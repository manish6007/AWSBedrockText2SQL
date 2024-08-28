import boto3
import csv
import json
import os
import sys
import re
import time
import shutil
from datetime import datetime
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockLLM
from langchain_community.chat_models import BedrockChat
import pandas as pd
import hashlib
import streamlit.components.v1 as components
import base64


# Crete bedrock client
bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

#S3 client
s3_client = boto3.client('s3')

# Create bedrock embedding
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def query_athena_to_get_first_row(database_name, table_name, output_location):
    athena_client = boto3.client('athena')

    query = f"SELECT * FROM {database_name}.{table_name} LIMIT 1"
    
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database_name},
        ResultConfiguration={'OutputLocation': output_location}
    )
    
    query_execution_id = response['QueryExecutionId']
    
    while True:
        result = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state = result['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break

    if state == 'SUCCEEDED':
        result = athena_client.get_query_results(QueryExecutionId=query_execution_id)
        if 'ResultSet' in result and 'Rows' in result['ResultSet']:
            return result['ResultSet']['Rows'][1]['Data']  # Skipping header row
        else:
            return None
    else:
        print(f"Query failed with state: {state}")
        return None


def get_glue_column_metadata_to_csv(catalog_name, output_file, athena_output_location):
    glue_client = boto3.client('glue')
    athena_client = boto3.client('athena')
    
    try:
        # List databases in the catalog
        response = athena_client.list_databases(CatalogName=catalog_name)
        databases = response['DatabaseList']
        
        # Prepare columns metadata list for CSV
        columns_metadata = [["database_name", "table_name", "column_name", "type", "comment"]]
        
        for database in databases:
            database_name = database['Name']
            
            # List tables in the database
            tables_response = glue_client.get_tables(DatabaseName=database_name)
            
            for table in tables_response['TableList']:
                table_name = table['Name']
                
                # Get the first record from the table in Athena
                first_row = query_athena_to_get_first_row(database_name, table_name, athena_output_location)
                first_row_values = [item['VarCharValue'] for item in first_row] if first_row else []

                # Get columns for each table
                columns_response = glue_client.get_table(DatabaseName=database_name, Name=table_name)
                for i, column in enumerate(columns_response['Table']['StorageDescriptor']['Columns']):
                    # Update comment with the first row value if available
                    comment = column.get('Comment', 'N/A')
                    if first_row_values and i < len(first_row_values):
                        comment = f"{first_row_values[i]}"
                    
                    column_metadata = [
                        database_name,
                        table_name,
                        column['Name'],
                        column['Type'],
                        comment
                    ]
                    columns_metadata.append(column_metadata)
        
        # Write columns metadata to CSV file
        with open(output_file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(columns_metadata)
        
        print(f"Column metadata saved to {output_file} successfully.")
        
    except glue_client.exceptions.EntityNotFoundException:
        print(f"Catalog '{catalog_name}' not found in Glue.")
    except Exception as e:
        print(f"Error: {e}")

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def extract_tables_and_columns(question, faiss_index):
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    response =  retriever.invoke(question)
    data_list = [response[n].page_content for n in range(len(response))]
    final_dict = {}
    table_names = []
    column_names = []
    for item in data_list:
        # Split the string into lines and create a temporary dictionary
        lines = item.split('\n')
        temp_dict = {line.split(': ')[0]: line.split(': ')[1] for line in lines}

        # Extract relevant information
        database_name = temp_dict['database_name']
        table_name = temp_dict['table_name']
        column_name = temp_dict['column_name']
        column_type = temp_dict['type']
        example_value = temp_dict['comment']
        if table_name not in table_names:
            table_names.append(table_name)
        column_names.append(column_name)

        # Initialize database and table if not already present in final_dict
        if database_name not in final_dict:
            final_dict[database_name] = {}

        if table_name not in final_dict[database_name]:
            final_dict[database_name][table_name] = {
                'Column_names': []
            }

        # Append the column details to the Column_names list
        final_dict[database_name][table_name]['Column_names'].append({
            'column_name': column_name,
            'type': column_type,
            'example_value': example_value
        })
    return database_name, table_names, column_names, final_dict


def create_context(table_names, column_names, additional_info):
    context = ""
    context += f"Table: {table_names}\n"
    context += f" - Column: {column_names}\n"
    context += f" - DB, Table, Column mapping: {additional_info}\n"
    return context

def get_response(llm, vectorstore, question):
    database_name,table_names, column_names, dict = extract_tables_and_columns(question, vectorstore)

    context = create_context(table_names, column_names, dict)
    print(context)
    knowledge_layer = load_knowledge_layer("knowledge_layer.json")
    knowledge_layer = json.dumps(knowledge_layer)

    details = "It is important that the SQL query complies with Athena syntax. \
        Query should be enclosed within ```sql and ``` only. \
    During join if column name are same please use alias ex llm.customer_id \
    use the joins only and only if you think its required \
    in select statement. table name should follow the format database_name.table_name ex athena_db.customers \
    It is also important to respect the type of columns: \
    if a column is string, the value should be enclosed in quotes. \
    If you are writing CTEs then include all the required columns. \
    For date columns comparing to string , please cast the string input. \
    . \
    "

    final_question = "\n\nHuman:"+details + context + question+ "refer below for joining condition or filter if required \n\n"+knowledge_layer+ "n\nAssistant:"
    print(final_question)
    answer = llm.predict(final_question)
    query_str = answer.split("```")[1] if "```" in answer else answer
    query_str = " ".join(query_str.split("\n")).strip()
    sql_query = query_str[3:] if query_str.startswith("sql") else query_str

    return sql_query


def get_llm():
    llm = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock_client,
                  model_kwargs={'max_tokens': 512, 'temperature': 0.5})
    return llm

def run_athena_query(query, database, output_location):
    athena_client = boto3.client('athena')
    
    # Start the query execution
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': database
        },
        ResultConfiguration={
            'OutputLocation': output_location
        }
    )
    
    query_execution_id = response['QueryExecutionId']
    
    # Wait for the query to finish
    while True:
        result = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        status = result['QueryExecution']['Status']['State']
        
        if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(1)
    
    if status == 'SUCCEEDED':
        # Get the query results
        result = athena_client.get_query_results(QueryExecutionId=query_execution_id)
        
        # Extract column names
        column_info = result['ResultSet']['ResultSetMetadata']['ColumnInfo']
        columns = [col['Name'] for col in column_info]
        
        # Extract rows
        rows = result['ResultSet']['Rows'][1:]  # Skip the header row
        data = []
        for row in rows:
            data.append([item.get('VarCharValue', '') for item in row['Data']])
        
        # Convert to a pandas DataFrame
        df = pd.DataFrame(data, columns=columns)
        return df
    else:
        raise Exception(f"Query failed with status: {status}")

def run_athena_query_to_validate(query, database, output_location):
    athena_client = boto3.client('athena')
    
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': database
        },
        ResultConfiguration={
            'OutputLocation': output_location
        }
    )
    
    query_execution_id = response['QueryExecutionId']
    
    while True:
        result = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        status = result['QueryExecution']['Status']['State']
        
        if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(1)
    
    if status == 'SUCCEEDED':
        return True, None
    else:
        return False, result['QueryExecution']['Status']['StateChangeReason']

def validate_query(query, database, output_location):
    try:
        valid, error = run_athena_query_to_validate(query, database, output_location)
        return valid, error
    except Exception as e:
        return False, str(e)

def get_valid_query(llm, vectorstore, query, database, output_location,bucket_name):
    count = 1
    while True:
        
        valid, error = validate_query(query, database, output_location)
        #print(f"inside get_valid{valid}, {error}")
        
        if valid:
            return query, True
        else:
            question = f"The query generated was invalid due to: {error}. Please provide a corrected query. Original query: {query}. "
            #query = get_response(llm, vectorstore, question)
            query = generate_query(question,bucket_name,"")
            query_str = query.split("```")[1] if "```" in query else query
            query_str = query_str.replace("sql","") if "sql" in query_str else query_str
            query = " ".join(query_str.split("\n")).strip()
            
        if count == 5:
            print("Unable to generate a valid query. Please try again.")
            return query, False
        count += 1


def load_knowledge_layer(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def upload_to_s3(file_path, bucket_name, object_name):
    s3_client = boto3.client('s3')
    try:
        # List all files in the directory
        for root, dirs, files in os.walk(file_path):
            for file in files:
                full_file_path = os.path.join(root, file)
                s3_key = os.path.join(object_name, file)
                
                # Upload each file
                s3_client.upload_file(full_file_path, bucket_name, s3_key)
                print(f"File {full_file_path} uploaded to {bucket_name}/{s3_key}.")
    except Exception as e:
        print(f"Error uploading files to S3: {e}")

def load_index(bucket_name, folder_path):
    s3_client = boto3.client('s3')
    root_path = "/".join(folder_path.split("/")[:-2])
    backup_folder_path = root_path + '/backup'
    os.makedirs(backup_folder_path, exist_ok=True)

    # Move all existing files in the folder path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_file_name = f"{file}_{timestamp}"
                backup_file_path = os.path.join(backup_folder_path, backup_file_name)
                shutil.move(file_path, backup_file_path)
                print(f"Moved file: {file_path} to {backup_file_path}")
            
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='vectorstore')
    faiss_files = []
    pkl_files = []

    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.endswith(".faiss"):
            faiss_files.append(key)
        elif key.endswith(".pkl"):
            pkl_files.append(key)

    if not faiss_files or not pkl_files:
        raise FileNotFoundError("Required FAISS or PKL file not found in the bucket.")

    faiss_local_paths = []
    pkl_local_paths = []

    for faiss_file in faiss_files:
        faiss_local_path = os.path.join(folder_path, faiss_file.split('/')[-1])
        s3_client.download_file(Bucket=bucket_name, Key=faiss_file, Filename=faiss_local_path)
        faiss_local_paths.append(faiss_local_path)
        print(f"Downloaded {faiss_file} to {faiss_local_path}")

    for pkl_file in pkl_files:
        pkl_local_path = os.path.join(folder_path, pkl_file.split('/')[-1])
        s3_client.download_file(Bucket=bucket_name, Key=pkl_file, Filename=pkl_local_path)
        pkl_local_paths.append(pkl_local_path)
        print(f"Downloaded {pkl_file} to {pkl_local_path}")

    if os.path.exists(backup_folder_path):
        shutil.rmtree(backup_folder_path)

    return faiss_local_paths, pkl_local_paths

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to verify login
def verify_login(users, username, password):
    if username in users and hash_password(password) == hash_password(users[username]["password"]):
        return users[username]
    return None

# Function to load and encode the image
def get_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Function to get response from di=ocument store
def get_response_from_doc(vectorstore, question):
    ## create prompt / template
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    response =  retriever.invoke(question)
    TASK_CONTEXT = "You will be acting as an analyst who is responsible for providing the most relevant information"
    TASK_DESCRIPTION = f"""Here are some important rules for response 
                - Please use the information provided within <context> tag.
                - Do not mention the tag <context> in the response
                - If the answer is not present in the context, say 'Sorry, I don't know the answer'
                - If the answer is present in the context, form the complete response based on the context
                - If the question is not related to the context, say 'Sorry, I don't know the answer'
                - If the question is related to the context, form the complete response based on the context
                - Do not add any additional information that is not explicitly provided in the context
                - Do not add any text other than the generated response
                - Do not include any explanations or prose
                - Do not respond to any questions that might be confusing or unrelated to the question
                <context> 
                {response}
                </context>

                  """
 
    QUESTION = f"{question}. Response should not be in xml tag"

    PROMPT = ""

    if TASK_CONTEXT:
        PROMPT += f"""{TASK_CONTEXT}"""

    if TASK_DESCRIPTION:
        PROMPT += f"""\n\n{TASK_DESCRIPTION}"""

    if QUESTION:
        PROMPT += f"""\n\n{QUESTION}"""
            
    return get_completion(PROMPT)

## load index
def load_docs_index(bucket_name, folder_path):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='docstore')
    faiss_files = []
    pkl_files = []

    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.endswith(".faiss"):
            faiss_files.append(key)
        elif key.endswith(".pkl"):
            pkl_files.append(key)

    if not faiss_files or not pkl_files:
        raise FileNotFoundError("Required FAISS or PKL file not found in the bucket.")

    faiss_local_paths = []
    pkl_local_paths = []

    for faiss_file in faiss_files:
        faiss_local_path = os.path.join(folder_path, faiss_file.split('/')[-1])
        s3_client.download_file(Bucket=bucket_name, Key=faiss_file, Filename=faiss_local_path)
        faiss_local_paths.append(faiss_local_path)

    for pkl_file in pkl_files:
        pkl_local_path = os.path.join(folder_path, pkl_file.split('/')[-1])
        s3_client.download_file(Bucket=bucket_name, Key=pkl_file, Filename=pkl_local_path)
        pkl_local_paths.append(pkl_local_path)

    return faiss_local_paths, pkl_local_paths

def get_completion(prompt, system_prompt=None, prefill=None):
    inference_config = {
        "temperature": 0.0,
         "maxTokens": 500
    }
    converse_api_params = {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": inference_config
    }
    if system_prompt:
        converse_api_params["system"] = [{"text": system_prompt}]
    if prefill:
        converse_api_params["messages"].append({"role": "assistant", "content": [{"text": prefill}]})
    try:
        response = bedrock_client.converse(**converse_api_params)
        text_content = response['output']['message']['content'][0]['text']
        return text_content

    except ClientError as err:
        message = err.response['Error']['Message']
        print(f"A client error occured: {message}")

def load_conversation_history(CONVERSATION_HISTORY_FILE):
    """Load conversation history from a JSON file."""
    if os.path.exists(CONVERSATION_HISTORY_FILE):
        with open(CONVERSATION_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def generate_query(question,bucket_name, context):
    knowledge_layer = load_knowledge_layer("knowledge_layer.json")
    knowledge_layer = json.dumps(knowledge_layer)
    query_history = load_conversation_history("conversation_history.json")
    query_history = json.dumps(query_history)
    faiss_local_paths= ['vectorstore/prompt_embeddings.index.faiss']
    pkl_local_paths=['vectorstore/prompt_embeddings.index.pkl']
    # bucket_name='athena-destination-store-cg'
    folder_path='vectorstore/'
    TASK_CONTEXT = "You will be acting as an SQL Developer who is responsible for writing sql queries. It is important that the SQL query complies with Athena syntax"
    TASK_DESCRIPTION = """Here are some important rules for writing query 
                - Use the joins only and only if you think its required 
                - Use the CTE only if you think it is really required.
                - If you are writing CTEs then include all the required columns. 
                - Use the only functions compatible with athena. Don't use STRING_AGG 
                - During join if column name are same please use alias ex llm.customer_id
                - In Query please use the table name as database_name.table_name ex athena_db.customers
                - It is also important to respect the type of columns. 
                - It is also important to use the distinct if there is possibility of cross join
                - If a column is string, the value should be enclosed in quotes. 
                - For date columns comparing to string , please cast the string input. """
    EXAMPLES = f"""Refer the information in <knowledge_layer> tag for joining and filter condition only if required:
                <knowledge_layer>
                {knowledge_layer}
                </knowledge_layer>"""
    OUTPUT_FORMATTING = "Put your response in sql``` ```."
    PREFILL = "sql```"
    PRECOGNITION = "Think about your answer first before you respond."
    METADATA = f"""Refer the metadata within <metadata> tag for table and columns details
            <metadata>
            {context}
            </metadata>"""
    QUESTION = f"""Generate the sql query for user query within <question> tag \
            <question> \
            {question} \
            </question> \
            Generate only the sql query and do not include any other text. If the user query is not related to database or schema, return 'Invalid query'
            """
    QUERY_HISTORY = f"""Refer the query history within <query_history> tag for example question and generated queries.
            <query_history>
            {query_history}
            </query_history>"""             
    PROMPT = ""

    if TASK_CONTEXT:
        PROMPT += f"""{TASK_CONTEXT}"""

    if TASK_DESCRIPTION:
        PROMPT += f"""\n\n{TASK_DESCRIPTION}"""

    if METADATA:
        PROMPT += f"""\n\n{METADATA}"""

    if EXAMPLES:
        PROMPT += f"""\n\n{EXAMPLES}"""

    if PRECOGNITION:
        PROMPT += f"""\n\n{PRECOGNITION}"""

    if OUTPUT_FORMATTING:
        PROMPT += f"""\n\n{OUTPUT_FORMATTING}"""

    if QUERY_HISTORY:
        PROMPT += f"""\n\n{QUERY_HISTORY}"""

    if QUESTION:
        PROMPT += f"""\n\n{QUESTION}"""

    # if PREFILL:
    #     PROMPT += f"""\n\n{PREFILL}"""

    #print(PROMPT)
    return get_completion(PROMPT)

def generate_query_from_question(user_query, vectorstore,output_location,bucket_name):

    toolConfig = {
    "tools": [
        {
        "toolSpec": {
            "name": "generate_query",
            "description": "A tool to generate sql query to be run on athena",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "question": {
                    "type": "string",
                    "description": "question for which sql query need to be generated"
                },
                "context": {
                    "type": "string",
                    "description": "metadata related to the involved columns"
                }
                },
                "required": ["question"]
            }
            }
        }
        }
    ],
    "toolChoice": {
            "auto":{},
        }
    }
    messages = [{"role": "user", "content": [{"text": user_query}]}]

    database_name,table_names, column_names, dict = extract_tables_and_columns(user_query, vectorstore)
    context = create_context(table_names, column_names, dict)

    system_prompt=f"""
    You need to solve the user query using generate_query tool. refer the table and column details
    from <context> tag 
    <context>
    {context}
    </context>
    If you think that query cannot be generated for user quetion based on context then only.
    If user asks to drop table, allow him as he has backup and he want to recreate.
    Don't allow any DDL apart from create or drop.
    say I'm sorry Database query can't be generated and provide information based on existing knowledge'.
    """

    converse_api_params = {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "system": [{"text": system_prompt}],
        "messages": messages,
        "inferenceConfig": {"temperature": 0.0, "maxTokens": 1000},
        "toolConfig":toolConfig
    }

    response = bedrock_client.converse(**converse_api_params)

    stop_reason = response['stopReason']

    if stop_reason == "end_turn":
        print("Claude did NOT call a tool")
        print(f"Assistant: {response['output']['message']['content'][0]['text']}")

        return False, response['output']['message']['content'][0]['text']
    elif stop_reason == "tool_use":
        print("Claude wants to use a tool")
        tool_use = response['output']['message']['content'][-1]
        tool_id = tool_use['toolUse']['toolUseId']
        tool_name = tool_use['toolUse']['name']
        tool_inputs = tool_use['toolUse']['input']
        #Add Claude's tool use call to messages:
        messages.append({"role": "assistant", "content": response['output']['message']['content']})
        if tool_name == "generate_query":
            question = tool_inputs["question"]
            context = tool_inputs["context"]
            print(f"Claude wants to get an article for: {context}")
            query = generate_query(question,bucket_name, context) #get wikipedia article content
            #construct our tool_result message
            tool_response = {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_id,
                            "content": [
                                {"text": query}
                            ]
                        }
                    }
                ]
            }
            messages.append(tool_response)
            print(query)
            query_str = query.split("```")[1] if "```" in query else query
            query_str = query_str.replace("sql","") if "sql" in query_str else query_str
            query = " ".join(query_str.split("\n")).strip()
            llm = get_llm()
            
            if 'drop' in query.lower() or 'create' in query.lower() or 'alter' in query.lower():
                #print("I am here")
                return True, query
            else:
                query, status = get_valid_query(llm, vectorstore, query, "default", output_location,bucket_name)
                print(query)
                print(status)
                return status, query

