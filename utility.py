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

def get_valid_query(llm, vectorstore, query, database, output_location):
    count = 1
    while True:
        
        valid, error = validate_query(query, database, output_location)
        
        if valid:
            return query, True
        else:
            question = f"The query generated was invalid due to: {error}. Please provide a corrected query. Original query: {query}. "
            query = get_response(llm, vectorstore, question)
            
        if count == 3:
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
def get_response_from_doc(llm, vectorstore, question):
    ## create prompt / template
    prompt_template = """
    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>
    Question: {question}
    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": question})
    return answer['result']

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

