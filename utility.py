import boto3
import csv
import json
import os
import sys
import re
import time
import shutil
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
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
        #print(f"Query failed with state: {state}")
        return None


def get_glue_column_metadata_to_csv(catalog_name, output_file, athena_output_location):
    glue_client = boto3.client('glue')
    athena_client = boto3.client('athena')
    
    try:
        # List databases in the catalog
        response = athena_client.list_databases(CatalogName=catalog_name)
        databases = response['DatabaseList']
        
        # Prepare columns metadata list for CSV
        columns_metadata = [["Database Name", "Table Name", "Column Name", "Type", "Comment"]]
        
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
                        comment = f"Example value: {first_row_values[i]}"
                    
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
        
        #print(f"Column metadata saved to {output_file} successfully.")
        
    except glue_client.exceptions.EntityNotFoundException:
        print(f"Catalog '{catalog_name}' not found in Glue.")
    except Exception as e:
        print(f"Error: {e}")

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def extract_tables_and_columns(question):
    # This regex assumes tables and columns are identified by capitalizing and separating words
    pattern = re.compile(r"([A-Z][a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)")
    return pattern.findall(question)

def find_closest_matches(retriever, items):
    validated_items = []
    for item in items:
        result = retriever.retrieve(item, search_type="similarity", search_kwargs={"k": 1})
        if result:
            validated_items.append(result[0]['text'])
        else:
            validated_items.append(item)  # fallback to original if no match is found
    return validated_items

def validate_items_from_vectorstore(vectorstore, items):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    return find_closest_matches(retriever, items)

def create_context(validated_tables_columns):
    context = ""
    for table, columns in validated_tables_columns.items():
        context += f"Table: {table}\n"
        for column in columns:
            context += f" - Column: {column}\n"
    return context

def get_response(llm, vectorstore, question):
    tables_columns = extract_tables_and_columns(question)
    table_names = list(set([tc[0] for tc in tables_columns]))
    column_names = list(set([tc[1] for tc in tables_columns]))

    validated_tables = validate_items_from_vectorstore(vectorstore, table_names)
    validated_columns = validate_items_from_vectorstore(vectorstore, column_names)

    validated_tables_columns = {table: [] for table in validated_tables}

    for table, column in tables_columns:
        if table in validated_tables and column in validated_columns:
            validated_tables_columns[table].append(column)

    context = create_context(validated_tables_columns)

    prompt_template = """Human: You are a SQL developer creating queries for Amazon Athena.
    Objective: Generate SQL queries to return data based on the user request and provided schema only. don't make up column names and  Use only functions relevant to Athena.
    - If multiple columns have the same name, include the column only once.
    - Append the table name with the database name in the format: database_name.table_name.
    - Use the CAST function for date columns, assuming the date is in YYYY-MM-DD format.
    Only return the SQL query without any additional information. Do not include any quotes.
    <context>
    {context}
    </context>
    Question: {question}
    Assistant:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context","question"]
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


def get_llm():
    llm = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock_client,
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

