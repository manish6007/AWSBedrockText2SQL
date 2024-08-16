import boto3
import streamlit as st
import os
import uuid
import yaml

yaml_file_path = '.\config.yaml'
with open(yaml_file_path, 'r') as file:
    config = yaml.safe_load(file)
## s3_client
s3_client = boto3.client("s3")
#BUCKET_NAME = os.getenv("BUCKET_NAME")
bucket_name = config['bucket_name']

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def delete_objects_in_folder(bucket_name, folder_path):
    """
    Delete all objects within a specified folder in an S3 bucket.

    :param bucket_name: The name of the S3 bucket.
    :param folder_path: The path to the folder within the S3 bucket.
    """
    s3 = boto3.client('s3')

    # Ensure folder path ends with a '/'
    if not folder_path.endswith('/'):
        folder_path += '/'

    # List all objects within the specified folder
    objects_to_delete = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)

    # Check if there are any objects to delete
    if 'Contents' in objects_to_delete:
        # Create a list of objects to delete
        delete_keys = [{'Key': obj['Key']} for obj in objects_to_delete['Contents']]

        # Delete the objects
        s3.delete_objects(Bucket=bucket_name, Delete={'Objects': delete_keys})

        print(f"Deleted {len(delete_keys)} objects from folder {folder_path} in bucket {bucket_name}.")
    else:
        print(f"No objects found in folder {folder_path} in bucket {bucket_name}.")

## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## create vector store
def create_vector_store(request_id, documents, bucket_name):
    if not documents:
        st.write("No documents to process for vector store.")
        return False

    try:
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        file_name = "document_store"
        folder_path = "documents"
        os.makedirs(folder_path, exist_ok=True)
        vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

        ## Delete existing files
        delete_objects_in_folder(bucket_name, "docstore")
        ## upload to S3
        s3_client.upload_file(Filename=f"{folder_path}/{file_name}.faiss", Bucket=bucket_name, Key=f"docstore/document_store.faiss")
        s3_client.upload_file(Filename=f"{folder_path}/{file_name}.pkl", Bucket=bucket_name, Key=f"docstore/document_store.pkl")

        return True
    except Exception as e:
        st.write(f"Error creating vector store: {e}")
        return False

## main method
def docs_main(bucket_name):
    st.write("Administrator Home Page...")
    uploaded_files = st.file_uploader("Choose files", type="pdf", accept_multiple_files=True)
    if uploaded_files is not None:
        all_splitted_docs = []
        for uploaded_file in uploaded_files:
            request_id = "document_store"
            st.write(f"Processing file with Request Id: {request_id}")
            saved_file_name = f"{request_id}.pdf"
            with open(saved_file_name, mode="wb") as w:
                w.write(uploaded_file.getvalue())

            loader = PyPDFLoader(saved_file_name)
            pages = loader.load_and_split()

            st.write(f"Total Pages in {request_id}: {len(pages)}")

            ## Split Text
            splitted_docs = split_text(pages, 1000, 200)
            if not splitted_docs:
                st.write(f"No text found in file {request_id}")
                continue

            st.write(f"Splitted Docs length for {request_id}: {len(splitted_docs)}")
            st.write("===================")
            st.write(splitted_docs[0])
            st.write("===================")
            st.write(splitted_docs[1])

            all_splitted_docs.extend(splitted_docs)

        if all_splitted_docs:
            st.write("Creating the Vector Store for all uploaded files")
            final_request_id = "document_store"
            result = create_vector_store(final_request_id, all_splitted_docs, bucket_name)

            if result:
                st.write(f"Hurray!! All PDFs processed successfully into Vector Store with Request Id: {final_request_id}")
            else:
                st.write("Error processing PDFs!! Please check logs.")
        else:
            st.write("No valid documents found to create a vector store.")

if __name__ == "__main__":
    docs_main(bucket_name)
