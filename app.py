import streamlit as st
import yaml
import boto3
import pandas as pd
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
import numpy as np
from utility import *
from admin import *
import json
import os

# Define the file path to store conversation history
CONVERSATION_HISTORY_FILE = "conversation_history.json"

def load_conversation_history():
    """Load conversation history from a JSON file."""
    if os.path.exists(CONVERSATION_HISTORY_FILE):
        with open(CONVERSATION_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_conversation_history():
    """Save conversation history to a JSON file."""
    with open(CONVERSATION_HISTORY_FILE, 'a') as f:
        json.dump(st.session_state.messages, f)

def refresh_vector_store_local(faiss_local_paths, pkl_local_paths, bucket_name, folder_path):
    try:
        dir_list = os.listdir(folder_path)
        faiss_indexes = []
        for faiss_local_path in faiss_local_paths:
            faiss_index = FAISS.load_local(
                index_name=faiss_local_path.split('/')[-1].replace('.faiss', ''),
                folder_path=folder_path,
                embeddings=bedrock_embeddings,
                allow_dangerous_deserialization=True
            )
            faiss_indexes.append(faiss_index)
        faiss_index = None
        for index in faiss_indexes:
            if faiss_index is None:
                faiss_index = index
            else:
                faiss_index.merge_from(index)    

        return faiss_index
    except Exception as e:
        st.error(f"An error occurred during metadata extraction: {e}")

def main(faiss_index, faiss_doc_index):
    st.sidebar.header("Metadata Refresh")
    st.sidebar.image("athena.png", width=200, caption="Athena - Your Q&A Buddy")
    if st.sidebar.button("Refresh"):
        catalog_name = 'AwsDataCatalog'
        get_glue_column_metadata_to_csv(catalog_name, metadata_file, output_location)
        loader = CSVLoader(file_path=metadata_file)
        documents = loader.load()
        docs = split_text(documents, 1000, 200)
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vector_store_file = "prompt_embeddings.index"
        vectorstore_faiss.save_local(index_name=vector_store_file, folder_path="vectorstore/")
        upload_to_s3(f"vectorstore/", bucket_name, f'vectorstore/')
        faiss_local_paths, pkl_local_paths = load_index(bucket_name, folder_path)
        st.sidebar.write("Metadata Refreshed")
    st.sidebar.image("images/docstore.png", width=200)
    if st.sidebar.button("Refresh Documents"):
        faiss_doc_paths, pkl_doc_paths = load_docs_index(bucket_name, doc_store_path)
        st.sidebar.write("Document store Refreshed Successfully")

    if st.session_state.role == 'admin':
        image_base64 = get_image_as_base64('images/docstore.png')
        link_url = "http://localhost:8052/"
        html = f"""
        <a href="{link_url}" target="_blank">
            <img src="data:image/jpeg;base64,{image_base64}" alt="Image" style="width:10%;">
        </a>
        """
        st.write("To upload the documents please click below icon..")
        st.markdown(html, unsafe_allow_html=True)

    llm = get_llm()
    
    # Initialize session state for history
    if "messages" not in st.session_state:
        st.session_state.messages = load_conversation_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if original_question := st.chat_input("Ask me the details you want from athena.."):
        st.session_state.messages.append({"role": "user", "content": original_question})
        with st.chat_message("user"):
            st.markdown(original_question)

        with st.chat_message("assistant"):
            question = f"{original_question}. Use the {knowledge_layer} only if the query involves a join, filter or calculation like SUM, AVG. Otherwise, refer to {st.session_state.messages} to see if a similar query has been asked earlier and refine the response accordingly."
            response = get_response(llm, faiss_index, original_question)
            print(response)
            if "drop" not in response.lower() and "delete" not in response.lower() and "truncate" not in response.lower() and "create" not in response.lower():
                query, status = get_valid_query(llm, faiss_index, response, "default", output_location)
                if status == False:
                    st.write("Could not generate a valid Athena query. Here's the information from the document vector store:")
                    response = get_response_from_doc(llm, faiss_doc_index, original_question)
                    st.write(response)
                else:
                    st.write(query)
                    df = run_athena_query(query, "default", output_location)
                    st.session_state.messages.append({"role": "assistant", "content": query})
                    st.write(df)
                    response = get_response_from_doc(llm, faiss_doc_index, original_question)
                    st.write(response)
            else:
                if st.session_state.role == 'admin':
                    run_athena_query(response, "default", output_location)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.success("Statement Executed Successfully.")
                else:
                    st.write("Sorry, You are not authorized to perform this action.")

        # Save conversation history after each interaction
        save_conversation_history()

if __name__ == "__main__":
    yaml_file_path = '.\config.yaml'
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    metadata_file = config['metadata_file']
    json_file_path = config['json_file_path']
    output_location = config['output_location']
    bucket_name = config['bucket_name']
    folder_path = config['folder_path']
    knowledge_layer_file = config['knowledge_layer_file']
    faiss_local_paths = config['faiss_local_paths']
    pkl_local_paths = config['pkl_local_paths']
    faiss_doc_paths = config['faiss_doc_paths']
    pkl_doc_paths = config['pkl_doc_paths']
    doc_store_path = config['doc_store_path']
    knowledge_layer = load_knowledge_layer(knowledge_layer_file)
    knowledge_layer = json.dumps(knowledge_layer)

    with open(json_file_path, 'r') as file:
        users = json.load(file)

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.role = None

    st.title("Talk to Athena")
    st.image(".\images\icon.png", width=200, caption="Athena - Your Q&A Buddy")
    if not st.session_state.logged_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = verify_login(users, username, password)
            st.session_state.messages = []
            if user:
                st.session_state.logged_in = True
                st.session_state.role = user['role']
                st.session_state.username = user['username']
            else:
                st.error("Invalid username or password")
    else:
        st.success(f"Welcome {st.session_state.username[0].upper()}{st.session_state.username[1:]}! You can access the {st.session_state.role} dashboard. ")
        faiss_index = refresh_vector_store_local(faiss_local_paths, pkl_local_paths, bucket_name, folder_path)
        faiss_doc_index = refresh_vector_store_local(faiss_doc_paths, pkl_doc_paths, bucket_name, doc_store_path)
        main(faiss_index, faiss_doc_index)
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.experimental_rerun()
