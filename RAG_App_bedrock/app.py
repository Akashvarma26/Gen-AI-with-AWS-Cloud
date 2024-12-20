# Importing libraries
import os
import json
import sys
import boto3
import streamlit as st
# Using Amazon Titan to generate embeddings using bedrock
from langchain_aws import BedrockEmbeddings
from langchain_aws import BedrockLLM
# Data Ingestion Libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# Initiating embedding model from AWS Bedrock using langchain
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embed=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

# Data Ingestion
def data_ingest():
    old_docs=PyPDFDirectoryLoader("data").load()
    text_split=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    docs=text_split.split_documents(old_docs)
    return docs

# Vector Embedding and storing
def get_vectordb(docs):
    faiss_vdb=FAISS.from_documents(docs,bedrock_embed)
    faiss_vdb.save_local("vdb_index")

# Models Initiating using Bedrock and langchain
def get_claudeAI_LLM():
    llm=BedrockLLM(model_id="anthropic.claude-instant-v1",client=bedrock,
                model_kwargs={'max_tokens_to_sample':500}
                )
    return llm

def get_Llama3_LLM():
    llm=BedrockLLM(model_id="meta.llama3-8b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':512}
                )
    return llm

# Prompt templates
Prompt_Template="""
Human: Use the following pieces of context to provide concise answers to the questions at the end. 
But use atleast 250 words to summarize when asked to summarize with detailed explanation.
Do Not make up an answer if you do not know.

<Context>
{context}
</context>

Question:{question}

Assistant:"""

prompt=PromptTemplate(
    template=Prompt_Template, input_variables=["context","question"]
)

# Getting the response from llm
def get_response_llm(llm,vdb,query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,chain_type="stuff",
        retriever=vdb.as_retriever(
            search_type="similarity", search_kwargs={"k":4}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":prompt}
    )
    answer=qa({"query":query})
    return answer['result']


# Streamlit App
def main():
    st.set_page_config("RAG_BEDROCK",page_icon="🔗")
    st.header("Chat with In house pdfs")
    user_input=st.text_input("Ask anything!!!")
    with st.sidebar:
        st.title("Menu")
        if st.button("Vectors Update"):
            with st.spinner("processing..."):
                docs=data_ingest()
                get_vectordb(docs)
                st.success("Done!!!")

    if st.button("ClaudeAI Answer"):
        faiss_index=FAISS.load_local("vdb_index", embeddings=bedrock_embed,allow_dangerous_deserialization=True)
        llm=get_claudeAI_LLM()
        st.success(get_response_llm(llm,faiss_index,query=user_input))

    if st.button("Llama3 Answer"):
        faiss_index=FAISS.load_local("vdb_index", embeddings=bedrock_embed,allow_dangerous_deserialization=True)
        llm=get_Llama3_LLM()
        st.success(get_response_llm(llm,faiss_index,query=user_input))

if __name__=="__main__":
    main()