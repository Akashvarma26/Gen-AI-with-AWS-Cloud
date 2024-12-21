# Importing libraries
import json
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
import base64

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
Prompt_Template_rag="""
Human: Use the following pieces of context to provide concise answers to the questions at the end.
But use atleast 250 words to summarize when asked to summarize with detailed explanation.
Do Not make up an answer if you do not know.

<Context>
{context}
</context>

Question:{question}

Assistant:"""

prompt_rag=PromptTemplate(
    template=Prompt_Template_rag, input_variables=["context","question"]
)

# Getting the response from llm
def get_response_llm_rag(llm,vdb,query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,chain_type="stuff",
        retriever=vdb.as_retriever(
            search_type="similarity", search_kwargs={"k":4}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":prompt_rag}
    )
    answer=qa({"query":query})
    return answer['result']

# Generate image using Stable Diffusion
def gen_image_from_llm(prompt_text):
    prompt_template = [{"text": prompt_text, "weight": 1}]
    payload = {
        "text_prompts": prompt_template,
        "cfg_scale": 15,
        "seed": 0,
        "steps": 50,
        "width": 512,
        "height": 512
    }
    body = json.dumps(payload)
    model_id = "stability.stable-diffusion-xl-v1"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response.get("body").read())
    artifact = response_body.get("artifacts")[0]
    image_encoded = artifact.get("base64").encode("utf-8")
    image_bytes = base64.b64decode(image_encoded)
    return image_bytes


# Streamlit App
def main():
    st.set_page_config("RAG BEDROCK APP",page_icon="🔗")
    st.header("Chat with In house pdfs or Generate Images")
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
        st.success(get_response_llm_rag(llm,faiss_index,query=user_input))

    if st.button("Llama3 Answer"):
        faiss_index=FAISS.load_local("vdb_index", embeddings=bedrock_embed,allow_dangerous_deserialization=True)
        llm=get_Llama3_LLM()
        st.success(get_response_llm_rag(llm,faiss_index,query=user_input))

    if st.button("Stable Diffusion Answer"):
        with st.spinner("Generating image..."):
            image_output=gen_image_from_llm(user_input)
            st.image(image_output, caption="Generated Image")

if __name__=="__main__":
    main()