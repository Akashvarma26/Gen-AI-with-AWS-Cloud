# Importing Libraries
import boto3 # For AWS services usage
import json # For parsing json data

# Input and prompt
prompt_data="""
Act like a pirate and write a poem about Machine Learning.
"""
# Connecting/ Interacting with bedrock service
bedrock=boto3.client(service_name="bedrock-runtime")

# Attributes needed in API
payload={
    "prompt":prompt_data,
    "max_gen_len":512,
    "temperature":0.5,
    "top_p":0.9
}

# Configurating the model
body=json.dumps(payload)
model_id="meta.llama3-8b-instruct-v1:0"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    contentType= "application/json",
    accept= "application/json",
)

# Response and output parsing
response_body=json.loads(response.get("body").read())
response_text=response_body['generation']
print(response_text)