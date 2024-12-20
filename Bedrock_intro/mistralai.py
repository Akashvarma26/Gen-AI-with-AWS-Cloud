# Importing Libraries
import boto3 # For AWS services usage
import json # For parsing json data

# Input and prompt
prompt_data="""
Act like a shakespear and write a poem about Machine Learning.
"""

# Connecting/ Interacting with bedrock service
bedrock=boto3.client(service_name="bedrock-runtime")

# Attributes needed in API
payload={
    "prompt":"[INST]"+prompt_data+"[/INST]",
    "max_tokens":200,
    "temperature":0.5,
    "top_p":0.9,
    "top_k":50
}

# Configurating the model
body=json.dumps(payload)
model_id="mistral.mistral-7b-instruct-v0:2"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    contentType= "application/json",
    accept= "application/json",
)

# Response and output parsing
response_body=json.loads(response.get("body").read())
response_text=response_body.get("outputs")[0].get("text")
print(response_text)
