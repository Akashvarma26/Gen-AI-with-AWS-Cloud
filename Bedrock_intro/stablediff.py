# Importing Libraries
import boto3 # For AWS services usage
import json # For parsing json data
import base64 # Encoding and decoding binary files and base64 files
import os

# Input and prompt
prompt_data="""
Show me a cartoonic image of robot typing in computer.
"""
prompt_template=[{"text":prompt_data,"weights":1}]

# Connecting/ Interacting with bedrock service
bedrock=boto3.client(service_name="bedrock-runtime")

# Attributes needed in API
payload={
    "text_prompts":prompt_template,
    "cfg_scale":10,
    "seed":0,
    "steps":50,
    "width":512,
    "height":512
}

# Configurating the model
body=json.dumps(payload)
model_id="stability.stable-diffusion-xl-v1"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    contentType= "application/json",
    accept= "application/json",
)

response_body=json.loads(response.get("body").read())
print(response_body)
artifact=response_body.get("artifacts")[0] # Extracting artifacts from response
image_encoded=artifact.get("base64").encode("utf-8") # Extracting base64 from artifacts and encoding it to bytes format
image_bytes=base64.b64decode(image_encoded) # Decoding the base64 to binary image format

# Save image file in output_dir/Images_gen
output_dir="Images_gen"
os.makedirs(output_dir,exist_ok=True)
file_name=f"{output_dir}/image.png"
with open(file_name,"wb") as f:
    f.write(image_bytes)