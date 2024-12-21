# Gen-AI-with-AWS-Cloud
## Bedrock_intro
### Steps:  
- Install requirements.txt   
- Create an User in IAM Service. Give it AdminAccess policy while creating and then get access key using cli option.
- Configure the user using awscli in terminal. Use "aws configure". Enter details like access key, password key, region and output format(json). 
- Access model access in AWS bedrock. Create a api in python file. Just like in llama3.py , mistralai.py or stablediff.py.
- run these python files.

## RAG_App_bedrock
### Steps:
- Follow first two steps of [Bedrock_intro](#Bedrock_intro)
- create app.py and type the code as given. create "data" folder with pdf files. 
- run the python file app.py.
- The stable diffusion model does not need vector store DBs. Generate Images with good prompts.

## AWS_Sagemaker_deployment
- 
