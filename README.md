# ☁️ Gen-AI-with-AWS-Cloud
## 🗀 Bedrock_intro
### Steps:  
- Install requirements.txt   
- Create an User in IAM Service. Give it AdminAccess policy while creating and then get access key using cli option.
- Configure the user using awscli in terminal. Use "aws configure". Enter details like access key, password key, region and output format(json). 
- Access model access in AWS bedrock. Create a api in python file. Just like in llama3.py , mistralai.py or stablediff.py.
- run these python files.

## 🗀 RAG_App_bedrock
### Steps:
- Follow first two steps of [Bedrock_intro](#Bedrock_intro)
- create app.py and type the code as given. create "data" folder with pdf files. 
- run the python file app.py.
- The stable diffusion model does not need vector store DBs. Generate Images with good prompts.

## 🗀 AWS_Sagemaker_deployment
### Steps:
- Go to Domains tab in AWS Sagemaker. Create a Domain and set up. After creating one, open "studio" in launch button of default user in user profile tab inside domains.
- Open JupyterLab from applications. create jupyterLab space and add instance, storage you would want to work with and run the space.
- Open JupyterLab. create a notebook and write the code as in "endpoint_creation.ipynb". by running first cell code, you have created a role.
- config model from HF as in second cell.
- Deploy it to sagemaker inferences and test it as in third cell.(It may take more than 20 mins to deploy as the endpoint for it is created.)
- To see your new endpoint, go back to the jupyter lab space page in AWS. Scroll down to deployments tab and select endpoints.click on your instance and add the request data(Like in the notebook file) under test inference tab next to settings. Send request to get output.


Note: Do not forget to Delete the services and instances to prevent incurring costs.