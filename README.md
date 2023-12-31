# jobsum
summary job requirements

# workflows

1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline
7. Update the main.py
8. Update the app.py

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 217423437775.dkr.ecr.ap-southeast-2.amazonaws.com/text-s
	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one

# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = ap-southeast-2

    AWS_ECR_LOGIN_URI = demo>>  217423437775.dkr.ecr.ap-southeast-2.amazonaws.com

    ECR_REPOSITORY_NAME = text-s

# references

[quackr](https://quackr.io/)

[quillbot](https://quillbot.com/summarize)

[itviec](https://itviec.com/)

[google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail)

[Web Scraping with Python - Beautiful Soup Crash Course](https://www.youtube.com/watch?v=XVv6mJpFOb0)

[Text-Summarization-NLP-Project](https://github.com/krishnaik06/Text-Summarization-NLP-Project)

[End To End NLP Project Implementation With Deployment Github Action- Text Summarization- Krish Naik](https://www.youtube.com/watch?v=p7V4Aa7qEpw)