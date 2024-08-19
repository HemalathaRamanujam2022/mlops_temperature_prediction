
### AWS Account Setup (Free tier account)

Create an AWS account with your email. This email should be keop carefully and should not be used for general AWS services for any project.

Using the adminstrator account, create an IAM user and an IAM role with AdministratorAccess and AmazonS3FullAccess permissions. This user acts as the admin for the project and will be used to create the infrastructure. However when the infrastructure is created each service will have its own IAM role with the least required permissions.

Log in to AWS by using IAM user account created as above.

Create an access key for the user. This will give you the AWS_ACCESS_KEY and AWS_SECRET_ACCESS_KEY which you'll need to configure the AWS CLI later. Access keys are secret, just like a password. Don’t share them.

### EC2 virtual machine setup 

Refer to this [video](https://www.youtube.com/watch?v=IXSiYkP23zo) by Alexey for understanding on how to setup an SSH environment on the Amazon EC2 machine. You can then follow the below steps.

Launch a new EC2 instance. An Ubuntu OS (Ubuntu Server 24.04 LTS (HVM), SSD Volume Type, Architecture 64-bit (x86)) and a t2.large instance type, a 30Gb gp2 storage are recommended.

Note - Billing will start as soon as the instance is created and run.

Create a new key pair (access keys) so later you can connect to the new instance using SSH.

Save the .pem file in the ~/.ssh directory.

Create a config file in your .ssh folder

```
code ~/.ssh/config
```
