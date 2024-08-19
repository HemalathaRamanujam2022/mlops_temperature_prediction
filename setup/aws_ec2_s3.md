### EC2 virtual machine setup 

Refer to this [video](https://www.youtube.com/watch?v=IXSiYkP23zo) by Alexey for understanding on how to setup an SSH environment on the Amazon EC2 machine. You can then follow the below steps.

Launch a new EC2 instance. An Ubuntu OS (Ubuntu Server 24.04 LTS (HVM), SSD Volume Type, Architecture 64-bit (x86)) and a t2.large instance type, a 30Gb gp2 storage are recommended.

Note - Billing will start as soon as the instance is created and run.

Create a new key pair (access keys) so later you can connect to the new instance using SSH.

Save the .pem file in the ~/.ssh directory.

Create a config file in your .ssh folder

```
```
