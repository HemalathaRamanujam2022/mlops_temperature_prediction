Ensure you have the following tools on the Linux or Ubuntu Operating System installed locally or on an Amazon EC2 instance.

1. Install Docker if it is not available. For Docker, run the following commands.  
```
# install docker n the GCP VM or local host machine with Ubuntu / Debian OS
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

2. Install git if not available. An active GitHub account is needed to clone the project repository.
```
   sudo apt install -y git
```

3. Clone the projct repository from GitHub.
   ```
   git clone https://github.com/HemalathaRamanujam2022/mlops_temperature_prediction.git
   cd mlops_temperature_prediction
   ```

4. Create a virtual environment for the project by running the following commands. The project was built and tested on python 3.9.19 . The required dependencies are already setup in the Pipfile. Invoke the virtual environment as below.
   ```
   pipenv shell
   ```
