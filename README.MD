# Project 2

To get started, we will need to install spark from:

> http://spark.apache.org/downloads.html

Run the following to unzip and move to the correct folder

```
tar -xzf {SPARK_DOWNLOAD}.tgz
mv {SPARK_DOWNLOAD} /opt/{SPARK_VERSION}
ln -s /opt/{SPARK_VERSION} /opt/spark̀
```

We need to setup a virtual environment for our python code

```
cd 442_assignment2
python3 -m venv .
source bin/activate
```

Next, we will need to install a few libraries. With the virtual environment up and running, run the following commands:

```
pip install wheel
pip install jupyter
pip install numpy
pip install pyspark
```

Make sure java is installed and add the JAVA_HOME environment variable to your system.

Add the following to your .bashrc file:

```
export SPARK_HOME=/opt/spark
export PATH=$SPARK_HOME/bin:$PATH
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='notebook'
#export PYSPARK_PYTHON=python3
```

To run pyspark use:

```
pyspark
```

To run jupyter notebook (a playground environment) use:

```
jupyter notebook
```

To run the main program use:

```
python3 src/main.py
```

To update requirements.txt use:

```
pip freeze > requirements.txt
```

# Docker Setup

Create an AWS EC2 instance and connect to it.

Then we will need git to get the repo onto the ec2 instance:

```bash
> sudo yum install git -y
```

Git requires ssh keys to be configured in order to pull code from a private repo so ssh keys were generated with the following command and using all the default options

```bash
> ssh-keygen
```

Next, the generated public key should be copied and pasted from the console into github to give the ec2 instance access:

```bash
> cd ~/.ssh
> cat id_rsa.pub
```

Once the keys for github have been setup, the code can be pulled onto the instances:

```bash
> cd ~
> git clone git@github.com:mgharlan/442_assignment2.git
> cd 442_assignment2
```

Run the following commands to setup docker on an ec2 instance

```bash
sudo yum update -y
sudo amazon-linux-extras install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
```

Log out and back in again.

Run the following commands to build the docker image

```bash
docker build -t harlangmason/cs442 .
docker run harlangmason/cs442
```

Log out of docker and back in again to update docker hub credentials.

```
docker logout
docker login
```

Push the new docker image to docker hub:

```
docker tag harlangmason/cs442 harlangmason/cs442:project2
docker push harlangmason/cs442:project2
```

To pull the image on another EC2 instance run:

```
docker push harlangmason/cs442:project2
```
