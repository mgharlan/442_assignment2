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
