# Project 2

To get started, we need to setup a virtual environment for our python code

```
cd 442_assignment2
python3 -m venv .
source bin/activate
```

Next we will need to install a few libraries. With the virtual environment up and running, run the following commands:

```
pip install wheel
pip install jupyter
pip install pyspark
```

Make sure java is installed and add the JAVA_HOME environment variable to your system.
Add the following to your .bashrc file:

```
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='notebook'
```

To run the jupyter notebook (a playground environment) use:

```
jupyter notebook
```

To run the main program use:

```
python3 src/main.py
```

To update requirements.txt use:

```
pip freeze  requirements.txt
```
