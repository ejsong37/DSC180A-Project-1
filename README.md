# Implementation of Various Multi-Armed Bandit Algorithms

<h2> <u> About </u> </h2>

This repository contains the code used for the Data Science Project 1 (DSC-180A) course from UC San Diego during Fall 2022 under Professor Yuhua Zhu. This project attempts to implement some of the most popular multi-armed bandit algorithms and explore their tradeoffs.
<hr>

<h2> <u> Contents </u> </h2>

* [Notebooks](https://github.com/ejsong37/dsc180a/tree/master/notebooks)

* [](https://github.com/ejsong/dsc180a/blob/master/references/master_list.txt)
<h2> <u> Installation </u> </h2>

Below are instructions to setup this project on your local machine using a Anaconda or miniconda environment. Please ensure that you have installed `git` on your machine beforehand. Further instructions can be found [here](https://git-scm.com/).

**Note: These instructions assume that the directory (`dsc180a`) containing all the files will be located in the path `C:\Users\<username>`. Feel free to change where you store your files, but ensure that you are in the same directory for every step.**

**1. Clone this project to your local machine.**

Fork the repository into your own GitHub account (a ```fork``` is a copy of the repository), and run the following on the Anaconda/miniconda prompt:

```
git clone https://github.com/<your-user-name>/dsc180a.git
```

You may be asked to enter your GitHub credentials.

**2. Create a conda environment.**

Using a [conda](https://docs.conda.io/en/latest/) environment will help manage modules/dependencies and isolate working environments. The file ```requirements.txt``` specifies the Python version and required libraries.

```
cd dsc180a
conda create --name dsc180a python=3.7
```

Once the environment is created, activate it in the Anaconda/miniconda console.

```
conda activate dsc180a
```

**3. Install modules.**

Using the `requirements.txt` file located in the `dsc180a` directory, install **Jupyter Notebook** and the required libraries using `pip` or `conda`.

```
pip install -U jupyter
pip install -r requirements.txt
```

**4. Launch the Jupyter Notebook.**

Once the environment is fully set-up, launch **Jupyter Notebook** in the console.

```
jupyter notebook
```

This will open up a web-browser where you will access all of the files associated with `dsc180a`.

**For Windows users:**
Alternatively, you can write a batch script to start-up the notebook instantly.

On `Notepad`, write the following script:

```
echo off

CALL C:\Users\<username>\miniconda3\Scripts\activate.bat C:\Users\<username>\miniconda3\envs\dsc180a
CD /D C:\Users\<username>\dsc180a
jupyter notebook

echo on
```
Save the file as `dsc180a.bat`. This script will automatically open a browser with **Jupyter Notebook** access.

<hr>
