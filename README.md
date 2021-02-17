# Data download

1. Download Data

* download by command (mac)
    ```shell script
    ./download.sh 
    ```

* download by link
    >   download and unzip under folder named 'data'
    >
    >   [data.zip](https://drive.google.com/uc?id=14-Bl89OzRtLM1MCW2H81Xvivq8EvTrmB)

2. Download trained models

    >   download and unzip models of the paper
    >
    >   [models_paper.zip](https://drive.google.com/uc?id=1lvexOJmZ8zGHecwOwAGBoEjxRj9YXBO4)
    >
    >   [models_paper_mean_std.zip](https://drive.google.com/uc?id=13ySFQd77kpqrBS61HuaK-YA9fs8hOSJH)


# Conda Setting

```shell script
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
sha256sum Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh

conda create -n maxwellfdfd-ai python=3.7
conda activate maxwellfdfd-ai

# pip install -r requirements.txt
# download manually 
pip install C:\{pip-dependency}\Keras-2.3.1-py2.py3-none-any.whl --no-deps
pip install C:\{pip-dependency}\tensorflow-2.0.1-cp37-cp37m-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\protobuf-3.11.0-cp37-cp37m-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\absl-py-0.8.1.tar.gz --no-deps
pip install c:\{pip-dependency}\wrapt-1.11.2.tar.gz --no-deps
pip install c:\{pip-dependency}\gast-0.2.2.tar.gz --no-deps
pip install c:\{pip-dependency}\astor-0.8.0-py2.py3-none-any.whl --no-deps
pip install c:\{pip-dependency}\termcolor-1.1.0.tar.gz --no-deps
pip install c:\{pip-dependency}\keras_applications-1.0.8-py3-none-any.whl --no-deps
pip install c:\{pip-dependency}\keras_preprocessing-1.1.0-py2.py3-none-any.whl --no-deps
pip install c:\{pip-dependency}\PyYAML-5.2-cp37-cp37m-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\scipy-1.3.3-cp37-cp37m-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\matplotlib-3.1.2-cp37-cp37m-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\pyparsing-2.4.5-py2.py3-none-any.whl --no-deps
pip install c:\{pip-dependency}\cycler-0.10.0-py2.py3-none-any.whl --no-deps
pip install c:\{pip-dependency}\kiwisolver-1.1.0-cp37-none-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\Pillow-6.2.2-cp37-cp37m-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\scikit_learn-0.21.3-cp37-cp37m-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\joblib-0.14.0-py2.py3-none-any.whl --no-deps


# download manually (linux)
pip install --no-deps pip-dependency/tensorflow_gpu-2.3.0-cp37-cp37m-manylinux2010_x86_64.whl
pip install --no-deps pip-dependency/protobuf-3.11.0-cp37-cp37m-manylinux2010_x86_64.whl

pip install c:\{pip-dependency}\PyYAML-5.2-cp37-cp37m-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\scipy-1.3.3-cp37-cp37m-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\matplotlib-3.1.2-cp37-cp37m-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\kiwisolver-1.1.0-cp37-none-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\Pillow-6.2.2-cp37-cp37m-win_amd64.whl --no-deps
pip install c:\{pip-dependency}\scikit_learn-0.21.3-cp37-cp37m-win_amd64.whl --no-deps
pandas-1.2.0-cp37-cp37m-manylinux1_x86_64.whl 

source py3-maxwellfdfd/bin/activate
module load cuda10.0 cudnn_v7.6.1_cuda10.0
```
 train.py -m cnn_single -tr 30000 -te 3000 -oe -oh

user.sub q=shr_gpu ngpu=1 gputype=titan info=tensorflow jname=0127Fhrmse ~/jsh2/bin/python ~/0Surr_model/1224SurrModel-real-h.py
user.sub q=shr_gpu ngpu=1 gputype=titan info=tensorflow jname=cnn_raw_rmse ../py3-maxwell/bin/python train.py -tr -1 -te -1 -m cnn_raw_rmse 
 bjobs, bstop, bkill

python train_4sym.py -oe -tr -1 -te -1  > train_4sym_maxep50_patience_8_2.log &


python train_7x.py -oe -tr -1 -te -1  > train_7x_maxep50_patience_8_2_batch512.log &
# Run script 

* Train
    - default
    ```shell script
    python train.py 
    ```
 
    - different model, loss function
    ```shell script
    python train.py -m rf -l diff_rmse
    ```


* Test
    ```shell script
    python test.py 
    python test_ensemble.py
    ```

* Evaluate single data

    ```shell script
    python evaluate.py 
    ```

## To generate the data please visit the following GitHub URL : 
https://github.com/wonderit/maxwellfdfd

## How to Cite
Kim, W., Seok, J. Simulation acceleration for transmittance of electromagnetic waves in 2D slit arrays using deep learning. Sci Rep 10, 10535 (2020). https://doi.org/10.1038/s41598-020-67545-x

test