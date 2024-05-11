# https://github.com/Kaggle/docker-python/releases
FROM gcr.io/kaggle-gpu-images/python:v145

# ruff がnotebook上で設定できないのでblackとisortを入れる
RUN python3 -m pip install --upgrade pip \
    &&  pip install --no-cache-dir \
    black isort \ 
    jupyterlab_code_formatter 

RUN pip install --no-cache-dir \
    hydra-core 

RUN pip install netCDF4 webdataset
RUN python3 -m pip install papermill
RUN pip install --upgrade seaborn scikit-learn
