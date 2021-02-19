FROM mcr.microsoft.com/mmlspark/release

RUN conda install -y pip=20.1.1 && \
    conda clean -ay && \
    pip install --no-cache-dir azureml-defaults \
    pip install azureml-core
