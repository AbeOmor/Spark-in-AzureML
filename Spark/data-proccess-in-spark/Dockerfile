FROM mcr.microsoft.com/mmlspark/release:0.15

RUN conda install -y pip=20.1.1 && \
    conda clean -ay && \
    pip install --no-cache-dir azureml-defaults \
    pip install azureml-core \
    pip install absl-py \
    pip install pip install -U scikit-learn
