# Set the base image using miniconda 
FROM continuumio/anaconda3

# Create environnement Python 3.7
RUN conda create -n envPython37 python=3.7

# Install packages in environnement workflow
# Run one environment workflow from env of anaconda
RUN echo "source activate envPython37" > ~/.bashrc

# redirection of environment
ENV PATH /opt/conda/envs/envPython37/bin:$PATH
RUN pip install -U protobuf && \
pip install tensorflow && \
pip install keras && \
pip install numpy && \
pip install h5py && \
pip install jupyter && \
pip install graphviz && \
pip install pandas && \
pip install pandas-gbq -U && \
pip install sklearn && \
pip install matplotlib && \
pip install ipyparallel && \
pip install seaborn
#pip install --upgrade google-api-workflow-client

# Use Kernel
RUN python -m ipykernel install --user --name envPython37 --display-name "Python (envPython37)"

# Set working directory 
WORKDIR /home/notebooks

EXPOSE 8000

CMD jupyter-lab --notebook-dir=/home/notebooks --no-browser --port=8888 --ip=0.0.0.0 --allow-root
