# Use Python 3.10.9 base image
FROM python:3.10.9

# Set the working directory
WORKDIR /workspace

# Copy requirements.txt first for caching
COPY requirements.txt .

RUN pip install --upgrade pip

# Install Python packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install JupyterLab
RUN pip install jupyterlab

# Copy the notebooks into the container
COPY notebooks/ ./notebooks/

# Open port 8888 for Jupyter
EXPOSE 8888

# Start JupyterLab when container runs
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

