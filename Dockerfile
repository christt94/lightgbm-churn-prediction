# Base image with Jupyter and Python
FROM jupyter/scipy-notebook:latest

# Switch to working directory
WORKDIR /home/jovyan/work

# Copy all project files into the container
COPY . /home/jovyan/work

# Install additional dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter Notebook port
EXPOSE 8888

# Start Jupyter
CMD ["start-notebook.sh"]
