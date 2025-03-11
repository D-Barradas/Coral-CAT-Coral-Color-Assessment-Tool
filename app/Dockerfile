# Use the official Ubuntu base image
FROM ubuntu:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
# Install necessary packages including OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    vim \
    curl \
    bzip2 \
    libglib2.0-0 \
    libglu1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh --no-check-certificate
RUN bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && rm Miniforge3-Linux-x86_64.sh

# Set path to conda
ENV PATH=/opt/conda/bin:$PATH

# create three new folders: /app, /checkpoints, /segment_anything
# RUN mkdir /app/checkpoints /app/segment_anything

# # Copy the application files
# COPY checkpoints/vit*.pth /app/checkpoints/
# # copy the segment_anything folder
# COPY segment_anything/* /app/segment_anything/
# COPY pages/ /app/pages/
COPY . /app/

# Set the working directory in the container
WORKDIR /app

# RUN conda env create -n coral -f environment.yml -y
RUN mamba create -n coral python=3.10 -y

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "coral", "/bin/bash", "-c"]

#RUN mamba env create -f environment.yml -n coral -y

# Install necessary packages
RUN pip install --upgrade pip --no-cache-dir && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir && \
    pip install jupyterlab opencv-python pycocotools matplotlib onnxruntime onnx plotly streamlit scikit-learn scikit-image streamlit-extras easyocr --no-cache-dir

# RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# RUN git clone git@github.com:facebookresearch/segment-anything.git
# RUN cd segment-anything
# RUN pip install -e .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Use the environment created by environment.yml
SHELL ["conda", "run", "-n", "coral", "/bin/bash", "-c"]

# The command to run the app
ENTRYPOINT ["conda", "run", "-n", "coral"]
CMD ["streamlit", "run", "streamlit_starting_page.py"]
