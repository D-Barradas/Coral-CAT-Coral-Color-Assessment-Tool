FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    bzip2 \
    libglib2.0-0 \
    libglu1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*



# Set the working directory in the container
WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

#RUN pip install git+https://github.com/facebookresearch/segment-anything.git

COPY app/ ./
COPY segment_anything/ ./segment_anything/

# Make port 8501 available to the world outside this container
EXPOSE 8501

# The command to run the app
CMD ["streamlit", "run", "streamlit_starting_page.py"]