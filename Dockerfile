# 1) Start from a minimal Python ba
FROM --platform=linux/amd64 python:3.11-slim-bullseye


# 2) Set pip to be more patient
RUN pip config set global.timeout 100


# 3) Install system dependencies and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libgl1 \
      libglib2.0-0 \
      tesseract-ocr \
    # Clean up the apt cache to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# 4) Set the working directory in the container
WORKDIR /app


# 5) Copy and install Python dependencies first
# This assumes requirements.txt is in your project's root directory
COPY requirements.txt .
RUN pip install  -r requirements.txt
RUN pip install "numpy<2"



# 6) Copy only the application folder into the container
# This copies the contents of your local 'app' folder to '/app' in the container
COPY app/ .


# 7) Specify the default command to run
CMD ["python", "utils.py"]
