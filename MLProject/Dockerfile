FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data and model files
COPY me_cfs_vs_depression_preprocessing/ ./me_cfs_vs_depression_preprocessing/
COPY modelling.py .

# Run training
CMD ["python", "modelling.py", "--n_estimators", "100", "--max_depth", "10"]
