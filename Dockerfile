FROM apache/airflow:2.1.1
COPY requirements.txt .
RUN pip install -r requirements.txt
