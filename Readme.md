# 🚀 **Titanic Survival Prediction Pipeline with Apache Airflow**

This project implements a **Titanic Survival Prediction Pipeline** using **Apache Airflow**, **Docker**, and **Python**. The pipeline performs the following steps:

- **Data Validation**: Ensures the schema, checks for missing values, and identifies duplicates.
- **Data Preprocessing**: Applies feature engineering, handles missing values, and prepares the data for model training and testing.
- **Model Training**: Uses Logistic Regression to train the model on the Titanic training data.
- **Model Validation & Prediction**: Tests the model on both labeled and unlabeled test datasets and generates submission files.

---

## ⚙️ **Directory Structure**

```
├── dags/                          # Airflow DAGs
│   ├── __init__.py                # Enables the module import/export
│   ├── pipeline.py                # Airflow DAG definition
│   └── func/                      # Data processing and ML functions
│       ├── __init__.py            # Enables the module import/export
│       ├── ml_pipeline.py         # Data preprocessing, training, and testing functions
│       └── data_validator.py      # Data validation functions
├── data/                          # Data directory (train and test CSVs)
│   └── train/                     # Training Data and Trained Model Directory
│       ├── train.csv              # Titanic training data
│       ├── processed_train.pkl    # Preprocessed training data
│       └── processed_train.pkl    # Trained Model
│   └── test/                      # Testing/Validation Data Directory
│       ├── test.csv               # Titanic test data
│       └── processed_test.pkl     # Preprocessed test data
│   └── submission/                # Model predictions saved as CSV
├── logs/                          # Airflow logs
├── plugins/                       # Airflow plugins
├── Dockerfile                     # Docker configuration
├── docker-compose.yaml            # Airflow services configuration
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🚀 **Setup and Execution**

### ✅ **Prerequisites**

- Docker and Docker Compose installed
- Python 3.x installed

---

### 🔥 **1️⃣ Build and Run the Project**

1. **Build the Docker containers**

```bash
docker-compose build --no-cache
```

2. **Start the Airflow services**

```bash
docker-compose up -d
```

3. **Access the Airflow UI**

- Open your browser and go to:
  ```
  http://localhost:8080
  ```
- Username: `airflow`
- Password: `airflow`

---

### 🔥 **2️⃣ File Descriptions**

#### **1. **`` - Data Validation Module

- Validates the schema, checks for missing values, and detects duplicates.
- Generates a validation report with:
  - **Schema Validation**
  - **Duplicate Rows**
  - **Missing Values**

✅ **Usage Example:**

```python
from func import data_validator
validator = data_validator.Validator(df)
report = validator.validate()
```

---

#### **2. **`` - ML Pipeline

- ``: Prepares the Titanic dataset by applying:
  - Schema validation
  - Duplicate removal
  - Feature engineering (e.g., `FamilySize`, `IsAlone`)
  - One-hot encoding
  - Missing value imputation
- ``:
  - Trains a Logistic Regression model on the training data.
  - Splits the data into training and validation sets.
  - Saves the trained model as `titanic_model.pkl`.
- ``:
  - Tests the model on labeled or unlabeled test data.
  - For labeled data:
    - Returns accuracy and predictions.
  - For unlabeled data:
    - Generates a Kaggle-style submission file.

✅ **Usage Example:**

```python
from func.ml_pipeline import prepare_data, train_model, test_model

# Prepare data
train_df = prepare_data('/data', 'train')

# Train model
model, accuracy = train_model('/data')

# Test model
predictions, accuracy = test_model('/data', 'titanic_model.pkl')
```

---

#### **3. **`` - Airflow DAG

- Defines the Airflow DAG with three tasks:
  - `prepare_data_task`: Prepares the Titanic data.
  - `model_training_task`: Trains the model and stores it in XCom.
  - `model_validation_task`: Validates the model and creates submission files.
- Automatically handles labeled and unlabeled test sets.

✅ **DAG Execution:**

- DAG ID: `titanic_survival_classifier`
- Schedule: `@monthly`
- Start Date: `2025-03-25`
- Dependencies:
  ```
  prepare_data_task >> model_training_task >> model_validation_task
  ```

---

#### **4. **`` - Docker Compose Configuration

- Defines the Airflow cluster using:
  - **PostgreSQL**: Backend database.
  - **Redis**: Message broker for Celery.
  - **Airflow Webserver, Scheduler, Worker**: Manages DAG execution.
  - **Flower**: Celery monitoring UI.
- Mounts the volumes:
  ```
  volumes:
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins
    - ./dags/func:/opt/airflow/dags/func
    - ./data:/opt/airflow/data
  ```

✅ **Docker Services:**

- **Airflow Webserver:** `localhost:8080`
- **Flower UI:** `localhost:5555`

---

#### **5. **`` - Python Dependencies

- Contains the required Python libraries:

```
pandas  
numpy  
scikit-learn
```

---

#### **6. **``

- Specifies the base Airflow image:

```dockerfile
FROM apache/airflow:2.1.1
COPY requirements.txt .
RUN pip install -r requirements.txt
```

---

## 🚀 **Usage Examples**

### 🛠️ **Train the Model**

1. Run the pipeline.
2. The model will be saved as:

```
/data/train/titanic_model.pkl
```

### 🛠️ **Validate the Model**

- For labeled test data:

```
Test Accuracy: 0.84
```

- For unlabeled test data:

```
Submission saved at /data/submission/submission.csv
```

---

## ✅ **Airflow Commands**

- **Stop Airflow:**

```bash
docker-compose down
```

- **Clear Docker Cache:**

```bash
docker system prune -a
```

- **Check Logs:**

```bash
docker-compose logs -f
```

---

## 🚀 **Key Features**

- **Modularized ML Pipeline:**
  - Validation, preprocessing, training, and testing as separate functions.
- **Automated DAG Execution:**
  - Prepares data, trains, and validates the model automatically.
- **Flexible Handling:**
  - Supports both labeled and unlabeled test data.
- **Dockerized Deployment:**
  - Easily deployable using Docker Compose.

---

## 📊 **Results**

- Generates predictions for both labeled and unlabeled datasets.
- Creates a Kaggle-style submission file for unlabeled data.
- Displays model accuracy for labeled test data.

---

- ✨ **[KAPIL CHAUDHARY]** – Developer and Machine Learning Engineer
- 📧 Contact: [kapilchaudhary1707@gmail.com](mailto\:kapilchaudhary1707@gmail.com)

---

## 📚 **References**

- [Apache Airflow](https://airflow.apache.org/)
- [Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- [Titanic Dataset - Kaggle](https://www.kaggle.com/c/titanic/data)

---
