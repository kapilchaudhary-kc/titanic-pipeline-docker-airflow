from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import pandas as pd

# Import ML functions once at the top
from func.ml_pipeline import prepare_data, train_model, test_model

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def prepare_data_task(**kwargs):
    """Prepare Titanic data"""
    base_data_path = os.getenv('TITANIC_DATA_PATH', '/opt/airflow/data')
    train_path = os.path.join(base_data_path, 'train')
    test_path = os.path.join(base_data_path, 'test')

    prepare_data(train_path, 'train')
    prepare_data(test_path, 'test')

    # Return paths (automatically pushed to XCom)
    return {'train_path': train_path, 'test_path': test_path}

def model_training_task(**kwargs):
    """Train model"""
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='prepare_data_task')  # Pull the whole dict
    train_path = paths['train_path']

    model, val_acc = train_model(train_path)

    # Return model accuracy and path
    return {
        'validation_accuracy': float(val_acc),
        'model_path': os.path.join(train_path, 'titanic_model.pkl')
    }

def model_validation_task(**kwargs):
    """Validate model"""
    ti = kwargs['ti']

    # Pull paths from XCom
    paths = ti.xcom_pull(task_ids='prepare_data_task')  
    test_path = paths['test_path']  # Use the test dataset path
    
    # Define submission path
    base_data_path = os.getenv('TITANIC_DATA_PATH', '/opt/airflow/data')
    submission_path = os.path.join(base_data_path, 'submission')

    # Pull the model path from the model training task
    model_info = ti.xcom_pull(task_ids='model_training_task')
    model_path = model_info['model_path']  # Model path from training task

    # Validate the model using the test dataset
    result = test_model(test_path, model_path)

    # Handle labeled and unlabeled test cases separately
    if len(result) == 2:
        if isinstance(result[1], pd.DataFrame):  # Unlabeled data
            test_pred, submission_df = result

            # Save submission DataFrame to CSV
            os.makedirs(submission_path, exist_ok=True)
            submission_file = os.path.join(submission_path, 'submission.csv')
            submission_df.to_csv(submission_file, index=False)
            
            print(f"Submission saved at {submission_file}")
            
            return {'submission_file': submission_file, 'test_predictions': test_pred.tolist()}
        
        else:  # Labeled data
            test_pred, test_acc = result
            return {
                'test_accuracy': float(test_acc),
                'test_predictions': test_pred.tolist()
            }


with DAG(
    dag_id='titanic_survival_classifier',
    schedule_interval='@monthly',
    start_date=datetime(2025, 3, 25),
    catchup=False,
    default_args=default_args,
    tags=['ml', 'titanic'],
) as dag:

    prepare_data_op = PythonOperator(
        task_id='prepare_data_task',  # Task ID updated
        python_callable=prepare_data_task
    )

    model_training_op = PythonOperator(
        task_id='model_training_task',  # Task ID updated
        python_callable=model_training_task
    )

    model_validation_op = PythonOperator(
        task_id='model_validation_task',  # Task ID updated
        python_callable=model_validation_task
    )

# Define dependencies
prepare_data_op >> model_training_op >> model_validation_op
