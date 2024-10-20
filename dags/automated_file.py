from prefect import task
import subprocess

@task(log_prints=True)
def trigger_docker_compose():
    try:
        # Run the docker-compose command
        result = subprocess.run(
            ['docker-compose', 'up', '-d'],  # '-d' runs containers in detached mode
            check=True,  # Raises CalledProcessError if the command exits with a non-zero status
            stdout=subprocess.PIPE,  # Captures standard output
            stderr=subprocess.PIPE  # Captures standard error
        )

        # Print the standard output
        print(result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        # Print error details if the command fails
        print(f"Error: {e.stderr.decode('utf-8')}")
        

if __name__ == "__main__":
    trigger_docker_compose()









































# import os
# import subprocess
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime, timedelta

# def trigger_training_pipeline():
#     # Getting the absolute path to main.py
#     file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../main.py'))
    
#     try:
#         # Running main.py using subprocess
#         result = subprocess.run(["python3", file_path], check=True, capture_output=True, text=True)
#         print(result.stdout)  # Log the standard output
#     except subprocess.CalledProcessError as e:
#         print(f"Error running script: {e.stderr}")  # Log the standard error
#         raise

# # Default arguments for the DAG
# default_args = {
#     'owner': 'airflow',
#     'depends_on_past': False,
#     'start_date': datetime(2023, 1, 1),  # Set your start date accordingly
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# # DAG definition
# dag = DAG(
#     'training_pipeline_trigger',
#     default_args=default_args,
#     description='A DAG to trigger the training pipeline by running main.py',
#     schedule_interval=timedelta(days=1),  # Runs daily; adjust as necessary
# )

# # Task definition using PythonOperator
# run_training_pipeline = PythonOperator(
#     task_id='run_training_pipeline',
#     python_callable=trigger_training_pipeline,
#     dag=dag)


# #Running the task
# run_training_pipeline



