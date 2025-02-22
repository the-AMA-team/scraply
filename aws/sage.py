from sagemaker.pytorch import PyTorch

role = "arn:aws:iam::842675993918:role/service-role/AmazonSageMaker-ExecutionRole-20250222T132005"

bucket = "sagebucket3"

input_data_path = f"s3://{bucket}/training-data/"

# Create a PyTorch estimator
estimator = PyTorch(
    entry_point='text_generation_transformer.py',  # Your training script
    role=role,  # SageMaker execution role
    framework_version='1.12',  # PyTorch version
    
    py_version='py38',  # Python version
    instance_count=1,  # Number of instances
    instance_type='ml.p3.2xlarge',  # Instance type (GPU recommended)
    output_path=f"s3://{bucket}/output/",  # Output path for model artifacts
    hyperparameters={
        'epochs': 200,
        'learning-rate': 0.001,
        'batch-size': 32,
        'embed-dim': 100,
        'num-layers': 2,
        'num-heads': 2,
    },
)

# Start the training job
# try:
    # print("Starting training job...")
estimator.fit({'train': input_data_path})
    # print("Training job submitted.")
# except Exception as e:
    # print(f"Error: {e}")