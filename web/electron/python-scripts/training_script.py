#!/usr/bin/env python3
"""
Standalone training script for Electron app
Executes model training without the FastAPI server
"""

import sys
import json
import os
import torch
import io
from contextlib import redirect_stdout

# Add the dynamic-model-api directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
api_dir = os.path.join(parent_dir, "dynamic-model-api")
sys.path.insert(0, api_dir)

try:
    from models import DynamicModel, Train

    print(
        json.dumps(
            {
                "status": "modules_imported",
                "message": "Successfully imported training modules",
            }
        )
    )
except ImportError as e:
    print(
        json.dumps(
            {"status": "error", "message": f"Failed to import modules: {str(e)}"}
        )
    )
    sys.exit(1)


def run_simple_training(trainer, n_epochs, batch_size):
    """Run simple training without streaming or websocket features"""
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # No need for status messages - frontend only cares about progress updates
    for epoch in range(n_epochs):

        # Train for one epoch (capture any debug output)
        train_output = io.StringIO()
        with redirect_stdout(train_output):
            avg_train_loss, train_avg_acc = trainer.train(n_epochs, batch_size)

        # Test after epoch (capture any debug output)
        test_output = io.StringIO()
        with redirect_stdout(test_output):
            test_result = trainer.test(output_info=False)
            (
                avg_test_loss,
                test_avg_acc,
                per_class_metrics,
                confusion_matrix_data,
                overall_metrics,
            ) = test_result

        train_losses.append(avg_train_loss)
        train_accs.append(train_avg_acc)
        test_losses.append(avg_test_loss)
        test_accs.append(test_avg_acc)

        # Send training progress in the format expected by frontend
        train_losses_so_far = [{"x": i, "y": v} for i, v in enumerate(train_losses)]
        test_losses_so_far = [{"x": i, "y": v} for i, v in enumerate(test_losses)]
        
        progress_data = {
            "epoch": epoch + 1,
            "total_epochs": n_epochs,
            "progress": ((epoch + 1) / n_epochs) * 100,
            "train_loss": avg_train_loss,
            "train_accuracy": train_avg_acc,
            "test_loss": avg_test_loss,
            "test_accuracy": test_avg_acc,
            "train_losses": train_losses_so_far,
            "test_losses": test_losses_so_far,
        }
        
        print(json.dumps(progress_data))

    # Calculate final averages
    avg_train_acc = sum(train_accs) / len(train_accs)
    avg_test_acc = sum(test_accs) / len(test_accs)
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_test_loss = sum(test_losses) / len(test_losses)

    # Format results
    train_losses_formatted = [{"x": i, "y": v} for i, v in enumerate(train_losses)]
    test_losses_formatted = [{"x": i, "y": v} for i, v in enumerate(test_losses)]

    results = {
        "training": {
            "train_losses": train_losses_formatted,
            "test_losses": test_losses_formatted,
            "avg_train_loss": avg_train_loss,
            "avg_test_loss": avg_test_loss,
            "avg_train_acc": avg_train_acc,
            "avg_test_acc": avg_test_acc,
        },
        "outputs_class": per_class_metrics,
        "outputs_overall": overall_metrics,
        "confusion_matrix": confusion_matrix_data,
        "random_samples": {},
        "top_misclassified": {},
    }

    return results


def run_training(config):
    """Run training with the provided configuration"""
    try:
        print(json.dumps({"status": "starting", "message": "Initializing training"}))

        # Extract parameters from config
        layers = config["layers"]
        loss = config["loss"]
        optimizer = config["optimizer"]
        n_epochs = config["epoch"]
        batch_size = config["batch_size"]
        inp = config["input"]

        # Initialize model
        model = DynamicModel(layers)
        print(
            json.dumps(
                {"status": "model_created", "message": "Model initialized successfully"}
            )
        )

        # Initialize trainer (capture device output to avoid JSON parsing issues)
        device_output = io.StringIO()
        with redirect_stdout(device_output):
            trainer = Train(
                model=model,
                input=inp,
                loss=loss,
                optimizer=optimizer,
                batch_size=batch_size,
            )

        # Extract device info from captured output
        device_info = device_output.getvalue().strip()
        if device_info:
            print(json.dumps({"status": "device_info", "message": device_info}))
        print(
            json.dumps(
                {
                    "status": "trainer_created",
                    "message": "Trainer initialized successfully",
                }
            )
        )

        # Run training using simple synchronous method
        results = run_simple_training(trainer, n_epochs, batch_size)

        print(
            json.dumps(
                {
                    "status": "completed",
                    "message": "Training completed successfully",
                    "results": results,
                }
            )
        )

        return results

    except Exception as e:
        error_msg = f"Training error: {str(e)}"
        print(json.dumps({"status": "error", "message": error_msg}))
        return None


def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No configuration provided"}))
        sys.exit(1)

    try:
        # Parse configuration from command line argument
        config_str = sys.argv[1]
        config = json.loads(config_str)

        # Run training
        result = run_training(config)

        if result is None:
            sys.exit(1)

    except json.JSONDecodeError as e:
        print(
            json.dumps(
                {"status": "error", "message": f"Invalid JSON configuration: {str(e)}"}
            )
        )
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Unexpected error: {str(e)}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
