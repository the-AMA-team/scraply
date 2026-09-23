#!/usr/bin/env python3
"""
Health check script for Python backend
Checks if all required dependencies are available
"""

import sys
import json
import os


def check_health():
    """Check if all required modules and dependencies are available"""
    try:
        # Add the dynamic-model-api directory to path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        api_dir = os.path.join(parent_dir, "dynamic-model-api")

        if not os.path.exists(api_dir):
            return {
                "status": "offline",
                "error": "dynamic-model-api directory not found",
            }

        sys.path.insert(0, api_dir)

        # Test importing required modules
        import torch
        import numpy as np
        from sklearn.model_selection import train_test_split

        # Try importing our custom modules
        from models import DynamicModel, Train
        from generate import Generate
        from params import DATALOADERS, LAYERS, ACTIVATIONS, LOSSES, OPTIMIZERS

        return {
            "status": "online",
            "message": "Python backend is running",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

    except ImportError as e:
        return {"status": "offline", "error": f"Missing dependency: {str(e)}"}
    except Exception as e:
        return {"status": "offline", "error": f"Health check failed: {str(e)}"}


def main():
    """Main function"""
    result = check_health()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
