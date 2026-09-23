#!/usr/bin/env python3
"""
Standalone notebook generation script for Electron app
Generates Jupyter notebooks without the FastAPI server
"""

import sys
import json
import os

# Add the dynamic-model-api directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
api_dir = os.path.join(parent_dir, "dynamic-model-api")
sys.path.insert(0, api_dir)

try:
    from generate import Generate

    print(
        json.dumps(
            {
                "status": "modules_imported",
                "message": "Successfully imported generation modules",
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


def generate_notebook(config):
    """Generate a Jupyter notebook with the provided configuration"""
    try:
        print(
            json.dumps(
                {"status": "starting", "message": "Starting notebook generation"}
            )
        )

        # Initialize generator
        generator = Generate(config)
        print(
            json.dumps(
                {
                    "status": "generator_created",
                    "message": "Generator initialized successfully",
                }
            )
        )

        # Generate notebook
        generator.generate_notebook()

        # Read the generated notebook file
        notebook_path = os.path.join(api_dir, "generated_notebook.ipynb")
        if os.path.exists(notebook_path):
            with open(notebook_path, "r") as f:
                notebook_content = f.read()

            print(
                json.dumps(
                    {
                        "status": "completed",
                        "message": "Notebook generated successfully",
                        "notebook_content": notebook_content,
                        "notebook_path": notebook_path,
                    }
                )
            )

            return notebook_content
        else:
            print(
                json.dumps(
                    {"status": "error", "message": "Generated notebook file not found"}
                )
            )
            return None

    except Exception as e:
        error_msg = f"Generation error: {str(e)}"
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

        # Generate notebook
        result = generate_notebook(config)

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
