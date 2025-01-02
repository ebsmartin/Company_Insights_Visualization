# Virtual Environment Setup and Activation Instructions

# Step 1: Navigate to the Project Directory
# Open PowerShell and change directory to the project folder:
# 
#   cd path\to\your\project
#
# Step 2: Create a Virtual Environment
# Run the following command to create a virtual environment in the project folder:
# 
#   python -m venv venv
#
# Step 3: Activate the Virtual Environment
# To activate the virtual environment, run the following command in PowerShell:
#
#   .\venv\Scripts\Activate
#
# Once activated, you should see `(venv)` at the beginning of the PowerShell prompt.
#
# Step 4: Install Dependencies
# While the virtual environment is active, install project dependencies using:
#
#   pip install -r requirements.txt
#
# Step 5: Deactivate the Virtual Environment
# To deactivate the virtual environment, simply run:
#
#   deactivate
#
# Notes:
# - If you encounter an error about script execution being disabled, run the following command
#   in PowerShell as Administrator to temporarily bypass the restriction:
#
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# - Ensure Python is installed and added to your PATH before starting.
