#!/bin/bash

# Exit immediately if any command fails
set -e

# Variables
PROJECT_NAME="LargeProject"
CS_PROJ_FILE="${PROJECT_NAME}.csproj"
GENERATED_CODE_DIR="GeneratedCode"
GENERATE_SCRIPT="generate_code.py"

# Step 1: Clean up any previous build artifacts
echo "Cleaning up previous build artifacts..."
rm -rf bin obj "$GENERATED_CODE_DIR"

# Step 2: Ensure the Python generation script exists
if [ ! -f "$GENERATE_SCRIPT" ]; then
    echo "Error: $GENERATE_SCRIPT not found. Ensure the file exists in the current directory." - >&2
    exit 1
fi

# Step 3: Generate files
echo "Generating files using $GENERATE_SCRIPT..."
python3 "$GENERATE_SCRIPT"

# Step 4: Ensure the project file exists
if [ ! -f "$CS_PROJ_FILE" ]; then
    echo "Error: $CS_PROJ_FILE not found. Please ensure the .csproj file is set up correctly." - >&2
    exit 1
fi

# Step 5: Restore NuGet packages (Display output for debugging)
echo "Restoring NuGet packages..."
dotnet restore "$CS_PROJ_FILE" >/dev/null

# Step 6: Build the project
echo "Building the project..."
dotnet build "$CS_PROJ_FILE"

# Step 7: Clean up generated files and build artifacts
echo "Cleaning up generated files and build artifacts..."
rm -rf "$GENERATED_CODE_DIR" bin obj
echo "Cleanup complete."

echo "Benchmark completed."