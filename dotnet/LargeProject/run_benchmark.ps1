# Exit on error
$ErrorActionPreference = "Stop"

# Variables
$ProjectName = "LargeProject"
$CsProjFile = "$ProjectName.csproj"
$GeneratedCodeDir = "GeneratedCode"
$GenerateScript = "generate_code.py"

# Step 1: Clean up previous build artifacts
Write-Host "Cleaning up previous build artifacts..."
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue bin, obj, $GeneratedCodeDir

# Step 2: Ensure the Python generation script exists
if (!(Test-Path $GenerateScript)) {
    Write-Host "Error: $GenerateScript not found. Ensure the file exists in the current directory." -ForegroundColor Red
    exit 1
}

# Step 3: Generate files
Write-Host "Generating files using $GenerateScript..."
python $GenerateScript

# Step 4: Ensure the project file exists
if (!(Test-Path $CsProjFile)) {
    Write-Host "Error: $CsProjFile not found. Please ensure the .csproj file is set up correctly." -ForegroundColor Red
    exit 1
}

# Step 5: Restore NuGet packages
Write-Host "Restoring NuGet packages..."
dotnet restore $CsProjFile | Out-Null

# Step 6: Build the project
Write-Host "Building the project..."
dotnet build $CsProjFile

# Step 7: Clean up generated files and build artifacts
Write-Host "Cleaning up generated files and build artifacts..."
Remove-Item -Recurse -Force $GeneratedCodeDir, bin, obj
Write-Host "Cleanup complete."

Write-Host "Benchmark completed."