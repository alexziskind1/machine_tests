
# Large .NET Project Benchmark

This repository contains a synthetic benchmark designed to evaluate the build performance of large .NET projects. The benchmark generates a vast number of C# files with computationally intensive code and includes commonly used NuGet packages. The setup is ideal for stress-testing the .NET build process on different hardware and configurations.

---

## **Methodology**

### 1. **Project Setup**
- The project targets `.NET 9.0` using the `Microsoft.NET.Sdk` template.
- A `LargeProject.csproj` file includes:
  - Compilation of files from the `GeneratedCode` directory.

### 2. **Generated Code**
- A Python script (`generate_code.py`) dynamically creates 100,000 C# classes.
- Each class includes:
  - A property (`Value`) with a unique value.
  - A `ComplexCalculation` method combining:
    - Recursive calculations.
    - Nested loops.
    - Array processing.

### 3. **Benchmark Execution**
- The benchmark measures:
  - **Build Time**: Time to compile all generated files and integrate package references.
- Post-build, the script cleans up generated files and build artifacts (`bin`, `obj`, and `GeneratedCode`).

---

## **Run Instructions**

### **On macOS**

1. **Prerequisites**
   - Install Python 3.x.
   - Install .NET SDK (version 9.0 or higher).
   - Ensure `dotnet` and `python3` are in your system’s PATH.

2. **Run the Benchmark**
   ```bash
   chmod +x run_benchmark.sh
   ./run_benchmark.sh
   ```

3. **Output**
   - The build process outputs timing information directly from `dotnet build`.

---

### **On Windows**

1. **Prerequisites**
   - Install Python 3.x.
   - Install .NET SDK (version 9.0 or higher).
   - Ensure `dotnet` and `python` are in your system’s PATH.

2. **Run the Benchmark**
   Open a PowerShell terminal and execute:
   ```powershell
   .\run_benchmark.ps1
   ```

3. **Output**
   - Timing information is displayed during the build step.

---

## **Customizations**

1. **Change Generated File Count**
   - Modify the `generate_code.py` script to adjust the number of files or class complexity.

2. **Invoke NuGet Package Features**
   - Add code to the generated files to use the included NuGet packages for a more realistic workload.

3. **Adjust Dependencies**
   - Edit `LargeProject.csproj` to include additional or alternate NuGet packages.

---

## **Cleanup**
All generated files and build artifacts are automatically cleaned up after the benchmark. If you need to retain them for debugging or analysis, comment out the cleanup commands in the scripts.

---

Feel free to customize and adapt this benchmark to suit your testing needs. Let me know if you’d like further assistance!
