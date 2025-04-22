# Disk Speed Benchmark using dd

This project benchmarks disk read and write performance using the `dd` command and visualizes the results.

## How it Works

1.  **`diskcopy_logger.sh`**:
    *   This shell script performs the core benchmark.
    *   It uses `dd` to write a large test file (`testfile.tmp` by default, 100GB) to the current directory, simulating a large file write operation.
    *   While `dd` is running, the script captures the progress output (bytes transferred) sent to stderr by `dd` (using `status=progress`).
    *   It logs the operation type (`write`), a Unix timestamp, and the cumulative bytes transferred to `dd_speed_log.csv`.
    *   After the write operation, it clears the buffer cache (`sync; sudo purge` on macOS, `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` on Linux might be needed for more accurate read tests, though not implemented in the current script).
    *   It then uses `dd` to read the test file back, logging the progress similarly with the operation type `read`.
    *   Finally, it removes the test file.
2.  **`dd_speed_log.csv`**:
    *   A CSV file storing the raw data logged by `diskcopy_logger.sh`.
    *   Columns: `operation` (write/read), `timestamp` (Unix timestamp), `bytes` (cumulative bytes transferred).
3.  **`plot_speed.py`**:
    *   A Python script that reads `dd_speed_log.csv`.
    *   It processes the data, resampling it into 1-second intervals and calculating the average transfer speed (MB/s) for each interval.
    *   It generates an interactive HTML plot (`speed_plot.html`) and a static PNG image (`speed_plot.png`) showing the MB/s over time for both read and write operations.
4.  **`requirements.txt`**:
    *   Lists the Python libraries needed for `plot_speed.py` (`pandas`, `plotly`, `numpy`).

## Usage

1.  **Make the script executable**:
    ```bash
    chmod +x diskcopy_logger.sh
    ```
2.  **Run the benchmark**:
    *   Execute the script. Note that it uses `sudo purge` (macOS) to attempt clearing the disk cache before the read test, so it might ask for your password.
    ```bash
    ./diskcopy_logger.sh
    ```
    *   This will create `testfile.tmp` (can take a while depending on disk speed and file size), log data to `dd_speed_log.csv`, and then delete the test file.
3.  **Install Python dependencies**:
    *   Ensure you have Python and pip installed.
    *   Install the required libraries (preferably in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
4.  **Generate the plot**:
    *   Run the plotting script:
    ```bash
    python plot_speed.py
    ```
    *   This will create/update `speed_plot.html` and `speed_plot.png` in the current directory and attempt to open the plot in your default browser.

## Customization

*   You can change the test file size and name by editing the `FILE_SIZE` and `TEST_FILE` variables within `diskcopy_logger.sh`.
*   The `BLOCK_SIZE` variable in the script can also be adjusted, which might affect performance.
