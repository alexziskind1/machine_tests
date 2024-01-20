#!/bin/bash

# Function to log date, time, battery percentage, CPU/memory usage, work done in last minute, and total work done
log_status() {
    local battery_info=$(pmset -g batt | grep 'InternalBattery')
    local battery_percentage=$(echo $battery_info | grep -o '[0-9]\+%')
    local charging_status=$(echo $battery_info | awk '{print $3}')
    local cpu_usage=$(top -l 1 -n 0 | grep 'CPU usage' | awk '{print $3, $5}') # user and system
    local memory_usage=$(top -l 1 -n 0 | grep 'PhysMem' | awk '{print $2, $6}') # used and free

    echo "$(date '+%Y-%m-%d %H:%M:%S'), Battery: $battery_percentage ($charging_status), CPU: $cpu_usage, Memory: $memory_usage, Work Done: $1 iterations, Total Work Done: $2 iterations" >> batlog.txt
}

# Handle script interruption
trap 'echo "Script interrupted, last work done: $iterations, total work done: $total_iterations"; log_status $iterations $total_iterations; exit' SIGINT SIGTERM

total_iterations=0

while true; do
    # Run the computational task for one minute, counting iterations
    iterations=0
    end=$((SECONDS+60))
    while [ $SECONDS -lt $end ]; do
        python ./benchmarksgame/python/mandelbrot/main.py 1000 >/dev/null

        # Check if the program ran successfully
        if [ $? -ne 0 ]; then
            echo "Mandelbrot program encountered an error at $(date)" >> batlog.txt
            break 2 # Exit from both the loop and the script
        fi

        ((iterations++))
    done

    # Update total iterations
    ((total_iterations+=iterations))

    # Log the status along with the number of iterations completed in the last minute and total iterations
    log_status $iterations $total_iterations

    # Sleep for one minute
    sleep 60
done
