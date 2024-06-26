# Function to log date, time, battery percentage, CPU/memory usage, work done in last minute, and total work done
function Log-Status {
    $batteryStatus = Get-WmiObject Win32_Battery
    $batteryPercentage = $batteryStatus.EstimatedChargeRemaining
    $chargingStatus = if ($batteryStatus.BatteryStatus -eq 2) {"Charging"} else {"Discharging"}

    $cpuUsage = Get-Counter '\Processor(_Total)\% Processor Time'
    $memoryUsage = Get-Counter '\Memory\Available MBytes'

    $logEntry = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), Battery: $($batteryPercentage)% ($chargingStatus), CPU: $($cpuUsage.CounterSamples.CookedValue)%," +
                " Memory: Available $(($memoryUsage.CounterSamples.CookedValue))MB, Work Done: $args[0] iterations, Total Work Done: $args[1] iterations"
    Add-Content -Path "batlog.txt" -Value $logEntry
}

# Registering script to handle interruptions
$script:totalIterations = 0
try {
    while ($true) {
        $iterations = 0
        $end = (Get-Date).AddSeconds(60)
        while ((Get-Date) -lt $end) {
            $output = python ../benchmarksgame/python/mandelbrot/main.py 1000 | Out-Null

            if ($LASTEXITCODE -ne 0) {
                $errorLog = "Mandelbrot program encountered an error at $(Get-Date)"
                Add-Content -Path "batlog.txt" -Value $errorLog
                break 2 # Exit from both the loop and the script
            }

            $iterations++
        }

        $script:totalIterations += $iterations

        Log-Status $iterations $totalIterations

        Start-Sleep -Seconds 60
    }
} catch {
    $interruptLog = "Script interrupted, last work done: $iterations, total work done: $totalIterations"
    Add-Content -Path "batlog.txt" -Value $interruptLog
    Log-Status $iterations $totalIterations
}
