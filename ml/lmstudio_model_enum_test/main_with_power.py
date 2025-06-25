#!/usr/bin/env python3
import argparse
import subprocess
import time
import requests
import csv
import sys
import json
import re
import platform
import psutil
import GPUtil
from cpuinfo import get_cpu_info
import threading
import os
from datetime import datetime

SERVER_URL = "http://localhost:1234"

def sanitize(s: str) -> str:
    return re.sub(r'[\W]+', '_', s).strip('_')

def detect_machine():
    host = sanitize(platform.node())
    system = platform.system()
    release = platform.release()
    info = get_cpu_info()
    cpu = sanitize(info.get("brand_raw", platform.machine()))
    total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    gpus = GPUtil.getGPUs()
    gpu_names = ";".join(g.name for g in gpus) or "N/A"
    gpu_mems  = ";".join(str(round(g.memoryTotal/1024,1)) for g in gpus) or "N/A"
    return {
        "host": host,
        "system": system,
        "release": release,
        "cpu": cpu,
        "total_ram_gb": total_ram_gb,
        "gpu_names": gpu_names,
        "gpu_memory_gb": gpu_mems
    }

class PowerMonitor:
    def __init__(self):
        self.monitoring = False
        self.power_readings = []
        self.monitor_thread = None
        self.system = platform.system()
        self.gpu_type = self._detect_gpu_type()
        self.sudo_authenticated = False
        
    def _authenticate_sudo(self):
        """Authenticate sudo access once at the beginning"""
        if self.sudo_authenticated:
            return True
        
        try:
            print("Power monitoring on macOS requires sudo access for powermetrics...")
            print("Please enter your password when prompted:")
            result = subprocess.run(["sudo", "-v"], check=True)
            self.sudo_authenticated = True
            print("Sudo authentication successful!")
            return True
        except subprocess.CalledProcessError:
            print("Failed to authenticate sudo access. Power monitoring will be disabled.", file=sys.stderr)
            return False
    
    def _detect_gpu_type(self):
        """Detect GPU type for appropriate power monitoring method"""
        gpus = GPUtil.getGPUs()
        if not gpus:
            return "none"
        
        # Check first GPU name to determine vendor
        gpu_name = gpus[0].name.lower()
        if "nvidia" in gpu_name or "geforce" in gpu_name or "rtx" in gpu_name or "gtx" in gpu_name:
            return "nvidia"
        elif "amd" in gpu_name or "radeon" in gpu_name:
            return "amd"
        elif "intel" in gpu_name or "arc" in gpu_name:
            return "intel"
        else:
            return "unknown"
    
    def _monitor_power_mac(self):
        """Monitor power usage on macOS using powermetrics"""
        while self.monitoring:
            try:
                # Try a simpler approach first - get CPU package power which includes GPU on Apple Silicon
                result = subprocess.run([
                    "sudo", "-n", "powermetrics", 
                    "--sample-count", "1",
                    "--sample-rate", "1000",
                    "--show-process-coalition",
                    "--show-process-gpu",
                    "--show-process-energy"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    output = result.stdout
                    
                    # Try to extract power information
                    power_value = self._extract_power_from_powermetrics_text(output)
                    if power_value is not None:
                        self.power_readings.append({
                            "timestamp": time.time(),
                            "gpu_power_watts": power_value
                        })
                        
                elif result.returncode == 1 and "sudo: a password is required" in result.stderr:
                    print("Warning: sudo authentication expired. Stopping power monitoring.", file=sys.stderr)
                    break
                else:
                    print(f"Warning: powermetrics failed with return code {result.returncode}: {result.stderr}", file=sys.stderr)
                    # Try fallback method without sudo (limited info but might work)
                    self._try_powermetrics_fallback()
                    
            except subprocess.TimeoutExpired:
                print("Warning: powermetrics timeout", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Power monitoring error: {e}", file=sys.stderr)
            
            time.sleep(2)  # Increased interval to reduce overhead
    
    def _try_powermetrics_fallback(self):
        """Try powermetrics without sudo (limited data available)"""
        try:
            result = subprocess.run([
                "powermetrics", 
                "--sample-count", "1",
                "--sample-rate", "1000",
                "-n", "cpu_power,gpu_power"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                power_value = self._extract_power_from_powermetrics_text(result.stdout)
                if power_value is not None:
                    self.power_readings.append({
                        "timestamp": time.time(),
                        "gpu_power_watts": power_value
                    })
        except:
            pass
    
    def _extract_power_from_powermetrics_text(self, output):
        """Extract power from powermetrics text output (works better than plist parsing)"""
        try:
            lines = output.split('\n')
            
            # First pass: Look specifically for GPU Power lines
            gpu_power = None
            for line in lines:
                line_stripped = line.strip()
                
                # Look for the specific GPU Power pattern: "GPU Power: 27317 mW"
                if line_stripped.startswith('GPU Power:'):
                    match = re.search(r'GPU Power:\s*(\d+(?:\.\d+)?)\s*mW', line_stripped)
                    if match:
                        gpu_power = float(match.group(1)) / 1000.0  # Convert mW to W
                        break
                    match = re.search(r'GPU Power:\s*(\d+(?:\.\d+)?)\s*W', line_stripped)
                    if match:
                        gpu_power = float(match.group(1))
                        break
            
            # Return GPU power if found
            if gpu_power is not None:
                return gpu_power
            
            # Second pass: Look for other power patterns as fallback
            for line in lines:
                line_stripped = line.strip()
                line_lower = line_stripped.lower()
                
                # Look for lowercase GPU power variant
                if line_stripped.lower().startswith('gpu power:'):
                    match = re.search(r'gpu power:\s*(\d+(?:\.\d+)?)\s*mw', line_lower)
                    if match:
                        return float(match.group(1)) / 1000.0  # Convert mW to W
                    match = re.search(r'gpu power:\s*(\d+(?:\.\d+)?)\s*w', line_lower)
                    if match:
                        return float(match.group(1))
                
                # Look for any "GPU Power" pattern anywhere in line
                if 'gpu power' in line_lower and ('mw' in line_lower or 'w' in line_lower):
                    # Extract number before 'mW' or 'W'
                    match = re.search(r'(\d+(?:\.\d+)?)\s*mw', line_lower)
                    if match:
                        return float(match.group(1)) / 1000.0  # Convert mW to W
                    match = re.search(r'(\d+(?:\.\d+)?)\s*w', line_lower)
                    if match:
                        return float(match.group(1))
            
            # Third pass: Look for CPU/package power as last resort (includes GPU on Apple Silicon)
            for line in lines:
                line_lower = line.strip().lower()
                
                if ('cpu power' in line_lower or 'package power' in line_lower) and ('mw' in line_lower or 'w' in line_lower):
                    match = re.search(r'(\d+(?:\.\d+)?)\s*mw', line_lower)
                    if match:
                        return float(match.group(1)) / 1000.0  # Convert mW to W
                    match = re.search(r'(\d+(?:\.\d+)?)\s*w', line_lower)
                    if match:
                        return float(match.group(1))
                        
        except Exception as e:
            print(f"Debug: Error parsing powermetrics output: {e}", file=sys.stderr)
            
        return None
    
    def _monitor_power_nvidia(self):
        """Monitor NVIDIA GPU power using nvidia-smi"""
        while self.monitoring:
            try:
                result = subprocess.run([
                    "nvidia-smi", 
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits"
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    gpu_powers = []
                    for line in lines:
                        try:
                            power = float(line.strip())
                            gpu_powers.append(power)
                        except ValueError:
                            continue
                    
                    if gpu_powers:
                        self.power_readings.append({
                            "timestamp": time.time(),
                            "gpu_power_watts": sum(gpu_powers),  # Total power across all GPUs
                            "individual_gpu_powers": gpu_powers
                        })
                else:
                    print(f"Warning: nvidia-smi failed: {result.stderr}", file=sys.stderr)
                    
            except subprocess.TimeoutExpired:
                print("Warning: nvidia-smi timeout", file=sys.stderr)
            except Exception as e:
                print(f"Warning: NVIDIA power monitoring error: {e}", file=sys.stderr)
            
            time.sleep(1)
    
    def _monitor_power_amd(self):
        """Monitor AMD GPU power - placeholder implementation"""
        print("AMD GPU power monitoring not yet implemented")
        # TODO: Implement AMD GPU power monitoring
        # Could use rocm-smi, amdgpu-top, or other AMD-specific tools
        while self.monitoring:
            # Placeholder - add actual AMD monitoring here
            time.sleep(1)
    
    def _monitor_power_intel(self):
        """Monitor Intel GPU power - placeholder implementation"""
        print("Intel GPU power monitoring not yet implemented")
        # TODO: Implement Intel GPU power monitoring
        # Could use intel_gpu_top or other Intel-specific tools
        while self.monitoring:
            # Placeholder - add actual Intel monitoring here
            time.sleep(1)
    
    def start_monitoring(self):
        """Start power monitoring in background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.power_readings = []
        
        if self.system == "Darwin":  # macOS
            # Authenticate sudo access first
            if not self._authenticate_sudo():
                print("Warning: Cannot authenticate sudo access. Power monitoring disabled.")
                return
            self.monitor_thread = threading.Thread(target=self._monitor_power_mac)
        elif self.gpu_type == "nvidia":
            self.monitor_thread = threading.Thread(target=self._monitor_power_nvidia)
        elif self.gpu_type == "amd":
            self.monitor_thread = threading.Thread(target=self._monitor_power_amd)
        elif self.gpu_type == "intel":
            self.monitor_thread = threading.Thread(target=self._monitor_power_intel)
        else:
            print(f"Warning: Power monitoring not supported for GPU type: {self.gpu_type}")
            return
        
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Started power monitoring ({self.system}, {self.gpu_type} GPU)")
    
    def stop_monitoring(self):
        """Stop power monitoring and return statistics"""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        if not self.power_readings:
            return {
                "avg_gpu_power_watts": None,
                "max_gpu_power_watts": None,
                "min_gpu_power_watts": None,
                "power_samples": 0
            }
        
        powers = [r["gpu_power_watts"] for r in self.power_readings if r["gpu_power_watts"] is not None]
        
        if not powers:
            return {
                "avg_gpu_power_watts": None,
                "max_gpu_power_watts": None,
                "min_gpu_power_watts": None,
                "power_samples": len(self.power_readings)
            }
        
        return {
            "avg_gpu_power_watts": round(sum(powers) / len(powers), 2),
            "max_gpu_power_watts": round(max(powers), 2),
            "min_gpu_power_watts": round(min(powers), 2),
            "power_samples": len(powers)
        }
    
    def test_powermetrics(self):
        """Test function to see what powermetrics returns"""
        print("Testing powermetrics output...")
        try:
            # Test without sudo first
            result = subprocess.run([
                "powermetrics", 
                "--sample-count", "1",
                "--sample-rate", "1000"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("=== Powermetrics output (no sudo) ===")
                print(result.stdout[:2000] + "..." if len(result.stdout) > 2000 else result.stdout)
                power = self._extract_power_from_powermetrics_text(result.stdout)
                print(f"Extracted power: {power}W")
            else:
                print(f"Powermetrics without sudo failed: {result.stderr}")
                
            # Test with sudo if authenticated
            if self.sudo_authenticated:
                result = subprocess.run([
                    "sudo", "-n", "powermetrics", 
                    "--sample-count", "1",
                    "--sample-rate", "1000"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    print("\n=== Powermetrics output (with sudo) ===")
                    # Show more output to see GPU Power line
                    print(result.stdout[:3000] + "..." if len(result.stdout) > 3000 else result.stdout)
                    power = self._extract_power_from_powermetrics_text(result.stdout)
                    print(f"Extracted power: {power}W")
                    
                    # Debug: Look specifically for GPU Power lines
                    gpu_lines = [line for line in result.stdout.split('\n') if 'gpu' in line.lower() and 'power' in line.lower()]
                    if gpu_lines:
                        print("\nFound GPU power related lines:")
                        for line in gpu_lines:
                            print(f"  {line.strip()}")
                    else:
                        print("\nNo GPU power lines found in output")
                        
                else:
                    print(f"Powermetrics with sudo failed: {result.stderr}")
                    
        except Exception as e:
            print(f"Test failed: {e}")
        print("=== End test ===\n")

def start_server():
    print("Starting server…")
    subprocess.run(["lms", "server", "start"], check=True)
    for _ in range(30):
        try:
            if requests.get(f"{SERVER_URL}/api/v0/models").ok:
                print("Server is up!")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("Error: server did not become healthy in time.", file=sys.stderr)
    sys.exit(1)

def list_models():
    raw = subprocess.check_output(["lms", "ls", "--json"], text=True)
    entries = json.loads(raw)
    llm_entries = [e for e in entries if e.get("type")=="llm" and e.get("modelKey")]
    if not llm_entries:
        print("No LLMs found in `lms ls --json`. Have you installed any?", file=sys.stderr)
        sys.exit(1)

    models = []
    print(f"Found {len(llm_entries)} LLM models:")
    for e in llm_entries:
        # CLI load ID (for loading)
        cli_id = e["modelKey"].replace("@","/")
        # extract filename from path and strip extension
        path = e.get("path","")
        fname = path.split("/")[-1]
        model_name = re.sub(r"\.gguf$", "", fname, flags=re.IGNORECASE)

        # parse paramsString, e.g. "27B"
        pstr   = e.get("paramsString","")
        params = float(pstr[:-1]) if isinstance(pstr,str) and pstr.upper().endswith("B") else None

        # parse sizeBytes
        sb     = e.get("sizeBytes")
        size   = sb/(1024**3) if isinstance(sb,(int,float)) else None

        arch = e.get("architecture","N/A")
        fmt  = e.get("format","N/A")

        print(f"  • {model_name} → load as {cli_id}; params={params or 'N/A'}B; "
              f"size={round(size,2) if size else 'N/A'}GB; arch={arch}; fmt={fmt}")

        models.append({
            "cli_id":       cli_id,
            "model_name":   model_name,
            "params_b":     params,
            "size_gb":      size,
            "architecture": arch,
            "format":       fmt
        })
    return models

def unload_all():
    subprocess.run(["lms","unload","--all"], check=True)

def load_model(cli_id):
    subprocess.run(["lms","load","--gpu","max","-y",cli_id], check=True)

def run_query(cli_id, prompt, power_monitor):
    # Start power monitoring for this query
    power_monitor.start_monitoring()
    
    try:
        resp = requests.post(
            f"{SERVER_URL}/api/v0/chat/completions",
            json={"model": cli_id, "messages":[{"role":"user","content":prompt}], "stream": False}
        )
        resp.raise_for_status()
        j = resp.json()
        stats = j.get("stats",{})
        usage = j.get("usage",{})
        
        # Stop power monitoring and get stats
        power_stats = power_monitor.stop_monitoring()
        
        return {
            "tokens_per_second":   stats.get("tokens_per_second"),
            "total_tokens":        usage.get("total_tokens"),
            "time_to_first_token": stats.get("time_to_first_token"),
            "generation_time":     stats.get("generation_time"),
            **power_stats
        }
    except Exception as e:
        # Make sure to stop monitoring even if query fails
        power_monitor.stop_monitoring()
        raise e

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs via LM Studio with power monitoring")
    parser.add_argument("--first-model", action="store_true",
                        help="Only benchmark the first model (for quick dev iterations)")
    parser.add_argument("--gpu-names",  type=str, default=None,
                        help="Override detected GPU names (semicolon-separated)")
    parser.add_argument("--gpu-memory", type=str, default=None,
                        help="Override detected GPU memory in GB (semicolon-separated)")
    parser.add_argument("--no-power", action="store_true",
                        help="Disable power monitoring")
    parser.add_argument("--test-power", action="store_true",
                        help="Test power monitoring setup and exit")
    args = parser.parse_args()

    mi = detect_machine()
    if args.gpu_names:
        mi["gpu_names"]     = args.gpu_names
    if args.gpu_memory:
        mi["gpu_memory_gb"] = args.gpu_memory

    # Initialize power monitor
    power_monitor = PowerMonitor()
    
    # Test power monitoring if requested
    if args.test_power:
        if power_monitor.system == "Darwin":
            power_monitor._authenticate_sudo()
        power_monitor.test_powermetrics()
        return

    # Build filename with machine + GPU info
    gpu_part = sanitize(mi["gpu_names"] + "_" + mi["gpu_memory_gb"])
    
    # Create tps_power directory if it doesn't exist
    output_dir = "tps_power"
    os.makedirs(output_dir, exist_ok=True)
    
    outfile = os.path.join(output_dir, f"benchmark_power_{mi['host']}_{mi['cpu']}_{gpu_part}.csv")

    # Configure power monitoring
    if args.no_power:
        print("Power monitoring disabled")
    elif power_monitor.system == "Darwin":
        # Pre-authenticate sudo on macOS to avoid password prompts during benchmarking
        print("Initializing power monitoring...")
        if not power_monitor._authenticate_sudo():
            print("Failed to authenticate sudo. Power monitoring will be disabled.")
            args.no_power = True

    start_server()
    models = list_models()
    if args.first_model:
        models = models[:1]
        print("→ [dev] Running only the first model:", models[0]["model_name"])

    prompt = "Hello, world! Summarize the latest AI news in ten sentences."
    results = []

    for m in models:
        name, cli_id = m["model_name"], m["cli_id"]
        print(f"\n→ Benchmarking {name} …")
        unload_all()
        try:
            load_model(cli_id)
        except subprocess.CalledProcessError:
            print(f"  ✗ Skipped {name}: load failed.", file=sys.stderr)
            continue

        try:
            if args.no_power:
                # Run without power monitoring (original behavior)
                resp = requests.post(
                    f"{SERVER_URL}/api/v0/chat/completions",
                    json={"model": cli_id, "messages":[{"role":"user","content":prompt}], "stream": False}
                )
                resp.raise_for_status()
                j = resp.json()
                stats = j.get("stats",{})
                usage = j.get("usage",{})
                query_stats = {
                    "tokens_per_second":   stats.get("tokens_per_second"),
                    "total_tokens":        usage.get("total_tokens"),
                    "time_to_first_token": stats.get("time_to_first_token"),
                    "generation_time":     stats.get("generation_time"),
                    "avg_gpu_power_watts": None,
                    "max_gpu_power_watts": None,
                    "min_gpu_power_watts": None,
                    "power_samples": None
                }
            else:
                # Run with power monitoring
                query_stats = run_query(cli_id, prompt, power_monitor)
            
            row = {
                **mi,
                "model_name":   name,
                "params_b":     m["params_b"],
                "size_gb":      m["size_gb"],
                "architecture": m["architecture"],
                "format":       m["format"],
                **query_stats
            }
            results.append(row)
            
            power_info = ""
            if not args.no_power and query_stats.get("avg_gpu_power_watts") is not None:
                power_info = f", avg power: {query_stats['avg_gpu_power_watts']}W"
            
            print(f"  ✔ {name}: {query_stats['tokens_per_second']} TPS, {query_stats['total_tokens']} tokens{power_info}")
        except Exception as e:
            print(f"  ✗ Error on {name}: {e}", file=sys.stderr)
        finally:
            unload_all()

    if not results:
        print("No successful benchmarks—nothing to write.", file=sys.stderr)
        sys.exit(1)

    # CSV columns (including power monitoring fields)
    fieldnames = [
        "host","system","release","cpu","total_ram_gb",
        "gpu_names","gpu_memory_gb",
        "model_name","params_b","size_gb","architecture","format",
        "tokens_per_second","total_tokens","time_to_first_token","generation_time",
        "avg_gpu_power_watts","max_gpu_power_watts","min_gpu_power_watts","power_samples"
    ]
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Done — results saved to {outfile}")

if __name__ == "__main__":
    main()
