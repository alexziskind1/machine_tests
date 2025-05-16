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

SERVER_URL = "http://localhost:1234"

def sanitize(s: str) -> str:
    # collapse any non-alphanumeric into underscores
    return re.sub(r'[\W]+', '_', s).strip('_')

def detect_machine():
    # Hostname
    host_raw = platform.node()
    host     = sanitize(host_raw)

    # OS info
    system   = platform.system()
    release  = platform.release()

    # CPU brand via py-cpuinfo
    info     = get_cpu_info()
    cpu_raw  = info.get("brand_raw", platform.machine())
    cpu      = sanitize(cpu_raw)

    # RAM
    total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)

    # GPUs
    gpus      = GPUtil.getGPUs()
    gpu_names = ";".join(g.name for g in gpus) or "N/A"
    gpu_mems  = ";".join(str(round(g.memoryTotal / 1024, 1)) for g in gpus) or "N/A"

    return {
        "host":           host,
        "system":         system,
        "release":        release,
        "cpu":            cpu,
        "total_ram_gb":   total_ram_gb,
        "gpu_names":      gpu_names,
        "gpu_memory_gb":  gpu_mems
    }

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
    llm_entries = [e for e in entries if e.get("type") == "llm" and e.get("modelKey")]
    if not llm_entries:
        print("No LLMs found in `lms ls --json`. Have you installed any?", file=sys.stderr)
        sys.exit(1)

    models = []
    print(f"Found {len(llm_entries)} LLM models:")
    for e in llm_entries:
        mk     = e["modelKey"]
        cli_id = mk.replace("@", "/")

        # parse paramsString, e.g. "27B"
        params = None
        pstr   = e.get("paramsString", "")
        if isinstance(pstr, str) and pstr.upper().endswith("B"):
            try:
                params = float(pstr[:-1])
            except ValueError:
                pass

        # parse sizeBytes (an integer)
        size = None
        sb   = e.get("sizeBytes")
        if isinstance(sb, (int, float)):
            size = sb / (1024 ** 3)

        arch = e.get("architecture", "N/A")
        fmt  = e.get("format", "N/A")

        print(f"  • {mk} → {cli_id}; "
              f"params={params or 'N/A'}B; size={round(size,2) if size else 'N/A'}GB; "
              f"arch={arch}; fmt={fmt}")

        models.append({
            "id":           cli_id,
            "params_b":     params,
            "size_gb":      size,
            "architecture": arch,
            "format":       fmt
        })
    return models

def unload_all():
    subprocess.run(["lms", "unload", "--all"], check=True)

def load_model(model_id):
    subprocess.run(["lms", "load", "--gpu", "max", "-y", model_id], check=True)

def run_query(model_id, prompt):
    resp = requests.post(
        f"{SERVER_URL}/api/v0/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    resp.raise_for_status()
    j = resp.json()
    stats = j.get("stats", {})
    usage = j.get("usage", {})
    return {
        "tokens_per_second":   stats.get("tokens_per_second"),
        "total_tokens":        usage.get("total_tokens"),
        "time_to_first_token": stats.get("time_to_first_token"),
        "generation_time":     stats.get("generation_time")
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs via LM Studio")
    parser.add_argument(
        "--first-model", action="store_true",
        help="Only benchmark the first model (for quick dev iterations)"
    )
    parser.add_argument(
        "--gpu-names", type=str, default=None,
        help="Override detected GPU names (semicolon-separated)"
    )
    parser.add_argument(
        "--gpu-memory", type=str, default=None,
        help="Override detected GPU memory (semicolon-separated in GB)"
    )
    args = parser.parse_args()

    machine_info = detect_machine()
    # apply overrides if given
    if args.gpu_names:
        machine_info["gpu_names"] = args.gpu_names
    if args.gpu_memory:
        machine_info["gpu_memory_gb"] = args.gpu_memory

    # sanitize GPU info for filename
    gpu_part = sanitize(machine_info["gpu_names"] + "_" + machine_info["gpu_memory_gb"])

    outfile = f"benchmark_{machine_info['host']}_{machine_info['cpu']}_{gpu_part}.csv"

    start_server()
    entries = list_models()
    if args.first_model:
        entries = entries[:1]
        print("→ [dev] Running only the first model:", entries[0]["id"])

    prompt  = "Hello, world! Summarize the latest AI news in one sentence."
    results = []

    for entry in entries:
        mid = entry["id"]
        print(f"\n→ Benchmarking {mid} …")
        unload_all()
        try:
            load_model(mid)
        except subprocess.CalledProcessError:
            print(f"  ✗ Skipped {mid}: could not load.", file=sys.stderr)
            continue

        try:
            stats = run_query(mid, prompt)
            row = {
                **machine_info,
                "model":           mid,
                "params_b":        entry["params_b"],
                "size_gb":         entry["size_gb"],
                "architecture":    entry["architecture"],
                "format":          entry["format"],
                **stats
            }
            results.append(row)
            print(f"  ✔ {mid}: {stats['tokens_per_second']} TPS, {stats['total_tokens']} tokens")
        except Exception as e:
            print(f"  ✗ Error querying {mid}: {e}", file=sys.stderr)
        finally:
            unload_all()

    if not results:
        print("No successful benchmarks—nothing to write.", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "host", "system", "release", "cpu", "total_ram_gb",
        "gpu_names", "gpu_memory_gb",
        "model", "params_b", "size_gb", "architecture", "format",
        "tokens_per_second", "total_tokens", "time_to_first_token", "generation_time"
    ]
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Done — results saved to {outfile}")

if __name__ == "__main__":
    main()
