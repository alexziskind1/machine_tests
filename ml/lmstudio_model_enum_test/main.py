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
import os

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

def run_query(cli_id, prompt):
    resp = requests.post(
        f"{SERVER_URL}/api/v0/chat/completions",
        json={"model": cli_id, "messages":[{"role":"user","content":prompt}], "stream": False}
    )
    resp.raise_for_status()
    j = resp.json()
    stats = j.get("stats",{})
    usage = j.get("usage",{})
    return {
        "tokens_per_second":   stats.get("tokens_per_second"),
        "total_tokens":        usage.get("total_tokens"),
        "time_to_first_token": stats.get("time_to_first_token"),
        "generation_time":     stats.get("generation_time")
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs via LM Studio")
    parser.add_argument("--first-model", action="store_true",
                        help="Only benchmark the first model (for quick dev iterations)")
    parser.add_argument("--gpu-names",  type=str, default=None,
                        help="Override detected GPU names (semicolon-separated)")
    parser.add_argument("--gpu-memory", type=str, default=None,
                        help="Override detected GPU memory in GB (semicolon-separated)")
    args = parser.parse_args()

    mi = detect_machine()
    if args.gpu_names:
        mi["gpu_names"]     = args.gpu_names
    if args.gpu_memory:
        mi["gpu_memory_gb"] = args.gpu_memory

    # Build filename with machine + GPU info
    gpu_part = sanitize(mi["gpu_names"] + "_" + mi["gpu_memory_gb"])
    
    # Create tps directory if it doesn't exist
    output_dir = "tps"
    os.makedirs(output_dir, exist_ok=True)
    
    outfile = os.path.join(output_dir, f"benchmark_{mi['host']}_{mi['cpu']}_{gpu_part}.csv")

    start_server()
    models = list_models()
    if args.first_model:
        models = models[:1]
        print("→ [dev] Running only the first model:", models[0]["model_name"])

    prompt = "Hello, world! Summarize the latest AI news in one sentence."
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
            stats = run_query(cli_id, prompt)
            row = {
                **mi,
                "model_name":   name,
                "params_b":     m["params_b"],
                "size_gb":      m["size_gb"],
                "architecture": m["architecture"],
                "format":       m["format"],
                **stats
            }
            results.append(row)
            print(f"  ✔ {name}: {stats['tokens_per_second']} TPS, {stats['total_tokens']} tokens")
        except Exception as e:
            print(f"  ✗ Error on {name}: {e}", file=sys.stderr)
        finally:
            unload_all()

    if not results:
        print("No successful benchmarks—nothing to write.", file=sys.stderr)
        sys.exit(1)

    # CSV columns
    fieldnames = [
        "host","system","release","cpu","total_ram_gb",
        "gpu_names","gpu_memory_gb",
        "model_name","params_b","size_gb","architecture","format",
        "tokens_per_second","total_tokens","time_to_first_token","generation_time"
    ]
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Done — results saved to {outfile}")

if __name__ == "__main__":
    main()
