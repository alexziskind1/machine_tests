#!/usr/bin/env python3

import argparse, subprocess, time, requests, csv, sys, json, re, platform, psutil

from cpuinfo import get_cpu_info

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
    return {
        "host": host,
        "system": system,
        "release": release,
        "cpu": cpu,
        "total_ram_gb": total_ram_gb,
        "gpu_names": "N/A",           # For macOS/Apple Silicon
        "gpu_memory_gb": "N/A"        # For macOS/Apple Silicon
    }

def mem_usage_gb() -> float:
    keywords = ["LM Studio", "LM Studio Helper"]
    matches = []
    for p in psutil.process_iter(['name']):
        try:
            pname = p.info['name']
            if pname and any(k in pname for k in keywords):
                matches.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    total = sum(p.memory_info().rss for p in matches)
    return round(total / (1024 ** 3), 2)

def gpu_usage_gb() -> float:
    # On macOS, always zero (no Apple VRAM access)
    if platform.system() == "Darwin":
        return 0.0
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        return round(sum(g.memoryUsed for g in gpus) / 1024, 2) if gpus else 0.0
    except ImportError:
        return 0.0

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
        cli_id = e["modelKey"].replace("@", "/")
        path = e.get("path", "")
        fname = path.split("/")[-1]
        model_name = re.sub(r"\.gguf$", "", fname, flags=re.IGNORECASE)
        pstr = e.get("paramsString", "")
        params = float(pstr[:-1]) if isinstance(pstr, str) and pstr.upper().endswith("B") else None
        sb = e.get("sizeBytes")
        size = sb / (1024 ** 3) if isinstance(sb, (int, float)) else None
        arch = e.get("architecture", "N/A")
        fmt = e.get("format", "N/A")
        print(f"  • {model_name} → load as {cli_id}; params={params or 'N/A'}B; "
              f"size={round(size,2) if size else 'N/A'}GB; arch={arch}; fmt={fmt}")
        models.append({
            "cli_id": cli_id,
            "model_name": model_name,
            "params_b": params,
            "size_gb": size,
            "architecture": arch,
            "format": fmt
        })
    return models

def unload_all():
    subprocess.run(["lms", "unload", "--all"], check=True)

def load_model(cli_id, context_length):
    subprocess.run([
        "lms", "load",
        "--gpu", "max",
        "--context-length", str(context_length),
        "-y", cli_id
    ], check=True)

def run_query(cli_id, prompt):
    resp = requests.post(
        f"{SERVER_URL}/api/v0/chat/completions",
        json={"model": cli_id, "messages": [{"role": "user", "content": prompt}], "stream": False}
    )
    resp.raise_for_status()
    j = resp.json()
    stats = j.get("stats", {})
    usage = j.get("usage", {})
    return {
        "tokens_per_second": stats.get("tokens_per_second"),
        "total_tokens": usage.get("total_tokens"),
        "time_to_first_token": stats.get("time_to_first_token"),
        "generation_time": stats.get("generation_time")
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs via LM Studio with multiple context lengths")
    parser.add_argument("--first-model", action="store_true",
                        help="Only benchmark the first model (for quick dev iterations)")
    parser.add_argument("--contexts", type=str, default="256,2048,8192",
                        help="Comma-separated context lengths to test (tokens)")
    args = parser.parse_args()

    # Parse context list
    context_list = sorted({int(x) for x in args.contexts.split(",") if x.strip().isdigit()})
    if not context_list:
        print("--contexts produced no valid integers.", file=sys.stderr)
        sys.exit(1)

    mi = detect_machine()
    gpu_part = sanitize(mi["gpu_names"] + "_" + mi["gpu_memory_gb"])
    outfile = f"benchmark_{mi['host']}_{mi['cpu']}_{gpu_part}.csv"

    start_server()
    models = list_models()
    if args.first_model:
        models = models[:1]
        print("→ [dev] Running only the first model:", models[0]["model_name"])

    results = []
    for m in models:
        name, cli_id = m["model_name"], m["cli_id"]
        print(f"\n→ Benchmarking {name} …")
        unload_all()
        for ctx in context_list:
            try:
                load_model(cli_id, ctx)
            except subprocess.CalledProcessError:
                print(f"  ✗ Skipped {name}: load failed.", file=sys.stderr)
                continue

            ram_loaded = mem_usage_gb()
            gpu_loaded = gpu_usage_gb()
            print(f"    RAM after load: {ram_loaded} GB | GPU VRAM: {gpu_loaded} GB")

            prompt = "Just say hello."
            try:
                stats = run_query(cli_id, prompt)
                ram_after = mem_usage_gb()
                gpu_after = gpu_usage_gb()
                row = {
                    **mi,
                    "model_name": name,
                    "params_b": m["params_b"],
                    "size_gb": m["size_gb"],
                    "architecture": m["architecture"],
                    "format": m["format"],
                    "context_tokens": ctx,
                    "ram_loaded_gb": ram_loaded,
                    "ram_after_query_gb": ram_after,
                    "gpu_loaded_gb": gpu_loaded,
                    "gpu_after_query_gb": gpu_after,
                    **stats
                }
                results.append(row)
                print(f"      • ctx={ctx:<5} → {stats['tokens_per_second']:>6.1f} TPS, "
                      f"{stats['total_tokens']} tok | RAM Δ {ram_after-ram_loaded:+.2f} GB | "
                      f"VRAM Δ {gpu_after-gpu_loaded:+.2f} GB")
            except Exception as e:
                print(f"      ✗ Error at ctx={ctx}: {e}", file=sys.stderr)

            unload_all()

    if not results:
        print("No successful benchmarks—nothing to write.", file=sys.stderr)
        sys.exit(1)

    # CSV
    fieldnames = [
        "host", "system", "release", "cpu", "total_ram_gb",
        "gpu_names", "gpu_memory_gb",
        "model_name", "params_b", "size_gb", "architecture", "format",
        "context_tokens", "ram_loaded_gb", "ram_after_query_gb",
        "gpu_loaded_gb", "gpu_after_query_gb",
        "tokens_per_second", "total_tokens", "time_to_first_token", "generation_time"
    ]
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Done — {len(results)} rows written to {outfile}")

if __name__ == "__main__":
    main()
