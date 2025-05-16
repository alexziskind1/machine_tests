#!/usr/bin/env python3
import argparse
import subprocess
import time
import requests
import csv
import sys
import json

SERVER_URL = "http://localhost:1234"

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
    """
    Call `lms ls --json`, parse the output for LLM entries, and
    extract:
      - modelKey → CLI load ID (replace '@' with '/')
      - parameter count (in billions) from paramsString
      - disk size (in GB) from sizeBytes
      - architecture
      - format
    Returns a list of dicts with keys: id, params_b, size_gb, architecture, format.
    """
    try:
        raw = subprocess.check_output(["lms", "ls", "--json"], text=True)
        entries = json.loads(raw)
    except Exception as e:
        print(f"Failed to list models via JSON: {e}", file=sys.stderr)
        sys.exit(1)

    llm_entries = [
        e for e in entries
        if e.get("type") == "llm" and e.get("modelKey")
    ]
    if not llm_entries:
        print("No LLMs found in `lms ls --json`. Have you installed any?", file=sys.stderr)
        sys.exit(1)

    models = []
    print(f"Found {len(llm_entries)} LLM models (modelKey → load ID; params; size; arch; format):")
    for e in llm_entries:
        mk = e["modelKey"]
        cli_id = mk.replace("@", "/")

        # parse paramsString, e.g. "27B"
        params_b = None
        pstr = e.get("paramsString")
        if isinstance(pstr, str) and pstr.upper().endswith("B"):
            try:
                params_b = float(pstr[:-1])
            except ValueError:
                pass

        # parse sizeBytes (an integer)
        size_gb = None
        sb = e.get("sizeBytes")
        if isinstance(sb, (int, float)):
            size_gb = sb / (1024 ** 3)

        architecture = e.get("architecture", "N/A")
        fmt = e.get("format", "N/A")

        params_str = f"{params_b:.1f}B" if params_b is not None else "N/A"
        size_str   = f"{size_gb:.2f}GB" if size_gb is not None else "N/A"
        print(f"  • {mk}  →  {cli_id};  params={params_str};  size={size_str};  arch={architecture};  fmt={fmt}")

        models.append({
            "id": cli_id,
            "params_b": params_b,
            "size_gb": size_gb,
            "architecture": architecture,
            "format": fmt
        })

    return models

def unload_all():
    subprocess.run(["lms", "unload", "--all"], check=True)

def load_model(model_id):
    # flags must come before the model_id
    subprocess.run(
        ["lms", "load", "--gpu", "max", "-y", model_id],
        check=True
    )

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
        "tokens_per_second": stats.get("tokens_per_second"),
        "total_tokens": usage.get("total_tokens"),
        "time_to_first_token": stats.get("time_to_first_token"),
        "generation_time": stats.get("generation_time")
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs via LM Studio")
    parser.add_argument(
        "--first-model",
        action="store_true",
        help="Only benchmark the first model (for quick dev iterations)"
    )
    args = parser.parse_args()

    start_server()
    entries = list_models()

    if args.first_model:
        entries = entries[:1]
        print("→ [dev] Running only the first model:", entries[0]["id"])

    prompt = "Hello, world! Summarize the latest AI news in one sentence."
    results = []

    for entry in entries:
        model_id = entry["id"]
        print(f"\n→ Benchmarking {model_id} …")
        unload_all()
        try:
            load_model(model_id)
        except subprocess.CalledProcessError:
            print(f"  ✗ Skipped {model_id}: could not load.", file=sys.stderr)
            continue

        try:
            stats = run_query(model_id, prompt)
            result = {
                "model": model_id,
                "params_b": entry["params_b"],
                "size_gb": entry["size_gb"],
                "architecture": entry["architecture"],
                "format": entry["format"],
                **stats
            }
            results.append(result)
            print(f"  ✔ {model_id}: {stats['tokens_per_second']} TPS, {stats['total_tokens']} tokens")
        except Exception as e:
            print(f"  ✗ Error querying {model_id}: {e}", file=sys.stderr)
        finally:
            unload_all()

    if not results:
        print("\nNo successful benchmarks—nothing to write.", file=sys.stderr)
        sys.exit(1)

    # write CSV
    keys = results[0].keys()
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print("\n✅ Done — results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
