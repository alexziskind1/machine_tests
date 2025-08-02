#!/usr/bin/env python3
"""
Migration script to reorganize existing results into the new nested folder structure.

This script will:
1. Scan existing CSV files in the results directory and subdirectories
2. Parse model information from the CSV content and filenames
3. Move files to the appropriate nested directories: /results/<model>/<quantization>/<hardware>/
"""

import os
import csv
import shutil
import re
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

def load_hardware_mapping() -> Dict:
    """Load hardware mapping configuration."""
    try:
        with open("hardware_mapping.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load hardware mapping: {e}")
        return {"hardware_mappings": {}, "quantization_mappings": {}, "model_mappings": {}}

def parse_model_info(model_name: str, hardware_mapping: Dict) -> Tuple[str, str]:
    """
    Parse model name to extract base model and quantization.
    Enhanced to handle LM Studio format patterns and use hardware mapping.
    """
    model_name = model_name.lower()
    original_model = model_name
    
    # Extract quantization pattern - enhanced for your data
    quant_patterns = [
        r'q\d+(?:_[a-z0-9]+)*',  # q4_k_m, q8_0, q4km, q3, etc.
        r'fp\d+',                # fp16, fp32
        r'int\d+',               # int4, int8
        r'bf\d+',                # bf16
        r'mlx@?\d*bit',          # mlx4bit, mlx8bit, mlx@4bit
        r'mlx\d+bit',            # mlx4bit, mlx8bit
    ]
    
    quantization = "unknown"
    base_model = model_name
    
    # First try to find quantization in the model name
    for pattern in quant_patterns:
        match = re.search(pattern, model_name)
        if match:
            quantization = match.group()
            # Normalize common patterns
            if quantization == 'q4km':
                quantization = 'q4_k_m'
            elif 'mlx' in quantization:
                quantization = re.sub(r'mlx@?(\d+)bit', r'mlx\1bit', quantization)
            
            # Remove quantization from model name to get base model
            base_model = re.sub(f'[-_:@]{re.escape(match.group())}', '', model_name)
            break
    
    # Clean up base model name
    base_model = re.sub(r'[-_:](instruct|chat|it)$', '', base_model)
    base_model = re.sub(r'[^\w\.]', '_', base_model)
    
    # Handle special cases where quantization might be in filename instead
    if quantization == "unknown":
        # Look for patterns like "q8_cuda12", "q4km_vulkan", etc.
        extended_patterns = [
            r'q\d+[a-z]*(?:_[a-z0-9]+)*',  # q8_cuda12, q4km_vulkan
        ]
        for pattern in extended_patterns:
            match = re.search(pattern, original_model)
            if match:
                found = match.group()
                # Extract just the quantization part
                quant_match = re.match(r'(q\d+[a-z]*)', found)
                if quant_match:
                    quantization = quant_match.group()
                    if quantization == 'q4km':
                        quantization = 'q4_k_m'
                break
    
    # Apply model mapping if available
    model_mappings = hardware_mapping.get("model_mappings", {})
    for key, mapped_name in model_mappings.items():
        if key in base_model:
            base_model = key
            break
            
    # Apply quantization mapping if available  
    quant_mappings = hardware_mapping.get("quantization_mappings", {})
    for key, mapped_quant in quant_mappings.items():
        if key in quantization:
            quantization = key
            break
    
    return base_model, quantization

def normalize_hardware_name(hardware: str, hardware_mapping: Dict) -> str:
    """Normalize hardware name using hardware mapping."""
    hardware_lower = hardware.lower()
    
    # Check direct mappings first
    hardware_mappings = hardware_mapping.get("hardware_mappings", {})
    if hardware_lower in hardware_mappings:
        return hardware_lower
    
    # Check fallback patterns
    fallback_patterns = hardware_mapping.get("fallback_patterns", {})
    for pattern, normalized in fallback_patterns.items():
        if pattern in hardware_lower:
            return pattern
            
    return hardware

def extract_hardware_from_filename(filename: str) -> str:
    """Extract hardware information from filename patterns."""
    filename_lower = filename.lower()
    
    # Hardware patterns to look for
    hardware_patterns = [
        (r'm\d+max_\d+gb', lambda m: m.group().replace('_', '')),  # m4max_128gb -> m4max128gb
        (r'm\d+pro_\d+gb', lambda m: m.group().replace('_', '')),  # m4pro_64gb -> m4pro64gb  
        (r'm\d+ultra_\d+gb', lambda m: m.group().replace('_', '')), # m3ultra_96gb -> m3ultra96gb
        (r'rtx\d+w?rtx\d+ti?_\d+gb', lambda m: m.group().replace('_', '')), # rtx5090wrtx5060ti_48gb
        (r'fw_ryzen_ai_\d+_\d+gb', lambda m: 'fw_ryzen_ai_' + m.group().split('_')[-2]), # fw_ryzen_ai_395_64gb
        (r'm\d+\w*', lambda m: m.group()),  # m1, m2, m3, m4 etc.
        (r'rtx\d+\w*', lambda m: m.group()),  # rtx3080, rtx4090 etc.
    ]
    
    for pattern, extractor in hardware_patterns:
        match = re.search(pattern, filename_lower)
        if match:
            return extractor(match)
    
    return "unknown_hardware"

def read_csv_info(csv_path: str) -> Optional[Dict]:
    """Read model information from a CSV file."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return None
            
            # Get model from first row
            first_row = rows[0]
            model = first_row.get('model', '')
            base_model = first_row.get('base_model', '')
            quantization = first_row.get('quantization', '')
            hardware = first_row.get('hardware', '')
            
            # If not in CSV, try to extract from filename
            filename = os.path.basename(csv_path)
            if not hardware:
                hardware = extract_hardware_from_filename(filename)
            
            return {
                'model': model,
                'base_model': base_model,
                'quantization': quantization,
                'hardware': hardware,
                'row_count': len(rows),
                'filename': filename
            }
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def migrate_file(csv_path: str, target_dir: str, hardware_mapping: Dict, dry_run: bool = True) -> bool:
    """Migrate a single CSV file to the new structure."""
    info = read_csv_info(csv_path)
    if not info:
        return False
    
    model = info['model']
    base_model = info.get('base_model')
    quantization = info.get('quantization')
    hardware = info.get('hardware', 'unknown_hardware')
    filename = info['filename']
    
    # If the CSV doesn't have parsed info, parse from model name
    if not base_model or not quantization:
        if model:
            base_model, quantization = parse_model_info(model, hardware_mapping)
            
            # If quantization is still unknown, try to extract from filename
            if quantization == "unknown":
                filename_lower = filename.lower()
                # Look for quantization patterns in filename
                if 'q4km' in filename_lower:
                    quantization = 'q4_k_m'
                elif 'q8' in filename_lower:
                    quantization = 'q8_0'
                elif 'q3' in filename_lower:
                    quantization = 'q3'
                elif 'mlx4bit' in filename_lower:
                    quantization = 'mlx4bit'
                elif 'mlx8bit' in filename_lower:
                    quantization = 'mlx8bit'
        else:
            # Try to parse from current directory structure or filename
            current_dir = os.path.basename(os.path.dirname(csv_path))
            if current_dir != 'results':  # We're in a subdirectory
                base_model, quantization = parse_model_info(current_dir, hardware_mapping)
                
                # Try filename patterns for quantization if still unknown
                if quantization == "unknown":
                    filename_lower = filename.lower()
                    if 'q4km' in filename_lower:
                        quantization = 'q4_k_m'
                    elif 'q8' in filename_lower:
                        quantization = 'q8_0'
                    elif 'q3' in filename_lower:
                        quantization = 'q3'
                    elif 'mlx4bit' in filename_lower:
                        quantization = 'mlx4bit'
                    elif 'mlx8bit' in filename_lower:
                        quantization = 'mlx8bit'
            else:
                print(f"No model information found in {csv_path}")
                return False
    
    # Normalize hardware name using mapping
    hardware = normalize_hardware_name(hardware, hardware_mapping)
    
    # Create target directory structure
    nested_dir = os.path.join(target_dir, base_model, quantization, hardware)
    target_path = os.path.join(nested_dir, filename)
    
    # Check if we're already in the correct location
    current_structure = os.path.relpath(csv_path, target_dir)
    expected_structure = os.path.join(base_model, quantization, hardware, filename)
    
    if current_structure == expected_structure:
        if dry_run:
            print(f"Already in correct location: {csv_path}")
        return True
    
    if dry_run:
        print(f"Would move: {csv_path}")
        print(f"       to: {target_path}")
        print(f"    Model: {model} -> {base_model}/{quantization}/{hardware}")
        print(f"     Rows: {info['row_count']}")
        print()
    else:
        try:
            os.makedirs(nested_dir, exist_ok=True)
            
            # Check if target file already exists
            if os.path.exists(target_path):
                print(f"Warning: Target file already exists: {target_path}")
                return False
                
            shutil.move(csv_path, target_path)
            print(f"Moved: {csv_path} -> {target_path}")
            
            # Clean up empty directories
            try:
                old_dir = os.path.dirname(csv_path)
                if old_dir != target_dir and not os.listdir(old_dir):
                    os.rmdir(old_dir)
                    print(f"Removed empty directory: {old_dir}")
            except OSError:
                pass  # Directory not empty or other issue
                
        except Exception as e:
            print(f"Error moving {csv_path}: {e}")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Migrate existing results to nested folder structure")
    parser.add_argument("--results-dir", default="results", help="Results directory to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually moving files")
    parser.add_argument("--force", action="store_true", help="Proceed even if target files exist")
    
    args = parser.parse_args()
    
    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found")
        return 1
    
    # Load hardware mapping
    hardware_mapping = load_hardware_mapping()
    
    # Find all CSV files in the results directory and subdirectories
    csv_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return 0
    
    print(f"Found {len(csv_files)} CSV files to migrate")
    if args.dry_run:
        print("DRY RUN - no files will be moved")
    print("=" * 80)
    
    migrated = 0
    skipped = 0
    for csv_file in csv_files:
        if migrate_file(csv_file, results_dir, hardware_mapping, dry_run=args.dry_run):
            migrated += 1
        else:
            skipped += 1
    
    print("=" * 80)
    if args.dry_run:
        print(f"Would migrate {migrated} files, skip {skipped} files out of {len(csv_files)} total")
        print("\nTo actually perform the migration, run:")
        print(f"python migrate_results.py --results-dir {results_dir}")
    else:
        print(f"Successfully migrated {migrated} files, skipped {skipped} files out of {len(csv_files)} total")
    
    return 0

if __name__ == "__main__":
    exit(main())
