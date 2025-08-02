# Hardware Mapping System

This document explains how to use the hardware mapping system for clean chart legends.

## Files

- `hardware_mapping.json` - Contains mappings from raw file path components to clean display names
- `prompt_hardware_comparison.py` - Updated script that uses the mapping file

## How It Works

1. The script extracts hardware names from file paths (e.g., "rtx5090", "m4max")
2. It looks up these raw names in `hardware_mapping.json`
3. If found, it uses the clean name (e.g., "RTX 5090 + RTX 5060Ti", "M4 Max 128GB")
4. If not found, it applies basic formatting to the raw name

## Usage

### See what hardware names are detected:
```bash
python prompt_hardware_comparison.py --show-hardware-names
```

This will show you:
- Raw names extracted from file paths
- Which ones are mapped vs unmapped
- Suggestions for adding new mappings

### Adding new hardware mappings:

1. Run the script with `--show-hardware-names` to see unmapped hardware
2. Edit `hardware_mapping.json` and add entries like:
```json
{
  "hardware_mappings": {
    "your_raw_name": "Your Clean Display Name",
    "rtx4090": "RTX 4090",
    "m5max": "M5 Max 256GB"
  }
}
```

3. The script will automatically use the clean names in charts

## Mapping Categories

- **hardware_mappings**: Exact matches for hardware names
- **fallback_patterns**: Regex patterns for broader matching
- **model_mappings**: Clean names for AI models
- **quantization_mappings**: Clean names for quantization levels

## Example Output

Before mapping:
- Legend shows: "rtx5090", "m4max", "fw ryzen"

After mapping:
- Legend shows: "RTX 5090 + RTX 5060Ti", "M4 Max 128GB", "Framework Ryzen AI 395"

## Benefits

✅ Clean, professional chart legends  
✅ No code changes needed for new hardware  
✅ Consistent naming across all charts  
✅ Easy to maintain and update  
