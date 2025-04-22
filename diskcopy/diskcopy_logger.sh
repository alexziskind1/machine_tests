#!/bin/bash

set -e

echo "== Thunderbolt Dock Speed Logger =="

# === Select target directory ===
read -e -p "Enter target directory (Thunderbolt drive mount point): " TARGET_DIR
if [[ ! -d "$TARGET_DIR" ]]; then
  echo "❌ Target directory not found."
  exit 1
fi

# === Ask for GB size ===
read -p "Enter size of test file in GB (e.g., 10, 50, 100): " GB_SIZE
if ! [[ "$GB_SIZE" =~ ^[0-9]+$ ]]; then
  echo "❌ Invalid input. Please enter a number."
  exit 1
fi

BLOCK_COUNT=$((GB_SIZE * 1024))
echo "📝 Creating test file of $GB_SIZE GB ($BLOCK_COUNT blocks of 1MB)..."

SRC_FILE="tb5_testfile_${GB_SIZE}GB.bin"
DEST_FILE="$TARGET_DIR/tb5_testfile_copy.bin"
READ_BACK_FILE="copy_back_tb5.bin"
CSV_LOG="dd_speed_log.csv"

# === Create source file ===
dd if=/dev/urandom of="$SRC_FILE" bs=1m count=$BLOCK_COUNT status=none

echo "operation,timestamp,bytes" > "$CSV_LOG"

# === WRITE TEST ===
echo ""
echo "== WRITE TEST =="
{
  dd if="$SRC_FILE" bs=1m count=$BLOCK_COUNT of="$DEST_FILE" status=progress conv=notrunc &
  PID=$!
  while kill -0 $PID 2> /dev/null; do
    BYTES=$(du -k "$DEST_FILE" 2>/dev/null | awk '{print $1 * 1024}')
    echo "write,$(date +%s),$BYTES"
    sleep 0.2
  done
  echo "write,$(date +%s),$(du -k "$DEST_FILE" | awk '{print $1 * 1024}')"
} >> "$CSV_LOG"

# === READ TEST ===
echo ""
echo "== READ TEST =="
{
  dd if="$DEST_FILE" bs=1m count=$BLOCK_COUNT of="$READ_BACK_FILE" status=progress conv=notrunc &
  PID=$!
  while kill -0 $PID 2> /dev/null; do
    BYTES=$(du -k "$READ_BACK_FILE" 2>/dev/null | awk '{print $1 * 1024}')
    echo "read,$(date +%s),$BYTES"
    sleep 0.2
  done
  echo "read,$(date +%s),$(du -k "$READ_BACK_FILE" | awk '{print $1 * 1024}')"
} >> "$CSV_LOG"

echo "✅ Logging complete. Data saved to: $CSV_LOG"

# === Cleanup ===
read -p "Delete test files? (y/n): " CLEANUP
if [[ "$CLEANUP" == "y" ]]; then
  rm -f "$SRC_FILE" "$DEST_FILE" "$READ_BACK_FILE"
  echo "🧹 Test files removed."
fi
