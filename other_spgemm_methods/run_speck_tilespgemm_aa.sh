#!/bin/bash

# Record start time
START_TIME=$(date +%s)

# Relative Path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Project Root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/data/test}"
spECK_LOG_DIR="$SCRIPT_DIR/logs/spECK"
TileSpGEMM_LOG_DIR="$SCRIPT_DIR/logs/TileSpGEMM"
mkdir -p "$spECK_LOG_DIR/AA" "$TileSpGEMM_LOG_DIR/AA" || exit 1
if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset directory does not exist: $DATASET_DIR" >&2
    exit 1
fi
DATASET_DIR="$(cd "$DATASET_DIR" && pwd)" || exit 1
TIMEOUT_SECONDS=150

cd "$SCRIPT_DIR/spECK" || exit 1
make speck || exit 1

echo "Running spECK_AA..."
find "$DATASET_DIR" -name "*.mtx" -type f | sort | while read -r mtx_file; do
    base_name_=$(basename "$mtx_file" .mtx)

    echo "Run: ${base_name_}"
    timeout $TIMEOUT_SECONDS ./speck "$mtx_file" > "$spECK_LOG_DIR/AA/${base_name_}.log"

    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "Warn: Time out(Up to $(($TIMEOUT_SECONDS/60)) minutes), Skip"
        continue
    elif [ $exit_code -ne 0 ]; then
        echo "Warn: Run error, exit code: $exit_code"
        continue
    fi
done

cd "$SCRIPT_DIR/TileSpGEMM/src" || exit 1
make || exit 1

echo "Running TileSpGEMM_AA..."
find "$DATASET_DIR" -name "*.mtx" -type f | sort | while read -r mtx_file; do
    base_name_=$(basename "$mtx_file" .mtx)

    echo "Run: ${base_name_}"
    timeout $TIMEOUT_SECONDS ./test -d 0 -aat 0 "$mtx_file" > "$TileSpGEMM_LOG_DIR/AA/${base_name_}.log"

    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "Warn: Time out(Up to $(($TIMEOUT_SECONDS/60)) minutes), Skip"
        continue
    elif [ $exit_code -ne 0 ]; then
        echo "Warn: Run error, exit code: $exit_code"
        continue
    fi
done

# Calculate and print total elapsed time
END_TIME=$(date +%s)
ELAPSED_SECONDS=$((END_TIME - START_TIME))
ELAPSED_MINUTES=$((ELAPSED_SECONDS / 60))
ELAPSED_REMAINING_SECONDS=$((ELAPSED_SECONDS % 60))

echo "All other SpGEMM methods finished!"
echo "You can find all logs in ./logs!"
echo "Total elapsed time: ${ELAPSED_MINUTES} minutes ${ELAPSED_REMAINING_SECONDS} seconds (${ELAPSED_SECONDS} seconds in total)"
