#!/usr/bin/env bash

set -u

# Usage: scripts/kernel_10_autotuner.sh [N_SAMPLES]
# Collects all valid configurations, then builds and benchmarks a random
# sample of N_SAMPLES of them (default 60).
N_SAMPLES="${1:-60}"

# Define the range of values for each parameter
# (pruned for the T4: BK=64 and TM/TN >= 16 exceed shared memory or spill)
BK_VALUES=(8 16 32)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
WM_VALUES=(32 64 128 256)
WN_VALUES=(32 64 128 256)
WNITER_VALUES=(1 2 4 8)
TM_VALUES=(4 8)
TN_VALUES=(4 8)
NUM_THREADS_VALUES=(128 256)

cd "$(dirname "$0")"
cd "../build"

RUNNER="../src/runner.cu"
OUTPUT="../benchmark_results/kernel_10_autotune_results.txt"
mkdir -p "$(dirname "$OUTPUT")"

# Restore runner.cu when the script exits, however it exits
cp "$RUNNER" "$RUNNER.bak"
trap 'mv "$RUNNER.bak" "$RUNNER"' EXIT

# Clear the output file
echo "" > $OUTPUT

# Set GPU to use
export DEVICE="0"
WARPSIZE=32
MAX_SMEM_BYTES=49152   # static shared memory per block on the T4
MAX_ACCUMULATORS=128   # threadResults floats per thread before spilling

# Pass 1: collect configurations that satisfy the kernel's preconditions
CONFIGS=()
for BK in "${BK_VALUES[@]}"; do
for BM in "${BM_VALUES[@]}"; do
for BN in "${BN_VALUES[@]}"; do
for WM in "${WM_VALUES[@]}"; do
for WN in "${WN_VALUES[@]}"; do
for WN_ITER in "${WNITER_VALUES[@]}"; do
for TM in "${TM_VALUES[@]}"; do
for TN in "${TN_VALUES[@]}"; do
for NUM_THREADS in "${NUM_THREADS_VALUES[@]}"; do
NUM_WARPS=$(( NUM_THREADS / 32 ))
(( BN % WN == 0 && BM % WM == 0 )) || continue
(( (BN / WN) * (BM / WM) == NUM_WARPS )) || continue
(( (WM * WN) % (WARPSIZE * TM * TN * WN_ITER) == 0 )) || continue
WM_ITER=$(( (WM * WN) / (WARPSIZE * TM * TN * WN_ITER) ))
(( WM % WM_ITER == 0 && WN % WN_ITER == 0 )) || continue
(( (NUM_THREADS * 4) % BK == 0 )) || continue
(( (NUM_THREADS * 4) % BN == 0 )) || continue
(( BN % (16 * TN) == 0 )) || continue
(( BM % (16 * TM) == 0 )) || continue
(( (BM * BK) % (4 * NUM_THREADS) == 0 )) || continue
(( (BN * BK) % (4 * NUM_THREADS) == 0 )) || continue
(( (BM + BN) * BK * 4 <= MAX_SMEM_BYTES )) || continue
(( WM_ITER * TM * WN_ITER * TN <= MAX_ACCUMULATORS )) || continue
CONFIGS+=("$BK $BM $BN $WM $WN $WN_ITER $TM $TN $NUM_THREADS")
done
done
done
done
done
done
done
done
done

echo "${#CONFIGS[@]} valid configurations, sampling $N_SAMPLES"

# Pass 2: build and benchmark a random sample
CONFIG_NUM=0
while read -r BK BM BN WM WN WN_ITER TM TN NUM_THREADS; do
CONFIG_NUM=$(( CONFIG_NUM + 1 ))
echo ""

# Update the parameters in the source code
sed -i "s/const uint K10_NUM_THREADS = .*/const uint K10_NUM_THREADS = $NUM_THREADS;/" $RUNNER
sed -i "s/const uint K10_BN = .*/const uint K10_BN = $BN;/" $RUNNER
sed -i "s/const uint K10_BM = .*/const uint K10_BM = $BM;/" $RUNNER
sed -i "s/const uint K10_BK = .*/const uint K10_BK = $BK;/" $RUNNER
sed -i "s/const uint K10_WM = .*/const uint K10_WM = $WM;/" $RUNNER
sed -i "s/const uint K10_WN = .*/const uint K10_WN = $WN;/" $RUNNER
sed -i "s/const uint K10_WNITER = .*/const uint K10_WNITER = $WN_ITER;/" $RUNNER
sed -i "s/const uint K10_TM = .*/const uint K10_TM = $TM;/" $RUNNER
sed -i "s/const uint K10_TN = .*/const uint K10_TN = $TN;/" $RUNNER

# Rebuild the program
make

echo "($CONFIG_NUM/$N_SAMPLES): BK=$BK BM=$BM BN=$BN WM=$WM WN=$WN WN_ITER=$WN_ITER TM=$TM TN=$TN NUM_THREADS=$NUM_THREADS" 2>&1 | tee -a $OUTPUT
# Run the benchmark and get the result
# Kill the program after 8 seconds if it doesn't finish
timeout -v 8 ./sgemm 10 --check && timeout -v 8 ./sgemm 10 --bench | tee -a $OUTPUT
done < <(printf '%s\n' "${CONFIGS[@]}" | shuf -n "$N_SAMPLES")
