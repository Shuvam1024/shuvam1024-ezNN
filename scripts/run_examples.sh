#!/bin/sh
# Train the two worked examples through the runezNN CLI and reject
# results below the gates in scripts/config.sh.
set -eu
cd "$(dirname "$0")/.."
. ./scripts/config.sh
mkdir -p build

if [ ! -x ./runezNN ]; then
    echo "runezNN is missing; run make cli" >&2
    exit 1
fi

export EZNN_SEED="$SEED"
export EZNN_VERBOSE=0

./runezNN MODE_MULTICAT_CLASSIFICATION "$IRIS_LAYERS" "$IRIS_ACTS" \
    "$IRIS_LR" "$IRIS_EPOCHS" "$IRIS_TRAIN,$IRIS_TEST" build/iris.model \
    > build/example_iris.log
cat build/example_iris.log
iris_acc=$(awk '/^test loss:/ {print $NF}' build/example_iris.log)
echo "cli_iris_test_accuracy $iris_acc"
awk -v v="$iris_acc" -v m="$IRIS_MIN_TEST_ACCURACY" 'BEGIN { if (v+0 < m+0) exit 1 }' \
    || { echo "iris test accuracy $iris_acc is below $IRIS_MIN_TEST_ACCURACY" >&2; exit 1; }

./runezNN MODE_REGRESSION_L2 "$REG_LAYERS" "$REG_ACTS" \
    "$REG_LR" "$REG_EPOCHS" "$REG_TRAIN,$REG_TEST" build/regression.model \
    > build/example_reg.log
cat build/example_reg.log
reg_loss=$(awk '/^test loss:/ {print $3}' build/example_reg.log)
echo "cli_regression_test_loss $reg_loss"
awk -v v="$reg_loss" -v m="$REG_MAX_TEST_LOSS" 'BEGIN { if (v+0 > m+0) exit 1 }' \
    || { echo "regression test loss $reg_loss is above $REG_MAX_TEST_LOSS" >&2; exit 1; }
