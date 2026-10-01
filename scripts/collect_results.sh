#!/bin/bash
# Rebuild the benchmark three ways and record every number it prints.
# Run from anywhere; the script cds to the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source ./scripts/config.sh
mkdir -p results build
OUT=results/bench_results.txt
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"

cpu=$(awk -F: '/model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)
{
    echo "# ezNN benchmark results"
    echo "# produced by scripts/collect_results.sh"
    echo "# utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# uname: $(uname -srm)"
    echo "# cpu: ${cpu}"
    echo "# nproc: $(nproc)"
    echo "# OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
    echo "# gcc: $(gcc --version | head -1)"
    echo "# clang: $(clang --version | head -1)"
    echo "# iris.data sha256: $(sha256sum data/iris.data)"
} > "$OUT"

run_variant() {
    local label="$1"
    shift
    make clean >/dev/null
    make runezNN build/bench_eznn "$@"
    {
        echo
        echo "## ${label}"
        echo "# make $*"
    } >> "$OUT"
    ./build/bench_eznn "$SEED" "$TRIALS" \
        "$IRIS_LAYERS" "$IRIS_ACTS" "$IRIS_LR" "$IRIS_EPOCHS" "$IRIS_TRAIN" "$IRIS_TEST" \
        "$REG_LAYERS" "$REG_ACTS" "$REG_LR" "$REG_EPOCHS" "$REG_TRAIN" "$REG_TEST" \
        >> "$OUT"
}

run_variant "default -O2" CC=gcc
{
    echo
    echo "## cli default -O2"
} >> "$OUT"
bash scripts/run_examples.sh >> "$OUT"

run_variant "no-vectorize -O2 -fno-tree-vectorize" CC=gcc NOVEC=1
run_variant "openmp -O2 -fopenmp" CC=gcc OPENMP=1

make clean >/dev/null
mkdir -p build
make test CC=gcc > build/unit_default.log
{
    echo
    echo "## unit tests gcc -O2"
} >> "$OUT"
cat build/unit_default.log >> "$OUT"

make clean >/dev/null
mkdir -p build
make test CC=gcc SANITIZE=1 > build/unit_asan.log
{
    echo
    echo "## unit tests gcc asan+ubsan"
} >> "$OUT"
cat build/unit_asan.log >> "$OUT"

make clean >/dev/null
mkdir -p build
make test CC=gcc OPENMP=1 > build/unit_openmp.log
{
    echo
    echo "## unit tests gcc openmp"
} >> "$OUT"
cat build/unit_openmp.log >> "$OUT"

make clean >/dev/null
mkdir -p build
make test CC=clang > build/unit_clang.log
{
    echo
    echo "## unit tests clang -O2"
} >> "$OUT"
cat build/unit_clang.log >> "$OUT"

make clean >/dev/null
mkdir -p build
make test CC=clang SANITIZE=1 > build/unit_clang_asan.log
{
    echo
    echo "## unit tests clang asan+ubsan"
} >> "$OUT"
cat build/unit_clang_asan.log >> "$OUT"

python3 - "$OUT" << 'PY'
import sys
path = sys.argv[1]
text = open(path).read()
parts = text.split("\n## ")
sections = {}
for part in parts[1:]:
    header = part.splitlines()[0].strip()
    sections[header] = part

def val(body, key):
    for line in body.splitlines():
        if line.startswith(key + " "):
            return float(line.split()[1])
    raise SystemExit("missing %s" % key)

default = sections["default -O2"]
novec = sections["no-vectorize -O2 -fno-tree-vectorize"]
omp = sections["openmp -O2 -fopenmp"]
d = val(default, "micro_median_seconds")
n = val(novec, "micro_median_seconds")
o = val(omp, "micro_median_seconds")
with open(path, "a") as fh:
    fh.write("\n## derived ratios\n")
    fh.write("# micro_openmp_speedup_vs_default = default_median / openmp_median\n")
    fh.write("# micro_vectorize_speedup_vs_novec = novec_median / default_median\n")
    fh.write("micro_median_default %.8f\n" % d)
    fh.write("micro_median_novec %.8f\n" % n)
    fh.write("micro_median_openmp %.8f\n" % o)
    fh.write("micro_openmp_speedup_vs_default %.8f\n" % (d / o))
    fh.write("micro_vectorize_speedup_vs_novec %.8f\n" % (n / d))
    for label, body in (("default", default), ("novec", novec), ("openmp", omp)):
        fh.write("iris_test_accuracy_%s %.8f\n" % (label, val(body, "iris_test_accuracy")))
        fh.write("regression_test_loss_%s %.8f\n" % (label, val(body, "regression_test_loss")))
PY

if python3 -c "import numpy" >/dev/null 2>&1; then
    python3 tools/check_numpy.py
    {
        echo
        echo "## numpy crosscheck summary"
    } >> "$OUT"
    cat results/numpy_crosscheck.txt >> "$OUT"
    {
        echo
        echo "## reference comparison summary"
    } >> "$OUT"
    cat results/reference_comparison.txt >> "$OUT"
fi

echo "wrote $OUT"
