# ezNN

ezNN is a dependency-free C99 multilayer perceptron. The library is `ezNN.c` and `ezNN.h` and links only libm. `runezNN` trains and evaluates a network from CSV files. Training is online stochastic gradient descent in `float`, one sample at a time.

The public struct and the original function signatures are unchanged. `ezNN_seed` and `ezNN_set_verbose` are additions.

## Build

```sh
make            # static library build/libeznn.a and the runezNN CLI
make test       # C unit tests
make examples   # regenerate data if needed, train the two examples
make bench      # rewrite results/bench_results.txt
make shared     # build/libeznn.so, used by tools/check_numpy.py
make clean
```

`make SANITIZE=1 test` builds with `-O1 -g -fsanitize=address,undefined`. `make OPENMP=1` defines `EZNN_OPENMP` and links `-fopenmp`. `make NOVEC=1` adds `-fno-tree-vectorize`. Flags are not part of the target names, so run `make clean` before switching them. The default line is `-std=c99 -Wall -Wextra -Werror -O2`.

CMake is optional and covers the same targets:

```sh
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake
ctest --test-dir build-cmake
```

`EZNN_OPENMP`, `EZNN_SANITIZE`, and `EZNN_NOVEC` are CMake options. CI uses the Makefile.

## Layout

`ezNNType` is one network. `nlayers` counts the input layer, so a net declared as `4,16,3` has `nlayers == 3`, two weight matrices, and two activations. `layer_sizes[]` holds those widths. `weights[layer]` is a `float**`: row `in`, column `out`.

```
z[out] = bias[out] + sum_in act_outputs[layer][in] * weights[layer][in][out]
```

Each weight row is its own allocation. That matches the original ABI and makes a single sample's matmul a short dot product per output, with the output index contiguous inside a row. It is a poor layout for a GEMM-style kernel: there is no one dense `in * out` slab, and a backward pass that reduces across outputs has to chase a pointer per input row.

`act_inputs`, `act_outputs`, `error`, and `grad` are single-sample buffers. `do_inference` and `do_training` overwrite them. Two calls on the same network are not safe to run at the same time, and the library is not reentrant. `rand` is process-global, so two networks trained with `reset != 0` also share one shuffle and init stream.

`init_ezNN` zeroes weights and biases. A fresh draw happens only inside `do_training` when `reset != 0`. The scale on layer `i` is

```
epsilon = sqrt(1 / (fan_in * fan_out))
```

with each parameter uniform on `[-epsilon, epsilon]` via `rand() / RAND_MAX`. `ezNN_seed(seed)` makes every later reset replay `srand(seed)`. If `ezNN_seed` was never called, that path still uses `srand((unsigned)time(NULL))`.

## Training

`do_training` shuffles sample indices each epoch with Fisher-Yates and `rand() % (k + 1)`. That modulo is biased; it is the historical shuffle and the tests lock the seeded sequence on a given libc. The sequence is not portable across C libraries. CI accuracy gates are wide for that reason.

For each sample the forward pass writes pre-activations and activations, then backprop writes `weights_grad` and `biases_grad` for that sample only (no accumulation across a batch) and steps

```
parameter -= learning_rate * gradient
```

There is no momentum, weight decay, or minibatch. Multiclass targets in the data are one class index. The index is expanded to a one-hot vector inside the step. An out-of-range index becomes an all-zero target and does not read off the end of the vector.

`do_inference` always returns the full output width, including multiclass, where those values are class scores. `do_classification_hard` writes 0/1 per unit for binary mode (threshold 0.5, ties go to 1) and writes the argmax into `outputs[i][0]` for multiclass. Accuracy helpers return a percentage in `[0, 100]`.

With verbose left on (the default), each epoch runs an extra forward pass over the training set only to print a line. `ezNN_set_verbose(0)` skips that pass. The parameter updates do not depend on the flag.

## Losses and activation derivatives

Backprop differentiates a per-sample objective. The printed loss is the mean of a related quantity and is not always that objective.

| Mode | Differentiated objective | Printed mean |
| --- | --- | --- |
| L2 | `0.5 * sum_j (a - y)^2`, so `dL/da = a - y` | `sum_j (a - y)^2` |
| L1 | `sum_j \|a - y\|`, `sign(0) = +1` | same sum of absolute errors |
| Binary | `-[y log a + (1 - y) log(1 - a)]` per unit | same, both classes |
| Multiclass | `-log(a_class)` | same |

The factor of two between the L2 objective and the printed metric is absorbed into the learning rate. Reported binary and multiclass losses clip probabilities to `[1e-7, 1 - 1e-7]` before the log. The gradient uses the same clip on the explicit `dL/da` path.

Two canonical pairs replace that quotient with an algebraically equal `dL/dz`, because `a(1-a)` in the sigmoid Jacobian and `a` in the softmax Jacobian cancel a division that blows up when a probability is ~0:

- sigmoid + binary cross-entropy: `dL/dz = a - y`
- softmax + multiclass cross-entropy: `dL/dz = a * sum(y) - y`

Any other pairing (L2 through sigmoid, cross-entropy through identity, and so on) still multiplies `dL/da` by the activation Jacobian. Softmax's Jacobian is the full `n` by `n` matrix, not a diagonal.

Sigmoid branches on the sign and evaluates `exp` in double, so a large negative logit does not overflow to a NaN. Softmax subtracts the max logit, again in double, then writes `float`. `tanh` is `tanhf`. ReLU at exactly zero uses the subgradient `1/2`, which is what a central difference sees when it straddles the kink. Away from zero the derivative is 0 or 1.

Hidden activations are whatever the caller passes. Binary mode does not require a sigmoid output, and multiclass mode does not require softmax. The fused gradient is used only for the pairs above. The CLI rejects a binary network whose last activation is not sigmoid, and a multiclass network whose last activation is not softmax, because those are the intended CLI configurations. The library itself accepts other combinations.

## Model file

```
byte 0: mode + (nlayers << 3)
then nlayers little-endian uint16 widths
then activations packed two per byte, low nibble first
then native-endian float weights, row by row, then biases, for each layer
```

Widths must fit in 16 bits. `nlayers` must be in `[2, 16]` and `mode` in `[0, 3]`. `load_model_from_file` expects a zeroed or uninitialized struct. On failure it zeroes the struct. Passing a live network leaks the previous allocations. `free_ezNN` is safe on a zeroed struct and safe to call twice.

## CLI

```
runezNN <mode> <layers> <activations> <learning_rate> <epochs> \
        <train.csv[,test.csv]> <model> [column_map]
```

Modes: `MODE_REGRESSION_L2`, `MODE_REGRESSION_L1`, `MODE_BINARY_CLASSIFICATION`, `MODE_MULTICAT_CLASSIFICATION`.

Activations: `ACT_IDENTITY`, `ACT_RELU`, `ACT_TANH`, `ACT_SIGMOID`, `ACT_SOFTMAX`. One per layer boundary, so one fewer than the layer list.

`EZNN_SEED` is an unsigned seed. `EZNN_VERBOSE=0` hides the per-epoch line and prints one training summary after the fit. When a test CSV is present the tool saves the model, loads it back, and prints a `test loss:` line (and accuracy for classification). The optional column map is a comma-separated list of CSV column indices, inputs then targets. Multiclass still has a single target column.

Worked examples, both with `EZNN_SEED=1` and the hyperparameters in `scripts/config.sh`:

```sh
runezNN MODE_MULTICAT_CLASSIFICATION 4,16,3 ACT_RELU,ACT_SOFTMAX \
        0.05 400 data/iris_train.csv,data/iris_test.csv build/iris.model

runezNN MODE_REGRESSION_L2 3,16,1 ACT_RELU,ACT_IDENTITY \
        0.05 400 data/regression_train.csv,data/regression_test.csv build/regression.model
```

## Data

`data/iris.data` is the UCI Iris file (150 records, no header). Its sha256 is recorded in `results/bench_results.txt`. `tools/gen_datasets.c` reads that file and writes:

- `data/iris.csv`, all 150 rows, class index `0/1/2`
- `data/iris_train.csv` (120) and `data/iris_test.csv` (30): within each block of 50, the first 40 are train and the last 10 are test
- features z-scored with the training-set population mean and standard deviation (divisor `n`, not `n - 1`)
- `data/regression_train.csv` (200) and `data/regression_test.csv` (50): `x` in `[-1, 1]` from an xorshift32 started at 1, `y = 0.5*x0 - 0.8*x0^3 + 0.4*x1^2 - 0.7*x2`

`data/MANIFEST.txt` is generator output, including the Iris moments. `make data` rebuilds these files. CI fails if the rebuild differs from the commit.

## Tests

`tests/test_eznn.c` is a single binary of plain checks. It covers invalid init, stable sigmoid and softmax, the loss values (binary loss includes the `y = 0` term), exact dyadic SGD steps, and central-difference checks for identity, ReLU (positive, negative, and the kink), tanh, sigmoid, and softmax, each under L2, plus L1, explicit binary and multiclass cross-entropy through an identity output, the two fused cross-entropy paths, and two-layer chains. It also checks hard classification when the sample count is not the output width, save/load for several depths, truncated files, a width that does not fit in 16 bits, seeded training, verbose versus quiet bitwise equality, CSV round-trip, and (when built with `OPENMP=1`) that 1 thread and 2 threads produce the same parameters on a wide layer.

`tools/check_numpy.py` is an optional developer check. It needs NumPy and, for the reference section, scikit-learn. It is not linked into the library. It compiles an `offsetof` probe, compares that to the ctypes view of `ezNNType`, and checks the same forward and gradient cases against a NumPy reimplementation.

## Results

Produced by `bash scripts/collect_results.sh` on the machine recorded in `results/bench_results.txt` (`2026-10-01T21:02:31Z`, 4-core Intel Xeon, `OMP_NUM_THREADS=4`, gcc 13.3.0). Quality numbers below are from the default `-O2` section. The no-vectorize and OpenMP builds printed the same Iris test accuracy and the same regression test loss (`results/bench_results.txt`, derived ratios). Times are medians of `do_training` wall time, including the random init on `reset != 0` and excluding the verbose full-set pass.

| Task | Architecture | Epochs | LR | Train | Test | Median seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Iris | 4, 16, 3 ReLU, softmax | 400 | 0.05 | loss 0.04871212, accuracy 98.33333588 | loss 0.02103265, accuracy 100.00000000 | 0.01518015 |
| Regression | 3, 16, 1 ReLU, identity | 400 | 0.05 | loss 0.00060739 | loss 0.00136819 | 0.01691436 |

Iris test accuracy 100 on this split is 30/30. The CLI, after a save/load round trip, printed `test loss: 0.021033, accuracy: 100.000000` and `test loss: 0.001368` (`results/bench_results.txt`, cli section). `scripts/run_examples.sh` rejects an Iris test accuracy below 90 or a regression test loss above 0.05. Those gates are in `scripts/config.sh`.

The same splits, same depth, learning rate, and epoch count, trained by other implementations (`results/reference_comparison.txt`, NumPy 2.4.4, scikit-learn 1.9.1, seed 1). ezNN uses libc `rand`. The NumPy trainer uses `numpy.random.default_rng` with the same init scale and the same online update, but a different stream and an unbiased shuffle. scikit-learn is Adam, `alpha=0`, full-batch updates, `random_state=1`, and ran all 400 iterations (`n_iter` 400 for both).

| Implementation | Iris test accuracy | Iris test loss | Regression test loss |
| --- | --- | --- | --- |
| ezNN online SGD | 100.00000000 | 0.02103265 | 0.00136819 |
| NumPy online SGD | 100.00000000 | 0.00253435 | 0.00161407 |
| sklearn Adam | 100.00000000 | 0.00140245 | 0.00272765 |

Gradient check against NumPy, 14 cases: max forward absolute error `2.38418578e-08`, max gradient absolute error `9.53674317e-08` (`results/numpy_crosscheck.txt`).

Unit tests in that same results file:

| Build | Result |
| --- | --- |
| gcc `-O2` | 223 checks passed, 0 failed |
| gcc ASan + UBSan | 223 checks passed, 0 failed |
| gcc OpenMP | 224 checks passed, 0 failed |
| clang `-O2` | 223 checks passed, 0 failed |
| clang ASan + UBSan | 223 checks passed, 0 failed |

The extra OpenMP check is the 1-thread versus 2-thread bitwise comparison.

## OpenMP and the inner loop

`EZNN_OPENMP` parallelizes only loops whose outputs (or input rows, for the backward matmul) are independent and are reduced in the original `j` order, so a parameter ends up bitwise-identical to the one-thread sum. The cut-in is a width of at least 64. Iris and the polynomial net are narrower than that, so their OpenMP times in the results file are a recompile with `-fopenmp`, not a parallel layer. The timing comparison is the synthetic microbench: 768-768-128, ReLU then identity, L2, 64 samples, 8 epochs, learning rate 0.001, seed 1, 3 trials. It is a timing probe; `micro_loss_finite 1` only says the loss was finite.

| Variant | Micro median seconds |
| --- | --- |
| default `-O2` | 0.44792263 |
| `-O2 -fno-tree-vectorize` | 0.52487286 |
| `-O2 -fopenmp`, 4 threads | 0.29323351 |

`micro_openmp_speedup_vs_default` is `0.44792263 / 0.29323351 = 1.52752879`. `micro_vectorize_speedup_vs_novec` is `0.52487286 / 0.44792263 = 1.17179358`. Four threads did not give a 4x wall-time drop. The matmul inner loop is contiguous in the output index so the auto-vectorizer has a straight-line saxpy, but each weight row is a separate allocation, and the reduction order is fixed so the compiler is not invited to reassociate with `-ffast-math`.

## CI

`.github/workflows/ci.yml` runs on push and pull request, gcc and clang. Each compiler runs the unit tests at `-O2`, under ASan+UBSan, and with OpenMP, regenerates `data/` and diffs it against the commit, then runs the CLI examples. It does not compare wall times.

## Limitations

- One sample in flight per network, and a process-global `rand` stream. Not thread-safe.
- Online SGD only.
- `float` parameters. The L2 gradient is the derivative of half the printed sum of squares.
- Shuffle bias from `rand() % n`, and a seed that replays only on the same libc.
- Model files are native-endian, 16-bit widths, at most 16 layers.
- Weight rows are not one contiguous matrix.
- OpenMP is a compile flag, off by default, and only splits layers of width at least 64.
- No MNIST runner. A download script for the raw files was not added.
