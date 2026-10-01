# Example and benchmark configuration. scripts/run_examples.sh,
# scripts/collect_results.sh, and tools/check_numpy.py all read this file.
IRIS_LAYERS=4,16,3
IRIS_ACTS=ACT_RELU,ACT_SOFTMAX
IRIS_LR=0.05
IRIS_EPOCHS=400
IRIS_TRAIN=data/iris_train.csv
IRIS_TEST=data/iris_test.csv
REG_LAYERS=3,16,1
REG_ACTS=ACT_RELU,ACT_IDENTITY
REG_LR=0.05
REG_EPOCHS=400
REG_TRAIN=data/regression_train.csv
REG_TEST=data/regression_test.csv
SEED=1
TRIALS=5
# CI gates, not measured scores. Measured scores are in results/.
IRIS_MIN_TEST_ACCURACY=90
REG_MAX_TEST_LOSS=0.05
