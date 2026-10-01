# ezNN — dependency-free C99 multilayer perceptron.
# Override CC, or add SANITIZE=1 / OPENMP=1 / NOVEC=1 on the make command.
# Changing those flags does not rebuild existing binaries; `make clean` first.

CC       ?= gcc
OPT      := -O2
EXTRA    :=
LDFLAGS  ?=
LDLIBS   ?= -lm
CPPFLAGS ?= -I.

ifeq ($(SANITIZE),1)
OPT   := -O1 -g -fno-omit-frame-pointer
EXTRA += -fsanitize=address,undefined -fno-sanitize-recover=undefined
LDFLAGS += -fsanitize=address,undefined
endif

ifeq ($(OPENMP),1)
EXTRA += -fopenmp -DEZNN_OPENMP
LDFLAGS += -fopenmp
endif

ifeq ($(NOVEC),1)
EXTRA += -fno-tree-vectorize
endif

CFLAGS := -std=c99 -Wall -Wextra -Werror $(OPT) $(EXTRA)

.PHONY: all lib cli test bench examples data clean check shared

all: lib cli

lib: build/libeznn.a

shared: build/libeznn.so

cli: runezNN

build/libeznn.a: ezNN.c ezNN.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -c ezNN.c -o build/ezNN.o
	ar rcs $@ build/ezNN.o

build/libeznn.so: ezNN.c ezNN.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared -o $@ ezNN.c $(LDFLAGS) $(LDLIBS)

runezNN: main.c ezNN.c ezNN.h readwrite_csv.c readwrite_csv.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ main.c readwrite_csv.c ezNN.c $(LDFLAGS) $(LDLIBS)

build/test_eznn: tests/test_eznn.c ezNN.c ezNN.h readwrite_csv.c readwrite_csv.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ tests/test_eznn.c readwrite_csv.c ezNN.c $(LDFLAGS) $(LDLIBS)

test: build/test_eznn
	./build/test_eznn

build/bench_eznn: bench/bench_eznn.c ezNN.c ezNN.h readwrite_csv.c readwrite_csv.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ bench/bench_eznn.c ezNN.c readwrite_csv.c $(LDFLAGS) $(LDLIBS)

build/gen_datasets: tools/gen_datasets.c data/iris.data
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ tools/gen_datasets.c $(LDFLAGS) $(LDLIBS)

data: build/gen_datasets
	./build/gen_datasets

bench: scripts/collect_results.sh
	bash scripts/collect_results.sh

examples: runezNN data
	sh scripts/run_examples.sh

check: test examples

clean:
	rm -rf build runezNN
