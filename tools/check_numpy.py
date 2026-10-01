#!/usr/bin/env python3
"""Optional dev check. Not a library dependency.

Loads build/libeznn.so, checks the ezNNType layout against offsetof, and
compares a NumPy reimplementation of the forward pass and parameter
gradients to do_training(..., learning_rate=0). Also trains the Iris and
polynomial examples three ways and writes:

  results/numpy_crosscheck.txt
  results/reference_comparison.txt

The NumPy and scikit-learn trainers do not share libc rand() with ezNN.
They are references, not bit-reproductions.
"""
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_LAYERS = 16

ACT_IDENTITY, ACT_RELU, ACT_TANH, ACT_SIGMOID, ACT_SOFTMAX = range(5)
MODE_L2, MODE_L1, MODE_BINARY, MODE_MULTI = range(4)
PROB_EPS = 1e-7
GRAD_LIMIT = 1e-3
FWD_LIMIT = 1e-4


class EzNN(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("nlayers", ctypes.c_int),
        ("layer_sizes", ctypes.c_int * MAX_LAYERS),
        ("weights", ctypes.POINTER(ctypes.POINTER(ctypes.c_float)) * (MAX_LAYERS - 1)),
        ("biases", ctypes.POINTER(ctypes.c_float) * (MAX_LAYERS - 1)),
        ("weights_grad", ctypes.POINTER(ctypes.POINTER(ctypes.c_float)) * (MAX_LAYERS - 1)),
        ("biases_grad", ctypes.POINTER(ctypes.c_float) * (MAX_LAYERS - 1)),
        ("activations", ctypes.c_int * (MAX_LAYERS - 1)),
        ("act_inputs", ctypes.POINTER(ctypes.c_float) * MAX_LAYERS),
        ("act_outputs", ctypes.POINTER(ctypes.c_float) * MAX_LAYERS),
        ("error", ctypes.POINTER(ctypes.c_float) * MAX_LAYERS),
        ("grad", ctypes.POINTER(ctypes.c_float) * MAX_LAYERS),
    ]


def load_config():
    cfg = {}
    for line in (ROOT / "scripts" / "config.sh").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        cfg[key.strip()] = val.strip()
    return cfg


def ensure_shared():
    so = ROOT / "build" / "libeznn.so"
    subprocess.check_call(["make", "shared", "CC=gcc"], cwd=ROOT)
    if not so.is_file():
        raise SystemExit("build/libeznn.so was not produced")
    return so


def check_layout(lib):
    src = r"""
#include <stddef.h>
#include <stdio.h>
#include "ezNN.h"
int main(void) {
    printf("sizeof %zu\n", sizeof(ezNNType));
#define F(name) printf("%s %zu %zu\n", #name, offsetof(ezNNType, name), sizeof(((ezNNType *)0)->name))
    F(mode);
    F(nlayers);
    F(layer_sizes);
    F(weights);
    F(biases);
    F(weights_grad);
    F(biases_grad);
    F(activations);
    F(act_inputs);
    F(act_outputs);
    F(error);
    F(grad);
    return 0;
}
"""
    probe_c = ROOT / "build" / "layout_probe.c"
    probe = ROOT / "build" / "layout_probe"
    probe_c.write_text(src)
    subprocess.check_call(
        ["gcc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), "-o", str(probe), str(probe_c)],
        cwd=ROOT,
    )
    text = subprocess.check_output([str(probe)], cwd=ROOT, text=True)
    parsed = {}
    for line in text.splitlines():
        parts = line.split()
        parsed[parts[0]] = tuple(int(x) for x in parts[1:])
    sizeof = parsed.pop("sizeof")[0]
    if ctypes.sizeof(EzNN) != sizeof:
        raise SystemExit("ctypes sizeof %d != C sizeof %d" % (ctypes.sizeof(EzNN), sizeof))
    for name, (off, size) in parsed.items():
        field = getattr(EzNN, name)
        if field.offset != off or field.size != size:
            raise SystemExit(
                "layout mismatch on %s: ctypes off %d size %d, C off %d size %d"
                % (name, field.offset, field.size, off, size)
            )
    return sizeof


def bind(lib):
    lib.init_ezNN.argtypes = [
        ctypes.POINTER(EzNN), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.free_ezNN.argtypes = [ctypes.POINTER(EzNN)]
    lib.ezNN_seed.argtypes = [ctypes.c_uint]
    lib.ezNN_set_verbose.argtypes = [ctypes.c_int]
    lib.do_training.argtypes = [
        ctypes.POINTER(EzNN),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
    ]
    lib.do_inference.argtypes = [
        ctypes.POINTER(EzNN),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ]
    fp = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
    for name in (
        "get_regression_l2_loss",
        "get_regression_l1_loss",
        "get_binary_classification_loss",
        "get_multicat_classification_loss",
        "get_binary_classification_accuracy",
        "get_multicat_classification_accuracy",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.POINTER(EzNN), fp, ctypes.c_int, fp]
        fn.restype = ctypes.c_float


def make_net(lib, mode, sizes, acts):
    nn = EzNN()
    s = (ctypes.c_int * len(sizes))(*sizes)
    a = (ctypes.c_int * len(acts))(*acts)
    lib.init_ezNN(ctypes.byref(nn), mode, len(sizes), s, a)
    if nn.nlayers != len(sizes):
        raise SystemExit("init_ezNN failed for %s" % (sizes,))
    return nn


def set_params(nn, weights, biases):
    for layer, (W, b) in enumerate(zip(weights, biases)):
        inn, out = W.shape
        for i in range(inn):
            row = nn.weights[layer][i]
            for o in range(out):
                row[o] = float(W[i, o])
        for o in range(out):
            nn.biases[layer][o] = float(b[o])


def read_grads(nn):
    weights, biases = [], []
    for layer in range(nn.nlayers - 1):
        inn = nn.layer_sizes[layer]
        out = nn.layer_sizes[layer + 1]
        W = np.empty((inn, out), dtype=np.float64)
        b = np.empty(out, dtype=np.float64)
        for i in range(inn):
            row = nn.weights_grad[layer][i]
            for o in range(out):
                W[i, o] = row[o]
        for o in range(out):
            b[o] = nn.biases_grad[layer][o]
        weights.append(W)
        biases.append(b)
    return weights, biases


def c_forward(lib, nn, x):
    nout = nn.layer_sizes[nn.nlayers - 1]
    xin = (ctypes.c_float * len(x))(*[float(v) for v in x])
    yout = (ctypes.c_float * nout)()
    rows = (ctypes.POINTER(ctypes.c_float) * 1)(xin)
    outs = (ctypes.POINTER(ctypes.c_float) * 1)(yout)
    lib.do_inference(ctypes.byref(nn), rows, 1, outs)
    return np.array([yout[i] for i in range(nout)], dtype=np.float64)


def c_grads(lib, nn, row):
    buf = (ctypes.c_float * len(row))(*[float(v) for v in row])
    rows = (ctypes.POINTER(ctypes.c_float) * 1)(buf)
    lib.ezNN_set_verbose(0)
    lib.do_training(ctypes.byref(nn), rows, 1, ctypes.c_float(0), 1, 0)
    return read_grads(nn)


def activate(act, z):
    z = np.asarray(z, dtype=np.float64)
    if act == ACT_IDENTITY:
        return z.copy()
    if act == ACT_RELU:
        return np.maximum(z, 0.0)
    if act == ACT_TANH:
        return np.tanh(z)
    if act == ACT_SIGMOID:
        out = np.empty_like(z)
        pos = z >= 0.0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out
    if act == ACT_SOFTMAX:
        shift = z - np.max(z)
        e = np.exp(shift)
        return e / np.sum(e)
    raise ValueError(act)


def dactivate(act, z, a, err):
    if act == ACT_IDENTITY:
        return err.copy()
    if act == ACT_RELU:
        deriv = np.where(z > 0.0, 1.0, np.where(z < 0.0, 0.0, 0.5))
        return err * deriv
    if act == ACT_TANH:
        return (1.0 - a * a) * err
    if act == ACT_SIGMOID:
        return a * (1.0 - a) * err
    if act == ACT_SOFTMAX:
        g = np.zeros_like(a)
        n = a.shape[0]
        for k in range(n):
            acc = 0.0
            for j in range(n):
                jac = a[k] * (1.0 - a[k]) if j == k else -a[k] * a[j]
                acc += err[j] * jac
            g[k] = acc
        return g
    raise ValueError(act)


def loss_dlda(mode, a, y):
    if mode == MODE_L2:
        return a - y
    if mode == MODE_L1:
        return np.where((a - y) >= 0.0, 1.0, -1.0)
    if mode == MODE_BINARY:
        ai = np.clip(a, PROB_EPS, 1.0 - PROB_EPS)
        return (ai - y) / (ai * (1.0 - ai))
    if mode == MODE_MULTI:
        ai = np.clip(a, PROB_EPS, 1.0 - PROB_EPS)
        return -y / ai
    raise ValueError(mode)


def fused_dz(mode, act, a, y):
    if mode == MODE_BINARY and act == ACT_SIGMOID:
        return a - y
    if mode == MODE_MULTI and act == ACT_SOFTMAX:
        return a * np.sum(y) - y
    return None


def numpy_forward(weights, biases, acts, x):
    a = np.asarray(x, dtype=np.float64).copy()
    zs = []
    acts_out = [a]
    for W, b, act in zip(weights, biases, acts):
        z = a @ W + b
        zs.append(z)
        a = activate(act, z)
        acts_out.append(a)
    return acts_out, zs


def numpy_grads(mode, sizes, acts, weights, biases, row):
    nin = sizes[0]
    nout = sizes[-1]
    x = np.asarray(row[:nin], dtype=np.float64)
    if mode == MODE_MULTI:
        y = np.zeros(nout, dtype=np.float64)
        cl = int(row[nin])
        if 0 <= cl < nout:
            y[cl] = 1.0
    else:
        y = np.asarray(row[nin:nin + nout], dtype=np.float64)
    acts_out, zs = numpy_forward(weights, biases, acts, x)
    nlayers = len(sizes)
    dz = [None] * nlayers
    fused = fused_dz(mode, acts[-1], acts_out[-1], y)
    if fused is not None:
        dz[-1] = fused
    else:
        dlda = loss_dlda(mode, acts_out[-1], y)
        dz[-1] = dactivate(acts[-1], zs[-1], acts_out[-1], dlda)
    for layer in range(nlayers - 2, 0, -1):
        Wnext = weights[layer]
        err = dz[layer + 1] @ Wnext.T
        dz[layer] = dactivate(acts[layer - 1], zs[layer - 1], acts_out[layer], err)
    gW, gb = [], []
    for layer in range(nlayers - 1):
        g = dz[layer + 1]
        a_prev = acts_out[layer]
        gW.append(np.outer(a_prev, g))
        gb.append(g.copy())
    return acts_out[-1], gW, gb


def W(*rows):
    return np.array(rows, dtype=np.float64)


def b(*vals):
    return np.array(vals, dtype=np.float64)


CASES = [
    dict(name="id-l2", mode=MODE_L2, sizes=[2, 2], acts=[ACT_IDENTITY],
         weights=[W((0.4, -0.2), (0.1, 0.3))], biases=[b(0.05, -0.1)],
         row=[0.5, -1.25, 0.25, -0.5]),
    dict(name="relu-pos-l2", mode=MODE_L2, sizes=[2, 2], acts=[ACT_RELU],
         weights=[W((0.8, 0.7), (0.6, 0.9))], biases=[b(0.5, 0.4)],
         row=[1.0, 1.0, 0.2, -0.4]),
    dict(name="relu-neg-l2", mode=MODE_L2, sizes=[2, 2], acts=[ACT_RELU],
         weights=[W((-0.8, -0.7), (-0.6, -0.9))], biases=[b(-0.5, -0.4)],
         row=[1.0, 1.0, 0.5, -0.2]),
    dict(name="relu-kink-l2", mode=MODE_L2, sizes=[2, 2], acts=[ACT_RELU],
         weights=[W((0.0, 0.0), (0.0, 0.0))], biases=[b(0.0, 0.0)],
         row=[1.0, 0.0, -1.0, 1.0]),
    dict(name="tanh-l2", mode=MODE_L2, sizes=[2, 2], acts=[ACT_TANH],
         weights=[W((0.3, -0.2), (0.15, 0.25))], biases=[b(0.1, -0.05)],
         row=[0.5, -0.25, 0.0, 0.2]),
    dict(name="sigmoid-l2", mode=MODE_L2, sizes=[2, 2], acts=[ACT_SIGMOID],
         weights=[W((0.2, -0.15), (0.1, 0.25))], biases=[b(0.05, -0.1)],
         row=[0.4, -0.3, 0.2, 0.7]),
    dict(name="softmax-l2", mode=MODE_L2, sizes=[2, 3], acts=[ACT_SOFTMAX],
         weights=[W((0.2, -0.1, 0.15), (-0.2, 0.25, 0.05))], biases=[b(0.1, -0.05, 0.0)],
         row=[0.4, -0.3, 0.2, 0.5, 0.3]),
    dict(name="bce-identity", mode=MODE_BINARY, sizes=[1, 2], acts=[ACT_IDENTITY],
         weights=[W((0.3, 0.5))], biases=[b(0.1, 0.2)],
         row=[1.0, 1.0, 0.0]),
    dict(name="bce-sigmoid", mode=MODE_BINARY, sizes=[2, 2], acts=[ACT_SIGMOID],
         weights=[W((0.3, -0.2), (0.15, 0.4))], biases=[b(0.1, -0.05)],
         row=[0.5, -0.25, 1.0, 0.0]),
    dict(name="ce-identity", mode=MODE_MULTI, sizes=[2, 3], acts=[ACT_IDENTITY],
         weights=[W((0.2, 0.5, 0.3), (0.1, 0.1, 0.1))], biases=[b(0.0, 0.0, 0.0)],
         row=[1.0, 0.0, 1.0]),
    dict(name="ce-softmax", mode=MODE_MULTI, sizes=[2, 3], acts=[ACT_SOFTMAX],
         weights=[W((0.2, -0.3, 0.15), (-0.1, 0.25, 0.05))], biases=[b(0.05, -0.1, 0.2)],
         row=[0.4, -0.2, 2.0]),
    dict(name="l1-identity", mode=MODE_L1, sizes=[2, 2], acts=[ACT_IDENTITY],
         weights=[W((0.5, -0.25), (0.0, 0.1))], biases=[b(0.25, 0.5)],
         row=[1.0, 0.5, 0.0, 1.0]),
    dict(name="chain-tanh-sigmoid-l2", mode=MODE_L2, sizes=[2, 3, 2], acts=[ACT_TANH, ACT_SIGMOID],
         weights=[
             W((0.3, -0.2, 0.4), (0.15, 0.25, -0.1)),
             W((0.4, -0.3), (0.2, 0.5), (-0.25, 0.15)),
         ],
         biases=[b(0.1, -0.05, 0.2), b(0.05, -0.1)],
         row=[0.4, -0.2, 0.2, 0.8]),
    dict(name="chain-relu-softmax-ce", mode=MODE_MULTI, sizes=[2, 3, 3], acts=[ACT_RELU, ACT_SOFTMAX],
         weights=[
             W((0.4, 0.5, 0.35), (0.45, 0.3, 0.55)),
             W((0.2, -0.1, 0.15), (-0.2, 0.3, 0.05), (0.1, 0.2, -0.25)),
         ],
         biases=[b(0.3, 0.25, 0.4), b(0.05, -0.05, 0.1)],
         row=[1.0, 1.0, 1.0]),
]


def run_cases(lib):
    lines = ["# numpy cross-check of forward outputs and parameter gradients",
             "# numpy %s" % np.__version__,
             "# threshold_grad_max_abs %.8g" % GRAD_LIMIT,
             "# threshold_forward_max_abs %.8g" % FWD_LIMIT]
    worst_g = 0.0
    worst_f = 0.0
    for case in CASES:
        nn = make_net(lib, case["mode"], case["sizes"], case["acts"])
        set_params(nn, case["weights"], case["biases"])
        c_y = c_forward(lib, nn, case["row"][:case["sizes"][0]])
        cW, cb = c_grads(lib, nn, case["row"])
        n_y, nW, nb = numpy_grads(
            case["mode"], case["sizes"], case["acts"], case["weights"], case["biases"], case["row"]
        )
        fwd = float(np.max(np.abs(c_y - n_y)))
        diffs = [np.max(np.abs(a - b)) for a, b in zip(cW, nW)]
        diffs += [np.max(np.abs(a - b)) for a, b in zip(cb, nb)]
        grad = float(max(diffs))
        worst_f = max(worst_f, fwd)
        worst_g = max(worst_g, grad)
        lines.append("%s forward_max_abs %.8e grad_max_abs %.8e" % (case["name"], fwd, grad))
        lib.free_ezNN(ctypes.byref(nn))
    lines.append("max_forward_abs %.8e" % worst_f)
    lines.append("max_grad_abs %.8e" % worst_g)
    lines.append("cases %d" % len(CASES))
    status = "pass" if worst_g <= GRAD_LIMIT and worst_f <= FWD_LIMIT else "fail"
    lines.append("status %s" % status)
    path = ROOT / "results" / "numpy_crosscheck.txt"
    path.write_text("\n".join(lines) + "\n")
    if status != "pass":
        raise SystemExit("numpy cross-check failed; see %s" % path)
    return path


def rows_from_csv(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def pack_rows(data):
    n, cols = data.shape
    storage = [(ctypes.c_float * cols)(*[float(v) for v in data[i]]) for i in range(n)]
    ptrs = (ctypes.POINTER(ctypes.c_float) * n)(*storage)
    return storage, ptrs


def eznn_train_eval(lib, mode, sizes, acts, train, test, lr, epochs, seed):
    lib.ezNN_seed(ctypes.c_uint(seed))
    lib.ezNN_set_verbose(0)
    nn = make_net(lib, mode, sizes, acts)
    _keep, ptrs = pack_rows(train)
    lib.do_training(ctypes.byref(nn), ptrs, train.shape[0], ctypes.c_float(lr), epochs, 1)
    metrics = eval_eznn(lib, nn, train, "train")
    metrics.update(eval_eznn(lib, nn, test, "test"))
    lib.free_ezNN(ctypes.byref(nn))
    return metrics


def eval_eznn(lib, nn, data, prefix):
    n, _cols = data.shape
    nout = nn.layer_sizes[nn.nlayers - 1]
    storage = [(ctypes.c_float * nout)() for _ in range(n)]
    outs = (ctypes.POINTER(ctypes.c_float) * n)(*storage)
    _keep, ptrs = pack_rows(data)
    lib.do_inference(ctypes.byref(nn), ptrs, n, outs)
    mode = nn.mode
    if mode == MODE_L2:
        loss = lib.get_regression_l2_loss(ctypes.byref(nn), ptrs, n, outs)
        return {"%s_loss" % prefix: float(loss)}
    if mode == MODE_MULTI:
        loss = lib.get_multicat_classification_loss(ctypes.byref(nn), ptrs, n, outs)
        acc = lib.get_multicat_classification_accuracy(ctypes.byref(nn), ptrs, n, outs)
        return {"%s_loss" % prefix: float(loss), "%s_accuracy" % prefix: float(acc)}
    raise SystemExit("eval mode %d is not used by the examples" % mode)


def targets_of(mode, row, nin, nout):
    if mode == MODE_MULTI:
        y = np.zeros(nout, dtype=np.float64)
        cl = int(row[nin])
        if 0 <= cl < nout:
            y[cl] = 1.0
        return y
    return row[nin:nin + nout].astype(np.float64)


def numpy_sgd(mode, sizes, acts, train, lr, epochs, seed):
    rng = np.random.default_rng(seed)
    weights, biases = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        eps = np.sqrt(1.0 / float(fan_in * fan_out))
        weights.append(rng.uniform(-eps, eps, size=(fan_in, fan_out)))
        biases.append(rng.uniform(-eps, eps, size=(fan_out,)))
    n = train.shape[0]
    nin = sizes[0]
    nout = sizes[-1]
    for _epoch in range(epochs):
        order = rng.permutation(n)
        for idx in order:
            row = train[idx]
            x = row[:nin]
            y = targets_of(mode, row, nin, nout)
            acts_out, zs = numpy_forward(weights, biases, acts, x)
            dz = [None] * len(sizes)
            fused = fused_dz(mode, acts[-1], acts_out[-1], y)
            if fused is not None:
                dz[-1] = fused
            else:
                dz[-1] = dactivate(acts[-1], zs[-1], acts_out[-1], loss_dlda(mode, acts_out[-1], y))
            for layer in range(len(sizes) - 2, 0, -1):
                err = dz[layer + 1] @ weights[layer].T
                dz[layer] = dactivate(acts[layer - 1], zs[layer - 1], acts_out[layer], err)
            for layer in range(len(sizes) - 1):
                g = dz[layer + 1]
                weights[layer] -= lr * np.outer(acts_out[layer], g)
                biases[layer] -= lr * g
    return weights, biases


def numpy_predict(weights, biases, acts, data, nin):
    out = []
    for row in data:
        acts_out, _zs = numpy_forward(weights, biases, acts, row[:nin])
        out.append(acts_out[-1])
    return np.vstack(out)


def multiclass_metrics(probs, labels):
    clipped = np.clip(probs, PROB_EPS, 1.0 - PROB_EPS)
    n = labels.shape[0]
    loss = 0.0
    correct = 0
    for i in range(n):
        cl = int(labels[i])
        loss -= np.log(clipped[i, cl])
        if int(np.argmax(probs[i])) == cl:
            correct += 1
    return loss / n, 100.0 * correct / n


def regression_mse(pred, y):
    d = pred.reshape(-1) - y.reshape(-1)
    return float(np.mean(d * d))


def run_references(lib, cfg):
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    seed = int(cfg["SEED"])
    iris_sizes = [int(x) for x in cfg["IRIS_LAYERS"].split(",")]
    reg_sizes = [int(x) for x in cfg["REG_LAYERS"].split(",")]
    iris_acts = [ACT_RELU, ACT_SOFTMAX]
    reg_acts = [ACT_RELU, ACT_IDENTITY]
    iris_lr = float(cfg["IRIS_LR"])
    reg_lr = float(cfg["REG_LR"])
    iris_epochs = int(cfg["IRIS_EPOCHS"])
    reg_epochs = int(cfg["REG_EPOCHS"])

    iris_train = rows_from_csv(ROOT / cfg["IRIS_TRAIN"])
    iris_test = rows_from_csv(ROOT / cfg["IRIS_TEST"])
    reg_train = rows_from_csv(ROOT / cfg["REG_TRAIN"])
    reg_test = rows_from_csv(ROOT / cfg["REG_TEST"])

    ez_iris = eznn_train_eval(
        lib, MODE_MULTI, iris_sizes, iris_acts, iris_train, iris_test, iris_lr, iris_epochs, seed
    )
    ez_reg = eznn_train_eval(
        lib, MODE_L2, reg_sizes, reg_acts, reg_train, reg_test, reg_lr, reg_epochs, seed
    )

    w, b = numpy_sgd(MODE_MULTI, iris_sizes, iris_acts, iris_train, iris_lr, iris_epochs, seed)
    tr_p = numpy_predict(w, b, iris_acts, iris_train, iris_sizes[0])
    te_p = numpy_predict(w, b, iris_acts, iris_test, iris_sizes[0])
    np_iris_tr_loss, np_iris_tr_acc = multiclass_metrics(tr_p, iris_train[:, -1])
    np_iris_te_loss, np_iris_te_acc = multiclass_metrics(te_p, iris_test[:, -1])

    w, b = numpy_sgd(MODE_L2, reg_sizes, reg_acts, reg_train, reg_lr, reg_epochs, seed)
    np_reg_tr = regression_mse(numpy_predict(w, b, reg_acts, reg_train, reg_sizes[0]), reg_train[:, -1])
    np_reg_te = regression_mse(numpy_predict(w, b, reg_acts, reg_test, reg_sizes[0]), reg_test[:, -1])

    clf = MLPClassifier(
        hidden_layer_sizes=(iris_sizes[1],),
        activation="relu",
        solver="adam",
        alpha=0.0,
        batch_size=iris_train.shape[0],
        learning_rate_init=iris_lr,
        max_iter=iris_epochs,
        random_state=seed,
        tol=0.0,
        n_iter_no_change=iris_epochs,
        early_stopping=False,
    )
    clf.fit(iris_train[:, :-1], iris_train[:, -1].astype(int))
    sk_te = clf.predict_proba(iris_test[:, :-1])
    sk_tr = clf.predict_proba(iris_train[:, :-1])
    # predict_proba column order follows clf.classes_
    sk_te_loss, sk_te_acc = multiclass_metrics(sk_te[:, clf.classes_.astype(int)], iris_test[:, -1])
    sk_tr_loss, sk_tr_acc = multiclass_metrics(sk_tr[:, clf.classes_.astype(int)], iris_train[:, -1])

    reg = MLPRegressor(
        hidden_layer_sizes=(reg_sizes[1],),
        activation="relu",
        solver="adam",
        alpha=0.0,
        batch_size=reg_train.shape[0],
        learning_rate_init=reg_lr,
        max_iter=reg_epochs,
        random_state=seed,
        tol=0.0,
        n_iter_no_change=reg_epochs,
        early_stopping=False,
    )
    reg.fit(reg_train[:, :-1], reg_train[:, -1])
    sk_reg_tr = regression_mse(reg.predict(reg_train[:, :-1]), reg_train[:, -1])
    sk_reg_te = regression_mse(reg.predict(reg_test[:, :-1]), reg_test[:, -1])

    import sklearn
    lines = [
        "# reference comparison on the committed example splits",
        "# ezNN uses libc rand via ezNN_seed, online SGD, no weight decay",
        "# numpy_sgd uses numpy.random.default_rng, the same init scale and online update, a different stream and an unbiased shuffle",
        "# sklearn uses Adam, alpha 0, full-batch minibatches, random_state as given",
        "numpy %s" % np.__version__,
        "sklearn %s" % sklearn.__version__,
        "seed %d" % seed,
        "iris_layers %s" % cfg["IRIS_LAYERS"],
        "iris_epochs %d" % iris_epochs,
        "iris_lr %.8g" % iris_lr,
        "iris_eznn_train_loss %.8f" % ez_iris["train_loss"],
        "iris_eznn_train_accuracy %.8f" % ez_iris["train_accuracy"],
        "iris_eznn_test_loss %.8f" % ez_iris["test_loss"],
        "iris_eznn_test_accuracy %.8f" % ez_iris["test_accuracy"],
        "iris_numpy_sgd_train_loss %.8f" % np_iris_tr_loss,
        "iris_numpy_sgd_train_accuracy %.8f" % np_iris_tr_acc,
        "iris_numpy_sgd_test_loss %.8f" % np_iris_te_loss,
        "iris_numpy_sgd_test_accuracy %.8f" % np_iris_te_acc,
        "iris_sklearn_train_loss %.8f" % sk_tr_loss,
        "iris_sklearn_train_accuracy %.8f" % sk_tr_acc,
        "iris_sklearn_test_loss %.8f" % sk_te_loss,
        "iris_sklearn_test_accuracy %.8f" % sk_te_acc,
        "iris_sklearn_n_iter %d" % int(clf.n_iter_),
        "regression_layers %s" % cfg["REG_LAYERS"],
        "regression_epochs %d" % reg_epochs,
        "regression_lr %.8g" % reg_lr,
        "regression_eznn_train_loss %.8f" % ez_reg["train_loss"],
        "regression_eznn_test_loss %.8f" % ez_reg["test_loss"],
        "regression_numpy_sgd_train_loss %.8f" % np_reg_tr,
        "regression_numpy_sgd_test_loss %.8f" % np_reg_te,
        "regression_sklearn_train_loss %.8f" % sk_reg_tr,
        "regression_sklearn_test_loss %.8f" % sk_reg_te,
        "regression_sklearn_n_iter %d" % int(reg.n_iter_),
    ]
    path = ROOT / "results" / "reference_comparison.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "build").mkdir(exist_ok=True)
    so = ensure_shared()
    lib = ctypes.CDLL(str(so))
    bind(lib)
    sizeof = check_layout(lib)
    cross = run_cases(lib)
    ref = run_references(lib, load_config())
    print("layout_sizeof %d" % sizeof)
    print("wrote %s" % cross)
    print("wrote %s" % ref)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print("check_numpy.py: %s" % exc, file=sys.stderr)
        raise
