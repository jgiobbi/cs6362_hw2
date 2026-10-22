#!/usr/bin/env python
"""Round 2, goal 1: parameter-count check against Table 1 of Zhang et al. (ICLR 2017).

Instantiates each model exactly as used in round 1 (make_model from train_v2.py, imported unmodified)
and prints its parameter count against Table 1. MLP 3x512 is not in train_v2.py, so it is defined here
(3 hidden layers of 512). CPU only, no data, runs in seconds.

Run from the folder containing train_v2.py, in the same environment as the training jobs:
    python param_check.py
"""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")
import train_v2 as v2          # round-1 sweep script, not modified; v2.make_model is the code that ran

# Table 1 reference counts (from the paper / the assignment notes)
TABLE1 = {"MLP 1x512": 1_209_866, "MLP 3x512": 1_735_178, "AlexNet": 1_387_786, "Inception": 1_649_402}


def mlp_3x512():
    return nn.Sequential(nn.Flatten(),
                         nn.Linear(3 * 28 * 28, 512), nn.ReLU(),
                         nn.Linear(512, 512), nn.ReLU(),
                         nn.Linear(512, 512), nn.ReLU(),
                         nn.Linear(512, 10))


MODELS = {"MLP 1x512": lambda: v2.make_model("mlp"),
          "MLP 3x512": mlp_3x512,
          "AlexNet": lambda: v2.make_model("alexnet"),
          "Inception": lambda: v2.make_model("inception")}


def count(model):
    total = sum(p.numel() for p in model.parameters())
    bn = sum(p.numel() for m in model.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters())
    conv_bias = sum(m.bias.numel() for m in model.modules() if isinstance(m, nn.Conv2d) and m.bias is not None)
    return total, bn, conv_bias


def describe_alexnet(model):
    print("\nAlexNet layer-by-layer (input 1x3x28x28):")
    x, flat_in, first_fc_in, lrn_found = torch.zeros(1, 3, 28, 28), None, None, False
    model.eval()
    for i, layer in enumerate(model):
        x = layer(x)
        extra = ""
        if isinstance(layer, nn.Conv2d):
            extra = (f"out_channels={layer.out_channels} kernel={layer.kernel_size} stride={layer.stride} "
                     f"padding={layer.padding} bias={layer.bias is not None}")
        elif isinstance(layer, nn.MaxPool2d):
            extra = f"kernel={layer.kernel_size} stride={layer.stride} padding={layer.padding}"
        elif isinstance(layer, nn.LocalResponseNorm):
            lrn_found = True
            extra = f"size={layer.size} alpha={layer.alpha} beta={layer.beta} k={layer.k}"
        elif isinstance(layer, nn.Linear):
            extra = f"in={layer.in_features} out={layer.out_features}"
            if first_fc_in is None: first_fc_in = layer.in_features
        elif isinstance(layer, nn.Flatten):
            flat_in = x.shape[1]
        n_par = sum(p.numel() for p in layer.parameters())
        print(f"  [{i:2d}] {type(layer).__name__:18s} out_shape={tuple(x.shape)}  params={n_par:>9,}  {extra}")
    print(f"  LRN present: {lrn_found}")
    print(f"  flattened size before first FC: {flat_in} (first Linear in_features = {first_fc_in})")


def main():
    print("torch", torch.__version__)
    rows, ok_all = [], True
    for name, build in MODELS.items():
        torch.manual_seed(0)
        model = build()
        out = model.eval()(torch.zeros(2, 3, 28, 28))
        assert tuple(out.shape) == (2, 10), (name, tuple(out.shape))
        total, bn, conv_bias = count(model)
        ref = TABLE1[name]
        m_total, m_exbn = total == ref, (total - bn) == ref
        ok_all &= (m_total or m_exbn)
        rows.append((name, total, bn, total - bn, conv_bias, ref, m_total, m_exbn))

    print("\n| model | params (all) | BN gamma+beta | params excl. BN | conv biases | Table 1 | match (all) | match (excl. BN) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, total, bn, exbn, cb, ref, a, b in rows:
        print(f"| {name} | {total:,} | {bn:,} | {exbn:,} | {cb:,} | {ref:,} | {a} | {b} |")
    print("\n(Inception in Table 1 matches only if conv biases are counted and BN gamma/beta are not.)")

    describe_alexnet(MODELS["AlexNet"]())

    print("\nVERDICT:", "all four counts match Table 1 under the stated convention" if ok_all
          else "MISMATCH, paste this whole output back and do not rerun anything")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
