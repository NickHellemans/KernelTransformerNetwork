#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch

from KernelTransformer.cfg import LAYERS
from KernelTransformer.cfg import MODEL_DIR
from KernelTransformer.evaluation import row_errors
from KernelTransformer.evaluation import run_validation
from KernelTransformer.Loader.model_loader import build_ktnconv
from KernelTransformer.Loader.model_loader import load_ktnnet
from KernelTransformer.Loader.data_loader import prepare_dataset
from KernelTransformer.util import enable_gpu


SRCS = ["pixel",] + LAYERS 


def load_ktnconv(source, transform, layer, **kwargs):
    ktnconv = build_ktnconv(layer, network=source)
    if layer == LAYERS[0]:
        return ktnconv

    model_name = f"{transform}{layer}.transform.pt"
    model_path = os.path.join(MODEL_DIR, model_name)
    sys.stderr.write("Load transformation from {}\n".format(model_path))
    ktn_state = torch.load(model_path, map_location=torch.device('cpu'))
    for name, params in ktnconv.named_parameters():
        if "src_kernel" in name or "src_bias" in name:
            ktn_state[name] = params
    ktnconv.load_state_dict(ktn_state)
    return ktnconv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--source', choices=["pascal", "imagenet", "coco"], default='pascal')
    parser.add_argument('--transform', choices=["pascal", "imagenet", "coco"], default='pascal')
    parser.add_argument('--input', choices=SRCS, default=None)
    parser.add_argument('target', choices=LAYERS)
    args = parser.parse_args()

    if args.input is None:
        ktnconv = load_ktnconv(args.source, args.transform, args.target)
        torch.save(ktnconv.state_dict(), "./models/ktn-model2.pt")
        _, valid_loader = prepare_dataset(args.target,
                                          src_cnn=args.source)
    else:
        ktnconv = load_ktnnet(args.source,
                              args.target,
                              transform=args.transform,
                              src=args.input)
        torch.save(ktnconv.state_dict(), "./models/ktn-modelwithinput.pt")
        _, valid_loader = prepare_dataset(args.target,
                                          src=args.input,
                                          src_cnn=args.source)

    
    if torch.cuda.is_available():
        sys.stderr.write("Enable GPU\n")
        ktnconv = enable_gpu(ktnconv, gpu=args.gpu)

    with torch.no_grad():
        diffs = run_validation(ktnconv, valid_loader)
    row_errors(diffs)

if __name__ == "__main__":
    main()

