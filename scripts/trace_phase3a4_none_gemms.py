#!/usr/bin/env python3
"""Minimal untimed process for external hipBLASLt logging of None GEMMs."""
import argparse, pathlib, sys
import torch

RELU_CASES=((1024,8,64,4,3,"Sigmoid"),(1024,8,128,4,8,"None"),(4096,32,64,2,16,"None"))

def run(tcnn,batch,inp,width,hidden,out,activation,outact):
    model=tcnn.Network(inp,out,{"otype":"HipBLASLtMLP","n_hidden_layers":hidden,"n_neurons":width,
        "activation":activation,"output_activation":outact},seed=20260721)
    x=(torch.randn(batch,inp,device="cuda")*.2).requires_grad_(); grad=torch.randn(batch,out,device="cuda")*.2
    model.zero_grad(set_to_none=True); model(x).backward(grad); torch.cuda.synchronize()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--bindings",required=True); p.add_argument("--sequence",choices=("none-only","relu-none"),required=True)
    a=p.parse_args(); sys.path.insert(0,str(pathlib.Path(a.bindings).resolve())); import tinycudann as tcnn
    if a.sequence=="relu-none":
        for batch,inp,width,hidden,out,outact in RELU_CASES:
            run(tcnn,batch,inp,width,hidden,out,"ReLU",outact)
    run(tcnn,4096,32,64,2,16,"None","None")
    print("PHASE3A4_NONE_GEMM_TRACE=COMPLETE")
if __name__=="__main__": main()
