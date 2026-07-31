#!/usr/bin/env python3
import hashlib, json, struct
from pathlib import Path

FILES=("forward.fp16.bin","dY.fp16.bin","dX.fp32.bin","dW.fp32.bin",
       "W_master.fp32.bin","W_compute.fp16.bin","m.fp32.bin","v.fp32.bin")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def first_difference(expected,actual,dtype):
    eb=Path(expected).read_bytes();ab=Path(actual).read_bytes()
    size={"fp16":2,"fp32":4}[dtype];fmt={"fp16":"<H","fp32":"<I"}[dtype]
    for i in range(0,min(len(eb),len(ab)),size):
        if eb[i:i+size]!=ab[i:i+size]:
            return {"linear_index":i//size,"expected_raw_bits":hex(struct.unpack(fmt,eb[i:i+size])[0]),
                    "actual_raw_bits":hex(struct.unpack(fmt,ab[i:i+size])[0])}
    return {"linear_index":min(len(eb),len(ab))//size,"expected_size":len(eb),"actual_size":len(ab)}
def compare_step(ref,out):
    rows=[]
    for name in FILES:
        e,a=ref/name,out/name;eh,ah=sha(e),sha(a)
        row={"tensor":name,"expected_sha256":eh,"actual_sha256":ah,"match":eh==ah}
        if eh!=ah: row["first_difference"]=first_difference(e,a,"fp16" if "fp16" in name else "fp32")
        rows.append(row)
    state=hashlib.sha256()
    for name in FILES: state.update(name.encode()+ (out/name).read_bytes())
    return rows,state.hexdigest()
