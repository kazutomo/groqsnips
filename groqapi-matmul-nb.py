#!/usr/bin/env python

import torch
import numpy as np
import groq.api as g
import groq.runner.tsp as tsp
import groq.api.nn as nn
import time, sys, os, re
import subprocess

N=1
M=1024
L=3000

NREPS = 1000
M1shape = (N, M)
M2shape = (M, L) # non-transposed shape
m1_data = []
m2_data = []
m2t_data = []
oracle_data = []
cpu_ets_usec = []

for i in range(0, NREPS):
    m1 = np.random.rand(N, M).astype(np.float16)
    m2 = np.random.rand(M, L).astype(np.float16)
    m2t = m2.transpose()
    m2t = m2t.copy(order='C')
    m1_data.append(m1)
    m2_data.append(m2)
    m2t_data.append(m2t)
    otmp = np.matmul(m1, m2, dtype=np.float16)
    oracle_data.append(otmp)

M1Tshape = M1shape[::-1]
M2Tshape = M2shape[::-1]

def rungroq_nonblocking(devfn=None):
    from groq.runtime import driver as runtime
    import groq.runtime

    matrix1 = g.input_tensor(shape=M1shape, dtype=g.float16, name="matrix1")
    matrix2 = g.input_tensor(shape=M2Tshape, dtype=g.float16, name="matrix2")

    class TopLevel(g.Component):  # Create our top level component
        def __init__(self):
            super().__init__()
            self.mm = nn.MatMul(name="MatMul", buffer_output=True, arith_mode_warmup=True)

        def build(self, mat1_mt, mat2_mt, time=0):
            with g.ResourceScope(name="mmscope", time=0) as mmscope:
                result_mt = self.mm(mat1_mt, mat2_mt, time=0)
                result_mt.name = "mm_result"
            return result_mt

    top = TopLevel()
    result = top(matrix1, matrix2, time=0)
    iop_file = g.compile(base_name="mmbench", result_tensor=result)

    #
    #g.write_visualizer_data("mmbench")

    p = subprocess.check_output(['iop-utils', 'stats', iop_file], encoding='utf8')
    cycles = int(re.findall("Program is (\S+)", p)[0])
    compute_usec = float(cycles)*1e-3/0.9 # 900MHz
    print('[[Cycles reported by iop-utils]]')
    print(f'  {iop_file}')
    print(f'  cycles={cycles}')
    print(f'  usec={compute_usec}')


    print('[[Groq nonblocking]]')
    shim = groq.runtime.DriverShim()
    if devfn:
        print(f"  devfn={devfn}")
        dptr = shim.get_device(devfn)
    else:
        dptr = shim.next_available_device()

    dptr.open() # open and lock the card
    prog = runtime.IOProgram(iop_file)
    dptr.load(prog[0])
    # entry points: monolihic, input, compute, output
    prog[0].entry_points[0] # monolithic EP, which includes input, compute, and output

    # the dimension of inputs and outputs. IODescriptor
    inputs_iodesc = prog[0].entry_points[0].input
    outputs_iodesc = prog[0].entry_points[0].output

    inbuf  = runtime.BufferArray(inputs_iodesc, NREPS)
    outbuf = runtime.BufferArray(outputs_iodesc, NREPS)

    for i in range(0, NREPS):
        inputsdic = {'matrix1': m1_data[i], 'matrix2': m2t_data[i] }
        if inputs_iodesc.tensors:
            for inpt in inputs_iodesc.tensors:
                #print(f'input_tensor: {inpt.name} {inpt.shape}')
                data = inputsdic.get(inpt.name)
                inpt.from_host(data, inbuf[i])
        else:
            print('Error: ep.input.tensors does not exist')
    dptr.invoke_nonblocking(inbuf[0], outbuf[0])

    maxloopcnt = 10000
    loopcntlog=[]
    ts_start = time.time()
    for i in range(1, NREPS):
        dptr.invoke_nonblocking(inbuf[i], outbuf[i])
        loopcnt = 0
        while not outbuf[i - 1].ready():
            loopcnt += 1
            if (loopcnt >= maxloopcnt):
                print('Timedout!')
                sys.exit(1)
        loopcntlog.append(loopcnt)
    loopcnt = 0
    while not outbuf[NREPS - 1].ready():
        loopcnt += 1
        if (loopcnt >= maxloopcnt):
            print('Timedout!')
            sys.exit(1)
    ts_stop = time.time()
    loopcntlog.append(loopcnt)
    perinvocation = (ts_stop-ts_start)/float(NREPS) * 1e6
    meanloopcnt = np.mean(loopcntlog)
    print(f'  Per invocation [usec]: {perinvocation:.3f}')
    #print(f'  Average wait loopcnt: {meanloopcnt:.1f}')

    for i in range(0, NREPS):
        res = {}
        for outt in outputs_iodesc.tensors:
            tmparr = outt.allocate_numpy_array()
            outt.to_host(outbuf[i], tmparr)
            res[outt.name] = tmparr

        mm_result=res['mm_result']
        v = np.allclose(oracle_data[i], res['mm_result'],
                        rtol=1e-2, atol=1e-2, equal_nan=True)
        if v == False:
            print('Error: Failed to verity')
            print(res['mm_result'])
            sys.exit(1)
        #print(np.abs(oracle_data[i]-res['mm_result']))
        #print(res['mm_result'])
    #print('Verification passed!')
    dptr.close()


if __name__ == "__main__":
    devfn = None

    if len(sys.argv) > 1:
        devfn = sys.argv[1]

    rungroq_nonblocking(devfn)
    print(f"L={L}")

    sys.exit(0)
