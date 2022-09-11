#!/usr/bin/env python

#
# benchmarking different host APIs (groqflow, tsprunner, and groq API)
#
# Kazutomo Yoshii <kazutomo.yoshii@gmail.com>
#

import groq.api as g
import groq.runner.tsp as tsp
import groq.api.nn as nn
from groqflow import groqit

import torch
import numpy as np
import time, sys, os, re
import subprocess


#
# Preparing data
#
NELEMS = 16 # args
mode = 'sq'

if mode == 'sq':
    S=0
    N=256   # args
    M1shape = (N, N)
    M2shape = (N, N)
    m1_data = []
    m2_data = []
    m2t_data = []
    oracle_data = []
    cpu_ets_usec = []
    print('prep data: ', end='')
    for i in range(0, NELEMS):
        #m1 = np.random.rand(N, N).astype(np.float16)
        m1 = np.identity(N).astype(np.float16)
        m2 = np.random.rand(N, N).astype(np.float16)
        #m2.fill(float(i))
        m2t = m2.transpose()
        m2t = m2t.copy(order='C')
        m1_data.append(m1)
        m2_data.append(m2)
        m2t_data.append(m2t)
        st = time.time()
        otmp = np.matmul(m1, m2, dtype=np.float16)
        et = time.time() - st
        cpu_ets_usec.append(et*1e6)
        oracle_data.append(otmp)
        print('.', end='')
    print()
    perinvocation_cpu = np.mean(cpu_ets_usec)
else:
    print('Unsupported mode: {mode}')
    sys.exit(0)

M1Tshape = M1shape[::-1]
M2Tshape = M2shape[::-1]

print(f'mode: {mode}')
print(f'M1shape: {M1shape}')
print(f'M2shape: {M2shape}')
print(f'CPU per invocation [usec]: {perinvocation_cpu:.3f}')

def rungroqflow():
    class SQMM(torch.nn.Module):
        def __init__(self):
            super(SQMM, self).__init__()
        def forward(self, a, b):
            return torch.matmul(a,b)

    inputs = {"a": torch.from_numpy(m1),
              "b": torch.from_numpy(m2)}
        
    sqmm = SQMM()
    pytorch_outputs = sqmm(**inputs)

    groq_model = groqit(sqmm, inputs)
    ets = []
    for i in range(10):
        st = time.time()
        groq_outputs = groq_model(**inputs)
        et = time.time() - st
        ets.append(et*1e6)
    
    v = np.allclose(pytorch_outputs, groq_outputs,
                    rtol=1e-3, atol=1e-3, equal_nan=True)
    if v == False:
        print('Error: Failed to verity')
        print(res['mm_result'])
        sys.exit(1)
    perinvocation = np.mean(ets)
    print(f'  Per invocation [usec]: {perinvocation:.3f}')

def rungroqflowabunch():
    class SQMM(torch.nn.Module):
        def __init__(self):
            super(SQMM, self).__init__()
        def forward(self, a, b):
            return torch.matmul(a,b)


    inputsone = {"a": torch.from_numpy(m1_data[0]), "b": torch.from_numpy(m2_data[0])}
    inputs = [{"a": torch.from_numpy(a), "b": torch.from_numpy(b)} for a, b in zip(m1_data, m2_data)]
        
    sqmm = SQMM()

    groq_model = groqit(sqmm, inputsone)
    st = time.time()
    groq_outputs = groq_model.run_abunch(input_collection=inputs)
    et = time.time() - st

    for a, b, res in zip(m1_data, m2_data, groq_outputs):
        pytorch_outputs = sqmm(a=torch.from_numpy(a), b=torch.from_numpy(b))
        v = np.allclose(pytorch_outputs, res,
                        rtol=1e-3, atol=1e-3, equal_nan=True)
        if v == False:
            print('Error: Failed to verity')
            print(res['mm_result'])
            sys.exit(1)

    perinvocation = et*1e6/float(NELEMS)
    print(f'  Per invocation [usec]: {perinvocation:.3f}')

    
# FLIP
#matrix1 = g.input_tensor(shape=M1Tshape, dtype=g.float32, name="matrix1")
#matrix2 = g.input_tensor(shape=M2shape, dtype=g.float16, name="matrix2")
matrix1 = g.input_tensor(shape=M1shape, dtype=g.float16, name="matrix1")
matrix2 = g.input_tensor(shape=M2Tshape, dtype=g.float16, name="matrix2")

class TopLevel(g.Component):  # Create our top level component
    def __init__(self):
        super().__init__()
        self.mm = nn.MatMul(name="MatMul", buffer_output=True)

    def build(self, mat1_mt, mat2_mt, time=0):
        with g.ResourceScope(name="mmscope", time=0) as mmscope:
            result_mt = self.mm(mat1_mt, mat2_mt, time=0)
            result_mt.name = "mm_result"
        return result_mt

top = TopLevel()
result = top(matrix1, matrix2, time=0)
iop_file = g.compile(base_name="mmbench", result_tensor=result)

def print_iop_stats(iopf):
    p = subprocess.check_output(['iop-utils', 'stats', iopf], encoding='utf8')
    cycles = int(re.findall("Program is (\S+)", p)[0])
    compute_usec = float(cycles)*1e-3
    print(f'{iopf}:')
    print(f'  cycles={cycles}')
    print(f'  usec={compute_usec}')
    print()

print_iop_stats(iop_file)





#
#
#
def rungroq_nonblocking():
    from groq.runtime import driver as runtime
    import groq.runtime
    print('[[Groq nonblocking]]')
    shim = groq.runtime.DriverShim()
    dptr = shim.next_available_device()
    dptr.open() # open and lock the card
    prog = runtime.IOProgram(iop_file)
    print('loading')
    dptr.load(prog[0])
    print('done loading')
    # entry points: monolihic, input, compute, output
    prog[0].entry_points[0] # monolithic EP, which includes input, compute, and output

    # the dimension of inputs and outputs. IODescriptor
    inputs_iodesc = prog[0].entry_points[0].input
    outputs_iodesc = prog[0].entry_points[0].output

    inbuf  = runtime.BufferArray(inputs_iodesc, NELEMS)
    outbuf = runtime.BufferArray(outputs_iodesc, NELEMS)

    for i in range(0, NELEMS):
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
    for i in range(1, NELEMS):
        dptr.invoke_nonblocking(inbuf[i], outbuf[i])
        loopcnt = 0
        while not outbuf[i - 1].ready():
            loopcnt += 1
            if (loopcnt >= maxloopcnt):
                print('Timedout!')
                sys.exit(1)
        loopcntlog.append(loopcnt)
    loopcnt = 0
    while not outbuf[NELEMS - 1].ready():
        loopcnt += 1
        if (loopcnt >= maxloopcnt):
            print('Timedout!')
            sys.exit(1)
    ts_stop = time.time()
    loopcntlog.append(loopcnt)
    perinvocation = (ts_stop-ts_start)/float(NELEMS) * 1e6
    meanloopcnt = np.mean(loopcntlog)
    print(f'  Per invocation [usec]: {perinvocation:.3f}')
    print(f'  Average wait loopcnt: {meanloopcnt:.1f}')

    for i in range(0, NELEMS):
        res = {}
        for outt in outputs_iodesc.tensors:
            tmparr = outt.allocate_numpy_array()
            outt.to_host(outbuf[i], tmparr)
            res[outt.name] = tmparr

        mm_result=res['mm_result']
        # FLIP
        #mm_result=mm_result.transpose()
        #print(f'mm_result.dtype={mm_result.dtype}')
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

def rungroq_tsprunner():
    print('[[Groq tsp_runner]]')

    program = tsp.create_tsp_runner(iop_file)
    ets = []

    for i in range(0, NELEMS):
        st = time.time()
        result = program(matrix1=m1_data[i], matrix2=m2t_data[i])
        et = time.time() - st
        ets.append(et*1e6)
        v = np.allclose(oracle_data[i], result['mm_result'],
                        rtol=1e-1, atol=1e-1, equal_nan=True)
        if v == False:
            print('Error: Failed to verity')
            print(res['mm_result'])
            sys.exit(1)
    #print('Verification passed!')
    perinvocation = np.mean(ets)
    print(f'  Per invocation [usec]: {perinvocation:.3f}')

rungroq_tsprunner()
rungroq_nonblocking()
rungroqflowabunch()
