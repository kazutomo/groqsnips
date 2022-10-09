# groqsnips

This repository includes my groq snippets for learning, testing and benchmarking purposes.

To enable the Groq environment, check
https://github.com/groq/groqflow/blob/main/docs/install.md

Tested with GroqWare Suite version 0.9.0

To run the benchmark, simply type:

     $ ./bench-hostapis-matmal.py N=1 M=320 L=300
     [Params]
     N=1
     M=320
     L=300
     T=16
     prep data: ................
     M1shape: (1, 320)
     M2shape: (320, 300)
     [[CPU]]
       Per invocation [usec]: xxxxxx.xxx
     [[Cycles reported by iop-utils]]
       cycles=x
       usec=x.xxx
     [[Groq tsp_runner]]
       Per invocation [usec]: xxx.xxx
     [[Groq nonblocking]]
       Per invocation [usec]: xx.xxx
       Average wait loopcnt: xxx.x
     [[Groqflow]]
       Per invocation [usec]: xxxxx.xx
     Done
