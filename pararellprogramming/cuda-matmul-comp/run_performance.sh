#!/bin/bash

srun --nodes=1 --partition samsung --exclusive numactl --physcpubind 0-31 ./main -n 10 8192 8192 8192
