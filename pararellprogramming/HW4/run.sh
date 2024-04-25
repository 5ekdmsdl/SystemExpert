#!/bin/bash

srun --nodes=1 --exclusive  numactl --physcpubind 0-31 ./main $@
