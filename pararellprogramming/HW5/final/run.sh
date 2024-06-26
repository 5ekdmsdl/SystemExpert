#!/bin/bash

: ${NODES:=2}

salloc -N $NODES --exclusive                         \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main $@
