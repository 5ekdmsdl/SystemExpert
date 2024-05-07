make clean
make

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31                         \
  ./main -t 32 -n 5 16384 8192 8192
