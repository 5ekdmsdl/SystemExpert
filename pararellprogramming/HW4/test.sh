make clean
make
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 26 831 538 2304
# ./run_performance.sh