mpic++ -o ./hello_world ./hello_world.cpp
salloc -N 2 mpirun -mca btl ^openib -npernode 2 ./hello_world
