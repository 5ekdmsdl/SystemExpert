#include <cstdio>
#include <mpi.h>
int main() {
  MPI_Init(NULL, NULL);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
 
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int hostnamelen;
  MPI_Get_processor_name(hostname, &hostnamelen);
  printf("hostname : %s, rank : %d, size : %d \n", hostname, rank, size);
 
  printf("Hello, I am rank %d of size %d world!\n", rank, size);
  MPI_Finalize();
  return 0;
}
