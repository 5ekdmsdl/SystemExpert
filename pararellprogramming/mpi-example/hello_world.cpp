#include <cstdio>
#include <mpi.h>

int main() {
  MPI_Init(NULL, NULL);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
 
  int hostnamelen; char hostname[MPI_MAX_PROCESSOR_NAME];

  if(rank == 0){
    MPI_Get_processor_name(hostname, &hostnamelen);
    printf("hostname : %s, rank : %d, size : %d \n", hostname, rank, size);
  }

  MPI_Bcast(&hostnamelen, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  
  if(rank == 0){
    MPI_Send(hostname, hostnamelen + 1, MPI_CHAR, 1, 1234, MPI_COMM_WORLD);
  }
  else {
    char recvBuf[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;
    MPI_Recv(recvBuf, hostnamelen + 1, MPI_CHAR, 0, 1234, MPI_COMM_WORLD, &status);
    printf("Received hostname at rank 1: %s\n", recvBuf);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  printf("Hello, I am rank %d of size %d world!\n", rank, size);
  MPI_Finalize();
  return 0;
}
