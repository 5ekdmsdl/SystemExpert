#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
//#include <iostream>
//#include <vector>

#define read_csr(reg) ({unsigned long __tmp; asm volatile ("csrr %0, " #reg : "=r"(__tmp)); __tmp;})

#define rMAX 16 // range of each element in the matrix


/*  Function 1: MatrixMultiplication8()
    TODO: Optimize matrix-matrix multiplication,
          when the matrix size is [8 x 8].
*/

#ifdef Matrix8
void MatrixMultiplication8(int** A, 
                           int** B,
                           int** C,
                           int size) 
{
  for (uint32_t i = 0; i < size; i++) 
  {
    for (uint32_t j = 0; j < size; j++) 
    {
      for (uint32_t k = 0; k < size; k++) 
      {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
#endif

/*  Function 2: MatrixMultiplication16
    TODO: Optimize matrix-matrix multiplication,
          when the matrix size is [16 x 16].
*/

#ifdef Matrix16
void MatrixMultiplication16(int** A, 
                            int** B,
                            int** C,
                            int size) 
{
  for (uint32_t i = 0; i < size; i++) 
  {
    for (uint32_t j = 0; j < size; j++) 
    {
      for (uint32_t k = 0; k < size; k++) 
      {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
#endif

/*  Function 3: MatrixMultiplication64
    TODO: Optimize matrix-matrix multiplication,
          when the matrix size is [64 x 64].
*/

#ifdef Matrix64
void MatrixMultiplication64(int** A, 
                            int** B,
                            int** C,
                            int size) 
{
  for (uint32_t i = 0; i < size; i++) 
  {
    for (uint32_t j = 0; j < size; j++) 
    {
      for (uint32_t k = 0; k < size; k++) 
      {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
#endif

/* Naive matrix multiplication function as a baseline */
void NaiveMatrixMultiplication(int** A, 
                               int** B,
                               int** C,
                               int size) 
{
  for (uint32_t i = 0; i < size; i++) 
  {
    for (uint32_t j = 0; j < size; j++) 
    {
      for (uint32_t k = 0; k < size; k++) 
      {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

// Used for checking results between your matrix multiplication and naive matrix multiplication.
void CorrectionCheck(int** C_ref, int** C, int size)
{
  int temp = 0;
  for (uint32_t i = 0; i < size; i++) {
    for (uint32_t j = 0; j < size; j++) {
      if (C_ref[i][j] == C[i][j])
        ;
      else {
        printf("wrong %d multiplication terminate program \n", size);
        printf( "C : %d , C_your : %d\n", C_ref[i][j], C[i][j]);
        printf( " on (i,j) = (%d, %d)\n",i ,j);
        temp += 1;
        continue;
      }
    }
  }
  if (temp == 0)
  {
    printf("Success!! \n");
  }
  else
  {
    printf("Total Error: %d\n", temp);
  }
}


void MatrixMultiplication(int** A, 
                          int** B,
                          int** C,
                          int size) 
{
  // Print matrix shape
  printf("Start Matrix Multiplication : Optimized\n");
  printf("[%d x %d] x [%d x %d] = ", size, size, size, size); 
  printf("[%d x %d]\n", size,size);

  uint32_t start_inst, end_inst;
  uint32_t start_cycle, end_cycle;

  switch (size)
  {
    #ifdef Matrix8
    case 8:
      start_inst = read_csr(instret); // read the current instruction count from instret csr register.
      start_cycle = read_csr(cycle);  // read the current cycle from cycle csr register.
      MatrixMultiplication8(A, B, C, size);
      end_cycle = read_csr(cycle);   // read the current cycle from cycle csr register.
      end_inst = read_csr(instret);  // read the current instruction count from instret csr register.
      break;
    #endif
    #ifdef Matrix16
    case 16:
      start_inst = read_csr(instret); // read the current instruction count from instret csr register.
      start_cycle = read_csr(cycle);  // read the current cycle from cycle csr register.
      MatrixMultiplication16(A, B, C, size);
      end_cycle = read_csr(cycle);   // read the current cycle from cycle csr register.
      end_inst = read_csr(instret);  // read the current instruction count from instret csr register.
      break;
    #endif
    #ifdef Matrix64
    case 64:
      start_inst = read_csr(instret); // read the current instruction count from instret csr register.
      start_cycle = read_csr(cycle);  // read the current cycle from cycle csr register.
      MatrixMultiplication64(A, B, C, size);
      end_cycle = read_csr(cycle);   // read the current cycle from cycle csr register.
      end_inst = read_csr(instret);  // read the current instruction count from instret csr register.
      break;
    #endif
    default:
      printf("There is a problem with the matrix size. Please enter the correct size. Ex: 8, 16, 64.\n");
      break;
  }

  printf("Matrix Multiplication done\n");
  printf("Instruction counts = %d \n", (end_inst - start_inst));
  printf("Execution cycles   = %d \n", (end_cycle - start_cycle));
  printf("IPC = %f \n", ((float)((double)(end_inst - start_inst))/((double)(end_cycle - start_cycle))));
}



/* Insert random values into the matrix  */ 
void randomInit(int** data, int size)
{
  srand(0);

  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; j++)
    {
      data[i][j] = rand() % rMAX; 
    }
  }
}

void zeroInit(int** data, int size)
{
    srand(0);

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; j++)
        {
            data[i][j] = 0;
        }
    }
}

int main(int argc, char** argv) 
{
  int matrix_size = 0;
  char *option;
  if(argc < 2) 
  {
    printf("Please insert arguments\n");
  }
  else 
  {
    matrix_size = atoi(argv[1]);
    option = argv[2];
  }

  //-------------------Matrix Declarations--------------------//
  int** A     = (int**)malloc(matrix_size * sizeof(int*));
  int** B     = (int**)malloc(matrix_size * sizeof(int*));
  int** C     = (int**)malloc(matrix_size * sizeof(int*));
  int** C_ref = (int**)malloc(matrix_size * sizeof(int*));

  for (int i = 0; i < matrix_size; i++){
    A[i]     = (int*)malloc(matrix_size * sizeof(int));
    B[i]     = (int*)malloc(matrix_size * sizeof(int));
    C[i]     = (int*)malloc(matrix_size * sizeof(int));
    C_ref[i] = (int*)malloc(matrix_size * sizeof(int));
  }

  //---Init---//
  randomInit(A, matrix_size);
  randomInit(B, matrix_size);
  zeroInit(C, matrix_size);
  zeroInit(C_ref, matrix_size);

  //-----Run Baseline------//
  if (strcmp(option,"all") == 0)
  {
    NaiveMatrixMultiplication(A, B, C_ref, matrix_size);
  }


  //-----Run Optimized------//
  if (strcmp(option,"opt") ==0 || strcmp(option,"all") == 0)
  {
    MatrixMultiplication(A, B, C, matrix_size);
  }

  //-----Functionaliry_Check-----//
  if (strcmp(option,"all") == 0)
  {
    CorrectionCheck(C_ref, C, matrix_size);
  }


  //-------------------Free--------------------//
  for (int i = 0; i < matrix_size; i++){
    free(A[i]);
    free(B[i]);
    free(C[i]);
    free(C_ref[i]);
  }

  free(A);
  free(B);
  free(C);
  free(C_ref);

  return 0;


}
