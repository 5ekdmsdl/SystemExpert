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

// [8x8] 3,000 cycle 이하
// [16x16] 17,000 cycle 이하
// [64x64] 1,400,000 cycle 이하

void MatrixMultiplicationAll(int** A, 
                           int** B,
                           int** C,
                           int size) {
  uint32_t i = 0;
  uint32_t j = 0;
  uint32_t k = 0;
  
  int j_stepSize = 4; // size / 2;
  // int k_stepSize = 8; // (16 < (size / 2))? 16: size / 2;
  
  for(int _j = 0; _j < size; _j += j_stepSize){
    for(i = 0; i < size; i++)   {
      for (j = _j; j < _j + j_stepSize; j += 2) {
          for(k = 0; k < size; k += 4){
            C[i][k + 0] += A[i][j + 1] * B[j + 1][k + 0] + A[i][j] * B[j][k + 0];
            C[i][k + 1] += A[i][j + 1] * B[j + 1][k + 1] + A[i][j] * B[j][k + 1];
            C[i][k + 2] += A[i][j + 1] * B[j + 1][k + 2] + A[i][j] * B[j][k + 2];
            C[i][k + 3] += A[i][j + 1] * B[j + 1][k + 3] + A[i][j] * B[j][k + 3];
          }
        }
    }
  }
  

}

#ifdef Matrix8
void MatrixMultiplication8(int** A, 
                           int** B,
                           int** C,
                           int size) 
{
  for(uint32_t i = 0; i < 8; i += 1)
  {
    for(uint32_t j = 0; j < 8; j += 1)
    {
        C[i][0] += A[i][j] * B[j][0];
        C[i][1] += A[i][j] * B[j][1];
        C[i][2] += A[i][j] * B[j][2];
        C[i][3] += A[i][j] * B[j][3];
        C[i][4] += A[i][j] * B[j][4];
        C[i][5] += A[i][j] * B[j][5];
        C[i][6] += A[i][j] * B[j][6];
        C[i][7] += A[i][j] * B[j][7];
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
  uint32_t i = 0;
  uint32_t j = 0;
  uint32_t k = 0;
  
  int j_stepSize = 4; // size / 2;
  // int k_stepSize = 8; // (16 < (size / 2))? 16: size / 2;
  
  for(int _j = 0; _j < size; _j += j_stepSize){
    for(i = 0; i < size; i++)   {
      for (j = _j; j < _j + j_stepSize; j += 2) {
          for(k = 0; k < size; k += 8){
            C[i][k + 0] += A[i][j + 1] * B[j + 1][k + 0] + A[i][j] * B[j][k + 0];
            C[i][k + 1] += A[i][j + 1] * B[j + 1][k + 1] + A[i][j] * B[j][k + 1];
            C[i][k + 2] += A[i][j + 1] * B[j + 1][k + 2] + A[i][j] * B[j][k + 2];
            C[i][k + 3] += A[i][j + 1] * B[j + 1][k + 3] + A[i][j] * B[j][k + 3];
            // k += 4;
            C[i][k + 4] += A[i][j + 1] * B[j + 1][k + 4] + A[i][j] * B[j][k + 4];
            C[i][k + 5] += A[i][j + 1] * B[j + 1][k + 5] + A[i][j] * B[j][k + 5];
            C[i][k + 6] += A[i][j + 1] * B[j + 1][k + 6] + A[i][j] * B[j][k + 6];
            C[i][k + 7] += A[i][j + 1] * B[j + 1][k + 7] + A[i][j] * B[j][k + 7];
          }
        }
    }
  }
  return;

  // int i_stepSize = 8;
  // // int j_stepSize = 4;
  // for(uint32_t _i = 0; _i < size; _i += i_stepSize)
  // {
  //   for (uint32_t i = _i; i < _i + i_stepSize; i++) 
  //   {
  //     for(uint32_t _j = 0; _j < size; _j += j_stepSize)
  //     {
  //       for (uint32_t j = _j; j < _j + j_stepSize; j++) 
  //       {
  //         C[i][0] += A[i][j] * B[j][0];
  //         C[i][1] += A[i][j] * B[j][1];
  //         C[i][2] += A[i][j] * B[j][2];
  //         C[i][3] += A[i][j] * B[j][3];
  //         C[i][4] += A[i][j] * B[j][4];
  //         C[i][5] += A[i][j] * B[j][5];
  //         C[i][6] += A[i][j] * B[j][6];
  //         C[i][7] += A[i][j] * B[j][7];
  //         C[i][8] += A[i][j] * B[j][8];
  //         C[i][9] += A[i][j] * B[j][9];
  //         C[i][10] += A[i][j] * B[j][10];
  //         C[i][11] += A[i][j] * B[j][11];
  //         C[i][12] += A[i][j] * B[j][12];
  //         C[i][13] += A[i][j] * B[j][13];
  //         C[i][14] += A[i][j] * B[j][14];
  //         C[i][15] += A[i][j] * B[j][15];
  //       }
  //     }
  //   }
  // }
  
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
  uint32_t i = 0;
  uint32_t j = 0;
  uint32_t k = 0;
  
  int j_stepSize = 4; // size / 2;
  // int k_stepSize = 8; // (16 < (size / 2))? 16: size / 2;
  
  for(int _j = 0; _j < size; _j += j_stepSize){
    for(i = 0; i < size; i++)   {
      for (j = _j; j < _j + j_stepSize; j += 2) {
          for(k = 0; k < size; k += 8){
            C[i][k + 0] += A[i][j + 1] * B[j + 1][k + 0] + A[i][j] * B[j][k + 0];
            C[i][k + 1] += A[i][j + 1] * B[j + 1][k + 1] + A[i][j] * B[j][k + 1];
            C[i][k + 2] += A[i][j + 1] * B[j + 1][k + 2] + A[i][j] * B[j][k + 2];
            C[i][k + 3] += A[i][j + 1] * B[j + 1][k + 3] + A[i][j] * B[j][k + 3];
            // k += 4;
            C[i][k + 4] += A[i][j + 1] * B[j + 1][k + 4] + A[i][j] * B[j][k + 4];
            C[i][k + 5] += A[i][j + 1] * B[j + 1][k + 5] + A[i][j] * B[j][k + 5];
            C[i][k + 6] += A[i][j + 1] * B[j + 1][k + 6] + A[i][j] * B[j][k + 6];
            C[i][k + 7] += A[i][j + 1] * B[j + 1][k + 7] + A[i][j] * B[j][k + 7];
          }
        }
    }
  }
  return;

  // int i_stepSize = 16;
  // int j_stepSize = 64;
  // int k_stepSize = 8;  

  // for(uint32_t _i = 0; _i < size; _i += i_stepSize)
  // {
  //   for (uint32_t i = _i; i < _i + i_stepSize; i++) 
  //   {

  //     for(uint32_t _j = 0; _j < size; _j += j_stepSize)
  //     {
  //       for (uint32_t j = _j; j < _j + j_stepSize; j++) 
  //       {

  //         for(uint32_t k = 0; k < size; k += k_stepSize){
  //           C[i][k + 0] += A[i][j] * B[j][k + 0];
  //           C[i][k + 1] += A[i][j] * B[j][k + 1];
  //           C[i][k + 2] += A[i][j] * B[j][k + 2];
  //           C[i][k + 3] += A[i][j] * B[j][k + 3];
  //           C[i][k + 4] += A[i][j] * B[j][k + 4];
  //           C[i][k + 5] += A[i][j] * B[j][k + 5];
  //           C[i][k + 6] += A[i][j] * B[j][k + 6];
  //           C[i][k + 7] += A[i][j] * B[j][k + 7];
  //         }
  //       }
  //     }
  //   }
  // }
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
