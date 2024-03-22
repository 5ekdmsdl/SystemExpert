// This code is for Project #2

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define nMAX 64
#define iMAX 1
#define rMAX 512

#define read_csr(reg) ({uint64_t __tmp; asm volatile ("csrr %0, " #reg : "=r"(__tmp)); __tmp;})

void quickSort(uint32_t arr[], uint32_t* L, uint32_t* R) {
  uint32_t pivot = (*L + *(L + 1) + *(L + 2)) / 3;
  uint32_t* left = L; uint32_t* right = R;
  uint32_t temp;

  while(right >= left){
    while(left <= right && *left < pivot) left++;
    while(left <= right && *right > pivot) right--;

    if(right >= left){
      temp = *right;
      *right = *left;
      *left = temp;

      left++; right--;
    }
  }

  if(right > L)
    quickSort(arr, L, right);

  if(R > left)
    quickSort(arr, left, R);
}

void your_sort(uint32_t array[])
{
  quickSort(array, &array[0], &array[nMAX - 1]);
}

// Reference code (Bubble sort)
void bubble_sort(uint32_t array[])
{
  int i, j;
  uint32_t temp;
  for (i = 0; i < (nMAX - 1); i++)
  {
    for (j = 0; j < (nMAX - i - 1); j++)
    {
      if (array[j] > array[j + 1])
      {
        temp = array[j];
        array[j] = array[j + 1];
        array[j + 1] = temp;
      }
    }
  }
}

// Check the sorted results between 
// reference code and your code implemented.
void check(uint32_t array1[], uint32_t array2[])
{
  int i;
  uint32_t temp = 0;
  for (i = 0; i < nMAX; i++)
  {
    if (array1[i] != array2[i])
    {
      printf("Error : Wrong at array random_num[%d]\n",i);
      temp = 1;
    }
  }
  if (temp == 0)
  {
    printf("Success!\n");
  }
}


int main()
{
  int i, j;
  uint32_t random_nums[nMAX] = {0, };
  uint32_t your_nums[nMAX]   = {0, };

  uint64_t start_inst, end_inst;
  uint64_t start_cycle, end_cycle;

  srand(0);
  for (i = 0; i < iMAX; i++)
  {
    for (j = 0; j < nMAX; j++)
    {
      random_nums[j] = rand() % rMAX;
      your_nums[j] = random_nums[j];
    }
    bubble_sort(random_nums);


    
    start_inst = read_csr(instret); // read the current instruction count from instret csr register.
    start_cycle = read_csr(cycle);  // read the current cycle from cycle csr register.


    //** run your function **//
    your_sort(your_nums);
    //***********************//
    
    end_inst = read_csr(instret);  // read the current instruction count from instret csr register.
    end_cycle = read_csr(cycle);   // read the current cycle from cycle csr register.

    printf("Instruction counts = %ld\n", end_inst - start_inst);
    printf("Execution cycles = %ld\n", end_cycle - start_cycle);
    printf("IPC = %f\n", ((float)((double)(end_inst - start_inst))/((double)(end_cycle - start_cycle))));

    check(random_nums, your_nums);
  }

  return 0;
}
