#!/bin/bash

srun -N 1 --partition samsung --exclusive ./vec_add
