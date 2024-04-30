#!/bin/bash

srun --nodes=1 --exclusive --partition=samsung ./main $@
