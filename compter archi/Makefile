riscvcc=/opt/riscv/bin/riscv64-unknown-elf-gcc
riscvcpp=/opt/riscv/bin/riscv64-unknown-elf-g++
CCFLAG= -O2 -static -fno-reorder-blocks-and-partition -fno-reorder-functions -fno-reorder-blocks -fno-inline-small-functions -fno-shrink-wrap 
CCFLAG_DEBUG=-g -O0 -static
gem5dir=/home/scale/gem5
SIZE ?=8

SRC = matrix_multiplication.c
#SRC = $(wildcard *.c)
INC = ./include

run: test
step1: cmp

step2: gem5

step3: cmp gem5

step4: cmp gem5_check

step5: gem5_debug

debug: cmp_debug

all: gem5_all

cmp: 
	$(riscvcc) $(CCFLAG) $(SRC) -DMatrix$(SIZE) -I$(INC) -o matrix_multiplication 

clean:
	@rm -rf matrix_multiplication matrix_multiplication_debug
	@rm -rf m5out
	@rm -rf stats

gem5:
	rm -rf m5out
	$(gem5dir)/build/RISCV/gem5.opt $(gem5dir)/configs/example/se.py --cpu-type=DerivO3CPU	\
	--caches --l1d_size=512B --l1i_size=512B --l2cache --l2_size=8kB --cacheline_size=64	\
	--sys-clock=1GHz --cpu-clock=1GHz --cmd=./matrix_multiplication -o '$(SIZE) opt'
	@mkdir -p stats
	@mv m5out/stats.txt stats/stats_$(SIZE)x$(SIZE).txt

gem5_pipeview:
	rm -rf m5out
	$(riscvcc) $(CCFLAG) $(SRC) -DMatrix$(SIZE) -I$(INC) -o matrix_multiplication 
	$(gem5dir)/build/RISCV/gem5.opt --debug-flags=O3PipeView --debug-start=1000 \
	--debug-file=trace.out $(gem5dir)/configs/example/se.py --cpu-type=DerivO3CPU	\
	--caches --l1d_size=512B --l1i_size=512B --l2cache --l2_size=8kB --cacheline_size=64	\
	--sys-clock=1GHz --cpu-clock=1GHz --cmd=./matrix_multiplication -o '$(SIZE) opt'
	$(gem5dir)/util/o3-pipeview.py -o pipeview.out --color m5out/trace.out

run_pipeview:
	less -r pipeview.out

gem5_check:
	rm -rf m5out
	@mkdir -p stats
	$(gem5dir)/build/RISCV/gem5.opt $(gem5dir)/configs/example/se.py --cpu-type=DerivO3CPU	\
	--caches --l1d_size=512B --l1i_size=512B --l2cache --l2_size=8kB --cacheline_size=64	\
  --sys-clock=1GHz --cpu-clock=1GHz --cmd=./matrix_multiplication -o '$(SIZE) all'

gem5_all:
	rm -rf m5out
	@mkdir -p stats
	$(riscvcc) $(CCFLAG) $(SRC) -DMatrix$(SIZE) -I$(INC) -o matrix_multiplication 
	$(gem5dir)/build/RISCV/gem5.opt $(gem5dir)/configs/example/se.py --cpu-type=DerivO3CPU	\
	--caches --l1d_size=512B --l1i_size=512B --l2cache --l2_size=8kB --cacheline_size=64	\
       	--sys-clock=1GHz --cpu-clock=1GHz --cmd=./matrix_multiplication -o '$(SIZE) all'
	$(riscvcc) $(CCFLAG) $(SRC) -DMatrix$(shell echo '$(SIZE)*2'|bc) -I$(INC) -o matrix_multiplication 
	$(gem5dir)/build/RISCV/gem5.opt $(gem5dir)/configs/example/se.py --cpu-type=DerivO3CPU	\
	--caches --l1d_size=512B --l1i_size=512B --l2cache --l2_size=8kB --cacheline_size=64	\
       	--sys-clock=1GHz --cpu-clock=1GHz --cmd=./matrix_multiplication -o '$(shell echo '$(SIZE)*2'|bc) all'
	$(riscvcc) $(CCFLAG) $(SRC) -DMatrix$(shell echo '$(SIZE)*8'|bc) -I$(INC) -o matrix_multiplication 
	$(gem5dir)/build/RISCV/gem5.opt $(gem5dir)/configs/example/se.py --cpu-type=DerivO3CPU	\
	--caches --l1d_size=512B --l1i_size=512B --l2cache --l2_size=8kB --cacheline_size=64 	\
	--sys-clock=1GHz --cpu-clock=1GHz --cmd=./matrix_multiplication -o '$(shell echo '$(SIZE)*8'|bc) all'

cmp_debug: 
	$(riscvcc) $(CCFLAG_DEBUG) $(SRC) -DMatrix$(SIZE) -I$(INC) -o matrix_multiplication_debug 

gem5_debug:
	rm -rf m5out
	$(gem5dir)/build/RISCV/gem5.opt $(gem5dir)/configs/example/se.py --cpu-type=DerivO3CPU	\
	--caches --l1d_size=512B --l1i_size=512B --l2cache --l2_size=8kB --cacheline_size=64 	\
	--sys-clock=1GHz --cpu-clock=1GHz --cmd=./matrix_multiplication_debug -o '$(SIZE) all' --wait-gdb=8888
	@mv m5out/stats.txt stats/stats_$(SIZE)_debug.txt
