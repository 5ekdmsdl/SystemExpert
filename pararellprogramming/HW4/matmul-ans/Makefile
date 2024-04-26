TARGET=main
OBJECTS=util.o mat_mul.o

CPPFLAGS=-std=c++11 -O3 -Wall -mavx2 -mfma -fopenmp -mno-avx512f
LDFLAGS=-lm 

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
