TARGET=main
OBJECTS=util.o prefix_sum.o

CPPFLAGS=-std=c++11 -O0 -Wall -march=native -mavx2 -mfma -mno-avx512f
LDFLAGS=-lm -pthread

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
