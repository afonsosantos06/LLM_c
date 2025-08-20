CC = gcc
CFLAGS = -Wall -Wno-implicit-function-declaration -O3 -march=native -ffast-math -DNDEBUG

SRC_DIR = src
MATRIX_DIR = matrix

SRCS = $(wildcard $(SRC_DIR)/*.c) $(wildcard $(MATRIX_DIR)/*.c)
OBJS = $(SRCS:.c=.o)

all: train predict

train: train.o $(OBJS)
	@echo "Linking train"
	$(CC) $(CFLAGS) -o $@ $^

predict: predict.o $(OBJS)
	@echo "Linking predict"
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	@echo "Compiling $<"
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@echo "Cleaning up..."
	rm -f $(OBJS) train.o predict.o train predict
