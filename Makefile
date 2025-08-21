CC = gcc
CFLAGS = -Wall -Wno-implicit-function-declaration -O3 -march=native -ffast-math -DNDEBUG

SRC_DIR = src
MATRIX_DIR = matrix

SRCS = $(wildcard $(SRC_DIR)/*.c) $(wildcard $(MATRIX_DIR)/*.c) main.c
OBJS = $(SRCS:.c=.o)

# Mini-batch build (uses main_mini_batch.c)
SRCS_MINI = $(wildcard $(SRC_DIR)/*.c) $(wildcard $(MATRIX_DIR)/*.c) main_mini_batch.c
OBJS_MINI = $(SRCS_MINI:.c=.o)

all: main mini

main: $(OBJS)
	@echo "Linking objects..."
	$(CC) $(CFLAGS) $(OBJS) -o main

mini: $(OBJS_MINI)
	@echo "Linking objects (mini)..."
	$(CC) $(CFLAGS) $(OBJS_MINI) -o mini

%.o: %.c
	@echo "Compiling $<"
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@echo "Cleaning up..."
	rm -f $(OBJS) $(OBJS_MINI) main mini train.o predict.o train predict
