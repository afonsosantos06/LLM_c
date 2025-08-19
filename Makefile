CC = gcc
CFLAGS = -Wall -Wno-implicit-function-declaration -g

# Source olders
SRC_DIR = src
MATRIX_DIR = matrix

# List of source files
SRCS = main.c $(wildcard $(SRC_DIR)/*.c) $(wildcard $(MATRIX_DIR)/*.c)

# Object files (replace .c with .o)
OBJS = $(SRCS:.c=.o)

all: main

# Link step
main: $(OBJS)
	@echo "Linking $(TARGET)"
	$(CC) $(CFLAGS) -o $@ $(OBJS)

# Generic rule to compile .c -> .o
%.o: %.c
	@echo "Compiling $<"
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@echo "Cleaning up..."
	rm -f $(OBJS) $(TARGET)
