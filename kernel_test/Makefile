CC := g++
CFLAGS := -std=c++11 -Wall -Wextra -I/usr/include
LDFLAGS := -L/usr/lib -lpthread -lOpenCL

SRCS := kernel.cpp main.cpp
OBJS := $(SRCS:.cpp=.o)
EXEC := kernel

.PHONY: all clean

kernel_test: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(EXEC)
