CC := g++
CFLAGS := -std=c++11 -Wall -Wextra -I/usr/include
LDFLAGS := -L/usr/lib -lpthread -lOpenCL

SRCS := matrix_multiply.cpp main.cpp
OBJS := $(SRCS:.cpp=.o)
EXEC := matrix_multiply

.PHONY: all clean

matrix: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(EXEC)
