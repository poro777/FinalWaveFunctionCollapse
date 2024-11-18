# Define the compiler and options
CXX = nvcc 


# Target executable
TARGET = a.out

# Source files
SRCS = main.cpp utils.cpp rules.cpp WFC.cpp myKernel.cu

# Header files
HDRS = utils.h rules.h myKernel.cuh myTimer.h setOperator.h WFC.h

# Default target to build and run the program
all: $(TARGET)

# Rule to compile the source files
$(TARGET): $(SRCS) $(HDRS)
	$(CXX) -std=c++20 -o $(TARGET) $(SRCS)

# Clean up by removing the compiled executable
clean:
	rm -f $(TARGET)