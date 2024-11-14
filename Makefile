# Define the compiler and options
CXX = nvcc 


# Target executable
TARGET = a.out

# Source files
SRCS = main.cu utils.cpp rules.cpp CudaWFC.cu

# Header files
HDRS = utils.h rules.h CudaWFC.h myTimer.h

# Default target to build and run the program
all: $(TARGET)

# Rule to compile the source files
$(TARGET): $(SRCS) $(HDRS)
	$(CXX) -o $(TARGET) $(SRCS)

# Clean up by removing the compiled executable
clean:
	rm -f $(TARGET)
