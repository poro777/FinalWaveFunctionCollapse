# Define the compiler and options
CXX = g++
CXXFLAGS = -O3 

# Target executable
TARGET = a.out

# Source files
SRCS = main.cpp utils.cpp

# Header files
HDRS = utils.h rules.h WFC.h

# Default target to build and run the program
all: $(TARGET)

# Rule to compile the source files
$(TARGET): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Clean up by removing the compiled executable
clean:
	rm -f $(TARGET)
