# Define the compiler and options
CXX = g++
CXXFLAGS = -O2 -std=c++20

# Target executable
TARGET = a.out

# Source files
SRCS = main.cpp utils.cpp rules.cpp WFC.cpp

# Header files
HDRS = utils.h rules.h WFC.h myTimer.h

# Default target to build and run the program
all: $(TARGET)

# Rule to compile the source files
$(TARGET): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) -fopenmp -Wall

# Clean up by removing the compiled executable
clean:
	rm -f $(TARGET)
