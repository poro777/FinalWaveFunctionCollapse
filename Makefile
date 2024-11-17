# Define the compilers and options
CXX = g++
NVCC = nvcc
CXXFLAGS = -O2 -std=c++20
NVCCFLAGS = -O2 -std=c++20 -dc

# CUDA paths (adjust if necessary)
CUDA_PATH = /usr/local/cuda
CUDA_LIB_PATH = $(CUDA_PATH)/lib64
CUDA_INCLUDE_PATH = $(CUDA_PATH)/include

# Target executable
TARGET = a.out

# Source files
SRCS = main.cpp utils.cpp rules.cpp WFC.cpp
CUDA_SRCS = myKernel.cu

# Header files
HDRS = utils.h rules.h WFC.h myTimer.h
CH_HDRS = WFC.h myKernel.cuh
# Object files directory
OBJ_DIR = build
OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRCS))
CUDA_OBJS = $(patsubst %.cu,$(OBJ_DIR)/%.o,$(CUDA_SRCS))

# Default target to build and run the program
all: $(TARGET)

# Create the object files directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Rule to compile C++ source files and place .o files in the build directory
$(OBJ_DIR)/%.o: %.cpp $(HDRS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(CUDA_INCLUDE_PATH) -c $< -o $@

# Rule to compile CUDA source files and place .o files in the build directory
$(OBJ_DIR)/%.o: %.cu $(CH_HDRS) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_INCLUDE_PATH) -c $< -o $@

# Rule to link the object files and create the executable
$(TARGET): $(OBJS) $(CUDA_OBJS)
	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(CUDA_OBJS) -L$(CUDA_LIB_PATH) -lcudart -lcudadevrt

# Clean up by removing the compiled executable and object files
clean:
	rm -f $(TARGET)
	rm -rf $(OBJ_DIR)
