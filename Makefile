NVCC = nvcc
ARCH ?= sm_61
NVCCFLAGS += -O2 -std=c++20

CUDA_PATH ?= $(shell echo $$CUDA_HOME)
ifneq ($(CUDA_PATH),)
    CUDA_LIB_PATH = $(CUDA_PATH)/lib64
    CUDA_INCLUDE_PATH = $(CUDA_PATH)/include
else
    $(error CUDA_PATH is not set. Please set CUDA_PATH or CUDA_HOME environment variable)
endif

TARGET = a.out

CUDA_SRCS = main.cu myKernel.cu rules.cu utils.cu WFC.cu 

OBJ_DIR = build
CUDA_OBJS = $(patsubst %.cu,$(OBJ_DIR)/%.o,$(CUDA_SRCS))

all: $(OBJ_DIR) $(TARGET)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: %.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_INCLUDE_PATH) -arch $(ARCH) -c $< -o $@

$(TARGET): $(CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(CUDA_OBJS) -L$(CUDA_LIB_PATH) -lcudart -lcudadevrt

clean:
	rm -f $(TARGET)
	rm -rf $(OBJ_DIR)

.PHONY: all clean
