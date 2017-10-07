SOURCE_DIR = "SRmeetsPS-GPU"
SOURCE_FILES = $(SOURCE_DIR)/devicecalls.cu $(SOURCE_DIR)/Main.cpp $(SOURCE_DIR)/SRPS.cu $(SOURCE_DIR)/Utilities.cpp
INCLUDE_DIRS = "matio/include,opencv/include"
LIBRARIES = "-lopencv_world -lmatio -lcublas -lcusparse"
LIBRARY_DIRS = "matio/lib,opencv/lib"
NVCC_FLAGS = "-std=c++11"
OUTPUT_PATH = "build/SRmeetsPS-CUDA"

SRmeetsPS-GPU:
	nvcc $(NVCC_FLAGS) -o $(OUTPUT_PATH) $(SOURCE_FILES) -I $(INCLUDE_DIRS) -L $(LIBRARY_DIRS) $(LIBRARIES) 
