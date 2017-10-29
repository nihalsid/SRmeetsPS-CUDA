PROJECT_ROOT_DIR = ../..
OUTPUT_DIR = bin
OUTPUT_EXE = SRmeetsPS-CUDA
SOURCE_DIR = $(PROJECT_ROOT_DIR)/SRmeetsPS-GPU
SOURCE_FILES = $(SOURCE_DIR)/devicecalls.cu $(SOURCE_DIR)/Main.cpp $(SOURCE_DIR)/SRPS.cu $(SOURCE_DIR)/Utilities.cpp
INCLUDE_DIRS = "$(PROJECT_ROOT_DIR)/matio/include,$(PROJECT_ROOT_DIR)/opencv/include"
LIBRARIES = "-lopencv_world -lmatio -lcublas -lcusparse"
LIBRARY_DIRS = "$(PROJECT_ROOT_DIR)/matio/lib,$(PROJECT_ROOT_DIR)/opencv/lib"
NVCC_FLAGS = "-std=c++11"
OUTPUT_PATH = $(OUTPUT_DIR)/$(OUTPUT_EXE)

all:
	mkdir $(OUTPUT_DIR)
	nvcc $(NVCC_FLAGS) -o $(OUTPUT_PATH) $(SOURCE_FILES) -I $(INCLUDE_DIRS) -L $(LIBRARY_DIRS) $(LIBRARIES) 
	
clean:
	rm -rf $(OUTPUT_DIR)
