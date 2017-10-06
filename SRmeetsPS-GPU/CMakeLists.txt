cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

set(PROJECT_NAME SRmeetsPS_GPU)
# Output Variables
# Folders files
include_directories(./../include)
#include_directories(./../matio/include)
#link_directories(./../matio/lib)

add_library(matio SHARED IMPORTED GLOBAL)
set_target_properties(matio PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../matio/lib/libmatio.so"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/../matio/include"
)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x -D_FORCE_INLINES")
endif()

list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -D_FORCE_INLINES")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Definition of Macros
add_definitions(
   -DTHRUST_DEBUG 
   -DWIN64 
   -D_DEBUG 
   -D_CONSOLE 
)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

#./../matio/include/matio.h ./../matio/include/matio_pubconf.h 
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} 
   ${SRC_FILES} devicecalls.cu Main.cpp SRPS.cu Utilities.cpp ./../include/Exceptions.h 
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS} matio ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})

