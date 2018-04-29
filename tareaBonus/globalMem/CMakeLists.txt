cmake_minimum_required(VERSION 2.8)
project( sobel_Gmem )
find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=sm_52)
include_directories( ${OpenCV_INCLUDE_DIRS} )
cuda_add_executable( sobel_Gmem sobel_Gmem.cu )
target_link_libraries( sobel_Gmem ${OpenCV_LIBS} )


