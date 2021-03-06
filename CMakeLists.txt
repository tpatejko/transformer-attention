cmake_minimum_required(VERSION 2.8)
project(transformer-attention)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CheckCXXCompilerFlag)

if(UNIX)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX17)
add_definitions(-std=c++17)
#elseif(COMPILER_SUPPORTS_CXX0X)
#add_definitions(-std=c++0x)
endif()
endif()

find_library(GFLAGS_LIB NAMES gflags)
find_path(GFLAGS_INCLUDES NAMES gflags.h PATHS /usr/include/gflags/)
message("GFLAGS INCLUDES: ${GFLAGS_INCLUDES}")
include_directories(SYSTEM ${GFLAGS_INCLUDES})

set(EIGENROOT "$ENV{EIGENROOT}")
set(MKLROOT "$ENV{MKLROOT}")
set(MKLDNNROOT "$ENV{MKLDNNROOT}")

if(EIGENROOT)
  find_path(EIGEN_INCLUDE_DIR signature_of_eigen3_matrix_library PATHS ${EIGENROOT}/include PATH_SUFFIXES eigen3)
  message(STATUS "Found Eigen (include: ${EIGEN_INCLUDE_DIR}")
  include_directories(${EIGEN_INCLUDE_DIR})
else()
  message(FATAL "Eigen not found")
endif()

if(MKLROOT)
  find_path(MKL_INCLUDE_DIR mkl_cblas.h PATHS ${MKLROOT} PATH_SUFFIXES include)
  find_library(MKL_LIBRARY
                 NAMES mklml_intel
                 PATHS ${MKLROOT}/lib)
  message(STATUS "Found MKL (include: ${MKL_INCLUDE_DIR}")
  include_directories(${MKL_INCLUDE_DIR})
else()
  message(FATAL "MKL not found")
endif()

if(MKLDNNROOT)
  find_path(MKLDNN_INCLUDE_DIR mkldnn.hpp PATHS ${MKLDNNROOT} PATH_SUFFIXES include)
  find_library(MKLDNN_LIBRARY
               NAMES mkldnn
               PATHS ${MKLDNNROOT}/lib)
  message(STATUS "Found MKLDNN (include: ${MKLDNN_INCLUDE_DIR})")
  include_directories(${MKLDNN_INCLUDE_DIR})
else()
  message(FATAL "MKLDNN not found")
endif()

add_executable(attention attention.cpp)

add_definitions(-Wall)

#target_link_libraries(attention ${MKL_LIBRARY} )
target_link_libraries(attention ${MKL_LIBRARY} ${MKLDNN_LIBRARY} ${GFLAGS_LIB})
