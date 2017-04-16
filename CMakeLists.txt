cmake_minimum_required(VERSION 3.6)
project(DLVM)

# Get LLVM paths
execute_process(COMMAND llvm-config --includedir
  OUTPUT_VARIABLE LLVM_INC_DIR)
string(STRIP ${LLVM_INC_DIR} LLVM_INC_DIR)
execute_process(COMMAND llvm-config --libdir
  OUTPUT_VARIABLE LLVM_LIB_DIR)
string(STRIP ${LLVM_LIB_DIR} LLVM_LIB_DIR)

set(CMAKE_VERBOSE_MAKEFILE TRUE)

if(CMAKE_BUILD_TYPE MATCHES RELEASE)
  set(SWIFTPM_BUILD_TYPE release)
else()
  set(SWIFTPM_BUILD_TYPE debug)
endif()

# DLVM Compute Primitives

add_subdirectory(Compute)
add_subdirectory(Runtime)

# DLVM Core

file(GLOB_RECURSE CORE_SOURCES Sources/*.swift)

add_custom_target(Core
  COMMAND swift build --build-path ${CMAKE_CURRENT_BINARY_DIR}/Core -c ${SWIFTPM_BUILD_TYPE} -Xcc -I${LLVM_INC_DIR} -Xlinker -lLLVM -Xlinker -L${LLVM_LIB_DIR}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "DLVM Core"
  SOURCES ${CORE_SOURCES})

set_directory_properties(PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/Core)

add_custom_target(dlvm-all ALL DEPENDS Core Compute Runtime)
