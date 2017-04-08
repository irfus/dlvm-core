cmake_minimum_required(VERSION 3.6)
project(DLVM)

set(CMAKE_VERBOSE_MAKEFILE TRUE)

if(CMAKE_BUILD_TYPE MATCHES RELEASE)
  set(SWIFTPM_BUILD_TYPE release)
else()
  set(SWIFTPM_BUILD_TYPE debug)
endif()

# DLVM Compute Primitives

add_subdirectory(Compute)

# DLVM Core

file(GLOB_RECURSE CORE_SOURCES Sources/*.swift)

add_custom_target(Core
  COMMAND swift build --build-path ${CMAKE_CURRENT_BINARY_DIR}/Core -c ${SWIFTPM_BUILD_TYPE}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "DLVM Core"
  SOURCES ${CORE_SOURCES})

set_directory_properties(PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/Core)

add_custom_target(dlvm-all ALL DEPENDS core compute)
