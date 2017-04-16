set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/include)

set(DLRT_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB DLRT_INC ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB DLRT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

# Set libc path if not generating Xcode (won't build if we specify
# libc path with Xcode)
if(NOT CMAKE_GENERATOR STREQUAL Xcode)
set(CC_LIBC_INC_FLAG -I${CMAKE_LIBRARY_PATH})
endif()

set(CC_BRIDGING_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/dlrt.h)
set(CC_FLAGS -O2 -fmodule-name="dlrt" -I${DLRT_INC_DIR} ${CC_LIBC_INC_FLAG} -ObjC -c -emit-llvm)

set_directory_properties(PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(Runtime
  COMMAND clang ${CC_FLAGS} ${DLRT_SRC}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "DLVM Runtime Routines"
  SOURCES ${DLRT_INC} ${DLRT_SRC}
  )
