execute_process(COMMAND llvm-config --cflags
  OUTPUT_VARIABLE LLVM_C_FLAGS)
string(STRIP "${LLVM_C_FLAGS}" LLVM_C_FLAGS)
execute_process(COMMAND llvm-config --cxxflags
  OUTPUT_VARIABLE LLVM_CXX_FLAGS)
string(STRIP "${LLVM_CXX_FLAGS}" LLVM_CXX_FLAGS)

set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/include)
set(DLRT_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB DLRT_INC ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB DLRT_CXX_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB DLRT_C_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

set(CC_BRIDGING_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/dlrt.h)
set(CC_FLAGS -O3 -fmodule-name="dlrt" -I${DLRT_INC_DIR} ${LLVM_C_FLAGS} -c -emit-llvm)
set(CXX_FLAGS -O3 -std=c++17 -stdlib=libc++ -fmodule-name="dlrt" -I${DLRT_INC_DIR} ${LLVM_CXX_FLAGS} -c -emit-llvm)

set_directory_properties(PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(Runtime
  # Compile to bitcode
  COMMAND clang++ ${CXX_FLAGS} ${DLRT_CXX_SRC}
  COMMAND clang ${CC_FLAGS} ${DLRT_C_SRC}
  # Merge multiple bitcode libs to one
  COMMAND rm -f dlrt.bc && llvm-link *.bc -o dlrt.bc
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "DLVM Runtime Routines"
  SOURCES ${DLRT_INC} ${DLRT_SRC}
  )
