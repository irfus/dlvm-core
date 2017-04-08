set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/include)

file(GLOB DLCOMPUTE_INC include/*.h)
file(GLOB DLCOMPUTE_SRC src/*.swift)

# Set libc path if not generating Xcode (won't build if we specify
# libc path with Xcode)
if(NOT CMAKE_GENERATOR STREQUAL Xcode)
set(SWIFTC_LIBC_INC_FLAG -Xcc -I${CMAKE_LIBRARY_PATH})
endif()

set(SWIFTC_BRIDGING_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/compute.h)
set(SWIFTC_FLAGS ${SWIFTC_LIBC_INC_FLAG} -c -emit-bc -import-objc-header ${SWIFTC_BRIDGING_HEADER})
set(DLCOMPUTE_MODULE_NAME)

set_directory_properties(PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(Compute
  COMMAND swiftc -module-name="DLVMCompute" ${SWIFTC_FLAGS} ${DLCOMPUTE_SRC}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "DLVM Compute Primitives"
  SOURCES ${DLCOMPUTE_INC} ${DLCOMPUTE_SRC}
  )
