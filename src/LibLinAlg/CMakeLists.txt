cmake_minimum_required(VERSION 3.22)
project(LibLinAlg)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_BINARY_DIR /home/gordon/openBliSSART)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
unset(HAVE_CUDA)

#set(CMAKE_PREFIX_PATH "/home/gordon/Qt/6.3.0/gcc_64")
#find_package(Qt6 REQUIRED COMPONENTS QuickWidgets)
#target_link_libraries(mytarget PRIVATE Qt6::QuickWidgets)
#find_package(Qt6 CONFIG REQUIRED COMPONENTS Core Gui)
#set(CMAKE_AUTOMOC TRUE)
#set(CMAKE_AUTORCC TRUE)

set (SOURCES
Vector.cpp 
RowVector.cpp 
ColVector.cpp 
GPUMatrix.cpp 
GPUUtil.cpp 
Matrix.cpp
)

set (HEADERS
../include/common.h 
../include/types.h 
../include/blissart/linalg/Vector.h 
../include/blissart/linalg/ColVector.h 
../include/blissart/linalg/GPUMatrix.h 
../include/blissart/linalg/GPUUtil.h 
../include/blissart/linalg/GPUMatrixKernels.h 
../include/blissart/linalg/Matrix.h 
../include/blissart/linalg/common.h 
../include/blissart/BinaryReader.h 
../include/blissart/BinaryWriter.h 
../include/blissart/linalg/generators/generators.h
)

set (LibAlg_HEADERS
Vector.h
ColVector.h
GPUMatrix.h 
GPUUtil.h 
GPUMatrixKernels.h 
Matrix.h 
common.h)
 
#../include/blissart/BinaryReader.h 
#../include/blissart/BinaryWriter.h 
#../include/blissart/linalg/generators/generators.h


#target must be built by this project
#target_include_directories(LibLinAlg
#INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
#../include)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ../include
    /usr/include
)

add_library(LinAlg SHARED
${SOURCES}
${HEADERS}
)

set(CMAKE_DEBUG_POSTFIX d)

#add_library(tutorial_compiler_flags INTERFACE)

set_target_properties(LinAlg
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(LinAlg PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_property(TARGET LinAlg PROPERTY VERSION "1.2.0")
set_property(TARGET LinAlg PROPERTY SOVERSION "1")


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#target_link_libraries(my_app PRIVATE Threads::Threads)


target_link_libraries(LinAlg
 PRIVATE Threads::Threads
 /usr/lib/x86_64-linux-gnu/libcblas.so 
 /usr/lib/x86_64-linux-gnu/libf77blas.so
 /usr/lib/x86_64-linux-gnu/libatlas.so)

install(TARGETS LinAlg DESTINATION /usr/local/blissart/lib)
#install(FILES ${LinAlgHEADERS} DESTINATION /usr/local/blissart/include)
