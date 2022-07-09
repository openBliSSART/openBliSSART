cmake_minimum_required(VERSION 3.22)

project(fextool)

set(CMAKE_CXX_STANDARD 17) # This is equal to QMAKE_CXX_FLAGS += -std=c++0x

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_SOURCE_DIR /home/gordon/openBliSSART/src/fextool) 
set(CMAKE_BINARY_DIR /home/gordon/openBliSSART)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 
# Generating the library testsuite
#set(project "testsuite")
# Generating the library benchmark
#set(project "benchmark")

set (SOURCES
main.cpp
)
# Generating the library fextool
set(project "fextool")

include_directories(
    /usr/local/include/SDL2
    /usr/local/include
    ${CMAKE_SOURCE_DIR}
    ../benchmark
    ../browser
    ../icatool
    ../include
    ../include/blissart
    ../include/blissart/audio
    ../include/blissart/feature
    ../include/blissart/ica
    ../include/blissart/linalg
    ../include/blissart/nmf
    ../include/blissart/transforms
    ../include/libsvm
    ../LibAudio
    ../LibICA
    ../LibLinAlg
    ../LibNMF
    ../LibFramework
    ../LibFeature
    ../browser
    /usr/include
)

add_executable(fextool ${SOURCES} )

#set_target_properties( fextool
#    PROPERTIES COMPILE_FLAGS "-I../include "
#)

set_target_properties(fextool
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#target_link_libraries(my_app PRIVATE Threads::Threads)

# note LinAlg then Feature then Framework
target_link_libraries( fextool 
PRIVATE Threads::Threads 
${LIBRARY_OUTPUT_DIRECTORY}/libFrameworkd.so 
${LIBRARY_OUTPUT_DIRECTORY}/libFeatured.so
${LIBRARY_OUTPUT_DIRECTORY}/libLinAlgd.so 
#${LIBRARY_OUTPUT_DIRECTORY}/libAudiod.so 
#${LIBRARY_OUTPUT_DIRECTORY}/libNMFd.so 
#${LIBRARY_OUTPUT_DIRECTORY}/libICAd.so 
#${LIBRARY_OUTPUT_DIRECTORY}/libFeatured.so 
#/usr/local/blissart/lib/libFrameworkd.so 
#/usr/local/lib/libSDL2main.a
#/usr/local/lib/libSDL2.so
#/usr/local/lib/libSDL2_sound.so
/usr/local/lib/libPocoFoundationd.so 
/usr/local/lib/libPocoDataSQLited.so
# /usr/local/lib/libPocoDatad.so
/usr/local/lib/libPocoUtild.so 
/usr/local/lib/libPocoXMLd.so 
/usr/lib/x86_64-linux-gnu/libfftw3.so 
)
