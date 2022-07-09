cmake_minimum_required(VERSION 3.22)

project(Testsuite)

set(CMAKE_CXX_STANDARD 17) # This is equal to QMAKE_CXX_FLAGS += -std=c++0x

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_SOURCE_DIR /home/gordon/openBliSSART/src/benchmark) 
set(CMAKE_BINARY_DIR /home/gordon/openBliSSART)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# Generating the library testsuite
#set(project "testsuite")
# Generating the library benchmark
#set(project "benchmark")

set (SOURCES
Benchmark.cpp
Benchmark.h
NMDBenchmark.cpp
NMDBenchmark.h
NMFBenchmark.cpp
NMFBenchmark.h
main.cpp
)

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

add_executable(benchmark ${SOURCES} )
#set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

#set_target_properties( benchmark
#    PROPERTIES COMPILE_FLAGS "-I../include "
#)

set_target_properties(benchmark
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#target_link_libraries(my_app PRIVATE Threads::Threads)


target_link_libraries( benchmark 
PRIVATE Threads::Threads
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libLinAlgd.so 
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libAudiod.so 
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libNMFd.so 
#${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libICAd.so 
#${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFeatured.so 
#${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFrameworkd.so 
/usr/local/lib/libSDL2main.a
/usr/local/lib/libSDL2.so
/usr/local/lib/libSDL2_sound.so
/usr/local/lib/libPocoFoundationd.so 
# /usr/local/lib/libPocoDataSQLited.so
# /usr/local/lib/libPocoDatad.so
/usr/local/lib/libPocoUtild.so 
/usr/local/lib/libPocoXMLd.so 
)

install(TARGETS benchmark DESTINATION /usr/local/blissart/bin)
#install(FILES ${LinAlgHEADERS} DESTINATION /usr/local/blissart/include)
