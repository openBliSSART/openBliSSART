cmake_minimum_required(VERSION 3.22)

project(Testsuite)

set(CMAKE_CXX_STANDARD 17) # This is equal to QMAKE_CXX_FLAGS += -std=c++0x

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_SOURCE_DIR /home/gordon/openBliSSART/src/Testsuite) 
set(CMAKE_BINARY_DIR /home/gordon/openBliSSART)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 
# Generating the library testsuite
#set(project "testsuite")

#set(${project}, "")
set (SOURCES 
BinaryReaderWriterTest.cpp 
BinaryReaderWriterTest.h 
CNMFTest.cpp 
CNMFTest.h 
DatabaseSubsystemTest.cpp 
DatabaseSubsystemTest.h 
FeatureExtractionTest.cpp 
FeatureExtractionTest.h 
FeatureSelectionTest.cpp 
FeatureSelectionTest.h 
HTKWriterTest.cpp 
HTKWriterTest.h 
ICATest.cpp 
ICATest.h 
MFCCTest.cpp 
MFCCTest.h 
MTrTest.cpp 
MTrTest.h 
MatrixTest.cpp 
MatrixTest.h 
MelFilterTest.cpp 
MelFilterTest.h 
MinHeapTest.cpp 
MinHeapTest.h 
NMDTest.cpp 
NMDTest.h 
NMFTest.cpp 
NMFTest.h 
PCATest.cpp 
PCATest.h 
#SNMFTest.cpp
#SNMFTest.h
SVMModelTest.cpp 
SVMModelTest.h 
ScalingTest.cpp 
ScalingTest.h 
SpectralAnalysisTest.cpp 
SpectralAnalysisTest.h 
StorageSubsystemTest.cpp 
StorageSubsystemTest.h 
TaskDepsTest.cpp 
TaskDepsTest.h 
Testable.cpp 
Testable.h 
VectorTest.cpp 
VectorTest.h 
WaveTest.cpp 
WaveTest.h 
main.cpp 
)

set (HEADERS
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/FeatureSet.h>
#include "BinaryReaderWriterTest.h"
#include <blissart/BinaryReader.h>
#include <blissart/BinaryWriter.h>
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


add_executable(testsuite ${SOURCES} )
#set_target_properties( testsuite
#    PROPERTIES COMPILE_FLAGS "-I../include "
#)

set(CMAKE_DEBUG_POSTFIX d)
set_target_properties(testsuite
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#target_link_libraries(my_app PRIVATE Threads::Threads)

set_property(TARGET testsuite PROPERTY VERSION "1.2.0")
set_property(TARGET testsuite PROPERTY SOVERSION "1")

target_link_libraries( testsuite 
 PRIVATE Threads::Threads 
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libLinAlg${CMAKE_DEBUG_POSTFIX}.so
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libAudio${CMAKE_DEBUG_POSTFIX}.so
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libNMF${CMAKE_DEBUG_POSTFIX}.so
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libICA${CMAKE_DEBUG_POSTFIX}.so
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFeature${CMAKE_DEBUG_POSTFIX}.so
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFramework${CMAKE_DEBUG_POSTFIX}.so
/usr/local/lib/libSDL2main.a
/usr/local/lib/libSDL2.so
/usr/local/lib/libSDL2_sound.so
/usr/local/lib/libPocoFoundation${CMAKE_DEBUG_POSTFIX}.so
# /usr/local/lib/libPocoDataSQLited.so
# /usr/local/lib/libPocoDatad.so
/usr/local/lib/libPocoUtil${CMAKE_DEBUG_POSTFIX}.so
/usr/local/lib/libPocoXML${CMAKE_DEBUG_POSTFIX}.so
)
