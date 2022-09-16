cmake_minimum_required(VERSION 3.20)
#set (project LibFeature)
project(LibFeature)

#add_library(${project} SHARED ${${project}_SOURCES} )

#project(browser) # Your project name

set(CMAKE_CXX_STANDARD 17) # This is equal to QMAKE_CXX_FLAGS += -std=c++0x

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_SOURCE_DIR /home/gordon/openBliSSART/src/LibFeature) 
set(CMAKE_BINARY_DIR /home/gordon/openBliSSART)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#include_directories(BEFORE ../include ../include/blissart)

#set(CMAKE_PREFIX_PATH "~/Qt/6.3.0/gcc_64")
# This will find the Qt6 files.
#find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets) # Equivalent of QT += core
#qt_standard_project_setup()
#find_package(Qt6Gui REQUIRED)
#find_package(Qt6Widgets REQUIRED) # Equivalent of QT += widgets

set(SOURCES
misc.cpp
mfcc.cpp
peak.cpp
)

set (HEADERS
../include/blissart/feature/mfcc.h 
../include/blissart/feature/misc.h 
../include/blissart/feature/peak.h 
../include/blissart/linalg/Vector.h  
../include/blissart/linalg/Matrix.h 
../include/blissart/linalg/generators/generators.h 
../include/blissart/audio/MelFilter.h
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

add_library(Feature SHARED
${SOURCES}
#${HEADERS}
)

set(CMAKE_DEBUG_POSTFIX d)

#add_library(tutorial_compiler_flags INTERFACE)


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#target_link_libraries(my_app PRIVATE Threads::Threads)

set_target_properties(Feature PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(Feature
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

set_property(TARGET Feature PROPERTY VERSION "1.2.0")
set_property(TARGET Feature PROPERTY SOVERSION "1")

target_link_libraries(Feature
 PRIVATE Threads::Threads
 ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libLinAlg${CMAKE_DEBUG_POSTFIX}.so
)


install(TARGETS Feature DESTINATION /usr/local/blissart/lib)
#install(FILES ${LinAlgHEADERS} DESTINATION /usr/local/blissart/include)
