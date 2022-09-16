cmake_minimum_required(VERSION 3.20)
project(LibAudio)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_SOURCE_DIR /home/gordon/openBliSSART/src/LibAudio)
set(CMAKE_BINARY_DIR /home/gordon/openBliSSART)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#set(CMAKE_PREFIX_PATH "/home/gordon/Qt/6.3.0/gcc_64")
#find_package(Qt6 REQUIRED COMPONENTS QuickWidgets)
#target_link_libraries(mytarget PRIVATE Qt6::QuickWidgets)
#find_package(Qt6 CONFIG REQUIRED COMPONENTS Core Gui)
#set(CMAKE_AUTOMOC TRUE)
#set(CMAKE_AUTORCC TRUE)
#${CMAKE_CURRENT_SOURCE_DIR}/MelFilter.cpp

set (SOURCES
MelFilter.cpp 
AudioData.cpp 
Sound.cpp 
audio.cpp 
WaveEncoder.cpp
)

set (HEADERS
../include/linalg/Matrix.h 
../include/blissart/audio/wave.h 
../include/blissart/audio/MelFilter.h 
../include/linalg/generators/generators.h 
../include/blissart/audio/Sound.h 
../include/blissart/audio/audio.h 
../include/blissart/audio/WaveEncoder.h 
../include/blissart/audio/AudioData.h 
../include/blissart/ProgressObserver.h
)

#fftw3.h
#cassert
#cstring
#cmath
#include <cstdio>
#include <cstdlib>
#include <Poco/Mutex.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_sound.h>
#include <SDL2/SDL_audio.h>

#ib_LTLIBRARIES = libAudio.la
#ibAudio_la_LIBADD = $(top_srcdir)/src/LibLinAlg/libLinAlg.la \
#                    -L/usr/local/lib -Wl,-rpath,/usr/local/lib -Wl,--enable-new-dtags -pthread -lSDL2 -lSDL2_sound -lfftw3 

include_directories(
    ${CMAKE_SOURCE_DIR} 
    ../include 
    /usr/local/include/SDL2 
    /usr/local/include 
)

add_library(Audio SHARED
${SOURCES}
#${HEADERS}
)

set(CMAKE_DEBUG_POSTFIX d)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#target_link_libraries(my_app PRIVATE Threads::Threads)

#find_package(PkgConfig REQUIRED)
#pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
#include_directories(PkgConfig::FFTW)
#ink_libraries     (PkgConfig::FFTW -lfftw3_thread)

set_target_properties(Audio PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(Audio
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_property(TARGET Audio PROPERTY VERSION "1.2.0")
set_property(TARGET Audio PROPERTY SOVERSION "1")

target_link_libraries(Audio
 PRIVATE Threads::Threads
 /usr/local/lib/libSDL2.so
 /usr/local/lib/libSDL2_sound.so
 /usr/lib/x86_64-linux-gnu/libfftw3.so)
 
install(TARGETS Audio DESTINATION /usr/local/blissart/lib)
