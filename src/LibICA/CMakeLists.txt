cmake_minimum_required(VERSION 3.22)
set (project LibICA)

#set(${project}_SOURCES
#	${CMAKE_CURRENT_SOURCE_DIR}/PCA.cpp
#	${CMAKE_CURRENT_SOURCE_DIR}/FastICA.cpp
#)

#add_library(${project} STATIC ${${project}_SOURCES} )

#cmake_minimum_required(VERSION 3.22)
#set (project LibFeature)
project(LibICA)

#add_library(${project} SHARED ${${project}_SOURCES} )

#project(browser) # Your project name

set(CMAKE_CXX_STANDARD 17) # This is equal to QMAKE_CXX_FLAGS += -std=c++0x

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_SOURCE_DIR /home/gordon/openBliSSART/src/LibICA) 
set(CMAKE_BINARY_DIR /home/gordon/openBliSSART)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
include_directories(BEFORE ../include ../include/blissart)

#set(CMAKE_PREFIX_PATH "~/Qt/6.3.0/gcc_64")
# This will find the Qt6 files.
#find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets) # Equivalent of QT += core
#qt_standard_project_setup()
#find_package(Qt6Gui REQUIRED)
#find_package(Qt6Widgets REQUIRED) # Equivalent of QT += widgets

set(SOURCES
PCA.cpp
FastICA.cpp
)

set (HEADERS
../include/blissart/ica/FastICA.h
../include/blissart/ica/PCA.h
../include/blissart/linalg/Matrix.h
../include/blissart/linalg/ColVector.h  
../include/blissart/linalg/RowVector.h 
../include/blissart/linalg/generators/generators.h 
)

add_library(ICA SHARED
${SOURCES}
${HEADERS}
)

set(CMAKE_DEBUG_POSTFIX d)

#add_library(tutorial_compiler_flags INTERFACE)

set_target_properties(ICA
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(ICA PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_property(TARGET ICA PROPERTY VERSION "1.2.0")
set_property(TARGET ICA PROPERTY SOVERSION "1")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#target_link_libraries(my_app PRIVATE Threads::Threads)


target_link_libraries(ICA
 PRIVATE Threads::Threads
)

install(TARGETS ICA DESTINATION /usr/local/blissart/lib)
#install(FILES ${LinAlgHEADERS} DESTINATION /usr/local/blissart/include)
