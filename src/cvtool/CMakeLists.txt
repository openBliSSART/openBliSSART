cmake_minimum_required(VERSION 3.22)

project(cvtool)

set(CMAKE_CXX_STANDARD 20) # This is equal to QMAKE_CXX_FLAGS += -std=c++0x

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_SOURCE_DIR /home/gordon/openBliSSART/src/cvtool) 
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

add_executable(cvtool ${SOURCES} )

set_target_properties(cvtool
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

set(CMAKE_DEBUG_POSTFIX d)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#target_link_libraries(my_app PRIVATE Threads::Threads)


target_link_libraries( cvtool 
PRIVATE Threads::Threads
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libLinAlg${CMAKE_DEBUG_POSTFIX}.so 
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libAudio${CMAKE_DEBUG_POSTFIX}.so 
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFeature${CMAKE_DEBUG_POSTFIX}.so
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFramework${CMAKE_DEBUG_POSTFIX}.so 
#${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libNMFd.so 
#${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libICAd.so 
#${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFeatured.so 
#${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFrameworkd.so 
#/usr/local/lib/libSDL2main.a
#/usr/local/lib/libSDL2.so
#/usr/local/lib/libSDL2_sound.so
/usr/local/lib/libPocoFoundationd.so 
# /usr/local/lib/libPocoDataSQLited.so
# /usr/local/lib/libPocoDatad.so
/usr/local/lib/libPocoUtild.so 
/usr/local/lib/libPocoXMLd.so 
)

#set_target_properties(ICA PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
#set_property(TARGET ICA PROPERTY VERSION "1.2.0")
#set_property(TARGET ICA PROPERTY SOVERSION "1")

 
install(TARGETS browser DESTINATION /usr/local/blissart/bin)
#install(FILES ${LinAlgHEADERS} DESTINATION /usr/local/blissart/include)
