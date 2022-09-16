cmake_minimum_required(VERSION 3.20)

project(cleanup)

set(CMAKE_CXX_STANDARD 20) # This is equal to QMAKE_CXX_FLAGS += -std=c++0x

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_SOURCE_DIR /home/gordon/openBliSSART/src/cleanup) 
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

add_executable(cleanup ${SOURCES} )

#set_target_properties( benchmark
#    PROPERTIES COMPILE_FLAGS "-I../include "
#)

target_link_libraries( cleanup 
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFramework${CMAKE_DEBUG_POSTFIX}.so 
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFeature${CMAKE_DEBUG_POSTFIX}.so
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libLinAlg${CMAKE_DEBUG_POSTFIX}.so 
#${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libAudiod.so 
#${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libNMF${CMAKE_DEBUG_POSTFIX}.so 
#${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libICA${CMAKE_DEBUG_POSTFIX}.so 
#${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFeature${CMAKE_DEBUG_POSTFIX}.so 
#${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFramework${CMAKE_DEBUG_POSTFIX}.so 
#/usr/local/lib/libSDL2main.a
#/usr/local/lib/libSDL2.so
#/usr/local/lib/libSDL2_sound.so
/usr/local/lib/libPocoFoundation${CMAKE_DEBUG_POSTFIX}.so 
# /usr/local/lib/libPocoDataSQLite${CMAKE_DEBUG_POSTFIX}.so
# /usr/local/lib/libPocoData${CMAKE_DEBUG_POSTFIX}.so
/usr/local/lib/libPocoUtil${CMAKE_DEBUG_POSTFIX}.so 
/usr/local/lib/libPocoXML${CMAKE_DEBUG_POSTFIX}.so 
)

#set(THREADS_PREFER_PTHREAD_FLAG ON)
#find_package(Threads REQUIRED)
#target_link_libraries(my_app PRIVATE Threads::Threads)


#target_link_libraries(ICA
# PRIVATE Threads::Threads
#)

#set_target_properties(ICA PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
#set_property(TARGET ICA PROPERTY VERSION "1.0.0")
#set_property(TARGET ICA PROPERTY SOVERSION "1")

 
install(TARGETS cleanup DESTINATION /usr/local/blissart/bin)
#install(FILES ${LinAlgHEADERS} DESTINATION /usr/local/blissart/include)
