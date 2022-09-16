cmake_minimum_required(VERSION 3.20)

#set (project LibFramework)

project(LibFramework)

set(CMAKE_CXX_STANDARD 20) # This is equal to QMAKE_CXX_FLAGS += -std=c++0x

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_SOURCE_DIR /home/gordon/openBliSSART/src/LibFramework) 
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
    AnovaFeatureSelector.cpp 
    AudioObject.cpp 
    BasicApplication.cpp 
    BasicTask.cpp 
    ClassificationObject.cpp 
    ClassificationTask.cpp 
    CleanupTask.cpp 
    CorrelationFeatureSelector.cpp 
    CrossValidator.cpp 
    DataDescriptor.cpp 
    DataSet.cpp 
    DatabaseEntity.cpp 
    DatabaseSubsystem.cpp 
    FTTask.cpp 
    Feature.cpp 
    FeatureDescriptor.cpp 
    FeatureExtractionTask.cpp 
    FeatureExtractor.cpp 
    FeatureSelector.cpp 
    FeatureSet.cpp 
    GnuplotWriter.cpp 
    HTKWriter.cpp 
    Label.cpp 
    MatrixTransform.cpp 
    MelFilterTransform.cpp 
    NMDTask.cpp 
    PowerTransform.cpp 
    Process.cpp 
    ProgressInterface.cpp 
    QueuedTaskManager.cpp 
    Response.cpp 
    SeparationTask.cpp 
    SlidingWindowTransform.cpp 
    SpectralSubtractionTransform.cpp 
    SVMModel.cpp 
    StorageSubsystem.cpp 
    TargetedDeconvolver.cpp 
    ThreadedApplication.cpp 
    WindowFunctions.cpp 
    exportDataSet.cpp 
    libsvm/svm.cpp
)

set (HEADERS
/usr/local/include/Poco/Data/TypeHandler.h 
#/usr/local/include/Poco/Data/AbstractTypeHandler.h 
/usr/local/include/Poco/Data/AbstractPreparation.h 
/usr/local/include/Poco/Data/AbstractBinder.h 
/usr/local/include/Poco/Data/AbstractExtractor.h 
/usr/local/include/Poco/Data/AbstractPreparator.h 
/usr/local/include/Poco/Util/AbstractConfiguration.h 
/usr/local/include/Poco/Util/Util.h 
TypeHandler.h 
../include/libsvm/svm.h 
../include/blissart/AnovaFeatureSelector.h 
../include/blissart/DatabaseSubsystem.h 
../include/blissart/FTTask.h 
../include/blissart/BasicApplication.h 
#config.h
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


add_library(Framework SHARED
${SOURCES}
${HEADERS}
)

set(CMAKE_DEBUG_POSTFIX d)

#add_library(tutorial_compiler_flags INTERFACE)


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)#target_link_libraries(my_app PRIVATE Threads::Threads)

set_target_properties(Framework PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(Framework
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

set_property(TARGET Framework PROPERTY VERSION "1.2.0")
set_property(TARGET Framework PROPERTY SOVERSION "1")

target_link_libraries(Framework
PRIVATE Threads::Threads 
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libLinAlg${CMAKE_DEBUG_POSTFIX}.so 
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libAudio${CMAKE_DEBUG_POSTFIX}.so 
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libNMF${CMAKE_DEBUG_POSTFIX}.so 
${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFeature${CMAKE_DEBUG_POSTFIX}.so 
/usr/local/lib/libSDL2main.a 
/usr/local/lib/libSDL2.so 
/usr/local/lib/libSDL2_sound.so 
/usr/local/lib/libPocoFoundation${CMAKE_DEBUG_POSTFIX}.so 
/usr/local/lib/libPocoDataSQLite${CMAKE_DEBUG_POSTFIX}.so 
/usr/local/lib/libPocoData${CMAKE_DEBUG_POSTFIX}.so 
/usr/local/lib/libPocoUtil${CMAKE_DEBUG_POSTFIX}.so 
/usr/local/lib/libPocoXML${CMAKE_DEBUG_POSTFIX}.so 
/usr/lib/x86_64-linux-gnu/libGL.so 
/usr/lib/x86_64-linux-gnu/libfftw3.so
)

install(TARGETS Framework DESTINATION /usr/local/blissart/lib)
#install(FILES ${LinAlgHEADERS} DESTINATION /usr/local/blissart/include)
