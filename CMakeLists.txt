cmake_minimum_required(VERSION 2.8.4)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules")
include_directories (BEFORE "/usr/local/include")

#include_directories(${CMAKE_SOURCE_DIR}/inc)
#link_directories(${CMAKE_SOURCE_DIR}/lib)
SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -fPIC -DPIC -O3 -MD -MT -MF -Wall -Wextra -fpermissive \
 -Woverloaded-virtual \
 -Wformat-nonliteral -Wformat-security -Winit-self -Wswitch-enum \
 -Wconversion -DNDEBUG -DBUILD_RELEASE \
 -I/usr/local/include -pthread")
#SET(GCC_COVERAGE_LINK_FLAGS    "-lgcov")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
#SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

PROJECT(Bliss)

set(Bliss_SOURCES
./src/LibAudio/audio.cpp
./src/LibAudio/AudioData.cpp
./src/LibAudio/MelFilter.cpp
./src/LibAudio/Sound.cpp
./src/LibAudio/WaveEncoder.cpp
./src/LibFeature/mfcc.cpp
./src/LibFeature/misc.cpp
./src/LibFeature/peak.cpp
./src/LibFramework/AnovaFeatureSelector.cpp
./src/LibFramework/AudioObject.cpp
./src/LibFramework/BasicApplication.cpp
./src/LibFramework/BasicTask.cpp
./src/LibFramework/ClassificationObject.cpp
./src/LibFramework/ClassificationTask.cpp
./src/LibFramework/CleanupTask.cpp
./src/LibFramework/CorrelationFeatureSelector.cpp
./src/LibFramework/CrossValidator.cpp
./src/LibFramework/DatabaseEntity.cpp
./src/LibFramework/DatabaseSubsystem.cpp
./src/LibFramework/DataDescriptor.cpp
./src/LibFramework/DataSet.cpp
./src/LibFramework/exportDataSet.cpp
./src/LibFramework/Feature.cpp
./src/LibFramework/FeatureDescriptor.cpp
./src/LibFramework/FeatureExtractionTask.cpp
./src/LibFramework/FeatureExtractor.cpp
./src/LibFramework/FeatureSelector.cpp
./src/LibFramework/FeatureSet.cpp
./src/LibFramework/FTTask.cpp
./src/LibFramework/GnuplotWriter.cpp
./src/LibFramework/HTKWriter.cpp
./src/LibFramework/Label.cpp
./src/LibFramework/libsvm/svm.cpp
./src/LibFramework/MatrixTransform.cpp
./src/LibFramework/MelFilterTransform.cpp
./src/LibFramework/NMDTask.cpp
./src/LibFramework/PowerTransform.cpp
./src/LibFramework/Process.cpp
./src/LibFramework/ProgressInterface.cpp
./src/LibFramework/QueuedTaskManager.cpp
./src/LibFramework/Response.cpp
./src/LibFramework/SeparationTask.cpp
./src/LibFramework/SlidingWindowTransform.cpp
./src/LibFramework/SpectralSubtractionTransform.cpp
./src/LibFramework/StorageSubsystem.cpp
./src/LibFramework/SVMModel.cpp
./src/LibFramework/TargetedDeconvolver.cpp
./src/LibFramework/ThreadedApplication.cpp
./src/LibFramework/WindowFunctions.cpp
./src/LibICA/FastICA.cpp
./src/LibICA/PCA.cpp
./src/LibLinAlg/ColVector.cpp
#./src/LibLinAlg/GPUMatrix.cpp
#./src/LibLinAlg/GPUUtil.cpp
./src/LibLinAlg/Matrix.cpp
./src/LibLinAlg/RowVector.cpp
./src/LibLinAlg/Vector.cpp
./src/LibNMF/Deconvolver.cpp
./src/LibNMF/randomGenerator.cpp
)

include_directories(
    /usr/local/include
    ${CMAKE_SOURCE_DIR}
    ./src/benchmark
    ./src/browser
    ./src/icatool
    ./src/include
    ./src/include/blissart
    ./src/include/blissart/audio
    ./src/include/blissart/feature
    ./src/include/blissart/ica
    ./src/include/blissart/linalg
    ./src/include/blissart/nmf
    ./src/include/blissart/transforms
    ./src/include/libsvm
    ./src/LibAudio
    ./src/LibFramework
    /usr/include
)

add_library(bliss ${Bliss_SOURCES})
link_directories("/usr/local/lib" "/usr/lib/x86_64-linux-gnu")

add_executable(septool ./src/septool/main.cpp)
#set(link_flags "-L/usr/local/lib" "-L/usr/lib/x86_64-linux-gnu")
#set(link_flags "-L/usr/local/lib" )
target_link_libraries(septool bliss PocoFoundation PocoUtil PocoXML PocoSQL PocoSQLSQLite SDL2_sound SDL2main SDL2 fftw3 pthread)

find_package(Threads)
add_executable(testSuite
	./src/Testsuite/BinaryReaderWriterTest.cpp
	./src/Testsuite/CNMFTest.cpp
	./src/Testsuite/DatabaseSubsystemTest.cpp
	./src/Testsuite/FeatureExtractionTest.cpp
	./src/Testsuite/FeatureSelectionTest.cpp
	./src/Testsuite/HTKWriterTest.cpp
#	./src/Testsuite/ICATest.cpp
	./src/Testsuite/main.cpp
	./src/Testsuite/MatrixTest.cpp
	./src/Testsuite/MelFilterTest.cpp
	./src/Testsuite/MFCCTest.cpp
	./src/Testsuite/MinHeapTest.cpp
	./src/Testsuite/MTrTest.cpp
	./src/Testsuite/NMDTest.cpp
	./src/Testsuite/NMFTest.cpp
	./src/Testsuite/PCATest.cpp
	./src/Testsuite/ScalingTest.cpp
	./src/Testsuite/SNMFTest.cpp
	./src/Testsuite/SpectralAnalysisTest.cpp
	./src/Testsuite/StorageSubsystemTest.cpp
	./src/Testsuite/SVMModelTest.cpp
	./src/Testsuite/TaskDepsTest.cpp
	./src/Testsuite/Testable.cpp
	./src/Testsuite/VectorTest.cpp
	./src/Testsuite/WaveTest.cpp
	)
target_link_libraries(testSuite PocoFoundation PocoUtil PocoXML PocoSQL PocoSQLSQLite SDL2_sound SDL2main SDL2 fftw3 bliss ${CMAKE_THREAD_LIBS_INIT})
