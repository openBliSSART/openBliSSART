cmake_minimum_required(VERSION 2.8.4)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules")

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
    /usr/local/include
)

add_library(bliss ${Bliss_SOURCES})
link_directories("/usr/local/lib")

add_executable(septool ./src/septool/main.cpp)
#set(link_flags "-L/usr/local/lib")
target_link_libraries(septool bliss PocoFoundation PocoUtil PocoXML PocoData PocoSQLite SDLMain SDL SDL_sound fftw3)
