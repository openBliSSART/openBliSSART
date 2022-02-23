#-------------------------------------------------
#
# Project created by QtCreator 2018-10-20T17:51:01
#
#-------------------------------------------------

#QMAKE_CXX               = /path/to/custom/g++
#QMAKE_LINK              = /path/to/custom/g++
#QMAKE_LFLAGS            += -custom-link-flags-here
#QMAKE_CC                = /path/to/custom/gcc
#QMAKE_LINK_C            = /path/to/custom/gcc

QT       -= core gui

TARGET = openBlis
TEMPLATE = lib

DEFINES += OPENBLIS_LIBRARY

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH = ./src/include
INCLUDEPATH += /usr/local/include

QMAKE_CXXFLAGS = -std=c++11 -fPIC -DPIC -O3 -MD -MT -MF -Wall -Wextra -fpermissive \
# -Woverloaded-virtual \
 -Wformat-nonliteral -Wformat-security -Winit-self -Wswitch-enum \
 -Wconversion -DNDEBUG -DBUILD_RELEASE \
 -I/usr/local/include -pthread

HEADERS += \
./src/LibAudio/wave.h \
./src/icatool/WaveExporter.h \
./src/icatool/AbstractExporter.h \
./src/icatool/SampleSeparator.h \
./src/icatool/AbstractSeparator.h \
./src/icatool/ARFFExporter.h \
./src/LibFramework/TypeHandler.h \
./src/browser/SamplesPreviewCanvas.h \
./src/browser/ResponseItem.h \
./src/browser/CreateProcessDialog.h \
./src/browser/BrowserController.h \
./src/browser/ThreadedDialog.h \
./src/browser/ClassificationObjectItem.h \
./src/browser/FeatureExtractionDialog.h \
./src/browser/TreeWidgetController.h \
./src/browser/DataDescriptorItem.h \
./src/browser/BrowserMainWindow.h \
./src/browser/PlaybackThread.h \
./src/browser/FilesSelectionWidget.h \
./src/browser/EditWidget.h \
./src/browser/EditWidgetClassificationObject.h \
./src/browser/ResponseQualityDlg.h \
./src/browser/ExportObjectsDlg.h \
./src/browser/EditWidgetLabel.h \
./src/browser/EditWidgetResponse.h \
./src/browser/EntityItem.h \
./src/browser/FeatureItem.h \
./src/browser/LabelItem.h \
./src/browser/SamplesPreviewWidget.h \
./src/browser/PreferencesDlg.h \
./src/browser/ProcessItem.h \
./src/browser/LabelSelectionDialog.h \
./src/Testsuite/TaskDepsTest.h \
./src/Testsuite/MTrTest.h \
./src/Testsuite/DatabaseSubsystemTest.h \
./src/Testsuite/SNMFTest.h \
./src/Testsuite/StorageSubsystemTest.h \
./src/Testsuite/FeatureSelectionTest.h \
./src/Testsuite/PCATest.h \
./src/Testsuite/Testable.h \
./src/Testsuite/HTKWriterTest.h \
./src/Testsuite/GPUMatrixTest.h \
./src/Testsuite/MatrixTest.h \
./src/Testsuite/VectorTest.h \
./src/Testsuite/BinaryReaderWriterTest.h \
./src/Testsuite/MelFilterTest.h \
./src/Testsuite/NMFTest.h \
./src/Testsuite/SpectralAnalysisTest.h \
./src/Testsuite/ICATest.h \
./src/Testsuite/MinHeapTest.h \
./src/Testsuite/NMDTest.h \
./src/Testsuite/FeatureExtractionTest.h \
./src/Testsuite/ScalingTest.h \
./src/Testsuite/CNMFTest.h \
./src/Testsuite/SVMModelTest.h \
./src/Testsuite/WaveTest.h \
./src/Testsuite/MFCCTest.h \
./src/include/blissart/Feature.h \
./src/include/blissart/BasicApplication.h \
./src/include/blissart/MinHeap.h \
./src/include/blissart/FeatureDescriptor.h \
./src/include/blissart/feature/mfcc.h \
./src/include/blissart/feature/peak.h \
./src/include/blissart/feature/misc.h \
./src/include/blissart/ProgressObserver.h \
./src/include/blissart/AudioObject.h \
./src/include/blissart/SVMModel.h \
./src/include/blissart/DataSet.h \
./src/include/blissart/Label.h \
./src/include/blissart/CorrelationFeatureSelector.h \
./src/include/blissart/CrossValidator.h \
./src/include/blissart/CleanupTask.h \
./src/include/blissart/transforms/SpectralSubtractionTransform.h \
./src/include/blissart/transforms/MelFilterTransform.h \
./src/include/blissart/transforms/PowerTransform.h \
./src/include/blissart/transforms/SlidingWindowTransform.h \
./src/include/blissart/MatrixTransform.h \
./src/include/blissart/FeatureSet.h \
./src/include/blissart/ClassificationTask.h \
./src/include/blissart/HTKWriter.h \
./src/include/blissart/DataDescriptor.h \
./src/include/blissart/BinaryReader.h \
./src/include/blissart/Response.h \
./src/include/blissart/ProgressInterface.h \
./src/include/blissart/AnovaFeatureSelector.h \
./src/include/blissart/nmf/Deconvolver.h \
./src/include/blissart/nmf/randomGenerator.h \
./src/include/blissart/nmf/DeconvolverKernels.h \
./src/include/blissart/ProgressObserverAdapter.h \
./src/include/blissart/FTTask.h \
./src/include/blissart/linalg/GPUUtil.h \
./src/include/blissart/linalg/RowVector.h \
./src/include/blissart/linalg/GPUMatrixKernels.h \
./src/include/blissart/linalg/Matrix.h \
./src/include/blissart/linalg/GPUMatrix.h \
./src/include/blissart/linalg/common.h \
./src/include/blissart/linalg/generators/generators.h \
./src/include/blissart/linalg/Vector.h \
./src/include/blissart/linalg/ColVector.h \
./src/include/blissart/SeparationTask.h \
./src/include/blissart/validators.h \
./src/include/blissart/FeatureExtractor.h \
./src/include/blissart/BasicTaskNotification.h \
./src/include/blissart/StorageSubsystem.h \
./src/include/blissart/WindowFunctions.h \
./src/include/blissart/DatabaseEntity.h \
./src/include/blissart/BinaryWriter.h \
./src/include/blissart/ThreadedApplication.h \
./src/include/blissart/QueuedTaskManager.h \
./src/include/blissart/NMDTask.h \
./src/include/blissart/Process.h \
./src/include/blissart/DatabaseSubsystem.h \
./src/include/blissart/ClassificationObject.h \
./src/include/blissart/TargetedDeconvolver.h \
./src/include/blissart/ica/FastICA.h \
./src/include/blissart/ica/PCA.h \
./src/include/blissart/FeatureSelector.h \
./src/include/blissart/GnuplotWriter.h \
./src/include/blissart/FeatureExtractionTask.h \
./src/include/blissart/audio/WaveEncoder.h \
./src/include/blissart/audio/audio.h \
./src/include/blissart/audio/MelFilter.h \
./src/include/blissart/audio/AudioData.h \
./src/include/blissart/audio/Sound.h \
./src/include/blissart/BasicTask.h \
./src/include/blissart/exportDataSet.h \
./src/include/baseName.h \
./src/include/types.h \
./src/include/libsvm/svm.h \
./src/include/common.h

SOURCES += \
./src/LibAudio/audio.cpp \
./src/LibAudio/AudioData.cpp \
./src/LibAudio/MelFilter.cpp \
./src/LibAudio/Sound.cpp \
./src/LibAudio/WaveEncoder.cpp \
./src/LibFeature/mfcc.cpp \
./src/LibFeature/misc.cpp \
./src/LibFeature/peak.cpp \
./src/LibFramework/AnovaFeatureSelector.cpp \
./src/LibFramework/AudioObject.cpp \
./src/LibFramework/BasicApplication.cpp \
./src/LibFramework/BasicTask.cpp \
./src/LibFramework/ClassificationObject.cpp \
./src/LibFramework/ClassificationTask.cpp \
./src/LibFramework/CleanupTask.cpp \
./src/LibFramework/CorrelationFeatureSelector.cpp \
./src/LibFramework/CrossValidator.cpp \
./src/LibFramework/DatabaseEntity.cpp \
./src/LibFramework/DatabaseSubsystem.cpp \
./src/LibFramework/DataDescriptor.cpp \
./src/LibFramework/DataSet.cpp \
./src/LibFramework/exportDataSet.cpp \
./src/LibFramework/Feature.cpp \
./src/LibFramework/FeatureDescriptor.cpp \
./src/LibFramework/FeatureExtractionTask.cpp \
./src/LibFramework/FeatureExtractor.cpp \
./src/LibFramework/FeatureSelector.cpp \
./src/LibFramework/FeatureSet.cpp \
./src/LibFramework/FTTask.cpp \
./src/LibFramework/GnuplotWriter.cpp \
./src/LibFramework/HTKWriter.cpp \
./src/LibFramework/Label.cpp \
./src/LibFramework/libsvm/svm.cpp \
./src/LibFramework/MatrixTransform.cpp \
./src/LibFramework/MelFilterTransform.cpp \
./src/LibFramework/NMDTask.cpp \
./src/LibFramework/PowerTransform.cpp \
./src/LibFramework/Process.cpp \
./src/LibFramework/ProgressInterface.cpp \
./src/LibFramework/QueuedTaskManager.cpp \
./src/LibFramework/Response.cpp \
./src/LibFramework/SeparationTask.cpp \
./src/LibFramework/SlidingWindowTransform.cpp \
./src/LibFramework/SpectralSubtractionTransform.cpp \
./src/LibFramework/StorageSubsystem.cpp \
./src/LibFramework/SVMModel.cpp \
./src/LibFramework/TargetedDeconvolver.cpp \
./src/LibFramework/ThreadedApplication.cpp \
./src/LibFramework/WindowFunctions.cpp \
./src/LibICA/FastICA.cpp \
./src/LibICA/PCA.cpp \
./src/LibLinAlg/ColVector.cpp \
#./src/LibLinAlg/GPUMatrix.cpp \
#./src/LibLinAlg/GPUUtil.cpp \
./src/LibLinAlg/Matrix.cpp \
./src/LibLinAlg/RowVector.cpp \
./src/LibLinAlg/Vector.cpp \
./src/LibNMF/Deconvolver.cpp \
./src/LibNMF/randomGenerator.cpp

QMAKE_CXXFLAGS += -std=c++11 -fpic -Wall -Wextra -Woverloaded-virtual \
 -Wformat-nonliteral -Wformat-security -Winit-self -Wswitch-enum \
 -Wconversion -DNDEBUG -DBUILD_RELEASE \
 -I/usr/local/include -pthread


unix:LIBS += -L/usr/local/lib -lpthread -lPocoFoundation -lPocoUtil -lPocoXML \
    -lPocoDataSQLite -lPocoData -lSDL2main -lSDL2 -lSDL2_sound -lfftw3

unix {
    target.path = /usr/local/lib
    INSTALLS += target
}


#add_library(bliss ${Bliss_SOURCES})
#link_directories("/usr/local/lib")

#add_executable(septool ./src/septool/main.cpp)
#set(link_flags "-L/usr/local/lib" "/usr/lib/x86_64-link-gnu")
#target_link_libraries(septool pthread bliss PocoFoundation PocoUtil PocoXML PocoData PocoDataSQLite SDLmain SDL SDL_sound fftw3)

TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        ./src/Testsuite/BinaryReaderWriterTest.cpp \
        ./src/Testsuite/CNMFTest.cpp \
        ./src/Testsuite/DatabaseSubsystemTest.cpp \
        ./src/Testsuite/FeatureExtractionTest.cpp \
        ./src/Testsuite/FeatureSelectionTest.cpp \
        ./src/Testsuite/HTKWriterTest.cpp \
        ./src/Testsuite/ICATest.cpp \
        ./src/Testsuite/main.cpp \
        ./src/Testsuite/MatrixTest.cpp \
        ./src/Testsuite/MelFilterTest.cpp \
        ./src/Testsuite/MFCCTest.cpp \
        ./src/Testsuite/MinHeapTest.cpp \
        ./src/Testsuite/MTrTest.cpp \
        ./src/Testsuite/NMDTest.cpp \
        ./src/Testsuite/NMFTest.cpp \
        ./src/Testsuite/PCATest.cpp \
        ./src/Testsuite/ScalingTest.cpp \
        ./src/Testsuite/SNMFTest.cpp \
        ./src/Testsuite/SpectralAnalysisTest.cpp \
        ./src/Testsuite/StorageSubsystemTest.cpp \
        ./src/Testsuite/SVMModelTest.cpp \
        ./src/Testsuite/TaskDepsTest.cpp \
        ./src/Testsuite/Testable.cpp \
        ./src/Testsuite/VectorTest.cpp \
        ./src/Testsuite/WaveTest.cpp

### link libraries for testSuite
#target_link_libraries(testSuite pthread PocoFoundation PocoUtil PocoXML PocoData PocoDataSQLite SDLmain SDL SDL_sound fftw3 bliss)



