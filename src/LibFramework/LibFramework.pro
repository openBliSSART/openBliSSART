#-------------------------------------------------
#
# Project created by QtCreator 2018-11-05T18:39:30
#
#-------------------------------------------------

QT       -= core gui

CONFIG = release
CONFIG += debug_and_release

CONFIG(debug, debug|release) {
    TARGET = Frameworkd
} else {
    TARGET = Framework
}

CONFIG(debug, debug|release) {
    LibFramework.depends = LibLinAlgd
    LibFramework.depends = LibFeatured
    LibFramework.depends = LibNMFd
} else {
    LibFramework.depends = LibLinAlg
    LibFramework.depends = LibFeature
    LibFramework.depends = LibNMF
}

#build_pass:CONFIG(debug, debug|release) {
#    TARGET = $$join(TARGET,,,d)
#    }
TEMPLATE = lib

DEFINES += FRAMEWORK_LIBRARY

DEFINES += QT_DEPRECATED_WARNINGS

QMAKE_CXXFLAGS = -std=c++11 -fPIC -DPIC -O3 -MD -MT -MF -Wall -Wextra -fpermissive \
# -Woverloaded-virtual \
 -Wformat-nonliteral -Wformat-security -Winit-self -Wswitch-enum \
 -Wconversion -DNDEBUG -DBUILD_RELEASE \
 -I/usr/local/include -pthread \
 -lLinAlg -lFeature -lNMF -lPocoFoundation -lPocoUtil -lPocoXML -lPocoDataSQLite -lPocoData -lPocoFeature

SOURCES += \
    AnovaFeatureSelector.cpp \
    AudioObject.cpp \
    BasicApplication.cpp \
    BasicTask.cpp \
    ClassificationObject.cpp \
    ClassificationTask.cpp \
    CleanupTask.cpp \
    CorrelationFeatureSelector.cpp \
    CrossValidator.cpp \
    DataDescriptor.cpp \
    DataSet.cpp \
    DatabaseEntity.cpp \
    DatabaseSubsystem.cpp \
    FTTask.cpp \
    Feature.cpp \
    FeatureDescriptor.cpp \
    FeatureExtractionTask.cpp \
    FeatureExtractor.cpp \
    FeatureSelector.cpp \
    FeatureSet.cpp \
    GnuplotWriter.cpp \
    HTKWriter.cpp \
    Label.cpp \
    MatrixTransform.cpp \
    MelFilterTransform.cpp \
    NMDTask.cpp \
    PowerTransform.cpp \
    Process.cpp \
    ProgressInterface.cpp \
    QueuedTaskManager.cpp \
    Response.cpp \
    SeparationTask.cpp \
    SlidingWindowTransform.cpp \
    SpectralSubtractionTransform.cpp \
    SVMModel.cpp \
    StorageSubsystem.cpp \
    TargetedDeconvolver.cpp \
    ThreadedApplication.cpp \
    WindowFunctions.cpp \
    exportDataSet.cpp \
    libsvm/svm.cpp

INCLUDEPATH += /usr/local/include/SDL /usr/local/include ../include ../../
#INCLUDEPATH += /usr/local/include ../include ../../

HEADERS += \
        /usr/local/include/Poco/Data/TypeHandler.h \
        /usr/local/include/Poco/Data/AbstractTypeHandler.h \
        /usr/local/include/Poco/Data/AbstractPreparation.h \
        /usr/local/include/Poco/Data/AbstractBinder.h \
        /usr/local/include/Poco/Data/AbstractExtractor.h \
        /usr/local/include/Poco/Data/AbstractPreparator.h \
        TypeHandler.h \
        ../include/blissart/AnovaFeatureSelector.h \
        ../include/blissart/DatabaseSubsystem.h \
        config.h

release {
    unix:LIBS += -L/usr/local/lib -lPocoFoundation -lPocoUtil -lPocoXML \
        -lPocoDataSQLite -lPocoData \
        -L../LibLinAlg -lLinAlg \
        -L../LibFeature -lFeature \
        -L../LibNMF -lNMF \
        -L/usr/local/lib -lSDL2main -lSDL2 -lSDL2_sound -lfftw3 -lGL -lpthread
    }
debug {
    unix:LIBS += -L/usr/local/lib -lPocoFoundationd -lPocoUtild -lPocoXMLd \
        -lPocoDataSQLited -lPocoDatad \
        -L../LibLinAlg -lLinAlgd \
        -L../LibFeature -lFeatured \
        -L../LibNMF -lNMFd \
        -L/usr/local/lib -lSDL2main \
        -L/usr/local/lib -lSDL2 \
        -L/usr/local/lib -lSDL2_sound \
        -lfftw3 -lGL -lpthread
    }


unix {
    target.path = /usr/local/blissart/lib
    INSTALLS += target
}


