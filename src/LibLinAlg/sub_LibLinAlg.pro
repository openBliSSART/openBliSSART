TEMPLATE = lib
CONFIG += debug dll
##CONFIG += staticlib
##CONFIG += debug_and_release dll
CONFIG -= qt

release:DESTDIR = release
debug:DESTDIR = debug

OBJECTS_DIR = $$DESTDIR/.obj
MOC_DIR = $$DESTDIR/.moc
RCC_DIR = $$DESTDIR/.qrc
UI_DIR = $$DESTDIR/.u

CONFIG(debug, debug|release) {
    TARGET = LinAlgd
##    QMAKE_LINK_SHLIB = libLinAlg.so
} else {
    TARGET = LinAlg
##    QMAKE_LINK_SHLIB = libLinAlgd.so
}

QMAKE_CXXFLAGS += "-std=c++11 -fPIC -DPIC -O2 -MD -MT -MF -Wall -Wextra -fpermissive"
##-I/home/gordon/openBliSSART/src/include -pthread"
##-Wl,--verbose 
## -Wl,--whole-archive libLinAlgd.a 
## -Woverloaded-virtual 
## -Wformat-nonliteral -Wformat-security -Winit-self -Wswitch-enum 

QMAKE_LFLAGS += "-shared -o libLinAlgd.so -lm -ggdb -pthread"

cuda {
DEFINES += "HAVE_CUDA = 1"

INCLUDEPATH += ../include

HEADERS += \
        ../include/common.h \
        ../include/types.h \
        ../include/blissart/linalg/Vector.h \
        ../include/blissart/linalg/ColVector.h \
        ../include/blissart/linalg/GPUMatrix.h \
        ../include/blissart/linalg/GPUUtil.h \
        ../include/blissart/linalg/GPUMatrixKernels.h \
        ../include/blissart/linalg/Matrix.h \
        ../include/blissart/linalg/common.h \
        ../include/blissart/BinaryReader.h \
        ../include/blissart/BinaryWriter.h \
        ../include/blissart/linalg/generators/generators.h


SOURCES += \
        Vector.cpp \
        RowVector.cpp \
        ColVector.cpp \
        GPUMatrix.cpp \
        GPUUtil.cpp \
        Matrix.cpp

release {
    lib1.files = linAlg.*
    lib1.path = /usr/local/blissart/lib
    unix:LIBS += -L/lib/x86_64-linux-gnu -lcblas \
        -L/usr/local/cuda/lib64 -lcublas \
        -L/usr/lib/x86_64-linux-gnu -lf77blas \
        -L/usr/lib/x86_64-linux-gnu -latlas \
        -L/usr/local/lib -lpthread
##    OBJECTS += $$OBJECTS_DIR/*.o
    INSTALLS += lib1
    }

debug {
    lib1.files = linAlgd.*
    lib1.path = /usr/local/blissart/lib
    unix:LIBS += -L/lib/x86_64-linux-gnu -lcblasd \
    -L/usr/local/cuda/lib64 -lcublas \
    -L/usr/lib/x86_64-linux-gnu -lf77blas \
    -L/usr/lib/x86_64-linux-gnu -latlas \
    -L/usr/local/lib -lpthread
##    OBJECTS += $$OBJECTS_DIR/*.*
    INSTALLS += lib1
    }
    
} else {
DEFINES += "HAVE_CUDA = 0"
#DEPENDPATH += $$PWD
INCLUDEPATH = ../include
HEADERS += \
        ../include/common.h \
        ../include/types.h \
        ../include/blissart/linalg/Vector.h \
        ../include/blissart/linalg/ColVector.h \
        ../include/blissart/linalg/Matrix.h \
        ../include/blissart/linalg/common.h \
        ../include/blissart/BinaryReader.h \
        ../include/blissart/BinaryWriter.h \
        ../include/blissart/linalg/generators/generators.h

SOURCES += \
        Vector.cpp \
        RowVector.cpp \
        ColVector.cpp \
        Matrix.cpp

release {
    lib1.files = linAlg.*
    lib1.path = /usr/local/blissart/lib
    LIBS += -L/usr/lib/x86_64-linux-gnu -lcblas \
        -L/usr/lib/x86_64-linux-gnu -lf77blas \
        -L/usr/lib/x86_64-linux-gnu -latlas \
        -L/usr/local/lib -lpthread
##    OBJECTS += *.o
    INSTALLS += lib1
    }

debug {
    LIBS += -L/usr/lib/x86_64-linux-gnu -lcblas \
    -L/usr/lib/x86_64-linux-gnu -lf77blas \
    -L/usr/lib/x86_64-linux-gnu -latlas \
    -L/usr/local/lib -lpthread
##    INSTALLS += lib1
    QT_INSTALL_LIBS = /home/gordon/openBliSSART/src/LibLinAlg
    libraryFiles.files = libLinAlgd.la libLinAlgd.so libLinAlgd.so.1 libLinAlgd.so.1.0 libLinAlgd.so.1.0.0
    libraryFiles.path = $$[QT_INSTALL_LIBS]
    INSTALLS += libraryFiles
    }
}


### CONFIG(debug, debug|release):libraryFiles.files = $$OUT_PWD/debug/*.a $$OUT_PWD/debug/*.prl
### CONFIG(release, debug|release):libraryFiles.files = $$OUT_PWD/release/*.a $$OUT_PWD/release/*.prl
##QT_INSTALL_LIBS = /home/gordon/openBliSSART/src/LibLinAlg/.libs
##libraryFiles.files = libLinAlgd.la libLinAlgd.so libLinAlgd.so.1 libLinAlgd.so.1.0 libLinAlgd.so.1.0.0
##libraryFiles.path = $$[QT_INSTALL_LIBS]
##INSTALLS += libraryFiles
