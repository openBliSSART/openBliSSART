TEMPLATE = lib
CONFIG += debug
#CONFIG += debug_and_release

CONFIG -= qt

release:DESTDIR = release
debug:DESTDIR = debug

OBJECTS_DIR = $$DESTDIR/.obj
MOC_DIR = $$DESTDIR/.moc
RCC_DIR = $$DESTDIR/.qrc
UI_DIR = $$DESTDIR/.u

CONFIG(debug, debug|release) {
    TARGET = LinAlgd
    QMAKE_LINK_SHLIB = libLinAlg.so
} else {
    TARGET = LinAlg
    QMAKE_LINK_SHLIB = libLinAlgd.so
}

QMAKE_CXXFLAGS = -std=c++11 -fPIC -DPIC -O3 -MD -MT -MF -Wall -Wextra -fpermissive \
 -Wformat-nonliteral -Wformat-security -Winit-self -Wswitch-enum \
 -Wconversion -Wstrict-aliasing \
 -I/usr/local/include -pthread

LIBS += "-Wl,--verbose"

cuda {
DEFINES += "HAVE_CUDA = 1"
HEADERS += \
        ../include/common.h \
        ../include/types.h \
        ../include/linalg/Vector.h \
        ../include/linalg/ColVector.h \
        ../include/linalg/GPUMatrix.h \
        ../include/linalg/GPUUtil.h \
        ../include/linalg/GPUMatrixKernels.h \
        ../include/linalg/Matrix.h \
        ../include/linalg/common.h \
        ../include/BinaryReader.h \
        ../include/BinaryWriter.h \
        ../include/linalg/generators/generators.h

INCLUDEPATH += ../include

SOURCES += \
        Vector.cpp \
        RowVector.cpp \
        ColVector.cpp \
        GPUMatrix.cpp \
        GPUUtil.cpp \
        Matrix.cpp

release {
    lib1.files = linAlg.*
    lib1.path = /usr/local/lib
    LIBS += -L/lib/x86_64-linux-gnu -lcblas \
        -L/usr/local/cuda/lib64 -lcublas \
        -L/lib/x86_64-linux-gnu -lf77blas \
        -L/lib/x86_64-linux-gnu -latlas \
        -L/usr/local/lib -lpthread
    OBJECTS += $$OBJECTS_DIR/*.o
    INSTALLS += lib1
    }

debug {
    lib1.files = linAlgd.*
    lib1.path = /usr/local/lib
    LIBS += -L/lib/x86_64-linux-gnu -lcblasd \
    -L/usr/local/cuda/lib64 -lcublas \
    -L/lib/x86_64-linux-gnu -lf77blasd \
    -L/lib/x86_64-linux-gnu -latlasd \
    -L/usr/local/lib -lpthread
    OBJECTS += $$OBJECTS_DIR/*.*
    INSTALLS += lib1
    }
    
} else {
DEFINES += "HAVE_CUDA = 0"
HEADERS += \
        ../include/common.h \
        ../include/types.h \
        ../include/linalg/Vector.h \
        ../include/linalg/ColVector.h \
        ../include/linalg/Matrix.h \
        ../include/linalg/common.h \
        ../include/BinaryReader.h \
        ../include/BinaryWriter.h \
        ../include/linalg/generators/generators.h

INCLUDEPATH += ../include

SOURCES += \
        Vector.cpp \
        RowVector.cpp \
        ColVector.cpp \
        Matrix.cpp

release {
    LIBS += -L$$PWD/$$OBJECTS_DIR -l*.o 
    LIBS += -L/lib/x86_64-linux-gnu -lcblas \
        -L/lib/x86_64-linux-gnu -lf77blas \
        -L/lib/x86_64-linux-gnu -latlas \
        -L/usr/local/lib -lpthread
    libraryFiles.files = libLinAlgd.la libLinAlgd.so libLinAlgd.so.1 libLinAlgd.so.1.0 libLinAlgd.so.1.0.0
    libraryFiles.path = $$[QT_INSTALL_LIBS]
    INSTALLS += libraryFiles

    }

debug {
    LIBS += -L$$PWD/$$OBJECTS_DIR -l*.o 
    LIBS += -L/lib/x86_64-linux-gnu -lcblas \
    -L/lib/x86_64-linux-gnu -lf77blas \
    -L/lib/x86_64-linux-gnu -latlas \
    -L/usr/local/lib -lpthread

    libraryFiles.files = libLinAlgd.la libLinAlgd.so libLinAlgd.so.1 libLinAlgd.so.1.0 libLinAlgd.so.1.0.0
    libraryFiles.path = $$[QT_INSTALL_LIBS]
    INSTALLS += libraryFiles
    }
}
