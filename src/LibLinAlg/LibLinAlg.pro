TEMPLATE = lib
CONFIG = release
CONFIG += debug_and_release

CONFIG(debug, debug|release) {
    TARGET = LinAlgd
} else {
    TARGET = LinAlg
}

QMAKE_CXXFLAGS = -std=c++11 -fPIC -DPIC -O3 -MD -MT -MF -Wall -Wextra -fpermissive

INCLUDEPATH = ../include /usr/local/cuda/include

HEADERS += \
        blissart/libalg/vector.h \
        blissart/linalg/rowvector.h \
        blissart/linalg/colvector.h \
        blissart/linalg/matrix.h \
        blissart/linalg/generators/

SOURCES += \
        Vector.cpp \
        RowVector.cpp \
        ColVector.cpp \
        GPUMatrix.cpp \
        GPUUtil.cpp \
        Matrix.cpp

unix {
    target.path = /usr/local/blissart/lib
    INSTALLS += target
}
