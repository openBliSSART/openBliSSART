TEMPLATE = lib
CONFIG = debug
#CONFIG += debug_and_release


CONFIG(debug, debug|release) {
    TARGET = NMFd
} else {
    TARGET = NMF
}

QMAKE_CXXFLAGS = -std=c++11 -fPIC -DPIC -O3 -MD -MT -MF -Wall -Wextra -fpermissive

INCLUDEPATH = ../include

HEADERS += \
        blissart/nmf/Deconvolver.h \
        blissart/nmf/DeconvolverKernels.h \
        blissart/nmf/randomGenerator.h 

SOURCES += \
        Deconvolver.cpp \
        randomGenerator.cpp

unix {
    target.path = /usr/local/blissart/lib
    INSTALLS += target
}
