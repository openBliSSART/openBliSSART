TEMPLATE = lib
CONFIG = debug
CONFIG += debug_and_release

DEFINES -= HAVE_CUDA

CONFIG(debug, debug|release) {
    TARGET = ICAd
    lib1.files = libICAd.*
} else {
    TARGET = ICA
    lib1.files = libICA.*
}

QMAKE_CXXFLAGS = -std=c++11 -fPIC -DPIC -O3 -MD -MT -MF -Wall -Wextra -fpermissive

INCLUDEPATH = ../include

HEADERS += \
        blissart/ica/FastICA.h \
        blissart/ica/PCA.h

SOURCES += \
	FastICA.cpp PCA.cpp
	
lib1.path = /usr/local/blissart/lib
#ib1.files = libICA*.so*

#inc2.path = /usr/local/include/dynamica_bits
#inc2.files +=

INSTALLS += lib1

unix {
#    target.path = /usr/local/blissart/lib
#    INSTALLS += target
}
