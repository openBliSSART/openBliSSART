TEMPLATE = lib
CONFIG = debug

CONFIG += debug_and_release

CONFIG(debug, debug|release) {
    TARGET = Featured
    lib1.files = libFeatured.*
} else {
    TARGET = Feature
    lib1.files = libFeature.*
}

QMAKE_CXXFLAGS = -std=c++11 -fPIC -DPIC -O3 -MD -MT -MF -Wall -Wextra -fpermissive

INCLUDEPATH = ../include

HEADERS += \
        blissart/feature/mfcc.h \
        blissart/feature/misc.h \
        blissart/feature/peak.h

SOURCES += \
        mfcc.cpp \
        misc.cpp \
        peak.cpp

lib1.path = /usr/local/blissart/lib
INSTALLS += lib1

unix {
#    target.path = /usr/local/blissart/lib
#    INSTALLS += target
}
