TEMPLATE = lib
CONFIG = release

CONFIG += debug_and_release

CONFIG(debug, debug|release) {
    TARGET = Featured
} else {
    TARGET = Feature
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

unix {
    target.path = /usr/local/blissart/lib
    INSTALLS += target
}
