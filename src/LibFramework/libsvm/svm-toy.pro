#-------------------------------------------------
#
# Project created by QtCreator 2018-11-05T18:39:30
#
#-------------------------------------------------
TEMPLATE = app

QT       += gui widgets


CONFIG += debug_and_release

CONFIG(debug, debug|release) {
    TARGET = svm-toyd
} else {
    TARGET = svm-toy
}

DEFINES += QT_DEPRECATED_WARNINGS

QMAKE_CXXFLAGS = -std=c++20 -fPIC -DPIC -O3 -MD -MT -MF -Wall -Wextra -fpermissive

SOURCES += \
    svm-toy.cpp \
    svm.cpp

INCLUDEPATH += /usr/local/include/SDL /usr/local/include /usr/include/c++/11 ../include ../../
#INCLUDEPATH += /usr/local/include ../include ../../


HEADERS += \
        svm.h

#     target.path = /usr/local/blissart/lib
#    INSTALLS += target
