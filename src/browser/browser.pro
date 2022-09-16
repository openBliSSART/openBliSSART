#
# This file is part of openBliSSART.
#
# Copyright (c) 2007-2009, Alexander Lehmann <lehmanna@in.tum.de>
#                          Felix Weninger <felix@weninger.de>
#                          Bjoern Schuller <schuller@tum.de>
#
# Institute for Human-Machine Communication
# Technische Universitaet Muenchen (TUM), D-80333 Munich, Germany
#
# openBliSSART is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 2 of the License, or (at your option) any later
# version.
#
# openBliSSART is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# openBliSSART.  If not, see <http:#www.gnu.org/licenses/>.
#

#CONFIG = debug
CONFIG += resources thread debug
QT += widgets
CONFIG += force_debug_info


SOURCES = main.cpp \
          BrowserController.cpp \
          BrowserMainWindow.cpp \
          ClassificationObjectItem.cpp \
          CreateProcessDialog.cpp \
          DataDescriptorItem.cpp \
          EditWidget.cpp \
          EditWidgetClassificationObject.cpp \
          EditWidgetLabel.cpp \
          EditWidgetResponse.cpp \
          EntityItem.cpp \
          ExportObjectsDlg.cpp \
          FeatureExtractionDialog.cpp \
          FeatureItem.cpp \
          FilesSelectionWidget.cpp \
          LabelItem.cpp \
          LabelSelectionDialog.cpp \
          PlaybackThread.cpp \
          PreferencesDlg.cpp \
          ProcessItem.cpp \
          ResponseItem.cpp \
          ResponseQualityDlg.cpp \
          SamplesPreviewWidget.cpp \
          SamplesPreviewCanvas.cpp \
          ThreadedDialog.cpp \
          TreeWidgetController.cpp

HEADERS = BrowserController.h \
          BrowserMainWindow.h \
          ClassificationObjectItem.h \
          CreateProcessDialog.h \
          DataDescriptorItem.h \
          EditWidget.h \
          EditWidgetClassificationObject.h \
          EditWidgetLabel.h \
          EditWidgetResponse.h \
          EntityItem.h \
          ExportObjectsDlg.h \
          FeatureExtractionDialog.h \
          FeatureItem.h \
          FilesSelectionWidget.h \
          LabelItem.h \
          LabelSelectionDialog.h \
          PlaybackThread.h \
          PreferencesDlg.h \
          ProcessItem.h \
          ResponseItem.h \
          ResponseQualityDlg.h \
          SamplesPreviewWidget.h \
          SamplesPreviewCanvas.h \
          ThreadedDialog.h \
          TreeWidgetController.h

FORMS = BrowserForm.ui \
        CreateProcessDialog.ui \
        EditWidgetClassificationObject.ui \
        EditWidgetLabel.ui \
        EditWidgetResponse.ui \
        FeatureExtractionDialog.ui \
        FilesSelectionWidget.ui \
        LabelSelectionDialog.ui \
        PreferencesDlg.ui \
        ResponseQualityDlg.ui

TARGET = browser

INCLUDEPATH += ../include
#/usr/include/x86_64-linux-gnu/qt5

##CONFIG(debug, debug|release) {
##    IMDIR = ../../build/debug
##    DEFINES += _DEBUG
##    win32:DESTDIR = ../../bin/debug
##}


##CONFIG(release, debug|release) {
##    IMDIR = ../../build/release
##    DEFINES += QT_NO_DEBUG_OUTPUT
##    win32:DESTDIR = ../../bin/release
##    unix {
##        QMAKE_POST_LINK = strip $${TARGET}
##    }
##}

mac {
    INCLUDEPATH += /sw/include
    LIBS += -L/sw/lib
    CONFIG -= app_bundle
    system(test "`uname -p`" == "powerpc") {
        QMAKE_LFLAGS += -bind_at_load
    }
}

unix {
    QMAKE_CXXFLAGS += -std=c++17 -fPIC -DPIC -g -O2 -MD -MT -MF -Wall -Wextra -fpermissive
    QMAKE_CXXFLAGS += '-I/home/gordon/Qt/6.3.0/gcc_64/include -DDEBUG -DQT_DEBUG -DPIC $${POCOCPPFLAGS}'
    QMAKE_CXXFLAGS += '-DDEBUG -DQT_DEBUG -DPIC $${CPPFLAGS}'
    QMAKE_LFLAGS += '$${POCOLDFLAGS}'
    QMAKE_LFLAGS += '$${LDFLAGS}'
#    LIBS += -L../LibLinAlg/.libs -lLinAlg \
#            -L../LibAudio/.libs -lAudio \
#            -L../LibNMF/.libs -lNMF \
#            -L../LibFramework -lFramework \
#            -lPocoFoundation -lPocoDataSQLite -lPocoData -lPocoUtil -lPocoXML
##    LIBS += -L/usr/local/blissart/lib -lLinAlg \
##            -L/usr/local/blissart/lib -lAudio \
##            -L/usr/local/blissart/lib -lNMF \
##            -L/usr/local/lib -lSDL2main \
##            -L/usr/local/lib -lSDL2 \
##            -L/usr/local/lib -lSDL2_sound \
##            -L../LibFramwork -lFramework \
##            -lPocoFoundation -lPocoDataSQLite -lPocoData -lPocoUtil -lPocoXML

    LIBS += -L/usr/local/blissart/lib -lLinAlgd \
            -L/usr/local/blissart/lib -lAudio \
            -L/usr/local/blissart/lib -lNMFd \
            -L/usr/local/lib -lSDL2main \
            -L/usr/local/lib -lSDL2 \
            -L/usr/local/lib -lSDL2_sound \
            -L../LibFramwork -lFrameworkd \
            -lPocoFoundationd -lPocoDataSQLited -lPocoDatad -lPocoUtild -lPocoXMLd
}

##win32 {
##    # Add CONFIG += console for messages on std(out|err).
##    CONFIG += windows embed_manifest_exe
##    QMAKE_CXXFLAGS += /Fd$(IntDir)\$(ProjectName).pdb
##}

UI_DIR = $$IMDIR
MOC_DIR = $$IMDIR
OBJECTS_DIR = $$IMDIR
RCC_DIR = $$IMDIR
