cmake_minimum_required(VERSION 3.20)

project(browser) 

set(CMAKE_CXX_STANDARD 20) # This is equal to QMAKE_CXX_FLAGS += -std=c++0x

SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++ -Wuninitialized -Wall -Wextra -Wpedantic")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_SOURCE_DIR /home/gordon/openBliSSART/src/browser)
# Add extension lib for libraries and bin for app executables
set(CMAKE_BINARY_DIR /home/gordon/openBliSSART)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
include_directories(BEFORE ../include ../include/blissart)

# Instruct CMake to run moc uic qrc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
option( Qt5_ON "Enable Qt5_ON" OFF )
option( Qt6_ON "Enable Qt6_ON" OFF )
if (Qt5_ON)
    #Option to Enable Qt6_ON
    #option( Qt6_ON "Enable Qt6_ON" OFF )
    find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
    set(Qt_LINK "Qt5:Core Qt5::Gui Qt5::Widgets")
    option( Qt6_ON "Enable Qt6_ON" OFF )
else ()
    set(CMAKE_PREFIX_PATH "~/Qt/6.3.0/gcc_64")
    find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets) # Equivalent of QT += core
    set(Qt_LINK "Qt6::Core Qt6::Gui Qt6::Widgets")
    #set(CMAKE_PREFIX_PATH "~/Qt/6.3.0/gcc_64")
    option( Qt5_ON "Enable Qt5_ON" OFF )
endif ()

#set(CMAKE_PREFIX_PATH "~/Qt/6.3.0/gcc_64")
# This will find the Qt6 files.
#find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets) # Equivalent of QT += core
#Qt6::Core Qt6::Gui Qt6::Widgets
#find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)
#qt_standard_project_setup()
#find_package(Qt6Gui REQUIRED)
#find_package(Qt6Widgets REQUIRED) # Equivalent of QT += widgets

set(SOURCES
main.cpp 
BrowserController.cpp 
BrowserMainWindow.cpp 
ClassificationObjectItem.cpp 
CreateProcessDialog.cpp 
DataDescriptorItem.cpp 
EditWidget.cpp 
EditWidgetClassificationObject.cpp 
EditWidgetLabel.cpp 
EditWidgetResponse.cpp 
EntityItem.cpp 
ExportObjectsDlg.cpp 
FeatureExtractionDialog.cpp 
FeatureItem.cpp 
FilesSelectionWidget.cpp 
LabelItem.cpp 
LabelSelectionDialog.cpp 
PlaybackThread.cpp 
PreferencesDlg.cpp 
ProcessItem.cpp 
ResponseItem.cpp 
ResponseQualityDlg.cpp 
SamplesPreviewWidget.cpp 
SamplesPreviewCanvas.cpp 
ThreadedDialog.cpp 
TreeWidgetController.cpp
) 
#set(HEADERS mainwindow.h testclass.h testitemlistmodel.h)
#set(UI mainwindow.ui)

# This will create you executable
#add_executable(model_view_playground ${SOURCES} ${HEADERS} ${UI})
# This will link necessary Qt5 libraries to your project
#target_link_libraries(model_view_playground Qt5::Widgets)
#cmake_minimum_required(VERSION 3.9)
#find_package(Qt6.3 CONFIG REQUIRED COMPONENTS Core Gui Widgets)

set (HEADERS
BrowserController.h
BrowserMainWindow.h 
ClassificationObjectItem.h 
CreateProcessDialog.h 
DataDescriptorItem.h 
EditWidget.h 
EditWidgetClassificationObject.h 
EditWidgetLabel.h 
EditWidgetResponse.h 
EntityItem.h 
ExportObjectsDlg.h 
FeatureExtractionDialog.h 
FeatureItem.h 
FilesSelectionWidget.h 
LabelItem.h 
LabelSelectionDialog.h 
PlaybackThread.h 
PreferencesDlg.h 
ProcessItem.h 
ResponseItem.h 
ResponseQualityDlg.h 
SamplesPreviewWidget.h 
SamplesPreviewCanvas.h 
ThreadedDialog.h 
TreeWidgetController.h
)

set (UIS 
BrowserForm.ui
EditWidgetClassificationObject.ui
EditWidgetLabel.ui
EditWidgetResponse.ui
FilesSelectionWidget.ui
LabelSelectionDialog.ui
PreferencesDlg.ui
ResponseQualityDlg.ui
)

add_executable(browser
${SOURCES}
${HEADERS}
${UIS}
)

include_directories(
    /usr/local/include/SDL2
    /usr/local/include
    ${CMAKE_SOURCE_DIR}
    ../benchmark
    ../browser
    ../icatool
    ../include
    ../include/blissart
    ../include/blissart/audio
    ../include/blissart/feature
    ../include/blissart/ica
    ../include/blissart/linalg
    ../include/blissart/nmf
    ../include/blissart/transforms
    ../include/libsvm
    ../LibAudio
    ../LibICA
    ../LibLinAlg
    ../LibNMF
    ../LibFramework
    ../LibFeature
    ../browser
    /usr/include
)

set(CMAKE_DEBUG_POSTFIX d)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
#target_link_libraries(my_app PRIVATE Threads::Threads)

set_target_properties(browser
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

target_link_libraries(browser 
 PRIVATE Threads::Threads 
 Qt::Core Qt::Gui Qt::Widgets
 ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libLinAlg${CMAKE_DEBUG_POSTFIX}.so 
 ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libAudio${CMAKE_DEBUG_POSTFIX}.so 
 ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libNMF${CMAKE_DEBUG_POSTFIX}.so 
 #${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFeature${CMAKE_DEBUG_POSTFIX}.so 
 /usr/local/lib/libSDL2main.a 
 /usr/local/lib/libSDL2.so 
 /usr/local/lib/libSDL2_sound.so 
 ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libFramework${CMAKE_DEBUG_POSTFIX}.so 
 /usr/local/lib/libPocoFoundation${CMAKE_DEBUG_POSTFIX}.so 
 /usr/local/lib/libPocoDataSQLite${CMAKE_DEBUG_POSTFIX}.so 
 /usr/local/lib/libPocoData${CMAKE_DEBUG_POSTFIX}.so 
 /usr/local/lib/libPocoUtil${CMAKE_DEBUG_POSTFIX}.so 
 /usr/local/lib/libPocoXML${CMAKE_DEBUG_POSTFIX}.so 
)

install(TARGETS browser DESTINATION /usr/local/blissart/bin)
