//
// $Id: SamplesPreviewWidget.h 855 2009-06-09 16:15:50Z alex $
//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2009, Alexander Lehmann <lehmanna@in.tum.de>
//                          Felix Weninger <felix@weninger.de>
//                          Bjoern Schuller <schuller@tum.de>
//
// Institute for Human-Machine Communication
// Technische Universitaet Muenchen (TUM), D-80333 Munich, Germany
//
// openBliSSART is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 2 of the License, or (at your option) any later
// version.
//
// openBliSSART is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// openBliSSART.  If not, see <http://www.gnu.org/licenses/>.
//


#ifndef __SAMPLESPREVIEWWIDGET_H__
#define __SAMPLESPREVIEWWIDGET_H__


#include <QWidget>


// Forward declarations
class QPushButton;
class QScrollBar;


namespace blissart {


// Forward declarations
class PlaybackThread;
namespace audio { class Sound; }
namespace internal { class SamplesPreviewCanvas; }


class SamplesPreviewWidget : public QWidget
{
    Q_OBJECT

public:
    /**
     * Constructs a new SamplesPreviewWidget.
     */
    SamplesPreviewWidget(QWidget *parent = 0);


    /**
     * Constructs a new SamplesPreviewWidget for the given samples.
     * @param  samples                  a pointer to the samples
     * @param  nSamples                 the number of samples
     * @param  sampleFreq               the sample frequency
     * @param  parent                   a pointer to the parent widget
     */
    SamplesPreviewWidget(const double *samples, size_t nSamples,
                         unsigned int sampleFreq, QWidget *parent = 0);



    /**
     * Destructs an instance of SamplesPreviewWidget.
     */
    virtual ~SamplesPreviewWidget();


    /**
     * Sets the samples that should be displayed. Creates a Sound object
     * if possible.
     * @param  samples                  a pointer to the samples
     * @param  nSamples                 the number of samples
     * @param  sampleFreq               the sample frequency
     * @return                          true iff a Sound object has been created
     */
    bool setSamples(const double *samples, size_t nSamples,
                    unsigned int sampleFreq);


protected slots:
    /**
     * Event-handler for the "Play" button.
     */
    void on_pbPlay_clicked();


    /**
     * Event-handler for the "Pause" button.
     */
    void on_pbPause_clicked();


    /**
     Event-handler for the "Rewind" button.
     */
    void on_pbRewind_clicked();


    /**
     * Event-handler for the "+" button.
     */
    void on_pbZoomIn_clicked();


    /**
     * Event-handler for the "-" button.
     */
    void on_pbZoomOut_clicked();


    /**
     * Event-handler for the positional scrollbar.
     */
    void on_sbPosition_valueChanged(int value);


    /**
     * Event-handler for the samples visualization canvas.
     */
    void on_svCanvas_posClicked(float pos);
    
    
    /**
     * Event-handler for the playback thread.
     */
    void on_playbackThread_playbackPosChanged(float pos);


protected:
    /**
     * Handles change-events such as QEvent::EnabledChanged.
     */
    virtual void changeEvent(QEvent *ev);


private:
    /**
     * Sets up the whole user interface, i.e. all child widgets and layouts.
     */
    void setupUi();


    /**
     * Adapts the scrollbar's range to the visualized samples.
     */
    void updateScrollBarRange();


    internal::SamplesPreviewCanvas *_svCanvas;
    QScrollBar                     *_sbPosition;
    QPushButton                    *_pbPlay;
    QPushButton                    *_pbPause;
    QPushButton                    *_pbRewind;
    PlaybackThread                 *_playbackThread;
};


} // namespace blissart


#endif

