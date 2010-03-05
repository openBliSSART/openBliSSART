//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2010, Alexander Lehmann <lehmanna@in.tum.de>
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


#include "SamplesPreviewWidget.h"
#include "SamplesPreviewCanvas.h"
#include "PlaybackThread.h"

#include <QApplication>
#include <QPushButton>
#include <QScrollBar>
#include <QHBoxLayout>
#include <QVBoxLayout>

#include <cmath>


// The zoom factor to be used for zoom-in and -out.
#define ZOOM_FACTOR 1.5f


using namespace blissart::audio;
using namespace std;


namespace blissart {



SamplesPreviewWidget::SamplesPreviewWidget(QWidget *parent) :
    QWidget(parent),
    _playbackThread(NULL)
{
    setupUi();
}


SamplesPreviewWidget::SamplesPreviewWidget(const double *samples,
                                           size_t nSamples,
                                           unsigned int sampleFreq,
                                           QWidget *parent) :
    QWidget(parent),
    _playbackThread(NULL)
{
    setupUi();
    setSamples(samples, nSamples, sampleFreq);
}


SamplesPreviewWidget::~SamplesPreviewWidget()
{
    if (_playbackThread) {
        _playbackThread->setPlaybackState(Sound::Pause);
        while (_playbackThread->isRunning()) {
            // Wait for the thread to terminate.
        }
        delete _playbackThread;
        _playbackThread = NULL;
    }
}


bool SamplesPreviewWidget::setSamples(const double *samples,
                                      size_t nSamples,
                                      unsigned int sampleFreq)
{
    // Delegate the samples to the canvas.
    _svCanvas->setSamples(samples, nSamples);

    // Delete a possible former PlaybackThread object.
    if (_playbackThread) {
        _playbackThread->setPlaybackState(Sound::Pause);
        delete _playbackThread;
        _playbackThread = NULL;
    }
    // Create a new PlaybackThread object if possible.
    if (sampleFreq >= Sound::MinSampleFreq && sampleFreq <= Sound::MaxSampleFreq) {
        _playbackThread = new PlaybackThread(samples, nSamples, sampleFreq, this);
        connect(_playbackThread, SIGNAL(finished()), SLOT(on_pbPause_clicked()));
        connect(_playbackThread, SIGNAL(playbackPosChanged(float)),
                SLOT(on_playbackThread_playbackPosChanged(float)));
    }

    // Enable the buttons only if a playback thread exists
    _pbPlay->setEnabled(_playbackThread != NULL);
    _pbPause->setEnabled(_playbackThread != NULL);
    _pbRewind->setEnabled(_playbackThread != NULL);

    return (_playbackThread != NULL);
}


void SamplesPreviewWidget::setupUi()
{
    QHBoxLayout *upperLayout = new QHBoxLayout;
    QHBoxLayout *lowerLayout = new QHBoxLayout;
    QVBoxLayout *overallLayout = new QVBoxLayout;

    // The upper layout.
    _sbPosition = new QScrollBar(Qt::Horizontal);
    _sbPosition->setRange(0, 0);
    _sbPosition->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    upperLayout->addWidget(_sbPosition);

    QPushButton *pbZoomIn = new QPushButton("+");
    upperLayout->addWidget(pbZoomIn);

    QPushButton *pbZoomOut = new QPushButton("-");
    upperLayout->addWidget(pbZoomOut);

    // The lower layout.
    _pbPlay = new QPushButton(tr("Play"));
    lowerLayout->addWidget(_pbPlay);

    _pbPause = new QPushButton(tr("Pause"));
    lowerLayout->addWidget(_pbPause);

    _pbRewind = new QPushButton(tr("Rewind"));
    lowerLayout->addWidget(_pbRewind);

    // The overall layout
    _svCanvas = new internal::SamplesPreviewCanvas;
    _svCanvas->setMarkerVisible(true);
    overallLayout->addWidget(_svCanvas);
    overallLayout->addLayout(upperLayout);
    overallLayout->addLayout(lowerLayout);
    setLayout(overallLayout);

    // Connect the neccessary signals.
    connect(_sbPosition,
            SIGNAL(valueChanged(int)),
            SLOT(on_sbPosition_valueChanged(int)));
    connect(pbZoomIn, SIGNAL(clicked()), SLOT(on_pbZoomIn_clicked()));
    connect(pbZoomOut, SIGNAL(clicked()), SLOT(on_pbZoomOut_clicked()));
    connect(_pbPlay, SIGNAL(clicked()), SLOT(on_pbPlay_clicked()));
    connect(_pbPause, SIGNAL(clicked()), SLOT(on_pbPause_clicked()));
    connect(_pbRewind, SIGNAL(clicked()), SLOT(on_pbRewind_clicked()));
    connect(_svCanvas,
            SIGNAL(posClicked(float)),
            SLOT(on_svCanvas_posClicked(float)));
}


void SamplesPreviewWidget::on_pbPlay_clicked()
{
    debug_assert(_playbackThread);
    _pbPlay->setEnabled(false);
    _pbPause->setEnabled(true);
    _playbackThread->setPlaybackState(Sound::Play);
}


void SamplesPreviewWidget::on_pbPause_clicked()
{
    debug_assert(_playbackThread);
    _playbackThread->setPlaybackState(Sound::Pause);
    if (_playbackThread->playbackPos() >= 1.0)
        on_pbRewind_clicked();
    _pbPlay->setEnabled(true);
    _pbPause->setEnabled(false);
}


void SamplesPreviewWidget::on_pbRewind_clicked()
{
    debug_assert(_playbackThread);
    _playbackThread->setPlaybackPos(0.0);
}


void SamplesPreviewWidget::on_pbZoomIn_clicked()
{
    // Compute and set the new minimum and maximum.
    float width = _svCanvas->max() - _svCanvas->min();
    float middle = _svCanvas->min() + width / 2.0;
    float newMin = middle - width / (2.0 * ZOOM_FACTOR);
    float newMax = middle + width / (2.0 * ZOOM_FACTOR);
    if (newMin >= newMax)
        return;
    _svCanvas->setMinMax(newMin, newMax);
    // Update the scrollbar.
    updateScrollBarRange();
    // Finally update the widget.
    _svCanvas->update();
}


void SamplesPreviewWidget::on_pbZoomOut_clicked()
{
    // Compute and set the new minimum and maximum.
    float width = _svCanvas->max() - _svCanvas->min();
    float newMin = std::max<float>(0.0, _svCanvas->min() - width / 4.0 * ZOOM_FACTOR);
    float newMax = std::min<float>(1.0, _svCanvas->max() + width / 4.0 * ZOOM_FACTOR);
    _svCanvas->setMinMax(newMin, newMax);
    // Update the scrollbar.
    updateScrollBarRange();
    // Finally update the widget.
    _svCanvas->update();
}


void SamplesPreviewWidget::on_sbPosition_valueChanged(int value)
{
    float window = _svCanvas->max() - _svCanvas->min();
    float newMin = value * window;
    float newMax = newMin + window;
    if (newMax > 1.0) {
        newMax = 1.0;
        newMin = newMax - window;
    }
    _svCanvas->setMinMax(newMin, newMax);
}


void SamplesPreviewWidget::on_svCanvas_posClicked(float pos)
{
    if (_playbackThread)
        _playbackThread->setPlaybackPos(pos);
    else
        _svCanvas->setMarkerPos(pos);
}


void SamplesPreviewWidget::on_playbackThread_playbackPosChanged(float pos)
{
    _svCanvas->setMarkerPos(pos);
}


void SamplesPreviewWidget::changeEvent(QEvent *ev)
{
    if (ev->type() == QEvent::EnabledChange) {
        // Pause playback.
        if (_playbackThread)
            _playbackThread->setPlaybackState(Sound::Pause);
    }
    // Call the base-class implementation.
    QWidget::changeEvent(ev);
}


void SamplesPreviewWidget::updateScrollBarRange()
{
    int steps = (int)ceilf(1.0 / (_svCanvas->max() - _svCanvas->min())) - 1;
    _sbPosition->blockSignals(true);
    _sbPosition->setValue(min<int>(steps, _sbPosition->value() + 1));
    _sbPosition->setRange(0, steps);
    _sbPosition->blockSignals(false);
}


} // namespace blissart

