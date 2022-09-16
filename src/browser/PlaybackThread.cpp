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


#include "PlaybackThread.h"
#include "SamplesPreviewCanvas.h"
#include <blissart/BasicApplication.h>
#include <Poco/Logger.h>


using namespace blissart::audio;
using namespace blissart::internal;


namespace blissart {


PlaybackThread::PlaybackThread(const double *samples, size_t nSamples,
                               unsigned int sampleFreq, QObject *parent) :
    QThread(parent),
    _logger(BasicApplication::instance().logger())
{
    _sound = new Sound(samples, nSamples, sampleFreq);
    _logger.debug("PlaybackThread initialized.");
};


PlaybackThread::~PlaybackThread()
{
    _mutex.lock();
    if (_sound) {
        _sound->setPlaybackState(Sound::Pause);
        delete _sound;
        _sound = nullptr;
    }
    _mutex.unlock();
    _logger.debug("PlaybackThread terminated.");
}


void PlaybackThread::run()
{
    while (true) {
        _mutex.lock();
        if (!_sound || 
            _sound->playbackPos() >= 1 ||
            _sound->playbackState() != Sound::Play) {
            _mutex.unlock();
            break;
        }
        emit playbackPosChanged(_sound->playbackPos());
        _mutex.unlock();
        // 100ms gives a refresh rate of 10Hz.
        msleep(100);
    }
    // Qt emits the QThread::finished() signal right after run() has terminated.
}


void PlaybackThread::setPlaybackState(Sound::PlaybackState state)
{
    _mutex.lock();
    _sound->setPlaybackState(state);
    _mutex.unlock();
    // Start the thread if the state is Sound::Play and the thread isn't
    // running already.
    if (state == Sound::Play && !isRunning())
        start();
}


void PlaybackThread::setPlaybackPos(float pos)
{
    _mutex.lock();
    _sound->setPlaybackPos(pos);
    emit playbackPosChanged(_sound->playbackPos());
    _mutex.unlock();
}


} // namespace blissart
