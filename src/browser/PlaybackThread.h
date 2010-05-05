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


#ifndef __PLAYBACKTHREAD_H__
#define __PLAYBACKTHREAD_H__


#include <QThread>
#include <QMutex>
#include <blissart/audio/Sound.h>


// Forward declaration
namespace Poco { class Logger; }


namespace blissart {


class PlaybackThread : public QThread
{
    Q_OBJECT

public:
    /**
     * Constructs an instance of PlaybackThread for the given samples and
     * sample frequency.
     * @param  samples          a pointer to the samples
     * @param  nSamples         the total number of samples
     * @param  sampleFreq       the sample frequency
     * @param  parent           a pointer to the parent object
     */
    PlaybackThread(const double *samples, size_t nSamples,
                   unsigned int sampleFreq, QObject *parent = 0);
    
    
    /**
     * Destructs an instance of PlaybackThread, thus stopping the playback
     * and freeing all allocated memory.
     */
    virtual ~PlaybackThread();

    
    /**
     * Sets the playback state of this thread to the given state.
     * @param  state            the new playback state
     */
    void setPlaybackState(audio::Sound::PlaybackState state);
    
    
    /**
     * Returns the playback state of this thread.
     */
    inline audio::Sound::PlaybackState playbackState() const;
    
    
    /**
     * Sets the playback position to the given value.
     * @param  pos              a real value within [0,1]
     */
    void setPlaybackPos(float pos);
    
    
    /**
     * Returns the playback position of this thread.
     */
    inline float playbackPos() const;
    
    
signals:
    /**
     * This signal is emitted every 100ms at best.
     */
    void playbackPosChanged(float pos);
    

protected:
    /**
     * The thread's main loop.
     */
    virtual void run();
    
    
private:
    // Forbid copy constructor and operator=.
    PlaybackThread(const PlaybackThread &);
    PlaybackThread& operator=(const PlaybackThread &);
    
    
    audio::Sound                   *_sound;
    QMutex                         _mutex;
    Poco::Logger                   &_logger;
};


// Inlines
audio::Sound::PlaybackState PlaybackThread::playbackState() const
{
    return _sound->playbackState();
}


float PlaybackThread::playbackPos() const
{
    return _sound->playbackPos();
}


} // namespace blissart


#endif
