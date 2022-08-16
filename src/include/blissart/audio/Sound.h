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


#ifndef __BLISSART_AUDIO_SOUND_H__
#define __BLISSART_AUDIO_SOUND_H__


#include <common.h>


// Get rid of SDL's "main" macro if it hasn't been defined beforehand.  For a
// more detailed description why this is done see AudioData.h.
#ifdef main
#  include <SDL2/SDL_stdinc.h>
#else
#  include <SDL2/SDL_stdinc.h>
#  undef main
#endif


// Forward declaration
struct SDL_AudioSpec;


namespace blissart {

namespace audio {


/**
 * \addtogroup audio
 * @{
 */

/**
 * Convenience class for the playback of 16-bit mono sounds.
 */
class LibAudio_API Sound
{
public:
    typedef enum { Play, Pause } PlaybackState;


    // SDL supports only a limited range of sample frequencies for playback,
    // hence:
    static const unsigned int MinSampleFreq = 8000;
    static const unsigned int MaxSampleFreq = 44100;
    
    
    /**
     * Constructs a new instance of Sound for the given samples (in floating
     * point format) and sample frequency.
     * Note that the Sound object doesn't take ownership of the given pointer.
     * @param  samples              a pointer to the raw samples
     * @param  len                  length of the the samples buffer
     * @param  sampleFreq           the desired playback frequency
     */  
    Sound(const double *samples, size_t len, unsigned int sampleFreq);
    
    
    /**
     * Destructs an instance of Sound and frees all formerly allocated memory.
     */
    virtual ~Sound();

    
    /**
     * Sets the playback state of the Sound object.
     * @param  state                a PlayState
     */
    void setPlaybackState(PlaybackState state);


    /**
     * Returns the playback state of the Sound object.
     */
    inline PlaybackState playbackState() const { return _state; }

    
    /**
     * Sets the playback position.
     * @param  pos                  the new pos [0,1]
     */
    void setPlaybackPos(double pos);
    
    
    /**
     * Returns the current playback position.
     * @return                      a value within [0,1]
     */
    double playbackPos() const;
    
    
private:
    // Forbid copy constructor and operator=.
    Sound(const Sound &);
    Sound& operator=(const Sound &);
    
    
    /**
     * Callback function for SDL.
     */
    static void pbCallback(void *user, Uint8 *buf, int size);
    
    
    short              *_samples = nullptr;
    const size_t       _len;
    const unsigned int _sampleFreq;
    size_t             _pos;
    PlaybackState      _state;
    SDL_AudioSpec      *_audioSpec;
};


/**
 * @}
 */


} // namespace audio

} // namespace blissart


#endif // __BLISSART_AUDIO_SOUND_H__
