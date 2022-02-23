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


#ifndef __BLISSART_AUDIO_AUDIO_H__
#define __BLISSART_AUDIO_AUDIO_H__


#include <stdexcept>
#include <common.h>


namespace blissart {


/**
 * Classes related to audio de-/encoding, preprocessing in the time domain,
 * and transformation to the spectral domain.
 */
namespace audio {


/**
 * \defgroup audio Audio functionality (LibAudio)
 * \addtogroup audio
 * @{
 */

/**
 * Base class for exceptions thrown by LibAudio.
 */
class AudioException : public std::runtime_error
{
public:
    AudioException(const std::string &what) :
        std::runtime_error(what)
    {}
};


/**
 * An exception that is thrown when LibAudio functions are used before
 * initialization of the library.
 */
class AudioNotInitializedException : public AudioException
{
public:
    AudioNotInitializedException() :
        AudioException("LibAudio has not been initialized.")
    {}
};


/**
 * Initializes LibAudio in terms of initializing the SDL library.
 * @return              true iff the whole initialization went fine
 */
bool LibAudio_API initialize();


/**
 * Shuts down LibAudio in terms of shutting down the SDL library.
 */
void LibAudio_API shutdown();


/**
 * Returns whether LibAudio has been initialized.
 */
bool LibAudio_API isInitialized();


/**
 * Convenience macro that is mainly used by LibAudio's own functions.
 */
#define ASSERT_AUDIO_INITIALIZED \
    { \
        if (!blissart::audio::isInitialized()) \
            throw AudioNotInitializedException(); \
    }


/**
 * @}
 */


} // namespace audio

} // namespace blissart



#endif // __BLISSART_AUDIO_AUDIO_H__

