//
// $Id: WaveEncoder.h 855 2009-06-09 16:15:50Z alex $
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


#ifndef __BLISSART_AUDIO_WAVEENCODER_H__
#define __BLISSART_AUDIO_WAVEENCODER_H__


#include <common.h>
#include <string>


namespace blissart {


// Forward declaration
namespace linalg { class Matrix; }


namespace audio {


// Forward declaration
class AudioData;


/**
 * Utility class for saving of PCM wave files.
 */
class LibAudio_API WaveEncoder {
public:
    /**
     * Save the given data as 16-bit WAVE file.
     * @param   data                a pointer to a double array with values
     *                              within [0,1]
     * @param   nrOfSamples         the total number of samples in the array,
     *                              i.e. the array's size
     * @param   sampleRate          the sample rate, e.g. 44100
     * @param   nrOfChannels        the number of channels, e.g. 2 for stereo
     * @param   fileName            the name of the output file
     * @return                      true iff no error occured
     */
    static bool saveAsWav(const double* data, unsigned int nrOfSamples,
                          unsigned int sampleRate, unsigned int nrOfChannels,
                          const std::string& fileName);


    /**
     * Save the given data as 16-bit WAVE file.
     * @param   sampleMatrix        a sample matrix with values within [0,1]
     * @param   sampleRate          the sample rate, e.g. 44100
     * @param   fileName            the name of the output file
     * @return                      true iff no error occured
     */
    static bool saveAsWav(const linalg::Matrix& sampleMatrix, 
                          unsigned int sampleRate,
                          const std::string& fileName);


    /**
     * Save the given AudioData object as 16-bit WAVE file.
     * @param   audioData           an AudioData object
     * @param   fileName            the name of the output file
     * @return                      true iff no error occured
     */
    static bool saveAsWav(const AudioData& audioData, const std::string& fileName);


private:
    WaveEncoder();
    WaveEncoder(const WaveEncoder&);
};


} // namespace audio

} // namespace blissart


#endif // __BLISSART_AUDIO_WAVEENCODER_H__
