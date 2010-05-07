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


#ifndef __BLISSART_AUDIO_MELFILTER_H__
#define __BLISSART_AUDIO_MELFILTER_H__


#include <common.h>
#include <cmath>


namespace blissart {


// Fwd declaration
namespace linalg { class Matrix; }


namespace audio {


/**
 * \addtogroup audio
 * @{
 */

/**
 * A filter bank that transforms a discrete Fourier spectrum onto the Mel 
 * scale by triangular filters.
 */
class LibAudio_API MelFilter
{
public:
    /**
     * Constructs a Mel filter bank with 26 filters, assuming a sample rate
     * of 44.1 kHz, ranging from 0 to 22.05 kHz (Nyquist frequency).
     */
    MelFilter();


    /**
     * Constructs a Mel filter bank with the specified number of filters,
     * sample rate and cut-off frequencies.
     */
    MelFilter(unsigned int nBands, unsigned int sampleRate, 
              double lowFreq, double highFreq);

    /**
     * Transforms the given spectrogram on the Mel scale.
     */
    linalg::Matrix* melSpectrum(const linalg::Matrix& spectrogram);


    /**
     * Resynthesizes (approximatively) the Fourier spectrum from the Mel 
     * spectrum.
     */
    void synth(const linalg::Matrix& melSpectrogram, linalg::Matrix& spectrogram);


    /**
     * Sets a scale factor that can be used e.g. for compatibility with HTK.
     */
    void setScaleFactor(double factor);


private:
    /**
     * Convert a frequency given in Hz to a frequency in Mel.
     * @param   hertzFrequency  the frequency in Hz
     * @return                  the frequency in Mel
     */
    static double hertzToMel(double hertzFrequency);


    /**
     * Convert a frequency given in Mel to a frequency in Hz.
     * @param   melFrequency    the frequency in Mel
     * @return                  the frequency in Hz
     */
    static double melToHertz(double melFrequency);


    // Computes the mapping of frequency bins to Mel filters, as well as the
    // Mel filter coefficients.
    void computeFilters(unsigned int nBins);

    int          _nBands;
    unsigned int _sampleRate;
    double       _lowFreq, _highFreq;
    double       _scaleFactor;

    unsigned int _nBins;
    double*      _filterCoeffs;
    int*         _filterIndex;
    unsigned int _lowestIndex, _highestIndex;
};


/**
 * @}
 */


inline void MelFilter::setScaleFactor(double factor)
{
    _scaleFactor = factor;
}


inline double MelFilter::hertzToMel(double hertzFrequency)
{
    return 1127.0 * log(1.0 + hertzFrequency / 700.0);
}


inline double MelFilter::melToHertz(double melFrequency)
{
    return 700.0 * (exp(melFrequency / 1127.0) - 1.0);
}


} // namespace audio


} // namespace blissart


#endif // __BLISSART_AUDIO_MELFILTER_H__
