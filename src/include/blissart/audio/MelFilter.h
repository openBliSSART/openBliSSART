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


class LibAudio_API MelFilter
{
public:
    /**
     * TODO: Document me!
     */
    MelFilter();


    /**
     * TODO: Document me!
     */
    MelFilter(unsigned int nBands, unsigned int sampleRate, 
              double lowFreq, double highFreq);

    /**
     * TODO: Document me!
     */
    linalg::Matrix* melSpectrum(const linalg::Matrix& spectrogram);


    /**
     * TODO: Document me!
     */
    void synth(const linalg::Matrix& melSpectrogram, linalg::Matrix& spectrogram);


    /**
     * TODO: Document me!
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
};


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
