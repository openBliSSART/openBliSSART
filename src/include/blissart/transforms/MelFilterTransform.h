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


#ifndef __BLISSART_MEL_FILTER_TRANSFORM_H__
#define __BLISSART_MEL_FILTER_TRANSFORM_H__


#include <common.h>
#include <blissart/MatrixTransform.h>


namespace blissart {


namespace transforms {


/**
 * \addtogroup framework
 * @{
 */

/**
 * Calculation of the power spectrum. Implements MatrixTransform interface.
 */
class LibFramework_API MelFilterTransform : public MatrixTransform
{
public:
    /**
     * Default constructor. Constructs a MelFilterTransform with the parameters
     * in the applications' configuration and the given sample rate.
     */
    MelFilterTransform(double sampleRate);


    /**
     * Implementation of MatrixTransform interface.
     */
    virtual const char * name() const;


    /**
     * Implementation of MatrixTransform interface.
     * Computes the Mel spectrum using the given number of bands, sample rate
     * and cut-off frequencies.
     */
    virtual linalg::Matrix * transform(linalg::Matrix* m) const;


    /** 
     * Implementation of MatrixTransform interface.
     * Reverts the Mel filter transformation.
     */
    virtual linalg::Matrix * inverseTransform(linalg::Matrix* m) const;


    /** 
     * Sets the number of bands.
     */
    void setBands(int nBands);


    /**
     * Sets the sample rate, which is used for calculation of the base 
     * frequency and thus the frequency boundaries.
     */
    void setSampleRate(double sampleRate);


    /**
     * Sets the lower cut-off frequency.
     */
    void setLowFreq(double lowFreq);


    /**
     * Sets the upper cut-off frequency.
     */
    void setHighFreq(double highFreq);


    /**
     * Sets the number of bins for the inverse transformation.
     */
    void setBins(unsigned int bins);


private:
    // Forbid default constructor.
    MelFilterTransform();

    int _nBands;
    double _sampleRate;
    double _lowFreq, _highFreq;
    unsigned int _nBins;
};


/**
 * @}
 */


inline void MelFilterTransform::setBands(int nBands)
{
    _nBands = nBands;
}


inline void MelFilterTransform::setSampleRate(double sampleRate)
{
    _sampleRate = sampleRate;
}


inline void MelFilterTransform::setLowFreq(double lowFreq)
{
    _lowFreq = lowFreq;
}


inline void MelFilterTransform::setHighFreq(double highFreq)
{
    _highFreq = highFreq;
}


inline void MelFilterTransform::setBins(unsigned int nBins)
{
    _nBins = nBins;
}


} // namespace transforms


} // namespace blissart


#endif // __BLISSART_MEL_FILTER_TRANSFORM_H__
