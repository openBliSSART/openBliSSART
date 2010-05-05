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


#ifndef __BLISSART_FEATURE_MISC_H__
#define __BLISSART_FEATURE_MISC_H__


#include <common.h>
#include <blissart/linalg/RowVector.h>


namespace blissart {


namespace feature {


/**
 * Calculates the sample mean of the given vector.
 */
double LibFeature_API mean(const linalg::Vector& data);


/**
 * Calculates the sample standard deviation of the given vector.
 */
double LibFeature_API stddev(const linalg::Vector& data);


/**
 * Calculates the sample skewness of the given vector.
 */
double LibFeature_API skewness(const linalg::Vector& data);


/**
 * Calculates the sample kurtosis of the given vector.
 */
double LibFeature_API kurtosis(const linalg::Vector& data);


/**
 * Calculates the correlation coefficient of the given vectors.
 */
double LibFeature_API correlation(const linalg::Vector& v1, 
                                  const linalg::Vector& v2);


/**
 * Calculates the auto-correlation of the given vector at a specified delay.
 */
double LibFeature_API autocorrelation(const linalg::Vector& data,
                                      unsigned int delay);


/**
 * Calculates the periodicity of the given vector, i.e. the maximum value
 * of the autocorrelations with delays corresponding to the given beat
 * frequencies.
 */
double LibFeature_API periodicity(const linalg::Vector& data,
                                  double gainsFrequency,
                                  int bpmMin, int bpmMax, int deltaBpm);


/**
 * Calculates the centroid of the given vector, interpreted as a spectrum 
 * with the given base frequency.
 */
double LibFeature_API centroid(const linalg::Vector& data,
                               double baseFrequency);


/**
 * Calculates the centroid of the given vector, interpreted as a spectrum 
 * with the given frequencies.
 */
double LibFeature_API centroid(const linalg::Vector& data,
                               const linalg::Vector& frequencies);


/**
 * Calculates the roll-off point of the given vector, interpreted as a spectrum
 * with the given base frequency.
 */
double LibFeature_API rolloff(const linalg::Vector& data, double baseFrequency,
                              double amount = 0.95);


/**
 * Calculates the zero-crossing-rate of the given vector and returns it as the
 * number of zero-crossing per millisecond.
 */
double LibFeature_API zeroCrossingRate(const linalg::Vector& data,
                                       double durationMS);


/**
 * Returns a vector where all values are zero except the local maxima of the
 * original vector.
 */
linalg::RowVector LibFeature_API findLocalMaxima(const linalg::Vector& data);


/**
 * Computes the noise-likeness (Uhle et al. 2003). Convolves the local maxima
 * of the given (spectral) vector with a Gaussian impulse with zero mean and 
 * returns the correlation to the original vector.
 * @param   sigma    the desired standard deviation of the Gaussian impulse
 */
double LibFeature_API noiseLikeness(const linalg::Vector& data, 
                                    double sigma = 1.0);


/**
 * Computes the percussiveness (Uhle et al. 2003).
 * Convolves the local maxima of the given (gains) vector with a linear decay 
 * function of the given width and calculates the correlation to the 
 * original vector.
 * @param   length   the length of the decay function (in vector elements)
 */
double LibFeature_API percussiveness(const linalg::Vector& data,
                                     unsigned int length);


/**
 * Computes the spectral dissonance (Uhle et al. 2003).
 */
double LibFeature_API spectralDissonance(const linalg::Vector& data,
                                         double baseFrequency);


/**
 * Computes the spectral flatness measure (ISO-IEC 15938-4).
 * This is the ratio of the geometrical to the arithmetical mean of the 
 * element-wise squared input vector.
 */
double LibFeature_API spectralFlatness(const linalg::Vector& data);


} // namespace feature


} // namespace blissart


#endif // __BLISSART_FEATURE_MISC_H__

