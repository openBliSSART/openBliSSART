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


#ifndef __BLISSART_FEATURE_PEAK_H__
#define __BLISSART_FEATURE_PEAK_H__


#include <common.h>


namespace blissart {


// Forward declaration
namespace linalg { class Vector; }


namespace feature {


/**
 * \addtogroup feature
 * @{
 */

/**
 * Computes the average length of peaks in the given Vector.
 * @param  data       the input Vector for which peaks should be measured
 * @param  threshold  a peak is an area which is above threshold * max(data).
 */
double LibFeature_API averagePeakLength(const linalg::Vector& data,
                                        double threshold = 0.2);

/**
 * Computes the standard deviation of peak lengths in the given Vector.
 * @param  data       the input Vector for which peaks should be measured
 * @param  threshold  a peak is an area which is above threshold * max(data).
 */
double LibFeature_API peakFluctuation(const linalg::Vector& data,
                                      double threshold = 0.2);


/**
 * @}
 */


} // namespace feature


} // namespace blissart


#endif // __BLISSART_FEATURE_PEAK_H__

