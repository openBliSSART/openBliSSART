//
// $Id: peak.cpp 855 2009-06-09 16:15:50Z alex $
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


#include <blissart/feature/peak.h>
#include <blissart/linalg/Vector.h>

#include <vector>
#include <cmath>


using blissart::linalg::Vector;


namespace blissart {

namespace feature {


double averagePeakLength(const Vector& data, double threshold)
{
    unsigned int peakLength = 0;
    double peakLengthSum = 0.0;
    unsigned int peakCount = 0;
    threshold *= data.maximum();
    for (unsigned int i = 0; i <= data.dim(); ++i) {
       if (i < data.dim() && data(i) >= threshold) {
           ++peakLength;
       }
       else {
           if (peakLength > 0) {
               peakLengthSum += (double) peakLength;
               ++peakCount;
           }
           peakLength = 0;
       }
    }
    return peakLengthSum / (double) peakCount;
}


double peakFluctuation(const Vector& data, double threshold)
{
    unsigned int peakLength = 0;
    double peakLengthSum = 0.0;
    std::vector<int> peakLengths;
    threshold *= data.maximum();
    for (unsigned int i = 0; i <= data.dim(); ++i) {
       if (i < data.dim() && data(i) >= threshold) {
           ++peakLength;
       }
       else {
           if (peakLength > 0) {
               peakLengths.push_back(peakLength);
               peakLengthSum += (double) peakLength;
           }
           peakLength = 0;
       }
    }

    if (peakLengths.size() <= 1)
        return 0.0;

    double avg = peakLengthSum / (double) peakLengths.size();
    double variance = 0.0;
    for (std::vector<int>::const_iterator itr = peakLengths.begin();
        itr != peakLengths.end(); ++itr)
    {
        double dev = (double) *itr - avg;
        variance += dev * dev;
    }

    debug_assert(variance >= 0.0);
    return sqrt(variance / (peakLengths.size() - 1));
}


} // namespace feature

} // namespace blissart

