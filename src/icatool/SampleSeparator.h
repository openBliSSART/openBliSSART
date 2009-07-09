//
// $Id: SampleSeparator.h 855 2009-06-09 16:15:50Z alex $
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

#ifndef __SAMPLESEPARATOR_H__
#define __SAMPLESEPARATOR_H__


#include "AbstractSeparator.h"
#include <vector>
#include <string>


/**
 * Performs separation of sound samples
 */
class SampleSeparator : public AbstractSeparator
{
public:
    /**
     * Constructs a SampleSeparator for the given sound files.
     * Unless force is specified all input files must have the same
     * sample rate and # of sample points.
     * Note that Only the first channel of each file is taken into account.
     * @param   nSources            the # number of sources to be separated
     * @param   sourceFileNames     a vector of names of input-files
     * @param   force               allow input-files with a variable number
     *                              of sample points. If such files are found
     *                              then all smaller datasets will be expanded
     *                              to the size of the biggest dataset by 
     *                              respectively appending the neccessary number
     *                              of the 'expected value' to the end of
     *                              the datasets.
     *                              This also suppresses the need for an
     *                              equal sample rate for all files.
     * @param   prec                the desired precision
     * @param   maxIter             the maximum # of iterations during FastICA
     * @throw   str_exception       only in case of errors
     */
    SampleSeparator(unsigned int nSources,
                    const std::vector<std::string>& sourceFileNames,
                    bool force, double prec, unsigned int maxIter);


    /**
     * Get the sample rate of the underlying data.
     * @return              the sample rate
     */
    inline unsigned int sampleRate() const { return _sampleRate; }


private:
    unsigned int _sampleRate;
};


#endif
