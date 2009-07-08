//
// $Id: SpectralAnalysisTest.h 855 2009-06-09 16:15:50Z alex $
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


#ifndef __SPECTRAL_TRAFO_TEST_H__
#define __SPECTRAL_TRAFO_TEST_H__


#include "Testable.h"


namespace Testing {


/**
 * Test spectral analysis functions by windowing a sine function,
 * doing a Fourier transformation and backtransformation, then overlapping
 * windows in the time domain. The results as well as the spectra are
 * written to standard output.
 */
class SpectralAnalysisTest : public Testable
{
public:
    virtual bool performTest();
    inline const char *name() {
        return "Spectral analysis test";
    }

private:
    bool performTest(int nSamples, int windowSize);
};


} // namespace Testing


#endif
