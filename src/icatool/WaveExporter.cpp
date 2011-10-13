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


#include "WaveExporter.h"
#include <blissart/linalg/Matrix.h>
#include <blissart/audio/WaveEncoder.h>

#include <sstream>
#include <iomanip>
#include <cassert>
#include <cmath>


using namespace std;
using namespace blissart;
using namespace blissart::audio;
using namespace blissart::linalg;


WaveExporter::WaveExporter(const string& prefix, unsigned int sampleRate) :
    _prefix(prefix),
    _sampleRate(sampleRate)
{
    assert(_sampleRate > 0);
}


bool WaveExporter::doExport(const Matrix& matrix)
{
    // Calculate the neccessary number of digits
    const unsigned int numDigits = 1 + (unsigned int)log10f((float)matrix.rows());

    double* row = 0;
    try {
        for (unsigned int i = 0; i < matrix.rows(); i++) {
            // Assemble the name of the output file.
            stringstream ss;
            ss << _prefix
               << right << setfill('0') << setw(numDigits)
               << (i+1) << ".wav";
            // Retrieve the audio data.
            row = new double[matrix.cols()];
            for (unsigned int j = 0; j < matrix.cols(); ++j) {
                row[j] = matrix(i, j);
            }
            // Convert to double.
            // Eventually export the data to the audio file.
            WaveEncoder::saveAsWav(row, matrix.cols(), _sampleRate, 1, ss.str());
            delete[] row;
            row = 0;
        }
    } catch (...) {
        if (row)
            delete[] row;
        return false;
    }

    if (row)
        delete[] row;
    return true;
}

