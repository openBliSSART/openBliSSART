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


#include "SampleSeparator.h"
#include <blissart/audio/AudioData.h>
#include <blissart/audio/WaveEncoder.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/RowVector.h>

#include <iostream>
#include <stdexcept>
#include <memory>


using namespace std;
using namespace blissart;
using namespace blissart::audio;
using namespace blissart::linalg;


SampleSeparator::SampleSeparator(unsigned int nSources,
                                 const vector<string>& sourceFileNames,
                                 bool force, double prec, unsigned int maxIter) :
    AbstractSeparator(nSources, prec, maxIter)
{
    unsigned int lastSampleRate = -1, maxNrOfSamples = -1;
    bool variableNrOfSamples = false;
    auto_ptr<AudioData>* decoders = new auto_ptr<AudioData>[sourceFileNames.size()];

    try {
        cout << "Reading files..." << endl;
        for (unsigned int i = 0; i < sourceFileNames.size(); i++) {
            // First decode the files.
            decoders[i] = auto_ptr<AudioData>(AudioData::fromFile(sourceFileNames.at(i)));
            // Print some information about the file.
            cout << "   file '" << sourceFileNames.at(i) << '\'' << endl;
            cout << "      # samples   = " << decoders[i]->nrOfSamples() << endl;
            cout << "      # channels  = " << decoders[i]->nrOfChannels() << endl;
            cout << "      sample rate = " << decoders[i]->sampleRate() << " Hz" << endl;
            // Assure that all sources have the same sample rate and the
            // same number of sample points.
            // If this is the first file, simply initialize the corresponding variables.
            if (i == 0) {
                lastSampleRate = decoders[i]->sampleRate();
                maxNrOfSamples = decoders[i]->nrOfSamples();
                variableNrOfSamples = false;
            } else if (!force && lastSampleRate != decoders[i]->sampleRate()) {
                // The sample rate of this file differs from the previous ones.
                throw runtime_error("All sources must have the same sample rate!");
            } else if (maxNrOfSamples != decoders[i]->nrOfSamples()) {
                if (force) {
                    variableNrOfSamples = true;
                    // Keep track of the maximum # of samples.
                    if (decoders[i]->nrOfSamples() > maxNrOfSamples)
                        maxNrOfSamples = decoders[i]->nrOfSamples();
                } else {
                    // The number of sample points of this file differs from
                    // the previous ones.
                    throw runtime_error("All sources must have the same number "
                                        "of sample points!");
                }
            }
        }
    } catch (...) {
        delete[] decoders;
        throw;
    }

    // Remember the sample rate.
    _sampleRate = lastSampleRate;

    // Allocate enough memory to hold all available sample points.
    _matrix = new Matrix((unsigned int)sourceFileNames.size(), maxNrOfSamples);
    // Copy the sample points to the newly created matrix and expand if
    // neccessary.
    for (unsigned int i = 0; i < _matrix->rows(); i++) {
        const double* buf = decoders[i]->getChannel(0);
        double expectedValue = 0;
        const double f = 1.0 / decoders[i]->nrOfSamples();
        // Copy the existing sample points and calculate the
        // expected value
        for (unsigned int j = 0; j < decoders[i]->nrOfSamples(); j++) {
            _matrix->setAt(i, j, buf[j]);
            if (variableNrOfSamples)
                expectedValue += f * buf[j];
        }
        // Append the expected value as often as needed to the end of the
        // current dataset to get an even number of samples.
        if (variableNrOfSamples && decoders[i]->nrOfSamples() < maxNrOfSamples) {
            cout << "'Expected value'-expansion for file '"
                 << sourceFileNames.at(i) << '\'' << endl;
            for (unsigned int j = decoders[i]->nrOfSamples(); j < maxNrOfSamples; j++)
                _matrix->setAt(i, j, expectedValue);
        }
    }

    // Clean up.
    delete[] decoders;
}
