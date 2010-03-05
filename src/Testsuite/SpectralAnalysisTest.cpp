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


#include "SpectralAnalysisTest.h"
#include <blissart/audio/AudioData.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/WindowFunctions.h>

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <memory>


using namespace std;
using namespace blissart;
using namespace blissart::audio;
using namespace blissart::linalg;


namespace Testing {


bool SpectralAnalysisTest::performTest(int nSamples, int windowSize)
{
    cout << "Testing ceilPowerOfTwo:" << endl;
    unsigned int r1 = AudioData::ceilPowerOfTwo(257);
    unsigned int r2 = AudioData::ceilPowerOfTwo(256);
    cout << "ceilPowerOfTwo(257) = " << r1 << endl;
    cout << "ceilPowerOfTwo(256) = " << r2 << endl;
    if (r1 != 512 || r2 != 256)
        return false;

    const double overlap = 0.5;
    double* data = new double[nSamples];
    const double pi = 4.0 * atan(1.0);

    srand((unsigned int) time(0));

    cout << "Original data:" << endl;
    for (int i = 0; i < nSamples; i++) {
        data[i] = sin(pi * i / (nSamples - 1));
        cout << data[i] << " ";
    }
    cout << endl << endl;

    AudioData audioData(vector<double*>(1, data), nSamples, 1000);

    pair<Matrix*, Matrix*> spectrogram =
        audioData.computeSpectrogram(SqHannFunction, windowSize, overlap, 0);
    auto_ptr<Matrix> pAmplitudeMatrix(spectrogram.first);
    auto_ptr<Matrix> pPhaseMatrix(spectrogram.second);
    
    cout << "Amplitude spectrogram:" << endl;
    cout << *pAmplitudeMatrix << endl;
    
    cout << "Phase spectrogram:" << endl;
    cout << *pPhaseMatrix << endl;

    auto_ptr<AudioData> pAudioData2(AudioData::fromSpectrogram(*pAmplitudeMatrix, *pPhaseMatrix,
        SqHannFunction, windowSize, overlap, 1000));

    cout << "Result data:" << endl;
    const double* data2 = pAudioData2->getChannel(0);
    for (unsigned int i = 0; i < pAudioData2->nrOfSamples(); i++) {
        cout << data2[i] << " ";
    }
    cout << endl;

    return true;
}


bool SpectralAnalysisTest::performTest()
{
    return performTest(50, 10) && performTest(52, 10) && performTest(52, 11);
}


} // namespace Testing
