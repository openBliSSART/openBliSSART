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

#include "WaveTest.h"
#include <blissart/audio/AudioData.h>
#include <blissart/audio/WaveEncoder.h>
#include <blissart/linalg/Matrix.h>
#include <iostream>
#include <memory>


using namespace std;
using namespace blissart;
using namespace blissart::audio;
using blissart::linalg::Matrix;


namespace Testing {


WaveTest::WaveTest(const std::string& filename) : _filename(filename)
{
}


bool WaveTest::performTest()
{
    try {
        // I/O
        // All channels WAV
        {
            const std::string testFileName = "test_1.wav";
            cout << "Reading '" << _filename << "' (all channels) ... ";
            auto_ptr<AudioData> adata = auto_ptr<AudioData>(AudioData::fromFile(_filename));
            cout << "ok." << endl
                 << "Saving '" << testFileName << "'... ";
            if (!WaveEncoder::saveAsWav(*adata, testFileName)) {
                cerr << "WaveTest::performTest - saveAsWav failed!" << endl;
                return false;
            }
            cout << "ok." << endl;
        }

        // Mono WAV
        {
            const std::string testFileName = "test_2.wav";
            cout << "Reading '" << _filename << "' (mono) ... ";
            auto_ptr<AudioData> adata = auto_ptr<AudioData>(AudioData::fromFile(_filename));
            cout << "ok." << endl
                 << "Saving '" << testFileName << "'... ";
            if (!WaveEncoder::saveAsWav(*adata, testFileName)) {
                cerr << "WaveTest::performTest - saveAsWav failed!" << endl;
                return false;
            }
            cout << "ok." << endl;
        }

        // Transformation stuff
        {
            const std::string testFileName = "test_3.wav";
            cout << "Reading '" << _filename << "' (mono) ... ";
            auto_ptr<AudioData> adata = auto_ptr<AudioData>(AudioData::fromFile(_filename));
            cout << "ok." << endl;
            cout << "Transformation ... ";
            pair<Matrix*, Matrix*> sp = adata->computeSpectrogram(HammingFunction,
                25, 0.6, 0);
            auto_ptr<Matrix> amplSp(sp.first);
            auto_ptr<Matrix> phaseSp(sp.second);
            cout << "ok." << endl;
            cout << "Backtransformation ... ";
            auto_ptr<AudioData> adataRec(AudioData::fromSpectrogram(
                *amplSp, *phaseSp, HammingFunction, 25, 0.6, adata->sampleRate()));
            cout << "ok." << endl
                 << "Saving '" << testFileName << "'...";
            if (!WaveEncoder::saveAsWav(*adataRec, testFileName)) {
                cerr << "WaveTest::performTest - saveAsWav failed!" << endl;
                return false;
            }
            cout << "ok." << endl;
        }
    } catch (exception& ex) {
        cout << "ERROR: " << ex.what() << endl;
        return false;
    }

    return true;
}


} // namespace Testing
