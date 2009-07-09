//
// $Id: WaveTest.cpp 855 2009-06-09 16:15:50Z alex $
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
#include <iostream>
#include <memory>


using namespace std;
using namespace blissart;
using namespace blissart::audio;


namespace Testing {


bool WaveTest::performTest()
{
    try {
        // MP3 -> All channels WAV
        {
            const char* fileName = "test_1.wav";
            cout << "Reading 'schnappi.mp3' (all channels)...";
            auto_ptr<AudioData> mp3_data = auto_ptr<AudioData>(AudioData::fromFile("schnappi.mp3"));
            cout << "ok." << endl
                 << "Saving '" << fileName << "'...";
            if (!WaveEncoder::saveAsWav(*mp3_data, fileName)) {
                cerr << "WaveTest::performTest - saveAsWav failed!" << endl;
                return false;
            }
            cout << "ok." << endl;
        }

        // MP3 -> Mono WAV
        {
            const char* fileName = "test_2.wav";
            cout << "Reading 'schnappi.mp3' (mono)...";
            auto_ptr<AudioData> mp3_data_mono = auto_ptr<AudioData>(AudioData::fromFile("schnappi.mp3", true));
            cout << "ok." << endl
                 << "Saving '" << fileName << "'...";
            if (!WaveEncoder::saveAsWav(*mp3_data_mono, fileName)) {
                cerr << "WaveTest::performTest - saveAsWav failed!" << endl;
                return false;
            }
            cout << "ok." << endl;
        }

        // WAV -> All channels WAV
        {
            const char* fileName = "test_3.wav";
            cout << "Reading 'horse.wav' (all channels)...";
            auto_ptr<AudioData> wav_data = auto_ptr<AudioData>(AudioData::fromFile("horse.wav"));
            cout << "ok." << endl
                 << "Saving '" << fileName << "'...";
            if (!WaveEncoder::saveAsWav(*wav_data, fileName)) {
                cerr << "WaveTest::performTest - saveAsWav failed!" << endl;
                return false;
            }
            cout << "ok." << endl;
        }

        // OGG -> All channels WAV
        {
            const char* fileName = "test_4.wav";
            cout << "Reading 'schnappi.ogg' (all channels)...";
            auto_ptr<AudioData> ogg_data = auto_ptr<AudioData>(AudioData::fromFile("schnappi.ogg"));
            cout << "ok." << endl
                 << "Saving '" << fileName << "'...";
            if (!WaveEncoder::saveAsWav(*ogg_data, fileName)) {
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
