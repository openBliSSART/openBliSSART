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


#include "MFCCTest.h"
#include <blissart/feature/mfcc.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/audio/AudioData.h>
#include <blissart/audio/audio.h>
#include <Poco/SharedPtr.h>

#include <iostream>
#include <cstdlib>
#include <ctime>


using namespace std;
using namespace blissart;
using namespace blissart::audio;
using namespace blissart::feature;
using namespace blissart::linalg;


namespace Testing {


MFCCTest::MFCCTest(const std::string& filename) : _filename(filename)
{
}


bool MFCCTest::performTest()
{
    {
        audio::initialize();

        Poco::SharedPtr<AudioData> pAd = AudioData::fromFile(_filename, true);
        pair<Matrix*, Matrix*> sp = 
            pAd->computeSpectrogram(HammingFunction, 25, 0.598, 0, true);
        cout << "size = " << sp.first->rows() << " x " << sp.first->cols() << endl;
        delete sp.second;
        Poco::SharedPtr<Matrix> pM(sp.first);
        Poco::SharedPtr<Matrix> mel = 
            melSpectrum(*pM, pAd->sampleRate(), 26, 32767.0);

        audio::shutdown();

        // 32767.0 for compatibility with HTK results
        cout << "Mel spectrum (first column):" << endl;
        for (unsigned int i = 0; i < 26; ++i) {
            cout << mel->at(i, 0) << " ";
        }
        cout << endl;
        cout << "Mel spectrum (2nd column):" << endl;
        for (unsigned int i = 0; i < 26; ++i) {
            cout << mel->at(i, 1) << " ";
        }
        cout << endl;
        Poco::SharedPtr<Matrix> mfcc = computeCepstrogram(*mel, 13);
        cout << setprecision(3) << fixed;
        cout << "Cepstrum (first column):" << endl;
        for (unsigned int i = 0; i < 13; ++i) {
            cout << mfcc->at(i, 0) << " ";
        }
        cout << endl << endl;
    }

    {
        const double melData1[] = { 64.552, 47.181, 50.020, 60.640, 79.966, 
                                    50.395, 71.996, 79.101, 61.153, 48.371,
                                    53.110, 40.053, 59.405, 55.779, 64.960,
                                    67.867, 99.894, 170.688, 135.126, 126.953,
                                    93.942, 81.654, 75.565, 117.853, 116.354, 
                                    160.975};
        const double melData2[] = { 89.907, 41.827, 43.779, 51.809, 66.028,
                                    47.591, 61.885, 89.544, 71.710, 42.118,
                                    52.093, 43.860, 53.031, 60.732, 62.036,
                                    66.274, 98.875, 151.064, 137.968, 119.193,
                                    97.601, 77.480, 75.805, 125.598, 127.930, 
                                    145.707};
        Matrix melMatrix(26, 2);
        melMatrix.setColumn(0, ColVector(26, melData1));
        melMatrix.setColumn(1, ColVector(26, melData2));
        Poco::SharedPtr<Matrix> mfcc = computeCepstrogram(melMatrix, 13);
        cout << setprecision(3) << fixed;
        cout << "Mel matrix:" << endl << melMatrix << endl;
        cout << "Cepstrum (first column):" << endl;
        for (unsigned int i = 0; i < 13; ++i) {
            cout << mfcc->at(i, 0) << " ";
        }
        cout << endl;
        const double mfccData1[] = { 31.205, -1.399, 0.496, 0.286, -0.540, 
                                     -0.523, 0.651, -0.327, 0.306, 0.081, 
                                     -0.215, 0.098, 0.262 };
        for (unsigned int i = 0; i < 13; ++i) {
            if (!epsilonCheck(mfcc->at(i, 0), mfccData1[i], 1e-3))
                return false;
        }

        Poco::SharedPtr<Matrix> mfccL = computeCepstrogram(melMatrix, 13, 22);
        cout << "Cepstrum (first column, liftered):" << endl;
        for (unsigned int i = 0; i < 13; ++i) {
            cout << mfccL->at(i, 0) << " ";
        }
        cout << endl;
        const double mfccLiftered1[] = { 31.205, -3.588, 2.034, 1.591, -3.754, 
                                         -4.292, 6.063, -3.356, 3.369, 0.940, 
                                         -2.553, 1.176, 3.113 };
        for (unsigned int i = 0; i < 13; ++i) {
            if (!epsilonCheck(mfccL->at(i, 0), mfccLiftered1[i], 1e-3))
                return false;
        }

        cout << "Cepstrum (2nd column):" << endl;
        for (unsigned int i = 0; i < 13; ++i) {
            cout << mfcc->at(i, 1) << " ";
        }
        cout << endl;
        const double mfccData2[] = { 31.069, -1.455, 0.496, 0.255, -0.422,
                                     -0.336, 0.820, -0.081, 0.442, 0.169,
                                     -0.237, 0.203, 0.298 };
        for (unsigned int i = 0; i < 13; ++i) {
            if (!epsilonCheck(mfcc->at(i, 1), mfccData2[i], 1e-3))
                return false;
        }

        cout << endl;
    }

    {
        cout << "Testing delta regression:" << endl;

        const double data[] = { 1, 2, 3, 4, 5, 
                                2, 3, 4, 5, 6, 
                                1, 2, 3, 4, 5, 
                                2, 3, 4, 5, 6 };
        const double delta[] = { 0.5, 0.8, 1.0, 0.8, 0.5,
                                 0.5, 0.8, 1.0, 0.8, 0.5,
                                 0.5, 0.8, 1.0, 0.8, 0.5,
                                 0.5, 0.8, 1.0, 0.8, 0.5 };

        Matrix m(4, 5, data);
        Matrix d(4, 5, delta);

        Matrix* result = deltaRegression(m, 2);
        
        cout << "Matrix M:" << endl << m << endl;
        cout << "Delta regression:" << endl << *result << endl;
        
        bool ok = (*result == d);
        delete result;
        if (!ok) return false;
    }

    return true;
}


} // namespace Testing
