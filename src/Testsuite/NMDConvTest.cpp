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

#include "NMDConvTest.h"
#include <blissart/nmf/Deconvolver.h>
#include <blissart/linalg/generators/generators.h>
#include <blissart/audio/AudioData.h>
#include <blissart/WindowFunctions.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>
#include <Poco/SharedPtr.h>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;
using namespace blissart::audio;
using blissart::nmf::Deconvolver;


namespace Testing {


bool NMDConvTest::convTest(bool modifiedHUpdate)
{
    //const unsigned int m = 500;
    //const unsigned int n = 1000;
    const unsigned int t = 5;
    const unsigned int r = 20;
    const unsigned int nTrials = 1;
    const unsigned int nIt = 200;
    const double beta[] = { 2.0, 1.0, 0.0 };
    const unsigned int nBeta = 3;
    // FIXME: command-line option for this
    const char* filelist = "nmdconv.scp";
    vector<string> files;
    ifstream filelistIS(filelist);
    if (!filelistIS.good()) {
        cerr << "Could not open file " << filelist << endl;
        return false;
    }
    while (!filelistIS.eof()) {
        string filename;
        getline(filelistIS, filename);
        if (filename != "")
            files.push_back(filename);
    }
    filelistIS.close();

    for (unsigned int b = 0; b < nBeta; ++b) {
        vector<vector<double> > cf(nIt, vector<double>(files.size()));
        vector<double>  minCf(nIt, -1);
        vector<double>  maxCf(nIt, -1);
        vector<double> meanCf(nIt, 0);
        vector<double>   norm(files.size());
        //for (unsigned int k = 0; k < nTrials; ++k) {
        for (unsigned int k = 0; k < files.size(); ++k) {
            cout << "file: " << files[k] << endl;
            Poco::SharedPtr<audio::AudioData> pAd = AudioData::fromFile(files[k], true);
            Poco::SharedPtr<Matrix> pAmpM, pPhaseM;
            std::pair<Matrix*, Matrix*> spec = pAd->computeSpectrogram(SqHannFunction, 60, 0.5, 0);
            pAmpM = spec.first;
            pPhaseM = spec.second;
            unsigned int seed = k * 20;
            cout << "Random seed: " << seed << endl;
            //cout << "Creating " << m << "x" << n << " random matrix" << endl;
            //pAmpM->apply(Matrix::mul, 1.0 / pAmpM->frobeniusNorm());
            nmf::Deconvolver d(*pAmpM, r, t);
            srand(seed);
            for (unsigned int s = 0; s < t; ++s) {
                Matrix w(pAmpM->rows(), r, nmf::gaussianRandomGenerator);
                d.setW(0, w);
            }
            Matrix h(r, pAmpM->cols(), nmf::gaussianRandomGenerator);
            d.setH(h);
            d.setNMDModifiedHUpdate(modifiedHUpdate);

            cout << "Deconvolving (t = " << t << ", r = "<< r << ")" << endl;
            norm[k] = pAmpM->frobeniusNorm(); //pAmpM->rows() * pAmpM->cols();
            //cout << "norm = " << norm << endl;
            for (unsigned int i = 0; i < nIt; ++i) {
                cout << i + 1 << " " << flush;
                d.factorizeNMDBreg(1, 0.0, beta[b]);
                double c = d.getCfValue(Deconvolver::BetaDivergence, beta[b]);
                //if (i == 0)
                //    norm = c;
                //c /= norm;
                cf[i][k] = c;
                cout << "c = " << c << " " << flush;
                meanCf[i] += c;
                if (k == 0 || maxCf[i] < c)
                    maxCf[i] = c;
                if (k == 0 || minCf[i] > c)
                    minCf[i] = c; 
            }
            cout << endl;
        }
        ostringstream lss;
        lss << "nmderror_" << beta[b];
        if (modifiedHUpdate) lss << "_modH";
        lss << ".dat";
        string tmp = lss.str();
        ofstream logStream(tmp.c_str());
        //cout << "--- beta = " << beta[b] << endl << endl;
        logStream << "name";
        for (unsigned int k = 0; k < files.size(); ++k) {
            logStream << '\t' << files[k];
        }
        logStream << endl;
        logStream << "norm";
        for (unsigned int k = 0; k < files.size(); ++k) {
            logStream << '\t' << norm[k];
        }
        logStream << endl;
        for (unsigned int i = 0; i < nIt; ++i) {
            meanCf[i] /= (double)files.size();
            logStream << i + 1;
            for (unsigned int k = 0; k < files.size(); ++k) {
                logStream << '\t' << cf[i][k];
            }
            /*logStream << '\t' << minCf[i] << '\t' << meanCf[i] 
                      << '\t' << maxCf[i] << endl;*/
            logStream << endl;
        }
        logStream.close();
    }
    return true;
}


bool NMDConvTest::performTest()
{
    return convTest(false) && convTest(true);
}


} // namespace Testing
