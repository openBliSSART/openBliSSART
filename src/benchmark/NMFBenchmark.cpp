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


#include "NMFBenchmark.h"

#include <blissart/nmf/Deconvolver.h>
#include <blissart/nmf/randomGenerator.h>
#include <Poco/Util/Application.h>
#include <Poco/NumberParser.h>

#include <sstream>


using blissart::nmf::Deconvolver;
using blissart::linalg::Matrix;
using namespace std;


namespace benchmark {


void NMFBenchmark::addOptions(Poco::Util::OptionSet& options)
{
    options.addOption(
        Poco::Util::Option("nmf-comp", "c", "Number of NMF components", 
        false, "<n>", true)
    );
}


void NMFBenchmark::setOptions(const Benchmark::OptionsMap& options)
{
    OptionsMap::const_iterator tmp = options.find("nmf-comp");
    if (tmp != options.end())
        _nComp = Poco::NumberParser::parse(tmp->second);
    else
        _nComp = 100;
}


void NMFBenchmark::run()
{
    // Numbers of components to consider
    const unsigned int nc[] = { 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000 } ;
    const unsigned int nnc  =   11;

    // Create 100x1000 Gaussian random matrix
	Matrix v(_nComp, 1000, blissart::nmf::gaussianRandomGenerator);

	// NMF, Euclidean distance, optimized for overcomplete fact.
    for (int i = 0; i < nnc; ++i) {
		Deconvolver d(v, nc[i], 1);
        d.setAlgorithm(Deconvolver::Overcomplete);
        stringstream bnStr;
        bnStr << "NMF-ED " << v.rows() << "x" << v.cols() 
              << " r=" << nc[i];
        logger().information(bnStr.str());
        {
            ScopedStopwatch s(*this, bnStr.str());
            // fixed number of iterations (100)
            d.decompose(Deconvolver::EuclideanDistance, 100, 0.0, this);
        }
	}

    // NMF, Euclidean distance, optimized for incomplete fact.
    for (int i = 0; i < nnc; ++i) {
		Deconvolver d(v, nc[i], 1);
        d.setAlgorithm(Deconvolver::Incomplete);
        stringstream bnStr;
        bnStr << "NMF-EDinc " << v.rows() << "x" << v.cols() 
              << " r=" << nc[i];
        logger().information(bnStr.str());
        {
            ScopedStopwatch s(*this, bnStr.str());
            // fixed number of iterations (100)
            d.decompose(Deconvolver::EuclideanDistance, 100, 0.0, this);
        }
	}

    return;

	// NMF, KL divergence
    for (int i = 0; i < nnc; ++i) {
		Deconvolver d(v, nc[i], 1);
        stringstream bnStr;
        bnStr << "NMF-KL " << v.rows() << "x" << v.cols() 
              << " r=" << nc[i];
        {
            ScopedStopwatch s(*this, bnStr.str());
            // fixed number of iterations (100)
            d.decompose(Deconvolver::KLDivergence, 100, 0.0, this);
        }
	}

	// Sparse NMF, Euclidean distance
    for (int i = 0; i < nnc; ++i) {
		Deconvolver d(v, nc[i], 1);
        stringstream bnStr;
        bnStr << "NMF-ED(s) " << v.rows() << "x" << v.cols() 
              << " r=" << nc[i];
        {
            ScopedStopwatch s(*this, bnStr.str());
            // fixed number of iterations (100)
            d.decompose(Deconvolver::EuclideanDistanceSparse, 100, 0.0, this);
        }
	}

    // Sparse NMF, KL divergence
    for (int i = 0; i < nnc; ++i) {
		Deconvolver d(v, nc[i], 1);
        stringstream bnStr;
        bnStr << "NMF-KL(s) " << v.rows() << "x" << v.cols() 
              << " r=" << nc[i];
        {
            ScopedStopwatch s(*this, bnStr.str());
            // fixed number of iterations (100)
            d.decompose(Deconvolver::KLDivergenceSparse, 100, 0.0, this);
        }
	}
}


void NMFBenchmark::progressChanged(float progress)
{
	cout << "\r"
		 << fixed << setw(6) << setprecision(2)
		 << (progress * 100.0) << "% complete ...";
	if (progress == 1.0)
		cout << endl;
	cout << flush;
}


} // namespace benchmark
