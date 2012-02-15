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
#include <Poco/LogStream.h>

#include <cuda_runtime.h>

#include <sstream>


using blissart::nmf::Deconvolver;
using blissart::linalg::Matrix;
using namespace std;


namespace benchmark {


NMFBenchmark::NMFBenchmark() : _cf("all"), _rows(100), _iter(100)
{
}


void NMFBenchmark::addOptions(Poco::Util::OptionSet& options)
{
    options.addOption(
        Poco::Util::Option("rows", "r", "Number of matrix rows to decompose", 
        false, "<n>", true)
    );
    options.addOption(
        Poco::Util::Option("cf", "f", "NMF cost function (or \"all\")",
        false, "<func>", true)
    );
    options.addOption(
        Poco::Util::Option("iter", "i", "Number of NMF iterations",
        false, "<n>", true)
    );
}


void NMFBenchmark::setOptions(const Benchmark::OptionsMap& options)
{
    OptionsMap::const_iterator tmp = options.find("rows");
    if (tmp != options.end())
        _rows = Poco::NumberParser::parse(tmp->second);

    tmp = options.find("cf");
    if (tmp != options.end())
        _cf = tmp->second;
        
    tmp = options.find("iter");
    if (tmp != options.end())
        _iter = Poco::NumberParser::parse(tmp->second);
}


void NMFBenchmark::run()
{
    // Numbers of components to consider
    const unsigned int nc[] = { 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000 } ;
    //const unsigned int nc[] = { 500 } ;
    const unsigned int nnc  =   11;
    //const unsigned int nnc  =   1;
    
#ifdef HAVE_CUDA
    Poco::LogStream ls(logger());
    size_t free, total;
    // Display GPU memory usage.
    cudaMemGetInfo(&free, &total);
    ls.information() << "Free: " << free << " / total: " << total << endl;
#endif


    // Create 100x1000 Gaussian random matrix
	Matrix v(_rows, 1000, blissart::nmf::gaussianRandomGenerator);

	// NMF, Euclidean distance, optimized for overcomplete fact.
    if (_cf == "all" || _cf == "ed") {
        for (int i = 0; i < nnc; ++i) {
            Deconvolver d(v, nc[i], 1);
            d.setAlgorithm(Deconvolver::Overcomplete);
            stringstream bnStr;
            bnStr << "NMF-ED " << v.rows() << "x" << v.cols() 
                  << " r=" << nc[i];
            logger().information(bnStr.str());
            {
                ScopedStopwatch s(*this, bnStr.str());
                d.decompose(Deconvolver::EuclideanDistance, _iter, 0.0, this);
            }
        }
        // NMF, Euclidean distance, optimized for incomplete fact.
        for (int i = 0; i < nnc; ++i) {
            Deconvolver d(v, nc[i], 1);
            d.setAlgorithm(Deconvolver::NMFEDIncomplete);
            stringstream bnStr;
            bnStr << "NMF-EDinc " << v.rows() << "x" << v.cols() 
                  << " r=" << nc[i];
            logger().information(bnStr.str());
            {
                ScopedStopwatch s(*this, bnStr.str());
                d.decompose(Deconvolver::EuclideanDistance, _iter, 0.0, this);
            }
        }
    } // _cf == ed

    if (_cf == "all" || _cf == "is") {
        // NMF, IS divergence
        for (int i = 0; i < nnc; ++i) {
            Deconvolver d(v, nc[i], 1);
            stringstream bnStr;
            bnStr << "NMF-IS " << v.rows() << "x" << v.cols() 
                  << " r=" << nc[i];
            logger().information(bnStr.str());
            {
                ScopedStopwatch s(*this, bnStr.str());
                d.decompose(Deconvolver::ISDivergence, _iter, 0.0, this);
            }
        }
    }

    if (_cf == "all" || _cf == "kl") {
        // NMF, KL divergence
        for (int i = 0; i < nnc; ++i) {
            Deconvolver d(v, nc[i], 1);
            stringstream bnStr;
            bnStr << "NMF-KL " << v.rows() << "x" << v.cols() 
                  << " r=" << nc[i];
            logger().information(bnStr.str());
            {
                ScopedStopwatch s(*this, bnStr.str());
                d.decompose(Deconvolver::KLDivergence, _iter, 0.0, this);
#ifdef HAVE_CUDA
                // Display GPU memory usage.
                cudaMemGetInfo(&free, &total);
                ls.information() << "Free: " << free << " / total: " << total << endl;
#endif
            }
        }
    }
    
#ifdef HAVE_CUDA
    // Display GPU memory usage.
    cudaMemGetInfo(&free, &total);
    ls.information() << "Free: " << free << " / total: " << total << endl;
#endif

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
