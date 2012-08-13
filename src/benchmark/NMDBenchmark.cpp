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


#include "NMDBenchmark.h"

#include <blissart/nmf/Deconvolver.h>
#include <blissart/nmf/randomGenerator.h>
#include <Poco/Util/Application.h>
#include <Poco/NumberParser.h>

#include <iomanip>
#include <sstream>
#include <iostream>


using blissart::nmf::Deconvolver;
using blissart::linalg::Matrix;
using namespace std;


namespace benchmark {


NMDBenchmark::NMDBenchmark() : _cf("all"), _iter(100), _rows(100)
{
}


void NMDBenchmark::addOptions(Poco::Util::OptionSet& options)
{
    // NMD Benchmark options are identical to NMF Benchmark.
}


void NMDBenchmark::setOptions(const Benchmark::OptionsMap& options)
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


void NMDBenchmark::run()
{
	Matrix v(_rows, 1000, blissart::nmf::gaussianRandomGenerator);

    unsigned int t = 5;
    // Numbers of components to consider
    const unsigned int nc[] = { 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000 } ;
    const unsigned int nnc  =   11;

    // Base lengths to consider
    const unsigned int bl[] = { 2, 5, 10 } ;
    const unsigned int nbl = 3;

	// NMD, Euclidean distance
    if (_cf == "all" || _cf == "ed")
	{
        for (int j = 0; j < nbl; ++j) {
            for (int i = 0; i < nnc; ++i) {
                Deconvolver d(v, nc[i], bl[j]);
                stringstream bnStr;
                bnStr << "NMD-ED " << v.rows() << "x" << v.cols() 
                      << " r=" << nc[i] << " t=" << bl[j];
                logger().information(bnStr.str());
                {
                    ScopedStopwatch s(*this, bnStr.str());
                    d.decompose(Deconvolver::EuclideanDistance, _iter, 0.0, Deconvolver::NoSparsity, false, this);
                }
            }
        }
	}

	// NMD, KL divergence
    if (_cf == "all" || _cf == "kl")
	{
        for (int j = 0; j < nbl; ++j) {
            for (int i = 0; i < nnc; ++i) {
                Deconvolver d(v, nc[i], bl[j]);
                stringstream bnStr;
                bnStr << "NMD-KL " << v.rows() << "x" << v.cols() 
                      << " r=" << nc[i] << " t=" << bl[j];
                logger().information(bnStr.str());
                {
                    ScopedStopwatch s(*this, bnStr.str());
                    d.decompose(Deconvolver::KLDivergence, _iter, 0.0, Deconvolver::NoSparsity, false, this);
                }
            }
        }
	}

	// NMD, IS divergence
    if (_cf == "all" || _cf == "is")
	{
        for (int j = 0; j < nbl; ++j) {
            for (int i = 0; i < nnc; ++i) {
                Deconvolver d(v, nc[i], bl[j]);
                stringstream bnStr;
                bnStr << "NMD-IS " << v.rows() << "x" << v.cols() 
                      << " r=" << nc[i] << " t=" << bl[j];
                logger().information(bnStr.str());
                {
                    ScopedStopwatch s(*this, bnStr.str());
                    d.decompose(Deconvolver::ISDivergence, _iter, 0.0, Deconvolver::NoSparsity, false, this);
                }
            }
        }
	}
}


void NMDBenchmark::progressChanged(float progress)
{
	cout << "\r"
		 << fixed << setw(6) << setprecision(2)
		 << (progress * 100.0) << "% complete ...";
	if (progress == 1.0)
		cout << endl;
	cout << flush;
}


} // namespace benchmark
