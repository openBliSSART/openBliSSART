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


using blissart::nmf::Deconvolver;
using blissart::linalg::Matrix;
using namespace std;


namespace benchmark {


NMDBenchmark::NMDBenchmark() : _cf("all"), _nComp(100)
{
}


void NMDBenchmark::addOptions(Poco::Util::OptionSet& options)
{
    options.addOption(
        Poco::Util::Option("nmf-comp", "c", "Number of rows in V matrix", 
        false, "<n>", true)
    );
    options.addOption(
        Poco::Util::Option("cf", "f", "NMF cost function (or \"all\")",
        false, "<func>", true)
    );
}


void NMDBenchmark::setOptions(const Benchmark::OptionsMap& options)
{
    OptionsMap::const_iterator tmp = options.find("nmf-comp");
    if (tmp != options.end())
        _nComp = Poco::NumberParser::parse(tmp->second);

    tmp = options.find("cf");
    if (tmp != options.end())
        _cf = tmp->second;
}


void NMDBenchmark::run()
{
	Matrix v(_nComp, 500, blissart::nmf::gaussianRandomGenerator);

    unsigned int t = 5;
    unsigned int r = 20;

	// NMD, Euclidean distance, 
	// 500x1000 Gaussian random matrix, 20 components, 5 spectra
	// fixed number of iterations (100)
	{
		Deconvolver d(v, 10, t);
        stringstream bnStr;
        bnStr << "NMD-ED " << v.rows() << "x" << v.cols() 
              << " r=" << r << " t=" << t;
        logger().information(bnStr.str());
        {
            ScopedStopwatch s(*this, bnStr.str());
            d.factorizeNMDBreg(100, 0.0, this, 2);
        }
	}

	// NMD, KL divergence
	// 500x1000 Gaussian random matrix, 20 components, 5 spectra
	// fixed number of iterations (100)
	{
		Deconvolver d(v, 10, t);
        stringstream bnStr;
        bnStr << "NMD-KL " << v.rows() << "x" << v.cols() 
              << " r=" << r << " t=" << t;
        logger().information(bnStr.str());
        {
            ScopedStopwatch s(*this, bnStr.str());
            d.factorizeNMDBreg(100, 0.0, this, 1);
        }
	}

	// NMD, IS divergence
	// 500x1000 Gaussian random matrix, 20 components, 5 spectra
	// fixed number of iterations (100)
	{
		Deconvolver d(v, 10, t);
        stringstream bnStr;
        bnStr << "NMD-IS " << v.rows() << "x" << v.cols() 
              << " r=" << r << " t=" << t;
        logger().information(bnStr.str());
        {
            ScopedStopwatch s(*this, bnStr.str());
            d.factorizeNMDBreg(100, 0.0, this, 0);
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
