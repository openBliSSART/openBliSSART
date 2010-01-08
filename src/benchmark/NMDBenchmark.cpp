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


#include "NMDBenchmark.h"

#include <blissart/nmf/Deconvolver.h>
#include <blissart/nmf/randomGenerator.h>

#include <iomanip>


using Poco::Timestamp;
using blissart::nmf::Deconvolver;
using blissart::linalg::Matrix;
using namespace std;


namespace benchmark {


void NMDBenchmark::run()
{
	Matrix v(500, 1000, blissart::nmf::gaussianRandomGenerator);

	// NMF, Euclidean distance,
	// 500x1000 Gaussian random matrix, 20 components
	// fixed number of iterations (100)
	{
		Deconvolver d(v, 20, 1);
		Timestamp start;
		d.factorizeED(100, 0.0, this);
		Timestamp end;
		_elapsedTimes["NMF-ED 500x1000 r=20"] = end - start;
	}

	// NMD, Euclidean distance, 
	// 500x1000 Gaussian random matrix, 20 components, 5 spectra
	// fixed number of iterations (100)
	{
		Deconvolver d(v, 20, 5);
		Timestamp start;
		d.factorizeED(100, 0.0, this);
		Timestamp end;
		_elapsedTimes["NMD-ED 500x1000 r=20 t=5"] = end - start;
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
