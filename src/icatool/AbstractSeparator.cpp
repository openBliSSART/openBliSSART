//
// $Id: AbstractSeparator.cpp 855 2009-06-09 16:15:50Z alex $
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


#include "AbstractSeparator.h"
#include <blissart/ica/FastICA.h>
#include <blissart/linalg/Matrix.h>

#include <iostream>
#include <memory>
#include <cmath>


using namespace std;
using namespace blissart::linalg;
using namespace blissart::ica;


AbstractSeparator::AbstractSeparator(unsigned int nSources,
                                     double prec, unsigned int maxIter) :
    _matrix(NULL),
    _nSources(nSources),
    _prec(prec),
    _maxIter(maxIter)
{
}


AbstractSeparator::~AbstractSeparator()
{
    if (_matrix)
        delete _matrix;
}


const Matrix* AbstractSeparator::separate(int* numErrors)
{
    cout << "Separating..." << endl
         << "   precision = " << _prec << endl
         << "   max_iter  = " << _maxIter << endl;

    // Initialize and perform FastICA while storing the
    // # of convergence errors iff desired
    auto_ptr<FastICA> f(FastICA::compute(_matrix, _nSources,
                                         false, // not centered yet
                                         _maxIter, _prec));

    // Is the callee interested in the # of convergence errors?
    if (numErrors)
        *numErrors = f->nrOfConvergenceErrors();

    // Adjust the result
    cout << "Adjusting...";
    normalizeMatrix();
    cout << "done." << endl;

    // Return the result
    return _matrix;
}


void AbstractSeparator::normalizeMatrix(void)
{
    for (unsigned int i = 0; i < _matrix->rows(); i++) {
        // Determine the maximum absolute value for every single row.
        double absMax = 0;
        for (unsigned int j = 0; j < _matrix->cols(); j++) {
            double absVal = fabs(_matrix->at(i, j));
            if (absVal > absMax)
                absMax = absVal;
        }
        // Then normalize and move that row.
        if (absMax > 0) {
            double scaleFactor = 1.0 / absMax;
            for (unsigned int j = 0; j < _matrix->cols(); j++) {
                _matrix->setAt(i, j, _matrix->at(i, j) * scaleFactor);
            }
        }
    }
}
