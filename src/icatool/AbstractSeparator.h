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

#ifndef __ABSTRACTSEPARATOR_H__
#define __ABSTRACTSEPARATOR_H__


#include <cassert>


// Forward declaration
namespace blissart { namespace linalg { class Matrix; } }


/**
 * Abstract base class for the component analysis of datasets
 */
class AbstractSeparator
{
public:
    /**
     * Destructs an instance of AbstractSeparator and frees all allocated memory.
     */
    virtual ~AbstractSeparator();


    /**
     * Trys to separate sources via Independent Component Analysis. Each row
     * of the returned matrix holds another independent random variable.
     * @param numErrors     an int pointer that will hold the number of convergence
     *                      errors that may have possibly occured during the ica
     *                      computation step. Can be NULL if this information isn't
     *                      needed
     * @return              a const pointer to the internal data matrix
     */
    virtual const blissart::linalg::Matrix* separate(int* numErrors = 0L);


protected:
    /**
     * Constructs an AbstractSeparator.
     * @param   nSources    the # of sources to be separated
     * @param   prec        the desired precision
     * @param   maxIter     the maximum # of iterations during FastICA
     */
    AbstractSeparator(unsigned int nSources, double prec, unsigned int maxIter);


    blissart::linalg::Matrix* _matrix;


private:
    // Forbid copy constructor and = operator
    AbstractSeparator(const AbstractSeparator&);
    AbstractSeparator& operator = (const AbstractSeparator&);


    /**
     * Normalizes every row of the matrix individually. The resulting values
     * are all within [-1, 1].
     */
    void normalizeMatrix(void);


    unsigned int       _nSources;
    const double       _prec;
    const unsigned int _maxIter;
};


#endif
