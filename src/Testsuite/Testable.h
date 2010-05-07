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


#ifndef __TESTABLE_H__
#define __TESTABLE_H__


#include <cmath>
#include <common.h>


// Forward declaration
namespace blissart { namespace linalg { class Matrix; } }


/**
 * Classes implementing component-level testing
 */
namespace Testing {


/**
 * An abstract base class for tests
 */
class Testable
{
public:
    /**
     * Performs the actual test.
     * Abstract method that must be implemented by derived classes.
     * @return  true if and only if all tests were ok
     */
    virtual bool performTest() = 0;


    /**
     * Returns the name of the test.
     * Abstract method that must be implemented by derived classes.
     * @return  a pointer to a string
     */
    virtual const char *name() = 0;


    virtual ~Testable() {}
    

protected:
    /**
     * Compares two real values with the given precision.
     * @param   a       the first double value
     * @param   b       the second double value
     * @param   prec    a double value specifying the desired precision, e.g. 1e-10
     *                  (which is also the default value)
     * @return          true if and only if the absolute value of a-b is less than
     *                  prec
     */
    inline static bool epsilonCheck(double a, double b, const double prec = 1e-10)
    {
        debug_assert(prec >= 0);
        return (fabs(a-b) <= prec);
    }


    /**
     * Compares two real matrices with the given precision.
     * @param   a       the first matrix
     * @param   b       the second matrix
     * @param   prec    a double value specifying the desired precision, e.g. 1e-10
     *                  (which is also the default value)
     * @return          true if and only if all entries have an absolute difference
                        of less than prec
     */
    static bool epsilonCheck(const blissart::linalg::Matrix& a,
                      const blissart::linalg::Matrix& b,
                      const double prec = 1e-10);


};


} // namespace Testing


#endif
