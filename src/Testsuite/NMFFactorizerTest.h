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

#ifndef __NMFFACTORIZERTEST_H__
#define __NMFFACTORIZERTEST_H__


#include "Testable.h"


namespace Testing {


/**
 * Test the non-negative matrix factorization algorithm by creating a random
 * matrix X, using the matrix 2*I (identity) as starting value for the first factor 
 * and the matrix X as starting value for the second factor.
 */
class NMFFactorizerTest : public Testable
{
public:
    bool performTest();
    inline const char *name() { return "NMFFactorizerTest"; }
};


} // namespace Testing


#endif
