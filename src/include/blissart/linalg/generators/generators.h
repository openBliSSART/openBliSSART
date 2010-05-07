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

#ifndef __BLISSART_LINALG_GENERATORS_GENERATORS_H__
#define __BLISSART_LINALG_GENERATORS_GENERATORS_H__


#include <cstdlib>


namespace blissart {
    
namespace linalg {


/**
 * Classes that implement generation of standard matrix types.
 */
namespace generators {


// Vector related

inline double random(unsigned int)
{
    return (double)rand() / (double)RAND_MAX;
}

    
inline double zero(unsigned int)
{
    return 0.0;
}


// Matrix related

inline double random(unsigned int, unsigned int)
{
    return (double)rand() / (double)RAND_MAX;
}


inline double identity(unsigned int i, unsigned int j)
{
    return (i == j ? 1.0 : 0.0);
}


inline double zero(unsigned int, unsigned int)
{
    return 0.0;
}


} // namespace generators

} // namespace linalg

} // namespace blissart


#endif // __BLISSART_LINALG_GENERATORS_GENERATORS_H__
