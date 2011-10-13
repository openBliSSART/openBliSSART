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
#include <blissart/linalg/common.h>


namespace blissart {
    
namespace linalg {


/**
 * Classes that implement generation of standard matrix and vector types.
 */
namespace generators {


/**
 * \addtogroup linalg
 * @{
 */


/**
 * @name Vector generators
 */


/**
 * Generates a vector filled with uniform random numbers from the interval
 * [0,1[.
 */
inline Elem random(unsigned int)
{
    return (Elem)rand() / (Elem)RAND_MAX;
}

    
/**
 * Generates an all-zero vector.
 */
inline Elem zero(unsigned int)
{
    return 0.0f;
}


/**
 * @}
 *
 *
 * @name Matrix generators
 */


/**
 * Generates a matrix filled with uniform random numbers from the interval
 * [0,1[.
 */
inline Elem random(unsigned int, unsigned int)
{
    return (Elem)rand() / (Elem)RAND_MAX;
}


/**
 * Generates the identity matrix.
 */
inline Elem identity(unsigned int i, unsigned int j)
{
    return (i == j ? 1.0 : 0.0);
}


/**
 * Generates an all-zero matrix.
 */
inline Elem zero(unsigned int, unsigned int)
{
    return 0.0;
}


/**
 * Generates an all-one matrix.
 */
inline Elem unity(unsigned int, unsigned int)
{
    return 1.0;
}


/**
 * @}
 */


/**
 * @}
 */


} // namespace generators

} // namespace linalg

} // namespace blissart


#endif // __BLISSART_LINALG_GENERATORS_GENERATORS_H__
