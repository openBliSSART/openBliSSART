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


#ifndef __BLISSART_SPECTRAL_SUBTRACTION_TRANSFORM_H__
#define __BLISSART_SPECTRAL_SUBTRACTION_TRANSFORM_H__


#include <common.h>
#include <blissart/MatrixTransform.h>


namespace blissart {


namespace transforms {


/**
 * \addtogroup framework
 * @{
 */

/**
 * This implements a very basic version of spectral subtraction and is 
 * currently intended for testing purposes only.
 */
class LibFramework_API SpectralSubtractionTransform : public MatrixTransform
{
public:
    /**
     * Default constructor. Constructs a SpectralSubtractionTransform with the
     * parameters given in the Application's LayeredConfiguration.
     */
    SpectralSubtractionTransform();


    /**
     * Implementation of MatrixTransform interface.
     */
    virtual const char * name() const;


    /**
     * Implementation of MatrixTransform interface.
     * Performs spectral subtraction.
     */
    virtual linalg::Matrix * transform(linalg::Matrix* m) const;


    /** 
     * Implementation of MatrixTransform interface.
     * Does nothing.
     */
    virtual linalg::Matrix * inverseTransform(linalg::Matrix* m) const;
};


/**
 * @}
 */


} // namespace transforms


} // namespace blissart


#endif // __BLISSART_SLIDING_WINDOW_TRANSFORM_H__
