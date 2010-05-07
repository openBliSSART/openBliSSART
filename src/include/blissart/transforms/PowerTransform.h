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


#ifndef __BLISSART_POWER_TRANSFORM_H__
#define __BLISSART_POWER_TRANSFORM_H__


#include <common.h>
#include <blissart/MatrixTransform.h>


namespace blissart {


namespace transforms {


/**
 * \addtogroup framework
 * @{
 */

/**
 * Calculation of the power spectrum. Implements MatrixTransform interface.
 */
class LibFramework_API PowerTransform : public blissart::MatrixTransform
{
public:
    /**
     * Default constructor. Constructs a PowerTransform that squares its input.
     */
    PowerTransform();


    /**
     * Implementation of MatrixTransform interface.
     */
    virtual const char * name() const;


    /**
     * Implementation of MatrixTransform interface.
     * Computes the power spectrum using the given gamma parameter (exponent).
     */
    virtual linalg::Matrix * transform(linalg::Matrix* m) const;


    /** 
     * Implementation of MatrixTransform interface.
     * Reverts the power spectrum transformation.
     */
    virtual linalg::Matrix * inverseTransform(linalg::Matrix* m) const;


    /** 
     * Sets the gamma parameter (exponent).
     */
    void setGamma(double gamma);


private:
    // Applies pow to a Matrix.
    void powMatrix(linalg::Matrix* m, double exp) const;

    double _gamma;
};


/**
 * @}
 */


inline void PowerTransform::setGamma(double gamma)
{
    _gamma = gamma;
}


} // namespace transforms


} // namespace blissart


#endif // __BLISSART_POWER_TRANSFORM_H__
