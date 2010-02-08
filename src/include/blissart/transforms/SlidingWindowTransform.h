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


#ifndef __BLISSART_SLIDING_WINDOW_TRANSFORM_H__
#define __BLISSART_SLIDING_WINDOW_TRANSFORM_H__


#include <common.h>
#include <blissart/MatrixTransform.h>


namespace blissart {


namespace transforms {


/**
 * Performs a "sliding window" transformation (i.e. create multiple-frame
 * observations.)
 * Implements MatrixTransform interface.
 */
class LibFramework_API SlidingWindowTransform : public MatrixTransform
{
public:
    /**
     * Default constructor. Constructs a SlidingWindowTransform with the
     * parameters given in the Application's LayeredConfiguration.
     */
    SlidingWindowTransform();


    /**
     * Implementation of MatrixTransform interface.
     */
    virtual const char * name();


    /**
     * Applies a "sliding window" to the given spectrum, i.e. multiple
     * columns are concatenated into single columns.
     */
    virtual linalg::Matrix * transform(linalg::Matrix* m);


    /** 
     * Implementation of MatrixTransform interface.
     * Reverts the sliding window transformation.
     */
    virtual linalg::Matrix * inverseTransform(linalg::Matrix* m);


    /** 
     * Sets the "frame rate" (in matrix columns).
     */
    void setFrameRate(int frameRate);


    /**
     * Sets the "frame size" (number of matrix columns to combine).
     */
    void setFrameSize(int frameSize);


private:
    int _frameRate, _frameSize;
};


inline void SlidingWindowTransform::setFrameRate(int frameRate)
{
    _frameRate = frameRate;
}


inline void SlidingWindowTransform::setFrameSize(int frameSize)
{
    _frameSize = frameSize;
}


} // namespace transforms


} // namespace blissart


#endif // __BLISSART_SLIDING_WINDOW_TRANSFORM_H__
