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


#ifndef __BLISSART_GNUPLOTWRITER_H__
#define __BLISSART_GNUPLOTWRITER_H__


#include <blissart/BasicTask.h>
#include <blissart/FTTask.h>
#include <blissart/WindowFunctions.h>
#include <blissart/ClassificationObject.h>


namespace blissart {


/**
 * Writes a matrix in Gnuplot format. Column-wise or row-wise output can
 * be chosen.
 */
class LibFramework_API GnuplotWriter
{
public:
    /**
     * Writes a matrix in Gnuplot format. 
     * @param m a reference to the Matrix object to be written
     * @param file file name for output
     * @param columnWise a flag indicating whether the matrix should be
     *                   output column-wisely or not. Note that "column-wisely"
     *                   results in the rows of the Matrix being the plotted
     *                   observations, and vice versa.
     */
    static void writeMatrixGnuplot(const linalg::Matrix& m, 
                                   const std::string& file,
                                   bool columnWise);


private:
    // Forbid construction of objects of this class.
    GnuplotWriter();
};


} // namespace blissart


#endif // __BLISSART_GNUPLOTWRITER_H__
