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


#ifndef __BLISSART_TARGETEDDECONVOLVER_H__
#define __BLISSART_TARGETEDDECONVOLVER_H__


#include <common.h>
#include <blissart/Response.h>
#include <blissart/ClassificationObject.h>
#include <blissart/nmf/Deconvolver.h>
#include <vector>


namespace blissart {


namespace linalg { class Matrix; }


/**
 * \addtogroup framework
 * @{
 */

/**
 * Performs NMD/NMF by initializing the spectral matrices with components from
 * the database. If less components are specified than the dimensionality of
 * the factorization, the remaining components are initialized randomly.
 */
class LibFramework_API TargetedDeconvolver: public nmf::Deconvolver
{
public:
    /**
     * Creates a TargetedDeconvolver using a vector of ClassificationObjects.
     * @param v         a Matrix to factorize
     * @param r         number of components (must be at least the size of 
     *                  clObjs); if it is smaller than the size of clObjs,
     *                  the remaining columns of W are initialized randomly
     * @param clObjs    the vector of ClassificationObjects (of type
     *                  NMDComponent) to use for initialization
     * @param wGenerator  generator function for the uninitialized columns
     *                  of the spectral matrix W
     * @param hGenerator  generator function for H
     */
    TargetedDeconvolver(linalg::Matrix& v, unsigned int r,
        const std::vector<ClassificationObjectPtr>& clObjs,
        blissart::linalg::Matrix::GeneratorFunction wGenerator 
        = nmf::gaussianRandomGenerator,
        blissart::linalg::Matrix::GeneratorFunction hGenerator 
        = nmf::gaussianRandomGenerator);

    /**
     * Creates a TargetedDeconvolver using a vector of IDs referring to
     * ClassificationObjects. The corresponding ClassificationObjects are then
     * fetched from the database.
     * @param v         a Matrix to factorize
     * @param r         number of components (must be at least the size of 
     *                  clObjs); if it is smaller than the size of clObjs,
     *                  the remaining columns of W are initialized randomly
     * @param clObjIDs  vector of IDs of ClassificationObjects (of type
     *                  NMDComponent) to use for initialization
     * @param wGenerator  generator function for the uninitialized columns
     *                  of the spectral matrix W
     * @param hGenerator  generator function for H
     */
    TargetedDeconvolver(linalg::Matrix& v, unsigned int r,
        const std::vector<int>& clObjIDs,
        blissart::linalg::Matrix::GeneratorFunction wGenerator 
        = nmf::gaussianRandomGenerator,
        blissart::linalg::Matrix::GeneratorFunction hGenerator 
        = nmf::gaussianRandomGenerator);

    /**
     * Creates a TargetedDeconvolver using a vector of strings referring to
     * binary matrix files to use for initialization.
     * @param v         a Matrix to factorize
     * @param r         number of components (must be at least the size of 
     *                  clObjs); if it is smaller than the size of clObjs,
     *                  the remaining columns of W are initialized randomly
     * @param clObjs    the vector of matrix files to use for initialization
     *                  (they will be concatenated in the given order)
     * @param wGenerator  generator function for the uninitialized columns
     *                  of the spectral matrix W
     * @param hGenerator  generator function for H
     */
    TargetedDeconvolver(linalg::Matrix& v, unsigned int r,
        const std::vector<std::string>& matrices,
        blissart::linalg::Matrix::GeneratorFunction wGenerator 
        = nmf::gaussianRandomGenerator,
        blissart::linalg::Matrix::GeneratorFunction hGenerator 
        = nmf::gaussianRandomGenerator,
        bool keepConstant = true);


private:
    void buildW(const std::vector<ClassificationObjectPtr>& clObjs);
    int buildW(const std::vector<std::string>& matrices);
    int getNrOfSpectra(int clObjID);
    int getNrOfSpectra(ClassificationObjectPtr clObj);
    int getNrOfSpectra(const std::string &file);
};


/**
 * @}
 */


} // namespace blissart


#endif // __BLISSART_TARGETEDDECONVOLVER_H__
