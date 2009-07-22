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
 * Performs NMD by initializing the spectral matrices with components from
 * the database.
 */
class LibFramework_API TargetedDeconvolver: public nmf::Deconvolver
{
public:
    TargetedDeconvolver(const linalg::Matrix& v, unsigned int r,
        const std::vector<ClassificationObjectPtr>& clObjs);
    TargetedDeconvolver(const linalg::Matrix& v, unsigned int r,
        const std::vector<int>& clObjIDs);

private:
    void buildW(const std::vector<ClassificationObjectPtr>& clObjs);
    int getNrOfSpectra(int clObjID);
    int getNrOfSpectra(ClassificationObjectPtr clObj);
};


} // namespace blissart


#endif // __BLISSART_TARGETEDDECONVOLVER_H__
