//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2011, Alexander Lehmann <lehmanna@in.tum.de>
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


#ifndef __BLISSART_NMF_DECONVOLVERKERNELS_H__
#define __BLISSART_NMF_DECONVOLVERKERNELS_H__


#include <blissart/linalg/common.h>


// Declare functions that apply CUDA kernels for NMF
// (pure C code, no CUDA extensions).


namespace blissart {


namespace nmf {


namespace gpu {


/**
 * Applies the multiplicative update for the W matrix in minimization
 * of the KL divergence.
 */
void apply_KLWUpdate(const Elem* w, const Elem *wUpdateNum, 
    const Elem *hRowSums, Elem* updatedW, int rows, int cols);
    
    
/**
 * Computes the multiplicative update for the H matrix in minimization
 * of the KL divergence.
 */
void compute_KLHUpdate(const Elem *hUpdateNum, 
    const Elem *wColSums, Elem* hUpdate, int p, int rows, int cols);
    
    
} // namespace gpu


} // namespace nmf


} // namespace blissart


#endif // __BLISSART_NMF_DECONVOLVERKERNELS_H__
