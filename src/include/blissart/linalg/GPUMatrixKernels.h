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


#ifndef __BLISSART_LINALG_GPUMATRIXKERNELS_H__
#define __BLISSART_LINALG_GPUMATRIXKERNELS_H__


// Declare functions that apply CUDA kernels (pure C code, no CUDA extensions).


namespace blissart {


namespace linalg {


namespace gpu {


/**
 * Used by GPUMatrix::add().
 */
void apply_add(const double* a, const double* b, double* c, int m, int n);


/**
 * Used by GPUMatrix::sub().
 */
void apply_sub(const double* a, const double* b, double* c, int m, int n);


/**
 * Used by GPUMatrix::elementWiseMult().
 */
void apply_mul(const double* a, const double* b, double* c, int m, int n);


/**
 * Used by GPUMatrix::elementWiseDiv().
 */
void apply_div(const double* a, const double* b, double* c, int m, int n);


/**
 * Used by GPUMatrix::elementWisePow().
 */
void apply_pow(const double* a, const double b, double* c, int m, int n);


/**
 * Used by GPUMatrix::zero().
 */
void set_to_zero(double* a, int m, int n, int startRow, int startCol, int endRow, int endCol);


} // namespace gpu


} // namespace linalg


} // namespace blissart


#endif // __BLISSART_LINALG_GPUMATRIXKERNELS_H__

