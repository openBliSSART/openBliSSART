##
## This file is part of openBliSSART.
##
## Copyright (c) 2007-2009, Alexander Lehmann <lehmanna@in.tum.de>
##                          Felix Weninger <felix@weninger.de>
##                          Bjoern Schuller <schuller@tum.de>
##
## Institute for Human-Machine Communication
## Technische Universitaet Muenchen (TUM), D-80333 Munich, Germany
##
## openBliSSART is free software: you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation, either version 2 of the License, or (at your option) any later
## version.
##
## openBliSSART is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
## FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## openBliSSART.  If not, see <http://www.gnu.org/licenses/>.
##
## TODO: introduce flag for NVCC


.cu.o:
	/usr/local/cuda/bin/nvcc -o $@ -c $<

.cu.lo:
	$(LIBTOOL) --tag=CXX --mode=compile /usr2/home/wen/bin/nvccf --compile -o $@ $^ -shared "--compiler-options -fPIC -DPIC" -I../include

lib_LTLIBRARIES = libLinAlg.la

libLinAlg_la_CPPFLAGS = -I../include @CUDA_CPPFLAGS@ @MATLAB_CPPFLAGS@

libLinAlg_la_LIBADD = @CBLAS_LIBS@ 

libLinAlg_la_CXXFLAGS = -fPIC

libLinAlg_la_SOURCES = ColVector.cpp \
                       Matrix.cpp \
                       RowVector.cpp \
                       Vector.cpp

if HAVE_CUDA
libLinAlg_la_SOURCES += GPUMatrixKernels.cu GPUMatrix.cpp
endif

EXTRA_DIST = LibLinAlg.vcproj

#$(LIBTOOL) --tag=CC --mode=compile /usr/local/cuda/bin/nvcc --compile #$(NVCC_FLAGS) -o $@ $^ -static
