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

##.cu.o:
##	/usr/local/cuda/bin/nvcc -o $@ -c $<

##.cu.lo:
##	$(LIBTOOL) --tag=CXX --mode=compile /usr2/home/wen/bin/nvccf $(NVCC_OPTIONS) -DHAVE_CONFIG_H --compile -o $@ $^ -shared "--compiler-##options -fPIC -DPIC" -I../include -I../..

lib_LTLIBRARIES = libNMF.la

libNMF_la_LIBADD = $(top_srcdir)/src/LibLinAlg/libLinAlg.la

## right now the CUDA_CPPFLAGS are only needed because of the way CUBLAS is
## initialized (with a static member for the handle in GPUMatrix), remove
## if this changes!
##libNMF_la_CPPFLAGS = -I../include @CUDA_CPPFLAGS@ 
libNMF_la_CPPFLAGS = -I../include

libNMF_la_CXXFLAGS = -fPIC

libNMF_la_SOURCES = Deconvolver.cpp randomGenerator.cpp

##if HAVE_CUDA
##libNMF_la_SOURCES += DeconvolverKernels.cu
##endif

EXTRA_DIST = LibNMF.vcproj
