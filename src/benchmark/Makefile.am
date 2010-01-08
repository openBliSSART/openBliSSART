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

bin_PROGRAMS = benchmark

benchmark_CPPFLAGS = -I../include

benchmark_LDADD = $(top_srcdir)/src/LibLinAlg/libLinAlg.la \
                  $(top_srcdir)/src/LibNMF/libNMF.la \
                  $(top_srcdir)/src/LibAudio/libAudio.la \
                  @POCO_LDFLAGS@ -lPocoFoundation -lPocoUtil -lPocoXML 

benchmark_SOURCES = NMDBenchmark.cpp \
                    NMDBenchmark.h \
                    main.cpp

