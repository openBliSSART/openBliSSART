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

bin_PROGRAMS = testsuite

testsuite_CPPFLAGS = -I../include

testsuite_LDADD = $(top_srcdir)/src/LibAudio/libAudio.la \
                  $(top_srcdir)/src/LibLinAlg/libLinAlg.la \
                  $(top_srcdir)/src/LibFramework/libFramework.la \
                  $(top_srcdir)/src/LibFeature/libFeature.la \
                  $(top_srcdir)/src/LibICA/libICA.la \
                  $(top_srcdir)/src/LibNMF/libNMF.la \
                  @POCO_LDFLAGS@ -lPocoFoundation -lPocoUtil -lPocoXML 

testsuite_SOURCES = BinaryReaderWriterTest.cpp \
					BinaryReaderWriterTest.h \
					DatabaseSubsystemTest.cpp \
					DatabaseSubsystemTest.h \
					FeatureExtractionTest.cpp \
					FeatureExtractionTest.h \
					FeatureSelectionTest.cpp \
					FeatureSelectionTest.h \
					HTKWriterTest.cpp \
					HTKWriterTest.h \
					ICATest.cpp \
					ICATest.h \
					MinHeapTest.cpp \
					MinHeapTest.h \
					MFCCTest.cpp \
					MFCCTest.h \
					MatrixTest.cpp \
					MatrixTest.h \
                    MTrTest.cpp \
                    MTrTest.h \
					NMDTest.cpp \
					NMDTest.h \
					NMFTest.cpp \
					NMFTest.h \
					PCATest.cpp \
					PCATest.h \
					SVMModelTest.cpp \
					SVMModelTest.h \
					ScalingTest.cpp \
					ScalingTest.h \
					SNMFTest.cpp \
					SNMFTest.h \
					SpectralAnalysisTest.cpp \
					SpectralAnalysisTest.h \
					StorageSubsystemTest.cpp \
					StorageSubsystemTest.h \
					TaskDepsTest.cpp \
					TaskDepsTest.h \
                    Testable.cpp \
					Testable.h \
					VectorTest.cpp \
					VectorTest.h \
					WaveTest.cpp \
					WaveTest.h \
					main.cpp
