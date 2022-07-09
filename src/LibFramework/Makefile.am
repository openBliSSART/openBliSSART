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

lib_LTLIBRARIES = libFramework.la

libFramework_la_LIBADD = $(top_srcdir)/src/LibLinAlg/libLinAlg.la \
                         $(top_srcdir)/src/LibFeature/libFeature.la \
                         $(top_srcdir)/src/LibAudio/libAudio.la \
                         $(top_srcdir)/src/LibNMF/libNMF.la \
                         @POCO_LDFLAGS@ -lPocoDataSQLite -lPocoData -lPocoFoundation

##libFramework_la_CPPFLAGS = -I../include @POCO_CPPFLAGS@ @CUDA_CPPFLAGS@
libFramework_la_CPPFLAGS = -I../include @POCO_CPPFLAGS@


libFramework_la_CXXFLAGS = -fPIC

libFramework_la_SOURCES = AnovaFeatureSelector.cpp \
						  AudioObject.cpp \
						  BasicApplication.cpp \
						  BasicTask.cpp \
						  ClassificationObject.cpp \
						  ClassificationTask.cpp \
						  CleanupTask.cpp \
						  CorrelationFeatureSelector.cpp \
						  CrossValidator.cpp \
						  DataDescriptor.cpp \
						  DataSet.cpp \
						  DatabaseEntity.cpp \
						  DatabaseSubsystem.cpp \
						  FTTask.cpp \
						  Feature.cpp \
						  FeatureDescriptor.cpp \
						  FeatureExtractionTask.cpp \
						  FeatureExtractor.cpp \
						  FeatureSelector.cpp \
						  FeatureSet.cpp \
                          GnuplotWriter.cpp \
						  HTKWriter.cpp \
						  Label.cpp \
                          MatrixTransform.cpp \
                          MelFilterTransform.cpp \
						  NMDTask.cpp \
                          PowerTransform.cpp \
						  Process.cpp \
						  ProgressInterface.cpp \
						  QueuedTaskManager.cpp \
						  Response.cpp \
						  SeparationTask.cpp \
                          SlidingWindowTransform.cpp \
                          SpectralSubtractionTransform.cpp \
						  SVMModel.cpp \
						  StorageSubsystem.cpp \
						  TargetedDeconvolver.cpp \
						  ThreadedApplication.cpp \
						  WindowFunctions.cpp \
						  exportDataSet.cpp \
						  libsvm/svm.cpp \
						  TypeHandler.h

EXTRA_DIST = LibFramework.vcproj
