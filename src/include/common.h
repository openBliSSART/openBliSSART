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


/**
 * \mainpage 
 *
 * This is the documentation of the openBliSSART framework API.
 * It is intended for developers that want to use parts of openBliSSART,
 * such as NMF algorithms or the database-driven storage module,
 * in their own applications.
 *
 * If you only want to use the openBliSSART source separation toolbox, please
 * refer to the reference manual, which is contained in LaTeX format in the 
 * openBliSSART source distribution.
 *
 * Keep in mind that openBliSSART is licensed under the GPL, which means that
 * applications using openBliSSART must be provided as free software as well.
 *
 * To get an overview of the libraries and classes that openBliSSART provides,
 * it is recommended to have a look at the Modules page first.
 */


// DLL import/export specifications for Windows build
#if defined(_WIN32) || defined(_MSC_VER)
# ifdef BUILD_LIBNMF
#  define LibNMF_API __declspec(dllexport)
# else
#  define LibNMF_API __declspec(dllimport)
# endif
# ifdef BUILD_LIBICA
#  define LibICA_API __declspec(dllexport)
# else
#  define LibICA_API __declspec(dllimport)
# endif
# ifdef BUILD_LIBFEATURE
#  define LibFeature_API __declspec(dllexport)
# else
#  define LibFeature_API __declspec(dllimport)
# endif
# ifdef BUILD_LIBAUDIO
#  define LibAudio_API __declspec(dllexport)
# else
#  define LibAudio_API __declspec(dllimport)
# endif
# ifdef BUILD_LIBFRAMEWORK
#  define LibFramework_API __declspec(dllexport)
# else
#  define LibFramework_API __declspec(dllimport)
# endif
# ifdef BUILD_LIBLINALG
#  define LibLinAlg_API __declspec(dllexport)
# else
#  define LibLinAlg_API __declspec(dllimport)
# endif
#else
# define LibNMF_API
# define LibICA_API
# define LibFeature_API
# define LibAudio_API
# define LibFramework_API
# define LibLinAlg_API
#endif


// Debug Assertions
#ifdef _DEBUG
# include <cassert>
# define debug_assert(cond) assert(cond)
#else
# define debug_assert(cond)
#endif


// Support config.h
#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
