//
// $Id: common.h 855 2009-06-09 16:15:50Z alex $
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
