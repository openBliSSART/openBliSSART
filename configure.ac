#
# $Id: configure.ac 860 2009-06-10 12:57:12Z alex $
#
# This file is part of openBliSSART.
#
# Copyright (c) 2007-2009, Alexander Lehmann <lehmanna@in.tum.de>
#                          Felix Weninger <felix@weninger.de>
#                          Bjoern Schuller <schuller@tum.de>
#
# Institute for Human-Machine Communication
# Technische Universitaet Muenchen (TUM), D-80333 Munich, Germany
#
# openBliSSART is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 2 of the License, or (at your option) any later
# version.
#
# openBliSSART is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# openBliSSART.  If not, see <http://www.gnu.org/licenses/>.
#

AC_PREREQ(2.13)
AC_INIT([openBliSSART], [1.0], [lehmanna@in.tum.de])
AC_CONFIG_SRCDIR([src/septool/main.cpp])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_AUX_DIR([acbuild])
AC_CONFIG_MACRO_DIR([m4])
dnl AC_PRESERVE_HELP_ORDER

AM_INIT_AUTOMAKE([foreign -Wall])

# Include custom macros.
sinclude([acbuild/m4/ax_check_fftw3.m4])
sinclude([acbuild/m4/ax_check_poco.m4])
sinclude([acbuild/m4/ax_check_qt.m4])
sinclude([acbuild/m4/ax_check_sdl.m4])
sinclude([acbuild/m4/ax_check_sdl_sound.m4])
sinclude([acbuild/m4/ax_check_cblas.m4])
sinclude([acbuild/m4/ax_enable_debug.m4])
dnl sinclude([acbuild/m4/ax_get_os_and_machine_type.m4])

# Provide support for debugging (--enable-debug).
AX_ENABLE_DEBUG

# Checks for programs.
AC_PROG_AWK
AC_PROG_CC
AC_PROG_CPP
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_PROG_LN_S
AC_PROG_MAKE_SET

# Checks for libraries.
AX_CHECK_FFTW3(:, AC_ERROR([Need fftw3!]))
AX_CHECK_POCO([yes], :, AC_ERROR([Need Poco Enterprise v1.3.4+!]))
# TODO: Actually, Qt is only neccessary for the building the browser application.
#       This should be taken into account as someone might only want to build the
#       libraries and/or command-line tools.
AX_CHECK_QT([4.4.2], :, AC_ERROR([Need Qt 4.4.2+!]))
AX_CHECK_SDL([1.2.12], :, AC_ERROR([Need SDL 1.2.12+!]))
AX_CHECK_SDL_SOUND(:, AC_ERROR([Need SDL_sound!]))
AX_CHECK_CBLAS

# Checks for header files.
AC_CHECK_HEADERS([float.h stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_REALLOC
AC_CHECK_FUNCS([floor sqrt strrchr atexit memset pow strchr])

dnl The following lines have been disabled because they make debugging impossible
dnl and also don't improve the speed as much as intended. I'll leave them here
dnl for possible later use (<lehmanna@in.tum.de> 2008/02/22).
dnl
dnl # Determine OS and machine type, then adapt optimization flags
dnl AC_ARG_ENABLE([optimizations],
dnl               AS_HELP_STRING([--enable-optimizations],
dnl                              [enable machine specific optimizations (default)]),
dnl               optimizations="$enableval", optimizations="yes")
dnl if test "x$optimizations" = "xyes"; then
dnl     AX_GET_OS_AND_MACHINE_TYPE
dnl     AC_MSG_NOTICE([*** enabling optimizations])
dnl     CXXFLAGS="${CXXFLAGS//-g /}"
dnl     CXXFLAGS="${CXXFLAGS//-O2 /}"
dnl     case "$TYPE_MACHINE" in
dnl         i386)
dnl             CXXFLAGS="$CXXFLAGS -O3 -march=prescott -mfpmath=sse"
dnl             ;;
dnl         powerpc)
dnl             if test "x$TYPE_OS" = "xDarwin"; then
dnl                 # Looks like we've found ourselves a rotten apple here
dnl                 CXXFLAGS="$CXXFLAGS -O3 -fast"
dnl             else
dnl                 CXXFLAGS="$CXXFLAGS -O3"
dnl             fi
dnl             ;;
dnl     esac
dnl fi

# Set the standard installation prefix to /usr/local/blissart.
AC_PREFIX_DEFAULT([/usr/local/blissart])

# Generate the Makefiles.
AC_CONFIG_FILES([Makefile
                 doc/Makefile
                 src/LibAudio/Makefile
                 src/LibFeature/Makefile
                 src/LibFramework/Makefile
                 src/LibICA/Makefile
                 src/LibLinAlg/Makefile
                 src/LibNMF/Makefile
                 src/Makefile
                 src/Testsuite/Makefile
                 src/cvtool/Makefile
                 src/fextool/Makefile
                 src/icatool/Makefile
                 src/include/Makefile
                 src/septool/Makefile
                 src/labeller/Makefile
                 src/export/Makefile
                 src/cleanup/Makefile
                 src/nmdtool/Makefile
                 src/browser/Makefile])
AC_OUTPUT
