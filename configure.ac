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

AC_PREREQ([2.71])
AC_INIT([openBliSSART],[2.0],[lehmanna@in.tum.de],[openBliSSART])
AC_CONFIG_SRCDIR([src/septool/main.cpp])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_AUX_DIR([acbuild])
AC_CONFIG_MACRO_DIR([m4])
dnl AC_PRESERVE_HELP_ORDER

AM_INIT_AUTOMAKE([foreign -Wall])

# Include custom macros.
sinclude([acbuild/m4/ax_float.m4])
sinclude([acbuild/m4/ax_check_fftw3.m4])
sinclude([acbuild/m4/ax_check_poco.m4])
sinclude([acbuild/m4/ax_check_qt.m4])
sinclude([acbuild/m4/ax_check_sdl.m4])
sinclude([acbuild/m4/ax_check_sdl_sound.m4])
sinclude([acbuild/m4/ax_check_cblas.m4])
sinclude([acbuild/m4/ax_check_matlab.m4])
sinclude([acbuild/m4/ax_check_cuda.m4])
sinclude([acbuild/m4/ax_enable_cpu.m4])
sinclude([acbuild/m4/ax_enable_debug.m4])
dnl sinclude([acbuild/m4/ax_get_os_and_machine_type.m4])

# Provide support for debugging (--enable-debug).
AX_ENABLE_CPU
AX_ENABLE_DEBUG


# Checks for programs.
AC_PROG_AWK
AC_PROG_CC
AC_PROG_CPP
AC_PROG_CXX
AC_PROG_INSTALL
LT_INIT
AC_PROG_LN_S
AC_PROG_MAKE_SET

# Determines float type.
AX_DETERMINE_FLOAT_TYPE

# Checks for libraries.
AX_CHECK_FFTW3(:, AC_MSG_ERROR([Need fftw3!]))
AX_CHECK_POCO([no], :, AC_MSG_ERROR([Need Poco Enterprise v2.0.0+!]))
# TODO: Actually, Qt is only neccessary for the building the browser application.
#       This should be taken into account as someone might only want to build the
#       libraries and/or command-line tools.
AX_CHECK_QT([6.3.0], :, AC_MSG_ERROR([Need Qt 6.3.0+!]))
AX_CHECK_SDL([2.0.0], :, AC_MSG_ERROR([Need SDL2 2.0.0+!]))
AX_CHECK_SDL_SOUND(:, AC_MSG_ERROR([Need SDL2_sound!]))
AX_CHECK_MATLAB
#echo NO_MATLAB=$NO_MATLAB
if test "$NO_MATLAB" == "yes"; then
  AX_CHECK_CBLAS
fi
AX_CHECK_CUDA
##AM_CONDITIONAL([AM_HAVE_CUDA], [test -z "$NO_CUDA"])
#NO_CUDA="yes"
#HAVE_CUDA="no"
AM_CONDITIONAL([HAVE_CUDA], [test -z "$NO_CUDA"])
#NO_CUDA="yes"
#HAVE_CUDA="no"
#CUDA_CPPFLAGS=""
#HAVE_CUDA_TRUE=""

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
#AC_FUNC_REALLOC
AC_CHECK_FUNCS([floor sqrt strrchr atexit memset pow strchr])

# Set the standard installation prefix to /usr/local/blissart.
AC_PREFIX_DEFAULT([/usr/local/blissart])

# Generate the Makefiles.
AM_PROG_AR
LT_INIT
AC_LANG([C++])
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
                 src/benchmark/Makefile
                 src/cvtool/Makefile
                 src/fextool/Makefile
                 src/icatool/Makefile
                 src/include/Makefile
                 src/septool/Makefile
                 src/export/Makefile
                 src/exportaudio/Makefile
                 src/cleanup/Makefile
                 src/browser/Makefile])
AC_OUTPUT
