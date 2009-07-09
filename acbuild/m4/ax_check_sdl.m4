dnl
dnl This file is part of openBliSSART.
dnl
dnl Copyright (c) 2007-2009, Alexander Lehmann <lehmanna@in.tum.de>
dnl                          Felix Weninger <felix@weninger.de>
dnl                          Bjoern Schuller <schuller@tum.de>
dnl
dnl Institute for Human-Machine Communication
dnl Technische Universitaet Muenchen (TUM), D-80333 Munich, Germany
dnl
dnl openBliSSART is free software: you can redistribute it and/or modify it under
dnl the terms of the GNU General Public License as published by the Free Software
dnl Foundation, either version 2 of the License, or (at your option) any later
dnl version.
dnl
dnl openBliSSART is distributed in the hope that it will be useful, but WITHOUT
dnl ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
dnl FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
dnl details.
dnl
dnl You should have received a copy of the GNU General Public License along with
dnl openBliSSART.  If not, see <http://www.gnu.org/licenses/>.
dnl


dnl AX_CHECK_SDL(MINIMUM-VERSION, [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl Test for SDL, and define SDL_CFLAGS and SDL_LIBS.
AC_DEFUN([AX_CHECK_SDL],
[
    dnl Provide --with-sdl-prefix=<prefix>.
    AC_ARG_WITH([sdl-prefix],
        [AS_HELP_STRING([--with-sdl-prefix=<prefix>],
                        [Prefix where SDL is installed (optional)])],
        [SDL_PREFIX="$withval"], [SDL_PREFIX=""]
    ) dnl AC_ARG_WITH

    if test -n "$SDL_PREFIX"; then
        SDL_CONFIG=$SDL_PREFIX/bin/sdl-config
    else
        dnl The user didn't provide a prefix, so we have to look for sdl-config.
        AC_PATH_PROG(SDL_CONFIG, sdl-config, no)
    fi

    AC_MSG_CHECKING([for SDL >= $1])
    if test "$SDL_CONFIG" = "no" ; then
        NO_SDL=yes
    else
        SDL_CFLAGS=`$SDL_CONFIG $sdlconf_args --cflags`
        SDL_LIBS=`$SDL_CONFIG $sdlconf_args --libs`
        dnl Do a version check.
        SDL_MAJOR=`$SDL_CONFIG $SDL_ARGS --version | \
                   sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\1/'`
        SDL_MINOR=`$SDL_CONFIG $SDL_ARGS --version | \
                   sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\2/'`
        SDL_MICRO=`$SDL_CONFIG $sdl_config_args --version | \
                   sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\3/'`
        WANTED_MAJOR=`echo $1 | sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\1/'`
        WANTED_MINOR=`echo $1 | sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\2/'`
        WANTED_MICRO=`echo $1 | sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\3/'`
        if test $SDL_MAJOR -lt $WANTED_MAJOR ||
           test $SDL_MAJOR -eq $WANTED_MAJOR -a \
                $SDL_MINOR -lt $WANTED_MINOR ||
           test $SDL_MAJOR -eq $WANTED_MAJOR -a \
                $SDL_MINOR -eq $WANTED_MINOR -a \
                $SDL_MICRO -lt $WANTED_MICRO
        then
            NO_SDL=yes
        fi
        dnl Clean up after ourselves.
        SDL_MAJOR=
        SDL_MINOR=
        SDL_MICRO=
        WANTED_MAJOR=
        WANTED_MINOR=
        WANTED_MICRO=
    fi

    dnl The result depends on $NO_SDL.
    if test -z "$NO_SDL"; then
        AC_MSG_RESULT([yes])
        AC_SUBST(SDL_CFLAGS)
        AC_SUBST(SDL_LIBS)
        ifelse([$2], [], :, [$2])     
    else
        AC_MSG_RESULT([no])
        ifelse([$3], [], :, [$3])
    fi
])
