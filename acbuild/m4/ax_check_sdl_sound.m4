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


dnl AX_CHECK_SDL_SOUND([ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl Test for SDL_sound, and define SDL_SOUND_LIBS.
AC_DEFUN([AX_CHECK_SDL_SOUND],
[
    dnl First check for the existence of SDL_sound.h
    AC_CHECK_HEADER([SDL2/SDL_sound.h], [], [NO_SDL_SOUND=yes])

    dnl Now check for the presence of the symbol Sound_NewSampleFromFile in the
    dnl SDL_sound library.
    if test -z "$NO_SDL_SOUND"; then
        AC_LANG_PUSH([C])
        AC_CHECK_LIB([SDL2_sound], [Sound_NewSampleFromFile], :, [NO_SDL_SOUND=yes])
        AC_LANG_POP([C])
    fi

    dnl The result depends on $NO_SDL_SOUND
    if test -z "$NO_SDL_SOUND"; then
        AC_SUBST([SDL_SOUND_LIBS], [-lSDL2_sound])
        ifelse([$1], [], :, [$1])
    else
        ifelse([$2], [], :, [$2])
    fi
])
