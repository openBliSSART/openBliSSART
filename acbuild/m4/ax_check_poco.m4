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


dnl AX_CHECK_POCO([ENTERPRISE, [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]]])
dnl Checks for the presence of Poco v1.3.6+. Sets POCO_LDFLAGS and
dnl POCO_CPPFLAGS accordingly. If ENTERPRISE is set to an arbitrary value, i.e.
dnl it isn't empty, also Poco::Data is required.
AC_DEFUN([AX_CHECK_POCO],
[
    dnl Provide --with-poco-prefix=<prefix>.
    AC_ARG_WITH([poco-prefix],
        [AS_HELP_STRING([--with-poco-prefix=<prefix>],
            [where to look for POCO @<:@default=/usr/local@:>@])],
        [POCO_PREFIX="$with_poco_prefix"],
        [POCO_PREFIX=/usr/local]
    ) dnl AC_ARG_WITH

    ifelse([$1], [],
        [AC_MSG_CHECKING([for Poco (Economy)])],
        [AC_MSG_CHECKING([for Poco (Enterprise)])]
    )

    dnl Check if Poco really exists at the given location.
    dnl Backup old compiler and linker settings first.
    dnl If the source code cannot be compiled, set NO_POCO to "yes".
    OLD_LDFLAGS="$LDFLAGS"
    OLD_LIBS="$LIBS"
    OLD_CPPFLAGS="$CPPFLAGS"

    CPPFLAGS="-I$POCO_PREFIX/include"
    LDFLAGS="-L$POCO_PREFIX/lib"
    AC_LANG_PUSH([C++])
    ifelse([$1], [],
           [dnl Economy.
            LIBS="-lPocoFoundation"
            AC_LINK_IFELSE(
                [AC_LANG_PROGRAM(
                    [[
                      #include <Poco/DateTime.h>
                      #include <Poco/Foundation.h>
                    ]],
                    [[
                      Poco::DateTime dt;
                      #if !(POCO_VERSION >= 0x01030600)
                      # error Wrong Poco version!
                      #endif
                    ]])
                ], [], [NO_POCO=yes]
            )], 
           [dnl Enterprise.
            LIBS="-lPocoFoundation -lPocoSQLite -lPocoData"
            AC_LINK_IFELSE(
                [AC_LANG_PROGRAM(
                    [[
                      #include <Poco/Data/SQLite/Utility.h>
                      #include <Poco/Foundation.h>
                    ]],
                    [[
                      Poco::Data::SQLite::Utility util;
                      #if !(POCO_VERSION >= 0x01030600)
                      # error Wrong Poco version!
                      #endif
                    ]])
                ], [], [NO_POCO=yes]
            )]
    ) dnl ifelse
    AC_LANG_POP([C++])

    dnl Restore old compiler and linker settings.
    LIBS="$OLD_LIBS"
    LDFLAGS="$OLD_LDFLAGS"
    CPPFLAGS="$OLD_CPPFLAGS"

    dnl The result depends on $NO_POCO.
    if test -z "$NO_POCO"; then
        AC_MSG_RESULT([yes])
        AC_SUBST([POCO_LDFLAGS], [-L$POCO_PREFIX/lib])
        AC_SUBST([POCO_CPPFLAGS], [-I$POCO_PREFIX/include])
        ifelse([$2], [], :, [$2])
    else
        AC_MSG_RESULT([no])
        ifelse([$3], [], :, [$3])
    fi
]) dnl AC_DEFUN

