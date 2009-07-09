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


dnl AX_CHECK_QT(MINIMUM-VERSION, [ACTION-IF-FOUND, [ACTION-IF-NOT-FOUND]])
dnl Checks for the presence of Qt. Sets QMAKE accordingly.
AC_DEFUN([AX_CHECK_QT],
[
    AC_REQUIRE([AC_PROG_AWK])
    AC_REQUIRE([AC_PROG_SED])

    dnl Provide --with-qt-prefix=<prefix>.
    AC_ARG_WITH([qt-prefix],
       [AS_HELP_STRING([--with-qt-prefix=<prefix>],
          [where to look for Qt @<@default=/usr/local/Trolltech/current@:>@])],
       [QT_PREFIX="$with_qt_prefix"],
       [QT_PREFIX=/usr/local/Trolltech/current]
    ) dnl AC_ARG_WITH
    
    dnl If the specified QT_PREFIX doesn't exist we try to fall back on
    dnl $QTDIR if possible.
    if test ! -d $QT_PREFIX; then
        if test -n "$QTDIR"; then
            QT_PREFIX="$QTDIR"
        else
            QT_PREFIX=
        fi
    fi

    dnl Try to find qmake.
    if test -n "$QT_PREFIX" -a -x "$QT_PREFIX/bin/qmake"; then
        QMAKE="$QT_PREFIX/bin/qmake"
    else
        dnl Look for qmake in the $PATH.
        AC_PATH_PROG([QMAKE], [qmake])
    fi

    dnl Be verbose.
    AC_MSG_CHECKING([for Qt >= $1])

    dnl Now check Qt's version.
    if test -n "$QMAKE"; then
        QT_VERSION=`$QMAKE --version | $AWK '/Qt version/ {print $[]4}'`
        QT_MAJOR=`echo $QT_VERSION | $SED 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\1/'`
        QT_MINOR=`echo $QT_VERSION | $SED 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\2/'`
        QT_MICRO=`echo $QT_VERSION | $SED 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\3/'`
        WANTED_MAJOR=`echo $1 | $SED 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\1/'`
        WANTED_MINOR=`echo $1 | $SED 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\2/'`
        WANTED_MICRO=`echo $1 | $SED 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\3/'`
        if test $QT_MAJOR -lt $WANTED_MAJOR ||
           test $QT_MAJOR -eq $WANTED_MAJOR -a \
                $QT_MINOR -lt $WANTED_MINOR ||
           test $QT_MAJOR -eq $WANTED_MAJOR -a \
                $QT_MINOR -eq $WANTED_MINOR -a \
                $QT_MICRO -lt $WANTED_MICRO
        then
            dnl This version of Qt won't work for us.
            unset QMAKE
        fi
        dnl This way or that way, we should clean up a little bit.
        QT_VERSION=
        QT_MAJOR=
        QT_MINOR=
        QT_MICRO=
        WANTED_MAJOR=
        WANTED_MINOR=
        WANTED_MICRO=
    fi

    dnl The result depends on $QMAKE.    
    if test -n "$QMAKE"; then
        AC_MSG_RESULT([yes])
        AC_SUBST(QMAKE)
        ifelse([$2], [], :, [$2])
    else
        AC_MSG_RESULT([no])
        unset QMAKE
        AC_SUBST(QMAKE)
        ifelse([$3], [], :, [$3])
    fi
]) dnl AC_DEFUN
