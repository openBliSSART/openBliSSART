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

all: Makefile.qt
	$(MAKE) -f Makefile.qt $(AM_MAKEFLAGS)

clean-local: Makefile.qt
	$(MAKE) -f Makefile.qt $(AM_MAKEFLAGS) clean
	-rm -f browser

EXTRA_DIST = *.cpp *.h *.ui browser.pro browser.vcproj make_vcproj.bat

.PHONY: install-exec-local Makefile.qt
install-exec-local: all
	$(mkdir_p) $(bindir)
	$(install_sh_PROGRAM) browser $(bindir)/browser

Makefile.qt: browser.pro
if DEBUG_BUILD
	$(QMAKE) -config debug 'POCOLDFLAGS=@POCO_LDFLAGS@' 'POCOCPPFLAGS=@POCO_CPPFLAGS@' 'CPPFLAGS=-DHAVE_CONFIG_H -I../.. $(CPPFLAGS)' 'LDFLAGS=$(LDFLAGS)' -o Makefile.qt
else
	$(QMAKE) -config release 'POCOLDFLAGS=@POCO_LDFLAGS@' 'POCOCPPFLAGS=@POCO_CPPFLAGS@' 'CPPFLAGS=-DHAVE_CONFIG_H -I../.. $(CPPFLAGS)' 'LDFLAGS=$(LDFLAGS)' -o Makefile.qt
endif
