//
// $Id: ProgressObserverAdapter.h 855 2009-06-09 16:15:50Z alex $
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


#ifndef __BLISSART_PROGRESSOBSERVERADAPTER_H__
#define __BLISSART_PROGRESSOBSERVERADAPTER_H__


#include <blissart/ProgressObserver.h>


namespace blissart {


/**
 * FIXME: Document me!
 */
template<class T>
class LibFramework_API ProgressObserverAdapter : public ProgressObserver
{
public:
    typedef void (T::*Callback)(float);


    ProgressObserverAdapter(T* object, Callback method) :
        _object(object),
        _separationMethod(method)
    {
    }


    virtual void progressChanged(float progress)
    {
        (_object->*_separationMethod)(progress);
    }


private:
    T*       _object;
    Callback _separationMethod;
};


} // namespace blissart


#endif // __BLISSART_PROGRESSOBSERVERADAPTER_H__
