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


#ifndef __BLISSART_BASICTASKNOTIFICATION_H__
#define __BLISSART_BASICTASKNOTIFICATION_H__


#include <common.h>
#include <blissart/BasicTask.h>
#include <Poco/Notification.h>
#include <Poco/AutoPtr.h>
#include <Poco/Exception.h>


namespace blissart {


/**
 * \addtogroup framework
 * @{
 */

/**
 * A notification that is posted by a BasicTask.
 */
class LibFramework_API BasicTaskNotification : public Poco::Notification
{
public:
    /**
     * The type of notification.
     */
    typedef enum {
        ProgressChanged,
        Cancelled,
        Finished,
        Failed
    } Type;
    
    
    /**
     * Constructs a BasicTaskNotification from the given task and of the given
     * type.
     */
    BasicTaskNotification(BasicTask *source, Type what) :
        _source(source, true /* duplicate */),
        _what(what)
    {
    }


    /**
     * Returns a pointer to the task that is the source of this notification.
     */
    inline BasicTaskPtr source() const { return _source; }
    
    
    /**
     * Returns the type of this notification.
     */
    inline Type what() const { return _what; }
    
    
private:
    const BasicTaskPtr _source;
    const Type         _what;
};


/**
 * Specialization of BasicTaskNotification for failed tasks.
 * Provides access to the exception that has been thrown.
 */
class LibFramework_API BasicTaskFailedNotification : public BasicTaskNotification
{
public:
    /**
     * Constructs a BasicTaskFailedNotification from the given task,
     * with the given exception.
     */
    BasicTaskFailedNotification(BasicTask *source, const Poco::Exception& exc) :
         BasicTaskNotification(source, Failed),
         _exc(exc)
    {
    }


    /**
     * Returns a reference to the exception that has been thrown.
     */
    inline const Poco::Exception& reason() const { return _exc; }


private:
    Poco::Exception _exc;
};


typedef Poco::AutoPtr<BasicTaskNotification> BasicTaskNotificationPtr;


/**
 * @}
 */


} // namespace blissart


#endif // __BLISSART_BASICTASKNOTIFICATION_H__
