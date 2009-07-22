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


#ifndef __TASKDEPSTEST_H__
#define __TASKDEPSTEST_H__


#include "Testable.h"
#include <blissart/BasicTask.h>
#include <blissart/BasicTaskNotification.h>

#include <Poco/AutoPtr.h>
#include <Poco/Mutex.h>

#include <map>


namespace Testing {


/**
 * Tests for the QueuedTaskManager w.r.t. inter-task dependencies.
 */
class TaskDepsTest : public Testable
{
public:
    TaskDepsTest() : Testable(), _globalErrFlag(false)
    {
    }


    bool performTest();


    inline const char *name() { return "TaskDepsTest"; }


private:
    void handleTaskNotification(
            const Poco::AutoPtr<blissart::BasicTaskNotification> &nf);


    bool            _globalErrFlag;
    Poco::FastMutex _mutex;
    std::map<blissart::BasicTask *, int> _timestamps;
};


} // namespace Testing


#endif

