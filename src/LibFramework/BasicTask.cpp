//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2010, Alexander Lehmann <lehmanna@in.tum.de>
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


#include <blissart/BasicTask.h>
#include <blissart/QueuedTaskManager.h>
#include <blissart/BasicApplication.h>

#include <Poco/Exception.h>
#include <Poco/AtomicCounter.h>


using Poco::SystemException;
using Poco::AtomicCounter;


namespace blissart {


static AtomicCounter _nextTaskID(1);


BasicTask::BasicTask(const std::string &name) :
    _taskID(_nextTaskID++),
    _name(name),
    _state(TASK_READY),
    _cancelled(false),
    _taskManager(0)
{
}


BasicTask::~BasicTask()
{
    logger().debug(nameAndTaskID() + " deleted.");
}


void BasicTask::run()
{
    // NOTE: Don't access this instance's variables after calls to either
    // taskCancelled(), taskFailed() or taskFinished() because if these methods
    // are called, the task manager sends notifications to registered observers
    // which might then in turn delete this instance (sic).
    
    _state = TASK_RUNNING;
    
    try {
    
        logger().debug(nameAndTaskID() + " started.");

        runTask();

        if (_cancelled) {
            logger().debug(nameAndTaskID() + " cancelled.");
            _state = TASK_CANCELLED;
            if (_taskManager)
                _taskManager->taskCancelled(this);
            // Don't add anything here (see above note).
        } else {
            logger().debug(nameAndTaskID() + " finished.");
            _state = TASK_FINISHED;
            if (_taskManager)
                _taskManager->taskFinished(this);
            // Don't add anything here (see above note).
        }

    } catch (Poco::Exception &ex) {
        logger().debug(nameAndTaskID() + " failed.");
        _state = TASK_FAILED;
        if (_taskManager)
            _taskManager->taskFailed(this, ex);
        // Don't add anything here (see above note).
    } catch (std::exception &ex) {
        logger().debug(nameAndTaskID() + " failed.");
        _state = TASK_FAILED;
        if (_taskManager)
            _taskManager->taskFailed(this, SystemException(ex.what()));
        // Don't add anything here (see above note).
    } catch (...) {
        logger().debug(nameAndTaskID() + " failed.");
        _state = TASK_FAILED;
        if (_taskManager)
            _taskManager->taskFailed(this, SystemException("Unknown exception."));
        // Don't add anything here (see above note).
    }
}


void BasicTask::onProgressChanged()
{
    if (_taskManager)
        _taskManager->taskProgressChanged(this);
}


void BasicTask::setTaskManager(QueuedTaskManager *taskManager)
{
    _taskManager = taskManager;
}


Poco::Logger& BasicTask::logger() const
{
    return BasicApplication::instance().logger();
}


} // namespace blissart
