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


#ifndef __BLISSART_BASICTASK_H__
#define __BLISSART_BASICTASK_H__


#include <common.h>

#include <Poco/Runnable.h>
#include <Poco/Logger.h>
#include <Poco/AutoPtr.h>

#include <blissart/ProgressInterface.h>

#include <sstream>


// Forward declaration
namespace Poco { class Logger; }


namespace blissart {


// Forward declaration
class QueuedTaskManager;


/**
 * \addtogroup framework
 * @{
 */

/**
 * Abstract base class for all tasks.
 */
class LibFramework_API BasicTask : public Poco::Runnable,
                                   public Poco::RefCountedObject,
                                   public ProgressInterface
{
public:
    typedef enum { TASK_READY,
                   TASK_STARTING,
                   TASK_RUNNING,
                   TASK_CANCELLED,
                   TASK_FINISHED,
                   TASK_FAILED
    } TaskState;
    
    
    /**
     * Constructs a new instance of BasicTask for the given name and assigns a
     * unique id to it.
     * @param  name             the desired name
     */
    BasicTask(const std::string &name);
    
    
    /**
     * Runs this task by calling runTask().
     * Note that this method must NOT be overwritten because it does some
     * exception handling and task manager related stuff.
     */
    virtual void run();


    /**
     * Returns then name of the task.
     */
    inline std::string name() const;


    /**
     * Returns the unique id of this task.
     */
    inline int taskID() const;


    /**
     * Returns a string comprised the task's name plus it's id.
     */
    inline std::string nameAndTaskID() const;
    
    
    /**
     * Returns the task's state.
     */
    inline TaskState state() const;
    
    
    /**
     * Cancels this tasks, i.e. sets it's cancel flag.
     */
    inline void cancel();
    
    
    /**
     * Returns whether this task has been cancelled.
     */
    inline bool isCancelled() const;


protected:
    virtual void onProgressChanged();

    // Forbid instantiation on the stack.
    virtual ~BasicTask();


    /**
     * This method is called once the thread has been started. It must be
     * overwritten by subclasses. It's safe to throw any exceptions as they are
     * handled by the caller.
     */
    virtual void runTask() = 0;
    
    
    /**
     * Returns a reference to the application's logger.
     * @return                  a reference to the application's logger
     */
    Poco::Logger& logger() const;


private:
    /**
     * Sets this task's state.
     * @param  state            the new state
     */
    inline void setState(TaskState state);

    
    /**
     * Sets the task manager for this task.
     * @param  taskManager      a pointer to the QueuedTaskManager
     */
    void setTaskManager(QueuedTaskManager *taskManager);


    /**
     * setState() and setTaskManager() are called from the QueuedTaskManager.
     */
    friend class QueuedTaskManager;


    const int          _taskID;
    const std::string  _name;
    TaskState          _state;
    bool               _cancelled;
    QueuedTaskManager* _taskManager;
};


typedef Poco::AutoPtr<BasicTask> BasicTaskPtr;


/**
 * @}
 */


// Inlines


inline std::string BasicTask::name() const
{
    return _name;
}


inline int BasicTask::taskID() const
{
    return _taskID;
}


inline std::string BasicTask::nameAndTaskID() const
{
    std::stringstream ss;
    ss << _name << ' ' << _taskID;
    return ss.str();
}


inline void BasicTask::setState(TaskState state)
{
    _state = state;
}


inline BasicTask::TaskState BasicTask::state() const
{
    return _state;
}


inline void BasicTask::cancel()
{
    _cancelled = true;
}


inline bool BasicTask::isCancelled() const
{
    return _cancelled;
}


} // namespace blissart


#endif // __BLISSART_BASICTASK_H__
