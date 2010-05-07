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


#ifndef __BLISSART_QUEUEDTASKMANAGER_H__
#define __BLISSART_QUEUEDTASKMANAGER_H__


#include <blissart/BasicTask.h>
#include "MinHeap.h"

#include <Poco/ThreadPool.h>
#include <Poco/Mutex.h>
#include <Poco/RWLock.h>
#include <Poco/Event.h>
#include <Poco/NotificationCenter.h>
#include <Poco/Timestamp.h>

#include <set>
#include <map>


namespace blissart {


// Forward declaration
class BasicTaskNotification;


/**
 * \addtogroup framework
 * @{
 */

/**
 * Serves as a task manager in terms of managing a queue of ready tasks and
 * executing them as soon as any threads become available. Notifies registered
 * observers upon certain events.
 */
class LibFramework_API QueuedTaskManager : public Poco::RefCountedObject
{
public:
    /**
     * Constructs a new instance of QueuedTaskManager. Initializes the internal
     * ThreadPool with the given minimum and maximum numbers of threads.
     * @param  minThreads       the minimum # of threads
     * @param  maxThreads       the maximum capacity of threads
     */
    QueuedTaskManager(unsigned int minThreads, unsigned int maxThreads);


    /**
     * Destructs an instance of QueuedTaskManager and frees the associated
     * ThreadPool.
     */
    virtual ~QueuedTaskManager();


    /**
     * Adds a BasicTask to the tasks queue. The task is started as soon as a
     * thread is available. The task manager doesn't assume ownership of the
     * task, thus the creator of the task is responsible for its deletion.
     * @param  task             the task to be added
     */
    void addTask(BasicTask &task);


    // Note: As a matter of fact, containers of references are forbidden in C++.
    typedef std::set<BasicTask *> TaskDeps;


    /**
     * Adds a BasicTasks to the tasks queue. The task is started as soon as a
     * thread is available _and_ all the tasks specified in taskDeps have
     * ended (no matter if they were cancelled or just failed or finished).
     * The task manager doesn't assume ownership of the task, thus the creator
     * of the task is responsible for its deletion.
     * @param  task             the task to be added
     * @param  taskDeps         a vector of BasicTasks on whose completion
     *                          the task to be added depends on
     */
    void addTask(BasicTask &task, const TaskDeps &taskDeps);


    /**
     * Removes the given non-active(!) task from the tasks queue.
     * @param  task             the task to be removed
     */
    void removeTask(BasicTask &task);


    /**
     * Waits for the completion of all tasks in the queue.
     */
    void joinAll();


    /**
     * Cancels all pending tasks.
     */
    void cancelAll();


    /**
     * Adds the given observer.
     * @param  observer         the observer (use the Poco::Observer template)
     */
    void addObserver(const Poco::AbstractObserver &observer);


    /**
     * Removes the given observer.
     * @param  observer         the observer (use the Poco::Observer template)
     */
    void removeObserver(const Poco::AbstractObserver &observer);


    /**
     * Sets the number of threads. As a side-effect, a new Poco::ThreadPool
     * is created. Calling this method is usually not neccessary as the
     * ThreadPool is created by the constructor.
     * Note that setting the number of threads is only allowed when there are
     * no pending or active threads.
     * @param  minThreads       the minimum # of threads
     * @param  maxThreads       the maximum capacity of threads
     */
    void setNumThreads(unsigned int minThreads, unsigned int maxThreads);


protected:
    /**
     * Posts the given notification using the internal NotificationCenter.
     */
    void postNotification(BasicTaskNotification *nf);


private:
    /**
     * Starts the next pending task if a thread is available and removes it from
     * the queue. If proposedTask is specified, then start this one instead of
     * taking an arbitrary task from the queue.
     */
    void startNextPendingTask(BasicTask *proposedTask = NULL);


    /**
     * Removes the given task from the _activeTasks list and, if specified,
     * updates the tasks that depend on it. If, thereafter, one of the dependent
     * tasks has a key <= 0, then this task is returned so that it can be
     * executed right away (this can safery huge amounts of memory if lots of
     * jobs are to be done).
     * @param  task             the task to be removed
     * @param  updateDeps       whether to update the corresponding dependencies
     */
    BasicTask *removeActiveTask(BasicTask *task, bool updateDeps);


    /**
     * Called by instances of BasicTask only.
     */
    void taskProgressChanged(BasicTask *task);


    /**
     * Called by instances of BasicTask only.
     */
    void taskCancelled(BasicTask *task);


    /**
     * Called by instances of BasicTask only.
     */
    void taskFailed(BasicTask *task, const Poco::Exception &ex);


    /**
     * Called by instances of BasicTask only.
     */
    void taskFinished(BasicTask *task);


    friend class BasicTask;


    unsigned int             _maxThreads;

    MinHeap<BasicTask *>     _pendingTasks;
    std::set<BasicTask *>    _activeTasks;
    std::map<BasicTask *, TaskDeps> _dependencies;

    Poco::ThreadPool*        _threadPool;
    Poco::FastMutex          _mutex;
    Poco::Event              _emptyQueueEvent;
    Poco::NotificationCenter _notificationCenter;
    Poco::Timestamp          _lastProgressNotification;

    Poco::Logger&            _logger;
};


typedef Poco::AutoPtr<QueuedTaskManager> QueuedTaskManagerPtr;


/**
 * @}
 */


} // namespace blissart


#endif // __BLISSART_QUEUEDTASKMANAGER_H__
