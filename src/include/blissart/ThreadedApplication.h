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


#ifndef __BLISSART_THREADEDAPPLICATION_H__
#define __BLISSART_THREADEDAPPLICATION_H__


#include <common.h>

#include <blissart/BasicApplication.h>
#include <blissart/BasicTask.h>
#include <blissart/BasicTaskNotification.h>
#include <blissart/QueuedTaskManager.h>
#include <blissart/ProgressInterface.h>

#include <Poco/Mutex.h>
#include <Poco/Event.h>
#include <Poco/NObserver.h>

#include <set>
#include <map>


namespace blissart {


/**
 * A base class for applications that operate using a number of BasicTasks
 * being managed by an instance of QueuedTaskManager. The number of threads
 * defaults to 1.
 *
 * Automatically calculates (and displays) the progress of the tasks. The
 * progress can be reset for further processing.
 *
 * Listeners can register with this calls in order to get information about
 * tasks once they are being removed, e.g. when they have finished, failed or
 * have been cancelled.
 */
class LibFramework_API ThreadedApplication : public BasicApplication,
                                             public ProgressInterface
{
public:
    /**
     * Constructs a ThreadedApplication, setting the number of threads to 1.
     */
    ThreadedApplication();


    /**
     * Destructs an instance of ThreadedApplication.
     */
    virtual ~ThreadedApplication();


    /**
     * Helper class that enables other classes to be notified in case a task
     * is being removed.
     */
    class AbstractListener {
    public:
        virtual ~AbstractListener() {};
        virtual void removeTask(const BasicTaskPtr &task) = 0;
    };


    /**
     * Registers a listener for tasks being removed.
     */
    inline void registerListener(AbstractListener *callback);


    /**
     * Unregisters a listener for tasks being removed.
     */
    inline void unregisterListener(AbstractListener *callback);


protected:
    /**
     * Initializes the ThreadedApplication's QueuedTaskManager.
     * The template parameter must be set to the class that handles the
     * tasks' notifications.
     */
    template <class NotificationHandler> void initializeTaskManager();


    /**
     * Sets the number of threads. This function actually delegates its work to
     * the task manager.
     * @param  numThreads       the number of threads
     */
    inline void setNumThreads(unsigned int numThreads);


    /**
     * Returns the number of threads.
     */
    inline unsigned int numThreads() const;


    /**
     * Adds the given task to the set of tasks as well as to the task manager.
     * @param  task             the task to be added
     */
    void addTask(BasicTaskPtr task);


    /**
     * Addes the given task to the set of tasks as well as to the task manager.
     * Provides support for dependencies.
     * @param  task             the task to be added
     * @param  deps             the task's dependencies
     */
    void addTask(BasicTaskPtr task, const QueuedTaskManager::TaskDeps &deps);


    /**
     * Removes a task from the set of tasks and from the task manager.
     * Deriving classes can override this method in case they need to perform
     * something w.r.t. to the removal of the given task. However, note that
     * calling the base implementation is MANDATORY!
     * @param  task             the task to be removed
     */
    virtual void removeTask(BasicTaskPtr task);


    /**
     * Cancels all threads.
     */
    inline void cancelAll();


    /**
     * Joins all threads.
     */
    inline void joinAll();


    /**
     * Provides a basic implementation of task notification handling.
     * If a task is finished, cancelled or failed, it is deleted.
     * If a task changes its progress, global progress is updated.
     */
    void handleTaskNotification(const BasicTaskNotificationPtr& nf);


    /**
     * Displays the overall task progress in a fancy ASCII progress bar.
     */
    void displayProgress();


    /**
     * Waits for completion of all tasks and displays progress information.
     */
    void waitForCompletion();


    // We want to be friends with AbstractThreadedDialog.
    friend class ThreadedDialog;


private:
    unsigned int                   _numThreads;
    int                            _myUniqueID;

    QueuedTaskManagerPtr           _taskManager;
    Poco::FastMutex                _mutex;
    std::set<BasicTaskPtr>         _tasks;
    Poco::Event                    _progressEvent;

    std::set<AbstractListener *>   _listeners;
};


template <class NotificationHandler>
void ThreadedApplication::initializeTaskManager()
{
    if (_taskManager)
        throw Poco::RuntimeException("TaskManager already initialized.");

    _taskManager = new QueuedTaskManager(_numThreads, _numThreads);
    Poco::NObserver<NotificationHandler, BasicTaskNotification>
        obs(*this, &NotificationHandler::handleTaskNotification);
    _taskManager->addObserver(obs);
}


// Inlines


void ThreadedApplication::setNumThreads(unsigned int numThreads)
{
    if (_taskManager)
        _taskManager->setNumThreads(numThreads, numThreads);

    _numThreads = numThreads;
}


unsigned int ThreadedApplication::numThreads() const
{
    return _numThreads;
}


void ThreadedApplication::registerListener(AbstractListener *callback)
{
    _listeners.insert(callback);
}


void ThreadedApplication::unregisterListener(AbstractListener *callback)
{
    _listeners.erase(callback);
}


void ThreadedApplication::cancelAll()
{
    _taskManager->cancelAll();
    // Erasing the tasks at this point is way too early!
}


void ThreadedApplication::joinAll()
{
    _taskManager->joinAll();
    // Erasing the tasks here is fine.
    _tasks.clear();
}


} // namespace blissart


#endif // __BLISSART_THREADEDAPPLICATION_H__
