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


#include <blissart/QueuedTaskManager.h>
#include <blissart/BasicTaskNotification.h>
#include <blissart/BasicApplication.h>

#include <Poco/Exception.h>
#include <Poco/NumberFormatter.h>


using namespace Poco;
using namespace std;


namespace blissart {


QueuedTaskManager::QueuedTaskManager(unsigned int minThreads,
                                     unsigned int maxThreads) :
    _maxThreads(maxThreads),
    _threadPool(new ThreadPool((int) minThreads, (int) (maxThreads + 1))),
    _logger(BasicApplication::instance().logger())
{
    // It is neccessary to initialize the ThreadPool's capacity with the
    // maximum # of threads plus 1 because otherwise the ThreadPool wouldn't
    // let us start a new thread when a BasicTask has finished and the # of
    // active threads in the ThreadPool is at the upper limit.
    // This is caused by the fact that when a BasicTask calls taskCancelled,
    // taskFailed or taskFinished, the associated thread is of course still
    // running (remember that the _thread itself_ actually executes the code
    // of the above mentioned methods).

    _logger.debug("QueuedTaskManager is alive with " +
                  NumberFormatter::format(minThreads) + "/" +
                  NumberFormatter::format(maxThreads) + " (min/max) threads."
    );
}


QueuedTaskManager::~QueuedTaskManager()
{
    _logger.debug("QueuedTaskManager shutting down.");

    // TODO: Add a _shuttingDown flag? This way, misbehaved addition and
    // execution of new tasks could be prevented in an obvious way.

    if (_threadPool) {
        _threadPool->joinAll();
        delete _threadPool;
        _threadPool = NULL;
    }
}


void QueuedTaskManager::addTask(BasicTask &task)
{
    FastMutex::ScopedLock lock(_mutex);

    // Assert that the task is ready.
    if (task.state() != BasicTask::TASK_READY) {
        throw Poco::RuntimeException("Only tasks with state TASK_READY "
                                     "can be added!");
    }

    // Assert that the task has not already been added.
    if (_pendingTasks.find(&task))
        throw Poco::RuntimeException("Task cannot be added more than once!");

    task.setTaskManager(this);
    _pendingTasks.insert(0, &task);
    _emptyQueueEvent.reset();

	startNextPendingTask();
}


// TODO: There's a lot of double code here. This should be condensed somehow.
void QueuedTaskManager::addTask(BasicTask &task, const TaskDeps &taskDeps)
{
    FastMutex::ScopedLock lock(_mutex);

    // Assert that the task is ready.
    if (task.state() != BasicTask::TASK_READY) {
        throw Poco::RuntimeException("Only tasks with state TASK_READY "
                                     "may be added!");
    }

    // Assert that the task has not already been added.
    if (_pendingTasks.find(&task))
        throw Poco::RuntimeException("Task cannot be added more than once!");

    // Assert that the task doesn't depend on itself.
    if (taskDeps.find(&task) != taskDeps.end())
        throw Poco::RuntimeException("The task cannot depend on itself!");

    task.setTaskManager(this);
    if (!taskDeps.empty()) {
        // First make sure that the list of tasks dependencies contains only
        // tasks that actually exist. If they don't, this is probably not a
        // programmer's fault, but that task has already finished in some way.
        for (TaskDeps::const_iterator it = taskDeps.begin();
             it != taskDeps.end(); ++it)
        {
            BasicTask *ct = *it;
            if (_pendingTasks.find(ct) || _activeTasks.find(ct) != _activeTasks.end()) {
                // Everything's ok, so we add the new task to the list of
                // tasks that will depend on the iterated task.
                TaskDeps &v = _dependencies[ct];
                v.insert(&task);
            }
        }
    }

    // Now push the new task on the heap of pending tasks. Its key will be the
    // number of tasks that it depends on.
    _pendingTasks.insert((int)taskDeps.size(), &task);
    _emptyQueueEvent.reset();

    startNextPendingTask();
}


void QueuedTaskManager::removeTask(BasicTask &task)
{
    FastMutex::ScopedLock lock(_mutex);

    if (_activeTasks.find(&task) != _activeTasks.end())
        throw Poco::RuntimeException("Active tasks cannot be removed!");

    _pendingTasks.remove(&task);
    if (_pendingTasks.empty())
        _emptyQueueEvent.set();
}


void QueuedTaskManager::joinAll()
{
    _logger.debug("QueuedTaskManager joining active tasks.");

    _emptyQueueEvent.wait();
    _threadPool->joinAll();
}


void QueuedTaskManager::cancelAll()
{
    FastMutex::ScopedLock lock(_mutex);

    _logger.debug("QueuedTaskManager cancelling all tasks.");

    // First remove any pending tasks in order to prevent new tasks from
    // starting once the active tasks have been cancelled.
    _pendingTasks.clear();
    _emptyQueueEvent.set();

    // Now cancel the currently active tasks.
    for (set<BasicTask *>::iterator it = _activeTasks.begin();
        it != _activeTasks.end(); ++it)
    {
        (*it)->cancel();
    }
}


void QueuedTaskManager::addObserver(const AbstractObserver &observer)
{
    _notificationCenter.addObserver(observer);
}


void QueuedTaskManager::removeObserver(const AbstractObserver &observer)
{
    _notificationCenter.removeObserver(observer);
}


void QueuedTaskManager::setNumThreads(unsigned int minT, unsigned int maxT)
{
    FastMutex::ScopedLock lock(_mutex);

    _logger.debug("QueuedTaskManager reinitializing with " +
            NumberFormatter::format(minT) + "/" +
            NumberFormatter::format(maxT) + " (min/max) threads."
    );

    if (!_pendingTasks.empty() || !_activeTasks.empty()) {
        throw Poco::RuntimeException("The number of threads cannot be set "
                                     "while there are any pending or active "
                                     "threads.");
    }

    _threadPool->joinAll();
    delete _threadPool;
    _threadPool = new ThreadPool(minT, maxT + 1 /* see constructor */);
    _maxThreads = maxT;
}


void QueuedTaskManager::postNotification(BasicTaskNotification *nf)
{
    // The notification center takes ownership of nf from now on.
    _notificationCenter.postNotification(nf);
}


void QueuedTaskManager::startNextPendingTask(BasicTask *proposedTask)
{
#if !defined(_WIN32) && !defined(_MSC_VER)
    // Assure that the _mutex has already been locked.
    debug_assert(!_mutex.tryLock());
#endif

    if (_activeTasks.size() >= _maxThreads)
        return;

    if (_pendingTasks.empty()) {
        // No more pending tasks, hence signal a corresponding event.
        _emptyQueueEvent.set();
    }
    else if (_pendingTasks.minKey() > 0) {
        // None of the tasks is ready due to dependencies. This can happen
        // if a task was cancelled or has failed. Users of the QueuedTaskManager
        // need to make sure that they handle this situation appropriately
        // through the observation mechanism.
        return;
    }
    else if (_threadPool->available() > 0) {
        // Fetch the next task for execution.
        BasicTask *newTask;
        if (proposedTask) {
            newTask = proposedTask;
            _pendingTasks.remove(proposedTask);
        } else
            newTask = _pendingTasks.extractMin();

        // DBG: Make sure that it really _is_ ready.
        debug_assert(newTask->state() == BasicTask::TASK_READY);

        // Insert the new task in the set of active tasks and then start it.
        newTask->setState(BasicTask::TASK_STARTING);
        _activeTasks.insert(newTask);
        _threadPool->start(*newTask);
    }
}


BasicTask* QueuedTaskManager::removeActiveTask(BasicTask *task)
{
#if !defined(_WIN32) && !defined(_MSC_VER)
    // Assure that the _mutex has already been locked.
    debug_assert(!_mutex.tryLock());
#endif

    // First remove the task from the set of active tasks.
    _activeTasks.erase(task);

    BasicTask* nextTaskToRun = NULL;

    // Only update the keys of the tasks that possibly depend on this one
    // if the task has actually finished. This is to avoid the starting of
    // possibly dependent threads in case this one was cancelled or failed.
    if (task->state() == BasicTask::TASK_FINISHED) {
        // See if any tasks depend on this one and update their keys
        // accordingly.
        map<BasicTask *, TaskDeps>::iterator it = _dependencies.find(task);
        if (it != _dependencies.end()) {
            TaskDeps &td = it->second;
            for (TaskDeps::iterator tdi = td.begin(); tdi != td.end(); ++tdi) {
                int newKey = _pendingTasks.decreaseKey(*tdi, 1);
                if (newKey <= 0 && !nextTaskToRun) {
                    // So this is a task that directly depends on the completion
                    // of the task being removed and that is ready to go now.
                    nextTaskToRun = *tdi;
                }
            }
        }
    }

    // Make sure that this task's dependencies (if any) are removed from the
    // respective map.
    _dependencies.erase(task);

    return nextTaskToRun;
}


void QueuedTaskManager::taskProgressChanged(BasicTask *task)
{
    // No more than one progress notification every 500 milliseconds.
    if (_lastProgressNotification.isElapsed(500000)) {
        postNotification(
            new BasicTaskNotification(task,
                BasicTaskNotification::ProgressChanged)
        );
        _lastProgressNotification.update();
    }
}


void QueuedTaskManager::taskCancelled(BasicTask *task)
{
    _mutex.lock();
    {
        BasicTask *proposedTask = removeActiveTask(task);
        startNextPendingTask(proposedTask);
    }
    _mutex.unlock();

    postNotification(
        new BasicTaskNotification(task, BasicTaskNotification::Cancelled)
    );
}


void QueuedTaskManager::taskFailed(BasicTask *task, const Poco::Exception &ex)
{
    _logger.error(task->nameAndTaskID() + " exception: " + ex.displayText());

    _mutex.lock();
    {
        BasicTask *proposedTask = removeActiveTask(task);
        startNextPendingTask(proposedTask);
    }
    _mutex.unlock();

    postNotification(new BasicTaskFailedNotification(task, ex));
}


void QueuedTaskManager::taskFinished(BasicTask *task)
{
    _mutex.lock();
    {
        BasicTask *proposedTask = removeActiveTask(task);
        startNextPendingTask(proposedTask);
    }
    _mutex.unlock();

    postNotification(
        new BasicTaskNotification(task, BasicTaskNotification::Finished)
    );
}


} // namespace blissart
