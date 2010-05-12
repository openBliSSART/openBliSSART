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


#include <blissart/ThreadedApplication.h>

#include <Poco/Exception.h>

#include <iostream>
#include <iomanip>
#include <functional>


using namespace Poco;
using namespace Poco::Util;
using namespace std;


namespace blissart {


ThreadedApplication::ThreadedApplication() :
    _numThreads(1),
    _myUniqueID(retrieveUniqueID())
{
}


ThreadedApplication::~ThreadedApplication()
{
    _listeners.clear();
}


void ThreadedApplication::addTask(BasicTaskPtr task)
{
    FastMutex::ScopedLock lock(_mutex);

    _tasks.insert(task);
    _taskManager->addTask(*task);
    registerTask(_myUniqueID, task->taskID());
}


void ThreadedApplication::addTask(BasicTaskPtr task,
                                  const QueuedTaskManager::TaskDeps &deps)
{
    FastMutex::ScopedLock lock(_mutex);

    _tasks.insert(task);
    _taskManager->addTask(*task, deps);
    registerTask(_myUniqueID, task->taskID());
}


void ThreadedApplication::removeTask(BasicTaskPtr task)
{
    FastMutex::ScopedLock lock(_mutex);

    // Notify possible listeners that this task is being removed.
    for (set<AbstractListener *>::const_iterator it = _listeners.begin();
         it != _listeners.end(); ++it)
    {
        (*it)->taskAboutToBeRemoved(task);
    }

    // Make sure that the task's progress is set to 100% in order to
    // assure correct overall progress reporting.
    setTaskProgress(_myUniqueID, task->taskID(), 1.0f);

    _taskManager->removeTask(*task);
    _tasks.erase(task);
}


void ThreadedApplication::handleTaskNotification(const BasicTaskNotificationPtr &nf)
{
    switch (nf->what()) {
        case BasicTaskNotification::ProgressChanged:
            setTaskProgress(_myUniqueID, nf->source()->taskID(), 
                nf->source()->progress());
            _progressEvent.set();
            break;

        case BasicTaskNotification::Cancelled:
        case BasicTaskNotification::Failed:
        case BasicTaskNotification::Finished:
            removeTask(nf->source());
            break;

        default:
            throw Poco::NotImplementedException("Unknown notification type.");
    }
}


void ThreadedApplication::displayProgress()
{
    static char tiller[] = {'|', '/', '-', '\\'};
    static int pos = 0;

    const float percentage = progress();
    cout << "\rWorking... "
         << setw(6) << fixed << setprecision(2)
         << std::min<float>(100.0f, 100.0f * percentage)
         << "% ";
    if (percentage < 1.0f)
        cout << tiller[pos];
    else
        cout << ' ' << endl;
    cout << flush;
    pos = (pos + 1) % 4;
}


void ThreadedApplication::waitForCompletion()
{
    // Wait for the completion of all tasks and display a small progress
    // information once in a while.
    while (progress() < 1.0f) {
        _progressEvent.tryWait(500);
        displayProgress();
    }
    // To make absolutely sure that every task has finished, we join them
    // all. Actually this isn't neccessary, but doesn't hurt either...
    _taskManager->joinAll();
}


} // namespace blissart

