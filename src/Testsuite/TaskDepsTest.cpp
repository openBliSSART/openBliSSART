//
// $Id: TaskDepsTest.cpp 889 2009-07-01 16:12:26Z felix $
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


#include "TaskDepsTest.h"
#include <blissart/QueuedTaskManager.h>

#include <Poco/NObserver.h>
#include <Poco/Timestamp.h>

#include <iostream>
#include <iomanip>
#include <cstdlib>


using namespace blissart;
using namespace std;


namespace Testing {


class TaskDepsTestHelper : public BasicTask {
public:
    TaskDepsTestHelper(const string &name) : BasicTask(name)
    {
    }

    virtual void runTask()
    {
        Poco::Timestamp::TimeDiff delay = 50000 * (1 + rand() % 5);

        Poco::Timestamp ts;
        while (!ts.isElapsed(delay)) {
            Poco::Timestamp::TimeDiff elapsed = ts.elapsed();

            // Update progress every 50ms.
            if (elapsed >= 50000) {
                // FIXME: Adapt this to new system
                //setProgress(min<float>(1.0f, elapsed * 100.0f / delay));
                delay -= elapsed;
                ts.update();
            }
        }

        //setProgress(1.0f);
    }
};


bool TaskDepsTest::performTest()
{
    QueuedTaskManager tm(1, 1);

    Poco::NObserver<TaskDepsTest, BasicTaskNotification>
        obs(*this, &TaskDepsTest::handleTaskNotification);
    tm.addObserver(obs);

    vector<BasicTask *> tasks;
    for (int i = 0; i < 10; ++i) {
        stringstream ss;
        ss << "HelperTask " << i;
        tasks.push_back(new TaskDepsTestHelper(ss.str()));
    }

    /*
     *  0 -> 1 ->  2 -> 3 -> 4
     *  |          |
     * \./        \./
     *  5 -> 6 ->  7 -> 8    9
     */
    BasicTask *deps[10][3] = {
            { 0 },
            { tasks[0], 0 },
            { tasks[1], 0 },
            { tasks[2], 0 },
            { tasks[3], 0 },
            { tasks[0], 0 },
            { tasks[5], 0 },
            { tasks[6], tasks[2], 0 },
            { tasks[7], 0 },
            { 0 } };

    // Start all tasks.
    cout << left << setw(45) << "Starting helper tasks...";
    cout.flush();
    for (int i = 0; i < 10; ++i) {
        QueuedTaskManager::TaskDeps v;
        for (int j = 0; j < 3; ++j ) {
            BasicTask *sd = deps[i][j];
            if (!sd)
                break;
            else {
                v.insert(sd);
            }
        }
        tm.addTask(*tasks[i], v);
    }
    cout << "ok" << endl;

    // Join all tasks.
    cout << setw(45) << "Joining (this may take up to ~3 seconds)...";
    cout.flush();
    tm.joinAll();
    cout << "ok" << endl;

    // Check their timestamps relations.
    cout << setw(45) << "Checking timestamps relations...";
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 3; ++j) {
            BasicTask *sd = deps[i][j];
            if (!sd)
                break;
            else {
                if (_timestamps[tasks[i]] <= _timestamps[sd])
                    return false;
            }
        }
    }
    cout << "ok" << endl;

    // Should a task have been cancelled, failed or thrown an exception,
    // then this would be indiciated by the _globalErrFlag.
    return _globalErrFlag ? false : true;
}


void TaskDepsTest::handleTaskNotification(
        const Poco::AutoPtr<BasicTaskNotification> &nf)
{
    Poco::FastMutex::ScopedLock lock(_mutex);

    static int timestamp = 0;

    switch (nf->what()) {
    case BasicTaskNotification::ProgressChanged:
        break;

    case BasicTaskNotification::Finished:
        _timestamps[nf->source().get()] = timestamp++;
        break;

    default:
        _globalErrFlag = true;
        break;
    }
}


} // namespace Testing
