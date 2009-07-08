//
// $Id: ThreadedDialog.h 889 2009-07-01 16:12:26Z felix $
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


#ifndef __THREADEDDIALOG_H__
#define __THREADEDDIALOG_H__


#include <blissart/ThreadedApplication.h>

#include <QDialog>


namespace blissart {


class ThreadedDialog : public QDialog,
                       public ThreadedApplication::AbstractListener
{
    Q_OBJECT


public:
    /**
     * Constructs a new instance of ThreadedDialog.
     */
    ThreadedDialog(QWidget *parent = 0);


    /**
     * Destructs an instance of ThreadedDialog.
     */
    virtual ~ThreadedDialog();


protected:
    /**
     * Sets the number of threads. Actually delegates its work to the main
     * ThreadedApplication. Resets the progress of the ThreadedApplication.
     */
    inline void setNumThreads(unsigned int numThreads)
    {
        resetProgress();
        _threadedApp.setNumThreads(numThreads);
    }


    /**
     * Adds the given task. Actually delegates its work to the main
     * ThreadedApplication.
     */
    inline void addTask(BasicTaskPtr task)
    {
        _threadedApp.addTask(task);
    }


    /**
     * Adds the given task using the given dependencies. Actually delegates its
     * work to the main ThreadedApplication.
     */
    inline
    void addTask(BasicTaskPtr task, const QueuedTaskManager::TaskDeps &deps)
    {
        _threadedApp.addTask(task, deps);
    }


    /**
     * Cancels all tasks. Actually delegates its work to the main
     * ThreadedApplication.
     */
    inline void cancelAll()
    {
        _threadedApp.cancelAll();
    }


    /**
     * Joins all tasks. Actually delegates its work to the main
     * ThreadedApplication.
     */
    inline void joinAll()
    {
        _threadedApp.joinAll();
    }


    /**
     * Returns the overall progress. Actually delegates its work to the main
     * ThreadedApplication.
     */
    inline float progress()
    {
       return _threadedApp.progress();
    }


    /**
     * Resets the overall progress. Actually delegates its work to the main
     * ThreadedApplication.
     */
    inline void resetProgress()
    {
        _threadedApp.resetProgress();
    }


    /**
     * Implementation of AbstractListener::removeTask. Can be overriden by
     * subclasses in case they are interested in knowing once a task is being
     * removed, e.g. when it has finished, failed or has been cancelled.
     *
     * It is advised that deriving subclasses should call this base class
     * implementation of removeTask for future compatibility.
     */
    virtual void removeTask(const BasicTaskPtr &)
    {
        /* Intentionally left blank */
    }


    /**
     * Sleeps for the given amount of milliseconds. This function exists for
     * convenience only.
     */
    inline void yield(unsigned int msecs)
    {
        _auxWaitEvent.reset();
        _auxWaitEvent.tryWait(msecs);
    }


    /**
     * Wait for completion of all tasks and provide the user with some progress
     * information. Also perform periodical checks whether the user cancelled
     * the operation.
     * @param  message              the message to be displayed during operation
     * @return                      whether the user cancelled the operation
     */
    bool waitForCompletion(const std::string &message);


    // We want to be friends with ThreadedApplication.
    friend class ThreadedApplication;


private:
    ThreadedApplication&  _threadedApp;
    Poco::Event           _auxWaitEvent;
};


} // namespace blissart


#endif
