//
// $Id: ProgressInterface.h 896 2009-07-06 13:50:33Z felix $
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


#ifndef __BLISSART_PROGRESSINTERFACE_H__
#define __BLISSART_PROGRESSINTERFACE_H__


#include <common.h>
#include <map>
#include <Poco/Mutex.h>


namespace blissart {


/**
 * Provides a standard interface and implementation for thread-safe progress
 * management.
 */
class LibFramework_API ProgressInterface
{
public:
    /**
     * Returns the actual progress. It is calculated by the formula
     * (actual progress + sum of progress of all tasks) / 
     * (maximum progress + number of registered tasks).
     * @return                  the actual progress
     */
    float progress();


	virtual ~ProgressInterface() {}


protected:
    /**
     * Creates a new instance of ProgressInterface.
     */
    ProgressInterface();


    /**
     * Increases the maximum progress by the given value.
     * @param   delta           the amount to be added to the maximum progress
     */
    void incMaxProgress(float delta);


    /**
     * Increases the actual progress by the given value.
     * @param   delta           the amount to be added to the actual progress
     */
    void incTotalProgress(float delta);


    /**
     * Subclasses can override this method to perform an action when progress
     * is incremented.
     */
    virtual void onProgressChanged();


    /**
     * Resets the progress and the progress of all tasks to zero.
     */
    void resetProgress();


    /**
     * Retrieves a unique ID that must be used in subsequent calls to
     * registerTask() and setTaskProgress().
     */
    int retrieveUniqueID();


    /**
     * Registers a task. Increases the maximum progress by 1.
     * See the documentation for progress().
     */
    void registerTask(int uniqueID, int taskID);


    /**
     * Sets the progress for the given task. See the documentation for
     * progress() on how this affects the actual progress.
     */
    void setTaskProgress(int uniqueID, int taskID, float progress);

    
private:
    // Forbid copy constructor
    ProgressInterface(const ProgressInterface &);


    float                 _progress;
    float                 _maxProgress;
    typedef std::map<std::pair<int, int>, float> ProgressMap;
    ProgressMap           _progressPerTask;
    static int            _currentUniqueID;
    Poco::FastMutex       _mutex;
};


} // namespace blissart


#endif

