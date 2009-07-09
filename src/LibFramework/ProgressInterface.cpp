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


#include <blissart/ProgressInterface.h>
#include <iostream>


namespace blissart {


int ProgressInterface::_currentUniqueID = 0;


ProgressInterface::ProgressInterface() :
    _progress(0.0f),
    _maxProgress(0.0f)
{
}


void ProgressInterface::incMaxProgress(float delta)
{
    _mutex.lock();
    _maxProgress += delta;
    _mutex.unlock();
}


void ProgressInterface::incTotalProgress(float delta)
{
    _mutex.lock();
    _progress = std::min<float>(_maxProgress, _progress + delta);
    _mutex.unlock();
    onProgressChanged();
}


void ProgressInterface::onProgressChanged()
{
}


void ProgressInterface::resetProgress()
{
    _mutex.lock();
    _progress = 0.0f;
    _progressPerTask.clear();
    _mutex.unlock();
}


int ProgressInterface::retrieveUniqueID()
{
    _mutex.lock();
    int newUniqueID = ++_currentUniqueID;
    _mutex.unlock();
    return newUniqueID;
}


float ProgressInterface::progress()
{
    if (_maxProgress + _progressPerTask.size() == 0) {
        return 1.0f;
    }

    _mutex.lock();
    float taskProgress = 0.0f;
    for (ProgressMap::const_iterator itr = _progressPerTask.begin();
        itr != _progressPerTask.end(); ++itr)
    {
        taskProgress += itr->second;
    }
    float result = (_progress + taskProgress) / 
                   (_maxProgress + (float)_progressPerTask.size());
    _mutex.unlock();
    return result;
}


void ProgressInterface::registerTask(int uniqueID, int taskID)
{
    setTaskProgress(uniqueID, taskID, 0.0f);
}


void ProgressInterface::setTaskProgress(int uniqueID, int taskID, 
                                        float progress)
{
    _mutex.lock();
    _progressPerTask[std::pair<int, int>(uniqueID, taskID)] = progress;
    _mutex.unlock();
    onProgressChanged();
}


} // namespace blissart
