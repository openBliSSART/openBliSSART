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


#ifndef __BLISSART_CLEANUPTASK_H__
#define __BLISSART_CLEANUPTASK_H__


#include <blissart/BasicTask.h>
#include <Poco/Path.h>
#include <set>
#include <vector>


namespace blissart {


class StorageSubsystem;


/**
 * \addtogroup framework
 * @{
 */

/**
 * A task that deletes all files from a directory that are not referenced
 * by any data descriptor in the database.
 */
class LibFramework_API CleanupTask : public BasicTask
{
public:
    /**
     * Constructs a new instance of CleanupTask for the directory given
     * by the application's blissart.storageDir configuration option.
     */
    CleanupTask();


    /**
     * Constructs a new instance of CleanupTask for the given directory.
     */
    CleanupTask(const Poco::Path& directory);


    /**
     * The task's main method.
     */
    virtual void runTask();


    /**
     * Returns a vector of names of files which have been removed.
     */
    std::vector<Poco::Path> removedFiles() const;


    /**
     * Enables/disables simulation mode, i.e. don't really remove anything.
     */
    void setSimulation(bool flag);


    /**
     * Enables/disables removal of data associated with data descriptors
     * marked as being not available.
     */
    void setRemoveNA(bool flag);


private:
    // Forbid copy constructor and operator=.
    CleanupTask(const CleanupTask &other);
    CleanupTask& operator=(const CleanupTask &other);

    void cleanup(const Poco::Path& directory);

    Poco::Path              _directory;
    bool                    _simulate = false;
    bool                    _removeNA = false;
    std::set<int>           _validIDs;
    std::vector<Poco::Path> _removedFiles;
    StorageSubsystem&       _sts;
};


/**
 * @}
 */


// Inlines


inline std::vector<Poco::Path> CleanupTask::removedFiles() const
{
    return _removedFiles;
}


inline void CleanupTask::setSimulation(bool flag)
{
    _simulate = flag;
}


inline void CleanupTask::setRemoveNA(bool flag)
{
    _removeNA = flag;
}


} // namespace blissart


#endif // __BLISSART_CLEANUPTASK_H__
