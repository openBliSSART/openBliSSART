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


#include <blissart/CleanupTask.h>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <Poco/DirectoryIterator.h>
#include <iostream>


using namespace std;


namespace blissart {


CleanupTask::CleanupTask(const Poco::Path& directory) :
    BasicTask("CleanupTask"),
    _directory(directory),
    _sts(BasicApplication::instance().getSubsystem<StorageSubsystem>())
{
}


CleanupTask::CleanupTask() :
    BasicTask("CleanupTask"),
    _directory(BasicApplication::instance()
        .config()
        .getString("blissart.storageDir")
    ),
    _sts(BasicApplication::instance().getSubsystem<StorageSubsystem>())
{
}


void CleanupTask::runTask()
{
    Poco::Util::Application& app = BasicApplication::instance();
    DatabaseSubsystem &dbs = app.getSubsystem<DatabaseSubsystem>();

    vector<ProcessPtr> processes = dbs.getProcesses();
    for (vector<ProcessPtr>::const_iterator itr = processes.begin();
        itr != processes.end(); ++itr)
    {
        vector<DataDescriptorPtr> dds = dbs.getDataDescriptors((*itr)->processID);
        for (vector<DataDescriptorPtr>::const_iterator dItr = dds.begin();
            dItr != dds.end(); ++dItr)
        {
            if (!_removeNA || (*dItr)->available) {
                _validIDs.insert((*dItr)->descrID);
            }
        }
    }

    cleanup(_directory);
}


void CleanupTask::cleanup(const Poco::Path& directory)
{
    Poco::DirectoryIterator sItr(directory);
    Poco::DirectoryIterator sEndItr;
    for (; sItr != sEndItr; ++sItr) {
        if (sItr->isDirectory()) {
            cleanup(sItr->path());
        }
        else {
            if (_validIDs.find(_sts.getDescriptorID(sItr->path()))
                == _validIDs.end())
            {
                _removedFiles.push_back(sItr->path());
                if (!_simulate) {
                    sItr->remove();
                }
            }
        }
    }
}


} // namespace blissart
