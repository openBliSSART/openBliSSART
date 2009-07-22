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


#ifndef __BLISSART_FEATUREEXTRACTIONTASK_H__
#define __BLISSART_FEATUREEXTRACTIONTASK_H__


#include <common.h>
#include <blissart/BasicTask.h>
#include <blissart/Feature.h>
#include <blissart/FeatureExtractor.h>
#include <blissart/DataDescriptor.h>
#include <blissart/Process.h>
#include <map>
#include <vector>


namespace blissart {


class DatabaseSubsystem;
class StorageSubsystem;


/**
 * Extracts features from matrices and vectors that are pointed to by
 * the given data descriptors.
 */
class LibFramework_API FeatureExtractionTask : public BasicTask
{
public:
    FeatureExtractionTask();

    /**
     * Advises the task to perform feature extraction for the given
     * data descriptor.
     */
    void addDataDescriptor(const DataDescriptorPtr dd);

    /**
     * Implementation of BasicTask interface.
     */
    virtual void runTask();

private:
    /**
     * Extract the features from the object in the database described by the
     * given DataDescriptor.
     */
    void extract(const DataDescriptorPtr descr);

    // Since some feature need information about the process that created the
    // DataDescriptor, we cache process information to speed up multiple
    // extractions.
    typedef std::map<int, ProcessPtr> ProcessMap;
    ProcessMap _processMap;

    std::vector<DataDescriptorPtr> _ddVec;

    // Store extracted features in memory and write them at once in one
    // transaction, once the task has been completed.
    std::vector<FeaturePtr>        _extractedFeatures;

    DatabaseSubsystem& _database;
    StorageSubsystem&  _storage;
    FeatureExtractor   _featureExtractor;
};


typedef Poco::AutoPtr<FeatureExtractionTask> FeatureExtractionTaskPtr;


} // namespace blissart


#endif // #ifndef __BLISSART_FEATUREEXTRACTIONTASK_H__
