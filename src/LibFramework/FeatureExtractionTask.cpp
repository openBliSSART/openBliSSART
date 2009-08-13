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


#include <blissart/FeatureExtractionTask.h>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/FeatureExtractor.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/Matrix.h>

#include <Poco/Exception.h>
#include <Poco/LogStream.h>

#include <sstream>
#include <iostream>


using namespace std;
using namespace blissart::linalg;


namespace blissart {


FeatureExtractionTask::FeatureExtractionTask() :
    BasicTask("FeatureExtractionTask"),
    _database(BasicApplication::instance().getSubsystem<DatabaseSubsystem>()),
    _storage(BasicApplication::instance().getSubsystem<StorageSubsystem>())
{
    incMaxProgress(1.0f);
}


void FeatureExtractionTask::addDataDescriptor(const DataDescriptorPtr dd)
{
    debug_assert(dd->available);
    _ddVec.push_back(dd);
    incMaxProgress(1.0f);
}


void FeatureExtractionTask::runTask()
{
    for (vector<DataDescriptorPtr>::const_iterator itr = _ddVec.begin();
        itr != _ddVec.end() && !isCancelled(); ++itr)
    {
        try {
            extract(*itr);
        }
        catch (const Poco::Exception& exc) {
            Poco::LogStream ls(logger());
            ls.error();
            ls << "Feature extraction failed for data descriptor #"
               << (*itr)->descrID << ": " << exc.displayText() << endl;
        }
        catch (const std::exception& exc) {
            Poco::LogStream ls(logger());
            ls.error();
            ls << "Feature extraction failed for data descriptor #"
               << (*itr)->descrID << ": " << exc.what() << endl;
        }
        incTotalProgress(1.0f);
        if (isCancelled())
            break;
    }
    _database.saveFeatures(_extractedFeatures);
    incTotalProgress(1.0f);
}


void FeatureExtractionTask::extract(const DataDescriptorPtr descr)
{
    ProcessMap::const_iterator itr = _processMap.find(descr->processID);
    ProcessPtr process;
    if (itr == _processMap.end()) {
        process = _processMap[descr->processID] = _database.getProcess(descr->processID);
    } else {
        process = itr->second;
    }
    
    // Check process parameters
    if (process->sampleFreq <= 0)
        throw Poco::InvalidArgumentException("Got Process with sample frequency 0");
    
    int windowSize = process->windowSize();
    if (windowSize <= 0) {
        ostringstream errStr;
        errStr << "Invalid parameter: windowSize = " 
               << windowSize << " in process with ID " 
               << process->processID;
        throw Poco::InvalidArgumentException(errStr.str());
    }
    
    double overlap = process->overlap();
    if (overlap < 0.0 || overlap >= 1.0) {
        ostringstream errStr;
        errStr << "Invalid parameter: overlap = " 
               << overlap << " in process with ID " 
               << process->processID;
        throw Poco::InvalidArgumentException(errStr.str());
    }

    FeatureExtractor::FeatureMap features;
    
    if (DataDescriptor::Gains == descr->type) {
        // f[kHz] = 1 / (windowSize[ms] * overlap)
        // f[Hz]  = 1 / (windowSize[ms] * overlap) * 10^3
        _featureExtractor.setGainsFrequency(
            1000.0 / (windowSize * (1.0 - overlap))
        );
        RowVector data(_storage.getLocation(descr).toString());
        features = _featureExtractor.extract(descr->type, data);
    }
    else if (DataDescriptor::Spectrum == descr->type ||
        DataDescriptor::MagnitudeMatrix == descr->type || 
        DataDescriptor::MelMatrix == descr->type) 
    {
        _featureExtractor.setSampleFrequency(process->sampleFreq);
        Matrix data(_storage.getLocation(descr).toString());
        features = _featureExtractor.extract(descr->type, data);
    }

    for (FeatureExtractor::FeatureMap::const_iterator itr = features.begin();
        itr != features.end(); ++itr)
    {
        _extractedFeatures.push_back(new Feature(descr->descrID, 
            itr->first.name, itr->first.params[0], itr->first.params[1],
            itr->first.params[2], itr->second));
    }
}


} // namespace blissart
