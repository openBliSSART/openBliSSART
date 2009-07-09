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


#include "DatabaseSubsystemTest.h"
#include <Poco/File.h>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/FeatureSet.h>
#include <iostream>


using namespace std;
using namespace blissart;


namespace Testing {


DatabaseSubsystemTest::DatabaseSubsystemTest() : 
    _keepDB(false),
    _keepData(false)
{
}


bool DatabaseSubsystemTest::performTest()
{
    DatabaseSubsystem& database = 
        BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
    
    try {
    
        //
        // Process
        //
    
        const string processName = "Test process";
        const string processInputFile = "inputFile.wav";
        const int processSampleFreq = 8000;
        map<string, string> processParameters;
        processParameters["testParam"] = "testValue";
        processParameters["testParam2"] = "testValue2";
    
        int processID = 0;
        cout << "Creating a process with parameters." << endl;
        {
            ProcessPtr process = new Process(processName, processInputFile, 
                processSampleFreq);
            process->parameters = processParameters;
            database.createProcess(process);
            processID = process->processID;
        }
    
        cout << "Retrieving process #" << processID << "." << endl;
        {
            ProcessPtr process = database.getProcess(processID);
            if (process.isNull() ||
                process->name != processName || 
                process->processID != processID ||
                process->inputFile != processInputFile || 
                process->startTime == 0 ||
                process->sampleFreq != processSampleFreq ||
                process->parameters.size() != processParameters.size()) 
            {
                return false;
            }
        }
    
        cout << "Updating process #" << processID << "." << endl;
        {
            ProcessPtr process = database.getProcess(processID);
            const string newProcName = "newTestProcess";
            process->name = newProcName;
            process->parameters["testParam3"] = "testValue3";
            database.updateProcess(process);
            process = database.getProcess(processID);
            if (process->name != newProcName || 
                process->parameters.size() != processParameters.size() + 1) 
            {
                return false;
            }
        }
    
        cout << endl;
    
        // 
        // DataDescriptor
        //
    
        int descrID = 0;
        cout << "Creating a data descriptor." << endl;
        {
            DataDescriptorPtr descr = new DataDescriptor;
            descr->processID = processID;
            descr->type = DataDescriptor::PhaseMatrix;
            descr->available = false;
            database.createDataDescriptor(descr);
            descrID = descr->descrID;
        }
    
        cout << "Retrieving data descriptor #" << descrID << "." << endl;
        {
            DataDescriptorPtr descr = database.getDataDescriptor(descrID);
            if (descr->descrID != descrID || descr->processID != processID ||
                descr->type != DataDescriptor::PhaseMatrix || descr->available)
            {
                return false;
            }
        }
    
        cout << "Retrieving data descriptor by process_id, type and index."
             << endl;
        {
            DataDescriptorPtr descr = 
                database.getDataDescriptor(processID, 
                                           DataDescriptor::PhaseMatrix, 0);
            if (descr->descrID != descrID || 
                descr->processID != processID ||
                descr->type != DataDescriptor::PhaseMatrix)
            {
                return false;
            }
        }
    
        cout << "Updating data descriptor #" << descrID << "." << endl;
        {
            DataDescriptorPtr descr = database.getDataDescriptor(descrID);
            descr->type = DataDescriptor::Gains;
            database.updateDataDescriptor(descr);
            if (descr->type != DataDescriptor::Gains)
                return false;
        }
    
        cout << endl;
    
        //
        // Feature
        //
    
        cout << "Inserting features." << endl;
        database.saveFeature(new Feature(descrID, "pf", 1.0));
        database.saveFeature(new Feature(descrID, "periodicity", 1, 2, 3, 2.0));
        database.saveFeature(new Feature(descrID, "periodicity", 2, 3, 4, 2.1));
    
        cout << "Retrieving single feature." << endl;
        if (database.getFeature(descrID, "pf")->value != 1.0 ||
            database.getFeature(descrID, "periodicity", 1, 2, 3)->value != 2.0 ||
            database.getFeature(descrID, "periodicity", 2, 3, 4)->value != 2.1)
        {
            return false;
        }
    
        cout << "Retrieving multiple features." << endl;
        {
            vector<FeaturePtr> features = database.getFeatures(descrID);
            if (features.size() != 3)
                return false;
        }
    
        cout << "Updating feature." << endl;
        database.saveFeature(new Feature(descrID, "pf", 3.0));
        if (database.getFeature(descrID, "pf")->value != 3.0)
            return false;
    
        cout << endl;
    
        // 
        // Label
        //
    
        cout << "Inserting some labels." << endl;
        LabelPtr labels[3] = { new Label("ClassLabel1"), 
                               new Label("ClassLabel2"), 
                               new Label("ClassLabel3") };
        for (int i = 0; i < 3; ++i)
            database.createLabel(labels[i]);
    
        cout << "Retrieving label #" << labels[0]->labelID << "." << endl;
        {
            LabelPtr label = database.getLabel(labels[0]->labelID);
            if (label->labelID != labels[0]->labelID ||
                label->text != labels[0]->text) 
            {
                return false;
            }
        }

        cout << "Retrieving label by text." << endl;
        {
            vector<LabelPtr> foundLabels = 
                database.getLabelsByText("ClassLabel1");
            if (foundLabels.size() < 1 ||
                foundLabels[0]->text != labels[0]->text)
            {
                return false;
            }
        }
    
        cout << "Updating label #" << labels[2]->labelID << "." << endl;
        {
            const string newLabelText = "ClassLabel3New";
            labels[2]->text = newLabelText;
            database.updateLabel(labels[2]);
            LabelPtr updatedLabel = database.getLabel(labels[2]->labelID);
            if (updatedLabel->text != newLabelText)
                return false;
        }
    
        cout << endl;
    
        //
        // ClassificationObject
        //
    
        cout << "Creating a classification object." << endl;
        int clObjID = 0;
        {
            ClassificationObjectPtr clObj = new ClassificationObject;
            clObj->type = ClassificationObject::NMFComponent;
            clObj->descrIDs.insert(descrID);
            clObj->labelIDs.insert(labels[0]->labelID);
            clObj->labelIDs.insert(labels[1]->labelID);
            database.createClassificationObject(clObj);
            clObjID = clObj->objectID;
        }
    
        cout << "Retrieving classification object #" << clObjID << "." << endl;
        {
            ClassificationObjectPtr clObj = 
                database.getClassificationObject(clObjID);
            if (clObj->type != ClassificationObject::NMFComponent ||
                clObj->descrIDs.size() != 1 ||
                clObj->labelIDs.size() != 2)
            {
                return false;
            }
        }
    
        cout << "Updating classification object #" << clObjID << "." << endl;
        {
            ClassificationObjectPtr clObj = 
                database.getClassificationObject(clObjID);
            clObj->labelIDs.erase(labels[0]->labelID);
            clObj->labelIDs.insert(labels[2]->labelID);
            clObj->descrIDs.erase(descrID);
            clObj->type = ClassificationObject::NMDComponent;
            database.updateClassificationObject(clObj);
            clObj = database.getClassificationObject(clObjID);
            if (clObj->type != ClassificationObject::NMDComponent)
                return false;
            if (clObj->labelIDs.find(labels[0]->labelID) != clObj->labelIDs.end())
                return false;
            if (clObj->labelIDs.find(labels[1]->labelID) == clObj->labelIDs.end())
                return false;
            if (clObj->labelIDs.find(labels[2]->labelID) == clObj->labelIDs.end())
                return false;
            if (clObj->descrIDs.size() != 0)
                return false;
            clObj->descrIDs.insert(descrID);
            database.updateClassificationObject(clObj);
            clObj = database.getClassificationObject(clObjID);
            if (clObj->descrIDs.find(descrID) == clObj->descrIDs.end())
                return false;
            // The type has to be reset to avoid constraint violation when
            // creating a response later on.
            clObj->type = ClassificationObject::NMFComponent;
            database.updateClassificationObject(clObj);
        }
    
        // This has to be done after updateClassificationObject test
        // to avoid constraint violation.
        cout << "Removing label #" << labels[2]->labelID << "." << endl;
        {
            database.removeLabel(labels[2]);
            LabelPtr removedLabel = database.getLabel(labels[2]->labelID);
            if (!removedLabel.isNull())
                return false;
        }
    
        cout << endl;
    
        //
        // Response
        //
    
        const string responseName = "responseName";
        const string responseD = "This is a response";
        int responseID = 0;
    
        cout << "Creating a response." << endl;
        {
            ResponsePtr response = new Response(responseName, responseD);
            response->labels[clObjID] = labels[0]->labelID;
            database.createResponse(response);
            responseID = response->responseID;
        }
        
        cout << "Retrieving response #" << responseID << "." << endl;
        {
            ResponsePtr response = database.getResponse(responseID);
            if (response->name != responseName || 
                response->description != responseD ||
                response->labels.size() != 1 ||
                response->labels[clObjID] != labels[0]->labelID)
            {
                return false;
            }
        }
    
        cout << "Retrieving DataSet for response #" << responseID << "." 
             << endl;
        {
            DataSet dataSet = 
                database.getDataSet(database.getResponse(responseID));
            DataPoint::ComponentMap components = dataSet[0].components;
            if (components.size() != 3)
                return false;
            for (DataPoint::ComponentMap::const_iterator itr = components.begin();
                itr != components.end(); ++itr)
            {
                if (itr->first.dataType != DataDescriptor::Gains)
                    return false;
                if (itr->first.name == "pf") {
                    if (itr->second != 3.0)
                        return false;
                }
                else if (itr->first.name == "periodicity" && 
                    itr->first.params[0] == 1)
                {
                    if (itr->first.params[1] != 2)
                        return false;
                    if (itr->first.params[2] != 3)
                        return false;
                    if (itr->second != 2.0)
                        return false;
                }
                else if (itr->first.name == "periodicity" && 
                    itr->first.params[0] == 2) 
                {
                    if (itr->first.params[1] != 3)
                        return false;
                    if (itr->first.params[2] != 4)
                        return false;
                    if (itr->second != 2.1)
                        return false;
                }
                else
                    return false;
            }
            if (dataSet[0].classLabel != labels[0]->labelID)
                return false;
        }
    
        cout << "Retrieving DataSet for response using FeatureSet." << endl;
        FeatureSet fs;
        fs.add(FeatureDescriptor("pf", DataDescriptor::Gains));
        fs.add(FeatureDescriptor("periodicity", DataDescriptor::Gains, 2, 3, 4));
        {
            DataSet dataSet = 
                database.getDataSet(database.getResponse(responseID), fs);
            if (dataSet[0].components.size() != 2)
                return false;
        }
        
        cout << "Retrieving DataSet using ClassificationObjectID- vs. LabelID-map"
             << endl;
        {
            map<int, int> cloLabelsMap;
            cloLabelsMap[clObjID] = labels[0]->labelID;
            DataSet dataSet = database.getDataSet(cloLabelsMap);
            if (dataSet.empty())
                return false;
        }
    
        cout << "Retrieving response listing." << endl;
        {
            vector<ResponsePtr> responseVec = database.getResponses();
            if (responseVec.size() == 0) {
                cerr << "Response listing was empty." << endl;
                return false;
            }
            ResponsePtr response;
            for (vector<ResponsePtr>::const_iterator itr = responseVec.begin();
                itr != responseVec.end(); ++itr)
            {
                if ((*itr)->responseID == responseID) {
                    response = *itr;
                    break;
                }
            }
            if (response->name != responseName || 
                response->description != responseD ||
                response->labels.size() != 1 ||
                response->labels[clObjID] != labels[0]->labelID)
            {
                cerr << "Listing did not contain previously inserted response."
                     << endl;
                return false;
            }
        }
    
        cout << "Testing DataSet consistency check." << endl;
        {
            DataDescriptorPtr dataDescr2 = new DataDescriptor;
            dataDescr2->type = DataDescriptor::Gains;
            dataDescr2->index = 1;
            dataDescr2->processID = processID;
            database.createDataDescriptor(dataDescr2);
            FeaturePtr feature2 = new Feature(dataDescr2->descrID, "pf", 5.0);
            database.saveFeature(feature2);
            ClassificationObjectPtr clObj2 = new ClassificationObject;
            clObj2->type = ClassificationObject::NMFComponent;
            clObj2->descrIDs.insert(dataDescr2->descrID);
            database.createClassificationObject(clObj2);
            ResponsePtr response = database.getResponse(responseID);
            response->labels[clObj2->objectID] = labels[0]->labelID;
            database.updateResponse(response);
            bool excThrown = false;
            try {
                database.getDataSet(response);
            }
            catch (const Poco::RuntimeException& exc) {
                cout << "Caught exception: " << exc.message() << endl;
                excThrown = true;
            }
            if (!excThrown)
                return false;
        }

        cout << "Updating response #" << responseID << "." << endl;
        {
            const string newResponseD = "This is a new response";
            ResponsePtr response = database.getResponse(responseID);
            response->description = newResponseD;
            response->labels[clObjID] = labels[1]->labelID;
            database.updateResponse(response);
            response = database.getResponse(responseID);
            if (response->description != newResponseD ||
                response->labels[clObjID] != labels[1]->labelID)
            {
                return false;
            }
        }
    
        cout << endl;
    
        //
        // Remove entries
        //
    
        if (!_keepData) {

            cout << "Removing features." << endl;
            {
                database.removeFeature(database.getFeature(descrID, "pf"));
                database.removeFeatures(descrID);
                if (!database.getFeature(descrID, "pf").isNull() ||
                    !database.getFeature(descrID, "periodicity", 1).isNull() ||
                    !database.getFeature(descrID, "periodicity", 2).isNull())
                {
                    return false;
                }
            }
        
            cout << "Removing response #" << responseID << "." << endl;
            {
                database.removeResponse(database.getResponse(responseID));
                if (!database.getResponse(responseID).isNull())
                    return false;
            }
        
            cout << "Removing classification object #" << clObjID << "." << endl;
            {
                ClassificationObjectPtr clObj =
                    database.getClassificationObject(clObjID);
                database.removeClassificationObject(clObj);
                clObj = database.getClassificationObject(clObjID);
                if (!clObj.isNull())
                    return false;
            }
        
            cout << "Removing data descriptor #" << descrID << "." << endl;
            {
                DataDescriptorPtr descr = database.getDataDescriptor(descrID);
                database.removeDataDescriptor(descr);
                descr = database.getDataDescriptor(descrID);
                if (!descr.isNull())
                    return false;
            }
        
            cout << "Removing process #" << processID << "." << endl;
            {
                ProcessPtr process = database.getProcess(processID);
                database.removeProcess(process);
                // Assert that process does not exist any more
                process = database.getProcess(processID);
                if (!process.isNull())
                    return false;
            }
        
            cout << endl;
    
        } // !_keepData

        cout << "Testing database disconnect (";
        if (!_keepDB)
            cout << "not ";
        cout << "keeping database file)." << endl;
        {    
            database.disconnect();
            if (!_keepDB) {
                database.destroy();
            }
        }

    } catch (Poco::Exception& exc) {
        cerr << exc.displayText() << endl;
        return false;
    }

    return true;
}


} // namespace Testing
