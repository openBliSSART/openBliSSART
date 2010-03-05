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


#include <blissart/exportDataSet.h>
#include <blissart/FeatureSet.h>
#include <blissart/Label.h>
#include <blissart/DatabaseSubsystem.h>
#include <Poco/Util/Application.h>
#include <Poco/Path.h>
#include <fstream>
#include <iostream>
#include <iomanip>


using namespace std;


namespace blissart {


bool exportDataSet(const DataSet& dataSet,
                   const std::string& fileName,
                   const std::string& name,
                   const std::string& description)
{
    ofstream outputFile(fileName.c_str());
    
    if (outputFile.fail()) {
        return false;
    }

    outputFile << "@RELATION '" << name << "'" << endl << endl;
    outputFile << "% " << description << endl << endl;

    // Get feature set and class labels
    FeatureSet featureSet;
    DatabaseSubsystem& dbss =
        Poco::Util::Application::instance().getSubsystem<DatabaseSubsystem>();
    map<int, LabelPtr> classLabels;
    for (DataSet::const_iterator dItr = dataSet.begin();
        dItr != dataSet.end(); ++dItr)
    {
        if (classLabels.find(dItr->classLabel) == classLabels.end())
            classLabels[dItr->classLabel] = dbss.getLabel(dItr->classLabel);
        for (DataPoint::ComponentMap::const_iterator cItr =
            dItr->components.begin(); cItr != dItr->components.end(); ++cItr)
        {
            if (!featureSet.has(cItr->first))
                featureSet.add(cItr->first);
        }
    }

    vector<FeatureDescriptor> features = featureSet.list();
    outputFile << "% " << setw(5) << dataSet.size() << " entities with " << endl
               << "% " << setw(5) << features.size() << " features" << endl
               << endl;

    // Output feature set
    for (vector<FeatureDescriptor>::const_iterator itr = features.begin();
        itr != features.end(); ++itr)
    {
        outputFile << "@ATTRIBUTE "
            << DataDescriptor::strForTypeShort(itr->dataType) << "_"
            << itr->name;
        for (int i = 0; i <= 2; ++i) {
            outputFile << "_" << itr->params[i];

        }
        outputFile << " NUMERIC" << endl;
    }

    // Output class labels
    outputFile << "@ATTRIBUTE class {";
    unsigned int i = 0;
    for (map<int, LabelPtr>::const_iterator itr = classLabels.begin();
        itr != classLabels.end(); ++itr, ++i)
    {
        outputFile << '"' << itr->second->text << '"';
        if (i < classLabels.size() - 1)
            outputFile << ",";
    }
    outputFile << "}" << endl << endl;

    outputFile << "@DATA" << endl << endl;

    for (DataSet::const_iterator dItr = dataSet.begin();
        dItr != dataSet.end(); ++dItr)
    {
        ProcessPtr process =
            dbss.getProcess(dbss.getClassificationObject(dItr->objectID));
        Poco::Path inputFilePath(process->inputFile);
        outputFile << "% Source: "
            << inputFilePath.getFileName()
            << " (object ID " << dItr->objectID << ")"
            << endl;

        for (vector<FeatureDescriptor>::const_iterator itr = features.begin();
            itr != features.end(); ++itr)
        {
            outputFile << dItr->components.find(*itr)->second;
            outputFile << ",";
        }
        outputFile << '"' << classLabels[dItr->classLabel]->text << '"' 
                   << endl;
    }

    outputFile.flush();
    outputFile.close();

    return true;
}


} // namespace blissart
