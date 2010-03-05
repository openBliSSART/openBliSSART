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


#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>


using namespace std;
using namespace blissart;
using namespace Poco::Util;


class LabellerApplication : public BasicApplication
{
public:
    LabellerApplication() : BasicApplication()
    {
        addSubsystem(new DatabaseSubsystem);
    }


    int main(const vector<string> &args)
    {
        if (args.size() == 0) {
            cout << "Usage: " << this->commandName() << " <filename>" << endl
                 << "where <filename> contains tab-separated labels for files"
                 << endl;
            return EXIT_USAGE;
        }
        
        ifstream inputFile(args[0].c_str());
        if (inputFile.fail()) {
            cerr << "Could not open file " << args[0] << endl;
            return EXIT_IOERR;
        }

        DatabaseSubsystem& dbs = getSubsystem<DatabaseSubsystem>();

        string filename, labelText;
        char line[1024];
        map<string, LabelPtr> knownLabels;

        vector<ClassificationObjectPtr> allObjects;

        while (!inputFile.eof()) {
            inputFile.getline(line, 1024);
            string lineStr(line);
            if (lineStr.length() == 0)
                continue;

            string::size_type tabPos = lineStr.find('\t');
            filename = lineStr.substr(0, tabPos);
#ifdef _DEBUG
            cout << "filename: " << filename << endl;
#endif

            vector<ClassificationObjectPtr> objects =
                dbs.getClassificationObjectsByFilename(filename);
            if (objects.size() == 0) {
                throw Poco::NotFoundException("No object found for filename \""
                    + filename + "\"");
            }

            while (tabPos < lineStr.length()) {
                lineStr = lineStr.substr(tabPos + 1);
                // If tab is not found, find returns npos, which makes
                // substr return the whole rest :)
                tabPos = lineStr.find('\t');
                labelText = lineStr.substr(0, tabPos);
#ifdef _DEBUG
                cout << "label: " << labelText << endl;
#endif

                // Find the labels in the database with the given text.
                LabelPtr label;
                map<string, LabelPtr>::const_iterator 
                    itr = knownLabels.find(labelText);
                if (itr == knownLabels.end()) {
                    vector<LabelPtr> labels = dbs.getLabelsByText(labelText);
                    if (labels.size() != 1) {
                        ostringstream excText;
                        excText << "Label \"" << labelText << "\" exists "
                                << labels.size() << " times in the database!";
                        throw Poco::InvalidArgumentException(excText.str());
                    }
                    label = labels[0];
                }
                else {
                    label = itr->second;
                }
                
                // Assign corresponding label ID to the CLOs if they don't
                // have it already.
                for (vector<ClassificationObjectPtr>::iterator itr = 
                    objects.begin(); itr != objects.end(); ++itr)
                {
#ifdef _DEBUG
                    cout << "Assigning label " << label->labelID << 
                        " to object " << (*itr)->objectID << endl;
#endif
                    if ((*itr)->labelIDs.find(label->labelID) ==
                        (*itr)->labelIDs.end()) 
                    {
                        (*itr)->labelIDs.insert(label->labelID);
                    }
                }
            }
            allObjects.insert(allObjects.end(), objects.begin(), objects.end());
            cout << ".";
        }

        cout << endl;
        // Data must not be saved until now, to avoid constructing responses
        // from erroneous input files.
        cout << "Saving..." << endl;
        dbs.updateClassificationObjects(allObjects);
        
        return EXIT_OK;
    }
};


POCO_APP_MAIN(LabellerApplication);
