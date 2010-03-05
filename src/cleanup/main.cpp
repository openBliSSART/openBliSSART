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
#include <blissart/StorageSubsystem.h>
#include <blissart/CleanupTask.h>

#include <Poco/Util/HelpFormatter.h>

#include <iostream>


using namespace std;
using namespace blissart;
using blissart::linalg::Matrix;
using namespace Poco::Util;


class CleanupTool : public BasicApplication
{
public:
    CleanupTool() : BasicApplication(),
        _displayUsage(false)
    {
        addSubsystem(new DatabaseSubsystem);
        addSubsystem(new StorageSubsystem);
    }


protected:
    void defineOptions(OptionSet &options)
    {
        Application::defineOptions(options);
        
        options.addOption(
            Option("help", "h",
                   "Displays usage information",
                   false));

        options.addOption(
            Option("simulate", "s", 
                   "Do not remove anything", 
                   false));

        options.addOption(
            Option("remove-na", "a", 
                   "Remove data belonging to data descriptors that are marked "
                   "as not being available", 
                   false));
    }


    void handleOption(const string &name, const string &value)
    {
        Application::handleOption(name, value);
        
        if (name == "help") {
            _displayUsage = true;
            stopOptionsProcessing();
        }
        else if (name == "remove-na") {
            _removeNA = true;
        }
        else if (name == "simulate") {
            _simulate = true;
        }
    }


    int main(const vector<string> &args)
    {
        if (_displayUsage) {
            HelpFormatter formatter(this->options());
            formatter.setUnixStyle(true);
            formatter.setAutoIndent();
            formatter.setUsage(this->commandName() + " <options>\n");
            formatter.setHeader("CleanupTool, removes unused data from storage");
            formatter.format(cout);
            return EXIT_USAGE;
        }
        
        CleanupTask ct;
        ct.setSimulation(_simulate);
        ct.setRemoveNA  (_removeNA);
        ct.runTask();
        cout << "Removed " << ct.removedFiles().size() << " file(s)" << endl;

        return EXIT_OK;
    }

    
private:
    bool _displayUsage;
    bool _removeNA;
    bool _simulate;
};


POCO_APP_MAIN(CleanupTool);
