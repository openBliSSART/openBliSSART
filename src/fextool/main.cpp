//
// $Id: main.cpp 898 2009-07-07 13:50:21Z felix $
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


#include <blissart/ThreadedApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/FeatureExtractionTask.h>
#include <blissart/QueuedTaskManager.h>
#include <blissart/BasicTaskNotification.h>
#include <blissart/validators.h>

#include <Poco/AutoPtr.h>
#include <Poco/Mutex.h>
#include <Poco/Event.h>
#include <Poco/NumberFormatter.h>
#include <Poco/NumberParser.h>
#include <Poco/DateTime.h>
#include <Poco/DateTimeFormatter.h>
#include <Poco/NObserver.h>
#include <Poco/Util/HelpFormatter.h>
#include <Poco/Util/RegExpValidator.h>

#include <iostream>
#include <vector>
#include <iomanip>


using namespace std;
using namespace blissart;
using namespace Poco;
using namespace Poco::Util;


class FeXTool : public ThreadedApplication
{
public:
    FeXTool() :
      _displayUsage(false),
      _extractAll(false)
    {
        addSubsystem(new DatabaseSubsystem());
        addSubsystem(new StorageSubsystem());
    }


protected:
    virtual void defineOptions(OptionSet &options)
    {
        Application::defineOptions(options);

        options.addOption(
            Option("help", "h",
                   "Displays usage information",
                   false));

        options.addOption(
            Option("all", "a",
                   "Extracts features for all data descriptors in the database.",
                   false));

        options.addOption(
            Option("process", "p",
                   "Extracts features for all data descriptors associated with "
                   "the given process IDs. Single numbers or ranges "
                   "(min..max) may be given.",
                   false, "<id>", true)
            .repeatable(true)
            .validator(new RegExpValidator("\\d+(\\.\\.\\d+)?")));

        options.addOption(
            Option("num-threads", "n",
                   "Use the given number of threads during feature extraction.",
                   false, "<nr>", true)
            .validator(new validators::RangeValidator<int>(1)));
    }


    virtual void handleOption(const string &name, const string &value)
    {
        Application::handleOption(name, value);

        if (name == "all") {
            _extractAll = true;
        }
        else if (name == "process") {
            int start, end;
            string::size_type pos = value.find("..");
            if (pos != string::npos) {
                start = Poco::NumberParser::parse(value.substr(0, pos));
                end = Poco::NumberParser::parse(value.substr(pos + 2));
            }
            else {
                start = end = Poco::NumberParser::parse(value);
            }
            for (int id = start; id <= end; ++id) {
                _processIDs.push_back(id);
            }
        }
        else if (name == "num-threads") {
            setNumThreads(Poco::NumberParser::parse(value));
        }
        else if (name == "help") {
            _displayUsage = true;
            stopOptionsProcessing();
        }
    }


    virtual int main(const vector<string> &args)
    {
        if (_displayUsage || args.size() != 0 || 
            !(_extractAll || _processIDs.size() > 0)) 
        {
            HelpFormatter formatter(this->options());
            formatter.setUnixStyle(true);
            formatter.setAutoIndent();
            formatter.setUsage(this->commandName() + " <options>\n");
            formatter.setHeader("FeXTool, a feature extraction tool");
            formatter.format(cout);
            return EXIT_USAGE;
        }

        cout << "FeXTool, "
             << DateTimeFormatter::format(DateTime(), "%Y/%m/%d %H:%M:%S")
             << endl << endl
             << setw(20) << "Process(es): ";
        if (_extractAll) {
            cout << "all" << endl;
        }
        else {
            for (vector<int>::const_iterator itr = _processIDs.begin();
                itr != _processIDs.end(); ++itr)
            {
                cout << *itr << " ";
            }
            cout << endl;
        }
        cout << setw(20) << "# of threads: " << numThreads() << endl
             << endl;

        // Get the processes for whose associated data the features should be
        // extracted.
        DatabaseSubsystem& database = getSubsystem<DatabaseSubsystem>();
        vector<ProcessPtr> processes;
        if (_extractAll) {
            processes = database.getProcesses();
        }
        else {
            for (vector<int>::const_iterator itr = _processIDs.begin();
                itr != _processIDs.end(); ++itr)
            {
                ProcessPtr process = database.getProcess(*itr);
                if (process.isNull()) {
                    throw Poco::InvalidArgumentException("Invalid process ID: " +
                        Poco::NumberFormatter::format(*itr));
                }
                processes.push_back(process);
            }
        }

        // Create the neccessary number of tasks and distribute the data
        // descriptors associated with the numerous processes among the tasks.
        vector<FeatureExtractionTaskPtr> fexTasks;
        for (unsigned int i = 0; i < numThreads(); i++) {
            fexTasks.push_back(new FeatureExtractionTask);
        }
        for (unsigned int i = 0; i < processes.size(); ++i) {
            vector<DataDescriptorPtr> data =
                database.getDataDescriptors(processes.at(i)->processID);
            for (vector<DataDescriptorPtr>::const_iterator dataItr = data.begin();
                 dataItr != data.end(); ++dataItr)
            {
                if ((*dataItr)->available)
                    fexTasks.at(i % numThreads())->addDataDescriptor(*dataItr);
            }
        }

        // Initialize the task manager and add the tasks.
        initializeTaskManager<ThreadedApplication>();
        for (vector<FeatureExtractionTaskPtr>::iterator it = fexTasks.begin();
             it != fexTasks.end(); ++it)
        {
            addTask(*it);
        }

        // Wait until all tasks have finished and display some progress
        // information.
        waitForCompletion();

        return EXIT_OK;
    }


private:
    bool         _displayUsage;
    vector<int>  _processIDs;
    int          _extractAll;
};


POCO_APP_MAIN(FeXTool);

