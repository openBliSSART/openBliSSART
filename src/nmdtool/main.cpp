//
// $Id: main.cpp 855 2009-06-09 16:15:50Z alex $
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


#include <Poco/Util/HelpFormatter.h>
#include <Poco/Util/RegExpValidator.h>
#include <Poco/NumberFormatter.h>
#include <Poco/NumberParser.h>
#include <Poco/NObserver.h>
#include <Poco/Mutex.h>
#include <Poco/Event.h>

#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/ThreadedApplication.h>
#include <blissart/QueuedTaskManager.h>
#include <blissart/BasicTaskNotification.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/feature/mfcc.h>
#include <blissart/validators.h>
#include <blissart/TargetedDeconvolver.h>
#include <blissart/audio/audio.h>
#include <blissart/validators.h>

#include <iostream>

#include "NMDGainsTask.h"


using namespace std;
using namespace blissart;
using namespace blissart::linalg;
using blissart::feature::computeMFCC;
using namespace Poco::Util;
using namespace Poco;


class NMDTool : public ThreadedApplication
{
public:
    NMDTool() :
        _displayUsage(false),
        _scripted(false),
        _responseID(0),
        _nComponents(0),
        _nIterations(100),
        _transformation(NMDGainsTask::NoTransformation)
    {
        addSubsystem(new DatabaseSubsystem());
        addSubsystem(new StorageSubsystem());
    }


protected:
    virtual void initialize(Application& self)
    {
        ThreadedApplication::initialize(self);

        // Initialize LibAudio.
        blissart::audio::initialize();
    }


    virtual void uninitialize()
    {
        // Shut down LibAudio.
        if (audio::isInitialized())
            audio::shutdown();

        ThreadedApplication::uninitialize();
    }


    virtual void defineOptions(OptionSet &options)
    {
        Application::defineOptions(options);

        options.addOption(
            Option("help", "h",
                   "Displays usage information",
                   false));

        options.addOption(
            Option("scripted", "S",
                   "Run in scripted mode, i.e. the input files contain "
                   "list of sound files to process.",
                   false));

		options.addOption(
			Option("export", "p",
			       "Do not save anything to the database, but export the NMF "
				   "activation matrix in HTK format.",
				   false));

		options.addOption(
			Option("export-dir", "",
			       "Directory where exported files should be stored.",
				   false, "<dir>", true));

        options.addOption(
            Option("num-threads", "n",
                   "The number of concurrent threads that should be started.",
                   false, "<number>", true)
            .validator(new validators::RangeValidator<int>(1, 16)));

        options.addOption(
            Option("response", "r",
                   "Response ID giving NMD initialization",
                   true, "<id>", true)
            .validator(new validators::RangeValidator<int>(1)));

        options.addOption(
            Option("components", "c",
                   "The number of components to separate with NMD. "
                   "If the number of components is larger than the number of "
                   "classification objects in the response, then additional "
                   "noise components are generated.",
                   false, "<number>", true)
            .validator(new validators::RangeValidator<int>(1)));

        options.addOption(
            Option("all-components", "a",
                   "Saves the gains for all components, including the noise "
                   "ones. Otherwise only gains for initialized components "
                   "are saved.",
                   false));

        options.addOption(
            Option("transform", "t",
                   "Transforms NMD gains. Available methods are \"linear\" "
                   "(normalize each column such that it sums to 1) "
                   "or \"dct\" (calculates DCT of logarithmic gains).",
                   false, "<method>", true)
            .validator(new RegExpValidator("(l|m|n).*")));

        options.addOption(
            Option("index-count", "C",
                   "The number of indices to output, if transformation to "
                   "maximal indices is desired.",
                   false, "<number>", true)
            .validator(new validators::RangeValidator<int>(1))
            .binding("blissart.nmdtool.index_count"));

        options.addOption(
            Option("iter", "i",
                   "Specifies the number of iteration steps.",
                   false, "<number>", true).
            validator(new validators::RangeValidator<int>(1)));
    }


    virtual void handleOption(const string &name, const string &value)
    {
        Application::handleOption(name, value);

        if (name == "help") {
            _displayUsage = true;
            stopOptionsProcessing();
        }
        else if (name == "scripted") {
            _scripted = true;
        }
		else if (name == "export") {
			_export = true;
		}
		else if (name == "export-dir") {
			_exportDir = value;
		}
        else if (name == "num-threads") {
            setNumThreads(Poco::NumberParser::parse(value));
        }
        else if (name == "response") {
            _responseID = Poco::NumberParser::parse(value);
        }
        else if (name == "components") {
            _nComponents = Poco::NumberParser::parse(value);
        }
        else if (name == "all-components") {
            _allComponents = true;
        }
        else if (name == "transform") {
            if (value[0] == 'l') {
                _transformation = NMDGainsTask::UnitSum;
            }
            else if (value[0] == 'd') {
                _transformation = NMDGainsTask::LogDCT;
            }
            else if (value[0] == 'm') {
                _transformation = NMDGainsTask::MaximalIndices;
            }
        }
    }


    virtual int main(const vector<string> &args)
    {
        if (_displayUsage || args.size() == 0) {
            HelpFormatter formatter(this->options());
            formatter.setUnixStyle(true);
            formatter.setAutoIndent();
            formatter.setUsage(this->commandName() +
                " <options> file1 file2 ...\n"
                "where file1, file2 ... can be WAV, MP3 or script files");
            formatter.setHeader("NMDTool, computes NMD gains from audio files");
            formatter.format(cout);
            return EXIT_USAGE;
        }

        vector<string> inputFiles = _scripted ?
            BasicApplication::parseScriptFiles(args) : args;

        // Get classification objects in given response.
        vector<ClassificationObjectPtr> clObjs;
        DatabaseSubsystem& dbs = getSubsystem<DatabaseSubsystem>();
        ResponsePtr response = dbs.getResponse(_responseID);
        if (response.isNull()) {
            throw Poco::InvalidArgumentException("Invalid response ID: " +
                Poco::NumberFormatter::format(_responseID));
        }
        for (Response::LabelMap::const_iterator itr = response->labels.begin();
            itr != response->labels.end(); ++itr)
        {
            clObjs.push_back(dbs.getClassificationObject(itr->first));
        }

        // Initialize the task manager.
        initializeTaskManager<ThreadedApplication>();

        // Setup parameters for NMDGainsTasks.
        for (vector<string>::const_iterator itr = inputFiles.begin();
            itr != inputFiles.end(); ++itr)
        {
			Poco::AutoPtr<NMDGainsTask> task = new NMDGainsTask(
                *itr,
                _nComponents,
                _nIterations,
                clObjs,
                _allComponents,
                _transformation
			);
			task->setExport(_export);
			task->setExportDir(_exportDir);
			addTask(task);
        }

        // Wait until all tasks have finished and display some progress
        // information.
        waitForCompletion();

        return EXIT_OK;
    }


private:
    bool   _displayUsage;
    bool   _scripted;
	bool   _export;
	string _exportDir;
    int    _responseID;
    int    _nComponents;
    bool   _allComponents;
    int    _nIterations;
    NMDGainsTask::TransformationMethod _transformation;
};


POCO_APP_MAIN(NMDTool);

