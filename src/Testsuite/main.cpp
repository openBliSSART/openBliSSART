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


#include <Poco/Util/Application.h>
#include <Poco/Util/HelpFormatter.h>

#include <vector>
#include <iostream>

#include "BinaryReaderWriterTest.h"
#include "DatabaseSubsystemTest.h"
#include "FeatureExtractionTest.h"
#include "FeatureSelectionTest.h"
#include "ICATest.h"
#include "HTKWriterTest.h"
#include "MFCCTest.h"
#include "MatrixTest.h"
#include "MinHeapTest.h"
#include "MelFilterTest.h"
#include "MTrTest.h"
#include "NMDTest.h"
#include "NMFTest.h"
#include "PCATest.h"
#include "ScalingTest.h"
#include "SNMFTest.h"
#include "CNMFTest.h"
#include "SVMModelTest.h"
#include "SpectralAnalysisTest.h"
#include "StorageSubsystemTest.h"
#include "TaskDepsTest.h"
#include "VectorTest.h"
#include "WaveTest.h"

#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/audio/audio.h>


using namespace std;
using namespace Testing;
using namespace Poco::Util;


class Testsuite : public Application
{
public:
    Testsuite()
    {
        setUnixOptions(true);
    }


protected:
    virtual void initialize(Application& self)
    {
        // Base class initialization.
        Application::initialize(self);

        self.config().setString("blissart.databaseFile", "Testsuite.db");
        self.config().setString("logging.loggers.root.channel.class", "ConsoleChannel");
        self.config().setString("logging.loggers.root.level",
#ifdef _DEBUG
            "debug"
#else
            "information"
#endif
        );

        // Initialize LibAudio.
        blissart::audio::initialize();
    }


    virtual void uninitialize()
    {
        Application::uninitialize();
        // Shut down LibAudio.
        blissart::audio::shutdown();
    }


    void defineOptions(OptionSet &options)
    {
        Application::defineOptions(options);
        options.addOption(Option("brw", "", "Test binary read/write functions"));
        options.addOption(Option("db", "", "Test database functions"));
        options.addOption(Option("keep-db", "", "Do not delete database file after database test"));
        options.addOption(Option("keep-data", "", "Do not delete data in database file after database test"));
        options.addOption(Option("fex", "", "Test feature extraction (except MFCCs)"));
        options.addOption(Option("fs", "", "Test feature selection"));
        options.addOption(Option("htk", "", "Test HTK export"));
        options.addOption(Option("ica", "", "Test ICA functions"));
        options.addOption(Option("matrix", "", "Test matrix functions"));
        options.addOption(Option("mfcc", "", "Test extraction of MFCCs", false, "<file>", true));
        options.addOption(Option("mel", "", "Test Mel filter and synth"));
        options.addOption(Option("mh", "", "Test Minimum Heap implementation"));
        options.addOption(Option("mt", "", "Test (spectrogram) matrix transformations"));
        options.addOption(Option("nmd", "", "Test NMD routines (correctness)"));
        options.addOption(Option("nmd-conv", "", "Test NMD routines (convergence speed)"));
        options.addOption(Option("nmf", "", "Test NMF routines"));
        options.addOption(Option("pca", "", "Test PCA routine"));
        options.addOption(Option("snmf", "", "Test sparse NMF routines"));
        options.addOption(Option("cnmf", "", "Test continuous NMF routines"));
        options.addOption(Option("spec", "", "Test spectral analysis functions"));
        options.addOption(Option("storage", "", "Test StorageManager functions"));
        options.addOption(Option("sc", "", "Test scaling and smoothing"));
        options.addOption(Option("svm", "", "Test classification using SVMModel class"));
        options.addOption(Option("td", "", "Test inter-task dependency functionality of the QueuedTaskManager"));
        options.addOption(Option("vector", "", "Test vector functions"));
        options.addOption(Option("wave", "", "Test i/o-capabilities of AudioData", false, "<audio-file>", true));
        options.addOption(Option("all", "", "Run all of the above tests"));
    }


    void handleOption(const string& name, const string& value)
    {
        Application::handleOption(name, value);

        static DatabaseSubsystemTest* dbTest = new DatabaseSubsystemTest();

        if (name == "all" || name == "matrix")
            pushTest(new MatrixTest());
        if (name == "all" || name == "spec")
            pushTest(new SpectralAnalysisTest());
        if (name == "all" || name == "vector")
            pushTest(new VectorTest());
        if (name == "all" || name == "wave")
            pushTest(new WaveTest(value));
        if (name == "all" || name == "ica")
            pushTest(new ICATest());
        if (name == "all" || name == "pca")
            pushTest(new PCATest());
        if (name == "all" || name == "mfcc")
            pushTest(new MFCCTest(value));
        if (name == "all" || name == "mh")
            pushTest(new MinHeapTest());
        if (name == "all" || name == "fex")
            pushTest(new FeatureExtractionTest());
        if (name == "all" || name == "fs")
            pushTest(new FeatureSelectionTest());
        if (name == "all" || name == "htk")
            pushTest(new HTKWriterTest());
        if (name == "all" || name == "sc")
            pushTest(new ScalingTest());
        if (name == "all" || name == "mel")
            pushTest(new MelFilterTest());
        if (name == "all" || name == "mt")
            pushTest(new MTrTest());
        if (name == "all" || name == "nmd")
            pushTest(new NMDTest());
        if (name == "all" || name == "nmf")
            pushTest(new NMFTest());
        if (name == "all" || name == "snmf") 
            pushTest(new SNMFTest());
        if (name == "all" || name == "cnmf")
            pushTest(new CNMFTest());
        if (name == "all" || name == "brw")
            pushTest(new BinaryReaderWriterTest());
        if (name == "all" || name == "td")
            pushTest(new TaskDepsTest());
        if (name == "all" || name == "db") {
            addSubsystem(new blissart::DatabaseSubsystem());
            pushTest(dbTest);
        }
        if (name == "keep-db") {
            dbTest->keepDB(true);
        }
        if (name == "keep-data") {
            dbTest->keepData(true);
        }
        if (name == "all" || name == "storage") {
            addSubsystem(new blissart::StorageSubsystem());
            pushTest(new StorageSubsystemTest());
        }
        if (name == "all" || name == "svm")
            pushTest(new SVMModelTest());
    }


    void pushTest(Testable* test)
    {
        vector<Testable *>::iterator it;
        for (it = _tests.begin(); it != _tests.end(); it++) {
            if (!strcmp((*it)->name(), test->name()))
                break;
        }
        if (it == _tests.end())
            _tests.push_back(test);
    }


    int main(const vector<string>& args) {
        int returnValue = EXIT_SUCCESS;

        if (_tests.size() == 0) {
            HelpFormatter formatter(options());
            formatter.setUsage(commandName() + " <options>");
            formatter.setHeader("openBliSSART Testsuite");
            formatter.setUnixStyle(true);
            formatter.setAutoIndent();
            formatter.format(cerr);
            returnValue = EXIT_USAGE;
        } else {
            // Perform all desired tests
            for (vector<Testable *>::iterator it = _tests.begin(); it != _tests.end(); it++) {
                cout << "#### Performing " << (*it)->name() << ":" << endl;
                if (!((*it)->performTest())) {
                    cout.flush();
                    cerr << "!!! ERROR" << endl;
                    break;
                }
            }

            // Deletion of the vector's elements has to be separate because
            // the testing process could be interrupted by errors
            for (vector<Testable *>::iterator it = _tests.begin(); it != _tests.end(); it++)
                delete (*it);
            _tests.clear();
        }

        return returnValue;
    }


    vector<Testable*> _tests;
};


POCO_APP_MAIN(Testsuite);

