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
#include <iomanip>
#include <cstring> // for strcmp
#include <fstream>

#include <blissart/audio/audio.h>

#ifdef HAVE_CUDA
#include <blissart/linalg/GPUUtil.h>
#endif

#include "NMDBenchmark.h"
#include "NMFBenchmark.h"


using namespace std;
using namespace benchmark;
using namespace Poco::Util;


class BenchmarkApp : public Application
{
public:
    BenchmarkApp()
    {
        setUnixOptions(true);
    }


protected:
    virtual void initialize(Application& self)
    {
        // Base class initialization.
        Application::initialize(self);

        self.config().setString("blissart.databaseFile", "BenchmarkApp.db");
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

#ifdef HAVE_CUDA
        blissart::linalg::GPUStart();
#endif
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
        void (*addOptions[])(Poco::Util::OptionSet&) = {
            &NMFBenchmark::addOptions,
            0
        };
        for (void (**fp)(Poco::Util::OptionSet&) = addOptions; *fp != 0; ++fp) {
            (*fp)(options);
        }
        options.addOption(Option("nmd", "", "Run NMD benchmark"));
        options.addOption(Option("nmf", "", "Run NMF benchmark"));
		options.addOption(Option("all", "", "Run all of the above benchmarks"));
        options.addOption(Option("log", "l", "File name for results", 
            false, "<file>", true));
    }


    void handleOption(const string& name, const string& value)
    {
        Application::handleOption(name, value);

        if (name == "all" || name == "nmd")
            pushBenchmark(new NMDBenchmark());
        else if (name == "all" || name == "nmf")
            pushBenchmark(new NMFBenchmark());
        else if (name == "log") {
            _outputToFile = true;
            _logFile = value;
        }
        else {
            _bOptions[name] = value;
        }
    }


    void pushBenchmark(Benchmark* test)
    {
        vector<Benchmark *>::const_iterator it;
        for (it = _benchs.begin(); it != _benchs.end(); it++) {
            if (!strcmp((*it)->name(), test->name()))
                break;
        }
        if (it == _benchs.end())
            _benchs.push_back(test);
    }


    int main(const vector<string>& args) {
		int returnValue = EXIT_OK;

        if (_benchs.size() == 0) {
            HelpFormatter formatter(options());
            formatter.setUsage(commandName() + " <options>");
            formatter.setHeader("openBliSSART benchmark suite");
            formatter.setUnixStyle(true);
            formatter.setAutoIndent();
            formatter.format(cerr);
            returnValue = EXIT_USAGE;
        } else {
            // Perform all desired tests
            for (vector<Benchmark *>::iterator it = _benchs.begin(); 
				it != _benchs.end(); it++) 
			{
                (*it)->setOptions(_bOptions);
                cout << "#### Performing " << (*it)->name() << ":" << endl;
				try {
					(*it)->run();
				}
				catch (const Poco::Exception& exc) {
					cerr << "ERROR: " << exc.displayText() << endl;
				}
				catch (const std::exception& exc) {
					cerr << "ERROR: " << exc.what() << endl;
				}
				catch (...) {
					cerr << "UNKNOWN ERROR" << endl;
				}

				Benchmark::ElapsedTime totalTime(0);
				Benchmark::ElapsedTimeMap et = (*it)->elapsedTimes();
				for (Benchmark::ElapsedTimeMap::const_iterator etItr = et.begin();
					etItr != et.end(); ++etItr)
				{
					cout << etItr->first << "\t" 
						 << fixed << setprecision(2)
						 << etItr->second << "s"
						 << endl;
                    if (_outputToFile) {
                        std::ofstream logS(_logFile.c_str(), ios_base::app | ios_base::out);
                        logS << etItr->first << "\t" 
						     << fixed << setprecision(2)
						     << etItr->second << "s"
						     << endl;
                    }
					totalTime += etItr->second;
				}
				cout << "-----------------------" << endl;
				cout << (*it)->name() << "\t"
					 << fixed << setprecision(2)
					 << totalTime << "s"
					 << endl;
            }

            // Deletion of the vector's elements has to be separate because
            // the benchmark process could be interrupted by errors
            for (vector<Benchmark *>::iterator it = _benchs.begin(); 
				it != _benchs.end(); it++)
			{
                delete (*it);
			}
            _benchs.clear();
        }

#ifdef HAVE_CUDA
        blissart::linalg::GPUStop();
#endif

        return returnValue;
    }


    vector<Benchmark*> _benchs;
    string _logFile;
    bool   _outputToFile;
    map<string, string> _bOptions;
};


POCO_APP_MAIN(BenchmarkApp);

