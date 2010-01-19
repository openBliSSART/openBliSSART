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


#include <Poco/Util/Application.h>
#include <Poco/Util/HelpFormatter.h>

#include <vector>
#include <iostream>
#include <iomanip>
#include <cstring> // for strcmp

#include <blissart/audio/audio.h>

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
        options.addOption(Option("nmd", "", "Run NMD benchmark"));
        options.addOption(Option("nmf", "", "Run NMF benchmark"));
		options.addOption(Option("all", "", "Run all of the above benchmarks"));
    }


    void handleOption(const string& name, const string& value)
    {
        Application::handleOption(name, value);

        if (name == "all" || name == "nmd")
            pushBenchmark(new NMDBenchmark());
        if (name == "all" || name == "nmf")
            pushBenchmark(new NMFBenchmark());
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

				Poco::Timestamp::TimeDiff totalTime(0);
				Benchmark::ElapsedTimeVec et = (*it)->elapsedTimes();
				for (Benchmark::ElapsedTimeVec::const_iterator etItr = et.begin();
					etItr != et.end(); ++etItr)
				{
					cout << etItr->first << " took " 
						 << fixed << setprecision(2)
						 << ((double) etItr->second) / 1.0e6 << "s (" 
						 << etItr->second << "us)"
						 << endl;
					totalTime += etItr->second;
				}
				cout << "-----------------------" << endl;
				cout << (*it)->name() << " took "
					 << fixed << setprecision(2)
					 << ((double) totalTime) / 1.0e6 << "s (" 
					 << totalTime << "us)"
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

        return returnValue;
    }


    vector<Benchmark*> _benchs;
};


POCO_APP_MAIN(BenchmarkApp);

