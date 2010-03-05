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
#include <Poco/Path.h>
#include <Poco/File.h>
#include <Poco/Mutex.h>
#include <Poco/Exception.h>
#include <fstream>
#include <iostream>


using namespace Poco;
using namespace Poco::Util;
using namespace std;


static Poco::FastMutex _fftwMutex;


namespace blissart {


BasicApplication::BasicApplication() : _echoCommand(false)
{
    setUnixOptions(true);
}


void BasicApplication::lockFFTW()
{
    _fftwMutex.lock();
}


void BasicApplication::unlockFFTW()
{
    _fftwMutex.unlock();
}


vector<string>
BasicApplication::parseScriptFiles(const vector<string>& fileNames)
{
    vector<string> result;

    for (vector<string>::const_iterator it = fileNames.begin();
        it != fileNames.end(); ++it)
    {
        ifstream scriptIS(it->c_str());
        if (scriptIS.fail())
            throw Poco::IOException("Could not open script file: " + *it);

        string fileName;
        while (!scriptIS.eof()) {
            getline(scriptIS, fileName);
            if (fileName != "")
                result.push_back(fileName);
        }
        scriptIS.close();
    }

    return result;
}


void BasicApplication::initialize(Application& self)
{
    // Don't use logger here because it is not configured yet.
    if (_echoCommand) {
        cout << "Executing: " << commandName() << _optionsString << endl;
    }

    initializeDirectories();
    initializeConfiguration();

    // Application::initialize must _NOT_ be called before
    // initialize(Directories|Configuration) or else any subsystems will be
    // initialized without any prior configuration of directories, etc.
    Application::initialize(self);
}


void BasicApplication::initializeDirectories()
{
    Path dir = Path::forDirectory(config().getString("application.dir"));

    config().setString("blissart.binDir", dir.toString());

    dir.popDirectory();
    config().setString("blissart.installDir", dir.toString());

    dir.pushDirectory("etc");
    File(dir).createDirectory();
    config().setString("blissart.configDir", dir.toString());
    dir.popDirectory();

    // The storage subsystem's realm...
    dir.pushDirectory("storage");
    File(dir).createDirectory();
    config().setString("blissart.storageDir", dir.toString());
    dir.popDirectory();

    // Where the database subsystem roams...
    dir.pushDirectory("db");
    File(dir).createDirectory();
    config().setString("blissart.databaseDir", dir.toString());
    dir.setFileName("openBliSSART.db");
    config().setString("blissart.databaseFile", dir.toString());
}


void BasicApplication::initializeConfiguration()
{
    config().setString("logging.loggers.root.channel.class", "ConsoleChannel");
    config().setString("logging.loggers.root.level",
#ifdef _DEBUG
        "debug"
#else
        "information"
#endif
    );

    if (!_userConfigFile.empty()) {
        logger().information("Using configuration file: " + _userConfigFile);
        loadConfiguration(_userConfigFile);
    }
    else {
        Path configFileName = Path::forDirectory(config().getString("blissart.configDir"));
        configFileName.setFileName("blissart.properties");
        if (!File(configFileName).exists())
            File(configFileName).createFile();
        loadConfiguration(configFileName.toString());
    }
}


void BasicApplication::handleOption(const string& name, const string& value)
{
    if (name == "echo") {
        _echoCommand = true;
    }
    else if (name == "config") {
        _userConfigFile = value;
    }
    _optionsString += " --" + name;
    if (!value.empty()) {
        _optionsString += '=' + value;
    }
}


void BasicApplication::defineOptions(OptionSet& options)
{
    Application::defineOptions(options);

    options.addOption(Option("echo", "A", 
        "Echo the command that was used to start this application.",
        false));
    options.addOption(Option("config", "C",
        "Specifies a config file to use instead of "
        "blissart-dir/etc/blissart.properties",
        false, "<filename>", true));
}


} // namespace blissart

