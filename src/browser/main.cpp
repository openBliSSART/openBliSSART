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


#include "BrowserMainWindow.h"
#include <blissart/ThreadedApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/audio/audio.h>

#include <Poco/Path.h>
#include <Poco/File.h>
#include <Poco/Util/PropertyFileConfiguration.h>

#include <QApplication>
#include <QMessageBox>


using namespace blissart;
using namespace std;


class BrowserApp : public ThreadedApplication
{
public:
    BrowserApp(int argc, char **argv) :
        _argc(argc),
        _argv(argv)
    {
        // Add the database- and storage subsystems.
        addSubsystem(new DatabaseSubsystem());
        addSubsystem(new StorageSubsystem());
    }

    virtual ~BrowserApp()
    {
        // Save the configuration.
        uninitializeConfiguration();

        // Shut down LibAudio.
        blissart::audio::shutdown();
    }


protected:
    virtual void initialize(Poco::Util::Application &self)
    {
        BasicApplication::initialize(self);

        // Initialize LibAudio.
        blissart::audio::initialize();

        // Initialize the configuration.
        initializeConfiguration();

        // Initialize the task manager.
        initializeTaskManager<ThreadedApplication>();
    }

    void initializeConfiguration()
    {
        // Call the base-class implementation first.
        BasicApplication::initializeConfiguration();

        // Determine the configuration's filename. Create the file if
        // neccessary.
        _cfgFileName =
            Poco::Path::forDirectory(config().getString("blissart.configDir"));
        _cfgFileName.setFileName("browser.properties");
        if (!Poco::File(_cfgFileName).exists())
            Poco::File(_cfgFileName).createFile();
        // Add the configuration.
        _config =
            new Poco::Util::PropertyFileConfiguration(_cfgFileName.toString());
        config().addWriteable(_config, PRIO_APPLICATION);
    }

    void uninitializeConfiguration()
    {
        _config->save(_cfgFileName.toString());
        _config->release();
    }

    virtual int main(const vector<string>&)
    {
        QApplication qtApp(_argc, _argv);

        try {
            BrowserMainWindow mw;
            mw.show();
            mw.raise();
            return qtApp.exec();
        } catch (Poco::Exception &ex) {
            logger().log(ex);
            QMessageBox::critical(NULL, QObject::tr("Error"),
                QObject::tr("The following exception was raised:\n\n%1")
                    .arg(QString::fromStdString(ex.displayText())));
            return EXIT_FAILURE;
        } catch (std::exception &ex) {
            logger().error(ex.what());
            QMessageBox::critical(NULL, QObject::tr("Error"),
                QObject::tr("The following exception was raised:\n\n%1")
                    .arg(QString::fromStdString(ex.what())));
            return EXIT_FAILURE;
        } catch (...) {
            logger().error("An unknown error occured!");
            return EXIT_FAILURE;
        }
    }


private:
    Poco::Util::PropertyFileConfiguration* _config;
    Poco::Path                             _cfgFileName;
    // _argc and _argv must be passed through to QApplication.
    int                                    _argc;
    char**                                 _argv;
};


int main(int argc, char **argv)
{
    Poco::AutoPtr<BrowserApp> app = new BrowserApp(argc, argv);

    try {
        app->init(argc, argv);
    } catch (Poco::Exception &ex) {
        app->logger().log(ex);
        return EXIT_FAILURE;
    }

    return app->run();
}
