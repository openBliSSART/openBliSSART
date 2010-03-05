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


#include "BrowserMainWindow.h"
#include "BrowserController.h"
#include "FeatureExtractionDialog.h"
#include "PreferencesDlg.h"
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>

#include <Poco/Util/LayeredConfiguration.h>

#include <QAction>
#include <QApplication>
#include <QMenuBar>
#include <QMessageBox>


using namespace std;
using Poco::Util::LayeredConfiguration;


#define MINIMUM_WIDTH 1024
#define MINIMUM_HEIGHT 768


namespace blissart {


BrowserMainWindow::BrowserMainWindow() : QMainWindow()
{
    setCentralWidget(new BrowserController(this));
    setupMenu();
    setMinimumSize(MINIMUM_WIDTH, MINIMUM_HEIGHT);

    LayeredConfiguration &cfg = BasicApplication::instance().config();
    resize(cfg.getInt("browser.mainwindow.width", MINIMUM_WIDTH),
           cfg.getInt("browser.mainwindow.height", MINIMUM_HEIGHT));
    if (cfg.getBool("browser.mainwindow.isMaximized", false))
        setWindowState(Qt::WindowMaximized);
};


void BrowserMainWindow::handlePreferences()
{
    PreferencesDlg dlg(this);
    dlg.exec();
}


void BrowserMainWindow::closeEvent(QCloseEvent *ev)
{
    LayeredConfiguration &cfg = BasicApplication::instance().config();
    cfg.setBool("browser.mainwindow.isMaximized", isMaximized());
    cfg.setInt("browser.mainwindow.width", normalGeometry().width());
    cfg.setInt("browser.mainwindow.height", normalGeometry().height());

    return QMainWindow::closeEvent(ev);
}


void BrowserMainWindow::setupMenu()
{
    QMenu *dbMenu = menuBar()->addMenu(tr("Database"));

#ifdef Q_WS_MAC
    dbMenu->addAction(tr("Preferences"), this, SLOT(handlePreferences()));
#else
    QMenu *editMenu = menuBar()->addMenu(tr("Edit"));
    editMenu->addAction(tr("Preferences"), this, SLOT(handlePreferences()));
#endif

    dbMenu->addAction(tr("Import audio"),
                        centralWidget(), SLOT(handleImportAudio()));
    dbMenu->addAction(tr("Create label"),
                        centralWidget(), SLOT(handleCreateLabel()));
    dbMenu->addAction(tr("Create response"),
                        centralWidget(), SLOT(handleCreateResponse()));
    dbMenu->addSeparator();
    dbMenu->addAction(tr("Extract features from all data descriptors"),
                        this, SLOT(handleFeatureExtraction()));

    QMenu *viewMenu = menuBar()->addMenu(tr("View"));
    viewMenu->addAction(tr("Refresh view"),
                        centralWidget(), SLOT(handleRefreshTreeWidget()))
            ->setShortcut(Qt::Key_F5);
}


void BrowserMainWindow::handleFeatureExtraction()
{
    DatabaseSubsystem &dbs =
        BasicApplication::instance().getSubsystem<DatabaseSubsystem>();

    set<DataDescriptorPtr> descriptors;
    foreach (const ProcessPtr &pPtr, dbs.getProcesses()) {
        vector<DataDescriptorPtr> dd = dbs.getDataDescriptors(pPtr->processID);
        foreach (const DataDescriptorPtr &dPtr, dd) {
            if (dPtr->available)
                descriptors.insert(dPtr);
        }
    }

    if (descriptors.empty()) {
        QMessageBox::information(this, windowTitle(),
            tr("There are currently no data descriptors in the database."));
        return;
    }

    FeatureExtractionDialog dlg(descriptors, this);
    dlg.exec();

    static_cast<BrowserController *>(centralWidget())->handleRefreshTreeWidget();
}


} // namespace blissart
