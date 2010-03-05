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


#include "ThreadedDialog.h"

#include <QProgressDialog>
#include <QApplication>


namespace blissart {


ThreadedDialog::ThreadedDialog(QWidget *parent) :
    QDialog(parent),
    _threadedApp(
        static_cast<ThreadedApplication &>(ThreadedApplication::instance())
    )
{
    _threadedApp.registerListener(this);
}


ThreadedDialog::~ThreadedDialog()
{
    _threadedApp.unregisterListener(this);
}


bool ThreadedDialog::waitForCompletion(const std::string &message)
{
    // Initialize a progress dialog.
    QProgressDialog progDlg(tr(message.c_str()), tr("Cancel"), 0, 100, this);
    progDlg.setWindowModality(Qt::WindowModal);
    progDlg.setAutoReset(false);
    progDlg.show();

    while (progress() < 1.0f) {
        yield(500);
        progDlg.setValue(std::min<int>(100, (int)(progress() * 100)));
        if (progDlg.wasCanceled())
            break;
        qApp->processEvents();
    }
    progDlg.hide();

    if (progDlg.wasCanceled()) {
        // Cancelled.
        QProgressDialog cancelDlg(
            tr("Cancelling any active threads. This may take a while..."),
            QString(), 0, 2, this
        );
        cancelDlg.setWindowModality(Qt::WindowModal);
        cancelDlg.setAutoReset(false);
        cancelDlg.setValue(1);
        cancelDlg.show();
        qApp->processEvents();

        cancelAll();
        joinAll();
    } else
        joinAll();

    return progDlg.wasCanceled();
}


} // namespace blissart
