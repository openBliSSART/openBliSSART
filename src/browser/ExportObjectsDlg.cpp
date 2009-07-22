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


#include "ExportObjectsDlg.h"
#include <blissart/audio/AudioData.h>
#include <blissart/audio/WaveEncoder.h>
#include <blissart/AudioObject.h>
#include <blissart/DatabaseSubsystem.h>

#include <Poco/LogStream.h>
#include <Poco/Message.h>

#include <QProgressDialog>
#include <QMessageBox>


using namespace std;
using namespace blissart::linalg;
using namespace blissart::audio;


namespace blissart {


ExportObjectsDlg::ExportObjectsDlg(const vector<ClassificationObjectPtr> &clos,
                                         QWidget *parent) :
    QFileDialog(parent),
    _clos(clos),
    _logger(BasicApplication::instance().logger())
{
    debug_assert(!_clos.empty());

    setFileMode(QFileDialog::DirectoryOnly);
    setAcceptMode(QFileDialog::AcceptSave);
}


void ExportObjectsDlg::accept()
{
    QFileDialog::accept();

    QString dest = selectedFiles().value(0);
    debug_assert(!dest.isEmpty());

    // Set up a progress dialog.
    QProgressDialog dlg(tr("Exporting the objects..."), tr("Cancel"),
                        0, _clos.size(), this);
    dlg.setMinimumDuration(500);

    // Now iterate over the list of classification objects and export each
    // object individually.
    const QDir destDir(dest);
    bool allWentWell = true;
    for (vector<ClassificationObjectPtr>::const_iterator it = _clos.begin();
        it != _clos.end(); ++it)
    {
        allWentWell &= exportClassificationObject(*it, destDir);
        dlg.setValue(dlg.value() + 1);
        if (dlg.wasCanceled())
            break;
    }

    if (!allWentWell) {
        QMessageBox::information(this, tr("Error"),
                tr("One or more objects couldn't be exported. Please check "
                   "the application's log for more details."));
    }
}


bool ExportObjectsDlg::exportClassificationObject(const ClassificationObjectPtr clo,
                                                  const QDir &destDir)
{
    DatabaseSubsystem &dbs =
        BasicApplication::instance().getSubsystem<DatabaseSubsystem>();

    Poco::LogStream ls(_logger);
    ls << "Exporting classification object with id ";
    ls << clo->objectID << endl;

    // Reconstruct AudioData object.
    Poco::SharedPtr<AudioData> pAd;
    try {
        pAd = AudioObject::getAudioObject(clo);
    }
    catch (Poco::Exception& exc) {
        ls.error();
        ls << "Object with id " + clo->objectID << " could not be exported"
           << endl;
        ls << "Reason: " << exc.displayText() << endl;
    }
    catch (std::exception& exc) {
        ls.error();
        ls << "Object with id " + clo->objectID << " could not be exported"
           << endl;
        ls << "Reason: " << exc.what() << endl;
    }

    // Determine the filename.
    ProcessPtr process = dbs.getProcess(clo);
    QString fileName = QString("%1_%2.wav")
        .arg(QFileInfo(QString::fromStdString(process->inputFile))
             .completeBaseName())
        .arg(clo->objectID);
    string finalPath = destDir.filePath(fileName).toStdString();

    // Eventually export the data and return the result.
    _logger.debug("Exporting " + finalPath);
    return WaveEncoder::saveAsWav(*pAd, finalPath);
}


} // namespace blissart
