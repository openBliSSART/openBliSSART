//
// $Id: EditWidgetClassificationObject.cpp 855 2009-06-09 16:15:50Z alex $
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


#include "EditWidgetClassificationObject.h"
#include <blissart/Label.h>
#include <blissart/AudioObject.h>
#include <blissart/audio/AudioData.h>

#include <vector>
#include <cassert>

#include <QMessageBox>
#include <QProgressDialog>


using namespace std;
using namespace blissart::audio;
using namespace blissart::linalg;


namespace blissart {


EditWidgetClassificationObject::EditWidgetClassificationObject(ClassificationObjectPtr clo,
                                                               QWidget *parent) :
    EditWidget(parent),
    _clo(clo),
    _hasSound(false),
    _progressDlg(0)
{
    _ui.setupUi(this);

    // Load all available labels from the database.
    loadLabels();

    // First display the data.
    displayData();

    // Enable or disable the preview groupbox depending on the preferences.
    _ui.groupBoxPreview->setChecked(
        BasicApplication::instance()
        .config().getBool("browser.preview.alwaysEnabled", true)
    );

    // Nothing's been edited until now. Reset the "dirty" flag.
    setDirty(false);
}


void EditWidgetClassificationObject::loadLabels()
{
    _ui.lwLabels->clear();
    vector<LabelPtr> labels = dbSubsystem().getLabels();
    foreach (LabelPtr l, labels) {
        QListWidgetItem *lwi = new QListWidgetItem(_ui.lwLabels);
        // The displated text of the item is a concatenation of the original;
        // label's text plus its id encloses in parentheses. This is neccessary
        // to circumvent a bug in Qt where labels with the same displayed text
        // sometimes get mixed up when the current item changes and stuff like
        // that. The reason for this is that Qt uses text-based search instead
        // of something more useful and is hence not able to distinguish between
        // model (ref AbstractItemModel) items at all times.
        lwi->setText(
                QString("%1 (%2)")
                .arg(QString::fromStdString(l->text))
                .arg(l->labelID)
        );
        lwi->setData(Qt::UserRole, l->labelID);
        lwi->setFlags(Qt::ItemIsSelectable |
                      Qt::ItemIsUserCheckable |
                      Qt::ItemIsEnabled);
        lwi->setCheckState(Qt::Unchecked);
    }
}


bool EditWidgetClassificationObject::loadDataDescriptorsAndComputeSamples()
{
    // Set up a progress dialog.
    QProgressDialog progDlg(tr("Converting the spectrum to samples..."),
                            QString(), 0, 100, this);
    progDlg.setWindowModality(Qt::WindowModal);
    progDlg.setAutoReset(false);
    progDlg.show();

    // Start the transformation to an AudioData object.
    _progressDlg = &progDlg;
    Poco::SharedPtr<AudioData> ad;
    try {
        ad = AudioObject::getAudioObject(_clo, this);
    }
    catch (Poco::Exception& ex) {
        QMessageBox::warning(this, windowTitle(),
            tr("The transformation failed for the following reason:\n\n%1")
            .arg(ex.displayText().c_str()));
        _progressDlg = 0;
        return false;
    }
    catch (std::exception &ex) {
        QMessageBox::warning(this, windowTitle(),
            tr("The transformation failed for the following reason:\n\n%1")
            .arg(ex.what()));
        _progressDlg = 0;
        return false;
    }
    _progressDlg = 0;

    // Normalize the samples.
    if (BasicApplication::instance()
        .config().getBool("browser.preview.normalizeAudio", true))
    {
        ad->normalize();
    }

    // Initialize the sample visualization widget.
    _hasSound = _ui.svWidget->setSamples(ad->getChannel(0),
                                         ad->nrOfSamples(),
                                         ad->sampleRate());
    if (!_hasSound) {
        _ui.groupBoxPreview->setTitle(
                _ui.groupBoxPreview->title() +
                tr(" (sample frequency %1Hz out of range)")
                .arg(ad->sampleRate())
        );
    }

    // We're all good.
    return true;
}


void EditWidgetClassificationObject::displayData()
{
    // Iterate over all labels and set their checked state depending on the
    // classification object's label ids.
    for (int i = 0; i < _ui.lwLabels->count(); i++) {
        QListWidgetItem *lwi = _ui.lwLabels->item(i);
        const int labelID = lwi->data(Qt::UserRole).toInt();
        set<int>::const_iterator it =
            std::find(_clo->labelIDs.begin(), _clo->labelIDs.end(), labelID);
        if (it == _clo->labelIDs.end())
            lwi->setCheckState(Qt::Unchecked);
        else
            lwi->setCheckState(Qt::Checked);
    }
}


void EditWidgetClassificationObject::handleSave()
{
    // Reflect the changes.
    _clo->labelIDs.clear();
    for (int i = 0; i < _ui.lwLabels->count(); i++) {
        QListWidgetItem *lwi = _ui.lwLabels->item(i);
        if (lwi->checkState() == Qt::Checked)
            _clo->labelIDs.insert(lwi->data(Qt::UserRole).toInt());
    }
    // Save the changes.
    dbSubsystem().updateClassificationObject(_clo);
    // Let mother do the rest.
    EditWidget::handleSave();
}


void EditWidgetClassificationObject::handleCancel()
{
    // Since the entity's properties haven't been updated yet it's sufficient to
    // just display all the data again.
    displayData();
    // Let mother do the rest.
    EditWidget::handleCancel();
}


void EditWidgetClassificationObject::on_lwLabels_itemChanged(QListWidgetItem *)
{
    bool dirty = false;
    for (int i = 0; i < _ui.lwLabels->count(); i++) {
        QListWidgetItem *lwi = _ui.lwLabels->item(i);
        const int labelID = lwi->data(Qt::UserRole).toInt();
        set<int>::const_iterator it = _clo->labelIDs.find(labelID);

        if (lwi->checkState() == Qt::Checked && it == _clo->labelIDs.end()) {
            // If the item is checked but doesn't appear in the classification
            // object's set of labelIDs, then we're dirty.
            dirty = true;
            break;
        } else if (it != _clo->labelIDs.end()) {
            // If the item isn't checked but does appear in the classification
            // object's set of labelIDs, then we're dirty.
            dirty = true;
            break;
        }
    }
    setDirty(dirty);
}

void EditWidgetClassificationObject::on_groupBoxPreview_toggled(bool on)
{
    if (!on || _hasSound)
        return;

    // Load the associated data descriptors and initialize the samples
    // visualization widget.
    // If either the return value from loadDataDescriptorsAndComputeSamples is
    // false or there's no _sound object, the groupbox is unchecked and
    // disabled.
    if (!loadDataDescriptorsAndComputeSamples()) {
        _ui.groupBoxPreview->setChecked(false);
        _ui.groupBoxPreview->setEnabled(false);
    }
}


DatabaseEntityPtr EditWidgetClassificationObject::entityPtr() const
{
	return _clo;
}


void EditWidgetClassificationObject::progressChanged(float progress)
{
    _progressDlg->setValue(std::min<int>(100, progress * 100));
}


} // namespace blissart
