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


#include "EditWidgetResponse.h"
#include "LabelSelectionDialog.h"
#include "ResponseQualityDlg.h"

#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/ClassificationObject.h>
#include <blissart/DataSet.h>

#include <QHeaderView>
#include <QMenu>
#include <QMessageBox>
#include <QProgressDialog>


using namespace std;


namespace blissart {


EditWidgetResponse::EditWidgetResponse(ResponsePtr response,
                                       QWidget *parent) :
    EditWidget(parent),
    _response(response)
{
    _ui.setupUi(this);
    _ui.twRelations->horizontalHeader()->setStretchLastSection(true);

    // Initialize the _clos and _labels map.
    initializeCLOsAndLabels();

    // First display the data.
    displayData();
    
    // Then connect the neccessary signals/slots for tracking the edit status.
    connect(_ui.leName,
            SIGNAL(textChanged(const QString &)),
            SLOT(setDirty()));
    connect(_ui.teDescription,
            SIGNAL(textChanged()),
            SLOT(setDirty()));
    connect(_ui.twRelations,
            SIGNAL(itemSelectionChanged()),
            SLOT(updateButtonStates()));
    
    // Nothing's been edited until now. Reset the "dirty" flag.
    setDirty(false);    
}


void EditWidgetResponse::handleSave()
{
    // Sanity check
    if (_ui.leName->text().isEmpty()) {
        QMessageBox::warning(this, tr("Stop"),
                             tr("The response's name must not be empty."));
        _ui.leName->setFocus();
        return;
    }
	// Reflect the changes.
    _response->name = _ui.leName->text().toStdString();
    _response->description = _ui.teDescription->toPlainText().toStdString();
    _response->labels = _cloLabelsRelation;
    // Save the changes.
    dbSubsystem().updateResponse(_response);
    // Let mother do the rest.
    EditWidget::handleSave();
}


void EditWidgetResponse::handleCancel()
{
    // Reinitialize the relevant data from the database.
    initializeCLOsAndLabels();
    // Since the entity's properties haven't been updated yet it's sufficient to
    // just display all the data again.
    displayData();
    // Let mother do the rest.
    EditWidget::handleCancel();
}


void EditWidgetResponse::on_pbAddByLabel_clicked()
{
    LabelSelectionDialog lsd(set<int>(), true, this);
    if (QDialog::Accepted != lsd.exec())
        return;
    
    vector<LabelPtr> selectedLabels = lsd.selectedLabels();
    if (selectedLabels.size() == 0)
        return;

    QProgressDialog progDlg(tr("Inserting or updating relations..."), QString(),
                            0, selectedLabels.size(), this);
    progDlg.setMinimumDuration(1000);
    progDlg.setWindowModality(Qt::WindowModal);
    
    int i = 0;
    _ui.twRelations->setSortingEnabled(false);
    foreach (LabelPtr l, selectedLabels) {
        debug_assert(!l.isNull());

        // Retrieve all classification objects that are associated with the selected
        // label's id.
        vector<ClassificationObjectPtr> clos =
            dbSubsystem().getClassificationObjectsForLabel(l->labelID);
    
        // Insert them into the table widget.
        foreach (ClassificationObjectPtr clo, clos) {
            insertOrUpdateRelation(clo, l);
        }
        progDlg.setValue(++i);
    }
    _ui.twRelations->setSortingEnabled(true);
    
    // Update the button states.
    updateButtonStates();
}


void EditWidgetResponse::on_pbSelectLabel_clicked()
{
    const int currentRow = _ui.twRelations->currentRow();
    QTableWidgetItem *item = _ui.twRelations->item(currentRow, 0);
    ClassificationObjectPtr clo = _clos[item->data(Qt::UserRole).toInt()];
    LabelSelectionDialog lsd(clo->labelIDs, false, this);
    if (QDialog::Accepted == lsd.exec()) {
        _ui.twRelations->setSortingEnabled(false);
        insertOrUpdateRelation(clo, lsd.selectedLabels()[0]);
        _ui.twRelations->setSortingEnabled(true);
    }
}


void EditWidgetResponse::on_pbRemoveSelected_clicked()
{
    // Unfortunately we cannot simply iterate over the selectedItems()
    // and remove the associated rows because the table widget consequently
    // deletes all items of a row. So if we still did we might (and will) access
    // void pointers.
    // This is why we first determine which rows to delete.
    set<int> delRows;
    foreach (QTableWidgetItem *item, _ui.twRelations->selectedItems())
        delRows.insert(item->row());
    
    for (set<int>::reverse_iterator rit = delRows.rbegin();
        rit != delRows.rend(); ++rit)
    {
        ClassificationObjectPtr clo =
            _clos[_ui.twRelations->item(*rit, 0)->data(Qt::UserRole).toInt()];
    }    

    // Then remove the rows in descending order (this is neccessary because row
    // numbers would change if we deleted bottom-up instead of top-down).
    QProgressDialog progDlg(tr("Removing relations..."), QString(),
            0, delRows.size(), this);
    int i = 0;
    for (set<int>::reverse_iterator rit = delRows.rbegin();
        rit != delRows.rend(); ++rit)
    {
        ClassificationObjectPtr clo =
            _clos[_ui.twRelations->item(*rit, 0)->data(Qt::UserRole).toInt()];
        removeRelation(clo);
        progDlg.setValue(++i);
    }
    
    // Update the button states.
    updateButtonStates();
}


void EditWidgetResponse::on_pbQualityFeedback_clicked()
{
    // Retrieve the dataset corresponding to the current relations of
    // classification objects and labels.
    DataSet _dataSet = dbSubsystem().getDataSet(_cloLabelsRelation);
    if (_dataSet.empty()) {
        QMessageBox::information(this, tr("Empty dataset"),
            tr("There are currently no features associated with this response. "
               "Aborting."));
        return;
    }
    // Shoot.
    ResponseQualityDlg dlg(_dataSet, _response, this);
    dlg.exec();
}


void EditWidgetResponse::on_twRelations_customContextMenuRequested(const QPoint &pos)
{
    if (_ui.twRelations->selectedItems().empty())
        return;
    
    QMenu pm(this);
    
    QTableWidgetItem *item = _ui.twRelations->item(_ui.twRelations->currentRow(), 0);
    int objectID = item->data(Qt::UserRole).toInt();
    if (objectID && _clos[objectID]->labelIDs.size() > 1) {
        pm.addAction(tr("Select label to use"), this, SLOT(handleSelectLabel()));
        pm.addSeparator();
    }
    
    pm.addAction(tr("Delete selected items"), 
                 this, SLOT(on_pbRemoveSelected_clicked()));
    
    pm.exec(_ui.twRelations->mapToGlobal(pos));
}


void EditWidgetResponse::updateButtonStates()
{
    QTableWidgetItem *current = _ui.twRelations->currentItem();
    
    _ui.pbRemoveSelected->setEnabled(current);
    _ui.pbQualityFeedback->setEnabled(_ui.twRelations->rowCount() > 0);
    
    bool enableLabelSelection = false;
    if (current) {
        QTableWidgetItem *cloItem = _ui.twRelations->item(current->row(), 0);
        int objectID = cloItem->data(Qt::UserRole).toInt();
        if (_clos[objectID]->labelIDs.size() > 1)
            enableLabelSelection = true;
    }
    _ui.pbSelectLabel->setEnabled(enableLabelSelection);
}


void EditWidgetResponse::displayData()
{
    _ui.leName->setText(QString::fromStdString(_response->name));
    _ui.teDescription->setText(QString::fromStdString(_response->description));
    
    // Set up the relations table widget.
    QStringList headers;
    headers << tr("Type") << tr("ID") << tr("Label");
    _ui.twRelations->setHorizontalHeaderLabels(headers);
    _ui.twRelations->clear();
    _ui.twRelations->setRowCount(_cloLabelsRelation.size());
    _ui.twRelations->setSortingEnabled(false);
    pair<int, int> kvp;
    int row = 0;
    foreach (kvp, _cloLabelsRelation) {
        ClassificationObjectPtr clo = _clos[kvp.first];
        LabelPtr label = _labels[kvp.second];
        setTableWidgets(clo, label, row++);
    }
    _ui.twRelations->sortItems(0);
    _ui.twRelations->setSortingEnabled(true);
    
    // Update the button states.
    updateButtonStates();
}


void EditWidgetResponse::initializeCLOsAndLabels()
{
    // Remove any possible prior mappings.
    _clos.clear();
    _labels.clear();
    _cloLabelsRelation.clear();
    
    // Get all classification objects and labels that are associated with the
    // underlying response.
    vector<CLOLabelPair> relations =
        dbSubsystem().getClassificationObjectsAndLabelsForResponse(_response);
    
    QProgressDialog progDlg(tr("Initializing the relations between "
            "classification objects and labels..."), QString(),
            0, relations.size(), this);
    progDlg.setMinimumDuration(1000);
    progDlg.setWindowModality(Qt::WindowModal);
    
    // Insert the relations.
    _ui.twRelations->setSortingEnabled(false);
    int i = 0;
    foreach (const CLOLabelPair& p, relations) {
        insertOrUpdateRelation(p.first, p.second);
        if (++i % 100 == 0)
            progDlg.setValue(i);
    }
    progDlg.setValue(relations.size());
    _ui.twRelations->setSortingEnabled(true);
}


void EditWidgetResponse::insertOrUpdateRelation(const ClassificationObjectPtr &clo,
                                                const LabelPtr &label)
{
    // Sorting must be disabled because otherwise sorting would occur as soon as
    // we insert a new item and then our internal tracking of the relevant row
    // would be erroneous.
    debug_assert(!_ui.twRelations->isSortingEnabled());
    
    int row = -1;
    if (_cloLabelsRelation.find(clo->objectID) == _cloLabelsRelation.end()) {
        // Insert.
        _clos[clo->objectID] = clo;
        _labels[label->labelID] = label;
        _cloLabelsRelation[clo->objectID] = label->labelID;
        _ui.twRelations->insertRow(0);
        row = 0;
    } else {
        // Update.
        LabelPtr oldLabel = _labels[_cloLabelsRelation[clo->objectID]];
        _clos[clo->objectID] = clo;
        _labels[label->labelID] = label;
        _cloLabelsRelation[clo->objectID] = label->labelID;
        removeLabelIfApplicable(oldLabel);
        row = findRowForClassificationObject(clo);
    }
    debug_assert(row >= 0 && row < _ui.twRelations->rowCount());

    // Then create the new items.
    setTableWidgets(clo, label, row);
    
    // Set the "dirty" flag.
    setDirty(true);
}


void EditWidgetResponse::removeRelation(ClassificationObjectPtr &clo)
{
    const int row = findRowForClassificationObject(clo);
    
    LabelPtr oldLabel = _labels[_cloLabelsRelation[clo->objectID]];
    _cloLabelsRelation.erase(clo->objectID);
    _clos.erase(clo->objectID);
    removeLabelIfApplicable(oldLabel);
    
    _ui.twRelations->removeRow(row);

    // Set the "dirty" flag.
    setDirty(true);
}


DatabaseEntityPtr EditWidgetResponse::entityPtr() const
{
	return _response;
}


void EditWidgetResponse::setTableWidgets(const ClassificationObjectPtr &clo,
                                         const LabelPtr &label, int row)
{
    debug_assert(row >= 0 && row < _ui.twRelations->rowCount());
    debug_assert(!_ui.twRelations->isSortingEnabled());
    
    // Remove possibly existing items.
    QTableWidgetItem *delItem = 0;
    if ((delItem == _ui.twRelations->takeItem(row, 0)))
        delete delItem;
    if ((delItem == _ui.twRelations->takeItem(row, 1)))
        delete delItem;
    if ((delItem == _ui.twRelations->takeItem(row, 2)))
        delete delItem;
    
    // Create new items.
    QTableWidgetItem *cloItem = new QTableWidgetItem;
    cloItem->setText(QString::fromStdString(clo->strForType(clo->type)));
    cloItem->setData(Qt::UserRole, clo->objectID);
    cloItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

    QTableWidgetItem *cloAuxItem = new QTableWidgetItem;
    cloAuxItem->setText(QString("%1")
                        .arg(clo->objectID, 7, 10, QLatin1Char('0')));
    cloAuxItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

    QTableWidgetItem *labelItem = new QTableWidgetItem;
    labelItem->setText(QString::fromStdString(label->text));
    labelItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

    // Eventually insert the new items.
    // Note: Be careful when changing the sequence of these items as some other
    // methods rely on this exact sequence.
    _ui.twRelations->setItem(row, 0, cloItem);
    _ui.twRelations->setItem(row, 1, cloAuxItem);
    _ui.twRelations->setItem(row, 2, labelItem);
}


void EditWidgetResponse::removeLabelIfApplicable(LabelPtr &label)
{
    // See if any classification objects are still related to the given label.
    for (map<int, int>::const_iterator it = _cloLabelsRelation.begin();
        it != _cloLabelsRelation.end(); ++it)
    {
        if (it->second == label->labelID) {
            // Yes, return without removal.
            return;
        }
    }
    // No, it can be safely removed.
    _labels.erase(label->labelID);
}


int EditWidgetResponse::findRowForClassificationObject(const ClassificationObjectPtr &clo) const
{
    // Note: This could of course be instead done with QAbstractItemModel::match().
    // This seems to cause problems, however, and hence we search manually.
    for (int i = 0; i < _ui.twRelations->rowCount(); i++) {
        if (_ui.twRelations->item(i, 0)->data(Qt::UserRole).toInt() == clo->objectID)
            return i;
    }
    throw Poco::RuntimeException("Inconsistent model.");
}


} // namespace blissart
