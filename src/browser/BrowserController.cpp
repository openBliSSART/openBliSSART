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


#include "BrowserController.h"
#include "CreateProcessDialog.h"
#include "EditWidgetClassificationObject.h"
#include "EditWidgetLabel.h"
#include "EditWidgetResponse.h"
#include "ExportObjectsDlg.h"
#include "FeatureExtractionDialog.h"
#include "LabelItem.h"
#include "LabelSelectionDialog.h"
#include "ResponseItem.h"

#include <QDateTime>
#include <QMenu>
#include <QMessageBox>
#include <QTimer>

#include <cassert>


using namespace std;


namespace blissart {


BrowserController::BrowserController(QWidget *parent) :
    QWidget(parent)
{
    // General UI setup and initialization of the tree widget.
    _ui.setupUi(this);
    _ui.editWidgetContainer->setLayout(new QVBoxLayout());

    // Connect the appropriate signals and slots.
    connect(_ui.pbCreateResponse, SIGNAL(clicked()), SLOT(handleCreateResponse()));
    connect(_ui.pbImportAudio, SIGNAL(clicked()), SLOT(handleImportAudio()));
    connect(_ui.pbCreateLabel, SIGNAL(clicked()), SLOT(handleCreateLabel()));
}


void BrowserController::handleRefreshTreeWidget()
{
    // Remove a possible active edit widget.
    QLayoutItem *li = _ui.editWidgetContainer->layout()->takeAt(0);
    if (li) {
        li->widget()->hide();
        delete li->widget();
    }

    // And then reinitialize the tree widget.
    _ui.treeWidget->reinitialize();
}


void BrowserController::on_treeWidget_currentItemChanged(QTreeWidgetItem *current,
                                                         QTreeWidgetItem *previous)
{
    if (current && previous && previous->type() == EntityItem::EntityType) {
        QLayoutItem *li = _ui.editWidgetContainer->layout()->itemAt(0);
        if (li) {
            EditWidget *ew = static_cast<EditWidget *>(li->widget());
            if (ew->isDirty() && QMessageBox::Yes !=
                QMessageBox::question(this, tr("Modifications"),
                    tr("Any modifications you have made will be lost if "
                       "you continue. Do you want to continue?"),
                    QMessageBox::Yes | QMessageBox::No,
                    QMessageBox::No
                ))
            {
                // The user doesn't want to continue, so we restore the
                // previous current item selection.
                _ui.treeWidget->blockSignals(true);
                _ui.treeWidget->setCurrentItem(previous);
                _ui.treeWidget->blockSignals(false);
                // Qt handles the selection event that corresponds to the event
                // that caused this event handler to be called only after we
                // return from this method. So when the user selects "No" and
                // the current item is reset to the previous item, the selection
                // won't be updated correctly and several items will be selected.
                // In order to prevent this, we simply post a user-type event
                // that will then assure the correct single selection of the
                // now current item.
                qApp->postEvent(this, new QEvent(QEvent::User));
                return;
            }
        }
    }

    // Remove a possible former edit widget and then add the provided widget.
    QLayoutItem *li = _ui.editWidgetContainer->layout()->takeAt(0);
    if (li) {
        li->widget()->hide();
        delete li->widget();
    }

    // Perform a small sanity check.
    if (!current || current->type() != EntityItem::EntityType)
        return;

    // Determine the entity's type and create the corresponding edit widget.
    EditWidget *editWidget = 0;

    DatabaseEntityPtr dbe = static_cast<EntityItem *>(current)->entityPtr();
    switch (dbe->entityType()) {
    case DatabaseEntity::ClassificationObject:
        editWidget =
          new EditWidgetClassificationObject(dbe.cast<ClassificationObject>());
        break;
    case DatabaseEntity::Label:
        editWidget = new EditWidgetLabel(dbe.cast<Label>());
        break;
    case DatabaseEntity::Response:
        editWidget = new EditWidgetResponse(dbe.cast<Response>());
        break;
    default:
        // Intentionally left blank
        break;
    }

    // Set the new widget and connect to the related signals iff applicable.
    if (editWidget) {
        _ui.editWidgetContainer->layout()->addWidget(editWidget);
        connect(editWidget, SIGNAL(entityUpdated(DatabaseEntityPtr)),
                this, SLOT(handleEntityUpdates(DatabaseEntityPtr)));
        connect(editWidget, SIGNAL(entityDirty(bool)),
        		_ui.pbSave, SLOT(setEnabled(bool)));
        connect(editWidget, SIGNAL(entityDirty(bool)),
        		_ui.pbCancel, SLOT(setEnabled(bool)));
    }
}


void BrowserController::on_treeWidget_customContextMenuRequested(const QPoint &pos)
{
    QMenu pm(this);

    pm.addAction(tr("Import audio"), this, SLOT(handleImportAudio()));
    pm.addAction(tr("Create response"), this, SLOT(handleCreateResponse()));
    pm.addAction(tr("Create label"), this, SLOT(handleCreateLabel()));

    // Provide more actions depending on the selection.
    if (!_ui.treeWidget->selectedItems().isEmpty()) {
        // Determine the possible actions by analyzing the selected items.
        // An action is only enabled if all selected items are of a type that
        // allows this action.
        bool provideFeatureExtraction = true;
        bool provideDeletion = true;
        bool provideExport = true;
        bool provideCopy = true;
        bool provideResponseCreation = true;
        bool provideLabelSelection = true;
        foreach (QTreeWidgetItem *item, _ui.treeWidget->selectedItems()) {
            EntityItem *ew = dynamic_cast<EntityItem *>(item);
            if (!ew) {
                // A non-entity-type item has been selected. Thus provide
                // neither feature extraction nor deletion at all.
                provideFeatureExtraction = false;
                provideDeletion = false;
                provideExport = false;
                provideCopy = false;
                provideResponseCreation = false;
                provideLabelSelection = false;
                break;
            }
            // Only classification objects can be exported.
            // Also response creation and label selection is only available for them.
            if (ew->entityPtr()->entityType() != DatabaseEntity::ClassificationObject) {
                provideExport = false;
                provideResponseCreation = false;
                provideLabelSelection = false;
            }
            // Features can be extracted from classification objects, data descr.
            // and processes.
            if (ew->entityPtr()->entityType() != DatabaseEntity::Process &&
                ew->entityPtr()->entityType() != DatabaseEntity::DataDescriptor &&
                ew->entityPtr()->entityType() != DatabaseEntity::ClassificationObject)
            {
                provideFeatureExtraction = false;
            }
            // Only labels and responses can be copied.
            if (ew->entityPtr()->entityType() != DatabaseEntity::Label &&
                ew->entityPtr()->entityType() != DatabaseEntity::Response)
            {
                provideCopy = false;
            }
        }

        if (provideFeatureExtraction) {
            pm.addSeparator();
            pm.addAction(tr("Extract features"),
                         this, SLOT(handleExtractFeatures()));
        }

        if (provideExport) {
            // A separator isn't always neccessary, hence:
            if (!provideFeatureExtraction)
                pm.addSeparator();
            pm.addAction(tr("Export selected objects"),
                         this, SLOT(handleExportComponents()));
        }

        if (provideDeletion) {
            pm.addSeparator();
            pm.addAction(tr("Delete selected items"),
                         this, SLOT(handleDeleteSelectedItems()));
        }

        if (provideCopy) {
            pm.addAction(tr("Copy selected items"),
                         this, SLOT(handleCopySelectedItems()));
        }

        if (provideResponseCreation) {
            pm.addSeparator();
            pm.addAction(tr("Create response from these items"),
                         this, SLOT(handleCreateResponseFromItems()));
        }

        if (provideLabelSelection) {
            pm.addAction(tr("Select label(s)"),
                         this, SLOT(handleSelectLabelForItems()));
        }
    }

    pm.addSeparator();
    pm.addAction(tr("Refresh view"), this, SLOT(handleRefreshTreeWidget()));
    pm.exec(_ui.treeWidget->mapToGlobal(pos));
}


void BrowserController::on_pbSave_clicked()
{
	// Delegate.
	foreach (QObject *obj, _ui.editWidgetContainer->children()) {
		if (obj->inherits("blissart::EditWidget"))
			static_cast<EditWidget *>(obj)->handleSave();
	}
}


void BrowserController::on_pbCancel_clicked()
{
	// Delegate.
	foreach (QObject *obj, _ui.editWidgetContainer->children()) {
		if (obj->inherits("blissart::EditWidget"))
			static_cast<EditWidget *>(obj)->handleCancel();
	}
}


void BrowserController::handleCreateResponse()
{
    // First create a new response in the database.
    ResponsePtr r =
        new Response(QDateTime::currentDateTime().toString().toStdString(), "");
    dbSubsystem().createResponse(r);
    // Then insert and select the new response into the tree widget.
    _ui.treeWidget->setCurrentItem(_ui.treeWidget->insertNewResponse(r));
    _ui.treeWidget->setFocus();
}


void BrowserController::handleImportAudio()
{
    CreateProcessDialog dlg(this);
    if (QDialog::Accepted == dlg.exec()) {
        // Refresh the tree widget.
        handleRefreshTreeWidget();
    }
}


void BrowserController::handleCreateLabel()
{
    // First create a new label in the database
    LabelPtr l =
        new Label(QDateTime::currentDateTime().toString().toStdString());
    dbSubsystem().createLabel(l);
    // Then insert and select the new label into the tree widget.
    _ui.treeWidget->setCurrentItem(_ui.treeWidget->insertNewLabel(l));
    _ui.treeWidget->setFocus();
}


void BrowserController::handleDeleteSelectedItems()
{
   // Safety exit
   if (QMessageBox::Yes != QMessageBox::question(this, windowTitle(),
       tr("Really delete the selected items? Are you sure?"),
       QMessageBox::Yes | QMessageBox::No)) {
       return;
   }

   // User hit YES, we can go ahead and delete the selected items.
   foreach (QTreeWidgetItem *item, _ui.treeWidget->selectedItems()) {
       if (item->type() != EntityItem::EntityType)
           continue;

       EntityItem *ew = static_cast<EntityItem *>(item);

       // Delete the entity from the database.
       switch (ew->entityPtr()->entityType()) {
       case DatabaseEntity::Process:
           dbSubsystem().removeProcess(ew->entityPtr().cast<Process>());
           break;
       case DatabaseEntity::Response:
           dbSubsystem().removeResponse(ew->entityPtr().cast<Response>());
           break;
       case DatabaseEntity::Label:
           dbSubsystem().removeLabel(ew->entityPtr().cast<Label>());
           break;
       case DatabaseEntity::ClassificationObject:
           dbSubsystem().removeClassificationObject(ew->entityPtr()
                        .cast<ClassificationObject>());
           break;
       case DatabaseEntity::Feature:
           dbSubsystem().removeFeature(ew->entityPtr().cast<Feature>());
           break;
       case DatabaseEntity::DataDescriptor:
           dbSubsystem().removeDataDescriptor(ew->entityPtr().cast<DataDescriptor>());
           break;
       default:
           // This must never happen!
           throw Poco::NotImplementedException("Unknown database entity.");
           break;
       }
   }

   // Refresh the tree widget.
   handleRefreshTreeWidget();
}


void BrowserController::handleCopySelectedItems()
{
   foreach (QTreeWidgetItem *item, _ui.treeWidget->selectedItems()) {
       if (item->type() != EntityItem::EntityType)
           continue;

       EntityItem *ew = static_cast<EntityItem *>(item);

       // Delete the entity from the database.
       switch (ew->entityPtr()->entityType()) {
       case DatabaseEntity::Response:
           {
               ResponsePtr response = ew->entityPtr().cast<Response>();
               response->name = "Copy of " + response->name;
               dbSubsystem().createResponse(response);
           }
           break;
       case DatabaseEntity::Label:
           {
               LabelPtr label = ew->entityPtr().cast<Label>();
               label->text = "Copy of " + label->text;
               dbSubsystem().createLabel(label);
           }
           break;
       default:
           // This must never happen!
           throw Poco::NotImplementedException("Cannot copy this database entity.");
           break;
       }
   }

   // Refresh the tree widget.
   handleRefreshTreeWidget();
}


void BrowserController::handleExtractFeatures()
{
    // Determine the set of data descriptors for which the features should be
    // extracted.
    set<DataDescriptorPtr> descriptors;
    foreach (QTreeWidgetItem *item, _ui.treeWidget->selectedItems()) {
        EntityItem *ew = dynamic_cast<EntityItem *>(item);
        if (!ew)
            continue;

        switch (ew->entityPtr()->entityType()) {
        case DatabaseEntity::DataDescriptor:
            {
                DataDescriptorPtr dPtr = ew->entityPtr().cast<DataDescriptor>();
                if (dPtr->available)
                    descriptors.insert(dPtr);
            }
            break;
        case DatabaseEntity::Process:
            {
                int processID = ew->entityPtr().cast<Process>()->processID;
                vector<DataDescriptorPtr> tmp =
                    dbSubsystem().getDataDescriptors(processID);
                foreach (const DataDescriptorPtr &dPtr, tmp) {
                    if (dPtr->available)
                        descriptors.insert(dPtr);
                }
            }
            break;
        case DatabaseEntity::ClassificationObject:
            {
                ClassificationObjectPtr clo =
                    ew->entityPtr().cast<ClassificationObject>();
                vector<DataDescriptorPtr> tmp =
                    dbSubsystem().getDataDescriptors(clo);
                foreach (const DataDescriptorPtr &dPtr, tmp) {
                    if (dPtr->available)
                        descriptors.insert(dPtr);
                }
            }
            break;
        default:
            // Intentionally left blank
            break;
        }
    }

    FeatureExtractionDialog dlg(descriptors, this);
    dlg.exec();

    // Refresh the tree widget.
    handleRefreshTreeWidget();
}


void BrowserController::handleExportComponents()
{
    // Build a list of the components that are to be exported.
    vector<ClassificationObjectPtr> clos;
    foreach (QTreeWidgetItem *item, _ui.treeWidget->selectedItems()) {
        EntityItem *ew = dynamic_cast<EntityItem *>(item);
        if (ew &&
            ew->entityPtr()->entityType() == DatabaseEntity::ClassificationObject)
        {
            clos.push_back(ew->entityPtr().cast<ClassificationObject>());
        }
    }

    if (!clos.empty()) {
        ExportObjectsDlg dlg(clos, this);
        dlg.exec();
    }
}


void BrowserController::handleCreateResponseFromItems()
{
    // Setup response object.
    ResponsePtr r =
        new Response(QDateTime::currentDateTime().toString().toStdString(), "");
    foreach (QTreeWidgetItem *item, _ui.treeWidget->selectedItems()) {
        EntityItem *ew = dynamic_cast<EntityItem *>(item);
        if (ew &&
            ew->entityPtr()->entityType() == DatabaseEntity::ClassificationObject)
        {
            ClassificationObjectPtr pClo = ew->entityPtr().
                cast<ClassificationObject>();
            if (pClo->labelIDs.size() == 0) {
                QMessageBox::critical(this, windowTitle(),
                    tr("Cannot create response from unlabeled classification object!"));
                return;
            }
            else {
                // Use first label ID assigned to the object for the response.
                r->labels[pClo->objectID] = *(pClo->labelIDs.begin());
            }
        }
    }

    // Create new response in the database.
    dbSubsystem().createResponse(r);

    // Insert and select the new response into the tree widget.
    _ui.treeWidget->setCurrentItem(_ui.treeWidget->insertNewResponse(r));
    _ui.treeWidget->setFocus();
}


void BrowserController::handleSelectLabelForItems()
{
    LabelSelectionDialog lsd;
    lsd.setMultipleSelection(true);
    if (QDialog::Accepted != lsd.exec())
        return;

    foreach (QTreeWidgetItem *item, _ui.treeWidget->selectedItems()) {
        EntityItem *ew = dynamic_cast<EntityItem *>(item);
        if (ew &&
            ew->entityPtr()->entityType() == DatabaseEntity::ClassificationObject)
        {
            ClassificationObjectPtr pClo = ew->entityPtr().
                cast<ClassificationObject>();
            foreach (LabelPtr label, lsd.selectedLabels()) {
                pClo->labelIDs.insert(label->labelID);
            }
            dbSubsystem().updateClassificationObject(pClo);
        }
    }

    // This is the easiest, yet not the most efficient way to update the
    // view of the currently selected ClObj.
    on_treeWidget_currentItemChanged(_ui.treeWidget->selectedItems().first(), 0);
}


void BrowserController::handleEntityUpdates(DatabaseEntityPtr dbe)
{
    // Update all matching entity items in the tree widget.
    _ui.treeWidget->updateEntityItems(dbe);
    // By returning the focus to the tree widget we force it to update it's
    // view. Unfortunately treeWidget->repaint() as well as treeWidget->update()
    // don't do the job.
    _ui.treeWidget->setFocus();
}


bool BrowserController::event(QEvent *ev)
{
    // For an explanation why we have to explicitly deal with user-type events
    // please refer to on_treeWidget_currentItemChanged.
    if (ev->type() == QEvent::User) {
        ev->accept();
        _ui.treeWidget->selectionModel()->select(
            _ui.treeWidget->currentIndex(),
            QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Rows
        );
        return true;
    }

    return QWidget::event(ev);
}


} // namespace blissart
