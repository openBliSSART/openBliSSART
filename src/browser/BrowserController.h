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


#ifndef __BROWSERCONTROLLER_H__
#define __BROWSERCONTROLLER_H__


#include "ui_BrowserForm.h"
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>


namespace blissart {


/**
 * \addtogroup browser
 * @{
 */

/**
 * The controlling class of the Browser window.
 */
class BrowserController : public QWidget
{
   Q_OBJECT

public:
   /**
    * Constructs a BrowserController.
    * @param  parent            a pointer to the parent QWidget (or NULL)
    */
    BrowserController(QWidget *parent = 0);


public slots:
    /**
     * Handles refreshes of the tree widget.
     */
    void handleRefreshTreeWidget();


protected slots:
    /**
     * Event-handler for QTreeWidget::currentItemChanged.
     */
    void on_treeWidget_currentItemChanged(QTreeWidgetItem *current,
                                          QTreeWidgetItem *previous);


    /**
     * Event-handler for QTreeWidget::customContextMenuRequested.
     */
    void on_treeWidget_customContextMenuRequested(const QPoint &pos);


    /**
     * Event-handler for the "Save" button.
     */
    void on_pbSave_clicked();


    /**
     * Event-handler for the "Cancel" button.
     */
    void on_pbCancel_clicked();


    /**
     * Handles the creation of responses.
     */
    void handleCreateResponse();


    /**
     * Handles the creation of processes.
     */
    void handleImportAudio();


    /**
     * Handles the creation of labels.
     */
    void handleCreateLabel();


    /**
     * Handles the deletion of the selected items.
     * @throw                   Poco::NotImplementedException
     */
    void handleDeleteSelectedItems();


    /**
     * Handles the copy of the selected items.
     * @throw                   Poco::NotImplementedException
     */
    void handleCopySelectedItems();


    /**
     * Handles the extraction of features for the selected items or for all
     * classification objects.
     */
    void handleExtractFeatures();


    /**
     * Handles the export of the selected components to audio files.
     */
    void handleExportComponents();


    void handleCreateResponseFromItems();


    void handleSelectLabelForItems();


    /**
     * Handles entity updates from edit widgets.
     * @param  dbe              a pointer to the just updated database entity
     */
    void handleEntityUpdates(DatabaseEntityPtr dbe);


protected:
    /**
     * Returns a reference to the database subsystem.
     */
    inline DatabaseSubsystem& dbSubsystem() const;


    /**
     * Handles all incoming events. Normally, this method isn't neccessary at
     * all, but Qt forces us to circumvent some currentItemChanged- and
     * selection-related annoyances. For further information please refer to
     * the implementation of on_treeWidget_currentItemChanged.
     */
    virtual bool event(QEvent *event);


private:
    Ui::BrowserForm  _ui;
};


/**
 * @}
 */


// Inlines


DatabaseSubsystem& BrowserController::dbSubsystem() const
{
    return BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
}


} // namespace blissart


#endif
