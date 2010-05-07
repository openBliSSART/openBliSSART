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


#ifndef __EDITWIDGETCLASSIFICATIONOBJECT_H__
#define __EDITWIDGETCLASSIFICATIONOBJECT_H__


#include "ui_EditWidgetClassificationObject.h"
#include "EditWidget.h"
#include <blissart/ClassificationObject.h>
#include <blissart/ProgressObserver.h>


// Forward declaration
class QProgressDialog;


namespace blissart {


/**
 * \addtogroup browser
 * @{
 */

/**
 * Represents a widget that implements GUI-based editing of 
 * ClassificationObjects, including playback and annotation.
 */
class EditWidgetClassificationObject : public EditWidget,
                                       public ProgressObserver
{
    Q_OBJECT

public:
    /**
     * Constructs a new instance of EditWidgetClassificationObject.
     * @param  clo              a pointer to a ClassificationObject
     * @param  parent           a pointer to the widget's parent
     */
    EditWidgetClassificationObject(ClassificationObjectPtr clo,
                                   QWidget *parent = 0);


protected slots:
    /**
     * Handles "save"-events.
     */
    virtual void handleSave();


    /**
     * Handles "cancel"-events.
     */
    virtual void handleCancel();


    /**
     * Event-handler for the list-widget.
     */
    void on_lwLabels_itemChanged(QListWidgetItem *);


    /**
     * Event-handler for the preview groupbox.
     */
    void on_groupBoxPreview_toggled(bool on);


protected:
    /**
     * Retrieves all Labels.
     */
    void loadLabels();


    /**
     * Loads the DataDescriptors that are related to the underlying
     * ClassificationObject. Also initializes the samples visualization widget.
     * @return                  true iff all data could be retrieved correctly
     */
    bool loadDataDescriptorsAndComputeSamples();


    /**
     * Sets all child widgets' data according to the underlying
     * ClassificationObject.
     */
    void displayData();


    /**
     * Returns a pointer to the associated ClassificationObject.
     */
    virtual DatabaseEntityPtr entityPtr() const;


private:
    /**
     * Implementation of ProgressObserver's progressChanged()-method in order
     * to give the user some information on the progress of the IFT.
     */
    void progressChanged(float progress);


    Ui::EditWidgetClassificationObject _ui;
    ClassificationObjectPtr            _clo;
    bool                               _hasSound;
    QProgressDialog*                   _progressDlg;
};


/**
 * @}
 */
    

} // namespace blissart


#endif

