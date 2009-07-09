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


#ifndef __EDITWIDGETRESPONSE_H__
#define __EDITWIDGETRESPONSE_H__


#include "ui_EditWidgetResponse.h"
#include "EditWidget.h"
#include <blissart/Response.h>
#include <blissart/Label.h>


namespace blissart {


class EditWidgetResponse : public EditWidget
{
    Q_OBJECT

public:
    /**
     * Constructs a new instance of EditWidgetResponse.
     * @param  response         a pointer to a Response
     * @param  parent           a pointer to the widget's parent
     */
    EditWidgetResponse(ResponsePtr response, QWidget *parent = 0);


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
     * Event-handler for the "Add CLO's by label" push button.
     */
    void on_pbAddByLabel_clicked();
    
    
    /**
     * Handles label selection for classification objects that have more than
     * one label associated with them.
     */
    void on_pbSelectLabel_clicked();


    /**
     * Event-handler for the "Remove Selected" push button.
     */
    void on_pbRemoveSelected_clicked();


    /**
     * Event-handler for the "Quality feedback" push button.
     */
    void on_pbQualityFeedback_clicked();
    
    
    /**
     * Event-handler for the custom context menu request of the table widget.
     */ 
    void on_twRelations_customContextMenuRequested(const QPoint &pos);
    
    
    /**
     * Updates the button states depending on, for example, the relations of
     * classification objects and labels.
     */
    void updateButtonStates();
    

protected:
    /**
     * Sets all child widgets' data according to the underlying response.
     */
    void displayData();
    
    
    /**
     * Gets all classification objects and labels that are related to the
     * underlying response and stores them in internal maps for faster access.
     */
    void initializeCLOsAndLabels();

    
    /**
     * Manages insertions and updates of the relations of classification objects
     * and labels. Takes care of the table widget, too.
     * @param  clo              a pointer to a classification object
     * @param  label            a pointer to a label
     */
    void insertOrUpdateRelation(const ClassificationObjectPtr &clo,
                                const LabelPtr &label);
    
    
    /**
     * Manages the removal of relations for a given classification object.
     * Takes care of the table widget, too.
     * @param  clo              a pointer to a classification object
     */
    void removeRelation(ClassificationObjectPtr &clo);
    
    
    /**
     * Returns a pointer to the associated response.
     */
    virtual DatabaseEntityPtr entityPtr() const;
    
    
private:
    /**
     * Sets the table widget items for the given classification object, label
     * and row. Deletes any old items if neccessary.
     * @param  clo              a pointer to a classification object
     * @param  label            a pointer to a label
     * @param  row              a valid row
     */
    void setTableWidgets(const ClassificationObjectPtr &clo,
                         const LabelPtr &label, int row);
    
    
    /**
     * Removes a label from the _labels map iff it is not related to any
     * classification objects.
     */
    void removeLabelIfApplicable(LabelPtr &label);
    

    /**
     * Find the table widget's row for the given ClassificationObject.
     * @param  clo              a pointer to a classification object
     * @return                  the corresponding table widget's row
     */
    int findRowForClassificationObject(const ClassificationObjectPtr &clo) const;


    typedef std::pair<ClassificationObjectPtr, LabelPtr> CLOLabelPair;
    
    
    Ui::EditWidgetResponse                 _ui;
    ResponsePtr                            _response;
    std::map<int, LabelPtr>                _labels;
    std::map<int, ClassificationObjectPtr> _clos;
    std::map<int, int>                     _cloLabelsRelation;
};


} // namespace blissart


#endif

