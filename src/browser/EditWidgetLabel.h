//
// $Id: EditWidgetLabel.h 855 2009-06-09 16:15:50Z alex $
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


#ifndef __EDITWIDGETLABEL_H__
#define __EDITWIDGETLABEL_H__


#include "ui_EditWidgetLabel.h"
#include "EditWidget.h"
#include <blissart/Label.h>


namespace blissart {


class EditWidgetLabel : public EditWidget
{
    Q_OBJECT

public:
    /**
     * Constructs a new instance of EditWidgetLabel.
     * @param  label            a pointer to a Label
     * @param  parent           a pointer to the widget's parent
     */
    EditWidgetLabel(LabelPtr label, QWidget *parent = 0);


protected slots:
    /**
     * Handles "save"-events.
     */
    virtual void handleSave();


    /**
     * Handles "cancel"-events.
     */
    virtual void handleCancel();


protected:
    /**
     * Sets all child widgets' data according to the underlying label.
     */
    void displayData();
    
    
    /**
     * Returns a pointer to the associated label.
     */
    virtual DatabaseEntityPtr entityPtr() const;


private:
    Ui::EditWidgetLabel _ui;
    LabelPtr            _label;
};


} // namespace blissart


#endif

