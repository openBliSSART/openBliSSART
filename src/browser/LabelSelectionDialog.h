//
// $Id: LabelSelectionDialog.h 855 2009-06-09 16:15:50Z alex $
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


#ifndef __LABELSELECTIONDIALOG_H__
#define __LABELSELECTIONDIALOG_H__


#include "ui_LabelSelectionDialog.h"
#include <blissart/Label.h>
#include <set>


namespace blissart {


class LabelSelectionDialog : public QDialog
{
    Q_OBJECT

public:
    /**
     * Constructs an instance of LabelSelectionDialog. It is possible to limit
     * the selectable labels to a given list by providing their respective ids. 
     * @param  limitIDs         a list of all the label ids whose associated
     *                          labels should be made available for selection
     * @param  parent           a pointer to the parent widget                         
     */
    LabelSelectionDialog(const std::set<int> limitIDs = std::set<int>(),
                         bool multipleSelection = false,
                         QWidget *parent = 0);


    /**
     * Returns the selected label.
     */
    inline std::vector<LabelPtr> selectedLabels() const { 
        return _selectedLabels; 
    }


    /**
     * Enables/disables multiple selection.
     */
    void setMultipleSelection(bool flag);


public slots:
    /**
     * Event-handler for the default button, i.e. "OK".
     */
    virtual void accept();


private:
    Ui::LabelSelectionDialog _ui;
    std::vector<LabelPtr>    _selectedLabels;
    std::map<int, LabelPtr>  _labels;
};


} // namespace blissart


#endif
