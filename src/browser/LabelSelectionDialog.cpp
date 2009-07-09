//
// $Id: LabelSelectionDialog.cpp 855 2009-06-09 16:15:50Z alex $
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


#include "LabelSelectionDialog.h"

#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>

#include <QPushButton>


using namespace std;


namespace blissart {


LabelSelectionDialog::LabelSelectionDialog(const set<int> limitIDs, 
                                           bool multipleSelection,
                                           QWidget *parent) :
    QDialog(parent)
{
    _ui.setupUi(this);

    // Get all labels from the database and insert them into the list widget
    // as well as the internal map for faster access.
    vector<LabelPtr> labels = BasicApplication::instance()
                              .getSubsystem<DatabaseSubsystem>()
                              .getLabels();
    foreach (LabelPtr l, labels) {
        // Make sure that the limiting list of IDs gets respected.
        if (!limitIDs.empty() &&
            std::find(limitIDs.begin(), limitIDs.end(), l->labelID) == limitIDs.end()) {
            continue;
        }
        // We're either not limited or the label's id is in the list.
        _labels[l->labelID] = l;
        QListWidgetItem *lwi = new QListWidgetItem(_ui.lwLabels);
        lwi->setText(QString::fromStdString(l->text));
        lwi->setData(Qt::UserRole, l->labelID);
    }
    
    // Enable the "Ok" button iff there's at least one label on the list.
    // This prevents false return values if there were no labels available for
    // selection and nevertheless the user could hit the "Ok" button.
    _ui.buttonBox->button(QDialogButtonBox::Ok)->setEnabled(_labels.size() > 0);

    if (multipleSelection) {
        _ui.lwLabels->setSelectionMode(QAbstractItemView::MultiSelection);
    }
}


void LabelSelectionDialog::setMultipleSelection(bool flag)
{
    _ui.lwLabels->setSelectionMode(flag ? 
        QAbstractItemView::MultiSelection :
        QAbstractItemView::SingleSelection);
}


void LabelSelectionDialog::accept()
{
    // Determine if a label has been selected and if so, store it's id
    // in the corresponding variable.
    _selectedLabels.clear();
    foreach (QListWidgetItem* item, _ui.lwLabels->selectedItems()) {
        _selectedLabels.push_back(_labels[item->data(Qt::UserRole).toInt()]);
    }
    // Let the base-class do the rest.
    QDialog::accept();
}


} // namespace blissart
