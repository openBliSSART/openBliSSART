//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2010, Alexander Lehmann <lehmanna@in.tum.de>
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


#include "EditWidgetLabel.h"
#include <QMessageBox>


namespace blissart {


EditWidgetLabel::EditWidgetLabel(LabelPtr label,
                                 QWidget *parent) :
    EditWidget(parent),
    _label(label)
{
    _ui.setupUi(this);

    // First display the data.
    displayData();
    
    // Then connect the neccessary signals/slots for tracking the edit status.
    connect(_ui.leText,
            SIGNAL(textChanged(const QString &)),
            SLOT(setDirty()));
    
    // Nothing's been edited until now. Reset the "dirty" flag.
    setDirty(false);    
}


void EditWidgetLabel::displayData()
{
    _ui.leText->setText(QString::fromStdString(_label->text));
}


void EditWidgetLabel::handleSave()
{
    // Sanity check.
    if (_ui.leText->text().isEmpty()) {
        QMessageBox::warning(this, tr("Stop"),
                tr("The label's text must not be empty."));
        _ui.leText->setFocus();
        return;
    }
    // Reflect the changes.
    _label->text = _ui.leText->text().toStdString();
    // Save the changes.
    dbSubsystem().updateLabel(_label);
    // Let mother do the rest.
    EditWidget::handleSave();
}


void EditWidgetLabel::handleCancel()
{
    // Since the entity's properties haven't been updated yet it's sufficient to
    // just display all the data again.
    displayData();
    // Let mother do the rest.
    EditWidget::handleCancel();
}


DatabaseEntityPtr EditWidgetLabel::entityPtr() const
{
    return _label;
}


} // namespace blissart

