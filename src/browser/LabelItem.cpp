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


#include "LabelItem.h"
#include <cassert>


namespace blissart {


LabelItem::LabelItem(QTreeWidgetItem *parent, LabelPtr label) :
    EntityItem(parent),
    _label(label)
{
}


QVariant LabelItem::data(int column, int role) const
{
    // We only care about Qt::DisplayRole and the first column.
    if (role == Qt::DisplayRole && column == 0)
        return QString::fromStdString(_label->text);
    // All the rest is handled by the superclass. This is the best way to allow
    // association of other data with this widget as well.
    return QTreeWidgetItem::data(column, role);
}


void LabelItem::setEntityPtr(DatabaseEntityPtr dbe)
{
    assert(dbe->entityType() == DatabaseEntity::Label);
    _label = dbe.cast<Label>();
}


} // namespace blissart
