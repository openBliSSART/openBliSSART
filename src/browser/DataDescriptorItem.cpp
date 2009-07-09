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


#include "DataDescriptorItem.h"
#include "FeatureItem.h"
#include <cassert>


using namespace std;


namespace blissart {


DataDescriptorItem::DataDescriptorItem(QTreeWidgetItem *parent,
                                       DataDescriptorPtr ddesc) :
    EntityItem(parent),
    _ddesc(ddesc)
{
    setChildIndicatorPolicy(ShowIndicator);
}


QVariant DataDescriptorItem::data(int column, int role) const
{
    // We only care about Qt::DisplayRole and the first two columns.
    if (role == Qt::DisplayRole) {
        if (column == 0)
            return QString("%1").arg(_ddesc->descrID, 7, 10, QLatin1Char('0'));
        else if (column == 1)
            return QString::number(_ddesc->index);
    }
    // All the rest is handled by the superclass. This is the best way to allow
    // association of other data with this widget as well.
    return QTreeWidgetItem::data(column, role);
}


void DataDescriptorItem::expand()
{
    // Expansion is handled only once.
    if (childCount() > 0)
        return;

    // Add a child widget for every related feature.
    vector<FeaturePtr> features = dbSubsystem().getFeatures(_ddesc->descrID);
    foreach (FeaturePtr feature, features)
        new FeatureItem(this, feature);
}


void DataDescriptorItem::setEntityPtr(DatabaseEntityPtr dbe)
{
    assert(dbe->entityType() == DatabaseEntity::DataDescriptor);
    _ddesc = dbe.cast<DataDescriptor>();
}


} // namespace blissart
