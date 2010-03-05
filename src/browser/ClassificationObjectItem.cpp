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


#include "ClassificationObjectItem.h"
#include "DataDescriptorItem.h"
#include <blissart/DataDescriptor.h>
#include <cassert>


using namespace std;


namespace blissart {


ClassificationObjectItem::ClassificationObjectItem(QTreeWidgetItem *parent,
                                                   ClassificationObjectPtr clo) :
    EntityItem(parent),
    _clo(clo)
{
    setChildIndicatorPolicy(QTreeWidgetItem::ShowIndicator);
}


QVariant ClassificationObjectItem::data(int column, int role) const
{
    // We only care about Qt::DisplayRole and the first column.
    if (role == Qt::DisplayRole) {
        if (column == 0) {
            return QString("%1 (%2)")
                   .arg(_clo->objectID, 7, 10, QLatin1Char('0'))
                   .arg(QString::fromStdString(_clo->strForType(_clo->type)));
        }
        else if (column == 1) {
            if (_clo->labelIDs.empty()) {
                return QObject::tr("Unlabeled");
            }
            else {
                string text;
                unsigned int i = 0;
                foreach (int labelID, _clo->labelIDs) {
                    LabelPtr label = dbSubsystem().getLabel(labelID);
                    text += label->text;
                    if (i < _clo->labelIDs.size() - 1)
                        text += ", ";
                    ++i;
                }
                return QObject::tr(text.c_str());
            }
        }
    }
    // All the rest is handled by the superclass. This is the best way to allow
    // association of other data with this widget as well.
    return QTreeWidgetItem::data(column, role);
}


void ClassificationObjectItem::expand()
{
    // Expansion is handled only once.
    if (childCount() > 0)
        return;

    getSourceFileName();

    _rootDataDescriptors =
        new QTreeWidgetItem(this, QStringList(QObject::tr("Data descriptors")));
    getDataDescriptors();
}


void ClassificationObjectItem::getSourceFileName()
{
    ProcessPtr process = dbSubsystem().getProcess(_clo);
    QTreeWidgetItem *originItem = new QTreeWidgetItem(this);
    originItem->setText(0, QObject::tr("Originates from"));
    originItem->setText(1, QString::fromStdString(process->inputFile));
}


void ClassificationObjectItem::getDataDescriptors()
{
    typedef QList<DataDescriptorPtr> ddlist_t;

    // Get the data descriptors and group them by their labels.
    vector<DataDescriptorPtr> descriptors = dbSubsystem().getDataDescriptors(_clo);
    map<DataDescriptor::Type, ddlist_t> map;
    foreach (DataDescriptorPtr descr, descriptors) {
        static_cast<ddlist_t &>(map[descr->type]).append(descr);
    }

    // Create a corresponding group of tree widget items for every label.
    pair<DataDescriptor::Type, ddlist_t> kvp;
    foreach (kvp, map) {
        QString strRepr =
            QString::fromStdString(DataDescriptor::strForType(kvp.first));
        QTreeWidgetItem *lgItem =
            new QTreeWidgetItem(_rootDataDescriptors, QStringList(strRepr));
        foreach (DataDescriptorPtr dd, kvp.second)
            new DataDescriptorItem(lgItem, dd);
    }
}


void ClassificationObjectItem::setEntityPtr(DatabaseEntityPtr dbe)
{
    assert(dbe->entityType() == DatabaseEntity::ClassificationObject);
    _clo = dbe.cast<ClassificationObject>();
}


} // namespace blissart
