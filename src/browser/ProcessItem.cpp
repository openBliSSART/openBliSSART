//
// $Id: ProcessItem.cpp 855 2009-06-09 16:15:50Z alex $
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


#include "ProcessItem.h"
#include "DataDescriptorItem.h"
#include <map>
#include <cassert>
#include <QDateTime>


using namespace std;


namespace blissart {


ProcessItem::ProcessItem(QTreeWidgetItem *parent, ProcessPtr process) :
    EntityItem(parent),
    _process(process)
{
    setChildIndicatorPolicy(ShowIndicator);
}


QVariant ProcessItem::data(int column, int role) const
{
    // We only care about Qt::DisplayRole and the first two columns.
    if (role == Qt::DisplayRole) {
        if (column == 0) {
            return QString("%1 (%2)")
                   .arg(_process->processID, 7, 10, QLatin1Char('0'))
                   .arg(QString::fromStdString(_process->name));
        } else if (column == 1)
            return QString::fromStdString(_process->inputFile);
    }
    // All the rest is handled by the superclass. This is the best way to allow
    // association of other data with this widget as well.
    return QTreeWidgetItem::data(column, role);
}


void ProcessItem::expand()
{
    // Expansion is handled only once.
    if (childCount() > 0)
        return;

    // Add an item for the sample frequency.
    QTreeWidgetItem *sfItem = new QTreeWidgetItem(this);
    sfItem->setText(0, QObject::tr("Sample freq."));
    sfItem->setText(1, QString::number(_process->sampleFreq));
    
    // Add an item for the startup time.
    QTreeWidgetItem *stItem = new QTreeWidgetItem(this);
    stItem->setText(0, QObject::tr("Startup time"));
    stItem->setText(1,
            QDateTime::fromTime_t(_process->startTime.epochTime()).toString());
    
    // Get the associated process parameters.
    _rootParameters =
        new QTreeWidgetItem(this, QStringList(QObject::tr("Parameters")));
    getProcessParameters();

    // Get the associated data descriptors.
    _rootDataDescriptors =
        new QTreeWidgetItem(this, QStringList(QObject::tr("Data descriptors")));
    getDataDescriptors();
}


void ProcessItem::getProcessParameters()
{
    // No need to get the parameters as they're already available
    // through the process itself.
    map<string, string>::const_iterator it = _process->parameters.begin();
    for (; it != _process->parameters.end(); it++) {
        QTreeWidgetItem *item = new QTreeWidgetItem(_rootParameters);
        item->setText(0, QString::fromStdString(it->first));
        item->setText(1, QString::fromStdString(it->second));
    }
}


void ProcessItem::getDataDescriptors()
{
    typedef QList<DataDescriptorPtr> ddlist_t;

    // Get the data descriptors and group them by their labels.
    vector<DataDescriptorPtr> descriptors =
        dbSubsystem().getDataDescriptors(_process->processID);
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


void ProcessItem::setEntityPtr(DatabaseEntityPtr dbe)
{
    assert(dbe->entityType() == DatabaseEntity::Process);
    _process = dbe.cast<Process>();
}


} // namespace blissart
