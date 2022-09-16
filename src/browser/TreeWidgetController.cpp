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


#include "TreeWidgetController.h"
#include "ClassificationObjectItem.h"
#include "LabelItem.h"
#include "ProcessItem.h"
#include "ResponseItem.h"
#include <Poco/Logger.h>


#include <cassert>
#include <map>
#include <queue>
#include <vector>

#include <QHeaderView>
#include <QProgressDialog>


using namespace std;


namespace blissart {


TreeWidgetController::TreeWidgetController(QWidget *parent) :
    QTreeWidget(parent),
    _logger(BasicApplication::instance().logger())
{
    initializeView();
    // To give items the possibility for further expansion we connect to the
    // appropriate signal.
    connect(this,
            SIGNAL(itemExpanded(QTreeWidgetItem *)),
            SLOT(handleItemExpanded(QTreeWidgetItem *)));
}


ResponseItem* TreeWidgetController::insertNewResponse(ResponsePtr r)
{
    // There seems to be a bug in Qt 4.3.4/4.4.0 that forces us to expand the
    // _rootResponses before adding new children.  This bug is related to sorted
    // item views only.
    expandItem(_rootResponses);
    return new ResponseItem(_rootResponses, r);
}


LabelItem* TreeWidgetController::insertNewLabel(LabelPtr l)
{
    // There seems to be a bug in Qt 4.3.4/4.4.0 that forces us to expand the
    // _rootLabels before adding new children.  This bug is related to sorted
    // item views only.
    expandItem(_rootLabels);
    return new LabelItem(_rootLabels, l);
}


//void TreeWidgetController::removeEntityItems(DatabaseEntityPtr dbe)
//{
//    // Delete all matching items.
//    foreach (QTreeWidgetItem *delItem, matchingItems(dbe)) {
//        removeAllChildren(delItem);
//        if (delItem->parent())
//            delItem->parent()->removeChild(delItem);
//        delete delItem;
//    }
//}
//
//
void TreeWidgetController::updateEntityItems(DatabaseEntityPtr dbe)
{
    foreach (EntityItem *item, matchingItems(dbe))
        item->setEntityPtr(dbe);
}
//foreach (EntityItem *item, matchingItems(dbe))
//    item->setEntityPtr(dbe);

//if (labels.size() != 0)
//{
//    for (auto& l : labels) {new LabelItem(_rootLabels, l);}
//}
//foreach (LabelPtr l, QVector<LabelPtr>::fromStdVector(labels))
//


void TreeWidgetController::reinitialize()
{
    QProgressDialog dlg(tr("Reinitializing view..."), QString(), 0, 3, this);

    // Perform a BFS and remember the expanded entity items.
    map<DatabaseEntity::EntityType, set<int> > expandedEntities;
    queue<QTreeWidgetItem *> q;
    q.push(invisibleRootItem());
    while (!q.empty()) {
        QTreeWidgetItem *currentItem = q.front();
        q.pop();
        
        // Don't bother with non-expanded items.
        if (currentItem != invisibleRootItem() && !currentItem->isExpanded())
            continue;
        
        // Enqueue the current item's children.
        for (int i = 0; i < currentItem->childCount(); i++)
            q.push(currentItem->child(i));
        
        // Don't bother with non-entity types.
        if (currentItem->type() != EntityItem::EntityType)
            continue;
        
        EntityItem *entityItem = static_cast<EntityItem *>(currentItem);
        if (entityItem->type() != DatabaseEntity::Feature) {
            const int id = getEntityID(entityItem->entityPtr());
            set<int> &s = expandedEntities[entityItem->entityPtr()->entityType()];
            s.insert(id);
        }
    }
    
    // Backup the expansion-states of the root items.
    const bool expandResponses = _rootResponses->isExpanded();
    const bool expandLabels = _rootLabels->isExpanded();
    const bool expandProcesses = _rootProcesses->isExpanded();
    const bool expandClassificationObjects = 
        _rootClassificationObjects->isExpanded();
    
    // Clear the model.
    if (_rootResponses)
        invisibleRootItem()->removeChild(_rootResponses);
    if (_rootLabels)
        invisibleRootItem()->removeChild(_rootLabels);
    if (_rootProcesses)
        invisibleRootItem()->removeChild(_rootProcesses);
    if (_rootClassificationObjects)
        invisibleRootItem()->removeChild(_rootClassificationObjects);
    dlg.setValue(1);
    
    // Reload.
    initializeView();
    dlg.setValue(2);
    
    // Perform a BFS and reopen the formerly expanded items.
    q.push(invisibleRootItem());
    while (!q.empty()) {
        QTreeWidgetItem *currentItem = q.front();
        q.pop();
        
        // Enqueue the current item's children.
        for (int i = 0; i < currentItem->childCount(); i++)
            q.push(currentItem->child(i));
        
        // Don't bother with non-entity types.
        if (currentItem->type() != EntityItem::EntityType)
            continue;
        
        // Determine the entity's id and reexpand this item if this id is stored
        // in the entity type's corresponding set of expanded ids.
        EntityItem *entityItem = static_cast<EntityItem *>(currentItem);
        const int id = getEntityID(entityItem->entityPtr());
        if (id) {
            set<int> &s = expandedEntities[entityItem->entityPtr()->entityType()];
            if (s.find(id) != s.end()) {
                QTreeWidgetItem *item = entityItem;
                while (item && item != invisibleRootItem()) {
                    item->setExpanded(true);
                    item = item->parent();
                }
            }
        }
    }
    dlg.setValue(3);
    
    // Restore the expansion-states of the root items.
    if (expandResponses && !_rootResponses->isExpanded())
        _rootResponses->setExpanded(true);
    if (expandLabels && !_rootLabels->isExpanded())
        _rootLabels->setExpanded(true);
    if (expandProcesses && !_rootProcesses->isExpanded())
        _rootProcesses->setExpanded(true);
    if (expandClassificationObjects && !_rootClassificationObjects->isExpanded())
        _rootClassificationObjects->setExpanded(true);
}



void TreeWidgetController::handleItemExpanded(QTreeWidgetItem *item)
{
    if (item->type() == EntityItem::EntityType)
        static_cast<EntityItem *>(item)->expand();
    header()->resizeSections(QHeaderView::ResizeToContents);
}


void TreeWidgetController::initializeView()
{
    _logger.debug("TreeWidgetController::initializeView.");
    // Disable sorting.
    setSortingEnabled(false);
    
    // Get the stored responses.
    _rootResponses = new QTreeWidgetItem(this, QStringList(tr("Responses")));
    _logger.debug("getResponses.");
    getResponses();
    
    // Get the stored labels.
    _rootLabels = new QTreeWidgetItem(this, QStringList(tr("Labels")));
    _logger.debug("getLabels.");
    getLabels();
    
     // Get the stored processes.
    _rootProcesses = new QTreeWidgetItem(this, QStringList(tr("Processes")));
    _logger.debug("getProcesses.");
    getProcesses();
    
    // Get the stored classification objects.
    _rootClassificationObjects =
        new QTreeWidgetItem(this, QStringList(tr("Classification objects")));
    _logger.debug("getClassificationObjects.");    
    getClassificationObjects();
    
    // "Prettify" the tree widget.
    header()->resizeSections(QHeaderView::ResizeToContents);
    QStringList labels;
    labels << "Entities" << "Additional information";
    setHeaderLabels(labels);
    
    // Reenable sorting.
    sortItems(0, Qt::AscendingOrder);
    setSortingEnabled(true);
}


void TreeWidgetController::getResponses()
{
    ResponsePtr r;
    // First remove all possible previous children.
    removeAllChildren(_rootResponses);
    // Retrieve the list of available responses.
    vector<ResponsePtr> responses = dbSubsystem().getResponses();
    // Create a new ResponseItem for every response.
    if (responses.size() != 0) 
    {
    	for (auto& r : responses) { new ResponseItem(_rootResponses, r);}
    }
}


void TreeWidgetController::getLabels()
{
    LabelPtr l;
    // First remove all possible previous children
    removeAllChildren(_rootLabels);
    // Retrieve the list of available labels.

    vector<LabelPtr> labels = dbSubsystem().getLabels();
    // Create a new LabelItem for every label.
    if (labels.size() != 0) 
    {
        for (auto& l : labels) {new LabelItem(_rootLabels, l);}
    }	
    //foreach (LabelPtr l, QVector<LabelPtr>::fromStdVector(labels))
    //    new LabelItem(_rootLabels, l);
}


void TreeWidgetController::getProcesses()
{
    ProcessPtr p;
    // First remove all possible previous children.
    removeAllChildren(_rootProcesses);
    // Retrieve the list of available processes.
    vector<ProcessPtr> processes = dbSubsystem().getProcesses();
    // Create a new ProcessItem for every process. Further data belonging to the
    // process will not be read until the first time the item is being expanded.
    if (processes.size() != 0) 
    {
    	for (auto& p : processes) {new ProcessItem(_rootProcesses, p);}
    }
    //foreach (ProcessPtr p, QVector<ProcessPtr>::fromStdVector(processes))
    //    new ProcessItem(_rootProcesses, p);
}


void TreeWidgetController::getClassificationObjects()
{
    _logger.debug("removeAllChildren.");
    // First remove all possible previous children.
    removeAllChildren(_rootClassificationObjects);
    // Retrieve the list of available classification objects.
    vector<ClassificationObjectPtr> clos = dbSubsystem().getClassificationObjects();
    // Create and insert a new tree widget item for every classification object.
    if (clos.size() != 0)
    {
    	foreach (ClassificationObjectPtr clo, clos)
    	    new ClassificationObjectItem(_rootClassificationObjects, clo);
    }
}


void TreeWidgetController::removeAllChildren(QTreeWidgetItem *parent)
{
    _logger.debug("removeAllChildren in.");
    queue<QTreeWidgetItem *> q;
    for (q.push(parent); !q.empty(); q.pop()) {
        QTreeWidgetItem *item = q.front();
        foreach (QTreeWidgetItem *child, item->takeChildren())
            q.push(child);
        if (item != parent)
            delete item;
        item = nullptr;
    }
    _logger.debug("removeAllChildren out.");
}


int TreeWidgetController::getEntityID(const DatabaseEntityPtr& dbe) const
{
    switch (dbe->entityType()) {
    case DatabaseEntity::ClassificationObject:
        return dbe.cast<ClassificationObject>()->objectID;
    case DatabaseEntity::Response:
        return dbe.cast<Response>()->responseID;
    case DatabaseEntity::Label:
        return dbe.cast<Label>()->labelID;
    case DatabaseEntity::DataDescriptor:
        return dbe.cast<DataDescriptor>()->descrID;
    case DatabaseEntity::Process:
        return dbe.cast<Process>()->processID;
    case DatabaseEntity::Feature:
        throw Poco::NotImplementedException("Features don't have ids.");
    default:
        throw Poco::NotImplementedException("Unknown entity type.");
    }
}


vector<EntityItem *>
TreeWidgetController::matchingItems(const DatabaseEntityPtr &dbe) const
{
    vector<EntityItem *> result;
    
    // Perform a BFS to find all entity items that match the given database
    // entity and put them into the result-vector in reverse order.
    queue<QTreeWidgetItem *> q;
    for (q.push(invisibleRootItem()); !q.empty(); q.pop()) {
        QTreeWidgetItem *item = q.front();
        // Enqueue all children (also non-entity-types, because they might have
        // respective children!).
        for (int i = 0; i < item->childCount(); i++)
            q.push(item->child(i));
        
        // Don't bother with non-entity types.
        if (item->type() != EntityItem::EntityType)
            continue;
        
        // Don't bother with entities of different types.
        DatabaseEntityPtr ep = static_cast<EntityItem *>(item)->entityPtr();
        if (dbe->entityType() != ep->entityType())
            continue;
        
        // Check if both entities are identical and insert the current item into
        // the result-vector if appropriate.
        if (ep->entityType() != DatabaseEntity::Feature) {
            if (getEntityID(dbe) == getEntityID(ep))
                result.insert(result.begin(), static_cast<EntityItem *>(item));
        } else {
            if (dbe.cast<Feature>()->descrID == 
                ep.cast<Feature>()->descrID &&
                dbe.cast<Feature>()->name == 
                ep.cast<Feature>()->name)
            {
                result.insert(result.begin(), static_cast<EntityItem *>(item));
            }
        }
    }
    
    return result;
}


} // namespace blissart
