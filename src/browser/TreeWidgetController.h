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


#ifndef __TREEWIDGETCONTROLLER_H__
#define __TREEWIDGETCONTROLLER_H__

#include <QTreeWidget>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <Poco/Exception.h>
#if __cplusplus >= 201703L
// Use C++17 if statement with initializer. User's code ends up in a else so
// scoping of different ifs is not broken
#define Q_FOREACH(variable, container)                                   \
for (auto _container_ = QtPrivate::qMakeForeachContainer(container);     \
     _container_.i != _container_.e;  ++_container_.i)                   \
    if (variable = *_container_.i; false) {} else
#else
// Explanation of the control word:
//  - it's initialized to 1
//  - that means both the inner and outer loops start
//  - if there were no breaks, at the end of the inner loop, it's set to 0, which
//    causes it to exit (the inner loop is run exactly once)
//  - at the end of the outer loop, it's inverted, so it becomes 1 again, allowing
//    the outer loop to continue executing
//  - if there was a break inside the inner loop, it will exit with control still
//    set to 1; in that case, the outer loop will invert it to 0 and will exit too
#define Q_FOREACH(variable, container)                                \
for (auto _container_ = QtPrivate::qMakeForeachContainer(container); \
     _container_.control && _container_.i != _container_.e;         \
     ++_container_.i, _container_.control ^= 1)                     \
    for (variable = *_container_.i; _container_.control; _container_.control = 0)
#endif


// Forward declaration
namespace Poco { class Logger; }


namespace blissart {


// Forward declaration
class EntityItem;
class LabelItem;
class ResponseItem;


/**
 * \addtogroup browser
 * @{
 */

/**
 * A specialization of TreeWidget that retrieves its items from the 
 * openBliSSART database.
 */
class TreeWidgetController : public QTreeWidget
{
    Q_OBJECT

public:
    /**
     * Constructs a new instance of TreeWidgetController for a given parent.
     */
    TreeWidgetController(QWidget *parent = 0);


    /**
     * Creates a ResponseItem for the given Response and inserts
     * it into the tree.
     * @param  r                a pointer to the Response
     * @return                  a pointer to the newly created ResponseItem
     */
    ResponseItem* insertNewResponse(ResponsePtr r);

    
    /**
     * Creates a LabelItem for the given Label and inserts
     * it into the tree.
     * @param  l                a pointer to the Label
     * @return                  a pointer to the newly created LabelItem
     */ 
    LabelItem* insertNewLabel(LabelPtr l);
    

    /**
     * Updates all entity items from the tree that match the given database
     * entity, i.e. assures that they contain the same information.
     * @param  dbe              a pointer to a database entity
     */
    void updateEntityItems(DatabaseEntityPtr dbe);
    
    
    /**
     * Reloads all data from the database.
     */
    void reinitialize();

protected slots:
    /**
     * Event-handler for QTreeWidget::itemExpanded.
     */
    void handleItemExpanded(QTreeWidgetItem *item);


protected:
    /**
     * Called once after the database manager has been set. Loads the first
     * level of entities.
     */
    void initializeView();


    /**
     * Gets all responsens and fits them into the tree widget.
     */
    void getResponses();


    /**
     * Gets all labels and fits them into the tree widget.
     */
    void getLabels();

    /**
     * Gets all processes and fits them into the tree widget.
     */
    void getProcesses();

    
    /**
     * Gets all classification objects and fits them into the tree widget.
     */
    void getClassificationObjects();


    /**
     * Recursively removes all children of the given item.
     * @param  parent           a pointer to the parent item
     */
    void removeAllChildren(QTreeWidgetItem *parent);
    
    
    /**
     * Determines the id of the given database entity independent of it's type.
     * @throw                   Poco::NotImplementedException
     */
    int getEntityID(const DatabaseEntityPtr& dbe) const;


    /**
     * Returns a depth-first list of all entity items in the tree that match the
     * given database entity.
     * @param  dbe              an abitrary database entity
     * @return                  a depth-first list of all matching entity items
     * @throw                   Poco::NotImplementedException
     */
    std::vector<EntityItem *> matchingItems(const DatabaseEntityPtr& dbe) const;
    
    
    /**
     * Returns a reference to the database subsystem.
     */
    inline DatabaseSubsystem& dbSubsystem() const;
    
    
    QTreeWidgetItem  *_rootResponses;
    QTreeWidgetItem  *_rootLabels;
    QTreeWidgetItem  *_rootProcesses;
    QTreeWidgetItem  *_rootClassificationObjects;
private:
    Poco::Logger                   &_logger;
};


/**
 * @}
 */
    

// Inlines


DatabaseSubsystem& TreeWidgetController::dbSubsystem() const
{
    return BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
}


} // namespace blissart


#endif
