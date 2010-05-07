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


#ifndef __ENTITYITEM_H__
#define __ENTITYITEM_H__


#include <QTreeWidgetItem>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>


namespace blissart {


/**
 * \addtogroup browser
 * @{
 */

/**
 * Encapsulation of a database entity within a tree widget item.
 */
class EntityItem : public QTreeWidgetItem
{
public:
    /**
     * The following enum helps the outside-world to distinguish
     * EntityItems from plain QTreeWidgetItems.
     */
    typedef enum { EntityType = QTreeWidgetItem::UserType };


    /**
     * Sets the encapsulated entity pointer to the given value.
     * @param  dbe              a pointer to the new database entity
     */
    virtual void setEntityPtr(DatabaseEntityPtr dbe) = 0;


    /**
     * Gets the associated entity pointer. Subclasses must overwrite this
     * method!
     */
    virtual DatabaseEntityPtr entityPtr() const = 0;


    /**
     * Subclasses must overwrite this method if they want to load further data
     * upon expansion. The default implementation does nothing.
     */
    virtual void expand();


protected:
    /**
     * Constructs an instance of EntityItem as child of a QTreeWidgetItem.
     * @param  parent           a pointer to the parent widget
     */
    EntityItem(QTreeWidgetItem *parent);


    /**
     * Returns a reference to the database subsystem.
     */
    inline DatabaseSubsystem& dbSubsystem() const;
};


/**
 * @}
 */
    

// Inlines


DatabaseSubsystem& EntityItem::dbSubsystem() const
{
    return BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
}


} // namespace blissart


#endif
