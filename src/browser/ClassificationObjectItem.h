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

#ifndef __CLASSIFICATIONOBJECTITEM_H__
#define __CLASSIFICATIONOBJECTITEM_H__


#include "EntityItem.h"
#include <blissart/ClassificationObject.h>


// Forward declaration
class QTreeWidgetItem;


namespace blissart {


/**
 * Represents an item in the Browser entity list that corresponds to a 
 * ClassificationObject in the database.
 */
class ClassificationObjectItem : public EntityItem
{
public:
    /**
     * Constructs a new instance of ClassificationObjectItem as child of a given
     * parent for a given ClassificationObject.
     * @param  parent           a pointer to the parent tree widget item
     * @param  clo              a pointer to the ClassificationObject
     */
    ClassificationObjectItem(QTreeWidgetItem *parent,
                             ClassificationObjectPtr clo);


    /**
     * Returns the data associated with a particular column and role.  Delegates
     * all unknown columns and role to the superclass.
     */
    virtual QVariant data(int column, int role) const;


    /**
     * Upon expansion loads all additional information for this classification
     * object.
     */
    virtual void expand();


    /**
     * Sets the encapsulated entity pointer to the given value.
     * @param  dbe              a pointer to the new database entity 
     */ 
    virtual void setEntityPtr(DatabaseEntityPtr dbe);
    
    
    /**
     * Returns a pointer to the associated ClassificationObject.
     */
    virtual DatabaseEntityPtr entityPtr() const { return _clo; }


private:
    /**
     * Gets the name of the source input file which is the origin of the data
     * descriptors associated with this classification object.
     */
    void getSourceFileName();
    
    
    /**
     * Gets all data descriptors associated with the underlying classification
     * object.
     */
    void getDataDescriptors();

    
    ClassificationObjectPtr _clo;
    QTreeWidgetItem         *_rootDataDescriptors;
};


} // namespace blissart


#endif
