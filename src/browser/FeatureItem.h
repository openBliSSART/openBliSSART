//
// $Id: FeatureItem.h 855 2009-06-09 16:15:50Z alex $
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

#ifndef __FEATUREITEM_H__
#define __FEATUREITEM_H__


#include "EntityItem.h"
#include <blissart/Feature.h>


namespace blissart {


class FeatureItem : public EntityItem
{
public:
    /**
     * Constructs a new instance of FeatureItem as child of a given parent for a
     * given Feature.
     * @param  parent           a pointer to the parent tree widget item
     * @param  feature          a pointer to the data descriptor
     */
    FeatureItem(QTreeWidgetItem *parent, FeaturePtr feature);


    /**
     * Returns the data associated with a particular column and role.  Delegates
     * all unknown columns and role to the superclass.
     */
    virtual QVariant data(int column, int role) const;


    /**
     * Sets the encapsulated entity pointer to the given value.
     * @param  dbe              a pointer to the new database entity 
     */ 
    virtual void setEntityPtr(DatabaseEntityPtr dbe);
    
    
    /**
     * Returns a pointer to the associated Feature.
     */
    virtual DatabaseEntityPtr entityPtr() const { return _feature; }


private:
    FeaturePtr _feature;
};


} // namespace blissart


#endif
