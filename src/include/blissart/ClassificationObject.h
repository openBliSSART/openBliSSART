//
// $Id: ClassificationObject.h 873 2009-06-24 15:55:46Z alex $
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


#ifndef __BLISSART_CLASSIFICATION_OBJECT_H__
#define __BLISSART_CLASSIFICATION_OBJECT_H__


#include <common.h>
#include <blissart/DatabaseEntity.h>

#include <set>
#include <map>


namespace blissart {


/**
 * A collection of data descriptors that represent an object to be classified,
 * e.g. an audio component is made up of a spectrum and time-varying gains.
 */
class LibFramework_API ClassificationObject : public DatabaseEntity
{
public:
    /**
     * An enumeration of the available classification object types.
     * NOTE: Assigning constant values to the enums assures database
     * integrity when adding new classification object types.
     */
    typedef enum {
        Invalid = 0,
        NMFComponent = 1,
        NMDComponent = 4,
        ImportedAudio = 3,
        Spectrogram = 5
    } Type;

    /**
     * Default constructor. Creates an empty ClassificationObject.
     */
    ClassificationObject();

    /**
     * Copies all data from another ClassificationObject.
     */
    ClassificationObject(const ClassificationObject& other);

    /**
     * Unique ID of the ClassificationObject.
     */
    int objectID;

    /**
     * The type of this classification object's data.
     */
    Type type;

    /**
     * A vector of IDs of data descriptors that make up the
     * ClassificationObject.
     */
    std::set<int> descrIDs;

    /**
     * A vector of IDs of labels that are assigned to the ClassificationObject.
     */
    std::set<int> labelIDs;

    /**
     * Returns a textual representation of the given type.
     * @throw                   Poco::NotImplementedException
     */
    static std::string strForType(Type type);
};


typedef Poco::AutoPtr<ClassificationObject> ClassificationObjectPtr;


} // namespace blissart


#endif // __BLISSART_CLASSIFICATION_OBJECT_H__
