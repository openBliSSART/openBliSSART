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


#ifndef __BLISSART_RESPONSE_H__
#define __BLISSART_RESPONSE_H__


#include <common.h>

#include <string>
#include <map>
#include <blissart/DatabaseEntity.h>

namespace blissart {


/**
 * \addtogroup framework
 * @{
 */

/**
 * Stores an assignment of labels to ClassificationObjects.
 */
class LibFramework_API Response : public DatabaseEntity
{
public:
    /**
     * Default constructor. Creates an empty Response object.
     */
    Response();
    
    /** 
     * Copies all data from another Response.
     */
    Response(const Response& other);
    
    /**
     * Creates a Response object with name and description.
     */
    Response(const std::string& name, const std::string& description);

    /**
     * A unique ID for the response.
     */
    int responseID;
    
    /**
     * A (usually short) name for the response.
     */
    std::string name;

    /** 
     * A textual description of the response.
     */
    std::string description;
    
    /**
     * Maps IDs of ClassificationObjects to IDs of labels.
     */
    typedef std::map<int, int> LabelMap;

    /** 
     * Response labels. Each pair assigns an ID of a ClassificationObject
     * to an ID of a label.
     */
    LabelMap labels;

    /**
     * Returns the number of classes, i.e. the number of distinct labels used
     * in the response.
     */
    int numberOfClasses() const;
};


typedef Poco::AutoPtr<Response> ResponsePtr;


/**
 * @}
 */


} // namespace blissart


#endif // __BLISSART_RESPONSE_H__
