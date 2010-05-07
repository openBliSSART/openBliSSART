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


#ifndef __BLISSART_LABEL_H__
#define __BLISSART_LABEL_H__


#include <common.h>
#include <string>
#include <blissart/DatabaseEntity.h>


namespace blissart {


/**
 * \addtogroup framework
 * @{
 */

/**
 * A label used for annotation of objects in the database.
 */
class LibFramework_API Label : public DatabaseEntity
{
public:
    /**
     * Default constructor. Creates an empty Label object.
     */
    Label();

    /**
     * Copies all data from another Label.
     */
    Label(const Label& other);

    /**
     * Creates a Label object with the given text.
     */
    Label(const std::string& aText);
    
    /**
     * Unique label ID.
     */
    int labelID;
    
    /**
     * The label's text.
     */
    std::string text;
};


typedef Poco::AutoPtr<Label> LabelPtr;


/**
 * @}
 */


} // namespace blissart


#endif // __BLISSART_LABEL_H__
