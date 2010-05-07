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


#ifndef __BLISSART_DATABASEENTITY_H__
#define __BLISSART_DATABASEENTITY_H__


#include <Poco/AutoPtr.h>
#include <Poco/RefCountedObject.h>
#include <common.h>
#include <string>


namespace blissart {


/**
 * \addtogroup framework
 * @{
 */

/**
 * Serves as a base class for all database entities.
 */
class LibFramework_API DatabaseEntity : public Poco::RefCountedObject
{
public:
    /**
     * An enumeration of the different database entity types.
     */
   typedef enum {
       ClassificationObject,
       DataDescriptor,
       Feature,
       Label,
       Process, 
       Response
   } EntityType;

   /**
    * Creates a DatabaseEntity object of the specified type.
    */
   DatabaseEntity(EntityType entityType);


   /**
    * Copies the type from another DatabaseEntity.
    */
   DatabaseEntity(const DatabaseEntity& other);
   
   
   /**
    * Tells the type of the entity.
    */
   inline EntityType entityType() const { return _entityType; }


   /**
    * Under Unix, formats a floating-point number using the "C" locale,
    * so that a consistent format within the database is guaranteed.
    * Under Windows, this does the same as Poco::NumberFormatter::format().
    */
   static std::string formatDouble(double value);


   /**
    * Under Unix, parses a floating-point number using the "C" locale,
    * so that portability of databases is preserved.
    * Under Windows, this does the same as Poco::NumberParser::parseFloat().
    */
   static double parseDouble(const std::string& str);


protected:
    EntityType _entityType;
};


typedef Poco::AutoPtr<DatabaseEntity> DatabaseEntityPtr;


/**
 * @}
 */


} // namespace blissart


#endif // __BLISSART_DATABASEENTITY_H__

