//
// $Id: StorageSubsystem.h 855 2009-06-09 16:15:50Z alex $
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


#ifndef __BLISSART_STORAGESUBSYSTEM_H__
#define __BLISSART_STORAGESUBSYSTEM_H__


#include <common.h>
#include <blissart/DataDescriptor.h>
#include <Poco/Path.h>
#include <Poco/Logger.h>
#include <Poco/Util/Subsystem.h>


namespace blissart {


// Forward declaration
namespace linalg {
    class Matrix;
    class Vector;
}


/**
 * Stores objects described by DataDescriptors as files, using a standard
 * storage layout.
 */
class LibFramework_API StorageSubsystem : public Poco::Util::Subsystem
{
public:
    /**
     * Default constructor. Sets up a StorageSubsystem that stores data in the
     * current application directory, in a subdirectory named "storage".
     */
    StorageSubsystem();

    /**
     * Stores a Matrix object described by a DataDescriptor.
     */
    void store(const blissart::linalg::Matrix& matrix, DataDescriptorPtr info);

    /**
     * Stores a Vector object described by a DataDescriptor.
     */
    void store(const blissart::linalg::Vector& vector, DataDescriptorPtr info);

    /**
     * Returns the location of the object described by a DataDescriptor.
     */
    Poco::Path getLocation(DataDescriptorPtr info);

    /**
     * Returns the ID of a DataDescriptor that describes the file at the 
     * given location.
     * Throws a Poco::InvalidArgumentException if the filename does not conform
     * to the StorageSubsystem's storage layout.
     */
    int getDescriptorID(const Poco::Path& filename);


protected:
    /**
     * Initialization of the storage subsystem.
     */
    virtual void initialize(Poco::Util::Application& app);


    /**
     * Uninitialization of the storage subsystem.
     */
    virtual void uninitialize();


    /**
     * Returns the name of this subsystem.
     */
    virtual const char* name() const;


private:
    // Forbid the copy constructor and operator=.
    StorageSubsystem(const StorageSubsystem&);
    StorageSubsystem& operator= (const StorageSubsystem&);


    Poco::Path             _storageDir;
    Poco::Logger&          _logger;
};


} // namespace blissart


#endif // __BLISSART_STORAGESUBSYSTEM_H__
