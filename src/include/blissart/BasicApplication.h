//
// $Id: BasicApplication.h 594 2008-11-14 12:27:36Z alex $
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


#ifndef __BLISSART_BASICAPPLICATION_H__
#define __BLISSART_BASICAPPLICATION_H__


#include <common.h>
#include <vector>
#include <string>
#include <Poco/Util/Application.h>


namespace blissart {


/**
 * An extension of Poco's Application class that provides some utility methods
 * and access to the openBliSSART directory tree.
 */
class LibFramework_API BasicApplication: public Poco::Util::Application
{
public:
    /**
     * Constructs an instance of BasicApplication.
     */
    BasicApplication();
    
    
    /**
     * Gets the FFTW lock. All threaded applications using FFTW must use 
     * this for thread-safety.
     */
    static void lockFFTW();
    
    
    /**
     * Releases the FFTW lock. All threaded applications using FFTW must use 
     * this for thread-safety.
     */
    static void unlockFFTW();


    /**
     * Parses the given script files and returns a vector of the input files
     * that were given in the script files.
     */
    static std::vector<std::string> 
        parseScriptFiles(const std::vector<std::string>& fileNames);
    
    
protected:
    /**
     * Overrides the corresponding Poco::Util::Application method.
     * Calls initializeDirectories() and initializeConfiguration() methods.
     */
    virtual void initialize(Poco::Util::Application& self);
    
    
    /**
     * Determines the openBliSSART directory structure and sets the
     * corresponding configuration (see implementation for details).  
     * Creates the directories if neccessary.
     */
    virtual void initializeDirectories();


    /**
     * Loads necessary configuration files after setting up the directory
     * sturcture.
     * Subclasses that overwrite this method must call the base-class
     * implementation at first!
     */
    virtual void initializeConfiguration();
};


} // namespace blissart


#endif // __BLISSART_BASICAPPLICATION_H__
