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


#ifndef __BLISSART_WINDOWFUNCTIONS_H__
#define __BLISSART_WINDOWFUNCTIONS_H__


#include <common.h>
#include <string>


namespace blissart {


    /**
     * The signature of a function that is used to window data. The first parameter
     * is the index of a data element, the second parameter is the total number of
     * data elements.
     */
    typedef double(*WindowFunction)(unsigned int, unsigned int);

    
    /**
     * The Hann function.
     */
    double LibFramework_API HannFunction(unsigned int index, unsigned int count);


    /**
     * The square root of the Hann function.
     */
    double LibFramework_API SqHannFunction(unsigned int index, unsigned int count);
    
    
    /**
     * The Hamming function.
     */
    double LibFramework_API HammingFunction(unsigned int index, unsigned int count);
    

    /**
     * The rectangle function.
     */
    double LibFramework_API RectangleFunction(unsigned int index, unsigned int count);


    /**
     * Returns the name of a window function if the function pointer is known.
     */
    std::string LibFramework_API
    windowFunctionName(WindowFunction wf);


    /**
     * Returns the short name of a window function if the function pointer 
     * is known.
     */
    std::string LibFramework_API
    windowFunctionShortName(WindowFunction wf);


    /**
     * Returns a function pointer to a window function of the given name,
     * or null if no function of the given name exists.
     */
    WindowFunction LibFramework_API
    windowFunctionForName(const std::string& wfName);


    /**
     * Returns a function pointer to a window function of the given name,
     * or null if no function of the given name exists.
     */
    WindowFunction LibFramework_API
    windowFunctionForShortName(const std::string& wfName);


} // namespace blissart


#endif // __BLISSART_WINDOWFUNCTIONS_H__
