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


#ifndef __BLISSART_FEATURE_H__
#define __BLISSART_FEATURE_H__


#include <common.h>
#include <blissart/DatabaseEntity.h>

#include <string>


namespace blissart {


/**
 * A named data feature with a real value.
 */
class LibFramework_API Feature : public DatabaseEntity
{
public:
    /**
     * Default constructor. Creates an empty Feature with zero value.
     */
    Feature();
    
    /**
     * Copies all data from another Feature.
     */
    Feature(const Feature& other);
    
    /**
     * Creates a named Feature corresponding to the data with the given 
     * descriptor.
     */
    Feature(int descrID, const std::string& name, double value);
    
    /**
     * Creates a named, parameterized Feature corresponding to the data with 
     * the given descriptor.
     */
    Feature(int descrID, const std::string& name, 
        double param1, double param2, double param3,
        double value);

    /**
     * Returns a textual description of this feature (name and parameters).
     */
    std::string description() const;
    
    /**
     * ID of associated data descriptor.
     */
    int descrID;
    
    /**
     * Name of the feature (e.g. "spectralCentroid").
     */
    std::string name;

    /**
     * Up to 3 optional feature parameters (e.g. MFCC index).
     */
    double params[3];

    /**
     * Value of the feature.
     */
    double value;
};


typedef Poco::AutoPtr<Feature> FeaturePtr;


} // namespace blissart


#endif // __BLISSART_FEATURE_H__
