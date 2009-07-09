//
// $Id: FeatureDescriptor.h 855 2009-06-09 16:15:50Z alex $
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


#ifndef __BLISSART_FEATUREDESCRIPTOR_H__
#define __BLISSART_FEATUREDESCRIPTOR_H__


#include <common.h>
#include <blissart/DataDescriptor.h>


namespace blissart {


/**
 * Describes a feature for use in feature extraction and classification.
 */
class LibFramework_API FeatureDescriptor
{
public:
    /**
     * Default constructor.
     */
    FeatureDescriptor(const std::string& name, DataDescriptor::Type dataType,
        double param1 = 0.0, double param2 = 0.0, double param3 = 0.0);

    /**
     * Copies data from another FeatureDescriptor.
     */
    FeatureDescriptor(const FeatureDescriptor& other);

    /**
     * Returns a string representation of the form "(name, dataType, param)".
     */
    std::string toString() const;

    /**
     * Name of the feature (e.g. "spectralCentroid").
     */
    std::string name;
   
    /**
     * The type of data descriptor associated with the feature.
     */
    DataDescriptor::Type dataType;

    /**
     * First feature parameter, e.g. the MFCC index.
     */
    double params[3];

    /**
     * Provides an ordering on FeatureDescriptors.
     */
    bool operator < (const FeatureDescriptor& other) const;

private:
    /**
     * Checks if the given feature name describes a valid feature for the
     * given data descriptor type.
     */
    static bool isValidFeature(const std::string& name, 
        DataDescriptor::Type dataType);
};


} // namespace blissart


#endif // __BLISSART_FEATUREDESCRIPTOR_H__
