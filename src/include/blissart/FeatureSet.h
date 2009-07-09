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


#ifndef __BLISSART_FEATURESET_H__
#define __BLISSART_FEATURESET_H__


#include <common.h>
#include <map>
#include <vector>
#include <string>

#include <blissart/FeatureDescriptor.h>
#include <blissart/DataDescriptor.h>


namespace blissart {


/**
 * A set of features to be extracted from data, or to be loaded from the database.
 */
class LibFramework_API FeatureSet
{
public:
    /**
     * Creates an empty FeatureSet.
     */
    FeatureSet();
    
    /**
     * Creates a FeatureSet by copying another FeatureSet.
     */
    FeatureSet(const FeatureSet& other);

    /**
     * Adds a feature described by a FeatureDescriptor to the feature set.
     * Does nothing if the feature is already in the feature set.
     */
    void add(const FeatureDescriptor& descr);
    
    /**
     * Removes a feature described by a FeatureDescriptor from the feature set.
     * Does nothing if the feature is not in the feature set.
     */
    void remove(const FeatureDescriptor& descr);
    
    /**
     * Tests if a feature described by a FeatureDescriptor belongs to the 
     * feature set.
     */
    bool has(const FeatureDescriptor& descr) const;

    /**
     * Lists all features in the FeatureSet.
     */
    std::vector<FeatureDescriptor> list() const;

    /**
     * Returns a vector of FeatureDescriptors in this FeatureSet that match 
     * the given type and name (ignoring the parameters).
     */
    std::vector<FeatureDescriptor> list(DataDescriptor::Type type,
        const std::string& name) const;

    /**
     * Get a FeatureDescriptor in the FeatureSet
     * that matches the type and name of the given
     * FeatureDescriptor (ignoring the parameters). 
     * In case of multiple matches, an arbitrary one
     * is chosen.
     * Returns true iff a match was found.
     */
    bool getAny(FeatureDescriptor& descr) const;

    /**
     * Returns the index of a feature described by a FeatureDescriptor within
     * the feature set.
     */
    int indexOf(const FeatureDescriptor& descr) const;

    /**
     * Returns the size of the FeatureSet.
     */
    unsigned int size() const;

    /**
     * Removes all entries from the FeatureSet.
     */
    void clear();
    
    /**
     * Creates and returns a standard FeatureSet according to the configuration.
     */
    static FeatureSet getStandardSet();

private:
    typedef std::map<FeatureDescriptor, int> FeatureMap;
    FeatureMap _featureMap;
    int        _maxFeatureIndex;
};


} // namespace blissart


#endif // __BLISSART_FEATURESET_H__
