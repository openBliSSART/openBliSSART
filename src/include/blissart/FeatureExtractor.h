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


#ifndef __BLISSART_FEATUREEXTRACTOR_H__
#define __BLISSART_FEATUREEXTRACTOR_H__


#include <common.h>
#include <blissart/DataDescriptor.h>
#include <blissart/FeatureDescriptor.h>
#include <blissart/FeatureSet.h>
#include <Poco/SharedPtr.h>


namespace blissart {


// Forward declarations
namespace linalg { 
    class Vector; 
    class Matrix;
}


/**
 * Extracts features from vectors and matrices, as specified by the 
 * application's configuration.
 */
class LibFramework_API FeatureExtractor
{
public:
    /**
     * Constructs a FeatureExtractor.
     */
    FeatureExtractor();

    /**
     * A mapping of feature descriptors to feature values.
     */
    typedef std::map<FeatureDescriptor, double> FeatureMap;

    /**
     * Returns a FeatureMap containing the features for the given Vector,
     * according to its data type.
     */
    FeatureMap extract(DataDescriptor::Type type,
        const linalg::Vector& data);

    /**
     * Returns a FeatureMap containing the features for the given Matrix,
     * according to its data type.
     */
    FeatureMap extract(DataDescriptor::Type type,
        const linalg::Matrix& data);

    /**
     * Sets the sample frequency that is assumed in processing spectral vectors.
     */
    void setSampleFrequency(double freq);
    
    /**
     * Sets the "gains frequency", i.e. the frequency corresponding to one
     * transformation window, which is assumed in processing gains vectors.
     */
    void setGainsFrequency(double freq);

private:
    /** 
     * Computes the length of the rows of the gains matrix of a NMD of the 
     * given Matrix which is initialized with a response.
     * This method is not in LibFeature, since it depends on the database
     * and would thus make LibFeature depend on LibFramework.
     */
    void computeNMDGain(FeatureMap& target, const linalg::Matrix& data,
        int responseID, const std::string& costf, int nComponents, 
        int nIterations, DataDescriptor::Type type);

    double     _sampleFreq;
    double     _gainsFreq;
};


// Inlines


inline void FeatureExtractor::setSampleFrequency(double freq)
{
    _sampleFreq = freq;
}


inline void FeatureExtractor::setGainsFrequency(double freq)
{
    _gainsFreq = freq;
}


} // namespace blissart


#endif // __BLISSART_FEATUREEXTRACTOR_H__
