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


#include <blissart/FeatureSet.h>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>

#include <Poco/Exception.h>
#include <Poco/Util/LayeredConfiguration.h>
#include <Poco/NumberFormatter.h>


using Poco::Util::LayeredConfiguration;
using namespace std;


namespace blissart {


FeatureSet::FeatureSet() :
_maxFeatureIndex(-1)
{
}


FeatureSet::FeatureSet(const FeatureSet& other) : 
_featureMap(other._featureMap),
_maxFeatureIndex(other._maxFeatureIndex)
{
}


void FeatureSet::add(const FeatureDescriptor& descr)
{
    if (!has(descr))
        _featureMap[descr] = ++_maxFeatureIndex;
}


void FeatureSet::remove(const FeatureDescriptor &descr)
{
    FeatureMap::iterator itr = _featureMap.find(descr);
    if (itr != _featureMap.end()) {
        int index = itr->second;
        for (FeatureMap::iterator decItr = _featureMap.begin();
            decItr != _featureMap.end(); ++decItr)
        {
            if (decItr->second > index)
                --decItr->second;
        }
        _featureMap.erase(itr);
        --_maxFeatureIndex;
    }
}


bool FeatureSet::has(const FeatureDescriptor& descr) const
{
    return _featureMap.find(descr) != _featureMap.end();
}


vector<FeatureDescriptor> 
FeatureSet::list() const
{
    vector<FeatureDescriptor> result;
    for (FeatureMap::const_iterator itr = _featureMap.begin();
        itr != _featureMap.end(); ++itr)
    {
        result.push_back(itr->first);
    }
    return result;
}


vector<FeatureDescriptor> 
FeatureSet::list(DataDescriptor::Type type, const string& name) const
{
    vector<FeatureDescriptor> result;
    for (FeatureMap::const_iterator itr = _featureMap.begin();
        itr != _featureMap.end(); ++itr)
    {
        if (itr->first.dataType == type && itr->first.name == name)
            result.push_back(itr->first);
    }
    return result;
}


bool
FeatureSet::getAny(FeatureDescriptor &descr) const
{
    bool result = false;
    for (FeatureMap::const_iterator itr = _featureMap.begin();
        itr != _featureMap.end(); ++itr)
    {
        if (itr->first.dataType == descr.dataType && 
            itr->first.name == descr.name) 
        {
            descr = itr->first;
            result = true;
        }
    }
    return result;
}


int FeatureSet::indexOf(const FeatureDescriptor& descr) const
{
    FeatureMap::const_iterator itr = _featureMap.find(descr);
    if (itr == _featureMap.end()) {
        throw Poco::NotFoundException("Feature not found: " + 
            descr.toString());
    }
    return itr->second;
}


unsigned int FeatureSet::size() const
{
    return (unsigned int) _featureMap.size();
}


void FeatureSet::clear()
{
    _featureMap.clear();
    _maxFeatureIndex = 0;
}


FeatureSet FeatureSet::getStandardSet()
{
    FeatureSet fs;
    LayeredConfiguration &config = BasicApplication::instance().config();
    
    // Global parameters.
    int nMFCC = std::max<int>(1, 
        config.getInt("blissart.global.mfcc.count", 13));
    int firstMFCC = 
        config.getBool("blissart.global.mfcc.mfcc0", true) ? 0 : 1;

    // Features for component spectra.
    if (config.getBool("blissart.features.spectrum.stddev", true))
        fs.add(FeatureDescriptor("stddev", DataDescriptor::Spectrum));
    if (config.getBool("blissart.features.spectrum.centroid", true))
        fs.add(FeatureDescriptor("centroid", DataDescriptor::Spectrum));
    if (config.getBool("blissart.features.spectrum.rolloff", true))
        fs.add(FeatureDescriptor("rolloff", DataDescriptor::Spectrum));
    if (config.getBool("blissart.features.spectrum.noiselikeness", true)) {
        fs.add(FeatureDescriptor("noise-likeness", DataDescriptor::Spectrum,
            config.getDouble("blissart.features.spectrum.noiselikeness.sigma", 5.0)));
    }
    if (config.getBool("blissart.features.spectrum.dissonance", false))
        fs.add(FeatureDescriptor("dissonance", DataDescriptor::Spectrum));
    if (config.getBool("blissart.features.spectrum.flatness", false))
        fs.add(FeatureDescriptor("flatness", DataDescriptor::Spectrum));

    // Features for gains vectors.
    if (config.getBool("blissart.features.gains.stddev", false))
        fs.add(FeatureDescriptor("stddev", DataDescriptor::Gains));
    if (config.getBool("blissart.features.gains.skewness", false))
        fs.add(FeatureDescriptor("skewness", DataDescriptor::Gains));
    if (config.getBool("blissart.features.gains.kurtosis", false))
        fs.add(FeatureDescriptor("kurtosis", DataDescriptor::Gains));
    if (config.getBool("blissart.features.gains.pl", true))
        fs.add(FeatureDescriptor("pl", DataDescriptor::Gains));
    if (config.getBool("blissart.features.gains.pf", true))
        fs.add(FeatureDescriptor("pf", DataDescriptor::Gains));
    if (config.getBool("blissart.features.gains.percussiveness", true)) {
        fs.add(FeatureDescriptor("percussiveness", DataDescriptor::Gains,
            config.getDouble("blissart.features.gains.percussiveness.length", 0.2)));
    }
    if (config.getBool("blissart.features.gains.periodicity", true)) {
        fs.add(FeatureDescriptor("periodicity", DataDescriptor::Gains,
            config.getInt("blissart.features.gains.periodicity.bpm_min", 35),
            config.getInt("blissart.features.gains.periodicity.bpm_max", 240),
            config.getInt("blissart.features.gains.periodicity.bpm_step", 5)));
    }

    // Features for component spectra, magnitude and mel matrices.
    const char* matrixTypeNames[] = { "spectrum",
                                      "magnitudematrix", 
                                      "melmatrix", 
                                      0 };
    DataDescriptor::Type matrixTypes[] = { DataDescriptor::Spectrum,
                                           DataDescriptor::MagnitudeMatrix, 
                                           DataDescriptor::MelMatrix };
    DataDescriptor::Type* matrixType = matrixTypes;
    for (const char** str = matrixTypeNames; *str != 0; ++str) {
        string typeName(*str);
        int nFrames = std::max<int>(1, 
            config.getInt("blissart.features." + typeName + ".mfcc.frame_count", 5));
        for (int i = firstMFCC; i < nMFCC; ++i) {
            for (int f = 0; f < nFrames; ++f) {
                if (config.getBool("blissart.features." + typeName + ".mfcc", true)) {
                    fs.add(FeatureDescriptor("mfcc", *matrixType, i, nFrames, f));
                }
                if (config.getBool("blissart.features." + typeName + ".mfccD", true)) {
                    fs.add(FeatureDescriptor("mfccD", *matrixType, i, nFrames, f));
                }
                if (config.getBool("blissart.features." + typeName + ".mfccA", true)) {
                    fs.add(FeatureDescriptor("mfccA", *matrixType, i, nFrames, f));
                }
            }
            if (config.getBool("blissart.features." + typeName + ".mean_mfcc", true)) {
                fs.add(FeatureDescriptor("mean_mfcc", *matrixType, i));
                if (config.getBool("blissart.features." + typeName + ".mfccD", true)) {
                    fs.add(FeatureDescriptor("mean_mfccD", *matrixType, i));
                }
                if (config.getBool("blissart.features." + typeName + ".mfccA", true)) {
                    fs.add(FeatureDescriptor("mean_mfccA", *matrixType, i));
                }
            }
            if (config.getBool("blissart.features." + typeName + ".stddev_mfcc", true)) {
                fs.add(FeatureDescriptor("stddev_mfcc", *matrixType, i));
                if (config.getBool("blissart.features." + typeName + ".mfccD", true)) {
                    fs.add(FeatureDescriptor("stddev_mfccD", *matrixType, i));
                }
                if (config.getBool("blissart.features." + typeName + ".mfccA", true)) {
                    fs.add(FeatureDescriptor("stddev_mfccA", *matrixType, i));
                }
            }
        }
        if (config.getBool("blissart.features." + typeName + ".nmd_gain", false)) {
            int responseID = config.getInt(
                "blissart.features." + typeName + ".nmd_gain.response");
            int nComponents = config.getInt(
                "blissart.features." + typeName + ".nmd_gain.components", 0);
            DatabaseSubsystem& dbs = BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
            ResponsePtr response = dbs.getResponse(responseID);
            if (response.isNull()) {
                throw Poco::InvalidArgumentException(
                    "Invalid response ID for NMD gain (" + typeName + "): "
                    + Poco::NumberFormatter::format(responseID));
            }
            Response::LabelMap labels = response->labels;
            if (nComponents == 0) {
                nComponents = (int) labels.size();
            }
            for (Response::LabelMap::const_iterator itr = labels.begin();
                itr != labels.end(); ++itr)
            {
                fs.add(FeatureDescriptor("nmd_gain", *matrixType,
                    responseID, nComponents, itr->first));
            }
        }
        ++matrixType;
    }
    
    return fs;
}


} // namespace blissart
