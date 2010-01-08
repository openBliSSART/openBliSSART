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


#include <blissart/FeatureExtractor.h>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/TargetedDeconvolver.h>

#include <blissart/linalg/Vector.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/Matrix.h>

#include <blissart/feature/mfcc.h>
#include <blissart/feature/peak.h>
#include <blissart/feature/misc.h>

#include <Poco/NumberFormatter.h>
#include <Poco/Exception.h>

#include <sstream>
#include <vector>
#include <algorithm>


using namespace std;
using namespace blissart::linalg;


namespace blissart {


FeatureExtractor::FeatureExtractor()
{
}


FeatureExtractor::FeatureMap
FeatureExtractor::extract(DataDescriptor::Type type, const Vector &data)
{
    Poco::Util::LayeredConfiguration& config = BasicApplication::instance().config();
    FeatureMap result;

    if (type == DataDescriptor::Gains) {
        // Standard deviation
        if (config.getBool("blissart.features.gains.stddev", false)) {
            result[FeatureDescriptor("stddev", type)] = feature::stddev(data);
        }

        // Skewness
        if (config.getBool("blissart.features.gains.skewness", false)) {
            result[FeatureDescriptor("skewness", type)] = feature::skewness(data);
        }

        // Kurtosis
        if (config.getBool("blissart.features.gains.kurtosis", false)) {
            result[FeatureDescriptor("kurtosis", type)] = feature::kurtosis(data);
        }

        // Periodicity
        if (config.getBool("blissart.features.gains.periodicity", true)) {
            int bpmMin = config.getInt(
                "blissart.features.gains.periodicity.bpm_min", 35);
            int bpmMax = config.getInt(
                "blissart.features.gains.periodicity.bpm_max", 240);
            int bpmStep = config.getInt(
                "blissart.features.gains.periodicity.bpm_step", 5);
            result[FeatureDescriptor("periodicity", type, bpmMin, bpmMax, bpmStep)] = 
                feature::periodicity(data, _gainsFreq, bpmMin, bpmMax, bpmStep);
        }

        // Peak length / fluctuation
        if (config.getBool("blissart.features.gains.pl", true)) {
            result[FeatureDescriptor("pl", type)] = feature::averagePeakLength(data);
        }
        if (config.getBool("blissart.features.gains.pf", true)) {
            result[FeatureDescriptor("pf", type)] = feature::peakFluctuation(data);
        }

        // Percussiveness
        if (config.getBool("blissart.features.gains.percussiveness", true)) {
            // The factor of 0.2 corresponds to 200 ms
            double length = config.getDouble(
                "blissart.features.gains.percussiveness.length", 0.2);
            result[FeatureDescriptor("percussiveness", type, length)] =
                feature::percussiveness(data, (unsigned int) (length * _gainsFreq));
        }
    } // Gains

    return result;
}


FeatureExtractor::FeatureMap
FeatureExtractor::extract(DataDescriptor::Type type, const Matrix &data)
{
    Poco::Util::LayeredConfiguration& config = BasicApplication::instance().config();
    FeatureMap result;

    if (type == DataDescriptor::Spectrum) {
        ColVector mean = data.meanColumnVector();

        // Standard deviation
        if (config.getBool("blissart.features.spectrum.stddev", true)) {
            result[FeatureDescriptor("stddev", type)] = feature::stddev(mean);
        }

        // Spectral centroid
        if (config.getBool("blissart.features.spectrum.centroid", true)) {
            result[FeatureDescriptor("centroid", type)] = feature::centroid(mean,
                _sampleFreq / 2.0);
        }

        // Roll-off point
        if (config.getBool("blissart.features.spectrum.rolloff", true)) {
            result[FeatureDescriptor("rolloff", type)] = feature::rolloff(mean,
                _sampleFreq / 2.0);
        }

        // Noise-likeness
        if (config.getBool("blissart.features.spectrum.noiselikeness", true)) {
            double sigma = config.getDouble(
                "blissart.features.spectrum.noiselikeness.sigma", 5.0);
            result[FeatureDescriptor("noise-likeness", type, sigma)] = 
                feature::noiseLikeness(mean, sigma);
        }

        // Dissonance
        if (config.getBool("blissart.features.spectrum.dissonance", false)) {
            result[FeatureDescriptor("dissonance", type)] =
                feature::spectralDissonance(mean, _sampleFreq / 2.0);
        }

        // Flatness
        if (config.getBool("blissart.features.spectrum.flatness", false)) {
            result[FeatureDescriptor("flatness", type)] =
                feature::spectralFlatness(mean);
        }
    } // Spectrum

    if (type == DataDescriptor::Spectrum ||
        type == DataDescriptor::MagnitudeMatrix || 
        type == DataDescriptor::MelMatrix) 
    {
        string typeName;
        if (type == DataDescriptor::Spectrum)
            typeName = "spectrum";
        else if (type == DataDescriptor::MagnitudeMatrix)
            typeName = "magnitudematrix";
        else if (type == DataDescriptor::MelMatrix)
            typeName = "melmatrix";

        // MFCC, delta and delta-delta (_A_cceleration)
        Poco::SharedPtr<Matrix> cepstrogram, cepstrogramD, cepstrogramA;
        bool mfcc  = config.getBool("blissart.features." + typeName + ".mfcc", true);
        bool mfccD = config.getBool("blissart.features." + typeName + ".mfccD", true);
        bool mfccA = config.getBool("blissart.features." + typeName + ".mfccA", true);

        // TODO: Save all these parameters in the feature descriptor
        int nCoeff = config.getInt("blissart.global.mfcc.count", 13);
        int firstCoeff = 0;
        if (!config.getBool("blissart.global.mfcc.mfcc0", true)) {
            ++firstCoeff;
        }
        int maxFrameCount = config.
            getInt("blissart.features." + typeName + "mfcc.frame_count", 5);
        unsigned int nBands = config.
            getInt("blissart.global.mel_bands", 26);
        unsigned int theta = (unsigned int) (config.
            getInt("blissart.global.deltaregression.theta", 2));
        double lifter = config.
            getDouble("blissart.global.mfcc.lifter", 22.0);
        double lowFreq = config.
            getDouble("blissart.global.mel_filter.low_freq", 0.0);
        double highFreq = config.
            getDouble("blissart.global.mel_filter.high_freq", 0.0);
    
        // Sampled MFCC
        if ((mfcc || mfccD || mfccA) && nCoeff > 0) {
            // We precompute MFCCs for all columns here, as we assume that 
            // in most cases, average MFCCs will be used too.
            cepstrogram = feature::computeMFCC(data, _sampleFreq, nCoeff, 
                nBands, lowFreq, highFreq, lifter);
            if (mfccD || mfccA) {
                cepstrogramD = feature::deltaRegression(*cepstrogram, theta);
            }
            if (mfccA) {
                cepstrogramA = feature::deltaRegression(*cepstrogramD, theta);
            }
            // Insert each desired MFCC into result structure.
            for (unsigned int mfccIndex = firstCoeff;
                 mfccIndex < (unsigned int)nCoeff; 
                 ++mfccIndex) 
            {
                // Here the 2nd feature parameter has to be set to the
                // maximal frame count, not the real one (which is 1).
                // Otherwise the feature won't be found later, as FeatureSet::
                // getStandardSet() uses the count from the config file
                // as parameter and doesn't now about the matrix properties.
                for (int frameIndex = 0; frameIndex < maxFrameCount; ++frameIndex) 
                {
                    // If the actual frame count is lower than maxFrameCount
                    // a frame might be taken into account more than once.
                    unsigned int col = (unsigned int)
                        ((double) frameIndex / (maxFrameCount - 1) * (data.cols() - 1));
                    if (mfcc) {
                        result[FeatureDescriptor("mfcc", type, 
                            mfccIndex, maxFrameCount, frameIndex)] = 
                            cepstrogram->at(mfccIndex, col);
                    }
                    if (mfccD) {
                        result[FeatureDescriptor("mfccD", type, 
                            mfccIndex, maxFrameCount, frameIndex)] = 
                            cepstrogramD->at(mfccIndex, col);
                    }
                    if (mfccA) {
                        result[FeatureDescriptor("mfccA", type, 
                            mfccIndex, maxFrameCount, frameIndex)] = 
                            cepstrogramA->at(mfccIndex, col);
                    }
                }
                result[FeatureDescriptor("mfcc", type, 
                    mfccIndex, maxFrameCount, 0)] = cepstrogram->at(mfccIndex, 0);
            }
        }

        // Average MFCCs, Delta- and Delta-Delta MFCCs
        if (config.getBool("blissart.features." + typeName + ".mean_mfcc", true) 
            && nCoeff > 0) 
        {
            if (cepstrogram.isNull()) {
                cepstrogram = feature::computeMFCC(data, _sampleFreq, nCoeff, nBands, 
                    lowFreq, highFreq, lifter);
            }
            if ((mfccD || mfccA) && cepstrogramD.isNull()) {
                cepstrogramD = feature::deltaRegression(*cepstrogram, theta);
            }
            if (mfccA && cepstrogramA.isNull()) {
                cepstrogramA = feature::deltaRegression(*cepstrogramD, theta);
            }
            ColVector mean = cepstrogram->meanColumnVector();
            for (unsigned int mfccIndex = firstCoeff;
                 mfccIndex < (unsigned int)nCoeff;
                 ++mfccIndex)
            {
                result[FeatureDescriptor("mean_mfcc", type, mfccIndex)] 
                    = mean.at(mfccIndex);
            }
            if (mfccD && !cepstrogramD.isNull()) {
                mean = cepstrogramD->meanColumnVector();
                for (unsigned int mfccIndex = firstCoeff;
                     mfccIndex < (unsigned int)nCoeff;
                     ++mfccIndex)
                {
                    result[FeatureDescriptor("mean_mfccD", type, mfccIndex)] 
                        = mean.at(mfccIndex);
                }
            }
            if (mfccA && !cepstrogramA.isNull()) {
                mean = cepstrogramA->meanColumnVector();
                for (unsigned int mfccIndex = firstCoeff;
                     mfccIndex < (unsigned int)nCoeff;
                     ++mfccIndex)
                {
                    result[FeatureDescriptor("mean_mfccA", type, mfccIndex)] 
                        = mean.at(mfccIndex);
                }
            }
        }

        // Standard deviation of MFCCs, Delta- and Delta-Delta MFCCs
        if (config.getBool("blissart.features." + typeName + ".stddev_mfcc", true) 
            && nCoeff > 0) 
        {
            if (cepstrogram.isNull()) {
                cepstrogram = feature::computeMFCC(data, _sampleFreq, nCoeff, nBands,
                    lowFreq, highFreq, lifter);
            }
            if ((mfccD || mfccA) && cepstrogramD.isNull()) {
                cepstrogramD = feature::deltaRegression(*cepstrogram, theta);
            }
            if (mfccA && cepstrogramA.isNull()) {
                cepstrogramA = feature::deltaRegression(*cepstrogramD, theta);
            }
            for (unsigned int mfccIndex = firstCoeff;
                 mfccIndex < (unsigned int)nCoeff;
                 ++mfccIndex)
            {
                result[FeatureDescriptor("stddev_mfcc", type, mfccIndex)] 
                    = feature::stddev(cepstrogram->nthRow(mfccIndex));
            }
            if (mfccD && cepstrogramD.isNull()) {
                for (unsigned int mfccIndex = firstCoeff;
                     mfccIndex < (unsigned int)nCoeff;
                     ++mfccIndex)
                {
                    result[FeatureDescriptor("stddev_mfccD", type, mfccIndex)] 
                        = feature::stddev(cepstrogramD->nthRow(mfccIndex));
                }
            }
            if (mfccA && cepstrogramA.isNull()) {
                for (unsigned int mfccIndex = firstCoeff;
                     mfccIndex < (unsigned int)nCoeff;
                     ++mfccIndex)
                {
                    result[FeatureDescriptor("stddev_mfccA", type, mfccIndex)] 
                        = feature::stddev(cepstrogramA->nthRow(mfccIndex));
                }
            }
        }

        // NMD gain
        if (type != DataDescriptor::Spectrum &&
            config.getBool("blissart.features." + typeName + ".nmd_gain", false)) 
        {
            int responseID = config.getInt(
                "blissart.features." + typeName + ".nmd_gain.response");
            int nIterations = config.getInt(
                "blissart.features." + typeName + ".nmd_gain.iterations", 200);
            // XXX: Use parameter "# noise components" instead?
            unsigned int nComponents = config.getInt(
                "blissart.features." + typeName + ".nmd_gain.components", 0);
            bool allComponents = config.getBool(
                "blissart.features." + typeName + ".nmd_gain.allcomponents", false);
            bool sumByLabel = config.getBool(
                "blissart.features." + typeName + ".nmd_gain.sumbylabel", false);
            string cfName = config.getString(
                "blissart.features." + typeName + ".nmd_gain.costfunction", "div");
            computeNMDGain(result, data, responseID, cfName, 
                nComponents, allComponents, sumByLabel, nIterations, type);
        }
    
    } // type == Spectrum || type == MagnitudeMatrix || 
      // type == DataDescriptor::MelMatrix

    return result;
}


void
FeatureExtractor::computeNMDGain(FeatureExtractor::FeatureMap& target, 
                                 const Matrix& data,
                                 int responseID, const string& cfName,
                                 int nComponents, bool allComponents,
                                 bool sumByLabel,
                                 int nIterations,
                                 DataDescriptor::Type type)
{
    DatabaseSubsystem& dbs = BasicApplication::instance().
        getSubsystem<DatabaseSubsystem>();
    ResponsePtr response = dbs.getResponse(responseID);
    if (response.isNull()) {
        throw Poco::InvalidArgumentException("Invalid response ID: " +
            Poco::NumberFormatter::format(responseID));
    }

    vector<int> nmdObjectIDs;
    for (Response::LabelMap::const_iterator itr = response->labels.begin();
        itr != response->labels.end(); ++itr)
    {
        nmdObjectIDs.push_back(itr->first);
    }

    if (nComponents == 0) {
        nComponents = (unsigned int) nmdObjectIDs.size();
    }

    TargetedDeconvolver d(data, nComponents, nmdObjectIDs);

    // All components initialized => keep whole W matrix constant
    if ((unsigned int)nComponents == nmdObjectIDs.size()) {
        d.keepWConstant(true);
    }
    // Some components randomized => only keep corresponding W cols constant
    else {
        for (unsigned int c = 0; c < nmdObjectIDs.size(); ++c) {
            d.keepWColumnConstant(c, true);
        }
    }

    if (cfName.substr(0, 3) == "div") {
        d.factorizeKL(nIterations, 0);
    }
    else if (cfName.substr(0, 4) == "dist") {
        d.factorizeED(nIterations, 0);
    }
    else {
        throw Poco::InvalidArgumentException("Invalid cost function: " +
            cfName);
    }

    const Matrix& h = d.getH();

    double totalLength = 0.0;
    unsigned int relevantComponents = allComponents ? 
                                      nComponents : 
                                      nmdObjectIDs.size();
    
    if (sumByLabel) {
        map<int, double> lengthByLabel;
        unsigned int compIndex = 0;
        for (vector<int>::const_iterator itr = nmdObjectIDs.begin();
            itr != nmdObjectIDs.end(); ++itr, ++compIndex) 
        {
            double l = h.nthRow(compIndex).length();
            lengthByLabel[response->labels[*itr]] += l;
            totalLength += l;
        }
        // Initialized components
        for (map<int, double>::const_iterator itr = lengthByLabel.begin();
            itr != lengthByLabel.end(); ++itr)
        {
            // XXX: Here the feature name differs from the configuration
            // option name.
            target[FeatureDescriptor("nmd_gain_label", type,
                responseID, nComponents, itr->first)] = 
                itr->second / totalLength;
        }
        // Uninitialized components (label 0)
        if (compIndex < relevantComponents) {
            double otherLength = 0.0;
            while (compIndex < relevantComponents) {
                double l = h.nthRow(compIndex).length();
                otherLength += l;
                totalLength += l;
                ++compIndex;
            }
            target[FeatureDescriptor("nmd_gain_label", type,
                responseID, nComponents, 0)] = 
                otherLength / totalLength;
        }
    }

    else {
        vector<double> lengths(nComponents);
        unsigned int compIndex = 0;
        for (; compIndex < h.rows(); ++compIndex) {
            double l = h.nthRow(compIndex).length();
            lengths[compIndex] = l;
            //if (compIndex < nmdObjectIDs.size()) {
                totalLength += l;
            //}
        }
        compIndex = 0;
        for (vector<int>::const_iterator itr = nmdObjectIDs.begin();
            itr != nmdObjectIDs.end(); ++itr, ++compIndex) 
        {
            // TODO: Include iterations parameter
            target[FeatureDescriptor("nmd_gain", type,
                responseID, nComponents, *itr)] = 
                lengths[compIndex] / totalLength;
        }
        // Gains from uninitialized components have the negative component index 
        // as parameter.
        while (compIndex < relevantComponents) {
            target[FeatureDescriptor("nmd_gain", type,
                responseID, nComponents, -((double)compIndex + 1))] = 
                lengths[compIndex] / totalLength;
            ++compIndex;
        }
    }
}


} // namespace blissart
