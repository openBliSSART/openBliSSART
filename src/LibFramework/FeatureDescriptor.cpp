//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2010, Alexander Lehmann <lehmanna@in.tum.de>
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


#include <blissart/FeatureDescriptor.h>
#include <Poco/NumberFormatter.h>


namespace blissart {


FeatureDescriptor::FeatureDescriptor(const std::string& name, DataDescriptor::Type dataType,
                                     double param1, double param2, double param3) :
    name(name), 
    dataType(dataType)
{
    if (!isValidFeature(name, dataType)) {
        throw Poco::InvalidArgumentException(name + 
            " is not a valid feature for data type " + 
            DataDescriptor::strForType(dataType));
    }
    params[0] = param1;
    params[1] = param2;
    params[2] = param3;
}


FeatureDescriptor::FeatureDescriptor(const FeatureDescriptor &other) :
    name(other.name), 
    dataType(other.dataType)
{
    params[0] = other.params[0];
    params[1] = other.params[1];
    params[2] = other.params[2];
}


std::string FeatureDescriptor::toString() const
{
    return "[" + name + ", " + DataDescriptor::strForType(dataType) + ", ("
        + Poco::NumberFormatter::format(params[0]) + ", "
        + Poco::NumberFormatter::format(params[1]) + ", "
        + Poco::NumberFormatter::format(params[2]) + ")]";
}


bool FeatureDescriptor::operator < (const blissart::FeatureDescriptor& other) const
{
    bool res;
    if (dataType < other.dataType) {
        res = true;
    }
    else if (dataType == other.dataType) {
        if (name < other.name) {
            res = true;
        }
        else if (name == other.name) {
            if (params[0] < other.params[0]) {
                res = true;
            }
            else if (params[0] == other.params[0]) {
                if (params[1] < other.params[1]) {
                    res = true;
                }
                else if (params[1] == other.params[1]) {
                    if (params[2] < other.params[2]) {
                        res = true;
                    }
                    else {
                        res = false;
                    }
                }
                else {
                    res = false;
                }
            }
            else {
                res = false;
            }
        }
        else {
            res = false;
        }
    }
    else {
        res = false;
    }
    return res;
}


bool FeatureDescriptor::isValidFeature(const std::string& name,
                                       DataDescriptor::Type type)
{
    const char* validNamesSpectrum[] = {
        "mfcc",
        "mfccD",
        "mfccA",
        "mean_mfcc",
        "mean_mfccD",
        "mean_mfccA",
        "stddev_mfcc",
        "stddev_mfccD",
        "stddev_mfccA",
        "stddev",
        "centroid",
        "rolloff",
        "noise-likeness",
        "dissonance",
        "flatness",
        0
    };
    const char* validNamesMagn[] = {
        "mfcc",
        "mfccD",
        "mfccA",
        "mean_mfcc",
        "mean_mfccD",
        "mean_mfccA",
        "stddev_mfcc",
        "stddev_mfccD",
        "stddev_mfccA",
        0
    };
    const char* validNamesGains[] = {
        "stddev",
        "skewness",
        "kurtosis",
        "periodicity",
        "pl",
        "pf",
        "percussiveness",
        0
    };

    switch (type) {
        case DataDescriptor::Spectrum:
            for (const char** str = validNamesSpectrum; *str != 0; ++str) {
                if (name == *str)
                    return true;
            }
            break;
        case DataDescriptor::MagnitudeMatrix:
            for (const char** str = validNamesMagn; *str != 0; ++str) {
                if (name == *str)
                    return true;
            }
            break;
        case DataDescriptor::Gains:
            for (const char** str = validNamesGains; *str != 0; ++str) {
                if (name == *str)
                    return true;
            }
            break;
        default:
            // Intentionally left blank
            break;
    }
    return false;
}


} // namespace blissart
