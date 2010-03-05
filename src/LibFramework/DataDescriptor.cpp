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


#include <blissart/DataDescriptor.h>
#include <Poco/Exception.h>


using Poco::NotImplementedException;


namespace blissart {


DataDescriptor::DataDescriptor() :
  DatabaseEntity(DatabaseEntity::DataDescriptor),
  descrID(0),
  processID(0),
  type(Invalid),
  index(0),
  index2(0),
  available(true)
{
}


DataDescriptor::DataDescriptor(const DataDescriptor &other) :
  DatabaseEntity(other),
  descrID(other.descrID),
  processID(other.processID),
  type(other.type),
  index(other.index),
  index2(other.index2),
  available(other.available)
{
}


std::string DataDescriptor::strForType(DataDescriptor::Type type)
{
    switch (type) {
    case MagnitudeMatrix:
        return "Magnitude Matrix";
        break;
    case PhaseMatrix:
        return "Phase Matrix";
        break;
    case Spectrum:
        return "Spectrum";
        break;
    case Gains:
        return "Gains";
        break;
    case FeatureMatrix:
        return "Feature matrix";
        break;
    default:
        throw NotImplementedException("Unknown data descriptor type.");
    }
}


std::string DataDescriptor::strForTypeShort(DataDescriptor::Type type)
{
    // 5 characters max!
    switch (type) {
    case MagnitudeMatrix:
        return "mmatr";
        break;
    case PhaseMatrix:
        return "phase";
        break;
    case Spectrum:
        return "spect";
        break;
    case Gains:
        return "gains";
        break;
    case FeatureMatrix:
        return "fmatr";
        break;
    default:
        throw NotImplementedException("Unknown data descriptor type.");
    }
}


} // namespace blissart

