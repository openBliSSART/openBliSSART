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


#include <blissart/ClassificationObject.h>
#include <Poco/Exception.h>


using Poco::NotImplementedException;


namespace blissart {


ClassificationObject::ClassificationObject() :
  DatabaseEntity(DatabaseEntity::ClassificationObject),
  objectID(0),
  type(Invalid)
{
}


ClassificationObject::ClassificationObject(const ClassificationObject &other) :
  DatabaseEntity(other),
  objectID(other.objectID),
  type(other.type),
  descrIDs(other.descrIDs),
  labelIDs(other.labelIDs)
{
}


std::string ClassificationObject::strForType(ClassificationObject::Type type)
{
    switch (type) {
    case NMDComponent:
        return "NMD Component";
        break;
    case ImportedAudio:
        return "Imported Audio";
        break;
    case Spectrogram:
        return "Spectrogram";
        break;
    default:
        throw NotImplementedException("Unknown classification object type.");
    }
}


} // namespace blissart

