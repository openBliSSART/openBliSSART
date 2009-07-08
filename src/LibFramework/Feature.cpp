//
// $Id: Feature.cpp 855 2009-06-09 16:15:50Z alex $
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


#include <blissart/Feature.h>
#include <sstream>


namespace blissart {


// XXX: these constructors should somehow check the parameters.
// But how, if there is no DataDescriptor::Type given?


Feature::Feature() :
  DatabaseEntity(DatabaseEntity::Feature),
  descrID(0),
  value(0.0)
{
    params[0] = 0.0;
    params[1] = 0.0;
    params[2] = 0.0;
}


Feature::Feature(const Feature &other) :
  DatabaseEntity(other),
  descrID(other.descrID),
  name(other.name),
  value(other.value)
{
    params[0] = other.params[0];
    params[1] = other.params[1];
    params[2] = other.params[2];
}


Feature::Feature(int descrID, const std::string& name, double value) :
  DatabaseEntity(DatabaseEntity::DataDescriptor),
  descrID(descrID),
  name(name),
  value(value)
{
    params[0] = 0.0;
    params[1] = 0.0;
    params[2] = 0.0;
}


Feature::Feature(int descrID, const std::string& name, 
                 double param1, double param2, double param3,
                 double value) :
  DatabaseEntity(DatabaseEntity::DataDescriptor),
  descrID(descrID),
  name(name),
  value(value)
{
    params[0] = param1;
    params[1] = param2;
    params[2] = param3;
}


std::string Feature::description() const
{
    std::ostringstream d;
    d << name 
      << "(" << params[0] << "," << params[1] << "," << params[2] << ")";
    return d.str();
}


} // namespace blissart

