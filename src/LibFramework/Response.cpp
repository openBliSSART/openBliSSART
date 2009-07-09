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


#include <blissart/Response.h>
#include <set>


namespace blissart {


Response::Response() :
  DatabaseEntity(DatabaseEntity::Response),
  responseID(0)
{
}


Response::Response(const Response &other) :
  DatabaseEntity(other),
  responseID(other.responseID),
  name(other.name),
  description(other.description),
  labels(other.labels)
{
}


Response::Response(const std::string &name, const std::string &description) :
  DatabaseEntity(DatabaseEntity::Response),
  responseID(0),
  name(name),
  description(description)
{
}


int Response::numberOfClasses() const
{
  std::set<int> distinctClasses;
  for (LabelMap::const_iterator itr = labels.begin();
      itr != labels.end(); ++itr)
  {
      distinctClasses.insert(itr->second);
  }
  return (int) distinctClasses.size();
}


} // namespace blissart
