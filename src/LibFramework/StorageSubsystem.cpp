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


#include <blissart/StorageSubsystem.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/ColVector.h>

#include <Poco/File.h>
#include <Poco/Util/Application.h>
#include <Poco/DigestStream.h>
#include <Poco/MD5Engine.h>
#include <Poco/NumberFormatter.h>
#include <Poco/NumberParser.h>


using Poco::Path;
using Poco::File;
using std::string;
using namespace blissart::linalg;


namespace blissart {


StorageSubsystem::StorageSubsystem() :
    _logger(Poco::Logger::get("openBliSSART.StorageSubsystem"))
{
}


void StorageSubsystem::store(const Matrix& matrix, DataDescriptorPtr info)
{
    Path location = getLocation(info);
    File locationDir(getLocation(info).parent());
    if (!locationDir.exists())
        locationDir.createDirectory();
    matrix.dump(location.toString());
    info->available = true;
}


void StorageSubsystem::store(const Vector& vector, DataDescriptorPtr info)
{
    Path location = getLocation(info);
    File locationDir(getLocation(info).parent());
    if (!locationDir.exists())
        locationDir.createDirectory();
    vector.dump(location.toString());
    info->available = true;
}


Path StorageSubsystem::getLocation(DataDescriptorPtr info)
{
    Poco::MD5Engine eng;
    Poco::DigestOutputStream dstr(eng);
    string storedFileName = Poco::NumberFormatter::format(info->descrID) + "." + 
                            Poco::NumberFormatter::format(info->type) + ".dat";
    dstr << storedFileName;
    dstr.close();
    string hash = Poco::DigestEngine::digestToHex(eng.digest()).substr(0, 2);
    Path location = _storageDir;
    location.pushDirectory(hash);
    location.setFileName(storedFileName);
    return location;
}


int StorageSubsystem::getDescriptorID(const Poco::Path& filename)
{
    int rv;
    Poco::MD5Engine eng;
    Poco::DigestOutputStream dstr(eng);
    dstr << filename.getFileName();
    dstr.close();
    string hash = Poco::DigestEngine::digestToHex(eng.digest()).substr(0, 2);
    if (filename.depth() < 1 ||
        filename.directory(filename.depth() - 1) != hash) 
    {
        throw Poco::InvalidArgumentException("Empty or wrong directory: "
            + filename.toString());
    }
    string::size_type dotPos = filename.getFileName().find_first_of('.');
    if (dotPos == string::npos) {
        throw Poco::InvalidArgumentException("Invalid file name: "
            + filename.toString());
    }
    string idString = filename.getFileName().substr(0, dotPos);
    if (!Poco::NumberParser::tryParse(idString, rv)) {
        throw Poco::InvalidArgumentException("Invalid file name: "
            + filename.toString());
    }
    return rv;
}


void StorageSubsystem::initialize(Poco::Util::Application& app)
{
    _storageDir = app.config().getString("blissart.storageDir", "");
}


void StorageSubsystem::uninitialize()
{
}


const char* StorageSubsystem::name() const
{
    return "Storage subsystem";
}


} // namespace blissart
