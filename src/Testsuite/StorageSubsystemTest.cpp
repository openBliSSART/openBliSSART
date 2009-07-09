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


#include "StorageSubsystemTest.h"
#include <blissart/StorageSubsystem.h>
#include <blissart/DataDescriptor.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/BasicApplication.h>
#include <blissart/linalg/generators/generators.h>

#include <iostream>
#include <string>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;


namespace Testing {


bool StorageSubsystemTest::performTest()
{
    StorageSubsystem& storage = BasicApplication::instance().
                                getSubsystem<StorageSubsystem>();

    DataDescriptorPtr data = new DataDescriptor;
    data->type = DataDescriptor::PhaseMatrix;
    data->descrID = 42;
    DataDescriptorPtr data2 = new DataDescriptor;
    data2->type = DataDescriptor::Spectrum;
    data2->descrID = 23;
    string location;
    try {
        location = storage.getLocation(data2).toString();
        cout << "Location of file 1: " << location << endl;
        location = storage.getLocation(data).toString();
        cout << "Location of file 2: " << location << endl;
        cout << "Descriptor ID from location: " 
             << storage.getDescriptorID(location) << endl;
        cout << "Storing a random matrix there." << endl;
        Matrix t(2, 2, generators::random);
        storage.store(t, data);
    }
    catch (const Poco::Exception& exc) {
        cerr << exc.displayText() << endl;
        return false;
    }
    return true;
}


} // namespace Testing
