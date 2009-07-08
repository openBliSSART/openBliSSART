//
// $Id: HTKWriterTest.cpp 855 2009-06-09 16:15:50Z alex $
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


#include "HTKWriterTest.h"
#include <blissart/HTKWriter.h>
#include <blissart/linalg/Matrix.h>
#include <iostream>
#include <fstream>
#include <Poco/TemporaryFile.h>
#include <blissart/BinaryReader.h>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;


namespace Testing {


HTKWriterTest::HTKWriterTest()
{
}


bool HTKWriterTest::performTest()
{
    bool ok = false;

    {
        const double data[] = { 1, 2, 3, 4,
                                2, 3, 4, 5,
                                3, 4, 5, 6 };
        Matrix m(3, 4, data);
        cout << "Matrix M:" << endl << m << endl;
        Poco::TemporaryFile tmpFile;
        string path = tmpFile.path();
        ofstream ofs(path.c_str(), ios_base::out | ios_base::binary);
        HTKWriter::writeMatrix(ofs, m, 100000);
        ofs.close();
        
        ifstream ifs(path.c_str(), ios_base::in | ios_base::binary);
        BinaryReader br(ifs, BinaryReader::BigEndian);
        Poco::Int32 i32;
        Poco::Int16 i16;
        br >> i32;
        if ((unsigned int) i32 != m.cols())
            goto end;
        br >> i32;
        if (i32 != 100000) 
            goto end;
        br >> i16;
        if ((unsigned int) i16 != m.rows() * sizeof(float))
            goto end;
        br >> i16;
        if (i16 != HTKWriter::USER)
            goto end;

        float elem;
        for (unsigned int j = 0; j < m.cols(); ++j) {
            for (unsigned int i = 0; i < m.rows(); ++i) {
                br >> elem;
                if (!epsilonCheck(m.at(i, j), elem, 1e-6)) 
                    goto end;
            }
        }

        ok = true;
    }

end:
    return ok;
}


} // namespace Testing
