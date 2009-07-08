//
// $Id: ARFFExporter.cpp 855 2009-06-09 16:15:50Z alex $
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


#include "ARFFExporter.h"
#include <blissart/linalg/Matrix.h>

#include <fstream>
#include <iomanip>
#include <ctime>
#include <cmath>


using namespace std;
using namespace blissart::linalg;


ARFFExporter::ARFFExporter(const string& fileName) :
    _fileName(fileName)
{
}


bool ARFFExporter::doExport(const Matrix& matrix)
{
    ofstream os(_fileName.c_str(), ios_base::out | ios_base::trunc);
    if (os.fail())
        return false;
    
    char strDateTime[80]; // 80 chars is more than enough
    time_t dateTime = time(NULL);
#if defined(_WIN32) || defined(_MSC_VER)
# pragma warning(push)
# pragma warning(disable:4996)
#endif
    strftime(strDateTime, 79, "%x %X", localtime(&dateTime));
#if defined(_WIN32) || defined(_MSC_VER)
# pragma warning(pop)
#endif

    // Header
    os << "% icatool, " << strDateTime << endl
       << "%" << endl
       << "% The exported dataset (matrix) has" << endl
       << "% " << setw(15) << matrix.rows() << " rows and" << endl
       << "% " << setw(15) << matrix.cols() << " columns." << endl
       << endl;

    // Relation name
    os << "@RELATION 'Independent Components'" << endl
       << endl;

    // Attributes description
    // One attribute per component
    const unsigned int numDigits = 1 + (unsigned int)log10f((float)matrix.rows());
    for (unsigned int i = 0; i < matrix.rows(); i++) {
        os << "@ATTRIBUTE component" << setfill('0') << setw(numDigits) << i
           << " NUMERIC" << endl;
    }
    os << endl;

    // Data
    os << "@DATA" << endl;
    for (unsigned int j = 0; j < matrix.cols(); j++) {
        for (unsigned int i = 0; i < matrix.rows()-1; i++) {
            os << matrix.at(i, j) << ",";
        }
        os << matrix.at(matrix.rows()-1, j) << endl;
    }

    os.close();
    return true;
}
