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


#include <blissart/HTKWriter.h>
#include <blissart/linalg/Matrix.h>
#include <Poco/Exception.h>


using Poco::IOException;
using blissart::linalg::Matrix;


namespace blissart {


HTKWriter::HTKWriter(std::ostream &str) :
Poco::BinaryWriter(str, Poco::BinaryWriter::BIG_ENDIAN_BYTE_ORDER)
{
}


void
HTKWriter::writeHeader(Poco::Int32 nSamples, Poco::Int32 samplePeriod, 
                       Poco::Int16 sampleSize, Poco::Int16 parmKind)
{
    *this << nSamples;
    *this << samplePeriod;
    *this << sampleSize;
    *this << parmKind;
    if (fail()) {
        throw Poco::IOException("Could not write HTK header");
    }
}


void
HTKWriter::writeMatrix(std::ostream& str,
                       const Matrix& matrix, Poco::Int32 samplePeriod,
                       Poco::Int16 parmKind)
{
    HTKWriter writer(str);
    writer.writeHeader(matrix.cols(), samplePeriod, 
        (Poco::Int16) sizeof(float) * matrix.rows(), 
        parmKind);
    for (unsigned int j = 0; j < matrix.cols(); ++j) {
        for (unsigned int i = 0; i < matrix.rows(); ++i) {
            writer << (float) matrix.at(i, j);
            if (writer.fail()) {
                throw Poco::IOException("Could not write data to stream");
            }
        }
    }
}


} // namespace blissart

