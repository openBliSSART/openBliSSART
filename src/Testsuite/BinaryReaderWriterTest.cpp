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


#include "BinaryReaderWriterTest.h"
#include <blissart/BinaryReader.h>
#include <blissart/BinaryWriter.h>

#include <Poco/TemporaryFile.h>

#include <fstream>
#include <iostream>


using namespace std;


namespace Testing {


bool BinaryReaderWriterTest::performTest()
{
    const float float_buf[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const double double_buf[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    Poco::TemporaryFile tmpFile;


    //{
    //    uint16_t uval16 = 0xAABB;
    //    uint32_t uval32 = 0xAABBCCDD;
    //    uint64_t uval64 = 0xAABBCCDDEEFFFEDCULL;
    //    flip16(&uval16);
    //    flip32(&uval32);
    //    flip64(&uval64);
    //    debug_assert(uval16 == 0xBBAA);
    //    debug_assert(uval32 == 0xDDCCBBAA);
    //    debug_assert(uval64 == 0xDCFEFFEEDDCCBBAAULL);
    //}

    cout << "BigEndian:" << endl;
    {
        ofstream fs(tmpFile.path().c_str(),
                    ios_base::out | ios_base::trunc | ios_base::binary);
        if (fs.fail()) {
            cerr << "Error while opening the temporary file." << endl;
            return false;
        }
        blissart::BinaryWriter bw(fs, blissart::BinaryWriter::BigEndian);

        cout << "\tWriting 16-, 32- and 64-bit unsigned and signed values."
             << endl;
        bw << (uint16_t)0xAABB
           << (uint32_t)0xAABBCCDD
           << (uint64_t)0xAABBCCDDEEFFFEDCULL
           << (sint16_t)-0x1234
           << (sint32_t)-0x12345678
           << (sint64_t)-0x1234567890ABCDEFLL
           << (double)1234.56
           << (float)7890.12;
        if (bw.fail())
            return false;

        cout << "\tWriting float array." << endl;
        if (bw.writeFloats(float_buf, 10) != 10 || bw.fail())
            return false;

        cout << "\tWriting double array." << endl;
        if (bw.writeDoubles(double_buf, 10) != 10 || bw.fail())
            return false;

        fs.close();
    }

    {
        ifstream fs(tmpFile.path().c_str(), ios_base::in | ios_base::binary);
        if (fs.fail()) {
            cerr << "Error while opening the temporary file." << endl;
            return false;
        }
        blissart::BinaryReader br(fs, blissart::BinaryReader::BigEndian);

        uint16_t uval16;
        uint32_t uval32;
        uint64_t uval64;
        sint16_t sval16;
        sint32_t sval32;
        sint64_t sval64;
        double dval;
        float fval;

        cout << "\tReading 16-, 32- and 64-bit unsigned and signed values."
             << endl;
        br >> uval16 >> uval32 >> uval64
           >> sval16 >> sval32 >> sval64
           >> dval >> fval;
        if (br.fail() ||
            uval16 != (uint16_t)0xAABB ||
            uval32 != (uint32_t)0xAABBCCDD ||
            uval64 != (uint64_t)0xAABBCCDDEEFFFEDCULL ||
            sval16 != (sint16_t)-0x1234 ||
            sval32 != (sint32_t)-0x12345678 ||
            sval64 != (sint64_t)-0x1234567890ABCDEFLL ||
            dval != (double)1234.56 ||
            fval != (float)7890.12)
        {
            return false;
        }

        cout << "\tReading float array." << endl;
        float float_tmp[10];
        if (br.readFloats(float_tmp, 10) != 10 || br.fail())
            return false;
        for (int i = 0; i < 10; i++)
            if (float_tmp[i] != float_buf[i])
                return false;

        cout << "\tReading double array." << endl;
        double double_tmp[10];
        if (br.readDoubles(double_tmp, 10) != 10 || br.fail())
            return false;
        for (int i = 0; i < 10; i++) {
            if (double_tmp[i] != double_buf[i])
                return false;
        }

        fs.close();
    }

    cout << "LittleEndian:" << endl;
    {
        ofstream fs(tmpFile.path().c_str(),
                    ios_base::out | ios_base::trunc | ios_base::binary);
        if (fs.fail()) {
            cerr << "Error while opening the temporary file." << endl;
            return false;
        }
        blissart::BinaryWriter bw(fs, blissart::BinaryWriter::LittleEndian);

        cout << "\tWriting 16-, 32- and 64-bit unsigned and signed values."
             << endl;
        bw << (uint16_t)0xAABB
           << (uint32_t)0xAABBCCDD
           << (uint64_t)0xAABBCCDDEEFFFEDCULL
           << (sint16_t)-0x1234
           << (sint32_t)-0x12345678
           << (sint64_t)-0x1234567890ABCDEFLL
           << (double)1234.56
           << (float)7890.12;
        if (bw.fail())
            return false;

        cout << "\tWriting float array." << endl;
        if (bw.writeFloats(float_buf, 10) != 10 || bw.fail())
            return false;

        cout << "\tWriting double array." << endl;
        if (bw.writeDoubles(double_buf, 10) != 10 || bw.fail())
            return false;

        fs.close();
    }

    {
        ifstream fs(tmpFile.path().c_str(), ios_base::in | ios_base::binary);
        if (fs.fail()) {
            cerr << "Error while opening the temporary file." << endl;
            return false;
        }
        blissart::BinaryReader br(fs, blissart::BinaryReader::LittleEndian);

        uint16_t uval16;
        uint32_t uval32;
        uint64_t uval64;
        sint16_t sval16;
        sint32_t sval32;
        sint64_t sval64;
        double dval;
        float fval;

        cout << "\tReading 16-, 32- and 64-bit unsigned and signed values."
             << endl;
        br >> uval16 >> uval32 >> uval64
           >> sval16 >> sval32 >> sval64
           >> dval >> fval;
        if (br.fail() ||
            uval16 != (uint16_t)0xAABB ||
            uval32 != (uint32_t)0xAABBCCDD ||
            uval64 != (uint64_t)0xAABBCCDDEEFFFEDCULL ||
            sval16 != (sint16_t)-0x1234 ||
            sval32 != (sint32_t)-0x12345678 ||
            sval64 != (sint64_t)-0x1234567890ABCDEFLL ||
            dval != (double)1234.56 ||
            fval != (float)7890.12)
        {
            return false;
        }

        cout << "\tReading float array." << endl;
        float float_tmp[10];
        if (br.readFloats(float_tmp, 10) != 10 || br.fail())
            return false;
        for (int i = 0; i < 10; i++) {
            if (float_tmp[i] != float_buf[i]) {
                std::cout << float_tmp[i] << " xxx " << float_buf[i] << std::endl;
                std::cout << std::hex << *((uint32_t *)(float_tmp + i)) << " xxx " << std::hex << *((uint32_t *)(float_buf + i)) << std::endl;
                return false;
            }
        }

        cout << "\tReading double array." << endl;
        double double_tmp[10];
        if (br.readDoubles(double_tmp, 10) != 10 || br.fail())
            return false;
        for (int i = 0; i < 10; i++) {
            if (double_tmp[i] != double_buf[i])
                return false;
        }

        fs.close();
    }

    return true;
}


} // namespace Testing

