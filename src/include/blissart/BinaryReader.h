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


#ifndef __BLISSART_BINARYREADER_H__
#define __BLISSART_BINARYREADER_H__


#include <istream>
#include <types.h>


// Assure that floats and doubles have the right sizes.
typedef int BinaryReaderDummy1[(sizeof(double) == sizeof(uint64_t)) * 2 - 1];
typedef int BinaryReaderDummy2[(sizeof(float) == sizeof(uint32_t)) * 2 - 1];


/**
 * Flips a 16-bit value.
 * @param  x            a pointer to a 16-bit value
 */
#define flip16(x) \
    *((uint16_t *)(x)) = ((*((uint16_t *)(x)) & 0xFF00) >> 8) | \
                         ((*((uint16_t *)(x)) & 0x00FF) << 8)


/**
 * Flips a 32-bit value.
 * @param  x            a pointer to a 32-bit value
 */
#define flip32(x) \
    *((uint32_t *)(x)) = ((*((uint32_t *)(x)) & 0xFF000000) >> 24) | \
                         ((*((uint32_t *)(x)) & 0x00FF0000) >> 8) | \
                         ((*((uint32_t *)(x)) & 0x0000FF00) << 8) | \
                         ((*((uint32_t *)(x)) & 0x000000FF) << 24)


/**
 * Flips a 64-bit value.
 * @param  x            a pointer to a 64-bit value
 */
#define flip64(x) \
    *((uint64_t *)(x)) = ((*((uint64_t *)(x)) & 0xFF00000000000000ULL) >> 56) | \
                         ((*((uint64_t *)(x)) & 0x00FF000000000000ULL) >> 40) | \
                         ((*((uint64_t *)(x)) & 0x0000FF0000000000ULL) >> 24) | \
                         ((*((uint64_t *)(x)) & 0x000000FF00000000ULL) >> 8) | \
                         ((*((uint64_t *)(x)) & 0x00000000FF000000ULL) << 8) | \
                         ((*((uint64_t *)(x)) & 0x0000000000FF0000ULL) << 24) | \
                         ((*((uint64_t *)(x)) & 0x000000000000FF00ULL) << 40) | \
                         ((*((uint64_t *)(x)) & 0x00000000000000FFULL) << 56)


namespace blissart {


/**
 * \defgroup misc Miscellaneous header-only classes and functions
 * \addtogroup misc
 * @{
 */


/**
 * Implementation of a binary reader that works on stl input-streams
 * and provides (big|little)-endian ordering.
 */
class BinaryReader
{
public:
    typedef enum { BigEndian, LittleEndian } Architecture;


    /**
     * Constructs a BinaryReader from the given input-stream and architecture.
     * @param  is               the input-stream
     * @param  wantedArch       one of BinaryReader::Architecture
     */
    BinaryReader(std::istream& is, Architecture wantedArch) :
        _is(is)
    {
        uint16_t testVal = 0xAABB;
        uint8_t *ptr = (uint8_t *)&testVal;
        Architecture haveArch = (*ptr == 0xBB) ? LittleEndian : BigEndian;
        _flip = (haveArch != wantedArch);
    }


    inline BinaryReader& operator >> (uint16_t& val)
    {
        _is.read((char *)&val, sizeof(uint16_t));
        if (_flip)
            flip16(&val);
        return *this;
    }


    inline BinaryReader& operator >> (uint32_t& val)
    {
        _is.read((char *)&val, sizeof(uint32_t));
        if (_flip)
            flip32(&val);
        return *this;
    }


    inline BinaryReader& operator >> (uint64_t& val)
    {
        _is.read((char *)&val, sizeof(uint64_t));
        if (_flip)
            flip64(&val);
        return *this;
    }


    inline BinaryReader& operator >> (sint16_t& val)
    {
        _is.read((char *)&val, sizeof(sint16_t));
        if (_flip)
            flip16(&val);
        return *this;
    }


    inline BinaryReader& operator >> (sint32_t& val)
    {
        _is.read((char *)&val, sizeof(sint32_t));
        if (_flip)
            flip32(&val);
        return *this;
    }


    inline BinaryReader& operator >> (sint64_t& val)
    {
        _is.read((char *)&val, sizeof(sint64_t));
        if (_flip)
            flip64(&val);
        return *this;
    }


    inline BinaryReader& operator >> (float& val)
    {
        _is.read((char *)&val, sizeof(float));
        if (_flip)
            flip32(&val);
        return *this;
    }


    inline BinaryReader& operator >> (double& val)
    {
        _is.read((char *)&val, sizeof(double));
        if (_flip)
            flip64(&val);
        return *this;
    }


    /**
     * Reads the given number of floats and stores them in the given buffer.
     * @param  buf              a pointer to the buffer
     * @param  size             the number of floats to be read
     * @return                  the number of read floats
     */
    inline size_t readFloats(float *buf, size_t size)
    {
        const size_t start = std::streamoff(_is.tellg());
        _is.read((char *)buf, (unsigned int)(size * sizeof(float)));
        if (_flip) {
            for (size_t i = 0; i < size; i++) {
                flip32(buf + i);
            }
        }
        return (std::streamoff(_is.tellg()) - start) / sizeof(float);
    }


    /**
     * Reads the given number of doubles and stores them in the given buffer.
     * @param  buf              a pointer to the buffer
     * @param  size             the number of doubles to be read
     * @return                  the number of read doubles
     */
    inline size_t readDoubles(double *buf, size_t size)
    {
        const size_t start = std::streamoff(_is.tellg());
        _is.read((char *)buf, (unsigned int)(size * sizeof(double)));
        if (_flip) {
            for (size_t i = 0; i < size; i++)
                flip64(buf + i);
        }
        return (std::streamoff(_is.tellg()) - start) / sizeof(double);
    }


    /**
     * Returns if the end-of-file has been reached.
     */
    inline bool eof() const { return _is.eof(); }


    /**
     * Returns if the underlying stream has the fail- or error-bit set.
     */
    inline bool fail() const { return _is.fail(); }


private:
    std::istream& _is;
    bool          _flip = 0;
};


/**
 * @}
 */


} // namespace blissart


// Clean up.
#undef flip64
#undef flip32
#undef flip16


#endif // __BLISSART_BINARYREADER_H__
