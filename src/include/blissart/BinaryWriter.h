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


#ifndef __BLISSART_BINARYWRITER_H__
#define __BLISSART_BINARYWRITER_H__


#include <ostream>


#if defined(_WIN32) || defined(_MSC_VER)
# define uint8_t  unsigned __int8
# define uint16_t unsigned __int16
# define uint32_t unsigned __int32
# define uint64_t unsigned __int64
# define sint8_t  __int8
# define sint16_t __int16
# define sint32_t __int32
# define sint64_t __int64
#else
# define sint8_t  int8_t
# define sint16_t int16_t
# define sint32_t int32_t
# define sint64_t int64_t
#endif


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
 * \addtogroup misc
 * @{
 */

/**
 * Implementation of a binary writer that works on stl output-streams
 * and provides (big|little)-endian ordering.
 */
class BinaryWriter
{
public:
    typedef enum { BigEndian, LittleEndian } Architecture;


    /**
     * Constructs a BinaryWriter from the given output-stream and architecture.
     * @param  os               the output-stream
     * @param  wantedArch       one of BinaryReader::Architecture
     */
    BinaryWriter(std::ostream& os, Architecture wantedArch) :
        _os(os)
    {
        uint16_t testVal = 0xAABB;
        uint8_t *ptr = (uint8_t *)&testVal;
        Architecture haveArch = (*ptr == 0xBB) ? LittleEndian : BigEndian;
        _flip = (haveArch != wantedArch);
    }


    inline BinaryWriter& operator << (const uint16_t& val)
    {
        if (_flip) {
            uint16_t tmp = val;
            flip16(&tmp);
            _os.write((const char *)&tmp, sizeof(uint16_t));
        } else
            _os.write((const char *)&val, sizeof(uint16_t));
        return *this;
    }


    inline BinaryWriter& operator << (const uint32_t& val)
    {
        if (_flip) {
            uint32_t tmp = val;
            flip32(&tmp);
            _os.write((const char *)&tmp, sizeof(uint32_t));
        } else
            _os.write((const char *)&val, sizeof(uint32_t));
        return *this;
    }


    inline BinaryWriter& operator << (const uint64_t& val)
    {
        if (_flip) {
            uint64_t tmp = val;
            flip64(&tmp);
            _os.write((const char *)&tmp, sizeof(uint64_t));
        } else
            _os.write((const char *)&val, sizeof(uint64_t));
        return *this;
    }


    inline BinaryWriter& operator << (const sint16_t& val)
    {
        if (_flip) {
            sint16_t tmp = val;
            flip16(&tmp);
            _os.write((const char *)&tmp, sizeof(sint16_t));
        } else
            _os.write((const char *)&val, sizeof(sint16_t));
        return *this;
    }


    inline BinaryWriter& operator << (const sint32_t& val)
    {
        if (_flip) {
            sint32_t tmp = val;
            flip32(&tmp);
            _os.write((const char *)&tmp, sizeof(sint32_t));
        } else
            _os.write((const char *)&val, sizeof(sint32_t));
        return *this;
    }


    inline BinaryWriter& operator << (const sint64_t& val)
    {
        if (_flip) {
            sint64_t tmp = val;
            flip64(&tmp);
            _os.write((const char *)&tmp, sizeof(sint64_t));
        } else
            _os.write((const char *)&val, sizeof(sint64_t));
        return *this;
    }


    inline BinaryWriter& operator << (const float& val)
    {
        if (_flip) {
            float tmp = val;
            flip32(&tmp);
            _os.write((const char *)&tmp, sizeof(float));
        } else
            _os.write((const char *)&val, sizeof(float));
        return *this;
    }


    inline BinaryWriter& operator << (const double& val)
    {
        if (_flip) {
            double tmp = val;
            flip64(&tmp);
            _os.write((const char *)&tmp, sizeof(double));
        } else
            _os.write((const char *)&val, sizeof(double));
        return *this;
    }


    /**
     * Writes the given number of floats from the given buffer.
     * @param  buf              a pointer to the buffer
     * @param  size             the number of floats to be written
     * @return                  the number of written floats
     */
    inline size_t writeFloats(const float *buf, size_t size)
    {
        if (_flip) {
            for (size_t i = 0; i < size; i++) {
                if (_os.fail())
                    return i;
                float tmp = buf[i];
                flip32(&tmp);
                _os.write((const char *)&tmp, sizeof(float));
            }
        } else {
            size_t start = std::streamoff(_os.tellp());
            _os.write((const char *)buf, (unsigned int)(size * sizeof(float)));
            size = (std::streamoff(_os.tellp()) - start) / sizeof(float);
        }

        return size;
    }


    /**
     * Writes the given number of doubles from the given buffer.
     * @param  buf              a pointer to the buffer
     * @param  size             the number of doubles to be written
     * @return                  the number of written doubles
     */
    inline size_t writeDoubles(const double *buf, size_t size)
    {
        if (_flip) {
            for (size_t i = 0; i < size; i++) {
                if (_os.fail())
                    return i;
                double tmp = buf[i];
                flip64(&tmp);
                _os.write((const char *)&tmp, sizeof(double));
            }
        } else {
            size_t start = std::streamoff(_os.tellp());
            _os.write((const char *)buf, (unsigned int)(size * sizeof(double)));
            size = (std::streamoff(_os.tellp()) - start) / sizeof(double);
        }
        return size;
    }


    /**
     * Returns if the underlying stream has the fail- or error-bit set.
     */
    inline bool fail() const { return _os.fail(); }


private:
    std::ostream& _os;
    bool          _flip;
};


/**
 * @}
 */


} // namespace blissart


// Clean up.
#undef flip64
#undef flip32
#undef flip16


#endif // __BLISSART_BINARYWRITER_H__
