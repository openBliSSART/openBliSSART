//
// $Id: wave.h 855 2009-06-09 16:15:50Z alex $
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

#ifndef __WAVE_H__
#define __WAVE_H__


#include <climits>


namespace blissart {

namespace audio {

namespace wave {


// The following conditional is vital for the correct
// length of the headers and chunks!
#if ((UCHAR_MAX != 0xFF) || \
     (USHRT_MAX != 0xFFFF) || \
     (UINT_MAX != 0xFFFFFFFF) || \
     (ULONG_MAX != 0xFFFFFFFFUL))
# error This machine does NOT support either \
        8-bit chars, 16-bit shorts, 32-bit ints or 32-bit longs!
#endif


// Byte-alignment is neccessary for the following structures
#ifdef _MSC_VER
#  pragma pack(push)
#endif
#pragma pack(1)


/**
 * Description of a RIFF header
 */
struct RiffHeader {
    char id[4];
    long lChunkSize;
    char type[4];
};


/**
 * Description of the PCM wave format chunk
 */
struct FormatChunk {
    char id[4];
    long lChunkSize;
    short wFormatTag;
    unsigned short wChannels;
    unsigned long dwSamplesPerSec;
    unsigned long dwAvgBytesPerSec;
    unsigned short wBlockAlign;
    unsigned short wBitsPerSample;
};


/**
 * Description of the PCM wave data chunk
 */
struct DataChunk {
    char id[4];
    long lChunkSize;
    unsigned char *data;
};


// No more byte-aligning
#ifdef _MSC_VER
#  pragma pack(pop)
#endif


} // namepace wave

} // namespace audio

} // namespace blissart


#endif
