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

#include <blissart/audio/WaveEncoder.h>
#include <blissart/audio/AudioData.h>
#include <blissart/linalg/Matrix.h>
#include <cassert>
#include <cstdio>
#include <SDL/SDL_sound.h>
#include "wave.h"


using namespace std;
using namespace blissart::linalg;


namespace blissart {

namespace audio {


#if SDL_BYTEORDER == SDL_BIG_ENDIAN
# define ISEP_MSB_TO_LSB_16(a)   ((((a) & 0x00FF) << 8) | \
                             (((a) & 0xFF00) >> 8))
# define ISEP_MSB_TO_LSB_32(a)   ((((a) & 0x000000FF) << 24) | \
                             (((a) & 0x0000FF00) << 8) | \
                             (((a) & 0x00FF0000) >> 8) | \
                             (((a) & 0xFF000000) >> 24))
#else
# define ISEP_MSB_TO_LSB_16(a)   (a)
# define ISEP_MSB_TO_LSB_32(a)   (a)
#endif


#if defined(_WIN32) || defined(_MSC_VER)
# pragma warning(push)
# pragma warning(disable : 4996)
#endif
bool WaveEncoder::saveAsWav(const double* data, unsigned int nrOfSamples,
                            unsigned int sampleRate, unsigned int nrOfChannels,
                            const string& fileName)
{
    bool result;
    FILE *fpOut;
    wave::RiffHeader rh;
    wave::FormatChunk fc;
    wave::DataChunk dc;

    // FormatChunk
    strncpy(fc.id, "fmt ", 4);
    fc.lChunkSize = ISEP_MSB_TO_LSB_32(sizeof(fc) - 8);
    fc.wFormatTag = ISEP_MSB_TO_LSB_16(1);               // no compression
    fc.wChannels = ISEP_MSB_TO_LSB_16(nrOfChannels);
    fc.dwSamplesPerSec = ISEP_MSB_TO_LSB_32(sampleRate);
    fc.wBitsPerSample = ISEP_MSB_TO_LSB_16(16);          // 16 Bit
    // XXX: The following line will cause problems when dealing with audio data
    // of less than 8 bit. However, we won't use such data *sigh*
    fc.wBlockAlign = ISEP_MSB_TO_LSB_16(fc.wChannels * fc.wBitsPerSample / 8);
    fc.dwAvgBytesPerSec = ISEP_MSB_TO_LSB_32(fc.dwSamplesPerSec * fc.wBlockAlign);

    // DataChunk
    strncpy(dc.id, "data", 4);
    try {
        short* buffer = new short[nrOfSamples];
        for (unsigned int i = 0; i < nrOfSamples; i++) {
            // Assure interval [-1,1]
            double val = data[i];
            if (val < -1)
                val = -1;
            else if (val > 1) 
                val = 1;
            // Map to [-32768,32767]
            buffer[i] = ISEP_MSB_TO_LSB_16((short)((val + 1) / 2.0 * 65535.0 - 32768.0));
        }
        dc.data = (unsigned char *)buffer;
    } catch (bad_alloc&) {
        return false;
    }

    // RiffHeader
    strncpy(rh.id, "RIFF", 4);
    strncpy(rh.type, "WAVE", 4);
    rh.lChunkSize = ISEP_MSB_TO_LSB_32(4 + fc.lChunkSize + 8 + dc.lChunkSize + 8);

    result = false;
    do {
        if (!(fpOut = fopen(fileName.c_str(), "wb")))
            break;

        // RiffHeader
        if (fwrite(&rh, sizeof(wave::RiffHeader), 1, fpOut) != 1)
            break;

        // FormatChunk
        if (fwrite(&fc, sizeof(wave::FormatChunk), 1, fpOut) != 1)
            break;
        if (sizeof(wave::FormatChunk) & 1)
            fputc(0, fpOut); // padding

        // DataChunk
        unsigned int rawSize = nrOfSamples << 1; // 16-bit
        dc.lChunkSize = ISEP_MSB_TO_LSB_32(rawSize);
        if (fwrite(&dc, sizeof(wave::DataChunk) - sizeof(const unsigned char *), 1, fpOut) != 1)
            break;
        if (fwrite(dc.data, 1, rawSize, fpOut) != rawSize)
            break;
        if (rawSize & 1)
            fputc(0, fpOut); // padding

        result = true;
    } while (0);
    
    // Clean up
    delete[] dc.data;
    if (fpOut)
        fclose(fpOut);
    
    return result;
}
#if defined(_WIN32) || defined(_MSC_VER)
# pragma warning(pop)
#endif


bool WaveEncoder::saveAsWav(const Matrix& sampleMatrix, unsigned int sampleRate,
                            const string& fileName)
{
    const unsigned int totalNrOfSamples = sampleMatrix.rows() * sampleMatrix.cols();
    double* data;

    try {
        data = new double[totalNrOfSamples];
    } catch (exception&) {
        return false;
    }

    double* tmp = data;
    for (unsigned int j = 0; j < sampleMatrix.cols(); j++)
        for (unsigned int i = 0; i < sampleMatrix.rows(); i++)
            *tmp++ = sampleMatrix.at(i,j);

    bool result = saveAsWav(data, 
                            totalNrOfSamples,
                            sampleRate,
                            sampleMatrix.rows(),
                            fileName);
    delete[] data;
    return result;
}


bool WaveEncoder::saveAsWav(const AudioData& audioData, const string& fileName)
{
    assert(audioData.nrOfChannels() > 0);

    // Determine the total # of samples and allocate enough memory to hold
    // all of them in a temporary buffer
    const unsigned int totalNrOfSamples = 
                        audioData.nrOfChannels() * audioData.nrOfSamples();
    double* buf = new double[totalNrOfSamples];
    // Copy and arrange the channels' data
    for (unsigned int i = 0; i < audioData.nrOfChannels(); i++) {
        double* buf_ptr = buf + i;
        const double* channel_ptr = audioData.getChannel(i);
        for (unsigned int j = 0; j < audioData.nrOfSamples(); j++) {
            *buf_ptr = *channel_ptr;
            buf_ptr += audioData.nrOfChannels();
            channel_ptr++;
        }
    }
    // Save
    bool result = saveAsWav(buf,
                            totalNrOfSamples,
                            audioData.sampleRate(),
                            audioData.nrOfChannels(),
                            fileName);
    // Free the temporary buffer and return the result
    delete[] buf;
    return result;
}


} // namespace audio

} // namespace blissart
