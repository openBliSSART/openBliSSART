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


#include <blissart/audio/AudioData.h>
#include <blissart/audio/audio.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/ProgressObserver.h>

#include <fftw3.h>

#include <cassert>
#include <cstring>
#include <cmath>

#include <Poco/Mutex.h>


using namespace std;
using namespace blissart::linalg;
using Poco::FastMutex;


namespace blissart {

namespace audio {


AudioData::AudioData(unsigned int nrOfSamples, unsigned int sampleRate) :
    _nrOfSamples(nrOfSamples),
    _sampleRate(sampleRate)
{
    assert(_nrOfSamples > 0);
}


AudioData::AudioData(const vector<double*> &data,
                     unsigned int nrOfSamples, unsigned int sampleRate,
                     bool useRawPointers) :
    _nrOfSamples(nrOfSamples),
    _sampleRate(sampleRate)
{
    assert(_nrOfSamples > 0);
    assert(!data.empty());

    for (unsigned int i = 0; i < data.size(); i++)
        addChannel(data.at(i), useRawPointers);
}


AudioData::~AudioData()
{
    removeAllChannels();
}


void AudioData::addChannel(const double* data, bool useRawPointer)
{
    if (useRawPointer) {
        // Simply store the pointer
        _data.push_back((double*)data);
    } else {
        // Copy the data and store a pointer to the copy
        assert(_nrOfSamples > 0);
        double* buf = new double[_nrOfSamples];
        memcpy(buf, data, _nrOfSamples * sizeof(double));
        _data.push_back(buf);
    }
}


void AudioData::makeMono()
{
    assert(_data.size() > 0);

    // Nothing to do in case of only one channel.
    if (_data.size() == 1)
        return;

    // Allocate memory for the new mono channel and compute the arithmetic mean
    // of all channels for all samples.
    double* const buf = new double[_nrOfSamples];
    for (unsigned int i = 0; i < _nrOfSamples; i++)
        buf[i] = 0.0;

    const double f = 1.0 / (double)_data.size();
    for (vector<double*>::iterator it = _data.begin(); it != _data.end(); it++) {
        for (unsigned int j = 0; j < _nrOfSamples; j++)
            buf[j] += f * (*it)[j];
    }

    // Replace the existing channels with the new one.
    removeAllChannels();
    addChannel(buf, true /*useRawPointer*/);
}


void AudioData::subRightFromLeftChannel()
{
    assert(_data.size() == 2);

    // Allocate memory for the new mono channel and fill it with the result
    // from subtracting the right from the left channel.
    double* buf = new double[_nrOfSamples];
    const double *left = _data.at(0);
    const double *right = _data.at(1);
    double absMax = 0.0;
    for (unsigned int i = 0; i < _nrOfSamples; ++i) {
        // Dividing by two is necessary in order to maintain [-1,1].
        buf[i] = (left[i] - right[i]) * 0.5;
        absMax = std::max<double>(abs(buf[i]), absMax);
    }

    // Normalize.
    if (absMax > 0.0) {
        double scale = 1.0 / absMax;
        for (unsigned int i = 0; i < _nrOfSamples; ++i)
            buf[i] *= scale;
    }

    // Replace the existing channels with the new one.
    removeAllChannels();
    addChannel(buf, true /*useRawPointer*/);
}


void AudioData::normalize()
{
    assert(_data.size() > 0);

    // Find the absolute maximum of all channels.
    double absMax = 0.0;
    for (vector<double*>::iterator it = _data.begin(); it != _data.end(); it++) {
        for (unsigned int j = 0; j < _nrOfSamples; j++) {
            double absVal = fabs((*it)[j]);
            if (absVal > absMax)
                absMax = absVal;
        }
        // There's no need to continue if absMax is already 1.
        if (absMax >= 1)
            return;
    }

    // Perform the normalization.
    if (absMax > 0) {
        double scale = 1.0 / absMax;
        for (vector<double*>::iterator it = _data.begin(); it != _data.end(); it++) {
            for (unsigned int j = 0; j < _nrOfSamples; j++)
                (*it)[j] *= scale;
        }
    }
}


void AudioData::preemphasize(double coeff)
{
    assert(_nrOfSamples > 0);
    for (vector<double*>::iterator it = _data.begin(); it != _data.end(); 
        ++it)
    {
        double* output = new double[_nrOfSamples];
        output[0] = (*it)[0];
        for (unsigned int j = 1; j < _nrOfSamples; ++j) {
            output[j] = (*it)[j] - coeff * (*it)[j - 1];
        }
        delete[] *it;
        *it = output;
    }
}


Matrix* AudioData::toMatrix() const
{
    assert(_data.size() > 0);

    Matrix* m = new Matrix((unsigned int)_data.size(), _nrOfSamples);
    for (unsigned int i = 0; i < _data.size(); i++) {
        const double* channel_ptr = _data.at(i);
        for (unsigned int j = 0; j < _nrOfSamples; j++)
            m->setAt(i, j, channel_ptr[j]);
    }

    return m;
}


void AudioData::removeAllChannels()
{
    vector<double*>::iterator it;
    for (it = _data.begin(); it != _data.end(); it++)
        delete[] (*it);
    _data.clear();
}


pair<Matrix*, Matrix*> AudioData::computeSpectrogram(WindowFunction windowFunc,
                                  int windowSize, double overlap,
                                  unsigned int channel, bool zeroPadding,
                                  bool removeDC) const
{
    // Check input parameters.
    if (overlap < 0 || overlap > 1)
        throw AudioException("Invalid overlap.");
    if (channel >= _data.size())
        throw AudioException("Invalid channel.");

    // The "jump" between frames has to be calculated with maximum precision
    // to avoid compatibility issues with systems like HTK where a frame rate
    // is specified directly.
    const int jump = (int) 
        ((1.0-overlap) * (double) windowSize * (double) _sampleRate / 1000.0);

    // Convert the windowSize parameter from msec to number of samples.
    windowSize = windowSize * _sampleRate / 1000;
    // Check the windowSize.
    if (windowSize < 0 || (unsigned int)windowSize > _nrOfSamples)
        throw AudioException("Invalid window size or window size too big.");

    const int nFrames = (_nrOfSamples - windowSize) / jump + 1;
    assert(nFrames >= 1);

    const int paddedWindowSize = zeroPadding ? 
        ceilPowerOfTwo(windowSize) :
        windowSize;

    // Allocate enough memory for in-place real-to-complex FFT
    // (n real values are transformed into (n+1)/2 complex values,
    //  which consist of two real values each).
    const int windowArraySize = paddedWindowSize + 2;
    double* windowData = new double[windowArraySize];

    fftw_complex* transformedData = (fftw_complex*) windowData;

    fftw_plan plan = fftw_plan_dft_r2c_1d(paddedWindowSize, windowData, transformedData, 0);

    Matrix* amplitudeMatrix = new Matrix(paddedWindowSize / 2 + 1, nFrames);
    Matrix* phaseMatrix = new Matrix(paddedWindowSize / 2 + 1, nFrames);

    double* dataPtr = _data.at(channel);

    for (int frame = 0; frame < nFrames; frame++) {
        if (removeDC) {
            // Compute window mean.
            double windowMean = 0.0;
            for (int i = 0; i < windowSize; i++) {
                windowMean += dataPtr[i];
            }
            windowMean /= (double)windowSize;
            // Window the data after mean has been subtracted.
            for (int i = 0; i < windowSize; i++) {
                windowData[i] = (dataPtr[i] - windowMean) 
                    * windowFunc(i, windowSize);
            }
        }
        else {
            for (int i = 0; i < windowSize; i++) {
                windowData[i] = dataPtr[i] * windowFunc(i, windowSize);
            }
        }
        // Perform zero-padding if desired.
        for (int i = windowSize; i < paddedWindowSize; i++) {
            windowData[i] = 0.0;
        }
        // Transform to frequency domain.
        fftw_execute_dft_r2c(plan, windowData, transformedData);
        // Split result into amplitude and phase.
        for (unsigned int i = 0; i < amplitudeMatrix->rows(); i++) {
            (*amplitudeMatrix)(i, frame) =
                sqrt(transformedData[i][0] * transformedData[i][0] +
                     transformedData[i][1] * transformedData[i][1]);
            (*phaseMatrix)(i, frame) =
                atan2(transformedData[i][1], transformedData[i][0]);
        }

        dataPtr += jump;
    }

    delete[] windowData;
    return pair<Matrix*, Matrix*>(amplitudeMatrix, phaseMatrix);
}


AudioData* AudioData::fromSpectrogram(const Matrix& amplitudeSpectrum,
                                      const Matrix& phaseSpectrum,
                                      WindowFunction windowFunc,
                                      int windowSize, double overlap,
                                      int sampleRate,
                                      ProgressObserver *observer)
{
    debug_assert(amplitudeSpectrum.cols() == phaseSpectrum.cols() &&
                 amplitudeSpectrum.rows() == phaseSpectrum.rows());

    // Check input parameters.
    if (overlap < 0 || overlap > 1)
        throw AudioException("Invalid overlap.");
    if (sampleRate < 0)
        throw AudioException("Invalid sample rate.");

    // See the remark in "computeSpectrogram".
    const int jump = (int) 
        ((1.0-overlap) * (double)windowSize * (double)sampleRate / 1000.0);

    // Convert the windowSize parameter from msec to number of samples.
    windowSize = windowSize * sampleRate / 1000;

    const int windowArraySize = amplitudeSpectrum.rows() + 2;
    fftw_complex* windowSpectrum = new fftw_complex[windowArraySize];
    double* windowData = (double*) windowSpectrum;

    const unsigned int nSamples = (amplitudeSpectrum.cols() - 1) * jump + windowSize;
    if (windowSize < 0 || (unsigned int)windowSize > nSamples) {
        throw AudioException("Invalid window size or window size too big.");
    }

    double* audioData = new double[nSamples];
    double* pAudioData = audioData;

    // Initialize audioData with 0's.
    for (unsigned int i = 0; i < nSamples; i++)
        audioData[i] = 0.0;

    // If data has been transformed with zero-padding, the actual dimension
    // of the transformation might be greater than the actual window size.
    unsigned int transformSize = (amplitudeSpectrum.rows() != windowSize / 2 + 1) ?
    (amplitudeSpectrum.rows() * 2 - 1) :
    windowSize;
    fftw_plan backtrafoPlan = fftw_plan_dft_c2r_1d(transformSize, windowSpectrum, windowData, 0);
    for (unsigned int frame = 0; frame < amplitudeSpectrum.cols(); frame++) {
        // Reconstruct complex number from amplitude + phase.
        for (unsigned int i = 0; i < amplitudeSpectrum.rows(); i++) {
            double x = amplitudeSpectrum(i, frame);
            double phi = phaseSpectrum(i, frame);
            windowSpectrum[i][0] = x * cos(phi);
            windowSpectrum[i][1] = x * sin(phi);
        }
        // Transform to time domain.
        fftw_execute_dft_c2r(backtrafoPlan, windowSpectrum, windowData);
        // Perform overlap-add of normalized window data.
        for (int i = 0; i < windowSize; i++)
            pAudioData[i] += (windowData[i] / transformSize) * windowFunc(i, windowSize);

        // Call the ProgressObserver every once in a while (if applicable).
        if (observer && frame % 250 == 0)
            observer->progressChanged((float)frame / (float)amplitudeSpectrum.rows());

        pAudioData += jump;
    }
    // Final call to the ProgressObserver (if applicable).
    if (observer)
        observer->progressChanged(1.0f);

    delete[] windowSpectrum;
    return new AudioData(vector<double*>(1, audioData), nSamples, sampleRate, true);
}


AudioData* AudioData::fromFile(const string& fileName, bool makeMono)
{
    ASSERT_AUDIO_INITIALIZED

    // Initialize the sample with a buffer size of 64kB.
    Sound_Sample* sample = Sound_NewSampleFromFile(fileName.c_str(), NULL, 65536);
    if (!sample)
        throw AudioException(Sound_GetError());

    // First stage decoding of the sample.
    if (!Sound_DecodeAll(sample) || !(sample->flags & SOUND_SAMPLEFLAG_EOF)) {
        const char* errorMsg = Sound_GetError();
        // Free the allocated sample
        Sound_FreeSample(sample);
        if (errorMsg) {
            throw AudioException(errorMsg);
        }
        else {
            throw AudioException("Could not decode audio sample!");
        }
    }

    // Determine important informations about the sample.
    bool isEightBit, isSigned, isLittleEndian;
    unsigned int nrOfSamplesPerChannel;
    if (!determineVitalSampleInfo(sample, &isEightBit, &isSigned,
                                  &isLittleEndian, &nrOfSamplesPerChannel)) {
        // Free the allocated sample.
        Sound_FreeSample(sample);
        // Forward the exception.
        throw AudioException("Unknown sample format!");
    }

    // Convert to native byte-order if neccessary.
    if (!isEightBit) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        if (!isLittleEndian) // big endian -> little endian
#else
        if (isLittleEndian) // little endian -> big endian
#endif
            switchByteOrder((unsigned short*)sample->buffer, sample->buffer_size >> 1);
    }

    // Convert the raw sample data into real values within [-1,1]. The result
    // is an array of channels, each being a double array.
    vector<double*> channels;
    rawToDouble(sample, &channels, isEightBit, isSigned, nrOfSamplesPerChannel, makeMono);

    // Make a backup of the sample rate before freeing Willy ;-)
    const unsigned int sampleRate = sample->actual.rate;
    // Free Willy!
    Sound_FreeSample(sample);

    // Construct resulting AudioData object and set the filename.
    AudioData *result = new AudioData(channels,
                        nrOfSamplesPerChannel,
                        sampleRate,
                        true /* useRawPointer */);
    result->_fileName = fileName;

    // Et volia.
    return result;
}


bool AudioData::determineVitalSampleInfo(const Sound_Sample* sample,
                                         bool* isEightBit, bool* isSigned,
                                         bool* isLittleEndian,
                                         unsigned int* nrOfSamples)
{
    *isEightBit = *isSigned = *isLittleEndian = false;
    switch (sample->actual.format) {
        case AUDIO_S8:
            *isSigned = true;
            // intentional fallthrough
        case AUDIO_U8:
            *isEightBit = true;
            *isLittleEndian = false;
            break;

        case AUDIO_S16LSB:
            *isLittleEndian = true;
            // intentional fallthrough
        case AUDIO_S16MSB:
            *isSigned = true;
            break;

        case AUDIO_U16LSB:
            *isLittleEndian = true;
            // intentional fallthrough
        case AUDIO_U16MSB:
            break;

        default:
            return false;
    }

    // Determine the number of samples (per channel)
    *nrOfSamples =
        (*isEightBit ? sample->buffer_size : (sample->buffer_size >> 1)) /
        sample->actual.channels;

    return true;
}


void AudioData::rawToDouble(const Sound_Sample* sample, vector<double*>* channels,
                            bool isEightBit, bool isSigned, unsigned int nrOfSamples,
                            bool makeMono)
{
    assert(channels->empty());

    // If makeMono has been specified as true but the # of available channels
    // is 1 then makeMono can be set to false at this point because no related
    // work has to be done
    if (makeMono && sample->actual.channels == 1)
        makeMono = false;

    // Determine the final # of channels
    const unsigned int finalNrOfChannels = makeMono ? 1 : sample->actual.channels;

    // Allocate memory for all samples and all channels
    double** const buf = new double*[finalNrOfChannels];
    for (unsigned int i = 0; i < finalNrOfChannels; i++)
        buf[i] = new double[nrOfSamples];

    // Although there's a lot of redundant code here, it's obviously faster
    // to perform the necessary if's only once, hence...
    if (isEightBit) {
        if (isSigned) { // -128 .. 127
            const char* buffer = (const char*)sample->buffer;
            if (makeMono) {
                // Mono
                const double f = 1.0 / sample->actual.channels;
                for (unsigned int i = 0; i < nrOfSamples; i++) {
                    buf[0][i] = 0.0;
                    for (unsigned int j = 0; j < sample->actual.channels; j++, buffer++)
                        buf[0][i] += f * (2.0 * (double)(*buffer) + 1.0) / 255.0;
                }
            } else {
                // All channels
                for (unsigned int i = 0; i < nrOfSamples; i++) {
                    for (unsigned int j = 0; j < sample->actual.channels; j++, buffer++)
                        buf[j][i] = (2.0 * (double)(*buffer) + 1.0) / 255.0;
                }
            }
        } else { // 0 .. 255
            const unsigned char* buffer = (const unsigned char*)sample->buffer;
            if (makeMono) {
                // Mono
                const double f = 1.0 / sample->actual.channels;
                for (unsigned int i = 0; i < nrOfSamples; i++) {
                    buf[0][i] = 0.0;
                    for (unsigned int j = 0; j < sample->actual.channels; j++, buffer++)
                        buf[0][i] += f * (2.0 * (double)(*buffer) - 255.0) / 255.0;
                }
            } else {
                // All channels
                for (unsigned int i = 0; i < nrOfSamples; i++) {
                    for (unsigned int j = 0; j < sample->actual.channels; j++, buffer++)
                        buf[j][i] = (2.0 * (double)(*buffer) - 255.0) / 255.0;
                }
            }
        }
    } else {
        if (isSigned) { // -32768 .. 32767
            const short* buffer = (const short*)sample->buffer;
            if (makeMono) {
                // Mono
                const double f = 1.0 / sample->actual.channels;
                for (unsigned int i = 0; i < nrOfSamples; i++) {
                    buf[0][i] = 0.0;
                    for (unsigned int j = 0; j < sample->actual.channels; j++, buffer++)
                        buf[0][i] += f * (2.0 * (double)(*buffer) + 1.0) / 65535.0;
                }
            } else {
                // All channels
                for (unsigned int i = 0; i < nrOfSamples; i++) {
                    for (unsigned int j = 0; j < sample->actual.channels; j++, buffer++)
                        buf[j][i] = (2.0 * (double)(*buffer) + 1.0) / 65535.0;
                }
            }
        } else { // 0 .. 65535
            const unsigned short* buffer = (const unsigned short*)sample->buffer;
            if (makeMono) {
                // Mono
                const double f = 1.0 / sample->actual.channels;
                for (unsigned int i = 0; i < nrOfSamples; i++) {
                    buf[0][i] = 0.0;
                    for (unsigned int j = 0; j < sample->actual.channels; j++, buffer++)
                        buf[0][i] += f * (2.0 * (double)(*buffer) - 65535.0) / 65535.0;
                }
            } else {
                // All channels
                for (unsigned int i = 0; i < nrOfSamples; i++) {
                    for (unsigned int j = 0; j < sample->actual.channels; j++, buffer++)
                        buf[j][i] = (2.0 * (double)(*buffer) - 65535.0) / 65535.0;
                }
            }
        }
    }

    // Add the double arrays to the channels vector
    for (unsigned int i = 0; i < finalNrOfChannels; i++)
        channels->push_back(buf[i]);
    // Delete only the array, not it's contents
    delete[] buf;
}


void AudioData::switchByteOrder(unsigned short* buffer, const unsigned int size)
{
    for (unsigned int i = 0; i < size; i++, buffer++)
        *buffer = ((*buffer & 0x00FF) << 8) | ((*buffer & 0xFF00) >> 8);
}


unsigned int AudioData::ceilPowerOfTwo(unsigned int number)
{
    float exp = log((float) number) / log(2.0f);
    return (unsigned int) pow(2.0f, (int) ceil(exp));
}


} // namespace audio

} // namespace blissart
