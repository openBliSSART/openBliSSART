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

#ifndef __BLISSART_AUDIO_AUDIODATA_H__
#define __BLISSART_AUDIO_AUDIODATA_H__


#include <blissart/WindowFunctions.h>
#include <common.h>
#include <vector>
#include <string>


// Since the authors of SDL_sound prefer typedef'ed anonymous structs we cannot
// have forward declarations but instead have to explicitly include SDL_sound.h.
// SDL_sound in turns includes SDL.h which has the bad habit of defining a
// "main" macro which can cause lots of confusion.
// Thus we undef this macro if it hasn't been defined beforehand. If, however,
// it has been defined before, we assume that the user knows what he's doing
// and leave it untouched.
#ifdef main
#  include <SDL/SDL_sound.h>
#else
#  include <SDL/SDL_sound.h>
#  undef main
#endif


namespace blissart {


// Forward declaration
class ProgressObserver;
namespace linalg { class Matrix; }


namespace audio {


/**
 * \addtogroup audio
 * @{
 */

/**
 * Encapsulation of audio data in terms of a number of channels, a number of
 * samples per channel and an overall sample rate.  The samples are represented
 * as real values within [-1,1].  Also offers methods to compute the spectogram
 * and create AudioData objects from spectograms as well as from sound files.
 */
class LibAudio_API AudioData
{
public:
    /**
     * Some tasks can be very time consuming and thus provide the possibility to
     * call a given callback function so that in turn the callee can provide
     * feedback to the user.
     */
    typedef void(*FeedbackCallback)(int frame, int maxFrames, void *user);


    /**
     * Constructs an AudioData object with 0 channels.
     * @param   nrOfSamples         the # of samples
     * @param   sampleRate          the sample rate
     */
    AudioData(unsigned int nrOfSamples, unsigned int sampleRate);


    /**
     * Constructs an AudioData object from the given data.
     * @param   data                a vector of pointers to samples for an arbitrary
     *                              number of channels
     * @param   nrOfSamples         the number of samples per channel
     * @param   sampleRate          the sample rate
     * @param   useRawPointers      make direct use of the given pointers instead
     *                              of copying all values from the given arrays
     *                              iff set to true (defaults to false). In this case
     *                              AudioData also assumes full ownership of the
     *                              pointers.
     */
    AudioData(const std::vector<double*> &data,
              unsigned int nrOfSamples, unsigned int sampleRate,
              bool useRawPointers = false);


    /**
     * Destructs an instance of AudioData and frees all allocated memory.
     */
    virtual ~AudioData();


    /**
     * Get the sample rate of this object's underlying audio data.
     * @return                      an unsigned int representing the sample rate
     */
    inline unsigned int sampleRate() const { return _sampleRate; }


    /**
     * Set the sample rate of this object to the given value.
     * Note that this doesn't perform any conversion of the underlying data from
     * an existing sample rate to the new value.
     * @param   sr                  the desired sample rate
     */
    inline void setSampleRate(unsigned int sr) { _sampleRate = sr; }


    /**
     * Get the number of samples.
     * @return                      the # of samples
     */
    inline unsigned int nrOfSamples() const { return _nrOfSamples; }


    /**
     * Get the number of channels.
     * @return                      the # of channels
     */
    inline unsigned int nrOfChannels() const { return (unsigned int)_data.size(); }


    /**
     * Returns the name of the source file.
     * @return                      the filename
     */
    inline std::string fileName() const { return _fileName; }


    /**
     * Adds a channel from the given data.
     * @param   data                a pointer to the samples
     * @param   useRawPointer       make direct use of the given pointer instead
     *                              of copying all values from the given array
     *                              iff set to true (defaults to false). In this case
     *                              AudioData also assumes full ownership of the
     *                              given pointer.
     */
    void addChannel(const double* data, bool useRawPointer = false);


    /**
     * Retrieve a pointer to the data of the given channel.
     * @param   channelNr           the # of the channel
     * @return                      a pointer to the data or NULL iff there's
     *                              no such channel
     */
    inline const double* getChannel(unsigned int channelNr) const
            { return (channelNr < _data.size() ? _data.at(channelNr) : NULL); }


    /**
     * Combines all available channels into one mono channel by
     * calculating their arithmetic means.
     */
    void makeMono();


    /**
     * Subtracts the right from the left channel, therefore reducing the mid-band
     * components of the samples. The AudioData object will be mono afterwards.
     */
    void subRightFromLeftChannel();


    /**
     * Normalizes all available channels.
     */
    void normalize();


    /**
     * Preemphasize (i.e. high-pass filter) all available channels, 
     * using the given preemphasis coefficient.
     */
    void preemphasize(double coeff);


    /**
     * Creates a matrix composed of all the encapsulated audio data.
     * @return                      a newly created Matrix
     */
    linalg::Matrix* toMatrix() const;


    /**
     * Computes the spectogram of this object's audio data.
     * Note that this method is NOT THREADSAFE because the fftw-library isn't.
     * @param   windowFunc          a window function
     * @param   windowSize          the size of each window (ms)
     * @param   overlap             a real value within [0..1] specifying the
     *                              amount that all windows overlap
     * @param   channel             the # of the channel of which the spectogram
     *                              should be computed
     * @param   zeroPadding         whether to apply zero padding to the frames
     *                              to achieve a power-of-two size for FFT
     *                              (note that the FFT algorithm can handle 
     *                              arbitrary sizes)
     * @param   removeDC            whether to subtract the mean from frames
     *                              (e.g. for compatibility with HTK)
     * @return                      a pair of Matrix objects of which the first
     *                              one represents the amplitudes and the second
     *                              one the phases.
     *                              It is up to the caller of this function to
     *                              delete these objects.
     */
    std::pair<linalg::Matrix*, linalg::Matrix*>
    computeSpectrogram(WindowFunction windowFunc, int windowSize, double overlap,
                       unsigned int channel, bool zeroPadding = false, 
                       bool removeDC = false) const;


    /**
     * Creates an AudioData object from the given spectrogram.
     * Note that this method is NOT THREADSAFE because the fftw-library isn't.
     * @param   amplitudeMatrix     the matrix containing the amplitudes
     * @param   phaseMatrix         the matrix containing the phases
     * @param   windowFunc          the window function to use for overlapping
     *                              time-domain data
     * @param   windowSize          the size of a window in the time domain (ms)
     * @param   overlap             the overlap of windows in the time domain
     * @param   sampleRate          the sample rate
     * @param   observer            a pointer to a ProgressObserver (or 0)
     * @return                      a new AudioData object
     */
    static AudioData* fromSpectrogram(const linalg::Matrix& amplitudeMatrix,
                                      const linalg::Matrix& phaseMatrix,
                                      WindowFunction windowFunc,
                                      int windowSize, double overlap,
                                      int sampleRate,
                                      ProgressObserver *observer = 0);


    /**
     * Creates and AudioData object from the give sound file.
     * @param   fileName            the name of the input sound file
     * @param   makeMono            whether to convert the sample into mono
     * @return                      an AudioData object
     * @throw                       an exception in case of an error
     */
    static AudioData* fromFile(const std::string& fileName, bool makeMono = false);


    /**
     * Returns the smallest power of two which is greater than or equal to 
     * the given integer.
     */
    static unsigned int ceilPowerOfTwo(unsigned int number);


private:
    // Forbid copy constructor and operator=.
    AudioData(const AudioData&);
    AudioData& operator=(const AudioData&);


    /**
     * Removes all channels and frees all associated memory.
     */
    void removeAllChannels();


    /**
     * Determine if the sound sample is 8- or 16-bit, the sample points are
     * signed or unsigned values and in case of 16-bit whether the data is
     * stored in little or big endian format.
     * Also compute the number of samples per channel.
     * @param   sample          a pointer to the sound sample
     * @param   isEightBit      a pointer to bool
     * @param   isSigned        a pointer to bool
     * @param   isLittleEndian  a pointer to bool
     * @param   nrOfSamples     a pointer to unsigned int
     * @return                  false iff the sample format is unknown
     */
    static bool determineVitalSampleInfo(const Sound_Sample* sample,
                                         bool* isEightBit, bool* isSigned,
                                         bool* isLittleEndian,
                                         unsigned int* nrOfSamples);


    /**
     * Converts the data of the given sound sample into double values.  Each
     * channel is processed. The final double values will be within [-1,1].
     * @param   sample          a pointer to the sound sample
     * @param   channels        a pointer to a vector that will contain the
     *                          channels after completion of this function
     * @param   isEightBit      indicates whether the sample's data are 8- or 16-bit
     * @param   isSigned        indicates whether the sample's data is signed or unsigned
     * @param   nrOfSamples     the # of samples per channel
     * @param   makeMono        whether to make the sample mono, i.e. store only
     *                          arithmetic mean of the sample points
     */
    static void rawToDouble(const Sound_Sample* sample, std::vector<double*>* channels,
                            bool isEightBit, bool isSigned, unsigned int nrOfSamples,
                            bool makeMono);


    /**
     * Switches 16-bit byte-order within the given buffer.
     * @param   buffer          a pointer to the buffer
     * @param   size            the size of the buffer
     */
    static void switchByteOrder(unsigned short* buffer, const unsigned int size);


// According to Microsoft the warning 4251 can be disabled if it's related to
// namespace std.
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable:4251)
    std::vector<double*> _data;
#  pragma warning(pop)
#else
    std::vector<double*> _data;
#endif
    const unsigned int   _nrOfSamples;
    unsigned int         _sampleRate;
    std::string          _fileName;
};


/**
 * @}
 */


} // namespace audio

} // namespace blissart


#endif // __BLISSART_AUDIO_AUDIODATA_H__
