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


#ifndef __BLISSART_FTTASK_H__
#define __BLISSART_FTTASK_H__


#include <blissart/BasicTask.h>
#include <blissart/WindowFunctions.h>
#include <blissart/MatrixTransform.h>
#include <blissart/Process.h>
#include <common.h>
#include <cassert>
#include <vector>


namespace blissart {


// Forward declaration
namespace audio { class AudioData; }
namespace linalg { class Matrix; }


/**
 * A base class for tasks that operate on audio data in the frequency domain.
 */
class LibFramework_API FTTask : public BasicTask
{
public:
    /**
     * Constructs a new instance of FTTask for the given file.
     * @param  typeIdentifier   Type identifier, e.g. "FTTask".
     * @param  fileName         the name of the input file
     */
    FTTask(const std::string& typeIdentifier, const std::string &fileName);


    /**
     * Destructs an instance of FTTask and frees all formerly allocated
     * memory.
     */
    virtual ~FTTask();


    /**
     * The task's main method.
     */
    virtual void runTask();


    /**
     * Sets the window function.
     * @param   wf              a pointer to a WindowFunction
     */
    inline void setWindowFunction(WindowFunction wf);


    /**
     * Returns the window function.
     * @return                  a pointer to a WindowFunction
     */
    inline WindowFunction windowFunction() const;


    /**
     * Sets the window-size.
     * @param   windowSize      the window-size (ms)
     */
    inline void setWindowSize(unsigned int windowSize);


    /**
     * Returns the window-size.
     * @return                  the window-size (ms)
     */
    inline unsigned int windowSize() const;


    /**
     * Sets the windows' overlap.
     * @param   overlap         the windows's overlap
     */
    inline void setOverlap(double overlap);


    /**
     * Returns the windows' overlap.
     * @return                  the windows's overlap
     */
    inline double overlap() const;


    /**
     * Returns the sample rate of the audio source.
     */
    inline unsigned int sampleRate() const;


    /**
     * Returns the file name of the audio source.
     */
    inline const std::string& fileName() const;


    /**
     * Sets whether the mids shall be reduced by subtracting the right from the
     * left channel when converting stereo files to mono.
     * @param  flag            whether to perform mid-reduction or not
     */
    inline void setReduceMids(bool flag);


    /**
     * Returns whether the mids are reduced.
     */
    inline bool reduceMids() const;


    /**
     * Sets the preemphasis coefficient.
     * @param  coeff           a number in the range [0,1).
     *                         0 indicates no preemphasis.
     */
    inline void setPreemphasisCoefficient(double coeff);


    /**
     * Returns the preemphasis coefficient.
     */
    inline double preemphasisCoefficient() const;


    /**
     * Sets a flag controlling whether the DC component (i.e. the mean) should
     * be removed from each window. This modification is applied before
     * applying the window function.
     * @param   flag           whether the DC should be removed
     */
    inline void setRemoveDC(bool flag);


    /**
     * Returns the flag controlling whether the DC component (i.e. the mean)
     * should be removed from each window.
     */
    inline bool removeDC() const;


    /**
     * Sets a flag controlling whether spectra should be padded with zeros
     * such that the length of each spectrum is a power of 2.
     * This option is mainly for compatibility with applications that do not
     * use sophisticated FFT algorithms like we do :)
     */
    inline void setZeroPadding(bool flag);


    /**
     * Returns the flag controlling whether spectra should be padded with
     * zeros such that the length of each spectrum is a power of 2.
     */
    inline bool zeroPadding() const;


    /**
     * Returns a reference to the computed phase matrix.
     * @return                  a Matrix reference
     */
    inline const linalg::Matrix& phaseMatrix() const;


    /**
     * Deletes the phase matrix. Can be called by subclasses once the
     * phase matrix isn't needed anymore, hence freeing some memory.
     */
    void deletePhaseMatrix();


    /**
     * Replaces the amplitude matrix with the given matrix. Deletes the old
     * object and _assumes ownership_ of the given object.
     * @param  amplitudeMatrix  a pointer to a Matrix
     */
    void replaceAmplitudeMatrix(linalg::Matrix *amplitudeMatrix);


    /**
     * Deletes the amplitude matrix. Can be called by subclasses once the
     * amplitude matrix isn't needed anymore, hence freeing some memory.
     */
    void deleteAmplitudeMatrix();


    /**
     * Returns a reference to the computed amplitude matrix.
     * @return                  a Matrix reference
     */
    inline const linalg::Matrix& amplitudeMatrix() const;


    /**
     * Advises the FTTask to perform one or more additional transformations
     * on the magnitude matrix obtained by Fourier transformation.
     */
    inline void addTransformation(MatrixTransform *tf);


    /**
     * Returns a reference to the vector containing addtional matrix 
     * transformations.
     */
    inline const std::vector<MatrixTransform*>& transforms() const;


protected:
    /**
     * Sets the number of completed steps.
     * @param   completedSteps  the number of already completed steps
     */
    void setCompletedSteps(unsigned int completedSteps);


    /**
     * Returns the number of completed steps.
     * @return                  the number of already completed steps
     */
    inline unsigned int completedSteps() const;


    /**
     * Reads audio from an audio file and and reduces the # of channels to 1
     * if necessary. Performs mid-reduction if desired.
     */
    void readAudioFile();


    /**
     * Computes the amplitude- and phase-matrices.
     */
    void computeSpectrogram();


    /**
     * Performs transformations of the matrices, such as Mel filtering of the
     * magnitude spectra.
     */
    void doAdditionalTransformations();


    /**
     * Fills in the attributes of the given Process entity according to the
     * actual parameters of this FTTask.
     */
    virtual void setProcessParameters(ProcessPtr process) const;


    /**
     * Stores the amplitude- and phase-matrix in the database.
     */
    void storeComponents() const;


private:
    // Forbid copy constructor and operator=.
    FTTask(const FTTask &other);
    FTTask& operator=(const FTTask &other);

    const std::string       _fileName;
    WindowFunction          _windowFunction;
    unsigned int            _windowSize;
    double                  _overlap;
    bool                    _reduceMids;
    double                  _preemphasisCoeff;
    bool                    _removeDC;
    bool                    _zeroPadding;

    std::vector<MatrixTransform*> _transforms;

    audio::AudioData*       _audioData;
    unsigned int            _sampleRate;

    linalg::Matrix*         _amplitudeMatrix;
    linalg::Matrix*         _phaseMatrix;
};


typedef Poco::AutoPtr<FTTask> FTTaskPtr;


// Inlines


inline void FTTask::setWindowFunction(WindowFunction wf)
{
    _windowFunction = wf;
}


inline WindowFunction FTTask::windowFunction() const
{
    return _windowFunction;
}


inline void FTTask::setWindowSize(unsigned int windowSize)
{
    _windowSize = windowSize;
}


inline unsigned int FTTask::windowSize() const
{
    return _windowSize;
}


inline void FTTask::setOverlap(double overlap)
{
    _overlap = overlap;
}


inline double FTTask::overlap() const
{
    return _overlap;
}


inline const linalg::Matrix& FTTask::phaseMatrix() const
{
    return *_phaseMatrix;
}


inline const linalg::Matrix& FTTask::amplitudeMatrix() const
{
    return *_amplitudeMatrix;
}


inline unsigned int FTTask::sampleRate() const
{
    return _sampleRate;
}


inline const std::string& FTTask::fileName() const
{
    return _fileName;
}


inline void FTTask::setReduceMids(bool yesNo)
{
    _reduceMids = yesNo;
}


inline bool FTTask::reduceMids() const
{
    return _reduceMids;
}


inline void FTTask::setPreemphasisCoefficient(double coeff)
{
    assert(0.0 <= coeff && coeff < 1);
    _preemphasisCoeff = coeff;
}


inline double FTTask::preemphasisCoefficient() const
{
    return _preemphasisCoeff;
}


inline void FTTask::setZeroPadding(bool flag)
{
    _zeroPadding = flag;
}


inline bool FTTask::zeroPadding() const
{
    return _zeroPadding;
}


inline void FTTask::setRemoveDC(bool flag)
{
    _removeDC = flag;
}


inline bool FTTask::removeDC() const
{
    return _removeDC;
}


inline void FTTask::addTransformation(MatrixTransform *tf)
{
    _transforms.push_back(tf);
}


inline const std::vector<MatrixTransform*>& FTTask::transforms() const
{
    return _transforms;
}


} // namespace blissart


#endif // __BLISSART_FTTASK_H__
