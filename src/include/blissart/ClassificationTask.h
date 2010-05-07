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


#ifndef __CLASSIFICATIONTASK_H__
#define __CLASSIFICATIONTASK_H__


#include <blissart/BasicTask.h>
#include <blissart/FeatureExtractor.h>
#include <blissart/DataSet.h>
#include <blissart/WindowFunctions.h>
#include <blissart/SeparationTask.h>


namespace blissart {


// Forward declaration
namespace linalg { class Matrix; }


/**
 * \addtogroup framework
 * @{
 */

/**
 * A task that classifies the audio components contained in a spectral and
 * gains matrix, as output by a non-negative matrix factorization, and
 * writes them to wave files.
 */
class LibFramework_API ClassificationTask : public BasicTask
{
public:
    /**
     * Constructs a new instance of ClassificationTask for the given
     * parameters.
     * @param  responseID       the id of the Response that should be used
     *                          for classification
     * @param  phaseMatrix      a pointer to a phase matrix
     * @param  componentSpectrograms a vector whose elements contain, for 
     *                               each separated component, the pointer to 
     *                               the spectrogram matrix (for NMF, this 
     *                               matrix has only 1 column)
     * @param  gainsMatrix      a pointer to a matrix whose rows contain the
     *                          corresponding separate gains
     * @param  sampleRate       the sample rate
     * @param  windowSize       the window-size that was used during FT (ms)
     * @param  overlap          the windows' overlap that was used during FT
     * @param  fileName         A file name that gives a pattern for the ouput
     *                          files. If it is "filename.wav", output files
     *                          are named subsequently as
     *                          "filename_<ClassLabel1>.wav", ...
     */
    ClassificationTask(int responseID,
                       const linalg::Matrix& phaseMatrix,
                       const std::vector<linalg::Matrix*>& componentSpectrograms,
                       const linalg::Matrix& gainsMatrix,
                       int sampleRate, int windowSize, double overlap,
                       const std::string& fileName);


    /**
     * Constructs a new instance of ClassificationTask for the given
     * SeparationTask. All data will be initialized from the SeparationTask
     * once this task is being run.
     * @param  responseID       the id of the Response that should be used
     *                          for classification
     * @param  sepTask          a pointer to the SeparationTask
     */
    ClassificationTask(int responseID, const SeparationTaskPtr& sepTask);

    /**
     * Destructs an instance of ClassificationTask and frees all formerly
     * allocated memory.
     */
    virtual ~ClassificationTask();


    /**
     * The task's main method.
     */
    virtual void runTask();


    inline SeparationTaskPtr separationTask() const;


    void presetClassLabel(int componentIndex, int classLabel);


protected:
    /**
     * Exports the IFT'ed classes' spectra as WAVE files.
     */
    void exportAsWav();


    /**
     * Creates the neccessary DataSet.
     */
    void createDataSet();


    /**
     * Returns a reference to the created DataSet.
     * @return                  a reference to the created DataSet
     */
    inline const DataSet& dataSet() const;


private:
    // Forbid copy constructor and operator=.
    ClassificationTask(const ClassificationTask &other);
    ClassificationTask& operator=(const ClassificationTask &other);


    FeatureExtractor               _extractor;
    const int                      _responseID;
    SeparationTaskPtr              _sepTask;  // Do NOT change to normal pointer
    const linalg::Matrix*          _phaseMatrix;
    std::vector<linalg::Matrix*>   _componentSpectrograms;
    const linalg::Matrix*          _gainsMatrix;
    unsigned int                   _sampleRate;
    int                            _windowSize;
    double                         _overlap;
    std::string                    _fileName;
    DataSet                        _dataSet;
    std::map<int, int>             _presetClassLabels;
    std::map<int, linalg::Matrix*> _spectraMap;
};


typedef Poco::AutoPtr<ClassificationTask> ClassificationTaskPtr;


/**
 * @}
 */


// Inlines


inline SeparationTaskPtr ClassificationTask::separationTask() const
{
    return _sepTask;
}


inline const DataSet& ClassificationTask::dataSet() const
{
    return _dataSet;
}


} // namespace blissart


#endif
