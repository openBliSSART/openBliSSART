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


#ifndef __BLISSART_SEPARATIONTASK_H__
#define __BLISSART_SEPARATIONTASK_H__


#include <blissart/BasicTask.h>
#include <blissart/FTTask.h>
#include <blissart/WindowFunctions.h>
#include <blissart/ClassificationObject.h>
#include <blissart/linalg/Matrix.h>
#include <Poco/SharedPtr.h>


namespace blissart {


/**
 * \addtogroup framework
 * @{
 */

/**
 * An abstract base class for tasks that perform component separation.
 */
class LibFramework_API SeparationTask : public FTTask
{
public:
    /**
     * Identifies the method to be used for component separation.
     */
    typedef enum {
        NMD
    } SeparationMethod;


    /**
     * Constructs a new instance of SeparationTask for the given parameters.
     * @param  sepMethod        the method to be used for component separation
     * @param  typeIdentifier   a string identifier for the type of the task
     * @param  fileName         the name of the input file
     *                          which should be separated
     * @param  nrOfComponents   the desired number of components
     * @param  nrOfSpectra      the desired number of spectra per component
     * @param  maxIterations    the maximum number of iterations
     * @param  epsilon          the desired precision
     * @param  isVolatile       store the resulting components iff true
     */
    SeparationTask(SeparationMethod sepMethod,
        const std::string &typeIdentifier,
        const std::string &fileName,
        unsigned int nrOfComponents, unsigned int nrOfSpectra,
        unsigned int maxIterations,
        double epsilon, bool isVolatile);


    /**
     * Destructs an instance of SeparationTask and frees all formerly allocated
     * memory.
     */
    virtual ~SeparationTask();


    /**
     * The task's main method. Should subclasses wish to overwrite this method,
     * they should call SeparationTask::runTask() at the beginning of their
     * implementation to assure existence of all matrices and the like.
     */
    virtual void runTask();


    /**
     * Returns the separation method.
     * @return                  separation method
     */
    inline SeparationMethod separationMethod() const;


    /**
     * Returns the number of components.
     * @return                  the number of components
     */
    inline unsigned int nrOfComponents() const;


    /**
     * Returns the number of spectra per component.
     * @return                  the number of spectra per component
     */
    inline unsigned int nrOfSpectra() const;


    /**
     * Returns the maximum number of iterations.
     * @return                  the maximum number of iterations
     */
    inline unsigned int maxIterations() const;


    /**
     * Returns the desired epsilon.
     * @return                  the desired epsilon
     */
    inline double epsilon() const;


    /**
     * Sets the generator function for initialization of the W and H matrices.
     * If W is initialized from objects, the function is ignored for W.
     */
    inline void setGeneratorFunction(linalg::Matrix::GeneratorFunction gf);


    /**
     * Controls whether separated components should be exported to audio files.
     */
    inline void setExportComponents(bool flag);


    /**
     * Controls whether the reconstructed spectrogram (product of spectra and 
     * gains) should be exported as an audio file.
     */
    inline void setExportSpectrogram(bool flag);


    /**
     * Controls whether the spectral matrices should be exported in binary
     * format.
     */
    inline void setExportSpectra(bool flag);


    /**
     * Controls whether the gains matrix should be exported in binary format.
     */
    inline void setExportGains(bool flag);


    /**
     * Sets the prefix to be used for the export of the separated components.
     * The filenames will be of the form <prefix>_<task_id>_<nr>.wav.
     */
    inline void setExportPrefix(const std::string& prefix);


    /**
     * Sets the objects to use for targeted initialization of the components'
     * spectral matrices and whether the initialized components' spectra should
     * remain constant.
     */
    void
    setInitializationObjects(const std::vector<ClassificationObjectPtr>& objects,
                             bool constant = false);


    /**
     * Returns the objects used for targeted initialization.
     */
    inline std::vector<ClassificationObjectPtr>& initializationObjects();


    /**
     * Returns whether the initialized components' spectra should remain
     * constant.
     */
    inline bool constantInitializedComponentSpectra() const;


    /**
     * Returns the number of targeted initialization objects.
     */
    inline unsigned int numInitializationObjects() const;


    /**
     * Returns the generator function used for matrix initialization.
     */
    inline linalg::Matrix::GeneratorFunction generatorFunction() const;


    /**
     * Returns a reference to the matrix whose columns contain the components'
     * magnitude spectra after separation.
     */
    virtual const linalg::Matrix& magnitudeSpectraMatrix(unsigned int index) const = 0;


    /**
     * Returns a reference to the matrix whose rows contain the components'
     * gains over time after separation.
     */
    virtual const linalg::Matrix& gainsMatrix() const = 0;


protected:
    /**
     * Performs initialization of the separation matrices, either randomly or
     * targeted according to the vectors in a set of ClassificationObjects.
     */
    virtual void initialize() = 0;


    /**
     * Performs the actual separation process.
     */
    virtual void performSeparation() = 0;


    /**
     * Sets the actual separation progress.
     * @param   progress        the current progress
     */
    void setSeparationProgress(float progress);


    /**
     * Sets the number of completed steps.
     * Overrides FTTask method.
     */
    void setCompletedSteps(unsigned int completedSteps);


    /**
     * Fills in some process parameters.
     * Overrides FTTask method.
     */
    virtual void setProcessParameters(ProcessPtr process) const;


    /**
     * Stores the phase-matrix and the separated components in the database.
     * Overrides FTTask method.
     */
    void storeComponents() const;


    /**
     * Exports the separated components to audio files.
     */
    void exportComponents() const;


    /**
     *
     */
    void exportSpectrogram() const;


    /**
     * Converts a magnitude spectrogram to a time signal, using the original
     * phase matrix, and save the result as a WAV file.
     */
    void spectrogramToAudioFile(
        Poco::SharedPtr<linalg::Matrix> magnitudeSpectrogram,
        const std::string& outputFile) const;


    /**
     * Exports the separation matrices (spectra and/or gains).
     */
    void exportMatrices() const;


    /**
     * Helper function for exportMatrices().
     */
    void exportMatrixHTK(const blissart::linalg::Matrix& m,
                         const std::string& filename) const;


    /**
     * Helper function to determine output file name for export.
     */
    std::string getExportPrefix() const;


private:
    // Forbid copy constructor and operator=.
    SeparationTask(const SeparationTask &other);
    SeparationTask& operator=(const SeparationTask &other);


    const SeparationMethod  _separationMethod;
    const unsigned int      _nrOfComponents;
    const unsigned int      _nrOfSpectra;

    std::vector<ClassificationObjectPtr> _initObjects;
    bool                    _constantInitializedComponentsSpectra;

    linalg::Matrix::GeneratorFunction _genFunc;

    const unsigned int      _maxIterations;
    const double            _epsilon;

    const bool              _isVolatile;
    bool                    _exportComponents;
    bool                    _exportSpectrogram;
    bool                    _exportSpectra;
    bool                    _exportGains;
    std::string             _exportPrefix;

    int                     _myUniqueID;
};


typedef Poco::AutoPtr<SeparationTask> SeparationTaskPtr;


/**
 * @}
 */


// Inlines


inline SeparationTask::SeparationMethod SeparationTask::separationMethod() const
{
    return _separationMethod;
}


inline unsigned int SeparationTask::nrOfComponents() const
{
    return _nrOfComponents;
}


inline unsigned int SeparationTask::nrOfSpectra() const
{
    return _nrOfSpectra;
}


inline unsigned int SeparationTask::maxIterations() const
{
    return _maxIterations;
}


inline double SeparationTask::epsilon() const
{
    return _epsilon;
}


inline void SeparationTask::setExportPrefix(const std::string& prefix)
{
    _exportPrefix = prefix;
}


inline void SeparationTask::setExportComponents(bool flag)
{
    _exportComponents = flag;
}


inline void SeparationTask::setExportSpectrogram(bool flag)
{
    _exportSpectrogram = flag;
}


inline void SeparationTask::setExportSpectra(bool flag)
{
    _exportSpectra = flag;
}


inline void SeparationTask::setExportGains(bool flag)
{
    _exportGains = flag;
}


inline void 
SeparationTask::setGeneratorFunction(linalg::Matrix::GeneratorFunction gf)
{
    _genFunc = gf;
}


inline linalg::Matrix::GeneratorFunction 
SeparationTask::generatorFunction() const
{
    return _genFunc;
}


inline
std::vector<ClassificationObjectPtr>& SeparationTask::initializationObjects()
{
    return _initObjects;
}


inline bool SeparationTask::constantInitializedComponentSpectra() const
{
    return _constantInitializedComponentsSpectra;
}


inline unsigned int SeparationTask::numInitializationObjects() const
{
    return (unsigned int)_initObjects.size();
}


} // namespace blissart


#endif // __BLISSART_SEPARATIONTASK_H__
