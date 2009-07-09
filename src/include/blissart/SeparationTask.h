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


namespace blissart {


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
        NMF,
        NMD
    } SeparationMethod;


    /**
     * Identifies the type of data on which component separation is done.
     */
    typedef enum {
        MagnitudeSpectrum,
        MelSpectrum
    } DataKind;


    /**
     * Constructs a new instance of SeparationTask for the given parameters.
     * @param  sepMethod        the method to be used for component separation
     * @param  typeIdentifier   a string identifier for the type of the task
     * @param  fileName         the name of the input file
     * @param  dataKind         the type of data (spectrum or Mel spectrum)
     *                          which should be separated
     * @param  nrOfComponents   the desired number of components
     * @param  maxIterations    the maximum number of iterations
     * @param  epsilon          the desired precision
     * @param  isVolatile       store the resulting components iff true
     */
    SeparationTask(SeparationMethod sepMethod,
        const std::string &typeIdentifier,
        DataKind dataKind,
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
     * Sets the prefix to be used for the export of the separated components as
     * audio files. Once set to a non-empty string, the components will be
     * exported after separation.
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
     * Stores the phase-matrix and the separated components in the database.
     * Overrides FTTask method.
     */
    void storeComponents() const;


    /**
     * Exports the separated components to audio files.
     */
    void exportComponents() const;


private:
    // Forbid copy constructor and operator=.
    SeparationTask(const SeparationTask &other);
    SeparationTask& operator=(const SeparationTask &other);


    const SeparationMethod  _separationMethod;
    const DataKind          _dataKind;
    const unsigned int      _nrOfComponents;
    const unsigned int      _nrOfSpectra;

    std::vector<ClassificationObjectPtr> _initObjects;
    bool                    _constantInitializedComponentsSpectra;

    const unsigned int      _maxIterations;
    const double            _epsilon;

    const bool              _isVolatile;
    std::string             _exportPrefix;

    int                     _myUniqueID;
};


typedef Poco::AutoPtr<SeparationTask> SeparationTaskPtr;


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
