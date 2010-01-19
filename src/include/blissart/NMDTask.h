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


#ifndef __BLISSART_NMDTASK_H__
#define __BLISSART_NMDTASK_H__


#include <blissart/SeparationTask.h>
#include <blissart/ProgressObserver.h>
#include <blissart/nmf/Deconvolver.h>


namespace blissart {


// Forward declarations
//namespace nmf { class Deconvolver; }


/**
 * A task that performs component separation using non-negative matrix 
 * deconvolution (NMD).
 */
class LibFramework_API NMDTask : public SeparationTask,
                                 public ProgressObserver
{
public:
    /**
     * Constructs a new instance of NMDTask for the given parameters.
     * @param  fileName         the name of the input file
     * @param  dataKind         the type of data (spectrum or Mel spectrum) 
     *                          which should be separated
     * @param  costFunction     the cost function which should be minimized
     * @param  nrOfComponents   the desired number of components
     * @param  nrOfSpectra      the desired number of spectra per component
     * @param  maxIterations    the maximum number of iterations
     * @param  epsilon          the desired precision
     * @param  isVolatile       store the resulting components iff true
     */
    NMDTask(const std::string &fileName, DataKind dataKind,
            nmf::Deconvolver::NMFCostFunction costFunction, 
            int nrOfComponents, int nrOfSpectra,
            int maxIterations, double epsilon, bool isVolatile);


    /**
     * Destructs an instance of NMDTask and frees all formerly allocated
     * memory.
     */
    virtual ~NMDTask();


    /**
     * Implementation of SeparationTask interface.
     */
    virtual void initialize();


    /**
     * Returns a reference to the matrix whose columns contain the components'
     * magnitude spectra after separation.
     */
    virtual const linalg::Matrix& magnitudeSpectraMatrix(unsigned int index) const;


    /**
     * Returns a reference to the matrix whose rows contain the components'
     * gains over time after separation.
     */
    virtual const linalg::Matrix& gainsMatrix() const;


    inline void setSparsity(double lambda);


    inline double getSparsity() const;

    
    inline void setNormalizeSpectra(bool flag);


    inline bool getNormalizeSpectra() const;


protected:
    /**
     * Performs the actual separation process.
     */
    virtual void performSeparation();


private:
    // Forbid copy constructor and operator=.
    NMDTask(const NMDTask &other);
    NMDTask& operator=(const NMDTask &other);


    /**
     * Implementation of ProgressObserver's progressChanged() method.
     */
    virtual void progressChanged(float);


    nmf::Deconvolver*                 _deconvolver;
    nmf::Deconvolver::NMFCostFunction _cf;
    double                            _sparsity;
    bool                              _normalizeSpectra;

};


void NMDTask::setNormalizeSpectra(bool flag)
{
    _normalizeSpectra = flag;
}


bool NMDTask::getNormalizeSpectra() const
{
    return _normalizeSpectra;
}


void NMDTask::setSparsity(double lambda)
{
    _sparsity = lambda;
}


double NMDTask::getSparsity() const
{
    return _sparsity;
}


typedef Poco::AutoPtr<NMDTask> NMDTaskPtr;


} // namespace blissart


#endif // __BLISSART_NMDTASK_H__
