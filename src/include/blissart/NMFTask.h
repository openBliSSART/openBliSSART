//
// $Id: NMFTask.h 855 2009-06-09 16:15:50Z alex $
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


#ifndef __BLISSART_NMFTASK_H__
#define __BLISSART_NMFTASK_H__


#include <blissart/SeparationTask.h>
#include <blissart/ProgressObserver.h>


namespace blissart {


// Forward declarations
namespace nmf { class Factorizer; }


/**
 * A task that performs component separation using non-negative matrix 
 * deconvolution (NMF).
 */
class LibFramework_API NMFTask : public SeparationTask,
                                 public ProgressObserver
{
public:
    /**
     * The NMF algorithm to use. Possible values:
     * - GradientDescent: Gradient descent minimizing squared error,
     * - MUDistance: Multiplicative update minimizing Euclidean distance,
     * - MUDivergence: Multiplicative update minimizing KL divergence.
     */
    typedef enum {
        GradientDescent,
        MUDistance,
        MUDivergence
    } Algorithm;


    /**
     * Constructs a new instance of NMFTask for the given parameters.
     * @param  fileName         the name of the input file
     * @param  dataKind         the type of data (spectrum or Mel spectrum) 
     *                          which should be separated
     * @param  algorithm        the algorithm to be used
     * @param  nrOfComponents   the desired number of components
     * @param  maxIterations    the maximum number of iterations
     * @param  epsilon          the desired precision
     * @param  isVolatile       store the resulting components iff true
     */
    NMFTask(const std::string &fileName, DataKind dataKind,
            Algorithm algorithm, int nrOfComponents,
            int maxIterations, double epsilon, bool isVolatile);


    /**
     * Destructs an instance of NMFTask and frees all formerly allocated
     * memory.
     */
    virtual ~NMFTask();


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


protected:
    /**
     * Performs the actual separation process.
     */
    virtual void performSeparation();


private:
    // Forbid copy constructor and operator=.
    NMFTask(const NMFTask &other);
    NMFTask& operator=(const NMFTask &other);


    /**
     * Implementation of ProgressObserver's progressChanged() method.
     */
    virtual void progressChanged(float);


    nmf::Factorizer*  _factorizer;
    Algorithm         _algorithm;
};


typedef Poco::AutoPtr<NMFTask> NMFTaskPtr;


} // namespace blissart


#endif // __BLISSART_NMFTASK_H__
