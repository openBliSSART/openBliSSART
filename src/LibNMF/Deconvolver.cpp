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


#include <blissart/nmf/Deconvolver.h>
#include <blissart/nmf/randomGenerator.h>
#include <blissart/linalg/generators/generators.h>
#include <blissart/ProgressObserver.h>
#include <blissart/linalg/RowVector.h>
#include <blissart/linalg/ColVector.h>

#include <stdexcept>
#include <sstream>
#include <cmath>
#include <vector>

#include <config.h>
#ifdef HAVE_CUDA
# include <blissart/nmf/DeconvolverKernels.h>
#endif


// for debugging ...
#include <iostream>
using namespace std;




//#define NMD_PT

using namespace blissart::linalg;


namespace blissart {


namespace nmf {


// Trivial constant to avoid division by zero in multiplicative updates.
#define DIVISOR_FLOOR 1e-9


const char* Deconvolver::costFunctionName(Deconvolver::NMDCostFunction cf)
{
    if (cf == Deconvolver::EuclideanDistance) 
        return "Squared Euclidean distance";
    if (cf == Deconvolver::KLDivergence)
        return "Extended KL divergence";
    if (cf == Deconvolver::ISDivergence)
        return "Itakura-Saito divergence";
    if (cf == Deconvolver::NormalizedEuclideanDistance)
        return "Squared ED (normalized basis)";
    if (cf == Deconvolver::BetaDivergence)
        return "Beta divergence";
    // should not occur ...
    return "Unknown";
}


Deconvolver::Deconvolver(const Matrix &v, unsigned int r, unsigned int t,
                         Matrix::GeneratorFunction wGenerator,
                         Matrix::GeneratorFunction hGenerator) :
    _alg(Deconvolver::Auto),
    _v(v),
    _approx(v.rows(), v.cols()), //, generators::zero),
    _oldApprox(0),
    _w(new Matrix*[t]),
    _wConstant(false),
    _wColConstant(new bool[r]),
    _t(t),
    _h(r, v.cols(), hGenerator),
    _s(r, v.cols(), generators::zero),      // zero --> no sparsity
    _c(r, v.cols(), generators::zero),      // zero --> no continuity
    _numSteps(0),
    _absoluteError(-1),
    _relativeError(-1),
    _vFrob(_v.frobeniusNorm()),
    _notificationDelay(25)
{
    if (t > v.cols()) {
        std::ostringstream errStr;
        errStr << "Invalid number of spectra: " << t
               << ": Matrix has only " << v.cols() << " columns!";
        throw std::runtime_error(errStr.str());
    }
    for (unsigned int l = 0; l < _t; ++l) {
        _w[l] = new Matrix(v.rows(), r);
    }
    for (unsigned int c = 0; c < r; ++c) {
        _wColConstant[c] = false;
    }
    generateW(wGenerator);
}


Deconvolver::~Deconvolver()
{
    for (unsigned int l = 0; l < _t; ++l) {
        delete _w[l];
    }
    delete[] _w;
    delete[] _wColConstant;
    if (_oldApprox) {
        delete _oldApprox;
        _oldApprox = 0;
    }
}


void Deconvolver::generateW(Matrix::GeneratorFunction generator)
{
    for (unsigned int l = 0; l < _t; l++) {
        for (unsigned int i = 0; i < _w[l]->rows(); i++) {
            for (unsigned int j = 0; j < _w[l]->cols(); j++) {
                _w[l]->at(i,j) = generator(i, j);
            }
        }
    }
}


void Deconvolver::generateH(Matrix::GeneratorFunction generator)
{
    for (unsigned int i = 0; i < _h.rows(); i++) {
        for (unsigned int j = 0; j < _h.cols(); j++) {
            _h(i,j) = generator(i, j);
        }
    }
}


void Deconvolver::setW(unsigned int i, const Matrix& w)
{
    assert(w.cols() == _w[i]->cols() && w.rows() == _w[i]->rows());
    delete _w[i];
    _w[i] = new Matrix(w);
}


void Deconvolver::setH(const Matrix& h)
{
    assert(h.cols() == _h.cols() && h.rows() == _h.rows());
    _h = h;
}


void Deconvolver::decompose(Deconvolver::NMDCostFunction cf,
                            unsigned int maxSteps, double eps,
                            bool sparse, bool continuous,
                            ProgressObserver *observer)
{
    // Select an optimal algorithm according to the given parameters.
    if (cf == EuclideanDistance) {
        if (_t == 1 && 
            (!isOvercomplete() || _alg == NMFEDIncomplete) &&
            !sparse && !continuous) 
        {
            factorizeNMFEDIncomplete(maxSteps, eps, observer);
        }
        else {
            factorizeNMDBeta(maxSteps, eps, 2, sparse, continuous, observer);
        }
    }
    else if (cf == KLDivergence) {
        factorizeNMDBeta(maxSteps, eps, 1, sparse, continuous, observer);
    }
    else if (cf == ISDivergence) {
        factorizeNMDBeta(maxSteps, eps, 0, sparse, continuous, observer);
    }
    else if (cf == NormalizedEuclideanDistance) {
        if (_t > 1) {
            throw std::runtime_error("NMD with normalized basis not implemented");
        }
        if (continuous) {
            throw std::runtime_error("Continuous NMF with normalized basis not implemented");
        }
        factorizeNMFEDSparseNorm(maxSteps, eps, observer);
    }
    else {
        throw std::runtime_error("Invalid cost function");
    }

    // Ensure the ProgressObserver sees that we have finished.
    if (observer)
        observer->progressChanged(1.0f);

    // Delete value of old approximation (used in convergence check), 
    // if it exists.
    if (_oldApprox) {
        delete _oldApprox;
        _oldApprox = 0;
    }
}


double 
Deconvolver::getCfValue(Deconvolver::NMDCostFunction cf, double beta) const
{
    // TODO : sparsity, continuity
    double res = 0.0;
    if (cf == Deconvolver::KLDivergence || 
       (cf == Deconvolver::BetaDivergence && beta == 1.0)) 
    {
        for (unsigned int j = 0; j < _v.cols(); ++j) {
            for (unsigned int i = 0; i < _v.rows(); ++i) {
                res += _v(i, j) * log(_v(i, j) / _approx(i, j)) 
                     - (_v(i, j) - _approx(i, j));
            }
        }
    }
    else if (cf == Deconvolver::EuclideanDistance ||
            (cf == Deconvolver::BetaDivergence && beta == 2.0))
    {
        // less operations than beta div. below...
        for (unsigned int j = 0; j < _v.cols(); ++j) {
            for (unsigned int i = 0; i < _v.rows(); ++i) {
                double tmp = _v(i, j) - _approx(i, j);
                res += tmp * tmp;
            }
        }
    }
    else if (cf == Deconvolver::ISDivergence || 
            (cf == Deconvolver::BetaDivergence && beta == 0.0))
    {
        for (unsigned int j = 0; j < _v.cols(); ++j) {
            for (unsigned int i = 0; i < _v.rows(); ++i) {
                double tmp = _v(i, j) / _approx(i, j);
                res += tmp - log(tmp);
            }
        }
        // summarize the -1 in element-wise cost function
        res -= _v.cols() * _v.rows(); 
    }
    else if (cf == Deconvolver::BetaDivergence) {
        double fac = 1.0 / (beta * (beta - 1.0));
        for (unsigned int j = 0; j < _v.cols(); ++j) {
            for (unsigned int i = 0; i < _v.rows(); ++i) {
                res += std::pow(_v(i, j), beta)
                     + (beta - 1.0) * std::pow(_approx(i, j), beta)
                     - beta * _v(i, j) * std::pow(_approx(i, j), beta - 1.0);
            }
        }
        res *= fac;
    }
    else if (cf == Deconvolver::NormalizedEuclideanDistance) {
        // FIXME: Implement this!
        throw std::runtime_error("Not implemented");
    }
    return res;
}


#ifdef HAVE_CUDA


void Deconvolver::factorizeNMDBeta(unsigned int maxSteps, double eps, 
                                   double beta, bool sparse, bool continuous,
                                   ProgressObserver *observer)
{
    GPUMatrix*  approxInv  = 0;
    // used for row / col sums
    GPUMatrix*  unityRcols = 0;
    GPUMatrix*  unityRrows = 0;
    GPUMatrix*  wpColSums  = 0;
    GPUMatrix*  hRowSums   = 0;
    GPUMatrix*  vLambdaInv = 0; // better name than vOverApprox
    // Holds a single H update. This might be made obsolete by using a 
    // specialized kernel in the future.
    GPUMatrix   hUpdate(_h.rows(), _h.cols());
    // Accumulates H updates.
    GPUMatrix   hUpdateAcc(_h.rows(), _h.cols());
    GPUMatrix   hUpdateNum(_h.rows(), _h.cols());
    GPUMatrix   hUpdateDenom(_h.rows(), _h.cols());
    
    GPUMatrix   wUpdateNum(_v.rows(), _h.rows());
    GPUMatrix   wUpdateDenom(_v.rows(), _h.rows());

    // GPU versions of V, W and H; transfer data to GPU
    GPUMatrix   vgpu(_v);
    GPUMatrix   hgpu(_h);
    GPUMatrix** wgpu = new GPUMatrix*[_t];
    for (unsigned int t = 0; t < _t; ++t) {
        wgpu[t] = new GPUMatrix(*_w[t]);
    }
    
    // GPU version of Lambda (Approximation)
    GPUMatrix   approxgpu(_v.rows(), _v.cols());
    
    if (beta == 2) {
        // for ED, exploit equalities by redirecting these pointers
        vLambdaInv = &vgpu;
        approxInv = &approxgpu;
    }
    else {
        if (beta != 1) {
            approxInv = new GPUMatrix(_v.rows(), _v.cols());
        }
        else {
            wpColSums = new GPUMatrix(1, _h.rows()); // 1xR
            hRowSums = new GPUMatrix(_h.rows(), 1);  // Rx1
            // TODO constructor like this for GPUMatrix?
            Matrix tmp1R(1, _h.rows(), linalg::generators::unity);
            unityRcols = new GPUMatrix(tmp1R);
            Matrix tmpR1(_h.rows(), 1, linalg::generators::unity);
            unityRrows = new GPUMatrix(tmpR1);
        }
        vLambdaInv = new GPUMatrix(_v.rows(), _v.cols());
    }

    //
    // Main iteration loop
    //

    /*cout << "W before iteration: " << endl << *_w[0] << endl;
    cout << "H before iteration: " << endl << _h << endl;    */

    _numSteps = 0;
    while (1) {
        computeApprox(wgpu, hgpu, &approxgpu);
        /*Matrix tmp(_v.rows(), _v.cols());
        approxgpu.getMatrix(&tmp);*/
        //cout << "Iteration #" << _numSteps << ": Approx =" << endl << tmp << endl;
        //break;

        if (_numSteps >= maxSteps)
            break;

        if (!_wConstant) {
            GPUMatrix* wpH = 0;
            for (unsigned int p = 0; p < _t; ++p) {
                // General Beta div. alg. (for ED, no computation needed)
                if (beta == 1) {
                    vgpu.elementWiseDiv(approxgpu, vLambdaInv);
                    // we explicitly compute row sums instead of using 
                    // approxInv which would be an all-one matrix
                    // compute row sum including columns 1 to N - p
                    hgpu.multWithMatrix(*unityRrows, hRowSums,
                        false, false,
                        _h.rows(), _h.cols() - p, 1,
                        0, 0, 0, 0, 0, 0);
                    //hgpu.multWithMatrix(*unityRrows, hRowSums);
                    hRowSums->floor(DIVISOR_FLOOR);
                    /*Matrix tmp4(_h.rows(), 1);
                    hRowSums->getMatrix(&tmp4);
                    cout << "H row sums: " << endl << tmp4 << endl;*/
                }
                else if (beta != 2) {
                    // TODO: KL as special case
                    // XXX: this might be done in a single kernel in the future...
                    approxgpu.elementWisePow(beta - 2, approxInv);
                    approxInv->elementWiseMult(vgpu, vLambdaInv);
                    // vOverApprox now contains Approx^{Beta - 2} .* V
                    approxInv->elementWiseMult(approxgpu, approxInv);
                    // approxInv now contains Approx^{Beta - 1};
                }
                /*approxInv->getMatrix(&tmp);
                cout << "Inverse of Approx = " << endl << tmp << endl;*/
                // W Update, Numerator
                vLambdaInv->multWithMatrix(hgpu, &wUpdateNum,
                    false, true,
                    _v.rows(), _v.cols() - p, _h.rows(),
                    0, p, 0, 0, 0, 0);

                // W Update, Denominator (for KL this is a all-one matrix)
                // for ED (beta = 2) the original approximation is used
                if (beta != 1) {
                    approxInv->multWithMatrix(hgpu, &wUpdateDenom,
                        false, true,
                        _v.rows(), _v.cols() - p, _h.rows(),
                        0, p, 0, 0, 0, 0);
                    wUpdateDenom.floor(DIVISOR_FLOOR);
                }

                // TODO move allocation out of loop!
                if (_t > 1) {
                    wpH = new GPUMatrix(_v.rows(), _v.cols());
                    // Difference-based calculation of new approximation
                    multWithShifted(*wgpu[p], hgpu, wpH, p);
                    approxgpu.sub(*wpH);
                }

                // W multiplicative update
                if (beta != 1) {
                    wgpu[p]->elementWiseMult(wUpdateNum,   wgpu[p]);
                    wgpu[p]->elementWiseDiv (wUpdateDenom, wgpu[p]);
                }
                else {
                    gpu::apply_KLWUpdate(wgpu[p]->dataPtr(), wUpdateNum.dataPtr(), 
                        hRowSums->dataPtr(), 
                        wgpu[p]->dataPtr(), 
                        wgpu[p]->rows(), wgpu[p]->cols());
                    /*Matrix tmp3(wgpu[p]->rows(), wgpu[p]->cols());
                    wgpu[p]->getMatrix(&tmp3);
                    cout << "W[" << p << "] after update: " << endl;
                    cout << tmp3 << endl;*/
                }
                
                // Difference-based calculation of new approximation
                if (_t > 1) {
                    multWithShifted(*wgpu[p], hgpu, wpH, p);
                    approxgpu.add(*wpH);
                    delete wpH;
                    approxgpu.floor(DIVISOR_FLOOR);
                }
            }
        }

        // For T > 1, approximation has been calculated above.
        // For T = 1, this is more efficient for T = 1.
        if (_t == 1) {
            computeApprox(wgpu, hgpu, &approxgpu);
        }

        // Now the approximation is up-to-date in any case.

        // see above
        
        if (beta == 1) {
            vgpu.elementWiseDiv(approxgpu, vLambdaInv);
            // we explicitly compute row sums instead of using 
            // approxInv which would be an all-one matrix
        }
        else if (beta != 2) {
            // XXX: this might be done in a single kernel in the future...
            approxgpu.elementWisePow(beta - 2, approxInv);
            approxInv->elementWiseMult(vgpu, vLambdaInv);
            // vOverApprox now contains Approx^{Beta - 2} .* V
            approxInv->elementWiseMult(approxgpu, approxInv);
            // approxInv now contains Approx^{Beta - 1};
        }

        // H Update
        hUpdateAcc.zero();
        
        // Compute H update for each W[p] and average later
        for (unsigned int p = 0; p < _t; ++p) {
            // TODO col sum stuff for KL

            // the last p columns will be zero but not computed
            hUpdateNum.zero();
            hUpdateDenom.zero();
            
            // Numerator
            wgpu[p]->multWithMatrix(*vLambdaInv, &hUpdateNum,
                // transpose W[p]
                true, false, 
                // target dimension: R x (N-p)
                wgpu[p]->cols(), wgpu[p]->rows(), vgpu.cols() - p,
                0, 0, 0, p, 0, 0);
            
            // Denominator
            if (beta != 1) {
                wgpu[p]->multWithMatrix(*approxInv, &hUpdateDenom,
                    // transpose W[p]
                    true, false, 
                    // target dimension: R x (N-p)
                    wgpu[p]->cols(), wgpu[p]->rows(), vgpu.cols() - p,
                    0, 0, 0, p, 0, 0);
                hUpdateDenom.floor(DIVISOR_FLOOR);
                hUpdateNum.elementWiseDiv(hUpdateDenom, &hUpdate);
            }
            else {
                // compute W[p] col sums
                unityRcols->multWithMatrix(*wgpu[p], wpColSums);
                wpColSums->floor(DIVISOR_FLOOR);
                /*Matrix tmp2(1, wgpu[p]->cols());
                wpColSums->getMatrix(&tmp2);
                cout << "Col sums: " << endl << tmp2 << endl;*/
                gpu::compute_KLHUpdate(hUpdateNum.dataPtr(), wpColSums->dataPtr(),
                    hUpdate.dataPtr(), hUpdate.rows(), hUpdate.cols());
            }

            /*Matrix tmp2(_h.rows(), _h.cols());
            hUpdateNum.getMatrix(&tmp2);
            cout << "H update matrix at p = " << p << ": num = " << endl << tmp2 << endl;
            hUpdateDenom.getMatrix(&tmp2);
            cout << "H update matrix at p = " << p << ": denom = " << endl << tmp2 << endl;*/
            hUpdateAcc.add(hUpdate);
        }

        const double alpha = 1.0f / (double) _t;
        hUpdateAcc.scale(alpha, 0, hgpu.cols() - _t);
        /*Matrix tmp3(_h.rows(), _h.cols());
        hUpdateAcc.getMatrix(&tmp3);*/
        //cout << "H update matrix: " << endl << tmp3 << endl;
        for (unsigned int j = hgpu.cols() - _t + 1; j < hgpu.cols(); ++j) {
            // we need to convert to const double for CUBLAS routine ...
            const double myalpha = 1.0f / (double) (hgpu.cols() - j);
            hUpdateAcc.scale(myalpha, j, j);
            //cout << "scale column " << j << " by " << myalpha << endl;
        }
        hgpu.elementWiseMult(hUpdateAcc, &hgpu);

        /*Matrix tmp3(_h.rows(), _h.cols());
        hgpu.getMatrix(&tmp3);
        cout << "H after update: " << endl << tmp3 << endl;*/
        
        nextItStep(observer, maxSteps);
    }

    // synchronize W and H to host
    hgpu.getMatrix(&_h);
    for (unsigned int t = 0; t < _t; ++t) {
        wgpu[t]->getMatrix(_w[t]);
    }
    
    // re-compute and synchronize approx
    computeApprox(wgpu, hgpu, &approxgpu);
    approxgpu.getMatrix(&_approx);

    /*cout << "Final approx: " << endl << _approx << endl;
    cout << "W after iteration: " << endl << *_w[0] << endl;
    cout << "H after iteration: " << endl << _h << endl;    */
    
    // delete GPU resources
    if (vLambdaInv && vLambdaInv != &vgpu)
        delete vLambdaInv;
    if (approxInv && approxInv != &approxgpu)
        delete approxInv;
    if (unityRcols)
        delete unityRcols;
    if (unityRrows)
        delete unityRrows;
    if (wpColSums)
        delete wpColSums;
    if (hRowSums)
        delete hRowSums;
    for (unsigned int t = 0; t < _t; ++t) {
        delete wgpu[t];
    }
    delete[] wgpu;
}        
        

void Deconvolver::factorizeNMFEDIncomplete(unsigned int maxSteps, double eps,
                                           ProgressObserver *observer)
{
    assert(_t == 1);

    // transfer V, W and H to GPU
    GPUMatrix vgpu( _v   );
    GPUMatrix hgpu( _h   );
    GPUMatrix wgpu(*_w[0]);

    GPUMatrix hUpdateNum  (_h.rows(), _h.cols());
    GPUMatrix hUpdateDenom(_h.rows(), _h.cols());
    GPUMatrix wUpdateNum  (_w[0]->rows(), _w[0]->cols());
    GPUMatrix wUpdateDenom(_w[0]->rows(), _w[0]->cols());

    GPUMatrix hhT(_h.rows(), _h.rows());
    GPUMatrix wTw(_h.rows(), _h.rows());

    _numSteps = 0;
    while (_numSteps < maxSteps) {
        // W Update
        vgpu.multWithTransposedMatrix(hgpu, &wUpdateNum);
        hgpu.multWithTransposedMatrix(hgpu, &hhT);
        wgpu.multWithMatrix(hhT, &wUpdateDenom);
        wUpdateDenom.floor(DIVISOR_FLOOR);
        wgpu.elementWiseMult(wUpdateNum,   &wgpu);
        wgpu.elementWiseDiv (wUpdateDenom, &wgpu);

        // H Update
        wgpu.transposedMultWithMatrix(vgpu, &hUpdateNum);
        wgpu.transposedMultWithMatrix(wgpu, &wTw);
        wTw.multWithMatrix(hgpu, &hUpdateDenom);
        hUpdateDenom.floor(DIVISOR_FLOOR);
        hgpu.elementWiseMult(hUpdateNum,   &hgpu);
        hgpu.elementWiseDiv (hUpdateDenom, &hgpu);

        nextItStep(observer, maxSteps);
    }
    
    // Sychronize GPU matrices to host.
    wgpu.getMatrix( _w[0]);
    hgpu.getMatrix(&_h   );

    // Update value of approximation (only once...)
    computeApprox();
}


#else


void Deconvolver::factorizeNMDBeta(unsigned int maxSteps, double eps, 
                                   double beta, bool sparse, bool continuous,
                                   ProgressObserver *observer)
{
    // TODO: remove oldH (not needed)
    // TODO: semi-supervised NMF with partial computation of W update matrix
    Matrix* approxInv = 0;
    Matrix* vOverApprox = 0; // "V Over Approx" from NMD-KL
    RowVector* wpColSums = 0;
    Matrix hUpdate(_h.rows(), _h.cols());
    Matrix hUpdateNum(_h.rows(), _h.cols());
    Matrix hUpdateDenom(_h.rows(), _h.cols());
    Matrix wUpdateNum(_v.rows(), _h.rows());
    Matrix wUpdateDenom(_v.rows(), _h.rows());

    // for sparsity / continuity
    ColVector *csplus = 0, *csminus  = 0;
    ColVector *ctplus = 0, *ctminus1 = 0, *ctminus2 = 0;
    Matrix    *oldH   = 0;
    if (sparse) {
        csplus  = new ColVector(_h.rows());
        csminus = new ColVector(_h.rows());
    }
    if (continuous) {
        ctplus   = new ColVector(_h.rows());
        ctminus1 = new ColVector(_h.rows());
        ctminus2 = new ColVector(_h.rows());
        oldH     = new Matrix   (_h.rows(), _h.cols());
    }

    if (beta == 2) {
        // for ED, exploit equalities by redirecting these pointers
        vOverApprox = (Matrix*) &_v;
        approxInv = &_approx;
    }
    else {
        vOverApprox = new Matrix(_v.rows(), _v.cols());
        // KL uses row sums instead of approxInv, which would be an all-1 
        // matrix
        if (beta != 1) {
            approxInv = new Matrix(_v.rows(), _v.cols());
        }
        else {
            wpColSums = new RowVector(_h.rows());
        }
    }

    /*cout << "W before iteration: " << endl << *_w[0] << endl;
    cout << "H before iteration: " << endl << _h << endl;    */

    _numSteps = 0;
    while (1) {
        computeApprox();
        
        //cout << "Entering iteration " << _numSteps << ". Approx = " << endl << _approx << endl;

        if (_numSteps >= maxSteps)
            break;

        // for NMF (T=1) we could also overwrite _approx ... but makes not much sense since we need 1 buffer matrix anyway...

        if (checkConvergence(eps, false))
            break;

        if (!_wConstant) {
            Matrix* wpH = 0;
            for (unsigned int p = 0; p < _t; ++p) {
                // KL divergence
                if (beta == 1) {
                    _v.apply(Matrix::div, _approx, vOverApprox);
                    // we explicitly compute row sums instead of using 
                    // approxInv which would be an all-one matrix
                }
                // General Beta div. alg. (for ED, no computation needed)
                else if (beta != 2) {
                    _approx.apply(std::pow, beta-2, approxInv);
                    approxInv->apply(Matrix::mul, _v, vOverApprox);
                    // vOverApprox now contains Approx^{Beta - 2} .* V
                    approxInv->apply(Matrix::mul, _approx, approxInv);
                    // approxInv now contains Approx^{Beta - 1};
                    //cout << "Inverse of approx: " << endl << *approxInv << endl;
                }
                // W Update, Numerator
                vOverApprox->multWithMatrix(_h, &wUpdateNum,
                    false, true,
                    _v.rows(), _v.cols() - p, _h.rows(),
                    0, p, 0, 0, 0, 0);
                //cout << "W update numerator: " << endl << wUpdateNum << endl;

                // W Update, Denominator (for KL this is a all-one matrix)
                // for ED (beta = 2) the original approximation is used
                if (beta != 1) {
                    approxInv->multWithMatrix(_h, &wUpdateDenom,
                        false, true,
                        _v.rows(), _v.cols() - p, _h.rows(),
                        0, p, 0, 0, 0, 0);
                    //cout << "W update denominator (before): " << endl << wUpdateDenom << endl;
                    ensureNonnegativity(wUpdateDenom);
                    //cout << "W update denominator (after): " << endl << wUpdateDenom << endl;
                }

                if (_t > 1) {
                    wpH = new Matrix(_v.rows(), _v.cols());
                    // Difference-based calculation of new approximation
                    computeWpH(p, *wpH);
                    _approx.sub(*wpH);
                }

                // for KL divergence: more efficient to use V over Approx (element wise division)
                // use row sum for denominator
                // for Euclidean Distance: nothing (pow with 1)
                // Finished calculation of vOverApprox / vOverApprox etc.
                // need this below, too, but maybe to complicated to get into separate function

                // W multiplicative update
                if (beta == 1) {
                    for (unsigned int j = 0; j < _w[p]->cols(); ++j) {
                        if (!_wColConstant[j]) {
                            double hRowSum = _h.rowSum(j, 0, _h.cols() - p - 1);
                            if (hRowSum <= 0.0) hRowSum = DIVISOR_FLOOR;
                            for (unsigned int i = 0; i < _w[p]->rows(); ++i) {
                                _w[p]->at(i, j) *= (wUpdateNum(i, j) / hRowSum);
                            }
                        }
                    }
                }
                else {
                    for (unsigned int j = 0; j < _w[p]->cols(); ++j) {
                        if (!_wColConstant[j]) {
                            for (unsigned int i = 0; i < _w[p]->rows(); ++i) {
                                _w[p]->at(i, j) *= (wUpdateNum(i, j) / wUpdateDenom(i, j));
                            }
                        }
                    }
                }
                
                //cout << "W after update: " << endl << *_w[p] << endl;

                // Difference-based calculation of new approximation
                if (_t > 1) {
                    computeWpH(p, *wpH);
                    _approx.add(*wpH);
                    delete wpH;
                    ensureNonnegativity(_approx);
                }
            }
        }

        // For T > 1, approximation has been calculated above.
        // For T = 1, this is more efficient for T = 1.
        if (_t == 1) {
            computeApprox();
        }

        // Now the approximation is up-to-date in any case.
        // see above
        if (beta == 1) {
            _v.apply(Matrix::div, _approx, vOverApprox);
        }
        else if (beta != 2) {
            _approx.apply(std::pow, beta-2, approxInv);
            approxInv->apply(Matrix::mul, _v, vOverApprox);
            approxInv->apply(Matrix::mul, _approx, approxInv);
        }

        // H Update
        hUpdate.zero();
        if (continuous) {
            *oldH = _h;
        }

        // Compute sparsity term
        // XXX: depends on H shifted, not H???!
        if (sparse) {
            double sqrtT = sqrt((double) _h.cols());
            double hRowSumSq, hRowLength;
            for (unsigned int i = 0; i < _h.rows(); ++i) {
                hRowSumSq = Matrix::dotRowRow(_h, i, _h, i);
                hRowLength = sqrt(hRowSumSq);
                csplus->at(i)   = sqrtT / hRowLength;
                csminus->at(i)  = sqrtT * _h.rowSum(i) / (hRowSumSq * hRowLength);
                //wColSums->at(i) = w.colSum(i);
            }
        }

        // Compute continuity term
        if (continuous) {
            double hRowSumSq, hDeltaSumSq;
            for (unsigned int i = 0; i < _h.rows(); ++i) {
                hRowSumSq = Matrix::dotRowRow(_h, i, _h, i);
                hDeltaSumSq = 0.0;
                for (unsigned int j = 1; j < _h.cols(); ++j) {
                    double hDelta = _h(i, j) - _h(i, j-1);
                    hDeltaSumSq += hDelta * hDelta;
                }

                (*ctplus)(i) = 4 * (double) _h.cols() / hRowSumSq;
                (*ctminus1)(i) = 2 * (double) _h.cols() / hRowSumSq;
                (*ctminus2)(i) = 2 * (double) _h.cols() * hDeltaSumSq / 
                                              (hRowSumSq * hRowSumSq);
            }
        }

        // Compute H update for each W[p] and average later
        for (unsigned int p = 0; p < _t; ++p) {
            if (beta == 1) {
                // Precalculation of column-sums of W_t
                for (unsigned int i = 0; i < _h.rows(); ++i) {
                    (*wpColSums)(i) = _w[p]->colSum(i);
                    if ((*wpColSums)(i) == 0.0)
                        (*wpColSums)(i) = DIVISOR_FLOOR;
                }
            }

            // Numerator
            _w[p]->multWithMatrix(*vOverApprox, &hUpdateNum,
                // transpose W[p]
                true, false, 
                // target dimension: R x (N-p)
                _w[p]->cols(), _w[p]->rows(), _v.cols() - p,
                0, 0, 0, p, 0, 0);
            // Denominator
            if (beta != 1) {
                _w[p]->multWithMatrix(*approxInv, &hUpdateDenom,
                    // transpose W[p]
                    true, false, 
                    // target dimension: R x (N-p)
                    _w[p]->cols(), _w[p]->rows(), _v.cols() - p,
                    0, 0, 0, p, 0, 0);
                ensureNonnegativity(hUpdateDenom, DIVISOR_FLOOR);
            }

            for (unsigned int j = 0; j < _h.cols() - p; ++j) {
                for (unsigned int i = 0; i < _h.rows(); ++i) {
                    double num   = hUpdateNum(i, j);
                    double denom = beta == 1 ? 
                                   wpColSums->at(i) : 
                                   hUpdateDenom(i, j);
                    // add sparsity and continuity terms to numerator
                    // and denominator of multiplicative update
                    // XXX: we might precompute the additive terms...
                    if (sparse) {
                        num   += _s(i, j) * _h(i, j) * csminus->at(i);
                        denom += _s(i, j) * csplus->at(i);
                    }
                    if (continuous) {
                        double l = j == 0 ? 0.0 : oldH->at(i, j - 1);
                        double r = j == _h.cols() - 1 ? 0.0 : _h(i, j + 1);
                        num   += _c(i,j) * ((l + r)  * ctminus1->at(i) + 
                                            _h(i, j) * ctminus2->at(i));
                        denom += _c(i, j) * _h(i, j) * ctplus->at(i);
                    }
                    //cout << "p = " << p << "; num(" << i << "," << j << "): " << num << endl;
                    //cout << "p = " << p << "; denom(" << i << "," << j << "): " << denom << endl;
                    hUpdate(i, j) += num / denom;
                }
            }
        }

        // Apply average update to H
        double updateNorm = _t;
        for (unsigned int j = 0; j <= _h.cols() - _t; ++j) {
            for (unsigned int i = 0; i < _h.rows(); ++i) {
                //cout << "H update (" << i << "," << j << "): " << hUpdate(i, j) << endl;
                _h(i, j) *= hUpdate(i, j) / updateNorm;
            }
        }
        for (unsigned int j = _h.cols() - _t + 1; j < _h.cols(); ++j) {
            --updateNorm;
            for (unsigned int i = 0; i < _h.rows(); ++i) {
                //cout << "H update (" << i << "," << j << "): " << hUpdate(i, j) << endl;
                _h(i, j) *= hUpdate(i, j) / updateNorm;
            }
            //cout << "scale column " << j << " by " << (1.0f / (double) updateNorm) << endl;
        }
        
        nextItStep(observer, maxSteps);
    }

    if (wpColSums)
        delete wpColSums;
    if (vOverApprox && vOverApprox != &_v)
        delete vOverApprox;
    if (approxInv && approxInv != &_approx)
        delete approxInv;
    if (oldH)
        delete oldH;
    if (ctminus2)
        delete ctminus2;
    if (ctminus1)
        delete ctminus1;
    if (ctplus)
        delete ctplus;
    if (csminus)
        delete csminus;
    if (csplus)
        delete csplus;

    //cout << "H after iteration: " << endl << _h << endl;
}


void Deconvolver::factorizeNMFEDIncomplete(unsigned int maxSteps, double eps,
                                           ProgressObserver *observer)
{
    // TODO: sparsity & continuity here

    assert(_t == 1);

    // helper variables
    double denom;
    Matrix& w = *(_w[0]); // for convenience

    Matrix hUpdateNum  (_h.rows(), _h.cols());
    Matrix hUpdateDenom(_h.rows(), _h.cols());
    Matrix wUpdateNum  ( w.rows(),  w.cols());
    Matrix wUpdateDenom( w.rows(),  w.cols());

    Matrix hhT(_h.rows(), _h.rows());
    Matrix wTw(_h.rows(), _h.rows());

    _numSteps = 0;
    while (_numSteps < maxSteps && !checkConvergence(eps, true)) {

        // W Update
        _v.multWithTransposedMatrix(_h, &wUpdateNum);
        _h.multWithTransposedMatrix(_h, &hhT);
        w.multWithMatrix(hhT, &wUpdateDenom);
        ensureNonnegativity(wUpdateDenom);

        for (unsigned int j = 0; j < w.cols(); ++j) {
            for (unsigned int i = 0; i < w.rows(); ++i) {
                w(i, j) *= (wUpdateNum(i, j) / wUpdateDenom(i, j));
            }
        }

        // H Update
        w.transposedMultWithMatrix(_v, &hUpdateNum);
        w.transposedMultWithMatrix(w, &wTw);
        wTw.multWithMatrix(_h, &hUpdateDenom);
        ensureNonnegativity(hUpdateDenom);

        for (unsigned int j = 0; j < _h.cols(); ++j) {
            for (unsigned int i = 0; i < _h.rows(); ++i) {
                _h(i, j) *= (hUpdateNum(i, j) / hUpdateDenom(i, j));
            }
        }

        nextItStep(observer, maxSteps);
    }

    // Update value of approximation (only once...)
    computeApprox();
}


#endif


void Deconvolver::factorizeNMFEDSparseNorm(unsigned int maxSteps, double eps,
                                           ProgressObserver *observer)
{
    assert(_t == 1);

    Matrix& w = *(_w[0]);
    //Matrix wNormalized = w;

    Matrix wTw(_h.rows(), _h.rows());                        // r x r
    Matrix hUpdateMatrixNum(_h.rows(), _h.cols());           // r x n
    Matrix hUpdateMatrixDenom(_h.rows(), _h.cols());         // r x n

    Matrix wUpdateMatrixNum1(_v.rows(), _h.rows());          // m x r
    Matrix hhT(_h.rows(), _h.rows());                        // r x r
    Matrix wUpdateMatrixDenom1(_v.rows(), _h.rows());        // m x r

    Matrix hvT(_h.rows(), _v.rows());                        // r x m
    Matrix wUpdateMatrixNum2(_h.rows(), _h.rows());          // r x r
    Matrix wUpdateMatrixDenom2(_h.rows(), _h.rows());        // r x r
    
    double num, denom;

    _numSteps = 0;
    while (_numSteps < maxSteps && !checkConvergence(eps, true)) {
        // Normalize W
        for (unsigned int j = 0; j < w.cols(); ++j) {
            double norm = sqrt(Matrix::dotColCol(w, j, w, j));
            for (unsigned int i = 0; i < w.rows(); ++i) {
                w(i, j) = w(i, j) / norm;
            }
        }

        // H Update
        // We should keep the value of wTw here, thus we don't use 
        // calculateNMFEDHUpdate().
        // Calculate W^T * V
        w.multWithMatrix(_v, &hUpdateMatrixNum, true, false,
            _h.rows(), _v.rows(), _h.cols(),
            0, 0, 0, 0, 0, 0);
        // Calculate W^T * W
        w.multWithMatrix(w, &wTw, true, false,
            _h.rows(), w.rows(), _h.rows(),
            0, 0, 0, 0, 0, 0);
        wTw.multWithMatrix(_h, &hUpdateMatrixDenom);
        for (unsigned int j = 0; j < _h.cols(); ++j) {
            for (unsigned int i = 0; i < _h.rows(); ++i) {
                denom = hUpdateMatrixDenom(i, j) + _s(i, j);
                if (denom <= 0.0) denom = DIVISOR_FLOOR;
                _h(i, j) *= hUpdateMatrixNum(i, j) / denom;
            }
        }

        // W Update
        _v.multWithTransposedMatrix(_h, &wUpdateMatrixNum1);
        _h.multWithTransposedMatrix(_h, &hhT);
        w.multWithMatrix(hhT, &wUpdateMatrixDenom1);
        hhT.multWithMatrix(wTw, &wUpdateMatrixNum2);
        _h.multWithTransposedMatrix(_v, &hvT);
        hvT.multWithMatrix(w, &wUpdateMatrixDenom2);
        for (unsigned int j = 0; j < w.cols(); ++j) {
            if (!_wColConstant[j]) {
                for (unsigned int i = 0; i < w.rows(); ++i) {
                    num   = wUpdateMatrixNum1(i, j)
                            + wUpdateMatrixNum2(j, j) * w(i, j);
                    denom = wUpdateMatrixDenom1(i, j) 
                            + wUpdateMatrixDenom2(j, j) * w(i, j);
                    if (denom <= 0.0) denom = DIVISOR_FLOOR;
                    w(i, j) *= num / denom;
                }
            }
        }

        nextItStep(observer, maxSteps);
    }
}


bool Deconvolver::checkConvergence(double eps, bool doComputeApprox)
{
    if (eps <= 0.0)
        return false;

    bool converged;
    if (_oldApprox == 0) {
        if (doComputeApprox) {
            computeApprox();
        }
        _oldApprox = new Matrix(_approx);
        converged = false;
    }
    else {
        if (doComputeApprox) {
            computeApprox();
        }
        Matrix approxDiff(_approx);
        approxDiff.sub(*_oldApprox);
        double zeta = approxDiff.frobeniusNorm() / _oldApprox->frobeniusNorm();
        converged = zeta < eps;
        *_oldApprox = _approx;
    }
    return converged;
}


void Deconvolver::computeApprox()
{
    if (_t == 1) {
        // this is much faster
        _w[0]->multWithMatrix(_h, &_approx);
    }
    else {
        Matrix wpH(_v.rows(), _v.cols());
        _approx.zero();
        for (unsigned int p = 0; p < _t; ++p) {
            computeWpH(p, wpH);
            _approx.add(wpH);
        }
    }
}


#ifdef HAVE_CUDA
void Deconvolver::computeApprox(GPUMatrix **w, const GPUMatrix &h, GPUMatrix *target)
{
    if (_t == 1) {
        // this is much faster
        w[0]->multWithMatrix(h, target);
    }
    else {
        GPUMatrix wpH(_v.rows(), _v.cols());
        target->zero();
        for (unsigned int p = 0; p < _t; ++p) {
            multWithShifted(*w[p], h, &wpH, p);
            target->add(wpH);
        }
    }
}
#endif


void Deconvolver::computeWpH(unsigned int p, Matrix& wpH)
{
    // Fill W[p]*H with zeros in the first p columns
    for (unsigned int j = 0; j < p; ++j) {
        for (unsigned int i = 0; i < wpH.rows(); ++i) {
            wpH(i, j) = 0.0;
        }
    }
    // Simulate multiplication with H shifted t spots to the right:
    // only use N - p columns of H for the matrix product
    // and store the result beginning at column p of W[p]*H
    // (for this reason W[p]*H had to be filled with zeros)
    _w[p]->multWithMatrix(_h, &wpH,
        false, false,
        _w[p]->rows(), _w[p]->cols(), _h.cols() - p,
        0, 0, 0, 0, 0, p);
}


#ifdef HAVE_CUDA
void Deconvolver::multWithShifted(const GPUMatrix &left, const GPUMatrix &right, 
    GPUMatrix *target, unsigned int shift) const
{
    if (shift > 0) {
        target->zero(0, 0, target->rows() - 1, shift - 1);
    }
    // See above.
    left.multWithMatrix(right, target,
        false, false,
        left.rows(), left.cols(), right.cols() - shift,
        0, 0, 0, 0, 0, shift);
}
#endif


void Deconvolver::computeError()
{
    computeApprox();
    Matrix errorMatrix(_approx);
    errorMatrix.sub(_v);
    _absoluteError = errorMatrix.frobeniusNorm();
    _relativeError = _absoluteError / _vFrob;
}


void Deconvolver::ensureNonnegativity(Matrix &m, double epsilon)
{
    for (unsigned int j = 0; j < m.cols(); ++j) {
        for (unsigned int i = 0; i < m.rows(); ++i) {
            if (m(i, j) <= 0.0) {
                m(i, j) = epsilon;
            }
        }
    }
}


void Deconvolver::normalizeMatrices(Deconvolver::MatrixNormalization method)
{
    switch (method)
    {
        case NormHFrob:
            normalizeHFrob();
            break;
        case NormWColumnsEucl:
            normalizeWColumnsEucl();
            break;
        case NoNorm: // fallthrough
        default:
            break;
    }
}


void Deconvolver::normalizeHFrob()
{
    // according to Wenwu Wang:
    // "We use the norm of the matrix H^q to normalise the each element in
    // H^q, i.e. each element in H^q is divided by the norm of H^q.
    // W^q(p) is normalised accordingly by multiplying each element with a norm
    // of matrix H^q(p), where H^q(p) is computed by shift H^q with p spots to
    // the right."
    double hNorm = _h.frobeniusNorm();
    for (unsigned int j = 0; j < _h.cols(); ++j) {
        for (unsigned int i = 0; i < _h.rows(); ++i) {
            _h(i, j) /= hNorm;
        }
    }

    // To simulate shift of H (as in the explanation above),
    // compute the Frobenius norms of the P-1 rightmost parts (submatrices) of H.
    // These are subtracted from the H norm.
    // Take care of p=0 by setting hNormRight[0] := 0.
    std::vector<double> hNormRight(_t, 0);
    unsigned int col = _h.cols() - 1;
    for (unsigned int p = 1; p < _t; ++p, --col) {
        if (p > 1)
            hNormRight[p - 1] = hNormRight[p - 2];
        hNormRight[p - 1] += Matrix::dotColCol(_h, col, _h, col);
    }
    for (unsigned int p = 0; p < _t; ++p) {
        for (unsigned int j = 0; j < _w[p]->cols(); ++j) {
            for (unsigned int i = 0; i < _w[p]->rows(); ++i) {
                _w[p]->at(i, j) *= sqrt(hNorm * hNorm - hNormRight[p]);
            }
        }
    }
}


void Deconvolver::normalizeWColumnsEucl()
{
    if (_t > 1) {
        throw std::runtime_error("Cannot normalize W columns for NMD");
    }

    // just a shortcut
    Matrix& w = *(_w[0]);
    // Normalize W and H
    for (unsigned int j = 0; j < w.cols(); ++j) {
        double norm = sqrt(Matrix::dotColCol(w, j, w, j));
        for (unsigned int i = 0; i < w.rows(); ++i) {
            w(i, j) = w(i, j) / norm;
        }
        for (unsigned int t = 0; t < _h.cols(); ++t) {
            _h(j, t) = _h(j, t) * norm;
        }
    }
}


void Deconvolver::nextItStep(ProgressObserver *observer, 
                             unsigned int maxSteps)
{
    ++_numSteps;
    // Call the ProgressObserver every once in a while (if applicable).
    if (observer && _numSteps % _notificationDelay == 0)
        observer->progressChanged((float)_numSteps / (float)maxSteps);
}


} // namespace nmf


} // namespace blissart
