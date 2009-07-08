//
// $Id: HTKWriter.h 858 2009-06-10 08:24:44Z alex $
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


#ifndef __BLISSART_HTKWRITER_H__
#define __BLISSART_HTKWRITER_H__


#include <common.h>
#include <iostream>
#include <Poco/BinaryWriter.h>
#include <Poco/Types.h>


namespace blissart {


namespace linalg { class Matrix; }


/**
 * Writes data in HTK format.
 */
class LibFramework_API HTKWriter : public Poco::BinaryWriter
{
public:
    /**
     * Sample kinds according to HTK book. Note that not all methods referred
     * to in these constants are implemented in openBliSSART.
     */
    typedef enum {
        WAVEFORM,           // sampled waveform
        LPC,                // linear prediction filter coefficients
        LPREFC,             // linear prediction reflection coefficients
        LPCEPSTRA,          // LPC cepstral coefficients
        LPDELCEP,           // LPC cepstra plus delta coefficients
        IREFC,              // LPC reflection coef in 16 bit integer format
        MFCC,               // mel-frequency cepstral coefficients
        FBANK,              // log mel-filter bank channel outputs
        MELSPEC,            // linear mel-filter bank channel outputs
        USER,               // user defined sample kind
        DISCRETE,           // vector quantised data
        PLP                 // PLP cepstral coefficients
    } HTKParmKind;


    /**
     * Parameter qualifiers according to HTK book. Note that not all methods 
     * referred to in these constants are implemented in openBliSSART.
     */
    typedef enum {
        _E = 0000100,       // has energy
        _N = 0000200,       // absolute energy suppressed
        _D = 0000400,       // has delta coefficients
        _A = 0001000,       // has acceleration coefficients
        _C = 0002000,       // is compressed
        _Z = 0004000,       // has zero mean static coef.
        _K = 0010000,       // has CRC checksum
        _O = 0020000,       // has 0’th cepstral coef.
        _V = 0040000,       // has VQ data
        _T = 0100000        // has third differential coef.
    } HTKParmQualifier;


    /**
     * Constructs a HTKWriter that writes to the given output stream.
     */
    HTKWriter(std::ostream& str);


    /**
     * Writes the HTK header.
     * @param   nSamples      Number of samples
     * @param   samplePeriod  Sampling period in 100ns intervals 
     *                        (conforming to HTK)
     * @param   sampleSize    Sample size in bytes
     * @param   parmKind      Sample kind, usually one of the values
     *                        in HTKParmKind bitwise OR'd with some
     *                        HTKParmQualifiers.
     */
    void writeHeader(Poco::Int32 nSamples, Poco::Int32 samplePeriod, 
        Poco::Int16 sampleSize, Poco::Int16 parmKind);


    /**
     * Convenience function to write a Matrix object. Only the sample period
     * and sample kind must be given.
     */
    static void writeMatrix(std::ostream& str,
        const linalg::Matrix& matrix, 
        Poco::Int32 samplePeriod, Poco::Int16 parmKind = USER);


private:
    HTKWriter();
};


} // namespace blissart


#endif // __BLISSART_HTKWRITER_H__

