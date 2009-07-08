//
// $Id: Process.h 855 2009-06-09 16:15:50Z alex $
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


#ifndef __BLISSART_PROCESS_H__
#define __BLISSART_PROCESS_H__


#include <blissart/DatabaseEntity.h>
#include <blissart/WindowFunctions.h>

#include <common.h>

#include <string>
#include <map>

#include <Poco/Timestamp.h>
#include <Poco/Exception.h>


namespace blissart {


/**
 * A computation process with parameters.
 */
class LibFramework_API Process : public DatabaseEntity
{
public:
    /**
     * Default constructor. Creates an empty Process object.
     * Sets the timestamp to the current time.
     */
    Process();

    /**
     * Copies all data from another Process.
     */
    Process(const Process& other);

    /**
     * Creates a Process object with the given name and input file.
     * Sets the timestamp to the current time.
     */
    Process(const std::string& name, const std::string& inputFile,
            int sampleFreq);

    /**
     * Unique process ID.
     */
    int processID;

    /**
     * A (usually short) name of the process, e.g. "NMF" or "ICA".
     */
    std::string name;

    /**
     * The full path name of the file that was given as input to the process.
     */
    std::string inputFile;

    /**
     * Timestamp of the time the process was started.
     */
    Poco::Timestamp startTime;

    /**
     * The sample frequency of the source audio file.
     */
    int sampleFreq;

    /**
     * Key-value pairs representing process parameters, e.g. window size for a
     * FFT or number of components for NMF factorization.
     */
    std::map<std::string, std::string> parameters;

    /**
     * @name Convenience functions
     * Note that these functions aren't for convience only, but also help to
     * assure correct process-parameter settings.
     * @{
     */

    /**
     * Searches this process' parameters for "windowSize" and returns the
     * associated value.
     * @return                  the window size parameter of this process (ms)
     * @throw                   Poco::NotFoundException
     */
    int windowSize() const;


    /**
     * Sets the corresponding parameter of this process to the given window size.
     * @param  windowSize       the window size
     * @throw                   Poco::InvalidArgumentException
     */
    void setWindowSize(unsigned int windowSize);


    /**
     * Search this process' parameters for "overlap" and returns the associated
     * value.
     * @return                  the overlap parameter of this process
     * @throw                   Poco::NotFoundException
     */
    double overlap() const;


    /**
     * Sets the corresponding parameter of this process to the given overlap.
     * @param  overlap          the overlap
     * @throw                   Poco::InvalidArgumentException
     */
    void setOverlap(double overlap);


    /**
     * Search this process' parameters for "windowFunction" and returns a
     * pointer to the associated window function.
     * @return                  a pointer to the window function of this process
     * @throw                   Poco::NotFoundException,
     *                          Poco::NotImplementedException
     */
    WindowFunction windowFunction() const;


    /**
     * Sets the corresponding parameter of this process to the given
     * window function.
     * @param  winFun           a pointer to the window function
     * @throw                   Poco::InvalidArgumentException
     */
    void setWindowFunction(WindowFunction winFun);

    
    /**
     * Searches this process' parameters for "components" and returns the
     * associated value.
     * @return                  the number of components 
     *                          (of a separation process)
     * @throw                   Poco::NotFoundException
     */
    int components() const;


    /**
     * Sets the corresponding parameter of this process to the given number
     * of components.
     * @param  r                the number of components
     * @throw                   Poco::InvalidArgumentException
     */
    void setComponents(unsigned int r);


    /**
     * Searches this process' parameters for "spectra" and returns the
     * associated value.
     * @return                  the number of spectra
     *                          (of a separation process)
     * @throw                   Poco::NotFoundException
     */
    int spectra() const;


    /**
     * Sets the corresponding parameter of this process to the given number
     * of spectra.
     * @param  t                the number of spectra
     * @throw                   Poco::InvalidArgumentException
     */
    void setSpectra(unsigned int t);


    /**
     * @}
     */
};


typedef Poco::AutoPtr<Process> ProcessPtr;


} // namespace blissart


#endif // __BLISSART_PROCESS_H__
