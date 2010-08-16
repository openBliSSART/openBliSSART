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


#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__


#include <cmath>
#include <common.h>
#include <map>
#include <string>
#include <ctime>


// Forward declaration
namespace Poco { class Logger; }


/**
 * Classes implementing component-level performance measures
 */
namespace benchmark {


/**
 * \defgroup benchmark Benchmarking framework
 * \addtogroup benchmark
 * @{
 */

/**
 * An abstract base class for benchmarks
 */
class Benchmark
{
public:
    /**
     * Runs the benchmark.
     * Abstract method that must be implemented by derived classes.
     */
	virtual void run() = 0;


    /**
     * Returns the name of the benchmark.
     * Abstract method that must be implemented by derived classes.
     * @return  a pointer to a string
     */
    virtual const char *name() const = 0;


    typedef double ElapsedTime;
	typedef std::map<std::string, ElapsedTime> ElapsedTimeMap;
    typedef std::map<std::string, std::string> OptionsMap;


    virtual void setOptions(const OptionsMap& optionsMap);
    Poco::Logger& logger() const;


    virtual ~Benchmark() {}
    

    typedef clock_t ClockTime;


	inline ElapsedTimeMap elapsedTimes();


protected:
    class ScopedStopwatch
    {
    public:
        ScopedStopwatch(Benchmark& parent, const std::string& id);
        virtual ~ScopedStopwatch();
    private:
        ClockTime _start;
        Benchmark& _parent;
        std::string _id;
    };


    inline void setElapsedTime(const std::string& id, ElapsedTime time);
	ElapsedTimeMap _elapsedTimes;
};


/**
 * @}
 */


Benchmark::ElapsedTimeMap Benchmark::elapsedTimes()
{
	return _elapsedTimes;
}


void Benchmark::setElapsedTime(const std::string& id, ElapsedTime time)
{
    _elapsedTimes[id] = time;
}


} // namespace benchmark


#endif // __BENCHMARK_H__
