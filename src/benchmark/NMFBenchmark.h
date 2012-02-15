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


#ifndef __NMF_BENCHMARK_H__
#define __NMF_BENCHMARK_H__


#include "Benchmark.h"
#include <blissart/ProgressObserver.h>


namespace Poco { namespace Util { class OptionSet; } }


namespace benchmark {


/**
 * Measures the performance of algorithms for non-negative matrix factorization
 * (NMF).
 */
class NMFBenchmark : public Benchmark, blissart::ProgressObserver
{
public:
    /**
     * Constructor, sets default options.
     */
    NMFBenchmark();


    /**
     * Implementation of Benchmark interface.
     */
	void run();


    /**
     * Implementation of ProgressObserver interface.
     */
	virtual void progressChanged(float progress);


    /**
     * Implementation of Benchmark interface.
     */
    inline const char *name() const 
	{ 
		return "NMFBenchmark"; 
	}


    /**
     * Overrides Benchmark method.
     */
    virtual void setOptions(const Benchmark::OptionsMap& optionsMap);


    /**
     * Overrides Benchmark method.
     */
    static void addOptions(Poco::Util::OptionSet& options);


private:
    unsigned int _rows;
    std::string  _cf;
    unsigned int _iter;
};


} // namespace benchmark


#endif // __NMD_BENCHMARK_H__

