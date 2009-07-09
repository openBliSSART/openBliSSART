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

#ifndef __WAVEEXPORTER_H__
#define __WAVEEXPORTER_H__


#include "AbstractExporter.h"
#include <string>


// Forward declaration
namespace blissart { namespace linalg { class Matrix; } }


/**
 * Exports datasets to PCM wave files
 */
class WaveExporter : public AbstractExporter
{
public:
    /**
     * Constructs a WaveExporter for the given prefix and sample rate.
     * The names of the files will be composed of the given prefix plus
     * an additional number depending on the number of available
     * matrix rows.
     * @param   prefix      the filename prefix
     * @param   sampleRate  the sample rate
     */
    WaveExporter(const std::string& prefix, unsigned int sampleRate);


    /**
     * Performs the actual export.
     * @param   matrix      the data matrix
     * @return              true iff no error occured during export
     */
    virtual bool doExport(const blissart::linalg::Matrix& matrix);


private:
    std::string  _prefix;
    unsigned int _sampleRate;
};


#endif
