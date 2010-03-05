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


#include "SampleSeparator.h"
#include "ARFFExporter.h"
#include "WaveExporter.h"

#include <blissart/linalg/Matrix.h>
#include <blissart/audio/audio.h>
#include <baseName.h>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>


using namespace std;
using namespace blissart::linalg;


#define MINIMUM_PRECISION   1e-20
#define DEFAULT_PRECISION   1e-10
#define DEFAULT_MAX_ITER    20


typedef enum { FormatWave, FormatARFF } FileFormat;


/**
 * Prints usage information.
 * @param   binaryName      the name of this executable
 */
static void printUsage(const char* binaryName)
{
    cerr << endl
         << "Usage: " << baseName(binaryName) << " [options] FILES ..." << endl
         << "  --prefix=PR     A prefix (PR) for the numbered output files" << endl
         << "  --nsources=x    The number (x > 1) of sources to be separated" << endl
         << "  --as-wave       Output result as PCM wave files (default)" << endl
         << "  --as-arff       Output result as ARFF file suitable for Weka" << endl
         << "  --force         If the input-files have a variable number of" << endl
         << "                  samples or differ in sample rate then you can" << endl
         << "                  force the process by respectively appending" << endl
         << "                  the neccessary number of their 'expected value'" << endl
         << "                  to the end of each affected dataset" << endl
         << "  --prec=p        the desired precision (i > " << MINIMUM_PRECISION
                               << ", defaults to " << DEFAULT_PRECISION << ")" << endl
         << "  --max-iter=m    the maximum number of iterations during FastICA" << endl
         << "                  (m, defaults to " << DEFAULT_MAX_ITER << ")" << endl
         << "  --help          This message" << endl;
}


/**
 * Determines if a particular file exists.
 * @param   fileName        the name of the file
 * @return                  true iff the file exists and could be opened for
 *                          reading
 */
static inline bool fileExists(const char* fileName)
{
    ifstream fs(fileName);
    return !fs.fail();
}


/**
 * Returns a pointer to the value associated with the given argument, e.g
 * for the string "prefix=abc" returns a pointer to the beginning of "abc".
 */
static inline const char *getArgValue(const char *arg)
{
    const char *result = strchr(arg, '=');
    return result ? result + 1 : 0;
}


/**
 * Parses the command-line arguments and stores the result in the corresponding
 * variables.
 * For a meaning of the parameters please see printUsage(...)
 * @return                  true iff all elements could be parsed correctly and
 *                          all values are within their allowed limits
 */
static bool parseCmdArgs(char** argv, string* prefix, unsigned int* nSources,
                         bool* force, double* prec, unsigned int* maxIter,
                         FileFormat* format, vector<string>* inFiles)
{
    // Default values
    *format = FormatWave;
    *nSources = 0;
    *force = false;
    *prec = DEFAULT_PRECISION;
    *maxIter = DEFAULT_MAX_ITER;

    for (; argv && *argv; argv++) {
        // See if a value has been provided for the current option.
        const char *value = getArgValue(*argv);

        if (value) {
            if (!strncmp(*argv, "--prefix", 8)) {
                *prefix = value;
            }
            else if (!strncmp(*argv, "--nsources", 10)) {
                *nSources = atoi(value);
            }
            else if (!strncmp(*argv, "--prec", 6)) {
                *prec = atof(value);
            }
            else if (!strncmp(*argv, "--max-iter", 10)) {
                *maxIter = atoi(value);
            }
        } else {
            if (!strcmp(*argv, "--as-wave")) {
                *format = FormatWave;
            }
            else if (!strcmp(*argv, "--as-arff")) {
                *format = FormatARFF;
            }
            else if (!strcmp(*argv, "--force")) {
                *force = true;
            }
            else if (!strcmp(*argv, "--help")) {
                return false;
            }
            else {
                // This is a filename.
                inFiles->push_back(*argv);
            }
        }
    }

    // In case --nsources wasn't specified the # of inFiles
    // will be considered as # of sources
    if (0 == *nSources)
        *nSources = (unsigned int)inFiles->size();

    // Check that prefix is non-empty
    if (prefix->empty()) {
        cerr << "No prefix specified!" << endl;
        return false;
    }

    // Check that multiple input files have been specified
    if (inFiles->size() <= 1) {
        cerr << "At least 2 input files must be given!" << endl;
        return false;
    }

    // Also check that nSources is >= 2
    if (*nSources <= 1) {
        cerr << "The # of sources to be separated must be > 1!" << endl;
        return false;
    }

    // Check if at least n input files have been specified
    // where n is the number of sources to be separated
    if (inFiles->size() < *nSources) {
        cerr << "At least n input files must be specified where n is the number" << endl
             << "of sources to be separated!" << endl;
        return false;
    }

    // Check if all input files exist
    for (vector<string>::const_iterator it = inFiles->begin();
        it != inFiles->end(); ++it)
    {
        if (!fileExists((*it).c_str())) {
            cerr << "Could not open file '" << *it << "'!" << endl;
            return false;
        }
    }

    // Check that prec is > MINIMUM_PREC
    if (*prec < MINIMUM_PRECISION) {
        cerr << "The precision must be > " << MINIMUM_PRECISION << "!" << endl;
        return false;
    }

    // Check that maxIter is >= 1
    if (*maxIter < 1) {
        cerr << "The maximum # of FastICA iterations must be > 1!" << endl;
        return false;
    }

    return true;
}


int main(int argc, char **argv)
{
    string prefix;
    unsigned int nSources, maxIter;
    bool force;
    double prec;
    FileFormat format;
    vector<string> inFiles;

    if (!parseCmdArgs(argv + 1, &prefix, &nSources, &force,
                      &prec, &maxIter, &format, &inFiles)) {
        printUsage(argv[0]);
        exit(1);
    }

    // Initialize LibAudio.
    blissart::audio::initialize();

    // Give some information
    cout << "Trying to separate " << nSources << " sources from "
         << inFiles.size() << " files." << endl;

    try {
        // Prepare separator and exporter
        auto_ptr<AbstractSeparator> separator;
        auto_ptr<AbstractExporter> exporter;
        switch (format) {
            case FormatWave:
                separator = auto_ptr<AbstractSeparator>(
                    new SampleSeparator(nSources, inFiles, force, prec, maxIter));
                exporter = auto_ptr<AbstractExporter>(new WaveExporter(prefix,
                    (static_cast<SampleSeparator*>(separator.get()))->sampleRate()));
                break;
            case FormatARFF:
                separator = auto_ptr<AbstractSeparator>(
                    new SampleSeparator(nSources, inFiles, force, prec, maxIter));
                exporter = auto_ptr<AbstractExporter>(new ARFFExporter(prefix + ".arff"));
                break;
            default:
                throw runtime_error("Format NOT supported yet!");
        }

        // Stress the CPU ;-)
        const Matrix* matrix = separator->separate();

        // Output
        cout << "Exporting...";
        if (!exporter->doExport(*matrix))
            throw runtime_error("Export failed!");
        cout << "done." << endl;
    } catch (exception& ex) {
        cout << endl << "ERR: " << ex.what() << endl;
        // Shutdown LibAudio.
        blissart::audio::shutdown();
        exit(1);
    } catch (...) {
        cout << "An unknown error occured!" << endl;
        // Shutdown LibAudio.
        blissart::audio::shutdown();
        exit(1);
    }

    // Shutdown LibAudio.
    blissart::audio::shutdown();

    return 0;
}
