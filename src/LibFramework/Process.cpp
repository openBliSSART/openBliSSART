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


#include <blissart/Process.h>
#include <Poco/NumberFormatter.h>
#include <Poco/NumberParser.h>
#include <blissart/transforms/SpectralSubtractionTransform.h>
#include <blissart/transforms/PowerTransform.h>
#include <blissart/transforms/MelFilterTransform.h>
#include <blissart/transforms/SlidingWindowTransform.h>

#include<iostream>
using namespace std;
using namespace blissart::transforms;


namespace blissart {


Process::Process() :
  DatabaseEntity(DatabaseEntity::Process),
  processID(0),
  sampleFreq(0)
{
    startTime.update();
}


Process::Process(const Process& other) :
  DatabaseEntity(other),
  processID(other.processID),
  name(other.name),
  inputFile(other.inputFile),
  startTime(other.startTime),
  sampleFreq(other.sampleFreq)
{
    cout << "Process::Process 1\n";
}


Process::Process(const std::string& processName, 
                 const std::string& inputFileName,
                 int aSampleFreq) :
  DatabaseEntity(DatabaseEntity::Process),
  name(processName),
  inputFile(inputFileName),
  sampleFreq(aSampleFreq)
{
    cout << "Process::Process 2\n";
    startTime.update();
}

Process::Process(int aprocess_id, const std::string& processName,
                 const std::string& inputFileName,
                 int astart_time,
                 int aSampleFreq) :
  DatabaseEntity(DatabaseEntity::Process),
  processID(aprocess_id),
  name(processName),
  inputFile(inputFileName),
  startTime(astart_time),
  sampleFreq(aSampleFreq)
{
    cout << "Process::Process 2a\n";
    //startTime.update();
}

// The following lines are neccessary to avoid superflous warnings about
// the throw(...) declarations in Visual C++.
#if defined(_WIN32) || defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable:4290)
#endif


int Process::windowSize() const
{
    map<string, string>::const_iterator it;
    if ((it = parameters.find("windowSize")) == parameters.end())
        throw Poco::NotFoundException("Missing windowSize.");
    //return (unsigned int)(Poco::NumberParser::parse(it->second.c_str()));
    return (int)(Poco::NumberParser::parse(it->second.c_str()));
}


void Process::setWindowSize(unsigned int windowSize)
{
    if (windowSize == 0)
        throw Poco::InvalidArgumentException("Invalid window size.");
    parameters["windowSize"] = Poco::NumberFormatter::format(windowSize);
}


double Process::overlap() const
{
    map<string, string>::const_iterator it;
    if ((it = parameters.find("overlap")) == parameters.end())
        throw Poco::NotFoundException("Missing overlap.");
    return DatabaseEntity::parseDouble(it->second.c_str());
}


void Process::setOverlap(double overlap)
{
    if (overlap < 0 || overlap >= 1)
        throw Poco::InvalidArgumentException("Invalid overlap.");
    parameters["overlap"] = DatabaseEntity::formatDouble(overlap);
}


WindowFunction Process::windowFunction() const
{
    map<string, string>::const_iterator it;
    if ((it = parameters.find("windowFunction")) == parameters.end())
        throw Poco::NotFoundException("Missing windowFunction.");
    return windowFunctionForName(it->second);
}


void Process::setWindowFunction(WindowFunction winFun)
{
    cout << "setWindowsFunction In\n";
    parameters["windowFunction"] = windowFunctionName(winFun);
    cout << "setWindowsFunction Out\n";
} 


const vector<MatrixTransform*> Process::transforms() const
{
    vector<MatrixTransform*> tfs;
    map<string, string>::const_iterator tcIt = parameters.find("transformCount");
    if (tcIt != parameters.end()) {
        int transformCount = Poco::NumberParser::parse(tcIt->second);
        for (int ti = 1; ti <= transformCount; ++ti) {
            string tfName;
            map<string, string>::const_iterator tnIt = 
                parameters.find("transform" + Poco::NumberFormatter::format(ti));
            if (tnIt != parameters.end()) {
                // TODO: Load parameters
                MatrixTransform* tf = 0;
                tfName = tnIt->second;
                if (tfName == "Power spectrum") {
                    tf = new PowerTransform();
                }
                else if (tfName == "Mel filter") {
                    tf = new MelFilterTransform(sampleFreq);
                    // TODO: Zero padding (introduce process param "fftSize"??)
                    int ws = windowSize() * sampleFreq / 1000;
                    ((MelFilterTransform*)tf)->setBins(ws / 2 + 1);
                }
                else if (tfName == "Sliding window spectrum") {
                    tf = new SlidingWindowTransform();
                }
                else if (tfName == "Spectral subtraction") {
                    tf = new SpectralSubtractionTransform;
                }
                else {
                    throw Poco::InvalidArgumentException(
                        "Invalid name for matrix transform: " + tfName);
                }
                tfs.push_back(tf);
            }
            else {
                throw Poco::NotFoundException("Transformation #" + 
                    Poco::NumberFormatter::format(ti) + 
                    " not found in parameters!");
            }
        }
    }
    return tfs;
}


void Process::setTransforms(const std::vector<MatrixTransform*>& tfs)
{
    // TODO: Implement me (now in FTTask)
}


int Process::components() const
{
    map<string, string>::const_iterator it;
    if ((it = parameters.find("components")) == parameters.end())
        throw Poco::NotFoundException("Missing # of components.");
    return (unsigned int)(Poco::NumberParser::parse(it->second.c_str()));
}


void Process::setComponents(unsigned int r)
{
    if (r == 0)
        throw Poco::InvalidArgumentException("Invalid # of components.");
    parameters["components"] = Poco::NumberFormatter::format(r);
}


int Process::spectra() const
{
    map<string, string>::const_iterator it;
    if ((it = parameters.find("spectra")) == parameters.end())
        throw Poco::NotFoundException("Missing # of spectra.");
    return (unsigned int)(Poco::NumberParser::parse(it->second.c_str()));
}


void Process::setSpectra(unsigned int t)
{
    if (t == 0)
        throw Poco::InvalidArgumentException("Invalid # of spectra.");
    parameters["spectra"] = Poco::NumberFormatter::format(t);
}


#if defined(_WIN32) || defined(_MSC_VER)
#  pragma warning(pop)
#endif


} // namespace blissart
