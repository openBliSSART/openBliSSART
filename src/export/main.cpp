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


#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/HTKWriter.h>
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/RowVector.h>
#include <Poco/Util/RegExpValidator.h>
#include <Poco/Util/HelpFormatter.h>
#include <Poco/NumberParser.h>
#include <Poco/NumberFormatter.h>
#include <Poco/Path.h>
#include <Poco/File.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>


using namespace std;
using namespace blissart;
using namespace blissart::linalg;
using namespace Poco::Util;


class ExportTool : public BasicApplication
{
public:
    ExportTool() : BasicApplication(),
        _displayUsage(false),
        _extractAll(false),
        _outputFormat(HTK),
        _dataType(DataDescriptor::FeatureMatrix),
        _concat(false),
        _addType(false)
    {
        addSubsystem(new DatabaseSubsystem);
        addSubsystem(new StorageSubsystem);
        _types[0] = DataDescriptor::Spectrum;
        _types[1] = DataDescriptor::MelSpectrum;
        _types[2] = DataDescriptor::Gains;
        _types[3] = DataDescriptor::MagnitudeMatrix;
        _types[4] = DataDescriptor::MelMatrix;
        _types[5] = DataDescriptor::PhaseMatrix;
        _types[6] = DataDescriptor::FeatureMatrix;
    }


protected:
    typedef enum {
        HTK,
        Gnuplot
    } OutputFormat;


    void defineOptions(OptionSet &options)
    {
        Application::defineOptions(options);

        options.addOption(
            Option("help", "h",
                   "Displays usage information",
                   false));

        options.addOption(
            Option("all", "a",
                   "Exports the data from all data descriptors of the given "
                   "type in the database.",
                   false)
            .group("Data selection"));

        options.addOption(
            Option("process", "p",
                   "Exports data descriptors associated with "
                   "the given process IDs. Single process IDs or ranges (x..y) "
                   "can be given and must be separated with commata.",
                   false, "<list>", true)
            .group("Data selection")
            .validator(new RegExpValidator("(\\d+(\\.\\.\\d+)?,)*\\d+(\\.\\.\\d+)?")));

        options.addOption(
            Option("format", "f",
                   "Selects an output format.",
                   false, "<format>", true)
            .validator(new RegExpValidator("(htk|gnuplot)")));

        options.addOption(
            Option("concat", "c",
                   "Concatenates data descriptors of the same type. "
                   "The type of concatenation (column- or row-wise) depends "
                   "on the type of data descriptor."));

        string typesRegexp = "(";
        string typesDescr;
        for (unsigned int i = 0; i < _nTypes; ++i) {
            typesRegexp += DataDescriptor::strForTypeShort(_types[i]);
            typesDescr += DataDescriptor::strForType(_types[i]) + " (" +
                DataDescriptor::strForTypeShort(_types[i]) + ")";
            if (i < _nTypes - 1) {
                typesRegexp += "|";
                typesDescr += ", ";
            }
        }
        typesRegexp += ")";

        options.addOption(
            Option("type", "t",
                   "Selects the type of data descriptor to export. "
                   "Available types are: " + typesDescr,
                   false, "<type>", true)
            .validator(new RegExpValidator(typesRegexp)));

        options.addOption(
            Option("strip-prefix", "",
                   "Strips the given path prefix from original file names.",
                   false, "<path>", true));

        options.addOption(
            Option("target-dir", "",
                   "Sets the target directory for output. Relative path names "
                   "are interpreted with respect to this directory.",
                   false, "<path>", true));

        options.addOption(
            Option("add-type", "T",
                   "Adds the type of data to the file names."));

    }


    void handleOption(const string &name, const string &value)
    {
        Application::handleOption(name, value);

        if (name == "all") {
            _extractAll = true;
        }
        else if (name == "process") {
            vector<string> ranges;
            string::size_type pos = 0, pos2;
            while (pos < value.length()) {
                pos2 = value.find_first_of(',', pos);
                if (pos2 == string::npos) {
                    ranges.push_back(value.substr(pos));
                    break;
                }
                else {
                    ranges.push_back(value.substr(pos, pos2 - pos));
                }
                pos = pos2 + 1;
            }
            for (vector<string>::const_iterator itr = ranges.begin();
                itr != ranges.end(); ++itr)
            {
                int start, end;
                pos = itr->find("..");
                if (pos != string::npos) {
                    start = Poco::NumberParser::parse(itr->substr(0, pos));
                    end = Poco::NumberParser::parse(itr->substr(pos + 2));
                }
                else {
                    start = end = Poco::NumberParser::parse(*itr);
                }
                for (int id = start; id <= end; ++id) {
                    _processIDs.push_back(id);
                }
            }
        }
        else if (name == "type") {
            for (unsigned int i = 0; i < _nTypes; ++i) {
                if (value == DataDescriptor::strForTypeShort(_types[i])) {
                    _dataType = _types[i];
                }
            }
        }
        else if (name == "format") {
            if (value == "htk") {
                _outputFormat = HTK;
            }
            else {
                _outputFormat = Gnuplot;
            }
        }
        else if (name == "concat") {
            _concat = true;
        }
        else if (name == "strip-prefix") {
            _stripPrefix = value;
        }
        else if (name == "target-dir") {
            _targetDir = value;
        }
        else if (name == "add-type") {
            _addType = true;
        }
        else if (name == "help") {
            _displayUsage = true;
            stopOptionsProcessing();
        }
    }


    void writeFile(const Matrix& m, string& inputFile, int index, int sampleFreq)
    {
        // Determine file name from input file and change its extension
        // according to output format.
        if (!_stripPrefix.empty()) {
            if (inputFile.substr(0, _stripPrefix.size()) == _stripPrefix) {
                inputFile.erase(0, _stripPrefix.size());
            }
            else {
                cerr << "File " << inputFile << 
                    " did not match path prefix " << _stripPrefix << endl;
            }
        }
        string inputFileNoExt = inputFile;
        inputFileNoExt.erase(inputFile.find_last_of('.'));
        if (_addType) {
            inputFileNoExt += '_';
            inputFileNoExt += DataDescriptor::strForTypeShort(_dataType);
        }
        if (index >= 0) {
            inputFileNoExt += '_';
            inputFileNoExt += Poco::NumberFormatter::format(index);
        }
        Poco::Path inputFilePath(inputFileNoExt);
        Poco::Path outputFilePath;
        if (_targetDir.empty() && inputFilePath.isAbsolute()) {
            outputFilePath = inputFilePath;
        }
        else {
            outputFilePath = _targetDir;
            outputFilePath.append(inputFilePath);
        }
        if (_outputFormat == HTK) {
            outputFilePath.setExtension("htk");
        }
        else {
            outputFilePath.setExtension("dat");
        }
        cout << "Writing " << outputFilePath.toString() << endl;

        // Ensure that output directory exists.
        Poco::Path container = outputFilePath;
        container.makeParent();
        if (container.depth() > 0) {
            Poco::File(container).createDirectories();
        }

        // Output matrix data.
        string outputFile = outputFilePath.toString();
        ofstream os(outputFile.c_str(),
            ios_base::out | ios_base::binary);
        if (_outputFormat == HTK) {
            HTKWriter::writeMatrix(os, m, sampleFreq);
        }
        else {
            // "Column-wise" Gnuplot output.
            if (_dataType == DataDescriptor::Gains || 
                _dataType == DataDescriptor::FeatureMatrix)
            {
                for (unsigned int j = 0; j < m.cols(); ++j) {
                    os << j << "\t";
                    for (unsigned int i = 0; i < m.rows(); ++i) {
                        os << m.at(i, j);
                        if (i < m.rows() - 1) os << "\t";
                    }
                    os << endl;
                }
            }
            // "Row-wise" Gnuplot output.
            else {
                for (unsigned int i = 0; i < m.rows(); ++i) {
                    os << i << "\t";
                    for (unsigned int j = 0; j < m.cols(); ++j) {
                        os << m.at(i, j);
                        if (j < m.cols() - 1) os << "\t";
                    }
                    os << endl;
                }
            }
        }
    }


    int main(const vector<string> &args)
    {
        if (_displayUsage || args.size() != 0 ||
            !(_extractAll || _processIDs.size() > 0))
        {
            HelpFormatter formatter(this->options());
            formatter.setUnixStyle(true);
            formatter.setAutoIndent();
            formatter.setUsage(this->commandName() + " <options>\n");
            formatter.setHeader("ExportTool, exports data in various formats");
            formatter.format(cout);
            return EXIT_USAGE;
        }

        DatabaseSubsystem& dbs = getSubsystem<DatabaseSubsystem>();
        StorageSubsystem& sts = getSubsystem<StorageSubsystem>();

        for (vector<int>::const_iterator itr = _processIDs.begin();
            itr != _processIDs.end(); ++itr)
        {
            ProcessPtr process = dbs.getProcess(*itr);
            if (process.isNull()) {
                throw Poco::InvalidArgumentException("Invalid process ID: " +
                    Poco::NumberFormatter::format(*itr));
            }
            vector<DataDescriptorPtr> dds = dbs.getDataDescriptors(*itr);
            vector<Poco::SharedPtr<Matrix> > matrices;
            map<int, int> indexMap;  // used for sorting DDs by index
            unsigned int totalDim = 0;
            unsigned int dim = 0;

            // Determine the type of concatenation.
            bool concatRows = _dataType == DataDescriptor::FeatureMatrix ||
                              _dataType == DataDescriptor::Gains;

            // Find matrices of the given type for the data descriptors of the
            // process.
            for (vector<DataDescriptorPtr>::const_iterator dItr = dds.begin();
                dItr != dds.end(); ++dItr)
            {
                if ((*dItr)->type == _dataType) {
                    Poco::SharedPtr<Matrix> data;
                    // Gains is a row vector. Convert it to a 1 x N matrix for
                    // further processing.
                    if (_dataType == DataDescriptor::Gains) {
                        RowVector gains(sts.getLocation(*dItr).toString());
                        data = new Matrix(1, gains.dim());
                        data->setRow(0, gains);
                    }
                    else {
                        data = new Matrix(sts.getLocation(*dItr).toString());
                    }
                    matrices.push_back(data);
                    indexMap[(int)matrices.size() - 1] = (*dItr)->index;
                    totalDim += (concatRows ? data->rows() : data->cols());
                    if (dim == 0) {
                        dim = concatRows ? data->cols() : data->rows();
                    }
                    else if ((concatRows && dim != data->cols()) ||
                             (!concatRows && dim != data->rows()))
                    {
                        throw Poco::InvalidArgumentException(
                            "Column number mismatch in data descriptor " +
                            Poco::NumberFormatter::format((*dItr)->descrID));
                    }
                }
            }

            if (_concat && concatRows) {
                // Row-wisely concatenate all matrices into one.
                Poco::SharedPtr<Matrix> allData = new Matrix(totalDim, dim);
                unsigned int rowIndex = 0;
                for (map<int, int>::const_iterator itr = indexMap.begin();
                    itr != indexMap.end(); ++itr)
                {
                    Poco::SharedPtr<Matrix> m = matrices[itr->second];
                    for (unsigned int i = 0; i < m->rows(); ++i) {
                        for (unsigned int j = 0; j < m->cols(); ++j) {
                            allData->at(rowIndex, j) = m->at(i, j);
                        }
                        ++rowIndex;
                    }
                }
                writeFile(*allData, process->inputFile, -1, 
                    process->sampleFreq);
            }
            else if (_concat && !concatRows) {
                // Column-wisely concatenate all matrices into one.
                Poco::SharedPtr<Matrix> allData = new Matrix(dim, totalDim);
                unsigned int colIndex = 0;
                for (map<int, int>::const_iterator itr = indexMap.begin();
                    itr != indexMap.end(); ++itr)
                {
                    Poco::SharedPtr<Matrix> m = matrices[itr->second];
                    for (unsigned int j = 0; j < m->cols(); ++j) {
                        for (unsigned int i = 0; i < m->rows(); ++i) {
                            allData->at(i, colIndex) = m->at(i, j);
                        }
                        ++colIndex;
                    }
                }
                writeFile(*allData, process->inputFile, -1, 
                    process->sampleFreq);
            }
            else /* (!concat) */ {
                for (map<int, int>::const_iterator indexItr = indexMap.begin();
                    indexItr != indexMap.end(); ++indexItr)
                {
                    writeFile(*matrices[indexItr->first], process->inputFile, 
                        indexItr->first, process->sampleFreq);
                }
            }
        } // foreach dd

        return EXIT_OK;
    }


private:
    bool                 _displayUsage;
    bool                 _extractAll;
    vector<int>          _processIDs;
    OutputFormat         _outputFormat;
    DataDescriptor::Type _dataType;
    bool                 _concat;
    string               _stripPrefix;
    string               _targetDir;
    bool                 _addType;

    static const unsigned int _nTypes = 7;
    DataDescriptor::Type _types[_nTypes];
};


POCO_APP_MAIN(ExportTool);
