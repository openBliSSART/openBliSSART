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


#include <blissart/audio/AudioData.h>
#include <blissart/audio/audio.h>
#include <blissart/ThreadedApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>
#include <blissart/NMDTask.h>
#include <blissart/SeparationTask.h>
#include <blissart/ClassificationTask.h>
#include <blissart/validators.h>

#include <Poco/Util/HelpFormatter.h>
#include <Poco/Util/RegExpValidator.h>
#include <Poco/DateTimeFormatter.h>
#include <Poco/NumberFormatter.h>
#include <Poco/DateTime.h>
#include <Poco/Mutex.h>

#include <iostream>
#include <iomanip>


using namespace std;
using namespace blissart;
using namespace blissart::audio;
using namespace blissart::validators;
using namespace Poco;
using namespace Poco::Util;


class SEPTool : public ThreadedApplication
{
public:
    SEPTool() :
        _scripted(false),
        _classify(false),
        _displayUsage(false),
        _volatile(false),
        _epsilon(0.0),
        _overlap(0.5),
        _clResponseID(0),
        _presetLabelID(0),
        _maxIter(100),
        _algName("Multiplicative update (divergence)"),
        _cfName("Extended KL divergence"),
        _nmdCostFunction(NMDTask::ExtendedKLDivergence),
        _nrComponents(20),
        _nrSpectra(5),
        _preserveInit(false),
        _windowSize(25),
        _wfName("Square root of Hann function"),
        _windowFunction(SqHannFunction),
        _reduceMids(false),
        _preemphasisCoeff(0.0),
        _zeroPadding(false),
        _removeDC(false),
        _separationMethod(SeparationTask::NMD),
        _dataKind(SeparationTask::MagnitudeSpectrum)
    {
        addSubsystem(new DatabaseSubsystem());
        addSubsystem(new StorageSubsystem());
    }


protected:
    virtual void initialize(Application &self)
    {
        ThreadedApplication::initialize(self);

        // Initialize LibAudio.
        blissart::audio::initialize();

        // Copy parameters from the configuration such that they can be
        // displayed later. Also ClassificationTasks have to use the same
        // window size and overlap parameters.
        _zeroPadding = config().getBool("blissart.fft.zeropadding", false);
        _removeDC    = config().getBool("blissart.audio.remove_dc", false);
        _reduceMids  = config().getBool("blissart.audio.reduce_mids", false);
        _preemphasisCoeff = config().getDouble("blissart.audio.preemphasis", 0.0);
        _windowFunction = windowFunctionForShortName(
            config().getString("blissart.fft.windowfunction", "sqhann"));
        _wfName      = windowFunctionName(_windowFunction);
        _windowSize  = config().getInt("blissart.fft.windowsize", 25);
        _overlap     = config().getDouble("blissart.fft.overlap", 0.5);
    }


    virtual void uninitialize()
    {
        // Shut down LibAudio.
        if (audio::isInitialized())
            audio::shutdown();

        ThreadedApplication::uninitialize();
    }


    virtual void defineOptions(OptionSet &options)
    {
        Application::defineOptions(options);

        options.addOption(
            Option("help", "h",
                   "Displays usage screen",
                   false));

        options.addOption(
            Option("scripted", "S",
                   "Run in scripted mode, i.e. the input files contain "
                   "list of sound files to process.",
                   false));

        options.addOption(
            Option("volatile", "v",
                   "Run in volatile mode, i.e. do not write anything to the "
                   "database.",
                   false));

        options.addOption(
            Option("window-function", "w",
                   "The window function for creating the spectrogram. "
                   "Must be one of \"hann\", \"hamming\", \"sqhann\" or "
                   "\"rectangle\". Default is \"sqhann\"",
                   false, "<function>", true)
            .validator(new RegExpValidator("hann|hamming|sqhann|rectangle"))
            .binding("blissart.fft.windowfunction"));

        options.addOption(
            Option("overlap", "o",
                   "Overlap (must be in the interval [0,1)). Default is " +
                   NumberFormatter::format(_overlap), false, "<number>",
                   true)
            .validator(new RangeValidator<double>(0, false, 1, true))
            .binding("blissart.fft.overlap"));

        options.addOption(
            Option("window-size", "s",
                   "Window size in milliseconds. Default is " +
                   NumberFormatter::format(_windowSize),
                   false, "<number>", true)
            .validator(new RangeValidator<int>(1))
            .binding("blissart.fft.windowsize"));

        options.addOption(
            Option("reduce-mids", "r",
                   "Subtract the right from the left channel when converting "
                   "from stereo to mono.",
                   false));

        options.addOption(
            Option("preemphasis", "k",
                   "Performs preemphasis with the given 0 <= k < 1.",
                   false, "<k>", true)
            .validator(new RangeValidator<double>(0.0, false, 1, true))
            .binding("blissart.audio.preemphasis"));

        options.addOption(
            Option("remove-dc", "d",
                   "Removes the DC component from each frame.",
                   false));

        options.addOption(
            Option("zero-padding", "z",
                   "Performs zero-padding before FFT.",
                   false));

        options.addOption(
            Option("method", "m",
                   "The method to be used for component separation. "
                   "Currently, this option has no effect.",
                   false, "<method>", true)
            .validator(new RegExpValidator("nmd")));

        options.addOption(
            Option("cost-function", "f",
                   "NMD cost function. Must be one of \"dist\" or \"div\". "
                   "Default is \"div\".",
                   false, "<name>", true)
            .validator(new RegExpValidator("dist|div")));

        options.addOption(
            Option("sparsity", "y",
                   "Sparsity weight for NMD cost function. "
                   "Implies --normalize if > 0",
                   false, "<number>", true)
            .validator(new RangeValidator<double>(0.0)));

        options.addOption(
            Option("normalize-spectra", "N",
                   "Normalize NMD spectra to unity length.",
                   false));

        options.addOption(
            Option("components", "c",
                   "The number of components. Default is " +
                   NumberFormatter::format(_nrComponents),
                   false, "<number>", true)
            .validator(new RangeValidator<int>(1)));

        options.addOption(
            Option("spectra", "T",
                   "The number of spectra per component (for NMD). Default is " +
                   NumberFormatter::format(_nrSpectra),
                   false, "<number>", true)
            .validator(new RangeValidator<int>(1)));

        options.addOption(
            Option("init", "I",
                   "A range of classification objects for initialization "
                   "of the spectra.",
                   false, "<min>..<max>", true)
            .repeatable(true)
            .validator(new RegExpValidator("\\d+\\.\\.\\d+")));

        options.addOption(
            Option("preserve", "P",
                   "Preserve initialization of the spectra during iteration.",
                   false));

        options.addOption(
            Option("precision", "e",
                   "The desired precision (epsilon) of the result. If set to "
                   "zero, perform the maximum number of iteration steps "
                   "anyway. Default is " + NumberFormatter::format(_epsilon),
                   false, "<number>", true)
            .validator(new RangeValidator<double>(0.0)));

        options.addOption(
            Option("max-iter", "i",
                   "The maximum number of iterations to perform. Default is " +
                   NumberFormatter::format(_maxIter),
                   false, "<number>", true)
            .validator(new RangeValidator<int>(1)));

        options.addOption(
            Option("classify", "l",
                   "Classify the components using the specified response.",
                   false, "<responseID>", true)
            .validator(new RangeValidator<int>(1)));

        options.addOption(
            Option("preset-label", "L",
                   "Assigns the given class label to the components which "
                   "have been initialized using the -I option, instead of the "
                   "class label predicted by the classifier.",
                   false, "<labelID>", true)
            .validator(new RangeValidator<int>(1)));

        options.addOption(
            Option("num-threads", "n",
                   "The number of concurrent threads (max 16). Default is " +
                   NumberFormatter::format(numThreads()),
                   false, "<number>", true)
            .validator(new RangeValidator<int>(1)));

        options.addOption(
            Option("prefix", "p",
                   "The prefix to be used for export of the separated "
                   "components.",
                   false, "<prefix>", true));

        options.addOption(
            Option("data-kind", "D",
                "The data kind to perform separation on "
                "(\"spectrum\" or \"melSpectrum\").",
                false, "<kind>", true)
                .validator(new RegExpValidator("(spectrum|melSpectrum)")));

    }


    virtual void handleOption(const string &name, const string &value)
    {
        Application::handleOption(name, value);

        if (name == "help") {
            _displayUsage = true;
            stopOptionsProcessing();
        }
        else if (name == "scripted") {
            _scripted = true;
        }
        else if (name == "volatile") {
            _volatile = true;
        }
        else if (name == "cost-function") {
            if (value == "dist") {
                _nmdCostFunction = NMDTask::EuclideanDistance;
                _cfName = "Euclidean distance";
            }
            else if (value == "div") {
                _nmdCostFunction = NMDTask::ExtendedKLDivergence;
                _cfName = "Extended KL divergence";
            }
        }
        else if (name == "sparsity") {
            _nmdSparsity = NumberParser::parseFloat(value);
            _nmdNormalize = true;
        }
        else if (name == "normalize-spectra") {
            _nmdNormalize = true;
        }
        else if (name == "components") {
            _nrComponents = NumberParser::parse(value);
        }
        else if (name == "spectra") {
            _nrSpectra = NumberParser::parse(value);
        }
        else if (name == "init") {
            int min = NumberParser::parse(
                value.substr(0, value.find_first_of('.')));
            int max = NumberParser::parse(
                value.substr(value.find_last_of('.') + 1));
            for (int id = min; id <= max; ++id) {
                _initObjectIDs.push_back(id);
            }
        }
        else if (name == "preserve") {
            _preserveInit = true;
        }
        else if (name == "precision") {
            _epsilon = NumberParser::parseFloat(value);
        }
        else if (name == "max-iter") {
            _maxIter = NumberParser::parse(value);
        }
        else if (name == "classify") {
            _classify = true;
            _clResponseID = NumberParser::parse(value);
        }
        else if (name == "preset-label") {
            _presetLabelID = NumberParser::parse(value);
        }
        else if (name == "num-threads") {
            setNumThreads(NumberParser::parse(value));
        }
        else if (name == "reduce-mids") {
            _reduceMids = true;
            config().setBool("blissart.audio.reduce_mids", true);
        }
        else if (name == "remove-dc") {
            _removeDC = true;
            config().setBool("blissart.audio.remove_dc", true);
        }
        else if (name == "zero-padding") {
            _zeroPadding = true;
            config().setBool("blissart.fft.zeropadding", true);
        }
        else if (name == "prefix") {
            _exportPrefix = value;
        }
        else if (name == "method") {
            if (value == "nmd")
                _separationMethod = SeparationTask::NMD;
            else
                throw Poco::NotImplementedException("Unknown separation method.");
        }
        else if (name == "data-kind") {
            if (value == "spectrum")
                _dataKind = SeparationTask::MagnitudeSpectrum;
            else if (value == "melSpectrum")
                _dataKind = SeparationTask::MelSpectrum;
            else
                throw Poco::NotImplementedException("Unknown data kind.");
        }
    }


    virtual void removeTask(BasicTaskPtr task)
    {
        if (_classify && task->state() != BasicTask::TASK_FINISHED) {
            SeparationTaskPtr sepTask = task.cast<SeparationTask>();
            if (!sepTask.isNull()) {
                // This SeparationTask failed. Thus add the corresponding
                // filename to the _failedFileNames list and remove the
                // corresponding ClassificationTask as well.
                ClassificationTaskPtr clTask;
                _genMutex.lock();
                {
                    _failedFileNames.push_back(sepTask->fileName());
                    clTask = _tasksMap[sepTask];
                    _tasksMap.erase(sepTask);
                }
                _genMutex.unlock();
                removeTask(clTask);
            }
        }

        ThreadedApplication::removeTask(task);
    }


    virtual int main(const vector<string> &args)
    {
        if (_displayUsage || args.empty()) {
            HelpFormatter formatter(this->options());
            formatter.setUnixStyle(true);
            formatter.setAutoIndent();
            formatter.setUsage(this->commandName() +
                " <options> FILE1 [FILE2 ...]\nwhere FILE can be one or more"
                " WAV, MP3 or script file(s)\n");
            formatter.setHeader(
                "SEPTool, a tool for blind source separation using NMD");
            formatter.format(cout);
            return EXIT_USAGE;
        }

        cout << "SEPTool, "
             << DateTimeFormatter::format(LocalDateTime(), "%Y/%m/%d %H:%M:%S")
             << endl << endl
             << setw(20) << "Method: ";
        switch (_separationMethod) {
        case SeparationTask::NMD:
            cout << "Non-Negative Matrix Deconvolution";
            break;
        default:
            throw Poco::NotImplementedException("Unknown separation method.");
        }
        cout << endl
             << setw(20) << "Window function: " << _wfName << endl
             << setw(20) << "Window size: " << _windowSize << " ms" << endl
             << setw(20) << "Overlap: " << _overlap << endl;

        if (_separationMethod == SeparationTask::NMD) {
            cout << setw(20) << "Cost function: " << _cfName << endl;
        }

        cout << setw(20) << "# of components: " << _nrComponents << endl;

        if (_separationMethod == SeparationTask::NMD) {
            cout << setw(20) << "# of spectra: " << _nrSpectra << endl;
        }

        cout << setw(20) << "Max. iterations: " << _maxIter << endl
             << setw(20) << "Epsilon: " << _epsilon
             << (_epsilon > 0 ? "" : " (all iterations)") << endl
             << setw(20) << "# of threads: " << numThreads() << endl
             << setw(20) << "Volatile: "
             << (_volatile ? "True " : "False") << endl
             << setw(20) << "Reduce mids: "
             << (_reduceMids ? "True" : "False") << endl
             << setw(20) << "Remove DC: "
             << (_removeDC ? "True" : "False") << endl
             << setw(20) << "Zero-Padding: "
             << (_zeroPadding ? "True" : "False") << endl
             << setw(20) << "Preemphasis (k): "
             << _preemphasisCoeff << endl
             << setw(20) << "Export prefix: " << _exportPrefix << endl;

        if (_classify) {
            cout << setw(20) << "Classification: " << "using Response #"
                 << _clResponseID << endl;
        }
        cout << endl;

        // Initialize the random generator.
        srand((unsigned)time(NULL));

        // Initialize the task manager.
        initializeTaskManager<ThreadedApplication>();

        // Determine the list of input files.
        vector<string> inputFiles = _scripted ? parseScriptFiles(args) : args;

        // Create tasks per input file.
        for (vector<string>::const_iterator it = inputFiles.begin();
            it != inputFiles.end(); ++it)
        {
            NMDTask* nmdTask;
            // SeparationTask:
            SeparationTaskPtr newSepTask;
            switch (_separationMethod) {
            case SeparationTask::NMD:
                nmdTask = new NMDTask(
                    *it, _dataKind, _nmdCostFunction,
                    _nrComponents, _nrSpectra, _maxIter,_epsilon, _volatile
                );
                nmdTask->setSparsity(_nmdSparsity);
                nmdTask->setNormalizeSpectra(_nmdNormalize);
                newSepTask = nmdTask;
                break;
            default:
                throw Poco::NotImplementedException("Unhandled method type.");
            }

            if (!_exportPrefix.empty())
                newSepTask->setExportPrefix(_exportPrefix);

            if (!_initObjectIDs.empty()) {
                vector<ClassificationObjectPtr> objects;
                DatabaseSubsystem& dbs = getSubsystem<DatabaseSubsystem>();
                for (vector<int>::const_iterator itr = _initObjectIDs.begin();
                    itr != _initObjectIDs.end(); ++itr)
                {
                    objects.push_back(dbs.getClassificationObject(*itr));
                    if (objects.back().isNull()) {
                        throw Poco::InvalidArgumentException(
                            "No classification object found with ID " +
                            Poco::NumberFormatter::format(*itr));
                    }
                }
                newSepTask->setInitializationObjects(objects, _preserveInit);
            }
            // Now add the new SeparationTask.
            addTask(newSepTask);

            // ClassificationTask:
            if (_classify) {
                ClassificationTaskPtr newClTask = new ClassificationTask(
                        _clResponseID,
                        newSepTask
                );
                
                // Preset class label for the initialized components,
                // if desired. They are at indices 0 .. N-1 where N
                // is the number of initialized components.
                if (_presetLabelID) {
                    for (int ci = 0; ci < (int)_initObjectIDs.size(); ++ci) {
                        newClTask->presetClassLabel(ci, _presetLabelID);
                    }
                }

                // The _taskMap stores pairing information of SeparationTasks
                // and ClassificationTasks. Hence it must be updated.
                _genMutex.lock();
                _tasksMap[newSepTask] = newClTask;
                _genMutex.unlock();

                // Set the corresponding SeparationTask as dependency and add
                // the new ClassificationTask.
                QueuedTaskManager::TaskDeps deps;
                deps.insert(newSepTask.get());
                addTask(newClTask, deps);
            }
        }

        // Wait until all tasks have finished and display some progress
        // information.
        waitForCompletion();

        // If any SeparationTasks failed, the correspondig fileNames have been
        // stored and can be displayed in terms of an error message now.
        if (!_failedFileNames.empty()) {
            cerr << endl
                 << "The following files were not processed:"
                 << endl;
            for (vector<string>::const_iterator it = _failedFileNames.begin();
                it != _failedFileNames.end(); ++it)
            {
                cerr << "\t" << *it << endl;
            }
        }

        return EXIT_OK;
    }


private:
    bool               _scripted;
    bool               _classify;
    bool               _displayUsage;
    bool               _volatile;
    double             _epsilon;
    double             _overlap;
    int                _clResponseID;
    int                _presetLabelID;
    int                _maxIter;
    string             _algName;
    string             _cfName;
    double             _nmdSparsity;
    bool               _nmdNormalize;
    NMDTask::CostFunction _nmdCostFunction;
    int                _nrComponents;
    int                _nrSpectra;
    vector<int>        _initObjectIDs;
    bool               _preserveInit;
    int                _windowSize;
    string             _wfName;
    WindowFunction     _windowFunction;
    bool               _reduceMids;
    double             _preemphasisCoeff;
    bool               _zeroPadding;
    bool               _removeDC;
    string             _exportPrefix;

    // The method to be used for component separation.
    SeparationTask::SeparationMethod   _separationMethod;

    // The type of data to perform separation on.
    SeparationTask::DataKind           _dataKind;

    // This vector holds the names of the files during whose processing errors
    // have occured.
    vector<string>                     _failedFileNames;

    // Tasks related stuff.
    typedef map<SeparationTaskPtr, ClassificationTaskPtr> TasksMap;
    TasksMap                           _tasksMap;

    // The following mutex is used to lock both _failedFilesNames and _tasksMap.
    FastMutex                          _genMutex;
};


POCO_APP_MAIN(SEPTool);

