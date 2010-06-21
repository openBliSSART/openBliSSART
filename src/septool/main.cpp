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


class SeparationTool : public ThreadedApplication
{
public:
    SeparationTool() :
        _scripted(false),
        _classify(false),
        _displayRelativeError(false),
        _displayUsage(false),
        _volatile(false),
        _epsilon(0.0),
        _overlap(0.5),
        _clResponseID(0),
        _presetLabelID(0),
        _maxIter(100),
        _algName("Multiplicative update (divergence)"),
        _cfName("Extended KL divergence"),
        _nmdSparsity(0.0),
        _nmdContinuity(0.0),
        _nmdNormalize(nmf::Deconvolver::NoNorm),
        _nmfCostFunction(nmf::Deconvolver::KLDivergence),
        _nrComponents(20),
        _nrSpectra(1),
        _preserveInit(false),
        _matrixGenFunc(nmf::gaussianRandomGenerator),
        _windowSize(25),
        _wfName("Square root of Hann function"),
        _windowFunction(SqHannFunction),
        _reduceMids(false),
        _preemphasisCoeff(0.0),
        _zeroPadding(false),
        _removeDC(false),
        _exportComponents(false),
        _exportSpectrogram(false),
        _exportSpectra(false),
        _exportGains(false),
        _separationMethod(SeparationTask::NMD)
    {
    }


protected:
    virtual void initialize(Application &self)
    {
        // Add storage and database subsystems if run in non-volatile mode,
        // or if classification is desired, which needs some input data.
        if (!_volatile || _classify || !_initObjectIDs.empty()) {
            addSubsystem(new DatabaseSubsystem());
            addSubsystem(new StorageSubsystem());
        }

        // Don't call the applications's initialize() until now, 
        // to make sure these subsystems are initialized.
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
        BasicApplication::defineOptions(options);

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
            Option("relative-error", "E",
                   "Displays the relative error for each file.",
                   false));

        options.addOption(
            Option("window-function", "w",
                   "The window function for creating the spectrogram. "
                   "Must be one of \"hann\", \"hamming\", \"sqhann\" or "
                   "\"rectangle\". Default is \"sqhann\"",
                   false, "<function>", true)
            .validator(new RegExpValidator("hann|hamming|sqhann|rectangle")));

        options.addOption(
            Option("overlap", "o",
                   "Overlap (must be in the interval [0,1)). Default is " +
                   NumberFormatter::format(_overlap), false, "<number>",
                   true)
            .validator(new RangeValidator<double>(0, false, 1, true)));

        options.addOption(
            Option("window-size", "s",
                   "Window size in milliseconds. Default is " +
                   NumberFormatter::format(_windowSize),
                   false, "<number>", true)
            .validator(new RangeValidator<int>(1)));

        options.addOption(
            Option("reduce-mids", "r",
                   "Subtract the right from the left channel when converting "
                   "from stereo to mono.",
                   false));

        options.addOption(
            Option("preemphasis", "k",
                   "Performs preemphasis with the given 0 <= k < 1.",
                   false, "<k>", true)
            .validator(new RangeValidator<double>(0.0, false, 1, true)));

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
                   "NMF cost function. Must be one of the following: "
                   "\"ed\", \"kl\", \"eds\", \"kls\", \"klc\", or \"edsn\". "
                   "Default is \"kl\".",
                   false, "<name>", true)
            .validator(new RegExpValidator("eds?|kl(c|s)?|edsn")));

        options.addOption(
            Option("sparsity", "y",
                   "Sparsity weight for NMF cost function. ",
                   false, "<number>", true)
            .validator(new RangeValidator<double>(0.0)));

        options.addOption(
            Option("continuity", "t",
                   "Continuity weight for NMF cost function. ",
                   false, "<number>", true)
            .validator(new RangeValidator<double>(0.0)));

        options.addOption(
            Option("normalize-matrices", "N",
                   "Normalize NMF/NMD matrices: "
                   "\"H\": normalize H such that it has unity Frobenius "
                   "norm. \"W\": normalize columns of W to unity Euclidean "
                   "norm (NMF only).",
                   false, "<method>", true)
            .validator(new RegExpValidator("W|H")));

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
            Option("generator", "g",
                   "Sets the generator function for initialization of the "
                   "matrices (gaussian, uniform or unity). "
                   "Default is gaussian.",
                   false, "<func>", true)
            .validator(new RegExpValidator("(gaussian|uniform|unity)")));

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
            Option("export-prefix", "",
                   "The prefix to be used for export of the separated "
                   "components. Default is input file name without extension",
                   false, "<prefix>", true));

        options.addOption(
            Option("export-components", "p",
                   "Export the separated components to WAV files.",
                   false));

        options.addOption(
            Option("export-spectrogram", "",
                   "Export the reconstructed spectrogram as a WAV file.",
                   false));

        options.addOption(
            Option("export-matrices", "",
                   "Export the separation matrices. Use \"W\" for spectra, "
                   "\"H\" for gains or \"WH\" for both (not the product!)",
                   false, "<name>", true)
            .validator(new RegExpValidator("(W|H|WH)")));
    }


    virtual void handleOption(const string &name, const string &value)
    {
        BasicApplication::handleOption(name, value);

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
        else if (name == "relative-error") {
            _displayRelativeError = true;
        }
        else if (name == "window-function") {
            config().setString("blissart.fft.windowfunction", value);
        }
        else if (name == "window-size") {
            config().setInt("blissart.fft.windowsize", 
                            NumberParser::parse(value));
        }
        else if (name == "overlap") {
            config().setDouble("blissart.fft.overlap", 
                               NumberParser::parseFloat(value));
        }
        else if (name == "preemphasis") {
            config().setDouble("blissart.audio.preemphasis", 
                               NumberParser::parseFloat(value));
        }
        else if (name == "cost-function") {
            if (value == "ed") {
                _nmfCostFunction = nmf::Deconvolver::EuclideanDistance;
            }
            else if (value == "kl") {
                _nmfCostFunction = nmf::Deconvolver::KLDivergence;
            }
            else if (value == "eds") {
                _nmfCostFunction = nmf::Deconvolver::EuclideanDistanceSparse;
            }
            else if (value == "kls") {
                _nmfCostFunction = nmf::Deconvolver::KLDivergenceSparse;
            }
            else if (value == "klc") {
                _nmfCostFunction = nmf::Deconvolver::KLDivergenceContinuous;
            }
            else if (value == "edsn") {
                _nmfCostFunction = nmf::Deconvolver::EuclideanDistanceSparseNormalized;
            }
            _cfName = nmf::Deconvolver::costFunctionName(_nmfCostFunction);
        }
        else if (name == "sparsity") {
            _nmdSparsity = NumberParser::parseFloat(value);
        }
        else if (name == "continuity") {
            _nmdContinuity = NumberParser::parseFloat(value);
        }
        else if (name == "normalize-matrices") {
            if (value == "W") {
                _nmdNormalize = nmf::Deconvolver::NormWColumnsEucl;
            }
            else if (value == "H") {
                _nmdNormalize = nmf::Deconvolver::NormHFrob;
            }
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
        else if (name == "generator") {
            _matrixGenFunc = nmf::randomGeneratorForName(value);
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
        else if (name == "export-prefix") {
            _exportPrefix = value;
        }
        else if (name == "export-components") {
            _exportComponents = true;
        }
        else if (name == "export-spectrogram") {
            _exportSpectrogram = true;
        }
        else if (name == "export-matrices") {
            if (value.find('W') != string::npos) 
                _exportSpectra = true;
            if (value.find('H') != string::npos)
                _exportGains = true;
        }
        else if (name == "method") {
            if (value == "nmd")
                _separationMethod = SeparationTask::NMD;
            else
                throw Poco::NotImplementedException("Unknown separation method.");
        }
    }


    virtual void removeTask(BasicTaskPtr task)
    {
        // Handle task instances that where cancelled or simply failed.
        if (task->state() != BasicTask::TASK_FINISHED) {
            // Handle SeparationTasks.
            SeparationTaskPtr sepTask = task.cast<SeparationTask>();
            if (!sepTask.isNull()) {
                // Push the input file name to the list of failed files.
                _failedFileNames.push_back(sepTask->fileName());
                // Remove the associated ClassificationTask, if applicable.
                if (_classify) {
                    ClassificationTaskPtr clTask;
                    _genMutex.lock();
                    {
                        clTask = _tasksMap[sepTask];
                        _tasksMap.erase(sepTask);
                    }
                    _genMutex.unlock();
                    ThreadedApplication::removeTask(clTask);
                }
            }
            // Handle ClassificationTasks.
            else {
                ClassificationTaskPtr clTask = task.cast<ClassificationTask>();
                if (!clTask.isNull()) {
                    // Push the input file name to the list of failed files.
                    _failedFileNames.push_back(clTask->fileName());
                }
            }
        }

        // Handle successful task instances.
        else {
            // Handle SeparationTasks.
            SeparationTaskPtr sepTask = task.cast<SeparationTask>();
            if (!sepTask.isNull() && _displayRelativeError) {
                _errorMap[sepTask->fileName()] = sepTask->relativeError();
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
                " WAV, OGG, FLAC, or script file(s)\n");
            formatter.setHeader(
                "SepTool, a tool for blind source separation using NMF/NMD");
            formatter.format(cout);
            return EXIT_USAGE;
        }

        cout << "SepTool, "
             << DateTimeFormatter::format(LocalDateTime(), "%Y/%m/%d %H:%M:%S")
             << endl << endl
             << setw(20) << "Method: ";
        switch (_separationMethod) {
        case SeparationTask::NMD:
            if (_nrSpectra == 1)
                cout << "Non-Negative Matrix Factorization";
            else
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

        if (_initObjectIDs.size() > 0) {
            cout << setw(20) << "# of initialized spectra: " 
                 << _initObjectIDs.size() << endl;
            cout << setw(20) << "Preserve initialized spectra: "
                 << (_preserveInit ? "True" : "False") << endl;
        }

        cout << setw(20) << "Initialization: "  
             << nmf::randomGeneratorName(_matrixGenFunc) << endl;

        if (_separationMethod == SeparationTask::NMD) {
            cout << setw(20) << "# of spectra: " << _nrSpectra << endl;
            if (_nmfCostFunction == nmf::Deconvolver::EuclideanDistanceSparse ||
                _nmfCostFunction == nmf::Deconvolver::KLDivergenceSparse ||
                _nmfCostFunction == nmf::Deconvolver::EuclideanDistanceSparseNormalized)
            {
                cout << setw(20) << "sparsity: " << _nmdSparsity << endl;
            }
            if (_nmfCostFunction == nmf::Deconvolver::KLDivergenceContinuous) {
                cout << setw(20) << "continuity: " << _nmdContinuity << endl;
            }
            cout << setw(20) << "normalize matrices: "
                 << (_nmdNormalize == nmf::Deconvolver::NoNorm ? "False" : "True") 
                 << endl;
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
             << setw(20) << "Export: ";
        if (_exportComponents)
            cout << "Components ";
        if (_exportSpectrogram)
            cout << "Spectrogram ";
        if (_exportSpectra)
            cout << "Spectra ";
        if (_exportGains)
            cout << "Gains";
        cout << endl;

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
                    *it, _nmfCostFunction,
                    _nrComponents, _nrSpectra, _maxIter,_epsilon, _volatile
                );
                nmdTask->setGeneratorFunction(_matrixGenFunc);
                nmdTask->setSparsity(_nmdSparsity);
                nmdTask->setContinuity(_nmdContinuity);
                nmdTask->setNormalizeMatrices(_nmdNormalize);
                newSepTask = nmdTask;
                break;
            default:
                throw Poco::NotImplementedException("Unhandled method type.");
            }

            newSepTask->setComputeRelativeError(_displayRelativeError);
            newSepTask->setExportComponents(_exportComponents);
            newSepTask->setExportSpectrogram(_exportSpectrogram);
            newSepTask->setExportSpectra(_exportSpectra);
            newSepTask->setExportGains(_exportGains);

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

        if (_displayRelativeError) {
            cout << endl << "Relative factorization error:" << endl;
            for (ErrorMap::const_iterator it = _errorMap.begin();
                it != _errorMap.end(); ++it)
            {
                cerr << "\t" << it->first << "\t" << it->second << endl;
            }
        }

        return EXIT_OK;
    }


private:
    bool               _scripted;
    bool               _classify;
    bool               _displayRelativeError;
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
    double             _nmdContinuity;
    nmf::Deconvolver::MatrixNormalization _nmdNormalize;
    nmf::Deconvolver::NMFCostFunction _nmfCostFunction;
    int                _nrComponents;
    int                _nrSpectra;
    vector<int>        _initObjectIDs;
    bool               _preserveInit;
    linalg::Matrix::GeneratorFunction _matrixGenFunc;
    int                _windowSize;
    string             _wfName;
    WindowFunction     _windowFunction;
    bool               _reduceMids;
    double             _preemphasisCoeff;
    bool               _zeroPadding;
    bool               _removeDC;
    bool               _exportComponents;
    bool               _exportSpectrogram;
    bool               _exportSpectra;
    bool               _exportGains;
    string             _exportPrefix;

    // The method to be used for component separation.
    SeparationTask::SeparationMethod   _separationMethod;

    // This vector holds the names of the files during whose processing errors
    // have occured.
    vector<string>                     _failedFileNames;

    // Tasks related stuff.
    typedef map<SeparationTaskPtr, ClassificationTaskPtr> TasksMap;
    TasksMap                           _tasksMap;

    // Store the relative error upon completion of SeparationTasks.
    typedef map<std::string, double> ErrorMap;
    ErrorMap                           _errorMap;

    // The following mutex is used to lock both _failedFilesNames and _tasksMap.
    FastMutex                          _genMutex;
};


POCO_APP_MAIN(SeparationTool);

