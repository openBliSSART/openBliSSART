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


#include "CreateProcessDialog.h"
#include <blissart/NMDTask.h>
#include <blissart/FTTask.h>

#include <QMessageBox>


using namespace std;
using namespace Poco;
using Poco::Util::LayeredConfiguration;


namespace blissart {


CreateProcessDialog::CreateProcessDialog(QWidget *parent) :
    ThreadedDialog(parent)
{
    _ui.setupUi(this);

    // Do NOT change the order of the following items:
    _ui.cbWindowFunction->addItem(tr("Square root of Hann function"));
    _ui.cbWindowFunction->addItem(tr("Hann function"));
    _ui.cbWindowFunction->addItem(tr("Hamming function"));
    _ui.cbWindowFunction->addItem(tr("Rectangle function"));

    // Do NOT change the order of the following items:
    _ui.cbCostFunction->addItem(tr("Extended KL divergence"));
    _ui.cbCostFunction->addItem(tr("Squared Euclidean distance"));

    LayeredConfiguration &config = BasicApplication::instance().config();

    _ui.cbWindowFunction->setCurrentIndex(
            config.getInt("browser.processCreation.windowFunction", 0));
    _ui.cbCostFunction->setCurrentIndex(
            config.getInt("browser.processCreation.costFunction", 0));
    _ui.sbWindowSize->setValue(
            config.getInt("browser.processCreation.windowSizeMS", 30));
    _ui.dsbOverlap->setValue(
            config.getDouble("browser.processCreation.overlap", 0.5));
    _ui.sbNumComponents->setValue(
            config.getInt("browser.processCreation.numComponents", 30));
    _ui.sbMaxIterations->setValue(
            config.getInt("browser.processCreation.maxIterations", 500));
    _ui.sbNumThreads->setValue(
            config.getInt("browser.processCreation.numThreads", 1));

    // Call to display parameter widgets correctly.
    on_cbProcessing_currentIndexChanged(0);

    connect(_ui.cbProcessing,
        SIGNAL(currentIndexChanged(int)),
        SLOT(on_cbProcessing_currentIndexChanged(int)));
}


void CreateProcessDialog::on_cbProcessing_currentIndexChanged(int index)
{
    switch (index) {
        case 0: // STFT + NMD
            _ui.lbAlgorithm->setVisible(true);
            _ui.cbCostFunction->setVisible(true);
            _ui.cbCostFunction->setEnabled(true);
            _ui.lbMaxIterations->setVisible(true);
            _ui.sbMaxIterations->setVisible(true);
            _ui.sbMaxIterations->setEnabled(true);
            _ui.lbNumComponents->setVisible(true);
            _ui.sbNumComponents->setVisible(true);
            _ui.sbNumComponents->setEnabled(true);
            _ui.lbNumSpectra->setVisible(true);
            _ui.sbNumSpectra->setVisible(true);
            _ui.sbNumSpectra->setEnabled(true);
            break;
        case 1: // STFT only
            _ui.lbAlgorithm->setVisible(false);
            _ui.cbCostFunction->setVisible(false);
            _ui.lbMaxIterations->setVisible(false);
            _ui.sbMaxIterations->setVisible(false);
            _ui.lbNumComponents->setVisible(false);
            _ui.sbNumComponents->setVisible(false);
            _ui.lbNumSpectra->setVisible(false);
            _ui.sbNumSpectra->setVisible(false);
            break;
    }
}


void CreateProcessDialog::accept()
{
    const QStringList fileNames = _ui.filesSelectionWidget->fileNames();
    if (fileNames.isEmpty()) {
        QMessageBox::information(this, windowTitle(),
                tr("No files were selected."));
        return;
    }

    // Determine the desired window function.
    WindowFunction windowFunction;
    switch (_ui.cbWindowFunction->currentIndex()) {
    case 0: windowFunction = SqHannFunction; break;
    case 1: windowFunction = HannFunction; break;
    case 2: windowFunction = HammingFunction; break;
    case 3: windowFunction = RectangleFunction; break;
    default: throw Poco::NotImplementedException("Unknown window function.");
    }

    // Determine the desired cost function.
    nmf::Deconvolver::NMFCostFunction cf;
    switch (_ui.cbCostFunction->currentIndex()) {
    case 0: 
        cf = nmf::Deconvolver::KLDivergence; 
        break;
    case 1: 
        cf = nmf::Deconvolver::EuclideanDistance; 
        break;
    default:
        cf = nmf::Deconvolver::KLDivergence;
        break;
    }

    // Set the number of threads.
    const unsigned int numThreads = _ui.sbNumThreads->value();
    setNumThreads(numThreads);

    // Since it's possible that the user first cancels and then restarts the
    // process, the list of filenames during whose processing errors have
    // occured must be explictly cleared now.
    _failedFileNames.clear();

    // Create and start one task per file.
    foreach (const QString &fileName, fileNames) {
        // TODO: Data kind, ICA, other options (preemphasis etc.)
        FTTaskPtr task;
        // STFT + NMD
        if (_ui.cbProcessing->currentIndex() == 0) {
            task = new NMDTask(fileName.toStdString(),
                cf,
                _ui.sbNumComponents->value(),
                _ui.sbNumSpectra->value(),
                _ui.sbMaxIterations->value(),
                0,
                false);
        }
        // STFT only
        else {
            task = new FTTask("FT", fileName.toStdString());
        }
        task->setWindowFunction(windowFunction);
        task->setWindowSize(_ui.sbWindowSize->value());
        task->setOverlap(_ui.dsbOverlap->value());

        addTask(task);
    }

    // Wait for completion.
    const bool cancelled = waitForCompletion("Processing files...");

    // Check if something went horribly wrong.
    if (!_failedFileNames.empty())
        showFailedFileNames();

    if (!cancelled)
        QDialog::accept();
}


void CreateProcessDialog::showFailedFileNames()
{
    QDialog dlg(this);
    dlg.setLayout(new QVBoxLayout());
    QLabel *l =
        new QLabel(tr("The were errors while processing the following files:"));
    dlg.layout()->addWidget(l);

    QListWidget *lw = new QListWidget;
    lw->setAlternatingRowColors(true);
    lw->setSortingEnabled(true);
    for (vector<string>::const_iterator it = _failedFileNames.begin();
        it != _failedFileNames.end(); ++it)
    {
        lw->addItem(QString::fromStdString(*it));
    }
    dlg.layout()->addWidget(lw);

    QPushButton *cb = new QPushButton(tr("Close"));
    connect(cb, SIGNAL(clicked()), &dlg, SLOT(accept()));
    dlg.layout()->addWidget(cb);

    dlg.exec();
}


void CreateProcessDialog::removeTask(BasicTaskPtr task)
{
    ThreadedDialog::removeTask(task);

    if (task->state() == BasicTask::TASK_CANCELLED ||
        task->state() == BasicTask::TASK_FAILED)
    {
        _genMutex.lock();
        _failedFileNames.push_back(task.cast<FTTask>()->fileName());
        _genMutex.unlock();
    }
}


} // namespace blissart
