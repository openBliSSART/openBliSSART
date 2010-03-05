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


#include "FeatureExtractionDialog.h"

#include <blissart/FeatureExtractionTask.h>

#include <Poco/Util/LayeredConfiguration.h>

#include <QMessageBox>


using namespace std;
using Poco::Util::LayeredConfiguration;


namespace blissart {


FeatureExtractionDialog::FeatureExtractionDialog(
        const set<DataDescriptorPtr> &descriptors, QWidget *parent) :
    ThreadedDialog(parent),
    _descriptors(descriptors)
{
    if (_descriptors.empty())
        throw Poco::InvalidArgumentException("No data descriptors provided.");

    _ui.setupUi(this);

    LayeredConfiguration &config = BasicApplication::instance().config();
    _ui.sbNumThreads->setValue(
            config.getInt("browser.featureExtraction.numThreads", 1));
}


void FeatureExtractionDialog::accept()
{
    const unsigned int numThreads = _ui.sbNumThreads->value();

    // Set the number of threads and create the corresponding tasks.
    setNumThreads(numThreads);
    vector<BasicTaskPtr> tasks;
    for (unsigned int i = 0; i < numThreads; i++)
        tasks.push_back(new FeatureExtractionTask);

    // Now distribute the data descriptors among the tasks.
    int taskNr = 0;
    foreach (const DataDescriptorPtr &dPtr, _descriptors) {
        FeatureExtractionTask *fexTask =
            tasks.at(taskNr % tasks.size()).cast<FeatureExtractionTask>();
        fexTask->addDataDescriptor(dPtr);
        taskNr++;
    }

    // Hide ourselves and start the feature extraction tasks.
    hide();
    resetProgress();
    foreach (const BasicTaskPtr &tsk, tasks)
        addTask(tsk);

    // Wait for completion.
    const bool cancelled = waitForCompletion("Extracting features...");

    if (!cancelled)
        QDialog::accept();
}


} // namespace blissart
