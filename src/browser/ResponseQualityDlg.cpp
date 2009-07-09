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


#include "ResponseQualityDlg.h"
#include <blissart/linalg/Matrix.h>
#include <blissart/linalg/ColVector.h>
#include <blissart/linalg/generators/generators.h>
#include <blissart/exportDataSet.h>

#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>

#include <QDateTime>
#include <QFileDialog>
#include <QHeaderView>
#include <QMessageBox>
#include <QProgressDialog>


using namespace std;
using namespace blissart::linalg;
using blissart::exportDataSet;


namespace blissart {


ResponseQualityDlg::ResponseQualityDlg(const DataSet &dataSet, 
                                       ResponsePtr response,
                                       QWidget *parent) :
    QDialog(parent),
    _dataSet(dataSet),
    _dpMatrix(NULL),
    _response(response)
{
    _ui.setupUi(this);

    initialize();
    setupTreeWidget();
}


ResponseQualityDlg::~ResponseQualityDlg()
{
    if (_dpMatrix) {
        delete _dpMatrix;
        _dpMatrix = NULL;
    }
}


void ResponseQualityDlg::on_pbSave_clicked()
{
    QString fileName = 
        QFileDialog::getSaveFileName(this, tr("Save as..."),
                                     "", tr("Weka files (*.arff)"));
    if (fileName.isEmpty()) {
        return;
    }
    
    // XXX: Provide the user with progress information?
    exportDataSet(_dataSet, fileName.toStdString(), 
                  _response->name, _response->description);
}


void ResponseQualityDlg::initialize()
{
    // Determine the distinct features, i.e. build a map of descriptor types and
    // feature names. Create an explicit index for every distinct feature.
    // Also count the classes.
    DescrReverseIndices descrReverseIndices;
    DescrFeatures features;
    int nextFeatureIndex = 0;
    unsigned int classCount = 1;
    foreach (const DataPoint &dp, _dataSet) {
        // Is this a "new" class?
        if (_classLabels.find(dp.classLabel) == _classLabels.end()) {
            // This is the first occurence of this particular class,
            // hence we give it an extra "identifier" (this is only
            // neccessary for the Weka export).
            _classLabels[dp.classLabel] = classCount++;
        }
        // Iterate over all features.
        for (DataPoint::ComponentMap::const_iterator cItr = dp.components.begin();
            cItr != dp.components.end(); cItr++) {
            const FeatureDescriptor &fdesc = cItr->first;
            // See if this feature has already been observed.
            FeatureSet &featuresForType = features[fdesc.dataType];
            if (featuresForType.find(fdesc.name) == featuresForType.end()) {
                // No, this is the first time we see this feature.
                // Insert it into the relevant set.
                featuresForType.insert(fdesc.name);
                // Assign a feature index.
                IndexMap &im = _descrIndices[fdesc.dataType];
                im[nextFeatureIndex] = fdesc.name;
                ReverseIndexMap &rim = descrReverseIndices[fdesc.dataType];
                rim[fdesc.name] = nextFeatureIndex;
                nextFeatureIndex++;
            }
        }
    }
    // At this point nextFeatureIndex resembles the total number of distinct
    // features.
    if (nextFeatureIndex <= 0)
        throw Poco::InvalidArgumentException("Empty dataset.");
    
    // FIXME: Since the DataSet is sparse, all non-existing features must be
    // replaced with their expected values in the following matrix!

    // Create a matrix from the data points and count the features.
    _dpMatrix = new Matrix(nextFeatureIndex + 1, _dataSet.size(), generators::zero);
    unsigned int col = 0;
    foreach (const DataPoint &dp, _dataSet) {
        for (DataPoint::ComponentMap::const_iterator cItr = dp.components.begin();
            cItr != dp.components.end(); cItr++)
        {
            ReverseIndexMap &rim = descrReverseIndices[cItr->first.dataType];
            int row = rim[cItr->first.name];
            _dpMatrix->setAt(row, col, cItr->second);
            _featureCount[row] += 1;
        }
        // The last row always contains the class label.
        _dpMatrix->setAt(_dpMatrix->rows() - 1, col, dp.classLabel);
        col++;
    }
}


void ResponseQualityDlg::setupTreeWidget()
{
    debug_assert(_dpMatrix);
    
    // Calculate the expected values and variances.
    ColVector ev = _dpMatrix->meanColumnVector();
    ColVector var = _dpMatrix->varianceRows();

    // Fill the tree widget.
    _ui.twFeatures->setSortingEnabled(false);
    for (DescrIndices::const_iterator it = _descrIndices.begin(); 
        it != _descrIndices.end(); it++)
    {
        // First create an item group for this particular data descriptor type.
        string typeAsString = DataDescriptor::strForType(it->first);
        QTreeWidgetItem *gi = new QTreeWidgetItem(_ui.twFeatures);
        gi->setText(0, QString::fromStdString(typeAsString));
        gi->setFirstColumnSpanned(true);

        // Then fill this group with the relevant data.
        for (IndexMap::const_iterator indexIt = it->second.begin();
            indexIt != it->second.end(); indexIt++) {
            QTreeWidgetItem *item = new QTreeWidgetItem(gi);
            item->setText(0, QString::fromStdString(indexIt->second));
            item->setText(1, QString::number(ev(indexIt->first), 'g', 4));
            item->setText(2, QString::number(var(indexIt->first), 'g', 4));
            item->setText(3, QString::number(_featureCount[indexIt->first]));
        }
    }
    _ui.twFeatures->expandAll();
    _ui.twFeatures->header()->resizeSections(QHeaderView::ResizeToContents);
    _ui.twFeatures->sortByColumn(0, Qt::AscendingOrder);
    _ui.twFeatures->setSortingEnabled(true);
}


} // namespace blissart
