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


#include "PreferencesDlg.h"
#include <blissart/BasicApplication.h>
#include <Poco/Exception.h>


using Poco::Util::LayeredConfiguration;


namespace blissart {


PreferencesDlg::PreferencesDlg(QWidget *parent) : QDialog(parent)
{
    _ui.setupUi(this);

    // Do NOT change the order of the following items:
    _ui.cbProcessCreationWindowFunction->addItem(tr("Square root of Hann function"));
    _ui.cbProcessCreationWindowFunction->addItem(tr("Hann function"));
    _ui.cbProcessCreationWindowFunction->addItem(tr("Rectangle function"));

    // Do NOT change the order of the following items:
    _ui.cbCostFunction->addItem(tr("Extended KL divergence"));
    _ui.cbCostFunction->addItem(tr("Squared Euclidean distance"));

    setupConfigMap();
    getConfig();
}


void PreferencesDlg::accept()
{
    setConfig();
    QDialog::accept();
}


void PreferencesDlg::setupConfigMap()
{
    // Preview
    _configMap.insert(_ui.cbPreviewAlwaysEnabled,
                      "browser.preview.alwaysEnabled");
    _configMap.insert(_ui.cbPreviewNormalizeAudio,
                      "browser.preview.normalizeAudio");

    // Process creation
    _configMap.insert(_ui.cbProcessCreationWindowFunction,
                      "browser.processCreation.windowFunction");
    _configMap.insert(_ui.sbProcessCreationWindowSize,
                      "browser.processCreation.windowSizeMS");
    _configMap.insert(_ui.cbCostFunction,
                      "browser.processCreation.costFunction");
    _configMap.insert(_ui.dbsProcessCreationOverlap,
                      "browser.processCreation.overlap");
    _configMap.insert(_ui.sbProcessCreationNumComponents,
                      "browser.processCreation.numComponents");
    _configMap.insert(_ui.sbProcessCreationMaxIterations,
                      "browser.processCreation.maxIterations");
    _configMap.insert(_ui.sbProcessCreationNumThreads,
                      "browser.processCreation.numThreads");

    // Feature extraction
    _configMap.insert(_ui.sbFeatureExtractionNumThreads,
                      "browser.featureExtraction.numThreads");
}


void PreferencesDlg::setConfig()
{
    LayeredConfiguration &config = BasicApplication::instance().config();

    // Iterate over the _configMap and set the widgets' values to whatever the
    // configuration says depending on the respective widget type.
    QMutableMapIterator<QWidget *, const char *> kvp(_configMap);
    while (kvp.hasNext()) {
        kvp.next();
        if (kvp.key()->inherits("QCheckBox")) {
            QCheckBox *cb = static_cast<QCheckBox*>(kvp.key());
            config.setBool(kvp.value(), cb->isChecked());
        }
        else if (kvp.key()->inherits("QComboBox")) {
            QComboBox *cb = static_cast<QComboBox*>(kvp.key());
            config.setInt(kvp.value(), cb->currentIndex());
        }
        else if (kvp.key()->inherits("QSpinBox")) {
            QSpinBox *sb = static_cast<QSpinBox*>(kvp.key());
            config.setInt(kvp.value(), sb->value());
        }
        else if (kvp.key()->inherits("QDoubleSpinBox")) {
            QDoubleSpinBox *dsb = static_cast<QDoubleSpinBox*>(kvp.key());
            config.setDouble(kvp.value(), dsb->value());
        }
        else {
            throw Poco::NotImplementedException("Unknown configuration widget.");
        }
    }
}


void PreferencesDlg::getConfig()
{
    LayeredConfiguration &config = BasicApplication::instance().config();

    // Iterate over the _configMap and set the widgets' values to whatever the
    // configuration says depending on the respective widget type.
    QMutableMapIterator<QWidget *, const char *> kvp(_configMap);
    while (kvp.hasNext()) {
        kvp.next();
        if (kvp.key()->inherits("QCheckBox")) {
            QCheckBox *cb = static_cast<QCheckBox*>(kvp.key());
            cb->setChecked(config.getBool(kvp.value(), true));
        }
        else if (kvp.key()->inherits("QComboBox")) {
            QComboBox *cb = static_cast<QComboBox*>(kvp.key());
            cb->setCurrentIndex(config.getInt(kvp.value(), 0));
        }
        else if (kvp.key()->inherits("QSpinBox")) {
            QSpinBox *sb = static_cast<QSpinBox*>(kvp.key());
            sb->setValue(config.getInt(kvp.value(), sb->minimum()));
        }
        else if (kvp.key()->inherits("QDoubleSpinBox")) {
            QDoubleSpinBox *dsb = static_cast<QDoubleSpinBox*>(kvp.key());
            dsb->setValue(config.getDouble(kvp.value(), dsb->minimum()));
        }
        else {
            throw Poco::NotImplementedException("Unknown configuration widget.");
        }
    }
}


} // namespace blissart
