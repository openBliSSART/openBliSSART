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


#include "FilesSelectionWidget.h"
#include <QFileDialog>


namespace blissart {


FilesSelectionWidget::FilesSelectionWidget(QWidget *parent) : QWidget(parent)
{
    _ui.setupUi(this);
}


QStringList FilesSelectionWidget::fileNames() const
{
    QList<QString> result;
    
    for (int i = 0; i < _ui.lwFiles->count(); i++)
        result.append(_ui.lwFiles->item(i)->text());
    
    return result;
}


void FilesSelectionWidget::on_pbAddFiles_clicked()
{
    QStringList files =
        QFileDialog::getOpenFileNames(this, tr("Select one or more files"),
                QString(), "Audio files (*.mp3 *.wav)");
    
    foreach (QString fileName, files) {
        bool found = false;
        for (int i = 0; i < _ui.lwFiles->count(); i++) {
            if (_ui.lwFiles->item(i)->text() == fileName) {
                found = true;
                break;
            }
        }
        if (!found)
            _ui.lwFiles->addItem(fileName);
    }
}


void FilesSelectionWidget::on_pbRemoveSelected_clicked()
{
    foreach (QListWidgetItem *item, _ui.lwFiles->selectedItems()) {
        // XXX: _ui.lwFiles->removeItem() fails for some unknown reasons?!
        delete _ui.lwFiles->takeItem(_ui.lwFiles->row(item));
    }
}


} // namespace blissart
