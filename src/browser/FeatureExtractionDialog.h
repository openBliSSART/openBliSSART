//
// $Id: FeatureExtractionDialog.h 855 2009-06-09 16:15:50Z alex $
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


#ifndef __FEATUREEXTRACTIONDIALOG_H__
#define __FEATUREEXTRACTIONDIALOG_H__


#include "ui_FeatureExtractionDialog.h"
#include "ThreadedDialog.h"

#include <blissart/DataDescriptor.h>

#include <set>


namespace blissart {


class FeatureExtractionDialog : public ThreadedDialog
{
    Q_OBJECT

public:
    /**
     * Constructs a new FeatureExtractionDialog for the given set of data
     * descriptors.
     * @param  descriptors  the set of data descriptors for which the features
     *                      should be extracted
     * @throw               Poco::InvalidArgumentException
     */
    FeatureExtractionDialog(const std::set<DataDescriptorPtr> &descriptors,
                            QWidget *parent = 0);


public slots:
    void accept();


private:
    Ui::FeatureExtractionDialog   _ui;
    std::set<DataDescriptorPtr>   _descriptors;
};


} // namespace blissart


#endif
