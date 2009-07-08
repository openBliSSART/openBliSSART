//
// $Id: ResponseQualityDlg.h 855 2009-06-09 16:15:50Z alex $
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


#ifndef __RESPONSEQUALITYDLG_H__
#define __RESPONSEQUALITYDLG_H__


#include "ui_ResponseQualityDlg.h"
#include <blissart/DataSet.h>
#include <blissart/Response.h>
#include <set>


namespace blissart {


// Forward declaration
namespace linalg { class Matrix; }


class ResponseQualityDlg : public QDialog
{
    Q_OBJECT
    
public:
    /**
     * Constructs an instance of ResponseQualityDlg for the given dataset.
     * @param   dataSet     a list of DataPoints
     * @param   response    a pointer to the corresponding Response
     * @throw               Poco::InvalidArgumentException
     */
    ResponseQualityDlg(const DataSet &dataSet, ResponsePtr _response,
                       QWidget *parent = 0);

    /**
     * Destructs an instance of ResponseQualityDlg and frees all previously
     * allocated memory.
     */
    virtual ~ResponseQualityDlg();
    
    
protected slots:
    /**
     * Event-handler for the "Save" button.
     */
    void on_pbSave_clicked();
    
    
protected:
    void initialize();
    void setupTreeWidget();
    
    
private:
    typedef std::map<int, std::string> IndexMap;
    typedef std::map<std::string, int> ReverseIndexMap;
    typedef std::set<std::string>      FeatureSet;
    typedef std::map<DataDescriptor::Type, FeatureSet>      DescrFeatures;
    typedef std::map<DataDescriptor::Type, IndexMap>        DescrIndices;
    typedef std::map<DataDescriptor::Type, ReverseIndexMap> DescrReverseIndices;

    
    Ui::ResponseQualityDlg _ui;
    const DataSet&         _dataSet;
    linalg::Matrix*        _dpMatrix;
    std::map<int, int>     _classLabels;
    std::map<int, int>     _featureCount;
    DescrIndices           _descrIndices;
    ResponsePtr            _response;
};


} // namespace blissart


#endif
