//
// $Id: NMDGainsTask.h 889 2009-07-01 16:12:26Z felix $
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


#ifndef __NMD_GAINS_TASK_H__
#define __NMD_GAINS_TASK_H__


#include <common.h>
#include <blissart/FTTask.h>
#include <blissart/TargetedDeconvolver.h>
#include <blissart/ProgressObserver.h>


/**
 * The type of task performed by NMDTool.
 */
class NMDGainsTask : public blissart::FTTask, blissart::ProgressObserver
{
public:
    typedef enum {
        NoTransformation,
        UnitSum,
        LogDCT,
        MaximalIndices
    } TransformationMethod;

    NMDGainsTask(const std::string &fileName,
        int nrOfComponents, int maxIterations,
        const std::vector<blissart::ClassificationObjectPtr>& initObjects,
        bool allComponents,
        TransformationMethod method);

    virtual ~NMDGainsTask();

    virtual void runTask();

    virtual void progressChanged(float progress);

    void setIndexCount(int count);

private:
    void performNMD();

    /**
     * Stores the calculated feature matrices (gains, MFCC) in the database.
     * Overrides FTTask method.
     */
    void storeComponents() const;

    std::vector<blissart::ClassificationObjectPtr> 
                                    _initObjects;
    blissart::TargetedDeconvolver*  _deconvolver;
    int                             _nrOfComponents;
    int                             _maxIterations;

    bool                            _allComponents;
    blissart::linalg::Matrix*       _gainsMatrix;
    TransformationMethod            _transformation;

    int                             _myUniqueID;
};


#endif // __NMD_GAINS_TASK_H__
