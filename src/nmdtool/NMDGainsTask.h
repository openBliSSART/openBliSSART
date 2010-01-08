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
 * Computes NMD activations (or "gains") using a given set of NMD components
 * from the database, and stores them in the database or HTK files.
 */
class NMDGainsTask : public blissart::FTTask, blissart::ProgressObserver
{
public:
    /**
     * TODO: Document or drop me.
     */
    typedef enum {
        NoTransformation,
        UnitSum,
        LogDCT,
        MaximalIndices
    } TransformationMethod;

    /**
     * Default constructor. Takes a variety of arguments concerning the actal
     * NMD parameters.
     * @param	fileName	input file (e.g. a WAV file)
     * @param	nrOfComponents	number of NMD components - if it is greater
     *							than the number of initialized objects, 
     *							additional "garbage" components are introduced.
     * @param	allComponents	whether to store the activations of the
     *							"garbage" components too
     * @param	method			a TransformationMethod for the activations
     */
    NMDGainsTask(const std::string &fileName,
        int nrOfComponents, int maxIterations,
        const std::vector<blissart::ClassificationObjectPtr>& initObjects,
        bool allComponents,
        TransformationMethod method = NoTransformation);

    /**
     * Destroys the NMDGainsTask and frees the memory allocated for NMD.
     */
    virtual ~NMDGainsTask();

    /**
     * Implementation of BasicTask interface; overrides FTTask method.
     */
    virtual void runTask();

    /**
     * Implementation of ProgressObserver interface.
     */
    virtual void progressChanged(float progress);

    /**
     * TODO: Document or drop me.
     */
    void setIndexCount(int count);

    /**
     * Sets a flag controlling whether to export activations to HTK files, 
     * or store them in the DB.
     */
    inline void setExport(bool flag);

    /**
     * Sets the directory where HTK files should be exported.
     */
    inline void setExportDir(const std::string& dir);

private:
    /**
     * Performs initialized NMD on the input file.
     */
    void performNMD();

    /**
     * Stores the calculated feature matrices (gains, MFCC) in the database.
     * Overrides FTTask method.
     */
    void storeComponents() const;

    /**
     * Exports the computed NMD activation matrix as an HTK file.
     */
    void exportHTKFile() const;

    std::vector<blissart::ClassificationObjectPtr> 
                                    _initObjects;
    blissart::TargetedDeconvolver*  _deconvolver;
    int                             _nrOfComponents;
    int                             _maxIterations;

    bool                            _allComponents;
    blissart::linalg::Matrix*       _gainsMatrix;
    TransformationMethod            _transformation;

    bool                            _export;
    std::string                     _exportDir;

    int                             _myUniqueID;
};


void NMDGainsTask::setExport(bool flag)
{
    _export = flag;
}


void NMDGainsTask::setExportDir(const std::string &dir)
{
    _exportDir = dir;
}


#endif // __NMD_GAINS_TASK_H__
