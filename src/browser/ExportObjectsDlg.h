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


#ifndef __EXPORTCOMPONENTSDLG_H__
#define __EXPORTCOMPONENTSDLG_H__


#include <QFileDialog>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>


namespace blissart {


/**
 * \addtogroup browser
 * @{
 */

/**
 * A dialog fitted to the needs of exporting lots of classification objects.
 */
class ExportObjectsDlg : public QFileDialog
{
public:
    /**
     * Constructs a new instance of ExportObjectsDlg for the given list
     * of classification objects.
     */
    ExportObjectsDlg(const std::vector<ClassificationObjectPtr> &clos,
                        QWidget *parent = 0);


    /**
     * Performs the actual export process.
     */
    virtual void accept();


protected:
    /**
     * Builds an audio file from the given classification object and related
     * data. Then exports it to the given directory using a name constructed of
     * the associated process' input file and the component's id.
     * @param  clo          a pointer to a ClassificationObject
     * @param  destDir      the destination directory
     * @return              true iff no errors occured during the export
     */
    bool exportClassificationObject(const ClassificationObjectPtr clo,
                                    const QDir &destDir);


    /**
     * Returns a copy of the underlying list of classification objects.
     */
    inline std::vector<ClassificationObjectPtr> getClassificationObjects() const
    { return _clos; }


private:
    std::vector<ClassificationObjectPtr> _clos;
    Poco::Logger                         &_logger;
};


/**
 * @}
 */
    

} // namespace blissart


#endif
