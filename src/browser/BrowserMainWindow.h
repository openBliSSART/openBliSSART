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


#ifndef __BROWSERMAINWINDOW_H__
#define __BROWSERMAINWINDOW_H__


#include <QMainWindow>


namespace blissart {


/**
 * Represents the main window of the browser GUI, including the menu bar.
 */
class BrowserMainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    /**
     * Constructs a new instance of BrowserMainWindow. Sets up the menubar and
     * restores any mainwindow related settings.
     */
    BrowserMainWindow();

    
protected slots:
    /**
     * Displays a Preferences dialog.
     */
    void handlePreferences();
    
    
    /**
     * Displays a FeatureExtractionDialog for all data descriptors.
     */
    void handleFeatureExtraction();
    
    
protected:
    /**
     * Event-handler for close-events. Stores mainwindow related settings.
     */
    virtual void closeEvent(QCloseEvent *ev);


private:
    /**
     * Adds a nice & small menu to this widget.
     */
    void setupMenu();
};


} // namespace blissart


#endif
