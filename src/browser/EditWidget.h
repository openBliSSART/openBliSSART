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


#ifndef __EDITWIDGET_H__
#define __EDITWIDGET_H__


#include <QWidget>
#include <blissart/BasicApplication.h>
#include <blissart/DatabaseSubsystem.h>
#include <blissart/StorageSubsystem.h>


namespace blissart {


class EditWidget : public QWidget
{
    Q_OBJECT

public:
    /**
     * Returns if there are any unsaved changes.
     */
    inline bool isDirty() const { return _dirty; }
    
    
signals:
    /**
     * This signal is emitted whenever the underlying entity is updated in
     * the database.
     */
    void entityUpdated(DatabaseEntityPtr);
    
    
    /**
     * This signal is emitted whenever the dirty-flag changes.
     */
    void entityDirty(bool yesNo);


public slots:
	/**
	 * Handles "save"-events. The default implementation emits the
	 * entityUpdated-signal and resets the dirty-flag.
	 */
	virtual void handleSave();
	
	
	/**
	 * Handles "cancel"-events. The default implementation resets the
	 * dirty-flag.
	 */
	virtual void handleCancel();
	

protected slots:
    /**
     * Handles edit-related events from widgets like (line|text)-edits and
     * alike. Subclasses must not forget to connect the related signals to this
     * slot!
     */
    void setDirty(bool dirty = true);

    
protected:
    // This class is should not be directly instantiated.
    EditWidget(QWidget *parent = 0);
    virtual ~EditWidget();

    
    /**
     * Subclasses must override this method and return a pointer to their
     * associated database entity.
     */
    virtual DatabaseEntityPtr entityPtr() const = 0;


    /**
     * Returns a reference to the database subsystem.
     */
    inline DatabaseSubsystem& dbSubsystem() const;
    
    
    /**
     * Returns a reference to the storage subsystem.
     */
    inline StorageSubsystem& stSubsystem() const;
    
    
private:
    bool _dirty;
};


// Inlines


DatabaseSubsystem& EditWidget::dbSubsystem() const
{
    return BasicApplication::instance().getSubsystem<DatabaseSubsystem>();
}


StorageSubsystem& EditWidget::stSubsystem() const
{
    return BasicApplication::instance().getSubsystem<StorageSubsystem>();
}


} // namespace blissart


#endif // EDITWIDGET_H
