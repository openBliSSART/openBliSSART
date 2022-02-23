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


#ifndef __BLISSART_DATABASESUBSYSTEM_H__
#define __BLISSART_DATABASESUBSYSTEM_H__


#include <string>
#include <map>

#include <common.h>
#include <blissart/DataDescriptor.h>
#include <blissart/Process.h>
#include <blissart/Feature.h>
#include <blissart/ClassificationObject.h>
#include <blissart/Response.h>
#include <blissart/Label.h>
#include <blissart/DataSet.h>

#include <Poco/Logger.h>
#include <Poco/Mutex.h>
#include <Poco/RWLock.h>
#include <Poco/Data/SessionPool.h>
#include <Poco/Util/Subsystem.h>


namespace blissart {


class FeatureSet;


/**
 * \addtogroup framework
 * @{
 */

/**
 * Provides functions for storing objects in a SQLite database.
 */
class LibFramework_API DatabaseSubsystem : public Poco::Util::Subsystem
{
public:
    /**
     * Default constructor. Registers the SQLite connector.
     */
    DatabaseSubsystem();

    /**
     * Destroys the DatabaseSubsystem. Unregisters the SQLite connector.
     */
    virtual ~DatabaseSubsystem();

    /**
     * Attaches the given database file.
     */
    void connect(const std::string& dbFilename);

    /**
     * Detaches the current database file.
     */
    void disconnect();

    /**
     * Detaches the current database file and deletes it.
     */
    void destroy();

    /**
     * @name Process
     * @{
     */

    /**
     * Creates a new Process object in the database.
     */
    void createProcess(ProcessPtr process);

    /**
     * Updates the given Process object.
     */
    void updateProcess(ProcessPtr process);

    /**
     * Removes the given Process object.
     */
    void removeProcess(ProcessPtr process);

    /**
     * Retrieves the Process object with the given ID.
     */
    ProcessPtr getProcess(int processID);

    /**
     * Retrieves the Process shared by the data descriptors associated with the
     * given ClassificationObject.
     */
    ProcessPtr getProcess(ClassificationObjectPtr clo);

    /**
     * Gets all processes.
     */
    std::vector<ProcessPtr> getProcesses();

    /**
     * @}
     */

    /**
     * @name DataDescriptor
     * @{
     */

    /**
     * Creates a new DataDescriptor object in the database.
     */
    void createDataDescriptor(DataDescriptorPtr data);

    /**
     * Updates the given DataDescriptor object.
     */
    void updateDataDescriptor(DataDescriptorPtr data);

    /**
     * Removes the given DataDescriptor object.
     */
    void removeDataDescriptor(DataDescriptorPtr data);

    /**
     * Retrieves the DataDescriptor object with the given ID.
     */
    DataDescriptorPtr getDataDescriptor(int descrID);

    /**
     * Retrieves the DataDescriptor object for the given process ID,
     * type and indices.
     */
    DataDescriptorPtr getDataDescriptor(int processID,
                                        DataDescriptor::Type type,
                                        int index,
                                        int index2 = 0);

    /**
     * Gets all data descriptors associated with a particular process.
     */
    std::vector<DataDescriptorPtr> getDataDescriptors(int processID);

    /**
     * Gets all data descriptors related to a particular classification object.
     */
    std::vector<DataDescriptorPtr> getDataDescriptors(ClassificationObjectPtr clo);

    /**
     * @}
     */

    /**
     * @name Feature
     * @{
     */

    /**
     * Saves the given Feature object in the database. If the corresponding
     * record does not exist, it is created, otherwise it is updated.
     */
    void saveFeature(FeaturePtr feature);

    /**
     * Saves the given Feature objects in the database, using a single
     * transaction for improved performance when doing multiple inserts.
     */
    void saveFeatures(const std::vector<FeaturePtr>& features);

    /**
     * Removes the given Feature object.
     */
    void removeFeature(FeaturePtr feature);

    /**
     * Removes all Feature objects associated with a DataDescriptor,
     * given by its ID.
     */
    void removeFeatures(int descrID);

    /**
     * Retrieves a specific unparameterized feature associated with the data
     * given by a DataDescriptor ID.
     */
    FeaturePtr getFeature(int descrID, const std::string& featureName,
        double param1 = 0.0, double param2 = 0.0, double param3 = 0.0);

    /**
     * Gets all features associated with the data given by a
     * DataDescriptor ID.
     */
    std::vector<FeaturePtr> getFeatures(int descrID);

    /**
     * @}
     */

    /**
     * @name ClassificationObject
     * @{
     */

    /**
     * Creates a new ClassificationObject in the database.
     */
    void createClassificationObject(ClassificationObjectPtr clObj);

    /**
     * Updates the given ClassificationObject.
     */
    void updateClassificationObject(ClassificationObjectPtr clObj);

    /**
     * Updates all ClassificationObjects in the given vector.
     */
    void updateClassificationObjects(
        const std::vector<ClassificationObjectPtr>& clObjs);

    /**
     * Removes the given ClassificationObject.
     */
    void removeClassificationObject(ClassificationObjectPtr clObj);

    /**
     * Retrieves a ClassificationObject by its unique ID.
     */
    ClassificationObjectPtr getClassificationObject(int clObjID);

    /**
     * Gets all classification objects.
     */
    std::vector<ClassificationObjectPtr> getClassificationObjects();

    /**
     * Gets all classification objects which are related to the label specified
     * by the given label id.
     */
    std::vector<ClassificationObjectPtr>
        getClassificationObjectsForLabel(int labelID);

    /**
     * Gets a map of all classification objects and their associated labels
     * for the given response.
     */
    std::vector<std::pair<ClassificationObjectPtr, LabelPtr> >
        getClassificationObjectsAndLabelsForResponse(ResponsePtr r);

    /**
     * Gets all classification objects belonging to the given file name.
     */
    std::vector<ClassificationObjectPtr>
        getClassificationObjectsByFilename(const std::string& filename);

    /**
     * @}
     */

    /**
     * @name Response
     * @{
     */

    /**
     * Creates a new Response object in the database.
     */
    void createResponse(ResponsePtr response);

    /**
     * Updates the given Response object.
     */
    void updateResponse(ResponsePtr response);

    /**
     * Removes the given Response object.
     */
    void removeResponse(ResponsePtr response);

    /**
     * Retrieves a Reponse object given by its unique ID.
     */
    ResponsePtr getResponse(int responseID);

    /**
     * Gets all responses.
     */
    std::vector<ResponsePtr> getResponses();

    /**
     * @}
     */

    /**
     * @name DataSet
     * @{
     */

    /**
     * Retrieves a DataSet for the given response that contains all available
     * features.
     */
    DataSet getDataSet(ResponsePtr response);

    /**
     * Retrieves a DataSet for the given response that contains the features
     * given by a FeatureSet.
     */
    DataSet getDataSet(ResponsePtr response, const FeatureSet& featureSet);

    /**
     * Retrieves all features associated with the given pairs of classification
     * objects and labels and returns a corresponding DataSet.
     */
    DataSet getDataSet(const std::map<int, int>& cloLabelsMap);

    /**
     * @}
     */

    /**
     * @name Label
     * @{
     */

    /**
     * Creates a new label in the database.
     */
    void createLabel(LabelPtr label);


    /**
     * Updates a label in the database.
     */
    void updateLabel(LabelPtr label);


    /**
     * Removes a label from the database.
     */
    void removeLabel(LabelPtr label);


    /**
     * Retrieves the label associated with the given id.
     */
    LabelPtr getLabel(int labelID);


    /**
     * Gets all labels.
     */
    std::vector<LabelPtr> getLabels();

    /**
     * Gets the label(s) with the given text.
     */
    std::vector<LabelPtr> getLabelsByText(const std::string& text);

    /**
     * Gets all labels related to the given response.
     */
    std::vector<LabelPtr> getLabelsForResponse(ResponsePtr r);

    /**
     * @}
     */


protected:
    /**
     * Initialization of the database subsystem.
     */
    virtual void initialize(Poco::Util::Application& app);

    /**
     * Uninitialization of the database subsystem.
     */
    virtual void uninitialize();

    /**
     * Returns the name of this subsystem.
     */
    virtual const char* name() const;

    /**
     * Further setup of the database subsystem, i.e. ensure all
     * tables/relations exist.
     */
    void setup();

    /**
     * Creates the trigger functions that are neccessary to enforce SQLite's
     * FOREIGN KEY constraints.
     */
    void setupTriggers();


    /**
     * Returns the id of the last inserted table row for the given session.
     */
    int lastInsertID(Poco::Data::Session& session);


    /**
     * Returns a Session object from the SessionPool.
     */
    inline Poco::Data::Session getSession();


    /**
     * @name Helper functions
     * NOTE: All of the helper functions assume that the _dbLock has been
     * already aquired!
     * @{
     */


    void insertProcessParams(Poco::Data::Session& session,
                             ProcessPtr process);


    void getProcessParams(Poco::Data::Session &session, ProcessPtr process);


    void insertClassificationObjectDescrIDs(Poco::Data::Session& session,
                                            ClassificationObjectPtr& clObj);


    void insertClassificationObjectLabelIDs(Poco::Data::Session& session,
                                            ClassificationObjectPtr& clObj);


    void getClassificationObjectDescrIDs(Poco::Data::Session &session,
                                         ClassificationObjectPtr& clObj);


    void getClassificationObjectLabelIDs(Poco::Data::Session &session,
                                         ClassificationObjectPtr& clObj);


    void insertResponseLabels(Poco::Data::Session& session,
                              ResponsePtr& response);


    void getResponseLabels(Poco::Data::Session& session, ResponsePtr& response);


    void saveFeature(Poco::Data::Session& session, FeaturePtr feature);


    void getAvailableFeatures(Poco::Data::Session& session,
                              const std::map<int, int>& clObjMap,
                              FeatureSet& featureSet);


    /**
     * @}
     */


    Poco::RWLock             _dbLock;


private:
    // Forbid the copy constructor and operator=.
    DatabaseSubsystem(const DatabaseSubsystem&);
    DatabaseSubsystem& operator=(const DatabaseSubsystem&);


    std::string              _dbFilename;
    Poco::Data::SessionPool* _pPool;
    Poco::FastMutex          _poolLock;
    Poco::Logger&            _logger;
};


/**
 * @}
 */


//
// Inlines
//


inline Poco::Data::Session DatabaseSubsystem::getSession()
{
    Poco::FastMutex::ScopedLock lock(_poolLock);
    poco_check_ptr(_pPool);
    return _pPool->get();
}


} // namespace blissart


#endif // __BLISSART_DATABASESUBSYSTEM_H__
