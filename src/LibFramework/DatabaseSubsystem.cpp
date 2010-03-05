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


#include "TypeHandler.h"

#include <blissart/DatabaseSubsystem.h>
#include <blissart/FeatureSet.h>

#include <Poco/File.h>
#include <Poco/Data/SQLite/Connector.h>
#include <Poco/Util/Application.h>

#include <iostream>
#include <cassert>
#include <algorithm>
#include <iterator>


using namespace std;
using namespace Poco::Data;
using Poco::FastMutex;
using Poco::RWLock;


namespace blissart {


DatabaseSubsystem::DatabaseSubsystem() :
    _pPool(0),
    _logger(Poco::Logger::get("openBliSSART.DatabaseSubsystem"))
{
    SQLite::Connector::registerConnector();
}


DatabaseSubsystem::~DatabaseSubsystem()
{
    SQLite::Connector::unregisterConnector();
}


const char* DatabaseSubsystem::name() const
{
    return "Database subsystem";
}


void DatabaseSubsystem::connect(const std::string& dbFilename)
{
    _poolLock.lock();
    if (_pPool)
        delete _pPool;
    _pPool = new Poco::Data::SessionPool("SQLite", dbFilename);
    _poolLock.unlock();
    _dbFilename = dbFilename;
    setup();
    setupTriggers();
}


void DatabaseSubsystem::disconnect()
{
    FastMutex::ScopedLock lock(_poolLock);
    if (_pPool) {
        delete _pPool;
        _pPool = 0;
    }
}


void DatabaseSubsystem::initialize(Poco::Util::Application& app)
{
    string dbFilename = app.config().getString("blissart.databaseFile", "openBliSSART.db");
    connect(dbFilename);
}


void DatabaseSubsystem::uninitialize()
{
    disconnect();
}


void DatabaseSubsystem::destroy()
{
    assert(!_pPool);
    Poco::File(_dbFilename).remove();
}


void DatabaseSubsystem::setup()
{
    RWLock::ScopedLock lock(_dbLock, true);

    _logger.debug("Creating tables and indices, if neccessary.");
    Session session = getSession();
    session.begin();
    session <<
        "CREATE TABLE IF NOT EXISTS process ("
        "  process_id     INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,"
        "  process_name   TEXT NOT NULL,"
        "  input_file     TEXT NOT NULL,"
        "  start_time     INTEGER NOT NULL,"
        "  sample_freq    INTEGER NOT NULL"
        ")",
        now;
    session <<
        "CREATE TABLE IF NOT EXISTS process_param ("
        "  process_id     INTEGER NOT NULL REFERENCES process(process_id),"
        "  param_name     TEXT NOT NULL,"
        "  param_value    TEXT NOT NULL,"
        "  PRIMARY KEY(process_id, param_name)"
        ")",
        now;
    session <<
        "CREATE TABLE IF NOT EXISTS data_descriptor ("
        "  descr_id       INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,"
        "  process_id     INTEGER NOT NULL REFERENCES process(process_id),"
        "  type           INTEGER NOT NULL,"
        "  idx            INTEGER NOT NULL,"
        "  idx2           INTEGER NOT NULL,"
        "  available      BOOL NOT NULL DEFAULT '0',"
        "  UNIQUE(process_id, type, idx, idx2)"
        ")",
        now;
    session <<
        "CREATE TABLE IF NOT EXISTS data_feature ("
        "  descr_id        INTEGER NOT NULL REFERENCES data_descriptor(descr_id),"
        "  feature_name    TEXT NOT NULL,"
        "  feature_param1  DOUBLE NOT NULL,"
        "  feature_param2  DOUBLE NOT NULL,"
        "  feature_param3  DOUBLE NOT NULL,"
        "  feature_value   DOUBLE NOT NULL,"
        "  PRIMARY KEY(descr_id, feature_name, feature_param1, feature_param2, feature_param3)"
        ")",
        now;
    session <<
        "CREATE TABLE IF NOT EXISTS label ("
        "  label_id       INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,"
        "  label_text     TEXT NOT NULL"
        ")",
        now;
    session <<
        "CREATE TABLE IF NOT EXISTS classification_object ("
        "  object_id      INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,"
        "  type           INTEGER NOT NULL"
        ")",
        now;
    session <<
        "CREATE TABLE IF NOT EXISTS classification_object_data ("
        "  object_id      INTEGER NOT NULL REFERENCES classification_object(object_id),"
        "  descr_id       INTEGER NOT NULL REFERENCES data_descriptor(descr_id),"
        "  PRIMARY KEY(object_id, descr_id)"
        ")",
        now;
    session <<
        "CREATE TABLE IF NOT EXISTS classification_object_label ("
        "  object_id      INTEGER NOT NULL REFERENCEs classification_object(object_id),"
        "  label_id       INTEGER NOT NULL REFERENCES label(label_id),"
        "  PRIMARY KEY(object_id, label_id)"
        ")",
        now;
    session <<
        "CREATE TABLE IF NOT EXISTS response ("
        "  response_id    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,"
        "  name           TEXT NOT NULL,"
        "  description    TEXT NOT NULL"
        ")",
        now;
    session <<
        "CREATE TABLE IF NOT EXISTS response_label ("
        "  response_id    INTEGER NOT NULL REFERENCES response(response_id),"
        "  object_id      INTEGER NOT NULL REFERENCES classification_object(object_id),"
        "  label_id       INTEGER NOT NULL REFERENCES label(label_id),"
        "  PRIMARY KEY(response_id, object_id, label_id)"
        ")",
        now;
    // A handy view that gives the file name that a classification object was created from.
    session <<
        "CREATE VIEW IF NOT EXISTS classification_object_file AS "
        "SELECT clo.object_id object_id, p.input_file input_file "
        "FROM classification_object clo JOIN classification_object_data USING (object_id) "
        "JOIN data_descriptor USING (descr_id) JOIN process p USING (process_id) "
        "GROUP BY object_id",
        now;
    session.commit();
    _logger.debug("Database setup successful.");
}


void DatabaseSubsystem::setupTriggers()
{
    RWLock::ScopedLock lock(_dbLock, true);

    _logger.debug("Setting up database triggers.");
    Session session = getSession();
    session.begin();

    // Create triggers that ensure that within a classification object,
    // all data descriptors have the same process ID.
    // This is enforced after each INSERT and UPDATE on the
    // classification_object_data relation, as well as after each UPDATE
    // on the data_descriptor table.
    session <<
        "CREATE TRIGGER IF NOT EXISTS "
        "classification_object_data_after_insert_unique_process_id "
        "AFTER INSERT ON classification_object_data "
        "FOR EACH ROW BEGIN "
        "  SELECT RAISE(ROLLBACK, 'Process ID must be unique within classification object') "
        "  WHERE ("
        "    SELECT COUNT(DISTINCT process_id) "
        "    FROM classification_object_data JOIN data_descriptor "
        "    ON classification_object_data.descr_id = data_descriptor.descr_id "
        "    WHERE classification_object_data.object_id = NEW.object_id"
        "  ) <> 1; "
        "END",
        now;
    session <<
        "CREATE TRIGGER IF NOT EXISTS "
        "classification_object_data_after_update_unique_process_id "
        "AFTER UPDATE ON classification_object_data "
        "FOR EACH ROW BEGIN "
        "  SELECT RAISE(ROLLBACK, 'Process ID must be unique within classification object') "
        "  WHERE ("
        "    SELECT COUNT(DISTINCT process_id) "
        "    FROM classification_object_data JOIN data_descriptor "
        "    ON classification_object_data.descr_id = data_descriptor.descr_id "
        "    WHERE classification_object_data.object_id = NEW.object_id"
        "  ) <> 1; "
        "END",
        now;
    session <<
        "CREATE TRIGGER IF NOT EXISTS "
        "data_descriptor_after_update_unique_process_id "
        "AFTER UPDATE ON data_descriptor "
        "FOR EACH ROW BEGIN "
        "  SELECT RAISE(ROLLBACK, 'Process ID must be unique within classification object') "
        "  WHERE ("
        "    SELECT COUNT(DISTINCT process_id) "
        "    FROM classification_object_data JOIN data_descriptor "
        "    ON classification_object_data.descr_id = data_descriptor.descr_id "
        "    WHERE classification_object_data.object_id IN "
        "    (SELECT object_id FROM classification_object_data WHERE descr_id = NEW.descr_id)"
        "  ) > 1; "
        "END",
        now;

    // Analogously, ensure that in every response all classification objects
    // have the same type.
    session <<
        "CREATE TRIGGER IF NOT EXISTS "
        "response_label_after_insert_unique_object_type "
        "AFTER INSERT ON response_label "
        "FOR EACH ROW BEGIN "
        "  SELECT RAISE(ROLLBACK, 'Classification object type must be unique within response') "
        "  WHERE ("
        "    SELECT COUNT(DISTINCT classification_object.type) "
        "    FROM response_label JOIN classification_object "
        "    ON response_label.object_id = classification_object.object_id "
        "    WHERE response_label.response_id = NEW.response_id"
        "  ) <> 1; "
        "END",
        now;
    session <<
        "CREATE TRIGGER IF NOT EXISTS "
        "response_label_after_update_unique_object_type "
        "AFTER UPDATE ON response_label "
        "FOR EACH ROW BEGIN "
        "  SELECT RAISE(ROLLBACK, 'Classification object type must be unique within response') "
        "  WHERE ("
        "    SELECT COUNT(DISTINCT classification_object.type) "
        "    FROM response_label JOIN classification_object "
        "    ON response_label.object_id = classification_object.object_id "
        "    WHERE response_label.response_id = NEW.response_id"
        "  ) <> 1; "
        "END",
        now;
    session <<
        "CREATE TRIGGER IF NOT EXISTS "
        "classification_object_after_update_unique_object_type "
        "AFTER UPDATE ON classification_object "
        "FOR EACH ROW BEGIN "
        "  SELECT RAISE(ROLLBACK, 'Classification object type must be unique within response') "
        "  WHERE ("
        "    SELECT COUNT(DISTINCT classification_object.type) "
        "    FROM response_label JOIN classification_object "
        "    ON response_label.object_id = classification_object.object_id "
        "    WHERE response_label.response_id IN "
        "    (SELECT response_id FROM classification_object_data WHERE object_id = NEW.object_id)"
        "  ) > 1; "
        "END",
        now;

    // Automatic trigger generation for foreign keys

    // Get a list of all tables.
    vector<string> tables;
    session << "SELECT tbl_name FROM sqlite_master WHERE type = 'table' AND "
            << "name <> 'sqlite_sequence'",
            into(tables), now;

    // Determine the respective foreign key constraints and create the
    // corresponding INSERT- and UPDATE-triggers for both the parent and the
    // children of each relation. See the SQLite documentation for further
    // explanations of the PRAGMA directive.
    map<string, string> parentDeletes, parentUpdates;
    for (vector<string>::const_iterator it = tables.begin();
         it != tables.end(); ++it)
    {
        vector<int> garbage1, garbage2;
        vector<string> dest, from, to;
        session << "PRAGMA foreign_key_list(" << *it << ")",
                   into(garbage1), into(garbage2),
                   into(dest), into(from), into(to), now;

        // Dropping is only neccessary when the tables' layout has changed.
        // Please do not delete the following lines, but leave them commented
        // out instead.
        session << "DROP TRIGGER IF EXISTS " << *it << "_before_insert", now;
        session << "DROP TRIGGER IF EXISTS " << *it << "_before_update", now;
        session << "DROP TRIGGER IF EXISTS " << *it << "_after_delete", now;
        session << "DROP TRIGGER IF EXISTS " << *it << "_after_update", now;

        if (dest.empty())
            continue;

        // Prepare the SELECT statements for the INSERT- and UPDATE-triggers of
        // the children. Also, build the DELETE- and UPDATE-triggers for the
        // parent.
        stringstream ss;
        for (unsigned int i = 0; i < dest.size(); ++i) {
            _logger.debug(*it + "." + from.at(i) + " references " +
                          dest.at(i) + "." + to.at(i));

            // The following SQL statement will be used to assure that children
            // only reference valid parent items before INSERT and UPDATE.
            ss << "SELECT RAISE(ROLLBACK, 'Invalid " << from.at(i) << ".') "
               << "WHERE (SELECT " << to.at(i) << " FROM " << dest.at(i)
               << " WHERE " << to.at(i) << " = NEW." << from.at(i) << ") "
               << "IS NULL;";

            // The following SQL statement will be used to assure that any
            // possibly existing children will be removed on parent DELETE.
            stringstream pdss(parentDeletes[dest.at(i)],
                              ios_base::out | ios_base::app | ios_base::ate);
            pdss << "DELETE FROM " << *it
                 << " WHERE " << from.at(i) << " = OLD." << to.at(i) << ";";
            parentDeletes[dest.at(i)] = pdss.str();

            // The following SQL statement will be used to assure that children
            // get updated accordingly on parent UPDATE.
            stringstream puss(parentUpdates[dest.at(i)],
                              ios_base::out | ios_base::app | ios_base::ate);
            puss << "UPDATE " << *it
                 << " SET " << from.at(i) << " = NEW." << to.at(i)
                 << " WHERE " << from.at(i) << " = OLD." << to.at(i) << ";";
            parentUpdates[dest.at(i)] = puss.str();
        }

        // Create the INSERT- and UPDATE-triggers.
        session << "CREATE TRIGGER IF NOT EXISTS " << *it << "_before_insert "
                << "BEFORE INSERT ON " << *it << " FOR EACH ROW BEGIN "
                << ss.str() << " END", now;
        session << "CREATE TRIGGER IF NOT EXISTS " << *it << "_before_update "
                << "BEFORE UPDATE ON " << *it << " FOR EACH ROW BEGIN "
                << ss.str() << " END", now;
    }

    // Create the parents' DELETE-triggers.
    for (map<string, string>::const_iterator it = parentDeletes.begin();
         it != parentDeletes.end(); ++it)
    {
        session << "CREATE TRIGGER IF NOT EXISTS " << it->first << "_after_delete "
                << "AFTER DELETE ON " << it->first << " FOR EACH ROW BEGIN "
                << it->second << " END", now;
    }

    // Create the parents' UPDATE-triggers.
    for (map<string, string>::const_iterator it = parentUpdates.begin();
         it != parentUpdates.end(); ++it)
    {
        session << "CREATE TRIGGER IF NOT EXISTS " << it->first << "_after_update "
                << "AFTER UPDATE ON " << it->first << " FOR EACH ROW BEGIN "
                << it->second << " END", now;
    }

    session.commit();
    _logger.debug("Trigger setup complete.");
}


int DatabaseSubsystem::lastInsertID(Session& session)
{
    int id;
    session << "SELECT last_insert_rowid()", into(id), now;
    return id;
}


void DatabaseSubsystem::insertProcessParams(Session& session, ProcessPtr process)
{
    for (map<string, string>::const_iterator itr = process->parameters.begin();
         itr != process->parameters.end();
         ++itr)
    {
        session <<
            "INSERT INTO process_param (process_id, param_name, param_value) "
            "VALUES (?, ?, ?)",
            use(process->processID),
            use(itr->first),
            use(itr->second),
            now;
    }
}


void DatabaseSubsystem::createProcess(ProcessPtr process)
{
    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    session.begin();
    session <<
        "INSERT INTO process (process_name, input_file, start_time, sample_freq) "
        "VALUES (?, ?, ?, ?)",
        use(process->name),
        use(process->inputFile),
        use(process->startTime),
        use(process->sampleFreq),
        now;
    process->processID = lastInsertID(session);
    insertProcessParams(session, process);
    session.commit();
}


void DatabaseSubsystem::updateProcess(ProcessPtr process)
{
    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    session.begin();
    session <<
        "UPDATE process SET process_name = ?, input_file = ?, start_time = ?, "
        "sample_freq = ?",
        use(process->name),
        use(process->inputFile),
        use(process->startTime),
        use(process->sampleFreq),
        now;
    session <<
        "DELETE FROM process_param WHERE process_id = ?",
        use(process->processID),
        now;
    insertProcessParams(session, process);
    session.commit();
}


void DatabaseSubsystem::removeProcess(ProcessPtr process)
{
    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    session.begin();
    // There's no "direct" relation between classification objects and
    // processes, hence we have to delete them manually before deleting the
    // process itself.
    session << "DELETE FROM classification_object WHERE object_id IN "
               "(SELECT object_id FROM classification_object_data WHERE descr_id IN "
               " (SELECT descr_id FROM data_descriptor WHERE process_id = ?))",
               use(process->processID), now;
    session << "DELETE FROM process WHERE process_id = ?",
               use(process->processID), now;
    session.commit();
}


void DatabaseSubsystem::getProcessParams(Session &session, ProcessPtr process)
{
    RWLock::ScopedLock lock(_dbLock);

    string paramName, paramValue;
    Statement stmt = (session <<
        "SELECT param_name, param_value FROM process_param WHERE process_id = ?",
        use(process->processID), range(0, 1), into(paramName), into(paramValue));
    while (!stmt.done()) {
        if (stmt.execute() == 1) {
            process->parameters[paramName] = paramValue;
        }
    }
}


ProcessPtr DatabaseSubsystem::getProcess(int processID)
{
    RWLock::ScopedLock lock(_dbLock);

    ProcessPtr result;
    Session session = getSession();
    session << "SELECT * FROM process WHERE process_id = ?",
               use(processID), into(result), now;
    if (!result.isNull())
        getProcessParams(session, result);
    return result;
}


ProcessPtr DatabaseSubsystem::getProcess(ClassificationObjectPtr clo)
{
    RWLock::ScopedLock lock(_dbLock);

    ProcessPtr result;
    Session session = getSession();
    session << "SELECT * FROM process INNER JOIN data_descriptor ON"
               " data_descriptor.process_id = process.process_id WHERE"
               " data_descriptor.descr_id IN "
               "  (SELECT descr_id FROM classification_object_data WHERE"
               "   object_id = ?)"
               "LIMIT 1",
            use(clo->objectID), into(result), now;
    if (!result.isNull())
        getProcessParams(session, result);
    return result;
}


vector<ProcessPtr> DatabaseSubsystem::getProcesses()
{
    RWLock::ScopedLock lock(_dbLock);

    vector<ProcessPtr> result;
    Session session = getSession();
    session << "SELECT * FROM process", into(result), now;
    for (vector<ProcessPtr>::const_iterator it = result.begin();
        it != result.end(); it++) {
        getProcessParams(session, *it);
    }
    return result;
}


void DatabaseSubsystem::createDataDescriptor(DataDescriptorPtr data)
{
    debug_assert(data->type != DataDescriptor::Invalid);

    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    session <<
        "INSERT INTO data_descriptor (process_id, type, idx, idx2, available) "
        "VALUES (?, ?, ?, ?, ?)",
        use(data->processID),
        use(data->type),
        use(data->index),
        use(data->index2),
        use(data->available),
        now;
    data->descrID = lastInsertID(session);
}


void DatabaseSubsystem::updateDataDescriptor(DataDescriptorPtr data)
{
    debug_assert(data->type != DataDescriptor::Invalid);

    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    session <<
        "UPDATE data_descriptor SET process_id = ?, type = ?, "
        "idx = ?, idx2 = ?, available = ? "
        "WHERE descr_id = ?",
        use(data->processID),
        use(data->type),
        use(data->index),
        use(data->index2),
        use(data->available),
        use(data->descrID),
        now;
}


void DatabaseSubsystem::removeDataDescriptor(DataDescriptorPtr data)
{
    RWLock::ScopedLock lock(_dbLock, true);

    getSession() << "DELETE FROM data_descriptor WHERE descr_id = ?",
                    use(data->descrID), now;
}


DataDescriptorPtr DatabaseSubsystem::getDataDescriptor(int descrID)
{
    RWLock::ScopedLock lock(_dbLock);

    DataDescriptorPtr result;
    Session session = getSession();
    session <<
        "SELECT * FROM data_descriptor WHERE descr_id = ?",
        use(descrID), into(result),
        now;
    return result;
}


DataDescriptorPtr DatabaseSubsystem::getDataDescriptor(int processID,
    const DataDescriptor::Type type, int index, int index2)
{
    RWLock::ScopedLock lock(_dbLock);

    DataDescriptorPtr result;
    Session session = getSession();
    session <<
        "SELECT * FROM data_descriptor WHERE process_id = ? "
        "AND type = ? AND idx = ? AND idx2 = ?",
        use(processID), use(type), use(index), use(index2),
        range(1, 1),
        into(result),
        now;
    return result;
}


vector<DataDescriptorPtr> DatabaseSubsystem::getDataDescriptors(int processID)
{
    RWLock::ScopedLock lock(_dbLock);

    vector<DataDescriptorPtr> result;
    getSession() <<
        "SELECT * FROM data_descriptor WHERE process_id = ? ",
        use(processID), into(result),
        now;
    return result;
}


vector<DataDescriptorPtr> 
DatabaseSubsystem::getDataDescriptors(ClassificationObjectPtr clo)
{
    RWLock::ScopedLock lock(_dbLock);

    vector<DataDescriptorPtr> result;
    getSession() <<
        "SELECT * FROM data_descriptor WHERE descr_id "
        "IN (SELECT descr_id FROM classification_object_data WHERE object_id = ?)",
        use(clo->objectID), into(result),
        now;
    return result;
}


void DatabaseSubsystem::saveFeature(Session& session, FeaturePtr feature)
{
    session <<
        "INSERT OR REPLACE "
        "INTO data_feature (descr_id, feature_name,"
        " feature_param1, feature_param2, feature_param3, feature_value) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        use(feature->descrID),
        use(feature->name),
        use(feature->params[0]),
        use(feature->params[1]),
        use(feature->params[2]),
        use(feature->value),
        now;
}


void DatabaseSubsystem::saveFeature(FeaturePtr feature)
{
    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    saveFeature(session, feature);
}


void DatabaseSubsystem::saveFeatures(const vector<FeaturePtr>& features)
{
    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    session.begin();

    for (vector<FeaturePtr>::const_iterator itr = features.begin();
        itr != features.end(); ++itr)
    {
        saveFeature(session, *itr);
    }

    session.commit();
}


void DatabaseSubsystem::removeFeature(FeaturePtr feature)
{
    RWLock::ScopedLock lock(_dbLock, true);

    getSession() << "DELETE FROM data_feature WHERE descr_id = ? "
                    "AND feature_name = ? AND feature_param1 = ? "
                    "AND feature_param2 = ? AND feature_param3 = ? ",
                    use(feature->descrID),
                    use(feature->name),
                    use(feature->params[0]),
                    use(feature->params[1]),
                    use(feature->params[2]),
                    now;
}


void DatabaseSubsystem::removeFeatures(int descrID)
{
    RWLock::ScopedLock lock(_dbLock, true);

    getSession() << "DELETE FROM data_feature WHERE descr_id = ?",
                    use(descrID), now;
}


FeaturePtr DatabaseSubsystem::getFeature(int descrID, const string &featureName,
                                         double param1, double param2, double param3)
{
    RWLock::ScopedLock lock(_dbLock);

    Session session = getSession();
    FeaturePtr result;
    session <<
        "SELECT * FROM data_feature WHERE descr_id = ? AND feature_name = ? "
        "  AND feature_param1 = ? AND feature_param2 = ? "
        "  AND feature_param3 = ?",
        use(descrID), use(featureName), 
        use(param1), use(param2), use(param3),
        into(result),
        now;
    return result;
}


vector<FeaturePtr> DatabaseSubsystem::getFeatures(int descrID)
{
    RWLock::ScopedLock lock(_dbLock);

    vector<FeaturePtr> result;
    getSession() <<
        "SELECT * FROM data_feature WHERE descr_id = ?",
        use(descrID),
        into(result),
        now;
    return result;
}


void DatabaseSubsystem::insertClassificationObjectDescrIDs(Session& session,
    ClassificationObjectPtr& clObj)
{
    int descrID;
    for (set<int>::const_iterator itr = clObj->descrIDs.begin();
         itr != clObj->descrIDs.end();
         ++itr)
    {
        descrID = *itr;
        session <<
            "INSERT INTO classification_object_data (object_id, descr_id) "
            "VALUES (?, ?)",
            use(clObj->objectID), use(descrID),
            now;
    }
}


void DatabaseSubsystem::insertClassificationObjectLabelIDs(Session& session,
    ClassificationObjectPtr& clObj)
{
    int labelID;
    for (set<int>::const_iterator itr = clObj->labelIDs.begin();
         itr != clObj->labelIDs.end();
         ++itr)
    {
        labelID = *itr;
        session <<
            "INSERT INTO classification_object_label (object_id, label_id) "
            "VALUES (?, ?)",
            use(clObj->objectID), use(labelID),
            now;
    }
}


void DatabaseSubsystem::getClassificationObjectDescrIDs(Session &session,
    ClassificationObjectPtr& clObj)
{
    session << "SELECT descr_id FROM classification_object_data "
               "WHERE object_id = ?",
               use(clObj->objectID),
               into(clObj->descrIDs),
               now;
}


void DatabaseSubsystem::getClassificationObjectLabelIDs(Session &session,
    ClassificationObjectPtr& clObj)
{
    session << "SELECT label_id FROM classification_object_label "
               "WHERE object_id = ?",
               use(clObj->objectID),
               into(clObj->labelIDs),
               now;
}


void DatabaseSubsystem::createClassificationObject(ClassificationObjectPtr clObj)
{
    debug_assert(clObj->type != ClassificationObject::Invalid);

    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    session.begin();
    session <<
        "INSERT INTO classification_object (type) VALUES (?)",
        use (clObj->type),
        now;
    clObj->objectID = lastInsertID(session);
    insertClassificationObjectDescrIDs(session, clObj);
    insertClassificationObjectLabelIDs(session, clObj);
    session.commit();
}


void DatabaseSubsystem::updateClassificationObject(ClassificationObjectPtr clObj)
{
    vector<ClassificationObjectPtr> clObjVec(1);
    clObjVec[0] = clObj;
    updateClassificationObjects(clObjVec);
}


void DatabaseSubsystem::
updateClassificationObjects(const vector<ClassificationObjectPtr>& clObjs)
{
    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    session.begin();

    // UPDATE
    // Update basic information (currently only type)
    for (vector<ClassificationObjectPtr>::const_iterator itr = clObjs.begin();
        itr != clObjs.end(); ++itr)
    {
        debug_assert((*itr)->type != ClassificationObject::Invalid);
        session <<
            "UPDATE classification_object SET type = ? WHERE object_id = ?",
            use ((*itr)->type), use((*itr)->objectID),
            now;
    }

    // SELECT
    // Determine current labels and data descriptor IDs of the objects
    map<int, set<int> > currentLabels;
    map<int, set<int> > currentDescrIDs;
    for (vector<ClassificationObjectPtr>::const_iterator itr = clObjs.begin();
        itr != clObjs.end(); ++itr)
    {
        session <<
            "SELECT label_id FROM classification_object_label WHERE object_id = ?",
            use((*itr)->objectID),
            into(currentLabels[(*itr)->objectID]),
            now;
        session <<
            "SELECT descr_id FROM classification_object_data WHERE object_id = ?",
            use((*itr)->objectID),
            into(currentDescrIDs[(*itr)->objectID]),
            now;
    }

    // DELETE
    // Delete labels in set currentLabels - newLabels and
    // data descriptor IDs in set currentDescrIDs - newDescrIDs
    for (vector<ClassificationObjectPtr>::const_iterator oItr = clObjs.begin();
        oItr != clObjs.end(); ++oItr)
    {
        set<int> labelsToDelete;
        set<int> descrIDsToDelete;
        set_difference(currentLabels[(*oItr)->objectID].begin(), 
            currentLabels[(*oItr)->objectID].end(),
            (*oItr)->labelIDs.begin(), (*oItr)->labelIDs.end(),
            insert_iterator<set<int> >(labelsToDelete, 
                                       labelsToDelete.end())
        );
        set_difference(currentDescrIDs[(*oItr)->objectID].begin(), 
            currentDescrIDs[(*oItr)->objectID].end(),
            (*oItr)->descrIDs.begin(), (*oItr)->descrIDs.end(),
            insert_iterator<set<int> >(descrIDsToDelete, 
                                       descrIDsToDelete.end())
        );
        for (set<int>::const_iterator lItr = labelsToDelete.begin();
            lItr != labelsToDelete.end(); ++lItr)
        {
            session <<
                "DELETE FROM classification_object_label WHERE object_id = ? "
                "AND label_id = ?",
                use((*oItr)->objectID),
                use(*lItr),
                now;
        }
        for (set<int>::const_iterator dItr = descrIDsToDelete.begin();
            dItr != descrIDsToDelete.end(); ++dItr)
        {
            session <<
                "DELETE FROM classification_object_data WHERE object_id = ? "
                "AND descr_id = ?",
                use((*oItr)->objectID),
                use(*dItr),
                now;
        }
    }

    // INSERT
    // Insert labels in set newLabels - currentLabels
    // and data descriptor IDs in set newDescrIDs - currentDescrIDs
    for (vector<ClassificationObjectPtr>::const_iterator oItr = clObjs.begin();
        oItr != clObjs.end(); ++oItr)
    {
        set<int> labelsToInsert;
        set<int> descrIDsToInsert;
        set_difference((*oItr)->labelIDs.begin(), (*oItr)->labelIDs.end(),
            currentLabels[(*oItr)->objectID].begin(), 
            currentLabels[(*oItr)->objectID].end(),
            insert_iterator<set<int> >(labelsToInsert, 
                                       labelsToInsert.end())
        );
        set_difference((*oItr)->descrIDs.begin(), (*oItr)->descrIDs.end(),
            currentDescrIDs[(*oItr)->objectID].begin(), 
            currentDescrIDs[(*oItr)->objectID].end(),
            insert_iterator<set<int> >(descrIDsToInsert, 
                                       descrIDsToInsert.end())
        );
        for (set<int>::const_iterator lItr = labelsToInsert.begin();
            lItr != labelsToInsert.end(); ++lItr)
        {
            session <<
                "INSERT INTO classification_object_label (object_id, label_id) "
                "VALUES (?, ?)",
                use((*oItr)->objectID),
                use(*lItr),
                now;
        }
        for (set<int>::const_iterator dItr = descrIDsToInsert.begin();
            dItr != descrIDsToInsert.end(); ++dItr)
        {
            session <<
                "INSERT INTO classification_object_data (object_id, descr_id) "
                "VALUES (?, ?)",
                use((*oItr)->objectID),
                use(*dItr),
                now;
        }
    }

    session.commit();
}


void DatabaseSubsystem::removeClassificationObject(ClassificationObjectPtr clObj)
{
    RWLock::ScopedLock lock(_dbLock, true);

    getSession() << "DELETE FROM classification_object WHERE object_id = ?",
                    use(clObj->objectID), now;
}


ClassificationObjectPtr DatabaseSubsystem::getClassificationObject(int clObjID)
{
    RWLock::ScopedLock lock(_dbLock);

    ClassificationObjectPtr result;
    Session session = getSession();
    session.begin();
    session <<
        "SELECT * FROM classification_object WHERE object_id = ?",
        use(clObjID), into(result),
        now;
    if (!result.isNull()) {
        getClassificationObjectDescrIDs(session, result);
        getClassificationObjectLabelIDs(session, result);
    }
    session.commit();
    return result;
}


vector<ClassificationObjectPtr>
DatabaseSubsystem::getClassificationObjects()
{
    RWLock::ScopedLock lock(_dbLock);

    vector<ClassificationObjectPtr> result;
    Session session = getSession();
    session.begin();
    session << "SELECT * FROM classification_object",
               into(result),
               now;
    for (vector<ClassificationObjectPtr>::iterator it = result.begin();
        it != result.end(); ++it)
    {
        getClassificationObjectDescrIDs(session, *it);
        getClassificationObjectLabelIDs(session, *it);
    }
    session.commit();
    return result;
}


vector<ClassificationObjectPtr>
DatabaseSubsystem::getClassificationObjectsForLabel(int labelID)
{
    RWLock::ScopedLock lock(_dbLock);

    vector<ClassificationObjectPtr> result;
    Session session = getSession();
    session.begin();
    session << "SELECT * FROM classification_object WHERE object_id "
               "IN (SELECT object_id FROM classification_object_label WHERE "
               "label_id = ?)",
               use(labelID),
               into(result),
               now;
    for (vector<ClassificationObjectPtr>::iterator it = result.begin();
        it != result.end(); ++it)
    {
        getClassificationObjectDescrIDs(session, *it);
        getClassificationObjectLabelIDs(session, *it);
    }
    session.commit();
    return result;
}


vector<ClassificationObjectPtr>
DatabaseSubsystem::getClassificationObjectsByFilename(const string& filename)
{
    // FIXME: This is erroneous if files with the same name exist in 
    // different dirs. But we don't want the user to mess with
    // full path names. Suggestions?

    RWLock::ScopedLock lock(_dbLock);

    vector<ClassificationObjectPtr> result;
    Session session = getSession();
    session.begin();
    session << "SELECT cof.object_id, co.type "
            << "FROM classification_object_file cof "
            << "JOIN classification_object co USING (object_id) "
            << "WHERE (cof.input_file LIKE ?) "
            << "OR (cof.input_file LIKE '%/' || ?) "
            << "OR (cof.input_file LIKE '%\\' || ?)",
            use(filename),
            use(filename),
            use(filename),
            into(result),
            now;
    for (vector<ClassificationObjectPtr>::iterator it = result.begin();
        it != result.end(); ++it)
    {
        getClassificationObjectDescrIDs(session, *it);
        getClassificationObjectLabelIDs(session, *it);
    }
    session.commit();
    return result;
}


void DatabaseSubsystem::insertResponseLabels(Session& session,
                                           ResponsePtr& response)
{
    for (map<int, int>::const_iterator itr = response->labels.begin();
         itr != response->labels.end(); ++itr)
    {
        session <<
            "INSERT INTO response_label (response_id, object_id, label_id) "
            "VALUES (?, ?, ?)",
            use(response->responseID), use(itr->first), use(itr->second),
            now;
    }
}


void DatabaseSubsystem::createResponse(ResponsePtr response)
{
    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    session.begin();
    session <<
        "INSERT INTO response (name, description) VALUES (?, ?)",
        use(response->name), use(response->description),
        now;
    response->responseID = lastInsertID(session);
    insertResponseLabels(session, response);
    session.commit();
}


void DatabaseSubsystem::updateResponse(ResponsePtr response)
{
    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    session.begin();
    session <<
        "UPDATE response SET name = ?, description = ? WHERE response_id = ?",
        use(response->name), use(response->description),
        use(response->responseID),
        now;
    session <<
        "DELETE FROM response_label WHERE response_id = ?",
        use(response->responseID),
        now;
    insertResponseLabels(session, response);
    session.commit();
}


void DatabaseSubsystem::removeResponse(ResponsePtr response)
{
    RWLock::ScopedLock lock(_dbLock, true);

    getSession() << "DELETE FROM response WHERE response_id = ?",
                    use(response->responseID), now;
}


void DatabaseSubsystem::getResponseLabels(Session& session,
                                          ResponsePtr& response)
{
    RWLock::ScopedLock lock(_dbLock);

    int objectID;
    int labelID;
    Statement stmt = (session <<
        "SELECT object_id, label_id FROM response_label WHERE response_id = ?",
        use(response->responseID), range(0, 1), into(objectID), into(labelID));
    while (!stmt.done()) {
        if (stmt.execute() == 1)
            response->labels[objectID] = labelID;
    }
}


ResponsePtr DatabaseSubsystem::getResponse(int responseID)
{
    RWLock::ScopedLock lock(_dbLock);

    ResponsePtr result;
    Session session = getSession();
    session.begin();
    session <<
        "SELECT * FROM response WHERE response_id = ?",
        use(responseID),
        into(result),
        now;
    if (!result.isNull()) {
        getResponseLabels(session, result);
    }
    session.commit();
    return result;
}


vector<ResponsePtr> DatabaseSubsystem::getResponses()
{
    RWLock::ScopedLock lock(_dbLock);

    vector<ResponsePtr> result;
    Session session = getSession();
    session.begin();
    session << "SELECT * FROM response", into(result), now;
    for (vector<ResponsePtr>::iterator itr = result.begin();
        itr != result.end(); ++itr)
    {
        getResponseLabels(session, *itr);
    }
    session.commit();
    return result;
}


vector<pair<ClassificationObjectPtr, LabelPtr> >
DatabaseSubsystem::getClassificationObjectsAndLabelsForResponse(ResponsePtr r)
{
    RWLock::ScopedLock lock(_dbLock);

    vector<pair<ClassificationObjectPtr, LabelPtr> > result;
    Session session = getSession();
    session.begin();
    for (map<int, int>::const_iterator it = r->labels.begin();
        it != r->labels.end(); it++) {
        ClassificationObjectPtr clo;
        LabelPtr label;
        session << "SELECT * FROM classification_object WHERE object_id = ?",
                   use(it->first), into(clo), now;
        getClassificationObjectDescrIDs(session, clo);
        getClassificationObjectLabelIDs(session, clo);
        session << "SELECT * FROM label where label_id = ?",
                   use(it->second), into(label), now;
        result.push_back(pair<ClassificationObjectPtr, LabelPtr>(clo, label));
    }
    session.commit();
    return result;
}


DataSet DatabaseSubsystem::getDataSet(ResponsePtr response,
                                      const FeatureSet& featureSet)
{
    RWLock::ScopedLock lock(_dbLock);

    DataSet result;
    Session session = getSession();
    session.begin();

    // This map counts the number of values for each feature.
    map<FeatureDescriptor, int> dataCount;

    for (map<int, int>::const_iterator labelItr = response->labels.begin();
        labelItr != response->labels.end(); ++labelItr)
    {
        int ddType;
        string featureName;
        double featureParam1, featureParam2, featureParam3;
        double featureValue;

        DataPoint point;
        point.objectID = labelItr->first;
        point.classLabel = labelItr->second;

        Statement stmt = (session <<
            "SELECT data_descriptor.type, data_feature.feature_name, "
            "  data_feature.feature_param1, data_feature.feature_param2, "
            "  data_feature.feature_param3, data_feature.feature_value "
            "FROM data_feature INNER JOIN data_descriptor "
            "ON (data_feature.descr_id = data_descriptor.descr_id) "
            "WHERE data_descriptor.descr_id IN ("
            "  SELECT descr_id FROM classification_object_data "
            "  WHERE object_id = ?"
            ")",
            use(labelItr->first),
            range(0, 1),
            into(ddType), into(featureName), 
            into(featureParam1), into(featureParam2), into(featureParam3), 
            into(featureValue)
        );

        while (!stmt.done()) {
            if (stmt.execute() == 1)
            {
                FeatureDescriptor featureDescr(featureName,
                    (DataDescriptor::Type) ddType, featureParam1, 
                    featureParam2, featureParam3);
                if (featureSet.has(featureDescr)) {
                    point.components[featureDescr] = featureValue;
                    ++dataCount[featureDescr];
                }
            }
        }

        // Reject "empty" data points.
        if (!point.components.empty())
            result.push_back(point);
    }
    session.commit();

    if (dataCount.size() > 0) {
        map<FeatureDescriptor, int>::const_iterator itr = dataCount.begin();
        int firstCount = itr->second;
        for (; itr != dataCount.end(); ++itr) {
            if (itr->second != firstCount) {
                throw Poco::RuntimeException("Feature count mismatch: " +
                    itr->first.toString());
            }
        }
    }

    return result;
}


DataSet DatabaseSubsystem::getDataSet(ResponsePtr response)
{
    RWLock::ScopedLock lock(_dbLock, false);

    FeatureSet features;

    Session session = getSession();
    getAvailableFeatures(session, response->labels, features);
    return getDataSet(response, features);
}


void DatabaseSubsystem::getAvailableFeatures(Session& session,
    const map<int, int>& clObjMap, FeatureSet& featureSet)
{
    int ddType;
    string featureName;
    double featureParam1, featureParam2, featureParam3;

    featureSet.clear();

    ostringstream clObjIDs;
    for (map<int, int>::const_iterator itr = clObjMap.begin();
         itr != clObjMap.end(); ++itr)
    {
        if (itr != clObjMap.begin()) {
            clObjIDs << ",";
        }
        clObjIDs << itr->first;
    }

    Statement stmt = (session <<
        "SELECT DISTINCT "
        "  data_descriptor.type, data_feature.feature_name, "
        "  data_feature.feature_param1, data_feature.feature_param2, "
        "  data_feature.feature_param3  "
        "FROM data_feature INNER JOIN data_descriptor "
        "ON (data_feature.descr_id = data_descriptor.descr_id) "
        "WHERE data_descriptor.descr_id IN ("
        "  SELECT descr_id FROM classification_object_data WHERE object_id IN ("
        << clObjIDs.str()
        << "))",
        range(0, 1),
        into(ddType), into(featureName), into(featureParam1), 
        into(featureParam2), into(featureParam3));

    while (!stmt.done()) {
        if (stmt.execute() == 1) {
            featureSet.add(FeatureDescriptor(featureName,
                (DataDescriptor::Type) ddType, featureParam1, featureParam2, 
                featureParam3));
        }
    }
}


DataSet DatabaseSubsystem::getDataSet(const map<int, int>& cloLabelsMap)
{
    ResponsePtr response = new Response;
    response->labels = cloLabelsMap;
    return getDataSet(response);
}


void DatabaseSubsystem::createLabel(LabelPtr label)
{
    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    session <<
        "INSERT INTO label (label_text) VALUES (?)",
        use(label->text),
        now;
    label->labelID = lastInsertID(session);
}


void DatabaseSubsystem::updateLabel(LabelPtr label)
{
    RWLock::ScopedLock lock(_dbLock, true);

    Session session = getSession();
    session <<
        "UPDATE label SET label_text = ? WHERE label_id = ?",
        use(label->text), use(label->labelID),
        now;
}


void DatabaseSubsystem::removeLabel(LabelPtr label)
{
    RWLock::ScopedLock lock(_dbLock, true);

    getSession() << "DELETE FROM label WHERE label_id = ?",
                    use(label->labelID), now;
}


LabelPtr DatabaseSubsystem::getLabel(int labelID)
{
    RWLock::ScopedLock lock(_dbLock);

    LabelPtr result;
    Session session = getSession();
    session <<
        "SELECT * FROM label WHERE label_id = ?",
        use(labelID),
        into(result),
        now;
    return result;
}


vector<LabelPtr> DatabaseSubsystem::getLabels()
{
    RWLock::ScopedLock lock(_dbLock);

    vector<LabelPtr> result;
    Session session = getSession();
    session << "SELECT * FROM label", into(result), now;
    return result;
}


vector<LabelPtr> DatabaseSubsystem::getLabelsByText(const string& text)
{
    RWLock::ScopedLock lock(_dbLock);

    vector<LabelPtr> result;
    Session session = getSession();
    session << "SELECT * FROM label WHERE label_text = ?", 
               use(text), into(result), now;
    return result;
}


vector<LabelPtr> DatabaseSubsystem::getLabelsForResponse(blissart::ResponsePtr r)
{
    RWLock::ScopedLock lock(_dbLock);

    vector<LabelPtr> result;
    Session session = getSession();
    session << "SELECT * FROM label WHERE label_id "
               "IN (SELECT label_id FROM response_label WHERE response_id = ?)",
               use(r->responseID), into(result), now;
    return result;
}


} // namespace blissart
