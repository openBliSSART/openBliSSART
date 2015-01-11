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
// TypeHandlers allow use of objects as targets for Poco's "into" modifier.


#include <Poco/Data/TypeHandler.h>

#include <blissart/DataDescriptor.h>
#include <blissart/Process.h>
#include <blissart/Feature.h>
#include <blissart/ClassificationObject.h>
#include <blissart/Response.h>
#include <blissart/Label.h>


/**
 * Classes that inherit the Poco (Portable Components) framework classes
 */
namespace Poco {

    
/**
 * Classes that inherit the Poco database access classes
 */
namespace Data {


/**
 * TypeHandler specialization for Poco::Timestamp.
 */
template<>
class TypeHandler<Poco::Timestamp>
{
public:
    static std::size_t size()
    {
        return 1;
    }


    static void bind(std::size_t pos, const Poco::Timestamp& data,
										 AbstractBinder* pBinder, AbstractBinder::Direction dir)
    {
			TypeHandler<int>::bind(pos++, (int) data.epochTime(), pBinder,dir);
    }


    static void prepare(std::size_t pos, const Poco::Timestamp& data,
        AbstractPreparator* pPrepare)
    {
        TypeHandler<int>::prepare(pos++, (int) data.epochTime(), pPrepare);
    }


    static void extract(std::size_t pos, Poco::Timestamp& data,
        const Poco::Timestamp& defVal, AbstractExtractor* pExtract)
    {
        int time;
        int defaultTime = static_cast<int>(defVal.epochTime());
        TypeHandler<int>::extract(pos++, time, defaultTime, pExtract);
        data = Poco::Timestamp::fromEpochTime(
            static_cast<std::time_t>(time));
    }
};

/**
 * TypeHandler specialization for DataDescriptor.
 */
template<>
class TypeHandler<blissart::DataDescriptor>
{
public:
    typedef blissart::DataDescriptor DataDescriptor;


    static std::size_t size()
    {
        return 6;
    }


    static void bind(std::size_t pos, const DataDescriptor& data,
										 AbstractBinder* pBinder, AbstractBinder::Direction dir)
    {
			TypeHandler<int>::bind(pos++, data.descrID, pBinder,dir);
			TypeHandler<int>::bind(pos++, data.processID, pBinder,dir);
			TypeHandler<int>::bind(pos++, data.type, pBinder,dir);
			TypeHandler<int>::bind(pos++, data.index, pBinder,dir);
			TypeHandler<int>::bind(pos++, data.index2, pBinder,dir);
			TypeHandler<bool>::bind(pos++, data.available, pBinder,dir);
    }


    static void prepare(std::size_t pos, const DataDescriptor& data,
        AbstractPreparator* pPrepare)
    {
        TypeHandler<int>::prepare(pos++, data.descrID, pPrepare);
        TypeHandler<int>::prepare(pos++, data.processID, pPrepare);
        TypeHandler<int>::prepare(pos++, data.type, pPrepare);
        TypeHandler<int>::prepare(pos++, data.index, pPrepare);
        TypeHandler<int>::prepare(pos++, data.index2, pPrepare);
        TypeHandler<bool>::prepare(pos++, data.available, pPrepare);
    }


    static void extract(std::size_t pos, DataDescriptor& data, 
        const DataDescriptor& defVal, AbstractExtractor* pExtract)
    {
        TypeHandler<int>::extract(pos++, data.descrID, 
            defVal.descrID, pExtract);
        TypeHandler<int>::extract(pos++, data.processID, 
            defVal.processID, pExtract);
        TypeHandler<int>::extract(pos++, (int&)data.type, 
            defVal.type, pExtract);
        TypeHandler<int>::extract(pos++, data.index, 
            defVal.index, pExtract);
        TypeHandler<int>::extract(pos++, data.index2, 
            defVal.index, pExtract);
        TypeHandler<bool>::extract(pos++, data.available,
            defVal.available, pExtract);
    }
};

/**
 * TypeHandler specialization for Process.
 */
template<>
class TypeHandler<blissart::Process>
{
public:
    typedef blissart::Process Process;


    static std::size_t size()
    {
        return 5;
    }


    static void bind(std::size_t pos, const Process& process,
										 AbstractBinder* pBinder,	AbstractBinder::Direction dir)
    {
        TypeHandler<int>::bind(pos++, process.processID, pBinder,dir);
        TypeHandler<std::string>::bind(pos++, process.name, pBinder,dir);
        TypeHandler<std::string>::bind(pos++, process.inputFile, pBinder,dir);
        TypeHandler<Poco::Timestamp>::bind(pos++, process.startTime, pBinder, dir);
        TypeHandler<int>::bind(pos++, process.sampleFreq, pBinder,dir);
    }


    static void prepare(std::size_t pos, const Process& process,
        AbstractPreparator* pPrepare)
    {
        TypeHandler<int>::prepare(pos++, process.processID, pPrepare);
        TypeHandler<std::string>::prepare(pos++, process.name, pPrepare);
        TypeHandler<std::string>::prepare(pos++, process.inputFile, pPrepare);
        TypeHandler<Poco::Timestamp>::prepare(pos++, process.startTime, pPrepare);
        TypeHandler<int>::prepare(pos++, process.sampleFreq, pPrepare);
    }


    static void extract(std::size_t pos, Process& process, 
        const Process& defVal, AbstractExtractor* pExtract)
    {
        TypeHandler<int>::extract(pos++, process.processID,
            defVal.processID, pExtract);
        TypeHandler<std::string>::extract(pos++, process.name,
            defVal.name, pExtract);
        TypeHandler<std::string>::extract(pos++, process.inputFile, 
            defVal.inputFile, pExtract);
        TypeHandler<Poco::Timestamp>::extract(pos++, process.startTime, 
            defVal.startTime, pExtract);
        TypeHandler<int>::extract(pos++, process.sampleFreq,
            defVal.sampleFreq, pExtract);
    }
};


/**
 * TypeHandler specialization for Feature.
 */
template<>
class TypeHandler<blissart::Feature>
{
public:
    typedef blissart::Feature Feature;


    static std::size_t size()
    {
        return 6;
    }


    static void bind(std::size_t pos, const Feature& feature,
										 AbstractBinder* pBinder,		AbstractBinder::Direction dir)
    {
			TypeHandler<int>::bind(pos++, feature.descrID, pBinder,dir);
			TypeHandler<std::string>::bind(pos++, feature.name, pBinder,dir);
        TypeHandler<double>::bind(pos++, feature.params[0], pBinder,dir);
        TypeHandler<double>::bind(pos++, feature.params[1], pBinder,dir);
        TypeHandler<double>::bind(pos++, feature.params[2], pBinder,dir);
        TypeHandler<double>::bind(pos++, feature.value, pBinder,dir);
    }


    static void prepare(std::size_t pos, const Feature& feature,
        AbstractPreparator* pPrepare)
    {
        TypeHandler<int>::prepare(pos++, feature.descrID, pPrepare);
        TypeHandler<std::string>::prepare(pos++, feature.name, pPrepare);
        TypeHandler<double>::prepare(pos++, feature.params[0], pPrepare);
        TypeHandler<double>::prepare(pos++, feature.params[1], pPrepare);
        TypeHandler<double>::prepare(pos++, feature.params[2], pPrepare);
        TypeHandler<double>::prepare(pos++, feature.value, pPrepare);
    }


    static void extract(std::size_t pos, Feature& feature, 
        const Feature& defVal, AbstractExtractor* pExtract)
    {
        TypeHandler<int>::extract(pos++, feature.descrID,
            defVal.descrID, pExtract);
        TypeHandler<std::string>::extract(pos++, feature.name,
            defVal.name, pExtract);
        TypeHandler<double>::extract(pos++, feature.params[0], 
            defVal.params[0], pExtract);
        TypeHandler<double>::extract(pos++, feature.params[1], 
            defVal.params[1], pExtract);
        TypeHandler<double>::extract(pos++, feature.params[2], 
            defVal.params[2], pExtract);
        TypeHandler<double>::extract(pos++, feature.value, 
            defVal.value, pExtract);
    }
};


/**
 * TypeHandler specialization for ClassificationObject.
 */
template<>
class TypeHandler<blissart::ClassificationObject>
{
public:
    typedef blissart::ClassificationObject ClassificationObject;


    static std::size_t size()
    {
        return 2;
    }


    static void bind(std::size_t pos, const ClassificationObject& clObj,
										 AbstractBinder* pBinder,		AbstractBinder::Direction dir)
    {
			TypeHandler<int>::bind(pos++, clObj.objectID, pBinder,dir);
			TypeHandler<int>::bind(pos++, clObj.type, pBinder,dir);
    }


    static void prepare(std::size_t pos, const ClassificationObject& clObj,
        AbstractPreparator* pPrepare)
    {
        TypeHandler<int>::prepare(pos++, clObj.objectID, pPrepare);
        TypeHandler<int>::prepare(pos++, clObj.type, pPrepare);
    }


    static void extract(std::size_t pos, ClassificationObject& clObj, 
        const ClassificationObject& defVal, AbstractExtractor* pExtract)
    {
        TypeHandler<int>::extract(pos++, clObj.objectID, 
            defVal.objectID, pExtract);
        TypeHandler<int>::extract(pos++, (int&)clObj.type, 
            defVal.objectID, pExtract);
    }
};


/**
 * TypeHandler specialization for Response.
 */
template<>
class TypeHandler<blissart::Response>
{
public:
    typedef blissart::Response Response;


    static std::size_t size()
    {
        return 3;
    }


    static void bind(std::size_t pos, const Response& response,
										 AbstractBinder* pBinder,			AbstractBinder::Direction dir)
    {
			TypeHandler<int>::bind(pos++, response.responseID, pBinder,dir);
			TypeHandler<std::string>::bind(pos++, response.name, pBinder,dir);
			TypeHandler<std::string>::bind(pos++, response.description, pBinder,dir);
    }


    static void prepare(std::size_t pos, const Response& response,
        AbstractPreparator* pPrepare)
    {
        TypeHandler<int>::prepare(pos++, response.responseID, pPrepare);
        TypeHandler<std::string>::prepare(pos++, response.name, pPrepare);
        TypeHandler<std::string>::prepare(pos++, response.description, 
            pPrepare);
    }


    static void extract(std::size_t pos, Response& response, 
        const Response& defVal, AbstractExtractor* pExtract)
    {
        TypeHandler<int>::extract(pos++, response.responseID,
            defVal.responseID, pExtract);
        TypeHandler<std::string>::extract(pos++, response.name,
            defVal.name, pExtract);
        TypeHandler<std::string>::extract(pos++, response.description,
            defVal.description, pExtract);
    }
};


/**
 * TypeHandler specialization for Label.
 */
template<>
class TypeHandler<blissart::Label>
{
public:
    typedef blissart::Label Label;


    static std::size_t size()
    {
        return 2;
    }


    static void bind(std::size_t pos, const Label& label,
										 AbstractBinder* pBinder,			AbstractBinder::Direction dir)
    {
			TypeHandler<int>::bind(pos++, label.labelID, pBinder,dir);
			TypeHandler<std::string>::bind(pos++, label.text, pBinder,dir);
    }


    static void prepare(std::size_t pos, const Label& label,
        AbstractPreparator* pPrepare)
    {
        TypeHandler<int>::prepare(pos++, label.labelID, pPrepare);
        TypeHandler<std::string>::prepare(pos++, label.text, pPrepare);
    }


    static void extract(std::size_t pos, Label& label, 
        const Label& defVal, AbstractExtractor* pExtract)
    {
        TypeHandler<int>::extract(pos++, label.labelID,
            defVal.labelID, pExtract);
        TypeHandler<std::string>::extract(pos++, label.text,
            defVal.text, pExtract);
    }
};


/**
 * @}
 */


} } // namespace Poco::Data
