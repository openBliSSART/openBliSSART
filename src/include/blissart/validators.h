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


#ifndef __BLISSART_VALIDATORS_H__
#define __BLISSART_VALIDATORS_H__


#include <Poco/Util/Validator.h>
#include <Poco/Util/OptionException.h>
#include <Poco/NumberParser.h>
#include <set>
#include <sstream>


namespace blissart {


/**
 * Classes that perform enhanced validation of command-line options.
 */
namespace validators {


/**
 * \addtogroup framework
 * @{
 */

/**
 * Validates a Poco::Util::Option against lower and upper bounds.
 */
template <typename ValueType>
class RangeValidator : public Poco::Util::Validator
{
public:
    /**
     * Constructs a RangeValidator for a lower but no upper bound.
     */
    RangeValidator(const ValueType& lower);
    
    /**
     * Constructs a RangeValidator for a lower and upper bound.
     */
    RangeValidator(const ValueType& lower, const ValueType& upper);
    
    /**
     * Constructs a RangeValidator for a lower and upper bound.
     * Either of them can be strict or not.
     */
    RangeValidator(const ValueType& lower, bool isStrictLower,
        const ValueType& upper, bool isStrictUpper);
    
    /**
     * Tests if the given Option has a value inside the allowed range.
     */
    virtual void validate(const Poco::Util::Option& option, 
        const std::string& value);
    
private:
    ValueType parse(const std::string& value);

    bool       _hasLower;
    bool       _isStrictLower;
    ValueType  _lower;
    bool       _hasUpper;
    bool       _isStrictUpper;
    ValueType  _upper;
};


/**
 * @}
 */


template <typename ValueType>
inline RangeValidator<ValueType>::RangeValidator(const ValueType& lower) : 
    _hasLower(true), _isStrictLower(false), _lower(lower),
    _hasUpper(false), _isStrictUpper(false), _upper(0)
{
}


template <typename ValueType>
inline RangeValidator<ValueType>::RangeValidator(const ValueType& lower, const ValueType& upper) :
    _hasLower(true), _isStrictLower(false), _lower(lower),
    _hasUpper(true), _isStrictUpper(false), _upper(upper)
{
}


template <typename ValueType>
inline RangeValidator<ValueType>::RangeValidator(const ValueType& lower, bool isStrictLower, 
                                                 const ValueType& upper, bool isStrictUpper) :
    _hasLower(true), _isStrictLower(isStrictLower), _lower(lower),
    _hasUpper(true), _isStrictUpper(isStrictUpper), _upper(upper)
{
}


template <typename ValueType>
inline void RangeValidator<ValueType>::validate(const Poco::Util::Option& option,
                                                const std::string& value)
{
    ValueType parsedValue = parse(value);
    std::ostringstream errStr;
    if (_hasLower && _isStrictLower && parsedValue <= _lower) {
        errStr << option.fullName() << " must be > " << _lower;
        throw Poco::Util::InvalidArgumentException(errStr.str());
    }
    else if (_hasLower && !_isStrictLower && parsedValue < _lower) {
        errStr << option.fullName() << " must be >= " << _lower;
        throw Poco::Util::InvalidArgumentException(errStr.str());
    }
    else if (_hasUpper && _isStrictUpper && parsedValue >= _upper) {
        errStr << option.fullName() << " must be < " << _upper;
        throw Poco::Util::InvalidArgumentException(errStr.str());
    }
    else if (_hasUpper && !_isStrictUpper && parsedValue > _upper) {
        errStr << option.fullName() << " must be <= " << _upper;
        throw Poco::Util::InvalidArgumentException(errStr.str());
    }
}


template <>
inline double RangeValidator<double>::parse(const std::string& value)
{
    double doubleValue;
    if (!Poco::NumberParser::tryParseFloat(value, doubleValue))
        throw Poco::Util::InvalidArgumentException(value);
    return doubleValue;
}


template <>
inline int RangeValidator<int>::parse(const std::string& value)
{
    int intValue;
    if (!Poco::NumberParser::tryParse(value, intValue))
        throw Poco::Util::InvalidArgumentException(value);
    return intValue;
}


} // namespace validators

} // namespace blissart


#endif  // __BLISSART_VALIDATORS_H__
