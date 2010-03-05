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

#include <cmath>
#include <stdexcept>
#include <blissart/WindowFunctions.h>


namespace blissart {


double HannFunction(unsigned int index, unsigned int count)
{
    const static double pi = 4 * atan(1.0);
    return 0.5 - 0.5 * cos(2 * pi * index / (count - 1));
}


double SqHannFunction(unsigned int index, unsigned int count)
{
    const static double pi = 4 * atan(1.0);
    return sqrt(0.5 - 0.5 * cos(2 * pi * index / (count - 1)));
}


double HammingFunction(unsigned int index, unsigned int count)
{
    const static double pi = 4 * atan(1.0);
    return 0.54 - 0.46 * cos(2 * pi * index / (count - 1));
}


double RectangleFunction(unsigned int index, unsigned int count)
{
    if (index < 0 || index >= count) return 0.0;
    else return 1.0;
}


std::string windowFunctionName(WindowFunction wf) 
{
    if (wf == HannFunction)
        return "Hann function";
    else if (wf == SqHannFunction)
        return "Square root of Hann function";
    else if (wf == HammingFunction)
        return "Hamming function";
    else if (wf == RectangleFunction)
        return "Rectangle function";
    else
        throw std::runtime_error("Unknown window function");
}   


std::string windowFunctionShortName(WindowFunction wf)
{
    if (wf == HannFunction)
        return "Hann function";
    else if (wf == SqHannFunction)
        return "Square root of Hann function";
    else if (wf == HammingFunction)
        return "Hamming function";
    else if (wf == RectangleFunction)
        return "Rectangle function";
    else
        throw std::runtime_error("Unknown window function");
}


WindowFunction windowFunctionForName(const std::string& wfName)
{
    if (wfName == "Square root of Hann function")
        return SqHannFunction;
    else if (wfName == "Hann function")
        return HannFunction;
    else if (wfName == "Hamming function")
        return HammingFunction;
    else if (wfName == "Rectangle function")
        return RectangleFunction;
    else
        throw std::runtime_error("Unknown window function: " + wfName);
}


WindowFunction windowFunctionForShortName(const std::string &wfName)
{
    if (wfName == "sqhann")
        return SqHannFunction;
    else if (wfName == "hann")
        return HannFunction;
    else if (wfName == "hamming")
        return HammingFunction;
    else if (wfName == "rectangle")
        return RectangleFunction;
    else
        throw std::runtime_error("Unknown window function: " + wfName);
}


} // namespace blissart
