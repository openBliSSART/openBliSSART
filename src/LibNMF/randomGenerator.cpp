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


#include <blissart/nmf/randomGenerator.h>
#include <cmath>
#include <string>
#include <stdexcept>


namespace blissart {

namespace nmf {


Elem gaussianRandomGenerator(unsigned int, unsigned int)
{
    static bool haveNumber = false;
    static Elem number;
    if (haveNumber) {
        haveNumber = false;
        return number;
    }
    else {
        Elem q = 2.0;
        Elem x, y;
        // Use Polar Method to obtain normally distributed random numbers.
        while (q > 1.0) {
            x = ((Elem)rand() / RAND_MAX) * 2.0 - 1.0;
            y = ((Elem)rand() / RAND_MAX) * 2.0 - 1.0;
            q = x * x + y * y;
        }
        Elem z = -2.0 * log(q) / q;
        // The std:: here is absolutely necessary to use the C++ (overloaded)
        // version of this function!!!
        number = std::abs(y * sqrt(z));
        haveNumber = true;
        return std::abs(x * sqrt(z));
    }
}


const char* randomGeneratorName(linalg::Matrix::GeneratorFunction gf)
{
    if (gf == unityGenerator)
        return "unity";
    if (gf == gaussianRandomGenerator)
        return "gaussian";
    if (gf == uniformRandomGenerator)
        return "uniform";
    throw std::runtime_error("randomGeneratorName: invalid generator function");
}


linalg::Matrix::GeneratorFunction randomGeneratorForName(const std::string& name)
{
    if (name == "unity")
        return unityGenerator;
    if (name == "gaussian")
        return gaussianRandomGenerator;
    if (name == "uniform")
        return uniformRandomGenerator;
    throw std::runtime_error("randomGeneratorForName: invalid name: " + name);
}


} // namespace nmf

} // namespace blissart
