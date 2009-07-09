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


#include <blissart/feature/misc.h>
#include <blissart/linalg/Vector.h>

#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>


using namespace blissart::linalg;


namespace blissart {

namespace feature {


double mean(const Vector& data)
{
    double result = 0.0;
    for (unsigned int i = 0; i < data.dim(); ++i) {
        result += data(i);
    }
    result /= (double) data.dim();
    return result;
}


double stddev(const Vector& data)
{
    double m = mean(data), variance = 0.0;
    for (unsigned int i = 0; i < data.dim(); ++i) {
        double dev = data(i) - m;
        variance += dev * dev;
    }
    variance /= (data.dim() - 1);
    return sqrt(variance);
}


double skewness(const Vector& data)
{
    double m = mean(data);
    double nom = 0.0, denom = 0.0;
    for (unsigned int i = 0; i < data.dim(); ++i) {
        double dev = data(i) - m;
        double dev2 = dev * dev;
        denom += dev2;
        dev *= dev2;
        nom += dev;
    }
    if (denom == 0.0)
        return 0.0;
    denom *= sqrt(denom);
    return sqrt((double) data.dim()) * nom / denom;
}


double kurtosis(const Vector& data)
{
    double m = mean(data);
    double nom = 0.0, denom = 0.0;
    for (unsigned int i = 0; i < data.dim(); ++i) {
        double dev = data(i) - m;
        dev *= dev;
        denom += dev;
        dev *= dev;
        nom += dev;
    }
    if (denom == 0.0)
        return 0.0;
    denom *= denom;
    return (double) data.dim() * nom / denom - 3.0;
}


double correlation(const Vector& v1, const Vector& v2)
{
    double result = 0.0;
    unsigned int i1 = 0, i2 = 0;
    for (; i1 < v1.dim() && i2 < v2.dim(); ++i1, ++i2)
    {
        result += v1(i1) * v2(i2);
    }
    return (result - v1.dim() * mean(v1) * mean(v2)) / 
        ((v1.dim() - 1) * stddev(v1) * stddev(v2));
}


double autocorrelation(const Vector& data, unsigned int delay)
{
    if (delay > data.dim())
        return 0.0;

    double mu = mean(data);
    double sigma = stddev(data);

    double c = 0.0;
    for (unsigned int i = 0; i < data.dim() - delay; ++i) {
        c += (data(i) - mu) * (data(i + delay) - mu);
    }
    return c / ((data.dim() - delay) * sigma * sigma);
}


double periodicity(const Vector& data, double gainsFrequency,
                   int bpmMin, int bpmMax, int deltaBpm)
{
    std::vector<double> periodicities;
    for (int bpm = bpmMin; bpm <= bpmMax; bpm += deltaBpm)
    {
        double beatFrequency = (double) bpm / 60.0;
        unsigned int delay = (unsigned int) (gainsFrequency / beatFrequency);
        periodicities.push_back(feature::autocorrelation(data, delay));
    }
    return *max_element(periodicities.begin(), periodicities.end());
}


double centroid(const Vector& data, double baseFrequency)
{
    double weightedSum = 0.0, sum = 0.0;
    for (unsigned int i = 0; i < data.dim(); ++i) {
        double frequency = (double) i / (double) data.dim() * baseFrequency;
        weightedSum += data(i) * frequency;
        sum += data(i);
    }
    return weightedSum / sum;
}


double centroid(const Vector& data, const Vector& frequencies)
{
    double weightedSum = 0.0, sum = 0.0;
    for (unsigned int i = 0; i < data.dim(); ++i) {
        weightedSum += data(i) * frequencies(i);
        sum += data(i);
    }
    return weightedSum / sum;
}


double rolloff(const Vector& data, double baseFrequency, double amount)
{
    double result = 0.0;
    double totalEnergy = 0.0;
    for (unsigned int i = 0; i < data.dim(); ++i) {
        totalEnergy += data(i) * data(i);
    }
    double threshold = amount * totalEnergy;
    double energy = 0.0;
    for (unsigned int i = 0; i < data.dim(); ++i) {
        energy += data(i) * data(i);
        if (energy > threshold) {
            result = (double) i / data.dim() * baseFrequency;
            break;
        }
    }
    return result;
}


template<typename T> 
static inline int signum(T val)
{
    return (val < 0 ? -1 : 1);
}


double zeroCrossingRate(const Vector& data, double durationMS)
{
    assert(durationMS > 0);
    unsigned int numCrossings = 0;
    for (unsigned int i = 1; i < data.dim(); i++) {
        if (signum<double>(data.at(i-1)) != signum<double>(data.at(i)))
            ++numCrossings;
    }
    return (double)numCrossings / durationMS;
}


RowVector findLocalMaxima(const Vector& data)
{
    RowVector result(data.dim());
    result(0) = data(0);
    double xp1 = 0.0, xp2 = 0.0;
    for (unsigned int i = 2; i < data.dim(); ++i) {
        if (data(i) > xp1 || xp2 > xp1)
            result(i - 1) = 0.0;
        else
            result(i - 1) = data(i - 1);
        xp2 = xp1;
        xp1 = data(i);
    }
    result(data.dim() - 1) = data(data.dim() - 1);
    return result;
}


double noiseLikeness(const Vector& data, double sigma)
{
    sigma = fabs(sigma);
    double sigma2 = sigma * sigma;
    RowVector maxima = findLocalMaxima(data);
    RowVector convolution(data.dim());
    // Convolve the local maxima with a gaussian impulse (mean = zero, max = 1,
    // stddev = sigma)
    for (unsigned int n = 0; n < maxima.dim(); ++n) {
        convolution(n) = 0.0;
        int kStart = (int) (n - 3 * sigma);
        unsigned int kEnd = (int) (n + 3 * sigma);
        kEnd = kEnd < maxima.dim() - 1 ? kEnd : maxima.dim() - 1;
        for (unsigned int k = kStart > 0 ? ((unsigned int) kStart) : 0; 
            k <= kEnd; ++k) 
        {
            if (maxima(k) == 0.0)
                continue;
            double arg = (double) n - (double) k;
            double g = exp(-0.5 * arg * arg / sigma2);
            convolution(n) += maxima(k) * g;
        }
    }
    return correlation(data, convolution);
}


double percussiveness(const Vector& data, unsigned int length)
{
    RowVector maxima = findLocalMaxima(data);
    RowVector convolution(data.dim());
    // Convolve the local maxima with a gaussian impulse (mean = zero, max = 1,
    // stddev = sigma)
    for (unsigned int n = 0; n < maxima.dim(); ++n) {
        convolution(n) = 0.0;
        int kStart = n - length;
        for (unsigned int k = kStart > 0 ? ((unsigned int) kStart) : 0; 
            k <= n; ++k) 
        {
            if (maxima(k) == 0.0)
                continue;
            double arg = (double) n - (double) k;
            double f = 1 - arg / length;
            convolution(n) += maxima(k) * f;
        }
    }
    return correlation(data, convolution);
}


double spectralDissonance(const Vector& data, double baseFrequency)
{
    double result = 0.0;
    for (unsigned int i = 0; i < data.dim(); ++i) {
        double f1 = (double) i / (double) data.dim() * baseFrequency;
        double s = 0.24 / (0.021 * f1 + 19.0);
        double as = -3.5 * s;
        // The paper by Uhle et al. contains an error here.
        // The sign of the argument to exp must be negative, otherwise
        // it will lead to overflow.
        double bs = -5.75 * s;
        for (unsigned int j = i + 1; j < data.dim(); ++j) {
            double f2 = (double) j / (double) data.dim() * baseFrequency;
            double deltaf = f2 - f1;
            result += data(i) * data(j) * 
                (exp(as * deltaf) - exp(bs * deltaf));
        }
    }
    return result;
}


double spectralFlatness(const Vector& data)
{
    double count = (double) data.dim();
    double prod = 1.0;
    double sum = 0.0;
    for (unsigned int i = 0; i < data.dim(); ++i) {
        double elem = data(i);
        // We have to calculate the nth square root here,
        // since otherwise for large vectors the product will get zero
        prod *= pow(elem, 2.0 / count);
        sum += elem * elem;
    }
    return prod * count * (1.0 / sum);
}


} // namespace feature

} // namespace blissart

