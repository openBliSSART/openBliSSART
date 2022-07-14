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


#include "SamplesPreviewCanvas.h"
#include <QPaintEvent>
#include <QPainter>
#include <QPainterPath>
#include <QRect>
#include <cassert>
#include <cmath>


namespace blissart {

namespace internal {


SamplesPreviewCanvas::SamplesPreviewCanvas(QWidget *parent) :
    QWidget(parent),
    _samples(NULL),
    _tMin(0),
    _tMax(1),
    _markerPos(0),
    _markerVisible(false)
{
    // Background color.
    setAutoFillBackground(true);
    QPalette palette;
    palette.setBrush(QPalette::Window, QBrush(Qt::lightGray));
    setPalette(palette);
}


SamplesPreviewCanvas:: SamplesPreviewCanvas(const double *samples,
                                            size_t nSamples,
                                            QWidget *parent) :
    QWidget(parent),
    _samples(NULL),
    _tMin(0),
    _tMax(1),
    _markerPos(0),
    _markerVisible(false)
{
    // Copy the samples.
    setSamples(samples, nSamples);
    // Background color.
    setAutoFillBackground(true);
    QPalette palette;
    palette.setBrush(QPalette::Window, QBrush(Qt::lightGray));
    setPalette(palette);
}


SamplesPreviewCanvas::~SamplesPreviewCanvas()
{
    if (_samples) {
        delete[] _samples;
        _samples = nullptr;
    }
}


void SamplesPreviewCanvas::setSamples(const double *samples,
                                            size_t nSamples)
{
    assert(nSamples > 0);
    _nSamples = nSamples; 
    if (_samples)
        delete _samples;
    _samples = new double[_nSamples];
    memcpy(_samples, samples, _nSamples * sizeof(double));
    // Schedule a paint-event.
    update();
}


void SamplesPreviewCanvas::setMin(float newMin)
{
    assert(newMin >= 0.0 && newMin <= 1.0);
    if (newMin < _tMax)
        _tMin = newMin;
    // Schedule a paint-event.
    update();
}


void SamplesPreviewCanvas::setMax(float newMax)
{
    assert(newMax >= 0.0 && newMax <= 1.0);
    if (newMax > _tMin)
        _tMax = newMax;
    // Schedule a paint-event.
    update();
}


void SamplesPreviewCanvas::setMinMax(float newMin, float newMax)
{
    assert(newMin >= 0.0 && newMax <= 1.0);
    if (newMin < newMax) {
        _tMin = newMin;
        _tMax = newMax;
    }
    // Schedule a paint-event.
    update();
}


QSize SamplesPreviewCanvas::sizeHint() const
{
    return QSize(400, 100);
}


void SamplesPreviewCanvas::paintEvent(QPaintEvent *ev)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    // "Cache" the widget's rect because we need it several times.
    QRect r(rect());

    // Draw the widget's frame if applicable.
    if (r.intersects(ev->rect()))
        p.drawRect(r);
    
    // Draw the samples.
    if (_samples) {
        // Note: If you change the samplesRect below you also have to adapt the
        // corresponding rectangle in updateMarker().
        const QRect samplesRect = r.adjusted(1, 1, -1, -1);
        // ev->rect() defines the rectangle that needs to be updated, but since
        // the ev->rect() can be bigger than the samplesRect and thus may cause
        // errors, we must only work inside the intersection of ev->rect() and
        // samplesRect.
        const QRect ir = samplesRect.intersected(ev->rect());
        // _tMin and _tMax define the displayed portion of the samples.
        const size_t minIndex = (size_t)(_tMin * (_nSamples - 1));
        const size_t maxIndex = (size_t)(_tMax * (_nSamples - 1));
        const float stepSize = (maxIndex - minIndex) / (float)samplesRect.width();
        const float middle = (samplesRect.top() + samplesRect.bottom()) / 2.0f;
        QPainterPath path;
        for (int i = ir.left(); i <= ir.right(); i++) {
            // We have to find every interval's absolute maximum in order to get
            // the best visual representation of the signal.
            size_t j = (size_t)(minIndex + i * stepSize);
            double maximum = _samples[j];
            double absMaximum = fabs(_samples[j]);
            if (i < samplesRect.right()) {
                // The above check for i < samplesRect.right() is neccessary due
                // to the fact that the interval [minIndex, maxIndex] is
                // distributed over the whole width of the samplesRect. Thus the
                // right edge of the samplesRect already represents the maxIndex.
                for (++j; j < (size_t)(minIndex + (i + 1) * stepSize); j++) {
                    double absVal = fabs(_samples[j]);
                    if (absVal > absMaximum) {
                        absMaximum = absVal;
                        maximum = _samples[j];
                    }
                }
            }
            if (i == ir.left())
                path.moveTo(i, middle + middle * maximum);
            else
                path.lineTo(i, middle + middle * maximum);
        }
        p.drawPath(path);

        // Draw the marker if applicable.
        if (_markerVisible) {
            int markerX = samplesRect.left() + 
                   (_markerPos - _tMin) * samplesRect.width() / (_tMax - _tMin);
            if (markerX >= ev->rect().left() && markerX <= ev->rect().right()) {
                QPen darkBluePen(Qt::darkBlue, 0);
                p.setPen(darkBluePen);
                p.setCompositionMode(QPainter::CompositionMode_SourceOver);
                p.drawLine(markerX, 0, markerX, samplesRect.bottom());
                p.setCompositionMode(QPainter::CompositionMode_Source);
            }
        }
    }

    // Gray out the whole area if the widget is disabled.
    if (!isEnabled()) {
        QBrush disabledBrush(
                palette().color(QPalette::Disabled, QPalette::Window),
                Qt::Dense4Pattern
        );
        p.fillRect(r, disabledBrush);
    }

    ev->accept();
}


void SamplesPreviewCanvas::mouseReleaseEvent(QMouseEvent *ev)
{
// was    float pos = _tMin + (_tMax - _tMin) * ev->x() / rect().width();
    float pos = _tMin + (_tMax - _tMin) * ev->position().x() / rect().width();
    if (pos >= 0.0 && pos <= 1.0)
        emit posClicked(pos);
    ev->accept();
}


void SamplesPreviewCanvas::setMarkerPos(float newPos)
{
    assert(newPos >= 0.0 && newPos <= 1.0);
    updateMarker();
    _markerPos = newPos;
    updateMarker();
}


void SamplesPreviewCanvas::setMarkerVisible(bool visible)
{
    if (visible == _markerVisible)
        return;

    _markerVisible = visible;
    updateMarker();
}


void SamplesPreviewCanvas::updateMarker()
{
    if (_markerPos >= _tMin && _markerPos <= _tMax) {
        QRect r(rect().adjusted(1, 1, -1, -1));
        int markerX = r.left() + (_markerPos - _tMin) * r.width() / (_tMax - _tMin);
        if (markerX >= r.left() && markerX <= r.right())
            update(markerX - 1, 0, 3, r.height()); 
    }
}


} // namespace internal

} // namespace blissart

