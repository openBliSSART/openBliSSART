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


#include "MinHeapTest.h"
#include <blissart/MinHeap.h>

#include <iostream>
#include <iomanip>
#include <cstdlib>


using blissart::MinHeap;
using namespace std;


namespace Testing {


bool MinHeapTest::performTest()
{
    MinHeap<int> h;

    // First generate 1000 elements that will be used throughout the tests.
    int *elem = new int[1000];
    for (int i = 0; i < 1000; i++)
        elem[i] = i + 1;

    cout << left;

    // Add them.
    {
        cout << setw(50) << "Adding 1000 elements in random order...";

        bool *added = new bool[1000];
        for (int i = 0; i < 1000; i++)
            added[i] = false;

        for (int i = 0; i < 1000; i++) {
            int idx = rand() % 1000;
            while (added[idx])
                idx = rand() % 1000;
            h.insert(elem[idx], elem[idx]);
            added[idx] = true;
        }

        delete[] added;

        cout << "ok" << endl;
    }

    // Check the heap's size.
    {
        cout << setw(50) << "Checking size...";
        if (h.size() != 1000)
            return false;
        cout << "ok" << endl;
    }

    // Decrease the highest key to -1. Then, the associated element should be on
    // top of the heap.
    {
        cout << setw(50) << "Decreasing highest key to -1...";
        h.decreaseKey(elem[999], 1001);
        if (h.peekMin() != 1000 || h.minKey() != -1)
            return false;
        cout << "ok" << endl;
    }

    // Restore the old key by increasing it.
    {
        cout << setw(50) << "Increasing the key again to 1000...";
        h.increaseKey(elem[999], 1001);
        cout << "if next tests are ok, increasing worked just as well!" << endl;
    }

    // Remove one item.
    {
        cout << setw(50) << "Removing item...";
        h.remove(elem[500]);
        cout << "if next tests are ok, increasing worked just as well!" << endl;
    }

    // NOTE: The tests above and below belong together, so don't move them!

    // Then extract them.
    {
        cout << setw(50) << "Extracting 1000 elements in linear order...";
        for (int i = 0; i < 1000; i++) {
            if (i == 500) // That's the one which we just removed from the heap.
                continue;
            int tmp = h.extractMin();
            if (tmp != elem[i]) {
                cout << tmp << " " << elem[i] << endl;
                delete[] elem;
                return false;
            }
        }
        cout << "ok" << endl;
    }

    // And check the size again.
    {
        cout << setw(50) << "Checking size...";
        if (h.size() != 0 || !h.empty())
            return false;
        cout << "ok" << endl;
    }

    // Free allocated memory.
    delete[] elem;

    return true;
}


} // namespace Testing
