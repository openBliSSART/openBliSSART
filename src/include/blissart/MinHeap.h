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


#ifndef __BLISSART_MINHEAP_H__
#define __BLISSART_MINHEAP_H__


#include <common.h>

#include <vector>
#include <stdexcept>


namespace blissart {


/**
 * \addtogroup framework
 * @{
 */

/**
 * Implementation of a Minimum Binary Heap, i.e. the key of an element on the
 * heap is always less or equal than the keys of its children.
 */
template <typename T>
class MinHeap {

public:
    /**
     * Destructs an instances of MinHeap and frees all allocated memory.
     */
    virtual ~MinHeap();


    /**
     * Inserts the given element on the heap using the given key.
     * @param key               The element's associated key.
     * @param data              The element to be inserted.
     */
    void insert(int key, const T data);


    /**
     * Extracts the minimum element, i.e. removes and returns the element with
     * the smallest key.
     * @throws                  std::runtime_error
     */
    const T extractMin();


    /**
     * Returns the minimum element, i.e. the element with the smallest key.
     * @throws                  std::runtime_error
     */
    const T peekMin() const;


    /**
     * Returns the minimum key of the heap's elements.
     */
    inline int minKey() const;


    /**
     * Removes the given element from the heap.
     * @return                  whether the element was found and removed
     */
    bool remove(const T data);


    /**
     * Decreases the key of the given element by the given amount.
     * @param data              The element who's key should be decreased.
     * @param delta             The amount by which the key should be changed.
     * @return                  The new key.
     * @throws                  std::runtime_error
     */
    int decreaseKey(const T data, unsigned int delta);


    /**
     * Increases the key of the given element by the given amount.
     * @param data              The element who's key should be increased.
     * @param delta             The amount by which the key should be changed.
     * @return                  The new key.
     * @throws                  std::runtime_error
     */
    int increaseKey(const T data, unsigned int delta);


    /**
     * Returns the number of elements on the heap.
     */
    inline int size() const;


    /**
     * Returns whether the heap is empty or not.
     */
    inline bool empty() const;


    /**
     * Returns whether the given element is on the heap or not.
     * @param data              The element to be found.
     */
    inline bool find(const T data) const;


    /**
     * Removes all elements from the heap.
     */
    inline void clear();


protected:
    /**
     * Restores the heap invariant by shifting up the element at the given index
     * in the heap's hierarchy as long as neccessary.
     * @param index             The index of the element that might cause a
     *                          violation of the heap invariant.
     */
    void bubbleUp(int index);


    /**
     * Restores the heap invariant by shifting down the element at the given
     * index in the heap's hierarchy as long as neccessary.
     * @param index             The index of the element that might cause a
     *                          violation of the heap invariant.
     */
    void reHeap(int index);


    /**
     * Swaps the two elements at the given indices.
     * @param i                 The first index.
     * @param j                 The second index.
     */
    inline void swap(int i, int j);


    /**
     * Returns the index of the given element.
     * @param data              The element who's index is sought.
     */
    int indexOf(const T data) const;


private:
    template <typename TT>
    struct Node {
        int      key;
        const TT data;

        Node(int aKey, const TT aData) :
            key(aKey),
            data(aData)
        {
        }
    };

    std::vector<Node<T> *> _nodes;
};


/**
 * @}
 */


template <typename T>
MinHeap<T>::~MinHeap()
{
    typename std::vector<Node<T> *>::iterator it;
    for (it = _nodes.begin(); it != _nodes.end(); ++it)
        delete *it;
    _nodes.clear();
}


template <typename T>
void MinHeap<T>::insert(int key, const T data)
{
    _nodes.push_back(new Node<T>(key, data));
    bubbleUp((int)_nodes.size() - 1);
}


template <typename T>
const T MinHeap<T>::extractMin()
{
    if (_nodes.empty())
        throw std::runtime_error("Empty heap!");

    // Backup the result and delete the root node.
    const T result = _nodes[0]->data;
    delete _nodes[0];

    // Now replace the root node with the last node and restore the heap
    // invariant.
    if (_nodes.size() > 1) {
        _nodes[0] = _nodes[_nodes.size() - 1];
        reHeap(0);
    }

    // Now that the heap shrunk by one element, get rid of the last element
    // of the _nodes vector.
    _nodes.pop_back();

    return result;
}


template <typename T>
const T MinHeap<T>::peekMin() const
{
    if (_nodes.empty())
        throw std::runtime_error("Empty heap!");

    return _nodes[0]->data;
}


template <typename T>
int MinHeap<T>::minKey() const
{
    if (_nodes.empty())
        throw std::runtime_error("Empty heap!");

    return _nodes[0]->key;
}


template <typename T>
bool MinHeap<T>::remove(const T data)
{
    int index = indexOf(data);
    if (index < 0)
        return false;

    // Decrease the key to an absolute minimum. This need only be done if there
    // are more than 1 elements on the heap.
    if (_nodes.size() > 1) {
        int diff = _nodes[index]->key - minKey() + 1;
        debug_assert(diff > 0);
        decreaseKey(data, diff);
    }
    extractMin();

    return true;
}


template <typename T>
int MinHeap<T>::decreaseKey(const T data, unsigned int delta)
{
    int index = indexOf(data);
    if (index < 0)
        throw std::runtime_error("Element not in heap!");

    _nodes[index]->key -= (int)delta;
    const int result = _nodes[index]->key;

    bubbleUp(index);

    return result;
}


template <typename T>
int MinHeap<T>::increaseKey(const T data, unsigned int delta)
{
    int index = indexOf(data);
    if (index < 0)
        throw std::runtime_error("Element not in heap!");

    _nodes[index]->key += (int)delta;
    const int result = _nodes[index]->key;

    reHeap(index);

    return result;
}


template <typename T>
int MinHeap<T>::size() const
{
    return (int)_nodes.size();
}


template <typename T>
bool MinHeap<T>::empty() const
{
    return _nodes.empty();
}


template <typename T>
bool MinHeap<T>::find(const T data) const
{
    return (indexOf(data) >= 0) ? true : false;
}


template <typename T>
void MinHeap<T>::clear()
{
    _nodes.clear();
}


template <typename T>
void MinHeap<T>::bubbleUp(int index)
{
    debug_assert(index >= 0 && index < (int)_nodes.size());

    // Assure heap invariant (bottom-up)
    while (index > 0) {
        int parentIndex = (index + 1) / 2 - 1;
        if (_nodes[index]->key < _nodes[parentIndex]->key) {
            swap(index, parentIndex);
            index = parentIndex;
        } else
            break;
    }
}


template <typename T>
void MinHeap<T>::reHeap(int index)
{
    debug_assert(index >= 0 && index < (int)_nodes.size());

    // Assure heap invariant (top-down)
    while ((unsigned int) index <= _nodes.size() / 2) {
        unsigned int leftChildIndex = index * 2 + 1;
        unsigned int rightChildIndex = index * 2 + 2;

        // Determine the index of the node with the smallest key
        int smallestKeyIndex = index;
        if (leftChildIndex < _nodes.size() &&
            _nodes[leftChildIndex]->key < _nodes[smallestKeyIndex]->key)
        {
            smallestKeyIndex = leftChildIndex;
        }
        if (rightChildIndex < _nodes.size() &&
            _nodes[rightChildIndex]->key < _nodes[smallestKeyIndex]->key)
        {
            smallestKeyIndex = rightChildIndex;
        }

        // If smallestKeyIndex and index don't match, the heap invariant
        // has to be restored. In case they match, we're done.
        if (smallestKeyIndex != index) {
            swap(smallestKeyIndex, index);
            index = smallestKeyIndex;
        } else
            break;
    }
}


template <typename T>
void MinHeap<T>::swap(int i, int j)
{
    debug_assert(i >= 0 && i < (int)_nodes.size());
    debug_assert(j >= 0 && j < (int)_nodes.size());

    Node<T> *tmp = _nodes[i];
    _nodes[i]    = _nodes[j];
    _nodes[j]    = tmp;
}


template <typename T>
int MinHeap<T>::indexOf(const T data) const
{
    int result = -1;

    typename std::vector<Node<T> *>::const_iterator it;
    for (it = _nodes.begin(); it != _nodes.end(); ++it) {
        ++result;
        if ((*it)->data == data)
            return result;
    }

    return -1;
}


} // namespace blissart


#endif // __BLISSART_MINHEAP_H__
