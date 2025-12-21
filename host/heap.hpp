#pragma once
#include <stdint.h>
#include <stdio.h>

#include "macro.hpp"
#include "task_utils.hpp"
#include "geometry.hpp"

#ifdef KNN_ON

template<typename IntType> void swap_int(IntType &a, IntType &b);
template<typename ObjType> void swap_object(ObjType &a, ObjType &b);

/* A max heap for kNN distances */
class heap_host {
public:
    uint8_t num;
    uint8_t max_k;
    int64_t *distance_storage;
    vectorT *vector_storage;

    heap_host(uint8_t max_k = MAX_KNN_SIZE): max_k(max_k), num(0) {
        distance_storage = new int64_t[max_k];
        vector_storage = new vectorT[max_k];
    }
    ~heap_host() {
        delete [] distance_storage;
        delete [] vector_storage;
    }

    void heapify_up(uint8_t index) {
        bool continue_signal = true;
        uint8_t idx_half;
        while(continue_signal) {
            idx_half = (index - 1) >> 1;
            if (index && this->distance_storage[idx_half] < this->distance_storage[index]) {
                swap_int(this->distance_storage[idx_half], this->distance_storage[index]);
                swap_object(this->vector_storage[idx_half], this->vector_storage[index]);
                index = idx_half;
            }
            else continue_signal = false;
        };
    }

    void heapify_down(uint8_t index) {
        uint8_t child, largest;
        bool continue_signal = true;
        int64_t distance_largest, distance_tmp;
        while(continue_signal) {
            largest = index;
            child = (index << 1) + 1;
            if(child < this->num) {
                // Left child
                distance_largest = this->distance_storage[largest];
                distance_tmp = this->distance_storage[child];
                if(distance_largest < distance_tmp) {
                    largest = child;
                    distance_largest = distance_tmp;
                }
                child++;
                if(child < this->num) {
                    // Right child
                    distance_tmp = this->distance_storage[child];
                    if(distance_largest < distance_tmp) {
                        largest = child;
                    }
                }
            }
            if(largest != index) {
                swap_int(this->distance_storage[index], this->distance_storage[largest]);
                swap_object(this->vector_storage[index], this->vector_storage[largest]);
                index = largest;
            }
            else continue_signal = false;
        };
    }

    void dequeue() {
        this->num--;
        this->distance_storage[0] = this->distance_storage[this->num];
        this->vector_storage[0] = this->vector_storage[this->num];
        this->heapify_down(0);
    }

    void enqueue(int64_t distance, vectorT *vec) {
        uint8_t pt;
        if(this->num >= this->max_k) {
            if(distance >= this->distance_storage[0]) return;
            else this->dequeue();
        }
        this->distance_storage[this->num] = distance;
        this->vector_storage[this->num] = *vec;
        this->heapify_up(this->num);
        this->num++;
    }
};

#endif
