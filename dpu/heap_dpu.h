#pragma once
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <stdint.h>
#include <stdio.h>

#include "macro.h"
#include "task_utils.h"
#include "geometry.h"

#ifdef KNN_ON

/* A max heap for kNN distances */
typedef struct heap_dpu {
    uint8_t num;
    uint8_t max_k;
    mpint64_t distance_storage;
    mpvector vector_storage;
    uint8_t arr[MAX_KNN_SIZE_DPU];
} heap_dpu;

static inline void heap_dpu_init(heap_dpu *heap, uint8_t max_k) {
    heap->max_k = max_k;
    heap->num = 0;
}

static inline heap_dpu* heap_dpu_new(uint8_t max_k, mpvoid data_storage) {
    heap_dpu *new_heap = (heap_dpu*) mem_alloc(sizeof(heap_dpu));
    heap_dpu_init(new_heap, max_k);
    new_heap->distance_storage = data_storage;
    new_heap->vector_storage = (mpvector)(data_storage + S64(MAX_KNN_SIZE_DPU));
    return new_heap;
}

static inline void swap_uint8(uint8_t *a, uint8_t *b) {
    *a ^= *b;
    *b ^= *a;
    *a ^= *b;
}

static inline void heapify_up(heap_dpu* heap, uint8_t index) {
    uint8_t *idx, *idx_half;
    bool continue_signal = true;
    while(continue_signal) {
        idx = &(heap->arr[index]);
        idx_half = &(heap->arr[(index - 1) >> 1]);
        if (index && heap->distance_storage[*idx_half] < heap->distance_storage[*idx]) {
            swap_uint8(idx, idx_half);
            index = (index - 1) >> 1;
        }
        else continue_signal = false;
    };
}

static inline void heapify_down(heap_dpu* heap, uint8_t index) {
    uint8_t child, largest;
    bool continue_signal = true;
    int64_t distance_largest, distance_tmp;
    while(continue_signal) {
        largest = index;
        child = (index << 1) + 1;
        if(child < heap->num) {
            // Left child
            distance_largest = heap->distance_storage[heap->arr[largest]];
            distance_tmp = heap->distance_storage[heap->arr[child]];
            if(distance_largest < distance_tmp) {
                largest = child;
                distance_largest = distance_tmp;
            }
            child++;
            if(child < heap->num) {
                // Right child
                distance_tmp = heap->distance_storage[heap->arr[child]];
                if(distance_largest < distance_tmp) {
                    largest = child;
                }
            }
        }
        if(largest != index) {
            swap_uint8(&(heap->arr[index]), &(heap->arr[largest]));
            index = largest;
        }
        else continue_signal = false;
    };
}

static inline void dequeue(heap_dpu* heap) {
    heap->num--;
    heap->arr[0] = heap->arr[heap->num];
    heapify_down(heap, 0);
}

static inline void enqueue(heap_dpu* heap, int64_t distance, vectorT *vec) {
    uint8_t pt;
    if(heap->num >= heap->max_k) {
        pt = heap->arr[0];
        if(distance >= heap->distance_storage[pt]) return;
        else dequeue(heap);
    }
    else {
        pt = heap->num;
    }
    heap->distance_storage[pt] = distance;
    heap->vector_storage[pt] = *vec;
    heap->arr[heap->num] = pt;
    heapify_up(heap, heap->num);
    heap->num++;
}

#endif
