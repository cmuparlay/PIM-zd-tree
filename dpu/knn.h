#pragma once
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <stdint.h>
#include <stdio.h>

#include "macro.h"
#include "task_utils.h"
#include "node_dpu.h"
#include "storage.h"
#include "utils_dpu.h"
#include "buffer_dpu.h"
#include "geometry.h"
#include "heap_dpu.h"

vectorT vector_add(vectorT *op1, vectorT *op2);
vectorT vector_sub(vectorT *op1, vectorT *op2);
void vector_ones(vectorT *src, COORD c);
vectorT vector_sub_zero_bounded(vectorT *op1, vectorT *op2);

#ifdef LX_NORM_ON_DPU
COORD vector_norm(vectorT *v);
bool radius_intersect_box(vectorT *v, COORD radius, vectorT *box_min, vectorT *box_max);
bool radius_contained_in_box(vectorT *v, COORD radius, vectorT *box_min, vectorT *box_max);
#else
COORD vector_norm_dpu(vectorT *v);
bool radius_intersect_box_dpu(vectorT *v, COORD radius, vectorT *box_min, vectorT *box_max);
bool radius_contained_in_box_dpu(vectorT *v, COORD radius, vectorT *box_min, vectorT *box_max);
#endif

#ifdef KNN_ON

static inline void knn(vectorT *center, int64_t radius, heap_dpu *heap, mpvoid buf, int buf_size) {
    pptr parent = b_search(coord_to_key(center), true), addr;
    mppptr pptr_buf = (mppptr)buf, pptr_buf_end = pptr_buf + buf_size / sizeof(pptr);
    mppptr pptr_head, pptr_tail;
    int8_t child_idx = -1;
    uint64_t key;
    mBptr b_addr = root;
    mPptr p_addr;
    Bnode bnode;
    Pnode pnode;
    vectorT box_min, box_max;
    int64_t distance;
    bool continue_signal = true;
    while(continue_signal) {
        // Load subtree nodes into stack
        pptr_tail = pptr_head = pptr_buf;
        if(parent.data_type == B_NODE_DATA_TYPE) {
            b_addr = pptr_to_mbptr(parent);
            m_read(b_addr->children, bnode.children, S64(DB_SIZE));
            for(int8_t i = 0; i < DB_SIZE; i++) {
                if(i != child_idx) {
                    addr = bnode.children[i];
                    if(valid_pptr(addr)) {
                        *pptr_tail = addr;
                        pptr_tail++;
                    }
                }
            }
        }
        else if(parent.data_type == P_NODE_DATA_TYPE) {
            *pptr_tail = parent;
            pptr_tail++;
        }
        // Brute-force search
        while(pptr_tail != pptr_head) {
            addr = *pptr_head;
            if(addr.data_type == P_NODE_DATA_TYPE) {
                p_addr = pptr_to_mpptr(addr);
                key = p_addr->num;
                m_read(p_addr->v, pnode.v, S64(MULTIPLY_NR_DIMENSION(key)));
                for(uint64_t i = 0; i < key; i++) {
                    box_max = vector_sub(pnode.v + i, center);
#ifdef LX_NORM_ON_DPU
                    distance = vector_norm(&box_max);
#else
                    distance = vector_norm_dpu(&box_max);
#endif
                    if(distance <= radius) {
                        enqueue(heap, distance, pnode.v + i);
                        if(heap->num == heap->max_k) radius = heap->distance_storage[heap->arr[0]];
                    }
                }
            }
            else if(addr.data_type == B_NODE_DATA_TYPE) {
                b_addr = pptr_to_mbptr(addr);
                m_read(&(b_addr->box_min), &box_min, S64(MULTIPLY_NR_DIMENSION(2)));
#ifdef LX_NORM_ON_DPU
                continue_signal = radius_intersect_box(center, radius, &box_min, &box_max);
#else
                continue_signal = radius_intersect_box_dpu(center, radius, &box_min, &box_max);
#endif
                if(continue_signal) {
                    m_read(b_addr->children, bnode.children, S64(DB_SIZE));
                    for(int8_t i = 0; i < DB_SIZE; i++) {
                        addr = bnode.children[i];
                        if(valid_pptr(addr)) {
                            *pptr_tail = addr;
                            pptr_tail++; if(pptr_tail >= pptr_buf_end) pptr_tail = pptr_buf;
                        }
                    }
                }
            }
            pptr_head++; if(pptr_head >= pptr_buf_end) pptr_head = pptr_buf;
        };
        // Prepare for the next iteration
        if(parent.data_type == B_NODE_DATA_TYPE) {
            b_addr = pptr_to_mbptr(parent);
            m_read(b_addr, &bnode, BNODE_METADATA_SIZE);
            key = bnode.key;
            box_min = bnode.box_min;
            box_max = bnode.box_max;
            b_addr = load_node_parent(bnode.parent);
        }
        else if(parent.data_type == P_NODE_DATA_TYPE) {
            p_addr = pptr_to_mpptr(parent);
            m_read(p_addr, &pnode, PNODE_METADATA_SIZE);
            key = pnode.key;
            box_min = pnode.box_min;
            box_max = pnode.box_max;
            b_addr = load_node_parent(pnode.parent);
        }
        parent = mbptr_to_pptr(b_addr);
#ifdef LX_NORM_ON_DPU
        continue_signal = !radius_contained_in_box(center, radius, &box_min, &box_max) && b_addr != INVALID_MBPTR;
#else
        continue_signal = !radius_contained_in_box_dpu(center, radius, &box_min, &box_max) && b_addr != INVALID_MBPTR;
#endif
        if(continue_signal) child_idx = lookup_next_bit_chunk(key, b_addr->height);
    };
}

static inline int64_t sqrt_dpu(uint64_t x) {
    // x is smaller than (1<<(64-__builtin_clzll))
    return ((int64_t)1) << (((64-__builtin_clzll(x)) >> 1) + 1);
}

static inline bool knn_first_round_finished(vectorT *center, int64_t radius) {
    vectorT vec;
#if LX_NORM == 2
    radius = sqrt_dpu(radius);
#endif
    vector_ones(&vec, radius);
    vec = vector_sub_zero_bounded(center, &vec);
    if(coord_to_key(&vec) < local_range_start) return false;
    vector_ones(&vec, radius);
    vec = vector_add(center, &vec);
    return coord_to_key(&vec) <= local_range_end;
}

#endif
