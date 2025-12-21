#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <parlay/primitives.h>
#include <utility>

#include "utils.h"
#include "pptr.h"
#include "geometry.hpp"

/* Return a randomly-hashed DPU ID based on keys */
static inline int hash_to_dpu(uint64_t key, uint64_t height, uint64_t max_dpu) {
    uint64_t v = parlay::hash64(key) + height;
    v = parlay::hash64(v);
    return v % max_dpu;
}

/* Only check address and data_type */
static inline bool equal_pptr_weak(pptr &a, pptr &b) {
    return a.addr == b.addr && a.id == b.id && a.data_type == b.data_type;
}

/* Check all things: address, data_type, and info */
static inline bool equal_pptr_strong(pptr &a, pptr &b) {
    return a.addr == b.addr && a.id == b.id && a.data_type == b.data_type && a.info == b.info;
}

template<typename IntType>
static inline void swap_int(IntType &a, IntType &b) {
    a ^= b;
    b ^= a;
    a ^= b;
}

template<typename ObjType>
static inline void swap_object(ObjType &a, ObjType &b) {
    ObjType obj = a;
    a = b;
    b = obj;
}

/* Swap two ints if min_val > max_val */
template<typename IntType>
static inline void ordered_swap_int(IntType &min_val, IntType &max_val) {
    if(min_val > max_val) {
        min_val ^= max_val;
        max_val ^= min_val;
        min_val ^= max_val;
    }
}

/* Ensure that every coordinate in box_min is smaller than box_max */
static inline void box_boundary_swap(vectorT &box_min, vectorT &box_max) {
#if NR_DIMENSION == 2
    ordered_swap_int(box_min.x, box_max.x);
    ordered_swap_int(box_min.y, box_max.y);
#elif NR_DIMENSION == 3
    ordered_swap_int(box_min.x, box_max.x);
    ordered_swap_int(box_min.y, box_max.y);
    ordered_swap_int(box_min.z, box_max.z);
#else
    for(int k = 0; k < NR_DIMENSION; k++)
        ordered_swap_int(box_min.x[k], box_max.x[k]);
#endif
}

/* Return the LITMAX and BIGMIN of a box split */
static inline std::pair<uint64_t, uint64_t> box_split(uint64_t key_min, uint64_t key_max) {
    // Tropf, Hermann and H. Herzog. “Multimensional Range Search in Dynamically Balanced Trees.” Angew. Inform. 23 (1981): 71-77.
    INT_HEIGHT match_height = maximum_match_height_precise(key_min, key_max);
    uint64_t split_key = key_max & (UINT64_MAX << (63 - match_height));
    uint64_t litmax = split_key, bigmin = split_key;
    int idx_lookup;

    /* Flag true, load 1000; Flag false, load 0111. */
    auto load_in_box_split = [&](uint64_t key, bool flag, INT_HEIGHT pos) -> uint64_t {
        bool first_time = true;
        for(int j = pos; j <= 64; j += NR_DIMENSION) {
            if((flag && first_time) || (!flag && !first_time)) key = set_bit_pos(key, j);
            else key = reset_bit_pos(key, j);
            first_time = false;
        }
        return key;
    };

    for(int i = match_height + 1; i <= 64; i++) {
        idx_lookup = (lookup_bit(split_key, i) << 2) | (lookup_bit(key_min, i) << 1) | lookup_bit(key_max, i);
        if(idx_lookup == 0b001) {
            // max = load(0111, max)
            // bigmin = load(1000, min)
            key_max = load_in_box_split(key_max, false, i);
            bigmin = load_in_box_split(key_min, true, i);
        }
        else if(idx_lookup == 0b011) {
            // bigmin = min. Finish
            bigmin = key_min;
            break;
        }
        else if(idx_lookup == 0b100) {
            // litmax = max. Finish
            litmax = key_max;
            break;
        }
        else if(idx_lookup == 0b101) {
            // litmax = load(0111, max)
            // min = load(1000, min)
            key_min = load_in_box_split(key_min, true, i);
            litmax = load_in_box_split(key_max, false, i);
        }
    }
    return std::make_pair(litmax, bigmin);
}