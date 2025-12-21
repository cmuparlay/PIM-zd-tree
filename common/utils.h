#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "macro_common.h"
#include "common.h"
#include "geometry_base.h"

// Takes in an integer and a position in said integer and returns whether the bit at that position is 0 or 1 (range [1, 64])
static inline bool lookup_bit(uint64_t interleave_integer, INT_HEIGHT pos){
    return (interleave_integer & (((uint64_t)1) << (64 - pos)));
}

// Set a bit in an integer to be 1 (range [1, 64])
static inline uint64_t set_bit_pos(uint64_t interleave_integer, INT_HEIGHT pos){
    return (interleave_integer | (((uint64_t)1) << (64 - pos)));
}

// Reset a bit in an integer to be 0 (range [1, 64])
static inline uint64_t reset_bit_pos(uint64_t interleave_integer, INT_HEIGHT pos){
    return (interleave_integer & ~(((uint64_t)1) << (64 - pos)));
}

static inline INT_HEIGHT prune_height(INT_HEIGHT height) {
    return ((height >> DB_SIZE_LOG_LOG) << DB_SIZE_LOG_LOG);
}

// Return the 4 bits in the next group of pos (range [1, 64])
static inline int lookup_next_bit_chunk(uint64_t interleave_integer, INT_HEIGHT pos) {
    // group_idx = (pos >> DB_SIZE_LOG_LOG);
    // Needs to fetch [group_idx*DB_SIZE_LOG+1, group_idx*DB_SIZE_LOG+DB_SIZE_LOG]
    if(pos > 64 - DB_SIZE_LOG) return interleave_integer & (uint64_t)(DB_SIZE - 1);
    return (interleave_integer >> (64 - DB_SIZE_LOG - prune_height(pos))) & (uint64_t)(DB_SIZE - 1);
}

/*
    Check the maximum leading bits that matches between two keys.
    Pruned with blocks of 4 bits.
*/
static inline INT_HEIGHT maximum_match_height(uint64_t key1, uint64_t key2) {
    if(key1 == key2) return 64;
    else return prune_height(__builtin_clzll(key1 ^ key2));
}

/*
    Check the maximum leading bits that matches between two keys.
    Precise heights returned, not pruned with 4-bit blocks.
*/
static inline INT_HEIGHT maximum_match_height_precise(uint64_t key1, uint64_t key2) {
    if(key1 == key2) return 64;
    else return __builtin_clzll(key1 ^ key2);
}

// Check whether two integers match for bits leq pos (range [1, 64])
static inline bool check_match_height(uint64_t key1, uint64_t key2, INT_HEIGHT pos) {
    if(pos >= 64) return key1 == key2;
    else if(pos == 0) return true;
    else return ((key1 ^ key2) >> (64 - pos)) == (uint64_t)0;
}

// Prune the tail bits of a key to 0s (Input range [1, 64])
static inline uint64_t prune_tail_bits(uint64_t key, INT_HEIGHT pos) {
    if(pos == 0) return (uint64_t)0;
    else return ((key >> (64 - pos)) << (64 - pos));
}


/* The next section is for Morton Ordering */

// Magic Numbers for Morton Ordering
#if NR_DIMENSION == 2
static inline uint64_t split_by_two(COORD x) {
    uint64_t key = x & 0xffffffff;
    key = (key | (key << 16)) & 0x0000ffff0000ffff;
    key = (key | (key << 8))  & 0x00ff00ff00ff00ff;
    key = (key | (key << 4))  & 0x0f0f0f0f0f0f0f0f;
    key = (key | (key << 2))  & 0x3333333333333333;
    key = (key | (key << 1))  & 0x5555555555555555;
    return key;
}
#elif NR_DIMENSION == 3
static inline uint64_t split_by_three(COORD x) {
    uint64_t key = x & 0x1fffff;
    key = (key | (key << 32)) & 0x001f00000000ffff;
    key = (key | (key << 16)) & 0x001f0000ff0000ff;
    key = (key | (key << 8))  & 0x100f00f00f00f00f;
    key = (key | (key << 4))  & 0x10c30c30c30c30c3;
    key = (key | (key << 2))  & 0x1249249249249249;
    return key;
}
#endif

// Morton Ordering
static inline uint64_t coord_to_key(vectorT *v) {
#if NR_DIMENSION == 2
    uint64_t x = split_by_two(COORD_PART_BY_TWO_IDX > KEY_START_POS ? ((v->x) >> (COORD_PART_BY_TWO_IDX - KEY_START_POS)) : ((v->x) << (KEY_START_POS - COORD_PART_BY_TWO_IDX))),
             y = split_by_two(COORD_PART_BY_TWO_IDX > KEY_START_POS ? ((v->y) >> (COORD_PART_BY_TWO_IDX - KEY_START_POS)) : ((v->y) << (KEY_START_POS - COORD_PART_BY_TWO_IDX)));
    return (x << 1) | y;
#elif NR_DIMENSION == 3
    uint64_t x = split_by_three(COORD_PART_BY_THREE_IDX > KEY_START_POS ? ((v->x) >> (COORD_PART_BY_THREE_IDX - KEY_START_POS)) : ((v->x) << (KEY_START_POS - COORD_PART_BY_THREE_IDX))),
             y = split_by_three(COORD_PART_BY_THREE_IDX > KEY_START_POS ? ((v->y) >> (COORD_PART_BY_THREE_IDX - KEY_START_POS)) : ((v->y) << (KEY_START_POS - COORD_PART_BY_THREE_IDX))),
             z = split_by_three(COORD_PART_BY_THREE_IDX > KEY_START_POS ? ((v->z) >> (COORD_PART_BY_THREE_IDX - KEY_START_POS)) : ((v->z) << (KEY_START_POS - COORD_PART_BY_THREE_IDX)));
    return (x << 3) | (y << 2) | (z << 1);
#else
    uint64_t res = 0;
    int i = 1, j = KEY_START_POS;
    int k;
    while(true) {
        for(k = 0; k < NR_DIMENSION; k++) {
            if(lookup_bit(v->x[k], j))
                res = set_bit_pos(res, i);
            i++;
            if(i > 64) return res;
        }
        j++;
        if(j > 64) return res;
    };
    return res;
#endif
}

// Reverse Magic Numbers for Reverse Morton Ordering
#if NR_DIMENSION == 2
static inline uint64_t merge_by_two(uint64_t key) {
    key &= 0xaaaaaaaaaaaaaaaa;
    key = (key | (key << 1))  & 0xcccccccccccccccc;
    key = (key | (key << 2))  & 0xf0f0f0f0f0f0f0f0;
    key = (key | (key << 4))  & 0xff00ff00ff00ff00;
    key = (key | (key << 8))  & 0xffff0000ffff0000;
    key = (key | (key << 16)) & 0xffffffff00000000;
    return key;
}
#elif NR_DIMENSION == 3
static inline uint64_t merge_by_three(uint64_t key) {
    key &= 0x9249249249249249;
    key = (key | (key << 2))  & 0xc30c30c30c30c30c;
    key = (key | (key << 4))  & 0xf00f00f00f00f00f;
    key = (key | (key << 8))  & 0xff0000ff0000ff00;
    key = (key | (key << 16)) & 0xffff00000000ffff;
    key = (key | (key << 32)) & 0xffffffff00000000;
    return key;
}
#endif

// Reverse Morton Ordering
static inline vectorT key_to_coord(uint64_t key, bool fill_with_one) {
    vectorT v;
#if NR_DIMENSION == 2
    v.x = merge_by_two(key) >> KEY_START_POS_MINUS_ONE;
    v.y = merge_by_two(key << 1) >> KEY_START_POS_MINUS_ONE;
#if KEY_START_POS_MINUS_ONE < 32
    if(fill_with_one) {
        uint64_t mask = UINT64_MAX >> (32 + KEY_START_POS_MINUS_ONE);
        v.x |= mask;
        v.y |= mask;
    }
#endif
#elif NR_DIMENSION == 3
    v.x = merge_by_three(key) >> KEY_START_POS_MINUS_ONE;
    v.y = merge_by_three(key << 1) >> KEY_START_POS_MINUS_ONE;
    v.z = merge_by_three(key << 2) >> KEY_START_POS_MINUS_ONE;
#if KEY_START_POS_MINUS_ONE < 43
    if(fill_with_one) {
        uint64_t mask = UINT64_MAX >> (21 + KEY_START_POS_MINUS_ONE);
        v.x |= mask;
        v.y |= mask;
        v.z |= mask;
    }
#endif
#else
    vector_ones(&v, 0);
    int i = 1, j = KEY_START_POS;
    int k;
    bool key_unfinished = true;
    while(true) {
        if(j > 64) return v;
        for(k = 0; k < NR_DIMENSION; k++) {
            if(key_unfinished) {
                if(lookup_bit(key, i))
                    v.x[k] = set_bit_pos(v.x[k], j);
                i++;
                if(i > 64) {
                    key_unfinished = false;
                    if(!fill_with_one) return v;
                }
            }
            else {
                v.x[k] = set_bit_pos(v.x[k], j);
            }
        }
        j++;
    };
#endif
    return v;
}
