#pragma once

#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <string.h>
#include "macro.h"
#include "utils.h"
#include "geometry.h"
#include "common.h"
#include "configs_dpu.h"

#define DPU_KEYS_STORED_IN_PNODE

/* -------------------------- Make Sure! sizeof(Everything) = 8x -------------------------- */
typedef __mram_ptr pptr* mppptr;
typedef __mram_ptr int64_t* mpint64_t;
typedef __mram_ptr uint8_t* mpuint8_t;
typedef __mram_ptr uint64_t* mpuint64_t;

typedef __mram_ptr void* mpvoid;

typedef __mram_ptr struct Bnode* mBptr;
typedef __mram_ptr struct Pnode* mPptr;

#define INVALID_MPPTR ((mPptr)-10)
#define INVALID_MBPTR ((mBptr)-10)

typedef struct Bnode {
    int16_t height;
    int16_t subtree_size;
    int32_t parent;
    uint64_t key;
    vectorT box_min __attribute__((aligned (8)));
    vectorT box_max __attribute__((aligned (8)));
    pptr children[DB_SIZE];
} Bnode;
#define BNODE_METADATA_SIZE S64(2 + MULTIPLY_NR_DIMENSION(2))

typedef struct Pnode {
    int16_t height;
    int16_t num;
    int32_t parent;
    uint64_t key;
    vectorT box_min __attribute__((aligned (8)));
    vectorT box_max __attribute__((aligned (8)));
    vectorT v[LEAF_SIZE];
#ifdef DPU_KEYS_STORED_IN_PNODE
    uint64_t keys[LEAF_SIZE];
#endif
} Pnode;
#define PNODE_METADATA_SIZE S64(2 + MULTIPLY_NR_DIMENSION(2))


/* Auxiliary Functions */

extern int64_t DPU_ID;
extern mBptr b_buffer;

#define INVALID_PARENT ((int32_t)-1)

static inline mBptr load_node_parent(int32_t parent_from_node) {
    return (parent_from_node == INVALID_PARENT ? INVALID_MBPTR : b_buffer + parent_from_node);
}

static inline int32_t store_node_parent(mBptr parent_to_node) {
    return (parent_to_node == INVALID_MBPTR ? INVALID_PARENT : (int32_t)(parent_to_node - b_buffer));
}

/* B nodes */

static inline pptr mbptr_to_pptr(const mBptr addr) {
    return ((addr == INVALID_MBPTR) ? null_pptr :
        (pptr) {
            .data_type = B_NODE_DATA_TYPE,
            .info = (int8_t)0,
            .id = DPU_ID,
            .addr = (uint32_t)(addr - b_buffer),
        }
    );
}

static inline mBptr pptr_to_mbptr(pptr x) {
    return (valid_pptr(x) ? (mBptr)(b_buffer + x.addr) : INVALID_MBPTR);
}

static inline int8_t get_mbptr_child_exist(mBptr addr, int idx) {
    return valid_pptr(addr->children[idx]);
}

static inline int8_t get_mbptr_child_also_b(mBptr addr, int idx) {
    pptr child = addr->children[idx];
    return child.data_type == B_NODE_DATA_TYPE;
}

/* P nodes */

extern mPptr p_buffer;

static inline pptr mpptr_to_pptr(const mPptr addr) {
    return ((addr == INVALID_MPPTR) ? null_pptr :
        (pptr) {
            .data_type = P_NODE_DATA_TYPE,
            .info = (int8_t)0,
            .id = DPU_ID,
            .addr = (uint32_t)(addr - p_buffer),
        }
    );
}

static inline mPptr pptr_to_mpptr(pptr x) {
    return (valid_pptr(x) ? (mPptr)(p_buffer + x.addr) : INVALID_MPPTR);
}
