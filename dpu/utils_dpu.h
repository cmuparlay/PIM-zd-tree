#pragma once
#include <stdint.h>
#include <stdio.h>
#include "utils.h"
#include "configs_dpu.h"
#include <mutex_pool.h>

/* Locks for BNodes */

MUTEX_POOL_INIT(bnode_lock_pool, BNODE_LOCK_NUM);

// A random hash function to decide the ID in the mutex_pool
static inline int bnode_mutex_hash(mBptr addr) {
    return (int)(addr - b_buffer) & BNODE_LOCK_NUM_MINUS_ONE;
}
