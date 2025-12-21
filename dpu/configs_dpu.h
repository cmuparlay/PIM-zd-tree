#pragma once

#include "configs.h"

#define MAX_KNN_SIZE_DPU (125)

/* DPU Buffer Size */

#define B_BUFFER_SIZE (12 << 20) // 12 MB
#define P_BUFFER_SIZE (36 << 20) // 36 MB

#define MRAM_BUFFER_SIZE (3 << 19) // 1.5 MB

/* DPU Locks */

#define BNODE_LOCK_NUM (16)
#define BNODE_LOCK_NUM_MINUS_ONE (15)  // BNODE_LOCK_NUM_MINUS_ONE = BNODE_LOCK_NUM - 1
#define BNODE_LOCK_NUM_LOG (4)  // BNODE_LOCK_NUM = 1 << BNODE_LOCK_NUM_LOG
