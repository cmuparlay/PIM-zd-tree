#pragma once
#include "configs.h"

// #define LX_NORM_ON_DPU

/* Data types */
#define DATA_TYPE_NUM ((uint8_t) 4)
#define DATA_TYPE_NUM_LOG (2)

#include "pptr.h"

#define B_NODE_DATA_TYPE ((uint8_t) 1)
#define P_NODE_DATA_TYPE ((uint8_t) 2)


/* Macros for node heights */
#define INT_HEIGHT int8_t
#define INVALID_NODE_HEIGHT ((INT_HEIGHT) -2)
#define ROOT_NODE_HEIGHT ((INT_HEIGHT) 0)

/* Used for Morton Ordering for 64-bit INT */
#define COORD_PART_BY_TWO_IDX (33)
#define COORD_PART_BY_THREE_IDX (44)

/* Auxilary functions to avoid multiplications */
#define MULTIPLY_DB_SIZE(x) ((x) << DB_SIZE_LOG)

#if NR_DIMENSION == 0
#define MULTIPLY_NR_DIMENSION(x) (0)
#elif NR_DIMENSION == 1
#define MULTIPLY_NR_DIMENSION(x) (x)
#elif NR_DIMENSION == 2
#define MULTIPLY_NR_DIMENSION(x) ((x) << 1)
#elif NR_DIMENSION == 3
#define MULTIPLY_NR_DIMENSION(x) (((x) << 1) + (x))
#elif NR_DIMENSION == 4
#define MULTIPLY_NR_DIMENSION(x) ((x) << 2)
#elif NR_DIMENSION == 5
#define MULTIPLY_NR_DIMENSION(x) (((x) << 2) + (x))
#elif NR_DIMENSION == 6
#define MULTIPLY_NR_DIMENSION(x) (((x) << 2) + ((x) << 1))
#elif NR_DIMENSION == 7
#define MULTIPLY_NR_DIMENSION(x) (((x) << 3) - (x))
#elif NR_DIMENSION == 8
#define MULTIPLY_NR_DIMENSION(x) ((x) << 3)
#elif NR_DIMENSION == 9
#define MULTIPLY_NR_DIMENSION(x) (((x) << 3) + (x))
#elif NR_DIMENSION == 10
#define MULTIPLY_NR_DIMENSION(x) (((x) << 3) + ((x) << 1))
#else
#define MULTIPLY_NR_DIMENSION(x) ((x) * NR_DIMENSION)
#endif

#define CPU_DPU_CONSENSUS_NO (114514)

/* Structure used by both the host and the dpu to communicate information */

static inline int lb(int64_t x) { return x & (-x); }

static inline int hh_dpu(int64_t key, uint64_t M) { return key & (M - 1); }

static inline int hash_to_addr(int64_t key, uint64_t M) {
    return hh_dpu(key, M);
}
