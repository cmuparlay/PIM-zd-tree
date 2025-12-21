#pragma once

#define NR_DIMENSION (3)

/* Size of the batch size */
#define BATCH_SIZE (2100000)

/* Maximum sizes of box range and kNN queries */
#define MAX_RANGE_QUERY_SIZE (1000)
#define MAX_KNN_SIZE (125)

/* Size threshold of a leaf node in the tree */
#define LEAF_SIZE (16)

/* Block size of internal nodes */
#define DB_SIZE (16)  // size of the data block
#define DB_SIZE_LOG (4)  // DB_SIZE = 2 ^ DB_SIZE_LOG
#define DB_SIZE_LOG_LOG (2)  // DB_SIZE_LOG = 2 ^ DB_SIZE_LOG_LOG
#define DB_SIZE_PLUS_ONE (17)  // DB_SIZE_PLUS_ONE = DB_SIZE + 1

/* Coord <-----> Key */
// 64-bit int
// #define COORD_MAX (INT64_MAX)
// #define KEY_START_POS (2)  // The first position where in coord start to convert into key [1, 64]
// #define KEY_START_POS_MINUS_ONE (1)  // KEY_START_POS_MINUS_ONE = KEY_START_POS - 1
// // 32-bit int
#define COORD_MAX ((int64_t)INT32_MAX)
#define KEY_START_POS (34)  // The first position where in coord start to convert into key [1, 64]
#define KEY_START_POS_MINUS_ONE (33)  // KEY_START_POS_MINUS_ONE = KEY_START_POS - 1

#define LX_NORM (1)

#define MAX_TASK_BUFFER_SIZE_PER_DPU (6396 << 10) // 6.4 MB
#define MAX_TASK_COUNT_PER_DPU_PER_BLOCK ((100 << 10) >> 3) // 100 KB = 12.5 K
