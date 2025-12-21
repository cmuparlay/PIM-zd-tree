#pragma once

#include <stdint.h>
#include <assert.h>

#include "common.h"
#include "geometry_base.h"
#include "debug_settings.h"

// name
// id
// fixed : true for fixed
// length (expected)
// content
#ifndef TASK
#define TASK(NAME, ID, FIXED, LENGTH, CONTENT)
#endif

#define EMPTY 0
TASK(empty_task_reply, 0, true, 0, {})

/* -------------------------- Switches to enable tasks -------------------------- */
// #define DPU_INIT_ON
// #define INSERT_NODE_ON
// #define BOX_RANGE_COUNT_ON
// #define BOX_RANGE_FETCH_ON
// #define SEARCH_TEST_ON
// #define FETCH_NODE_ON
// #define DPU_STORAGE_STAT_ON


/* -------------------------- Single node queries -------------------------- */

#define SINGLE_SEARCH_TSK 101
TASK(Single_search_task, 101, true, sizeof(Single_search_task), {
    uint64_t key;
})

#define SINGLE_SEARCH_REP 102
TASK(Single_search_reply, 102, true, sizeof(Single_search_reply), {
    pptr addr;
})

#ifdef INSERT_NODE_ON
#define SINGLE_INSERT_TSK 103
TASK(Single_insert_task, 103, false, sizeof(Single_insert_task), {
    pptr addr;
    int64_t len;
    vectorT v[];
})
#define SINGLE_INSERT_TSK_SIZE(x) S64(2 + MULTIPLY_NR_DIMENSION(x))
#endif

#ifdef SEARCH_TEST_ON
#define SINGLE_KEY_SEARCH_TSK 104
TASK(Single_key_search_task, 104, true, sizeof(Single_key_search_task), {
    uint64_t key;
})

#define SINGLE_KEY_SEARCH_REP 105
TASK(Single_key_search_reply, 105, true, sizeof(Single_key_search_reply), {
    uint64_t key;
})
#endif


/* -------------------------- Single node debug tools -------------------------- */

#ifdef FETCH_NODE_ON
#define FETCH_NODE_W_KEY_TSK 106
TASK(Fetch_node_w_key_task, 106, true, sizeof(Fetch_node_w_key_task), {
    uint64_t key;
})
#define FETCH_NODE_W_PPTR_TSK 107
TASK(Fetch_node_w_pptr_task, 107, true, sizeof(Fetch_node_w_pptr_task), {
    pptr addr;
})
#define FETCH_NODE_REP 108
TASK(Fetch_node_reply, 108, true, sizeof(Fetch_node_reply), {
    pptr addr;
    pptr parent;
    uint64_t key;
    int64_t height;
    int64_t len;
    uint64_t keys[LEAF_SIZE];
})
#endif

#ifdef DPU_STORAGE_STAT_ON
#define DPU_STORAGE_STAT_TSK 109
TASK(dpu_storage_stat_task, 109, true, sizeof(dpu_storage_stat_task), {
    int64_t dpu_id;
})
#define DPU_STORAGE_STAT_REP 110
TASK(dpu_storage_stat_reply, 110, true, sizeof(dpu_storage_stat_reply), {
    int64_t bcnt;
    int64_t pcnt;
})
#endif


/* -------------------------- Box Range queries -------------------------- */

#ifdef BOX_RANGE_COUNT_ON

#define BOX_COUNT_TSK 201
TASK(Box_count_task, 201, true, sizeof(Box_count_task), {
    vectorT vec_min;
    vectorT vec_max;
})

#define BOX_COUNT_REP 202
TASK(Box_count_reply, 202, true, sizeof(Box_count_reply), {
    uint64_t count;
})

#endif

#ifdef BOX_RANGE_FETCH_ON

#define BOX_FETCH_TSK 203
TASK(Box_fetch_task, 203, true, sizeof(Box_fetch_task), {
    vectorT vec_min;
    vectorT vec_max;
})

#define BOX_FETCH_REP 204
TASK(Box_fetch_reply, 204, false, sizeof(Box_fetch_reply), {
    int64_t len;
    vectorT v[];
})
#define BOX_FETCH_REP_SIZE(x) S64(1 + MULTIPLY_NR_DIMENSION(x))

#endif


/* -------------------------- kNN queries -------------------------- */

#ifdef KNN_ON

#define KNN_TSK 301
TASK(knn_task, 301, true, sizeof(knn_task), {
    int64_t k;
    vectorT center;
})

#define KNN_REP 302
TASK(knn_reply, 302, false, sizeof(knn_reply), {
    int64_t len;
    vectorT v[];
})
#define KNN_REP_SIZE(x) S64(1 + MULTIPLY_NR_DIMENSION(x))

#define KNN_BOUNDED_TSK 303
TASK(knn_bounded_task, 303, true, sizeof(knn_bounded_task), {
    int64_t k;
    vectorT center;
    int64_t radius;
})

#endif


/* -------------------------- Util -------------------------- */
// #define STATISTICS_TSK 1001
// TASK(statistic_task, 1001, true, sizeof(statistic_task), { int64_t dpu_id; })

#ifdef DPU_INIT_ON

#define INIT_TSK 1002
TASK(dpu_init_task, 1002, true, sizeof(dpu_init_task), {
    int32_t dpu_id;
    int32_t nr_of_dpus;
})

#define INIT_REP 1003
TASK(dpu_init_reply, 1003, true, sizeof(dpu_init_reply), {})

#define INIT_RANGE_TSK 1004
TASK(dpu_init_range_task, 1004, true, sizeof(dpu_init_range_task), {
    uint64_t range_start;
    uint64_t range_end;
})

#endif