#include <stdint.h>
#include <stdio.h>

#include "driver.h"
#include "task_utils.h"
#include "common.h"

#include "configs_dpu.h"
#include "storage.h"
#include "buffer_dpu.h"
#include "single_node.h"
#include "box_range.h"
#include "knn.h"

BARRIER_INIT(exec_barrier, NR_TASKLETS);

MUTEX_INIT(dpu_lock);

extern volatile int64_t recv_epoch_number;

/* -------------- Storage -------------- */

// Node Buffers & Hash Tables

#ifdef DPU_ENERGY
__host uint64_t op_count;
__host uint64_t db_size_count;
__host uint64_t cycle_count;
#endif

void execute(int l, int r) {
    uint32_t tasklet_id = me();
    // int length = r - l;

    switch (recv_block_task_type) {
#ifdef DPU_INIT_ON
        case INIT_TSK: {
            init_block_with_type(dpu_init_task, empty_task_reply);
            if (tasklet_id == 0) {
                init_task_reader(0);
                dpu_init_task it = *((dpu_init_task*)get_task_cached(0));
                dpu_init_func(it.dpu_id, it.nr_of_dpus);
                bnode_init();
            }
            break;
        }

        case INIT_RANGE_TSK: {
            init_block_with_type(dpu_init_range_task, empty_task_reply);
            if (tasklet_id == 0) {
                init_task_reader(0);
                dpu_init_range_task it = *((dpu_init_range_task*)get_task_cached(0));
                local_range_start = it.range_start;
                local_range_end = it.range_end;
                range_per_dpu = local_range_end - local_range_start;
            }
            break;
        }
#endif

#ifdef SEARCH_TEST_ON
        case SINGLE_KEY_SEARCH_TSK: {
            init_block_with_type(Single_key_search_task, Single_key_search_reply);
            init_task_reader(l);
            uint64_t key;
            Single_key_search_task* tsk;
            Single_key_search_reply tsr;
            pptr addr;
            for (int i = l; i < r; i++) {
                tsk = (Single_key_search_task*)get_task_cached(i);
                key = tsk->key;
                addr = b_search(key, true);
                if(addr.data_type == P_NODE_DATA_TYPE) {
                    tsr.key = p_search(pptr_to_mpptr(addr), key);
                }
                else tsr.key = INVALID_KEY;
                push_fixed_reply(i, &tsr);
            }
            break;
        }
#endif

#ifdef INSERT_NODE_ON
        case SINGLE_SEARCH_TSK: {
            init_block_with_type(Single_search_task, Single_search_reply);
            init_task_reader(l);
            uint64_t key;
            Single_search_task* tsk;
            Single_search_reply tsr;
            for (int i = l; i < r; i++) {
                tsk = (Single_search_task*)get_task_cached(i);
                key = tsk->key;
                tsr.addr = b_search(key, true);
                push_fixed_reply(i, &tsr);
            }
            break;
        }

        case SINGLE_INSERT_TSK: {
            init_block_with_type(Single_insert_task, empty_task_reply);
            int buf_size = MRAM_BUFFER_SIZE / NR_TASKLETS;
            mpvoid buf = (mpvoid)mrambuffer + buf_size * tasklet_id;
            uint64_t key_buf_wram[INSERT_WRAM_KEY_BUF_SIZE];
            for (int i = l; i < r; i++) {
                __mram_ptr Single_insert_task* tsk = (__mram_ptr Single_insert_task*)get_task(i);
                pptr addr = tsk->addr;
                int len = tsk->len;
                mBptr b_addr;
                if(addr.data_type == P_NODE_DATA_TYPE) {
                    mPptr p_addr = pptr_to_mpptr(addr);
                    b_addr = load_node_parent(p_addr->parent);
                    p_insert(p_addr, addr.info, len, tsk->v, buf, buf_size, key_buf_wram);
                    maintain_ancestor_counter(b_addr, len);
                }
                else if(addr.data_type == B_NODE_DATA_TYPE) {
                    b_addr = pptr_to_mbptr(addr);
                    b_addr = b_insert(b_addr, addr.info, len, tsk->v, buf, buf_size, key_buf_wram);
                    maintain_ancestor_counter(b_addr, len);
                }
            }
            break;
        }
#endif

#ifdef BOX_RANGE_COUNT_ON
        case BOX_COUNT_TSK: {
            init_block_with_type(Box_count_task, Box_count_reply);
            init_task_reader(l);
            int buf_size = MRAM_BUFFER_SIZE / NR_TASKLETS;
            mpvoid buf = (mpvoid)mrambuffer + buf_size * tasklet_id;
            Box_count_task tsk;
            Box_count_reply tsr;
            for (int i = l; i < r; i++) {
                tsk = *((Box_count_task*)get_task_cached(i));
                tsr.count = box_range_count(&(tsk.vec_min), &(tsk.vec_max), buf);
                push_fixed_reply(i, &tsr);
            }
            break;
        }
#endif

#ifdef BOX_RANGE_FETCH_ON
        case BOX_FETCH_TSK: {
            init_block_with_type(Box_fetch_task, Box_fetch_reply);
            init_task_reader(l);
            Box_fetch_task tsk;
            int buf_size = MRAM_BUFFER_SIZE / NR_TASKLETS;
            mpvoid buf = (mpvoid)mrambuffer + buf_size * tasklet_id;
            int buf_size2 = buf_size / (MULTIPLY_DB_SIZE(NR_DIMENSION) >> 1);
            varlen_buffer_in_mram *varlen_buf;
            varlen_buf = varlen_buffer_in_mram_new(buf + buf_size2);
            int64_t num;
            for (int i = l; i < r; i++) {
                tsk = *((Box_fetch_task*)get_task_cached(i));
                num = box_range_fetch(&(tsk.vec_min), &(tsk.vec_max), varlen_buf, buf);
                IN_DPU_ASSERT(varlen_buf->len == MULTIPLY_NR_DIMENSION(num), "Box fetch err\n");
                __mram_ptr Box_fetch_reply *replyptr = (__mram_ptr Box_fetch_reply*)push_variable_reply_zero_copy(tasklet_id, BOX_FETCH_REP_SIZE(num));
                replyptr->len = num;
                varlen_buffer_in_mram_to_mram(varlen_buf, (mpint64_t)(replyptr->v), varlen_buf->len);
                varlen_buffer_in_mram_reset(varlen_buf);
            }
            break;
        }
#endif

#ifdef KNN_ON
        case KNN_TSK: {}
        case KNN_BOUNDED_TSK: {
            if(recv_block_task_type == KNN_TSK) {init_block_with_type(knn_task, knn_reply);}
            else {init_block_with_type(knn_bounded_task, knn_reply);}
            init_task_reader(l);
            COORD radius;
            int buf_size = MRAM_BUFFER_SIZE / NR_TASKLETS;
            mpvoid buf = (mpvoid)mrambuffer + buf_size * tasklet_id;
            heap_dpu *heap = heap_dpu_new(0, buf);
            buf += S64(MAX_KNN_SIZE_DPU + MULTIPLY_NR_DIMENSION(MAX_KNN_SIZE_DPU));
            buf_size -= S64(MAX_KNN_SIZE_DPU + MULTIPLY_NR_DIMENSION(MAX_KNN_SIZE_DPU));
            knn_task knn_tsk;
            uint8_t k_max;
            for (int i = l; i < r; i++) {
                knn_tsk = *((knn_task*)get_task_cached(i));
                radius = ((recv_block_task_type == KNN_TSK) ? INT64_MAX : ((knn_bounded_task*)get_task_cached(i))->radius);
#if (LX_NORM == 2) && !defined(LX_NORM_ON_DPU)
                k_max = knn_tsk.k + (knn_tsk.k >> 1);
#else
                k_max = knn_tsk.k;
#endif
                heap_dpu_init(heap, GEOMETRY_MIN(k_max, MAX_KNN_SIZE_DPU));
                knn(&knn_tsk.center, radius, heap, buf, buf_size);
                __mram_ptr knn_reply *replyptr = (__mram_ptr knn_reply*)push_variable_reply_zero_copy(tasklet_id, KNN_REP_SIZE(heap->num));
#if (LX_NORM == 2) && !defined(LX_NORM_ON_DPU)
                replyptr->len = heap->num;
#else
                /*
                    If it is knn_task, then must return k values. rep->len is useless, thus we are storing radius here.
                    Else if it is knn_bounded_task, then still store len.
                */
                if(recv_block_task_type == KNN_TSK) {
                    radius = heap->distance_storage[heap->arr[0]];
                    if(knn_first_round_finished(&knn_tsk.center, radius)) replyptr->len = -1;
                    else replyptr->len = radius;
                }
                else replyptr->len = heap->num;
#endif
                if(heap->num > 0) mram_to_mram(replyptr->v, heap->vector_storage, S64(MULTIPLY_NR_DIMENSION(heap->num)));
            }
            break;
        }
#endif

#ifdef FETCH_NODE_ON
        case FETCH_NODE_W_KEY_TSK: {
            init_block_with_type(Fetch_node_w_key_task, Fetch_node_reply);
            init_task_reader(l);
            uint64_t key;
            Fetch_node_w_key_task* tsk;
            pptr addr;
            for (int i = l; i < r; i++) {
                tsk = (Fetch_node_w_key_task*)get_task_cached(i);
                key = tsk->key;
                addr = b_search(key, false);
                fetch_single_node(addr, i);
            }
            break;
        }

        case FETCH_NODE_W_PPTR_TSK: {
            init_block_with_type(Fetch_node_w_pptr_task, Fetch_node_reply);
            init_task_reader(l);
            Fetch_node_w_pptr_task* tsk;
            pptr addr;
            for (int i = l; i < r; i++) {
                tsk = (Fetch_node_w_pptr_task*)get_task_cached(i);
                addr = tsk->addr;
                fetch_single_node(addr, i);
            }
            break;
        }
#endif

#ifdef DPU_STORAGE_STAT_ON
        case DPU_STORAGE_STAT_TSK: {
            init_block_with_type(dpu_storage_stat_task, dpu_storage_stat_reply);
            if (tasklet_id == 0) {
                init_task_reader(0);
                dpu_storage_stat_reply tsr;
                tsr.bcnt = bcnt;
                tsr.pcnt = pcnt;
                push_fixed_reply(0, &tsr);
            }
            break;
        }
#endif

        default: {
            // printf("TT = %lld\n", recv_block_task_type);
            // IN_DPU_ASSERT(false, "WTT\n");
            break;
        }
    }

    finish_reply(recv_block_task_cnt, tasklet_id);
}

void init() {
    
}

int main() {
    run();
}