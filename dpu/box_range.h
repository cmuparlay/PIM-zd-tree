#pragma once
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>

#include "macro.h"
#include "task_utils.h"
#include "task_framework_dpu.h"
#include "node_dpu.h"
#include "storage.h"
#include "utils_dpu.h"
#include "buffer_dpu.h"
#include "geometry.h"

#define BOX_QUERY_WRAM_BUFFER_SIZE (10)

bool vector_in_box(vectorT *v, vectorT *box_min, vectorT *box_max);
bool box_intersect(vectorT *box1_min, vectorT *box1_max, vectorT *box2_min, vectorT *box2_max);
bool box_contain(vectorT *small_box_min, vectorT *small_box_max, vectorT *large_box_min, vectorT *large_box_max);

/* Count nr_points in Box Range Queries */
#ifdef BOX_RANGE_COUNT_ON
static inline uint64_t box_range_count(vectorT *vec_min, vectorT *vec_max, mpvoid buf) {
    uint64_t nr_count = 0;
    mppptr pptr_buf_mram = (mppptr)buf;
    pptr pptr_buf_wram[BOX_QUERY_WRAM_BUFFER_SIZE];
    int pptr_mram_num = 0, pptr_wram_num = 1;
    pptr_buf_wram[0] = mbptr_to_pptr(root);
    pptr addr;
    mBptr b_addr;
    mPptr p_addr;
    Bnode bnode;
    Pnode pnode;
    int64_t i;
    bool to_contunue_signal;
    while(pptr_wram_num > 0 || pptr_mram_num > 0) {
        if(pptr_wram_num > 0) {
            pptr_wram_num--;
            addr = pptr_buf_wram[pptr_wram_num];
        }
        else {
            m_read(pptr_buf_mram + pptr_mram_num - (BOX_QUERY_WRAM_BUFFER_SIZE >> 1), pptr_buf_wram, S64(BOX_QUERY_WRAM_BUFFER_SIZE >> 1));
            pptr_wram_num = (BOX_QUERY_WRAM_BUFFER_SIZE >> 1) - 1;
            addr = pptr_buf_wram[pptr_wram_num];
            pptr_mram_num -= (BOX_QUERY_WRAM_BUFFER_SIZE >> 1);
        }
        if(addr.data_type == P_NODE_DATA_TYPE) {
            p_addr = pptr_to_mpptr(addr);
            m_read(p_addr, &pnode, PNODE_METADATA_SIZE);
            to_contunue_signal = box_intersect(&pnode.box_min, &pnode.box_max, vec_min, vec_max);
        }
        else if(addr.data_type == B_NODE_DATA_TYPE) {
            b_addr = pptr_to_mbptr(addr);
            m_read(b_addr, &bnode, BNODE_METADATA_SIZE);
            to_contunue_signal = box_intersect(&bnode.box_min, &bnode.box_max, vec_min, vec_max);
        }
        if(to_contunue_signal) {
            if(addr.data_type == P_NODE_DATA_TYPE) {
                if(box_contain(&pnode.box_min, &pnode.box_max, vec_min, vec_max)) {
                    nr_count += pnode.num;
                }
                else {
                    m_read(p_addr->v, pnode.v, S64(MULTIPLY_NR_DIMENSION(pnode.num)));
                    for(i = 0; i < pnode.num; i++) {
                        if(vector_in_box(pnode.v + i, vec_min, vec_max))
                            nr_count++;
                    }
                }
            }
            else if(addr.data_type == B_NODE_DATA_TYPE) {
                if(bnode.subtree_size < MAX_RANGE_QUERY_SIZE && box_contain(&bnode.box_min, &bnode.box_max, vec_min, vec_max)) {
                    nr_count += bnode.subtree_size;
                }
                else {
                    m_read(b_addr->children, bnode.children, S64(DB_SIZE));
                    for(i = 0; i < DB_SIZE; i++) {
                        addr = bnode.children[i];
                        if(valid_pptr(addr)) {
                            if(pptr_wram_num < BOX_QUERY_WRAM_BUFFER_SIZE) {
                                pptr_buf_wram[pptr_wram_num] = addr;
                                pptr_wram_num++;
                            }
                            else {
                                m_write(pptr_buf_wram, pptr_buf_mram + pptr_mram_num, S64(pptr_wram_num));
                                pptr_mram_num += pptr_wram_num;
                                pptr_buf_wram[0] = addr;
                                pptr_wram_num = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    return nr_count;
}
#endif

/* Fetch all points in Box Range Queries */
#ifdef BOX_RANGE_FETCH_ON

static inline int check_fetch_pnode_to_buffer(bool fetch_all, Pnode *pnode_pt, vectorT *vec_min, vectorT *vec_max, varlen_buffer_in_mram *varlen_buf) {
    if(fetch_all) {
        varlen_buffer_in_mram_push_bulk(varlen_buf, pnode_pt->v, MULTIPLY_NR_DIMENSION(pnode_pt->num));
        return pnode_pt->num;
    }
    else {
        vectorT *vec = pnode_pt->v;
        int nr_count = 0;
        for(int8_t i = 0; i < pnode_pt->num; i++) {
            if(vector_in_box(vec, vec_min, vec_max)) {
                varlen_buffer_in_mram_push_vector(varlen_buf, vec);
                nr_count++;
            }
            vec++;
        }
        return nr_count;
    }
}

static inline int box_range_fetch(vectorT *vec_min, vectorT *vec_max, varlen_buffer_in_mram *varlen_buf, mpvoid buf) {
    int nr_count = 0;
    mppptr pptr_buf_mram = (mppptr)buf;
    pptr pptr_buf_wram[BOX_QUERY_WRAM_BUFFER_SIZE];
    int pptr_mram_num = 0, pptr_wram_num = 1;
    pptr_buf_wram[0] = mbptr_to_pptr(root);
    pptr addr;
    mBptr b_addr;
    mPptr p_addr;
    Bnode bnode;
    Pnode pnode;
    bool fetch_all, to_contunue_signal;
    int i;
    while(pptr_wram_num > 0 || pptr_mram_num > 0) {
        if(pptr_wram_num > 0) {
            pptr_wram_num--;
            addr = pptr_buf_wram[pptr_wram_num];
        }
        else {
            m_read(pptr_buf_mram + pptr_mram_num - (BOX_QUERY_WRAM_BUFFER_SIZE >> 1), pptr_buf_wram, S64(BOX_QUERY_WRAM_BUFFER_SIZE >> 1));
            pptr_wram_num = (BOX_QUERY_WRAM_BUFFER_SIZE >> 1) - 1;
            addr = pptr_buf_wram[pptr_wram_num];
            pptr_mram_num -= (BOX_QUERY_WRAM_BUFFER_SIZE >> 1);
        }
        fetch_all = addr.info != 0;
        if(addr.data_type == P_NODE_DATA_TYPE) {
            p_addr = pptr_to_mpptr(addr);
            to_contunue_signal = fetch_all;
            if(!fetch_all) {
                m_read(p_addr, &pnode, PNODE_METADATA_SIZE);
                to_contunue_signal = box_intersect(&pnode.box_min, &pnode.box_max, vec_min, vec_max);
                fetch_all = box_contain(&pnode.box_min, &pnode.box_max, vec_min, vec_max);
            }
            if(to_contunue_signal) {
                m_read(p_addr->v, pnode.v, S64(MULTIPLY_NR_DIMENSION(pnode.num)));
                nr_count += check_fetch_pnode_to_buffer(fetch_all, &pnode, vec_min, vec_max, varlen_buf);
            }
        }
        else if(addr.data_type == B_NODE_DATA_TYPE) {
            b_addr = pptr_to_mbptr(addr);
            if(!fetch_all) {
                m_read(b_addr, &bnode, BNODE_METADATA_SIZE);
                to_contunue_signal = box_intersect(&bnode.box_min, &bnode.box_max, vec_min, vec_max);
                fetch_all = box_contain(&bnode.box_min, &bnode.box_max, vec_min, vec_max);
            }
            if(to_contunue_signal) {
                m_read(b_addr->children, bnode.children, S64(DB_SIZE));
                for(i = 0; i < DB_SIZE; i++) {
                    addr = bnode.children[i];
                    if(valid_pptr(addr)) {
                        addr.info = (int8_t)fetch_all;
                        if(pptr_wram_num < BOX_QUERY_WRAM_BUFFER_SIZE) {
                            pptr_buf_wram[pptr_wram_num] = addr;
                            pptr_wram_num++;
                        }
                        else {
                            m_write(pptr_buf_wram, pptr_buf_mram + pptr_mram_num, S64(pptr_wram_num));
                            pptr_mram_num += pptr_wram_num;
                            pptr_buf_wram[0] = addr;
                            pptr_wram_num = 1;
                        }
                    }
                }
            }
        }
    }
    return nr_count;
}

#endif