#pragma once
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <alloc.h>

#include "common.h"
#include "task_framework_dpu.h"
#include "configs_dpu.h"
#include "geometry.h"

#ifdef BOX_RANGE_FETCH_ON

#define VARLEN_BUFFER_SIZE (60)
#define VARLEN_BUFFER_SIZE_IN_BYTES (512)  // VARLEN_BUFFER_SIZE_IN_BYTES = VARLEN_BUFFER_SIZE * sizeof(int64_t)

typedef struct varlen_buffer_in_mram {
    int64_t len; // total data size
    int llen; // WRAM data size
    int64_t ptr[VARLEN_BUFFER_SIZE];
    mpint64_t ptr_mram;
} varlen_buffer_in_mram;

static inline void varlen_buffer_in_mram_init(varlen_buffer_in_mram* buf, mpint64_t mptr) {
    buf->len = 0;
    buf->llen = 0;
    buf->ptr_mram = mptr;
}

static inline varlen_buffer_in_mram* varlen_buffer_in_mram_new(mpint64_t mptr) {
    varlen_buffer_in_mram* buf = (varlen_buffer_in_mram*)mem_alloc(sizeof(varlen_buffer_in_mram));
    varlen_buffer_in_mram_init(buf, mptr);
    return buf;
}

static inline void varlen_buffer_in_mram_push(varlen_buffer_in_mram* buf, int64_t v) {
    if(buf->llen == VARLEN_BUFFER_SIZE) {
        m_write(buf->ptr, (buf->ptr_mram + buf->len - VARLEN_BUFFER_SIZE), VARLEN_BUFFER_SIZE_IN_BYTES);
        buf->llen = 1;
        buf->ptr[0] = v;
    }
    else{
        buf->ptr[buf->llen] = v;
        buf->llen++;
    }
    buf->len++;
}

static inline void varlen_buffer_in_mram_to_mram(varlen_buffer_in_mram* buf, mpint64_t mptr, int len) {
    if(buf->len < len) len = buf->len;
    if(len > buf->len - buf->llen) {
        m_write(buf->ptr, (mptr + buf->len - buf->llen), S64(len - buf->len + buf->llen));
        len = buf->len - buf->llen;
    }
    mram_to_mram(mptr, buf->ptr_mram, S64(len));
}

static inline void varlen_buffer_in_mram_reset(varlen_buffer_in_mram* buf) {
    buf->len = 0;
    buf->llen = 0;
}

static inline int64_t varlen_buffer_in_mram_element(varlen_buffer_in_mram* buf, int64_t idx) {
    if(idx >= buf->len - buf->llen) {
        return buf->ptr[idx - buf->len + buf->llen];
    }
    else {
        int64_t res;
        m_read(buf->ptr_mram + idx, &res, 8);
        return res;
    }
}

static inline void varlen_buffer_in_mram_set_element(varlen_buffer_in_mram* buf, int64_t idx, int64_t value) {
    if(idx >= buf->len - buf->llen) {
        buf->ptr[idx - buf->len + buf->llen] = value;
    }
    else {
        m_write(&value, buf->ptr_mram + idx, 8);
    }
}

static inline void varlen_buffer_in_mram_push_bulk(varlen_buffer_in_mram* buf, void *v, int size_in_s64) {
    m_write(v, buf->ptr_mram + buf->len - buf->llen, S64(size_in_s64));
    buf->len += size_in_s64;
}

static inline void varlen_buffer_in_mram_push_vector(varlen_buffer_in_mram* buf, vectorT *v) {
    if(buf->llen > (int)(VARLEN_BUFFER_SIZE - sizeof(vectorT))) {
        m_write(buf->ptr, (buf->ptr_mram + buf->len - buf->llen), S64(buf->llen));
        buf->llen = 0;
    }
#if NR_DIMENSION == 2
    varlen_buffer_in_mram_push(buf, PPTR_TO_I64(v->x));
    varlen_buffer_in_mram_push(buf, PPTR_TO_I64(v->y));
#elif NR_DIMENSION == 3
    varlen_buffer_in_mram_push(buf, PPTR_TO_I64(v->x));
    varlen_buffer_in_mram_push(buf, PPTR_TO_I64(v->y));
    varlen_buffer_in_mram_push(buf, PPTR_TO_I64(v->z));
#else
    for(int i = 0; i < NR_DIMENSION; i++)
        varlen_buffer_in_mram_push(buf, PPTR_TO_I64(v->x[i]));
#endif
}

#endif
