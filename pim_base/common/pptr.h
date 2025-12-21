#pragma once

#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <stdbool.h>

#define DEFAULT_DATA_TYPE ((uint8_t) 0)
#define INVALID_DATA_TYPE (UINT8_MAX)
#define INVALID_DPU_ID (UINT16_MAX)
#define INVALID_DPU_ADDR (UINT32_MAX)

typedef struct pptr {
    uint8_t data_type;
    int8_t info;
    uint16_t id;
    uint32_t addr;
} pptr __attribute__((aligned (8)));

typedef struct offset_pptr {
    uint16_t id;
    uint16_t offset;
    uint32_t addr;
} offset_pptr __attribute__((aligned(8)));

const pptr null_pptr = (pptr){
    .data_type = INVALID_DATA_TYPE,
    .info = 0,
    .id = INVALID_DPU_ID,
    .addr = INVALID_DPU_ADDR,
};

#ifndef DATA_TYPE_NUM
#define DATA_TYPE_NUM (1)
#define DATA_TYPE_NUM_LOG (0)
#endif
