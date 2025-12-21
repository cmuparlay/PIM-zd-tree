#pragma once

#ifndef NR_DPUS
#define NR_DPUS (2560)
#endif

#ifndef NR_TASKLETS
#define NR_TASKLETS (16)
#endif

#define S32(x) ((x) << 2) // x * sizeof(int32_t)
#define S64(x) ((x) << 3) // x * sizeof(int64_t)

#define RETURN_X_IF_Y(f, x, y) \
    {                          \
        if ((f) == x) {        \
            return y;          \
        }                      \
    }

#define RETURN_FALSE_IF_FALSE(x) \
    { RETURN_X_IF_Y(x, false, false); }

#define RETURN_TRUE_IF_TRUE(x) \
    { RETURN_X_IF_Y(x, true, true); }
