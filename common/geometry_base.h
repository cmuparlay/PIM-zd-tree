#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "common.h"

#ifndef COORD
#define COORD int64_t
#endif

#define INVALID_KEY (UINT64_MAX)
#define INVALID_COORD (UINT64_MAX)

#define GEOMETRY_MAX(x, y) (((x) < (y)) ? (y) : (x))
#define GEOMETRY_MIN(x, y) (((x) < (y)) ? (x) : (y))

#define L2_NORM_MAX (2147483648)

static inline int64_t approx_square(int64_t x) {
    int64_t ret = 0;
    int64_t y = x;
    int8_t a;
    a = 63 - __builtin_clzll(y);
    if(a <= 0) return ret;
    ret += x << a;
    y -= 1 << a;
    a = 63 - __builtin_clzll(y);
    if(a <= 0) return ret;
    ret += x << a;
    y -= 1 << a;
    a = 63 - __builtin_clzll(y);
    if(a <= 0) return ret;
    ret += x << a;
    y -= 1 << a;
    a = 63 - __builtin_clzll(y);
    if(a <= 0) return ret;
    ret += x << a;
    return ret;
}

#if NR_DIMENSION == 2

struct vector2D {
    COORD x;
    COORD y;
};

inline struct vector2D vector_add(struct vector2D *op1, struct vector2D *op2) {
    struct vector2D res;
    res.x = ((op1->x < COORD_MAX - op2->x) ? (op1->x + op2->x) : COORD_MAX);
    res.y = ((op1->y < COORD_MAX - op2->y) ? (op1->y + op2->y) : COORD_MAX);
    return res;
}

inline struct vector2D vector_sub(struct vector2D *op1, struct vector2D *op2) {
    struct vector2D res;
    res.x = llabs(op1->x - op2->x);
    res.y = llabs(op1->y - op2->y);
    return res;
}

inline struct vector2D vector_sub_zero_bounded(struct vector2D *op1, struct vector2D *op2) {
    struct vector2D res;
    res.x = ((op1->x > op2->x) ? (op1->x - op2->x) : 0);
    res.y = ((op1->y > op2->y) ? (op1->y - op2->y) : 0);
    return res;
}

#if LX_NORM == 0
inline COORD vector_norm_dpu(struct vector2D *v) {
    return GEOMETRY_MAX(v->x, v->y);
}
inline COORD vector_norm(struct vector2D *v) {
    return GEOMETRY_MAX(v->x, v->y);
}
#elif LX_NORM == 1
inline COORD vector_norm_dpu(struct vector2D *v) {
    if(v->y >= INT64_MAX - v->x) return INT64_MAX;
    return (v->x + v->y);
}
inline COORD vector_norm(struct vector2D *v) {
    if(v->y >= INT64_MAX - v->x) return INT64_MAX;
    return (v->x + v->y);
}
#elif LX_NORM == 2
inline COORD vector_norm_dpu(struct vector2D *v) {
    if(v->x >= L2_NORM_MAX || v->y >= L2_NORM_MAX) return INT64_MAX;
    COORD distance = approx_square(v->x), tmp = approx_square(v->y);
    if(tmp >= INT64_MAX - distance) return INT64_MAX;
    return distance + tmp;
}
inline COORD vector_norm(struct vector2D *v) {
    if(v->x >= L2_NORM_MAX || v->y >= L2_NORM_MAX) return INT64_MAX;
    COORD distance = v->x * v->x, tmp = v->y * v->y;
    if(tmp >= INT64_MAX - distance) return INT64_MAX;
    return distance + tmp;
}
#else
#endif

inline void vector_max(struct vector2D *src, struct vector2D *dst) {
    dst->x = GEOMETRY_MAX(dst->x, src->x);
    dst->y = GEOMETRY_MAX(dst->y, src->y);
}

inline void vector_min(struct vector2D *src, struct vector2D *dst) {
    dst->x = GEOMETRY_MIN(dst->x, src->x);
    dst->y = GEOMETRY_MIN(dst->y, src->y);
}

inline void vector_ones(struct vector2D *src, COORD c) {
    src->x = c;
    src->y = c;
}


inline bool vector_in_box(struct vector2D *v, struct vector2D *box_min, struct vector2D *box_max) {
    return (v->x >= box_min->x) && (v->y >= box_min->y) && (v->x <= box_max->x) && (v->y <= box_max->y);
}

inline bool box_intersect(struct vector2D *box1_min, struct vector2D *box1_max,
                          struct vector2D *box2_min, struct vector2D *box2_max) {
    return (box1_min->x <= box2_max->x) && (box2_min->x <= box1_max->x)
        && (box1_min->y <= box2_max->y) && (box2_min->y <= box1_max->y);
}

inline bool radius_intersect_box(struct vector2D *v, COORD radius, struct vector2D *box_min, struct vector2D *box_max) {
    struct vector2D v1 = vector_sub(v, box_min), v2 = vector_sub(v, box_max);
    if((v->x >= box_min->x) && (v->x <= box_max->x)) v1.x = 0;
    else v1.x = GEOMETRY_MIN(v1.x, v2.x);
    if((v->y >= box_min->y) && (v->y <= box_max->y)) v1.y = 0;
    else v1.y = GEOMETRY_MIN(v1.y, v2.y);
    return vector_norm(&v1) <= radius;
}

inline bool radius_intersect_box_dpu(struct vector2D *v, COORD radius, struct vector2D *box_min, struct vector2D *box_max) {
    struct vector2D v1 = vector_sub(v, box_min), v2 = vector_sub(v, box_max);
    if((v->x >= box_min->x) && (v->x <= box_max->x)) v1.x = 0;
    else v1.x = GEOMETRY_MIN(v1.x, v2.x);
    if((v->y >= box_min->y) && (v->y <= box_max->y)) v1.y = 0;
    else v1.y = GEOMETRY_MIN(v1.y, v2.y);
    return vector_norm_dpu(&v1) <= radius;
}

inline bool radius_contained_in_box(struct vector2D *v, COORD radius, struct vector2D *box_min, struct vector2D *box_max) {
    if(!vector_in_box(v, box_min, box_max)) return false;
    struct vector2D vec = vector_sub(v, box_min);
    COORD distance = GEOMETRY_MIN(vec.x, vec.y);
    vec = vector_sub(v, box_max);
    distance = GEOMETRY_MIN(distance, vec.x);
    distance = GEOMETRY_MIN(distance, vec.y);
#if LX_NORM == 2
    if(distance >= L2_NORM_MAX) distance = INT64_MAX;
    else distance *= distance;
#endif
    return (distance >= radius);
}

inline bool radius_contained_in_box_dpu(struct vector2D *v, COORD radius, struct vector2D *box_min, struct vector2D *box_max) {
    if(!vector_in_box(v, box_min, box_max)) return false;
    struct vector2D vec = vector_sub(v, box_min);
    COORD distance = GEOMETRY_MIN(vec.x, vec.y);
    vec = vector_sub(v, box_max);
    distance = GEOMETRY_MIN(distance, vec.x);
    distance = GEOMETRY_MIN(distance, vec.y);
#if LX_NORM == 2
    if(distance >= L2_NORM_MAX) distance = INT64_MAX;
    else distance = approx_square(distance);
#endif
    return (distance >= radius);
}

#define vectorT struct vector2D

#elif NR_DIMENSION == 3

struct vector3D {
    COORD x;
    COORD y;
    COORD z;
};

inline struct vector3D vector_add(struct vector3D *op1, struct vector3D *op2) {
    struct vector3D res;
    res.x = ((op1->x < COORD_MAX - op2->x) ? (op1->x + op2->x) : COORD_MAX);
    res.y = ((op1->y < COORD_MAX - op2->y) ? (op1->y + op2->y) : COORD_MAX);
    res.z = ((op1->z < COORD_MAX - op2->z) ? (op1->z + op2->z) : COORD_MAX);
    return res;
}

inline struct vector3D vector_sub(struct vector3D *op1, struct vector3D *op2) {
    struct vector3D res;
    res.x = llabs(op1->x - op2->x);
    res.y = llabs(op1->y - op2->y);
    res.z = llabs(op1->z - op2->z);
    return res;
}

inline struct vector3D vector_sub_zero_bounded(struct vector3D *op1, struct vector3D *op2) {
    struct vector3D res;
    res.x = ((op1->x > op2->x) ? (op1->x - op2->x) : 0);
    res.y = ((op1->y > op2->y) ? (op1->y - op2->y) : 0);
    res.z = ((op1->z > op2->z) ? (op1->z - op2->z) : 0);
    return res;
}

#if LX_NORM == 0
inline COORD vector_norm_dpu(struct vector3D *v) {
    COORD tmp = GEOMETRY_MAX(v->x, v->y);
    return GEOMETRY_MAX(tmp, v->z);
}
inline COORD vector_norm(struct vector3D *v) {
    COORD tmp = GEOMETRY_MAX(v->x, v->y);
    return GEOMETRY_MAX(tmp, v->z);
}
#elif LX_NORM == 1
inline COORD vector_norm_dpu(struct vector3D *v) {
    COORD distance = v->x;
    if(v->y >= INT64_MAX - distance) return INT64_MAX;
    distance += v->y;
    if(v->z >= INT64_MAX - distance) return INT64_MAX;
    return distance + v->z;
}
inline COORD vector_norm(struct vector3D *v) {
    COORD distance = v->x;
    if(v->y >= INT64_MAX - distance) return INT64_MAX;
    distance += v->y;
    if(v->z >= INT64_MAX - distance) return INT64_MAX;
    return distance + v->z;
}
#elif LX_NORM == 2
inline COORD vector_norm_dpu(struct vector3D *v) {
    if(v->x >= L2_NORM_MAX || v->y >= L2_NORM_MAX || v->z >= L2_NORM_MAX) return INT64_MAX;
    COORD distance = approx_square(v->x), tmp = approx_square(v->y);
    if(tmp >= INT64_MAX - distance) return INT64_MAX;
    distance += tmp;
    tmp = approx_square(v->z);
    if(tmp >= INT64_MAX - distance) return INT64_MAX;
    return distance + tmp;
}
inline COORD vector_norm(struct vector3D *v) {
    if(v->x >= L2_NORM_MAX || v->y >= L2_NORM_MAX || v->z >= L2_NORM_MAX) return INT64_MAX;
    COORD distance = v->x * v->x, tmp = v->y * v->y;
    if(tmp >= INT64_MAX - distance) return INT64_MAX;
    distance += tmp;
    tmp = v->z * v->z;
    if(tmp >= INT64_MAX - distance) return INT64_MAX;
    return distance + tmp;
}
#else
#endif

inline void vector_max(struct vector3D *src, struct vector3D *dst) {
    dst->x = GEOMETRY_MAX(dst->x, src->x);
    dst->y = GEOMETRY_MAX(dst->y, src->y);
    dst->z = GEOMETRY_MAX(dst->z, src->z);
}

inline void vector_min(struct vector3D *src, struct vector3D *dst) {
    dst->x = GEOMETRY_MIN(dst->x, src->x);
    dst->y = GEOMETRY_MIN(dst->y, src->y);
    dst->z = GEOMETRY_MIN(dst->z, src->z);
}

inline void vector_ones(struct vector3D *src, COORD c) {
    src->x = c;
    src->y = c;
    src->z = c;
}

inline bool vector_in_box(struct vector3D *v, struct vector3D *box_min, struct vector3D *box_max) {
    return (v->x >= box_min->x) && (v->y >= box_min->y) && (v->z >= box_min->z)
        && (v->x <= box_max->x) && (v->y <= box_max->y) && (v->z <= box_max->z);
}

inline bool box_intersect(struct vector3D *box1_min, struct vector3D *box1_max,
                          struct vector3D *box2_min, struct vector3D *box2_max) {
    return (box1_min->x <= box2_max->x) && (box2_min->x <= box1_max->x)
        && (box1_min->y <= box2_max->y) && (box2_min->y <= box1_max->y)
        && (box1_min->z <= box2_max->z) && (box2_min->z <= box1_max->z);
}

inline bool radius_intersect_box(struct vector3D *v, COORD radius, struct vector3D *box_min, struct vector3D *box_max) {
    struct vector3D v1 = vector_sub(v, box_min), v2 = vector_sub(v, box_max);
    if((v->x >= box_min->x) && (v->x <= box_max->x)) v1.x = 0;
    else v1.x = GEOMETRY_MIN(v1.x, v2.x);
    if((v->y >= box_min->y) && (v->y <= box_max->y)) v1.y = 0;
    else v1.y = GEOMETRY_MIN(v1.y, v2.y);
    if((v->z >= box_min->z) && (v->z <= box_max->z)) v1.z = 0;
    else v1.z = GEOMETRY_MIN(v1.z, v2.z);
    return vector_norm(&v1) <= radius;
}

inline bool radius_intersect_box_dpu(struct vector3D *v, COORD radius, struct vector3D *box_min, struct vector3D *box_max) {
    struct vector3D v1 = vector_sub(v, box_min), v2 = vector_sub(v, box_max);
    if((v->x >= box_min->x) && (v->x <= box_max->x)) v1.x = 0;
    else v1.x = GEOMETRY_MIN(v1.x, v2.x);
    if((v->y >= box_min->y) && (v->y <= box_max->y)) v1.y = 0;
    else v1.y = GEOMETRY_MIN(v1.y, v2.y);
    if((v->z >= box_min->z) && (v->z <= box_max->z)) v1.z = 0;
    else v1.z = GEOMETRY_MIN(v1.z, v2.z);
    return vector_norm_dpu(&v1) <= radius;
}

inline bool radius_contained_in_box(struct vector3D *v, COORD radius, struct vector3D *box_min, struct vector3D *box_max) {
    if(!vector_in_box(v, box_min, box_max)) return false;
    struct vector3D vec = vector_sub(v, box_min);
    COORD distance = GEOMETRY_MIN(vec.x, vec.y);
    distance = GEOMETRY_MIN(distance, vec.z);
    vec = vector_sub(v, box_max);
    distance = GEOMETRY_MIN(distance, vec.x);
    distance = GEOMETRY_MIN(distance, vec.y);
    distance = GEOMETRY_MIN(distance, vec.z);
#if LX_NORM == 2
    if(distance >= L2_NORM_MAX) distance = INT64_MAX;
    else distance *= distance;
#endif
    return (distance >= radius);
}

inline bool radius_contained_in_box_dpu(struct vector3D *v, COORD radius, struct vector3D *box_min, struct vector3D *box_max) {
    if(!vector_in_box(v, box_min, box_max)) return false;
    struct vector3D vec = vector_sub(v, box_min);
    COORD distance = GEOMETRY_MIN(vec.x, vec.y);
    distance = GEOMETRY_MIN(distance, vec.z);
    vec = vector_sub(v, box_max);
    distance = GEOMETRY_MIN(distance, vec.x);
    distance = GEOMETRY_MIN(distance, vec.y);
    distance = GEOMETRY_MIN(distance, vec.z);
#if LX_NORM == 2
    if(distance >= L2_NORM_MAX) distance = INT64_MAX;
    else distance = approx_square(distance);
#endif
    return (distance >= radius);
}

#define vectorT struct vector3D

#else

struct vector_ {
    COORD x[NR_DIMENSION];
};

inline struct vector_ vector_add(struct vector_ *op1, struct vector_ *op2) {
    struct vector_ res;
    for(int i = 0; i < NR_DIMENSION; i++)
        res.x[i] = ((op1->x[i] < COORD_MAX - op2->x[i]) ? (op1->x[i] + op2->x[i]) : COORD_MAX);
    return res;
}

inline struct vector_ vector_sub(struct vector_ *op1, struct vector_ *op2) {
    struct vector_ res;
    for(int i = 0; i < NR_DIMENSION; i++)
        res.x[i] = llabs(op1->x[i] - op2->x[i]);
    return res;
}

inline struct vector_ vector_sub_zero_bounded(struct vector_ *op1, struct vector_ *op2) {
    struct vector_ res;
    for(int i = 0; i < NR_DIMENSION; i++)
        res.x[i] = ((op1->x[i] > op2->x[i]) ? (op1->x[i] - op2->x[i]) : 0);
    return res;
}

#if LX_NORM == 0
inline COORD vector_norm_dpu(struct vector_ *v) {
    COORD res = v->x[0];
    for(int i = 1; i < NR_DIMENSION; i++)
        res = GEOMETRY_MAX(res, v->x[i]);
    return res;
}
inline COORD vector_norm(struct vector_ *v) {
    COORD res = v->x[0];
    for(int i = 1; i < NR_DIMENSION; i++)
        res = GEOMETRY_MAX(res, v->x[i]);
    return res;
}
#elif LX_NORM == 1
inline COORD vector_norm_dpu(struct vector_ *v) {
    COORD res = v->x[0], tmp;
    for(int i = 1; i < NR_DIMENSION; i++) {
        tmp = v->x[i];
        if(tmp >= INT64_MAX - res) return INT64_MAX;
        res += tmp;
    }
    return res;
}
inline COORD vector_norm(struct vector_ *v) {
    COORD res = v->x[0], tmp;
    for(int i = 1; i < NR_DIMENSION; i++) {
        tmp = v->x[i];
        if(tmp >= INT64_MAX - res) return INT64_MAX;
        res += tmp;
    }
    return res;
}
#elif LX_NORM == 2
inline COORD vector_norm_dpu(struct vector_ *v) {
    COORD res = 0, tmp;
    for(int i = 0; i < NR_DIMENSION; i++) {
        if(v->x[i] >= L2_NORM_MAX) return INT64_MAX;
        tmp = approx_square(v->x[i]);
        if(tmp >= INT64_MAX - res) return INT64_MAX;
        res += tmp;
    }
    return res;
}
inline COORD vector_norm(struct vector_ *v) {
    COORD res = 0, tmp;
    for(int i = 0; i < NR_DIMENSION; i++) {
        if(v->x[i] >= L2_NORM_MAX) return INT64_MAX;
        tmp = v->x[i] * v->x[i];
        if(tmp >= INT64_MAX - res) return INT64_MAX;
        res += tmp;
    }
    return res;
}
#else
#endif

inline void vector_max(struct vector_ *src, struct vector_ *dst) {
    for(int i = 0; i < NR_DIMENSION; i++)
        dst->x[i] = GEOMETRY_MAX(dst->x[i], src->x[i]);
}

inline void vector_min(struct vector_ *src, struct vector_ *dst) {
    for(int i = 0; i < NR_DIMENSION; i++)
        dst->x[i] = GEOMETRY_MIN(dst->x[i], src->x[i]);
}

inline void vector_ones(struct vector_ *src, COORD c) {
    for(int i = 0; i < NR_DIMENSION; i++)
        src->x[i] = c;
}

inline bool vector_in_box(struct vector_ *v, struct vector_ *box_min, struct vector_ *box_max) {
    for(int i = 0; i < NR_DIMENSION; i++)
        if(v->x[i] < box_min->x[i] || v->x[i] > box_max->x[i])
            return false;
    return true;
}

inline bool box_intersect(struct vector_ *box1_min, struct vector_ *box1_max,
                          struct vector_ *box2_min, struct vector_ *box2_max) {
    for(int i = 0; i < NR_DIMENSION; i++)
        if((box1_min->x[i] > box2_max->x[i]) || (box2_min->x[i] > box1_max->x[i]))
            return false;
    return true;
}

inline bool radius_intersect_box(struct vector_ *v, COORD radius, struct vector_ *box_min, struct vector_ *box_max) {
    struct vector_ v1 = vector_sub(v, box_min), v2 = vector_sub(v, box_max);
    for(int i = 0; i < NR_DIMENSION; i++) {
        if((v->x[i] >= box_min->x[i]) && (v->x[i] <= box_max->x[i])) v1.x[i] = 0;
        else v1.x[i] = GEOMETRY_MIN(v1.x[i], v2.x[i]);
    }
    return vector_norm(&v1) <= radius;
}

inline bool radius_intersect_box_dpu(struct vector_ *v, COORD radius, struct vector_ *box_min, struct vector_ *box_max) {
    struct vector_ v1 = vector_sub(v, box_min), v2 = vector_sub(v, box_max);
    for(int i = 0; i < NR_DIMENSION; i++) {
        if((v->x[i] >= box_min->x[i]) && (v->x[i] <= box_max->x[i])) v1.x[i] = 0;
        else v1.x[i] = GEOMETRY_MIN(v1.x[i], v2.x[i]);
    }
    return vector_norm_dpu(&v1) <= radius;
}

inline bool radius_contained_in_box(struct vector_ *v, COORD radius, struct vector_ *box_min, struct vector_ *box_max) {
    if(!vector_in_box(v, box_min, box_max)) return false;
    struct vector_ vec = vector_sub(v, box_min);
    COORD distance = INT64_MAX;
    for(int i = 0; i < NR_DIMENSION; i++) distance = GEOMETRY_MIN(distance, vec.x[i]);
    vec = vector_sub(v, box_max);
    for(int i = 0; i < NR_DIMENSION; i++) distance = GEOMETRY_MIN(distance, vec.x[i]);
#if LX_NORM == 2
    if(distance >= L2_NORM_MAX) distance = INT64_MAX;
    else distance *= distance;
#endif
    return (distance >= radius);
}

inline bool radius_contained_in_box_dpu(struct vector_ *v, COORD radius, struct vector_ *box_min, struct vector_ *box_max) {
    if(!vector_in_box(v, box_min, box_max)) return false;
    struct vector_ vec = vector_sub(v, box_min);
    COORD distance = INT64_MAX;
    for(int i = 0; i < NR_DIMENSION; i++) distance = GEOMETRY_MIN(distance, vec.x[i]);
    vec = vector_sub(v, box_max);
    for(int i = 0; i < NR_DIMENSION; i++) distance = GEOMETRY_MIN(distance, vec.x[i]);
#if LX_NORM == 2
    if(distance >= L2_NORM_MAX) distance = INT64_MAX;
    else distance = approx_square(distance);
#endif
    return (distance >= radius);
}

#define vectorT struct vector_

#endif

inline bool box_contain(vectorT *small_box_min, vectorT *small_box_max,
                        vectorT *large_box_min, vectorT *large_box_max) {
    return vector_in_box(small_box_min, large_box_min, large_box_max)
        && vector_in_box(small_box_max, large_box_min, large_box_max);
}
