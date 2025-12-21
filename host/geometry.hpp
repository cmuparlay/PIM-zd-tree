#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "common.h"

#define COORD int64_t

#include "geometry_base.h"

// template <class COORD>
// class vector_host {
// public:
//     COORD x[NR_DIMENSION];
//     vector_host(COORD *p) {
//         for(int i = 0; i < NR_DIMENSION; i++)
//             x[i] = p[i];
//     }
//     vector_host operator+(vector_host &v) {
//         vector_host res;
//         for(int i = 0; i < NR_DIMENSION; i++)
//             res.x[i] = x[i] + v.x[i];
//         return res;
//     }
//     vector_host operator-(vector_host &v) {
//         vector_host res;
//         for(int i = 0; i < NR_DIMENSION; i++)
//             res.x[i] = x[i] - v.x[i];
//         return res;
//     }
//     vector_host operator*(COORD c) {
//         vector_host res;
//         for(int i = 0; i < NR_DIMENSION; i++)
//             res.x[i] = x[i] * c;
//         return res;
//     }
//     vector_host operator/(COORD c) {
//         vector_host res;
//         for(int i = 0; i < NR_DIMENSION; i++)
//             res.x[i] = x[i] / c;
//         return res;
//     }
//     COORD& operator[](int i) {return x[i];}
//     COORD dot(vector_host &v) {
//         COORD res = 0;
//         for(int i = 0; i < NR_DIMENSION; i++)
//             res += x[i] * v.x[i];
//         return res;
//     }
//     COORD l0() {
//         COORD res = abs(x[0]);
//         for(int i = 1; i < NR_DIMENSION; i++)
//             res = __max(res, abs(x[i]));
//         return res;
//     }
//     COORD l1() {
//         COORD res = abs(x[0]);
//         for(int i = 1; i < NR_DIMENSION; i++)
//             res += abs(x[i]);
//         return res;
//     }
//     COORD l2() {
//         COORD res = x[0] * x[0];
//         for(int i = 1; i < NR_DIMENSION; i++)
//             res += x[i] * x[i];
//         return res;
//     }
//     bool vector_in_box(vector_host &box_min, vector_host &box_max) {
//         for(int i = 1; i < NR_DIMENSION; i++)
//             if(x[i] < box_min.x[i] || x[i] > box_max.x[i])
//                 return false;
//         return true;
//     }
// };
