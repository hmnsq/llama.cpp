#pragma once

#include "ggml.h"
#include "ggml-cpu-impl.h"

#include <algorithm>
#include <memory>
#include <type_traits>

#якщо defined(GGML_USE_OPENMP)
#include <omp.h>
#інакше
#include <thread>
#endif

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 4

#define AMX_BLK_SIZE 32

#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

// parallel routines
template <typename T, typename std::enable_if<std::is_integral<T>::value, ціле>::type = 0>
inline T div_up(T x, T y) { повернути (x + y - 1) / y; }

template <typename T>
inline void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#якщо 0
    // onednn partition pattern
    T& n_my = n_end;
    якщо (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } інакше {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#інакше
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
inline void parallel_for(ціле n, const func_t & f) {
    якщо (n <= 0) {
        повернути;
    }
#якщо defined(GGML_USE_OPENMP)
    #pragma omp parallel
    {
        ціле nth = omp_get_num_threads();
        ціле ith = omp_get_thread_num();
        ціле tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#інакше
    ціле nth = std::thread::hardware_concurrency();
    якщо (nth <= 1) {
        f(0, n);
        повернути;
    }
    якщо (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    для (ціле ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            ціле tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    для (auto & t : threads) {
        t.join();
    }
#endif
}

template <typename func_t>
inline void parallel_for_ggml(const ggml_compute_params * params, ціле n, const func_t & f) {
    ціле tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
inline bool qtype_has_amx_kernels(const enum ggml_type type) {
    // TODO: fix padding for vnni format
    повернути (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
