#pragma once

#include "ggml.h"
#include "ggml-cpu-impl.h"

#include <algorithm>
#include <memory>
#include <type_traits>

#အကယ်၍ defined(GGML_USE_OPENMP)
#include <omp.h>
#မဟုတ်ပါက
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
template <typename T, typename std::enable_if<std::is_integral<T>::value, ကိန်းပြည့်>::type = 0>
anta_hstha T div_up(T x, T y) { ပြန်ပေး (x + y - 1) / y; }

template <typename T>
anta_hstha void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#အကယ်၍ 0
    // onednn partition pattern
    T& n_my = n_end;
    အကယ်၍ (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } မဟုတ်ပါက {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#မဟုတ်ပါက
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
anta_hstha void parallel_for(ကိန်းပြည့် n, const func_t & f) {
    အကယ်၍ (n <= 0) {
        ပြန်ပေး;
    }
#အကယ်၍ defined(GGML_USE_OPENMP)
    #pragma omp parallel
    {
        ကိန်းပြည့် nth = omp_get_num_threads();
        ကိန်းပြည့် ith = omp_get_thread_num();
        ကိန်းပြည့် tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#မဟုတ်ပါက
    ကိန်းပြည့် nth = std::thread::hardware_concurrency();
    အကယ်၍ (nth <= 1) {
        f(0, n);
        ပြန်ပေး;
    }
    အကယ်၍ (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    အတွက် (ကိန်းပြည့် ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            ကိန်းပြည့် tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    အတွက် (auto & t : threads) {
        t.join();
    }
#endif
}

template <typename func_t>
anta_hstha void parallel_for_ggml(const ggml_compute_params * params, ကိန်းပြည့် n, const func_t & f) {
    ကိန်းပြည့် tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
anta_hstha bool qtype_has_amx_kernels(const ga_nanaa ggml_type type) {
    // TODO: fix padding for vnni format
    ပြန်ပေး (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
