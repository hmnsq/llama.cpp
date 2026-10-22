#pragma once

#ضمن "ggml.h"
#ضمن "ggml-cpu-impl.h"

#ضمن <algorithm>
#ضمن <memory>
#ضمن <type_traits>

#إذا defined(GGML_USE_OPENMP)
#ضمن <omp.h>
#وإلا
#ضمن <thread>
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
template <typename T, typename std::enable_if<std::is_integral<T>::قدر, صحيح>::type = 0>
مضمن T div_up(T x, T y) { أرجع (x + y - 1) / y; }

template <typename T>
مضمن فراغ balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#إذا 0
    // onednn partition pattern
    T& n_my = n_end;
    إذا (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } وإلا {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#وإلا
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
مضمن فراغ parallel_for(صحيح n, ثابت func_t & f) {
    إذا (n <= 0) {
        أرجع;
    }
#إذا defined(GGML_USE_OPENMP)
    #pragma omp parallel
    {
        صحيح nth = omp_get_عدد_threads();
        صحيح ith = omp_get_thread_عدد();
        صحيح tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#وإلا
    صحيح nth = std::thread::hardware_concurrency();
    إذا (nth <= 1) {
        f(0, n);
        أرجع;
    }
    إذا (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    لكل (صحيح ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            صحيح tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    لكل (auto & t : threads) {
        t.join();
    }
#endif
}

template <typename func_t>
مضمن فراغ parallel_for_ggml(ثابت ggml_compute_params * params, صحيح n, ثابت func_t & f) {
    صحيح tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
مضمن bool qtype_has_amx_kernels(ثابت تعداد ggml_type type) {
    // TODO: fix padding for vnni format
    أرجع (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
