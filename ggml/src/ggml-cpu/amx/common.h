#pragma once

#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "ggml-cpu-impl.h"

#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <memory>
#অন্তর্ভাৱ <type_traits>

#যদি defined(GGML_USE_OPENMP)
#অন্তর্ভাৱ <omp.h>
#অন্যথা
#অন্তর্ভাৱ <thread>
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
template <typename T, typename std::enable_if<std::is_integral<T>::মান, পূর্ণাঙ্ক>::type = 0>
অন্তঃস্থ T div_up(T x, T y) { ঘূৰাই (x + y - 1) / y; }

template <typename T>
অন্তঃস্থ শূন্য balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#যদি 0
    // onednn partition pattern
    T& n_my = n_end;
    যদি (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } অন্যথা {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#অন্যথা
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
অন্তঃস্থ শূন্য parallel_for(পূর্ণাঙ্ক n, ধ্রুৱ func_t & f) {
    যদি (n <= 0) {
        ঘূৰাই;
    }
#যদি defined(GGML_USE_OPENMP)
    #pragma omp parallel
    {
        পূর্ণাঙ্ক nth = omp_get_অঙ্ক_threads();
        পূর্ণাঙ্ক ith = omp_get_thread_অঙ্ক();
        পূর্ণাঙ্ক tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#অন্যথা
    পূর্ণাঙ্ক nth = std::thread::hardware_concurrency();
    যদি (nth <= 1) {
        f(0, n);
        ঘূৰাই;
    }
    যদি (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    বাবে (পূর্ণাঙ্ক ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            পূর্ণাঙ্ক tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    বাবে (auto & t : threads) {
        t.join();
    }
#endif
}

template <typename func_t>
অন্তঃস্থ শূন্য parallel_for_ggml(ধ্রুৱ ggml_compute_params * params, পূর্ণাঙ্ক n, ধ্রুৱ func_t & f) {
    পূর্ণাঙ্ক tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
অন্তঃস্থ bool qtype_has_amx_kernels(ধ্রুৱ গণনা ggml_type type) {
    // TODO: fix padding for vnni format
    ঘূৰাই (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
