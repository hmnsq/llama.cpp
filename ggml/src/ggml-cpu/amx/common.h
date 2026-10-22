#pragma once

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-cpu-impl.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <memory>
#अन्तर्भाव <type_traits>

#यदि defined(GGML_USE_OPENMP)
#अन्तर्भाव <omp.h>
#अन्यथा
#अन्तर्भाव <thread>
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
template <typename T, typename std::enable_if<std::is_integral<T>::मूल्य, पूर्णाङ्क>::type = 0>
अन्तःस्थ T div_up(T x, T y) { प्रत्यागम (x + y - 1) / y; }

template <typename T>
अन्तःस्थ शून्य balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#यदि 0
    // onednn partition pattern
    T& n_my = n_end;
    यदि (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } अन्यथा {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#अन्यथा
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
अन्तःस्थ शून्य parallel_for(पूर्णाङ्क n, ध्रुव func_t & f) {
    यदि (n <= 0) {
        प्रत्यागम;
    }
#यदि defined(GGML_USE_OPENMP)
    #pragma omp parallel
    {
        पूर्णाङ्क nth = omp_get_अंक_threads();
        पूर्णाङ्क ith = omp_get_thread_अंक();
        पूर्णाङ्क tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#अन्यथा
    पूर्णाङ्क nth = std::thread::hardware_concurrency();
    यदि (nth <= 1) {
        f(0, n);
        प्रत्यागम;
    }
    यदि (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    प्रति (पूर्णाङ्क ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            पूर्णाङ्क tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    प्रति (auto & t : threads) {
        t.join();
    }
#endif
}

template <typename func_t>
अन्तःस्थ शून्य parallel_for_ggml(ध्रुव ggml_compute_params * params, पूर्णाङ्क n, ध्रुव func_t & f) {
    पूर्णाङ्क tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
अन्तःस्थ bool qtype_has_amx_kernels(ध्रुव गणना ggml_type type) {
    // TODO: fix padding for vnni format
    प्रत्यागम (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
