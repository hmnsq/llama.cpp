#pragma once

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <type_traits>

#ಯದಿ defined(GGML_USE_OPENMP)
#ಅನ್ತರ್ಭಾವ <omp.h>
#ಅನ್ಯಥಾ
#ಅನ್ತರ್ಭಾವ <thread>
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
template <typename T, typename std::enable_if<std::is_integral<T>::ಮೌಲ್ಯ, ಪೂರ್ಣಾಙ್ಕ>::type = 0>
ಅನ್ತಃಸ್ಥ T div_up(T x, T y) { ಪ್ರತ್ಯಾಗಮ (x + y - 1) / y; }

template <typename T>
ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#ಯದಿ 0
    // onednn partition pattern
    T& n_my = n_end;
    ಯದಿ (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } ಅನ್ಯಥಾ {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#ಅನ್ಯಥಾ
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ parallel_for(ಪೂರ್ಣಾಙ್ಕ n, ಧ್ರುವ func_t & f) {
    ಯದಿ (n <= 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }
#ಯದಿ defined(GGML_USE_OPENMP)
    #pragma omp parallel
    {
        ಪೂರ್ಣಾಙ್ಕ nth = omp_get_ಅಂಕೆ_threads();
        ಪೂರ್ಣಾಙ್ಕ ith = omp_get_thread_ಅಂಕೆ();
        ಪೂರ್ಣಾಙ್ಕ tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#ಅನ್ಯಥಾ
    ಪೂರ್ಣಾಙ್ಕ nth = std::thread::hardware_concurrency();
    ಯದಿ (nth <= 1) {
        f(0, n);
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಯದಿ (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            ಪೂರ್ಣಾಙ್ಕ tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    ಪ್ರತಿ (auto & t : threads) {
        t.join();
    }
#endif
}

template <typename func_t>
ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ parallel_for_ggml(ಧ್ರುವ ggml_compute_params * params, ಪೂರ್ಣಾಙ್ಕ n, ಧ್ರುವ func_t & f) {
    ಪೂರ್ಣಾಙ್ಕ tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
ಅನ್ತಃಸ್ಥ bool qtype_has_amx_kernels(ಧ್ರುವ ಗಣನಾ ggml_type type) {
    // TODO: fix padding for vnni format
    ಪ್ರತ್ಯಾಗಮ (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
