#pragma once

#شامل "ggml.h"
#شامل "ggml-cpu-impl.h"

#شامل <algorithm>
#شامل <memory>
#شامل <type_traits>

#اگر defined(GGML_USE_OPENMP)
#شامل <omp.h>
#وگرنه
#شامل <thread>
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
template <typename T, typename std::enable_if<std::is_integral<T>::مقدار, صحیح>::type = 0>
درونخطی T div_up(T x, T y) { بازگشت (x + y - 1) / y; }

template <typename T>
درونخطی تهی balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#اگر 0
    // onednn partition pattern
    T& n_my = n_end;
    اگر (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } وگرنه {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#وگرنه
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
درونخطی تهی parallel_for(صحیح n, ثابت func_t & f) {
    اگر (n <= 0) {
        بازگشت;
    }
#اگر defined(GGML_USE_OPENMP)
    #pragma omp parallel
    {
        صحیح nth = omp_get_رقم_threads();
        صحیح ith = omp_get_thread_رقم();
        صحیح tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#وگرنه
    صحیح nth = std::thread::hardware_concurrency();
    اگر (nth <= 1) {
        f(0, n);
        بازگشت;
    }
    اگر (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    برای (صحیح ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            صحیح tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    برای (auto & t : threads) {
        t.join();
    }
#endif
}

template <typename func_t>
درونخطی تهی parallel_for_ggml(ثابت ggml_compute_params * params, صحیح n, ثابت func_t & f) {
    صحیح tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
درونخطی bool qtype_has_amx_kernels(ثابت شمارش ggml_type type) {
    // TODO: fix padding for vnni format
    بازگشت (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
