#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu-impl.h"

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <type_traits>

#ଯଦି defined(GGML_USE_OPENMP)
#ଅନ୍ତର୍ଭାଵ <omp.h>
#ଅନ୍ୟଥା
#ଅନ୍ତର୍ଭାଵ <thread>
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
template <typename T, typename std::enable_if<std::is_integral<T>::ମୂଲ୍ୟ, ପୂର୍ଣ୍ଣସଂଖ୍ୟା>::type = 0>
ଧାଡ଼ିଭିତରେ T div_up(T x, T y) { ଫେରନ୍ତୁ (x + y - 1) / y; }

template <typename T>
ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#ଯଦି 0
    // onednn partition pattern
    T& n_my = n_end;
    ଯଦି (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } ଅନ୍ୟଥା {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#ଅନ୍ୟଥା
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ parallel_for(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ func_t & f) {
    ଯଦି (n <= 0) {
        ଫେରନ୍ତୁ;
    }
#ଯଦି defined(GGML_USE_OPENMP)
    #pragma omp parallel
    {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = omp_get_ଅଙ୍କ_threads();
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = omp_get_thread_ଅଙ୍କ();
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#ଅନ୍ୟଥା
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = std::thread::hardware_concurrency();
    ଯଦି (nth <= 1) {
        f(0, n);
        ଫେରନ୍ତୁ;
    }
    ଯଦି (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    ପାଇଁ (auto & t : threads) {
        t.join();
    }
#endif
}

template <typename func_t>
ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ parallel_for_ggml(ଅଚଳ ggml_compute_params * params, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ func_t & f) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
ଧାଡ଼ିଭିତରେ bool qtype_has_amx_kernels(ଅଚଳ ଗଣନା ggml_type type) {
    // TODO: fix padding for vnni format
    ଫେରନ୍ତୁ (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
