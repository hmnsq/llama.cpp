#pragma once

#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "ggml-cpu-impl.h"

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <type_traits>

#જો defined(GGML_USE_OPENMP)
#અન્તર્ભાવ <omp.h>
#નહીંતર
#અન્તર્ભાવ <thread>
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
template <typename T, typename std::enable_if<std::is_integral<T>::કિંમત, પૂર્ણાંક>::type = 0>
અંતઃરેખ T div_up(T x, T y) { પાછા (x + y - 1) / y; }

template <typename T>
અંતઃરેખ શૂન્ય balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#જો 0
    // onednn partition pattern
    T& n_my = n_end;
    જો (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } નહીંતર {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#નહીંતર
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
અંતઃરેખ શૂન્ય parallel_for(પૂર્ણાંક n, અચલ func_t & f) {
    જો (n <= 0) {
        પાછા;
    }
#જો defined(GGML_USE_OPENMP)
    #pragma omp parallel
    {
        પૂર્ણાંક nth = omp_get_અંક_threads();
        પૂર્ણાંક ith = omp_get_thread_અંક();
        પૂર્ણાંક tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#નહીંતર
    પૂર્ણાંક nth = std::thread::hardware_concurrency();
    જો (nth <= 1) {
        f(0, n);
        પાછા;
    }
    જો (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    માટે (પૂર્ણાંક ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            પૂર્ણાંક tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    માટે (auto & t : threads) {
        t.join();
    }
#endif
}

template <typename func_t>
અંતઃરેખ શૂન્ય parallel_for_ggml(અચલ ggml_compute_params * params, પૂર્ણાંક n, અચલ func_t & f) {
    પૂર્ણાંક tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
અંતઃરેખ bool qtype_has_amx_kernels(અચલ ગણના ggml_type type) {
    // TODO: fix padding for vnni format
    પાછા (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
