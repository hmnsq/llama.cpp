#pragma once

#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "ggml-cpu-impl.h"

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <type_traits>

#എങ്കിൽ defined(GGML_USE_OPENMP)
#അന്തര്ഭാവ <omp.h>
#അല്ലെങ്കിൽ
#അന്തര്ഭാവ <thread>
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
template <typename T, typename std::enable_if<std::is_integral<T>::മൂല്യം, പൂർണ്ണസംഖ്യ>::type = 0>
വരിയിൽ T div_up(T x, T y) { തിരികെ (x + y - 1) / y; }

template <typename T>
വരിയിൽ ശൂന്യം balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#എങ്കിൽ 0
    // onednn partition pattern
    T& n_my = n_end;
    എങ്കിൽ (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } അല്ലെങ്കിൽ {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#അല്ലെങ്കിൽ
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
വരിയിൽ ശൂന്യം parallel_for(പൂർണ്ണസംഖ്യ n, സ്ഥിരം func_t & f) {
    എങ്കിൽ (n <= 0) {
        തിരികെ;
    }
#എങ്കിൽ defined(GGML_USE_OPENMP)
    #pragma omp parallel
    {
        പൂർണ്ണസംഖ്യ nth = omp_get_അക്കം_threads();
        പൂർണ്ണസംഖ്യ ith = omp_get_thread_അക്കം();
        പൂർണ്ണസംഖ്യ tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#അല്ലെങ്കിൽ
    പൂർണ്ണസംഖ്യ nth = std::thread::hardware_concurrency();
    എങ്കിൽ (nth <= 1) {
        f(0, n);
        തിരികെ;
    }
    എങ്കിൽ (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    വേണ്ടി (പൂർണ്ണസംഖ്യ ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            പൂർണ്ണസംഖ്യ tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    വേണ്ടി (auto & t : threads) {
        t.join();
    }
#endif
}

template <typename func_t>
വരിയിൽ ശൂന്യം parallel_for_ggml(സ്ഥിരം ggml_compute_params * params, പൂർണ്ണസംഖ്യ n, സ്ഥിരം func_t & f) {
    പൂർണ്ണസംഖ്യ tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
വരിയിൽ bool qtype_has_amx_kernels(സ്ഥിരം എണ്ണൽ ggml_type type) {
    // TODO: fix padding for vnni format
    തിരികെ (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
