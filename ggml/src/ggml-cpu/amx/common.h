#आशय once

#समावेश "ggml.h"
#समावेश "ggml-cpu-impl.h"

#समावेश <algorithm>
#समावेश <memory>
#समावेश <type_traits>

#अगर defined(GGML_USE_OPENMP)
#समावेश <omp.h>
#अन्यथा
#समावेश <thread>
#पूर्ण_अगर

#घोषणा TILE_M 16
#घोषणा TILE_N 16
#घोषणा TILE_K 32
#घोषणा VNNI_BLK 4

#घोषणा AMX_BLK_आकार 32

#घोषणा TMM0 0
#घोषणा TMM1 1
#घोषणा TMM2 2
#घोषणा TMM3 3
#घोषणा TMM4 4
#घोषणा TMM5 5
#घोषणा TMM6 6
#घोषणा TMM7 7

// parallel routines
ढाँचा <typename T, typename std::enable_if<std::is_integral<T>::मूल्य, पूर्णांक>::type = 0>
अंतरभूत T div_up(T x, T y) { वापस (x + y - 1) / y; }

ढाँचा <typename T>
अंतरभूत व्योम balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#अगर 0
    // onednn partition pattern
    T& n_my = n_end;
    अगर (nth <= 1 || n == 0) {
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
#पूर्ण_अगर
}

ढाँचा <typename func_t>
अंतरभूत व्योम parallel_for(पूर्णांक n, स्थिर func_t & f) {
    अगर (n <= 0) {
        वापस;
    }
#अगर defined(GGML_USE_OPENMP)
    #आशय omp parallel
    {
        पूर्णांक nth = omp_get_अंक_threads();
        पूर्णांक ith = omp_get_thread_अंक();
        पूर्णांक tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#अन्यथा
    पूर्णांक nth = std::thread::hardware_concurrency();
    अगर (nth <= 1) {
        f(0, n);
        वापस;
    }
    अगर (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    क्रम (पूर्णांक ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            पूर्णांक tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    क्रम (स्वतः & t : threads) {
        t.join();
    }
#पूर्ण_अगर
}

ढाँचा <typename func_t>
अंतरभूत व्योम parallel_for_ggml(स्थिर ggml_compute_params * params, पूर्णांक n, स्थिर func_t & f) {
    पूर्णांक tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
अंतरभूत bool qtype_has_amx_kernels(स्थिर क्रमागत ggml_type type) {
    // TODO: fix padding for vnni format
    वापस (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
