#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <type_traits>

#ಆದರೆ defined(GGML_USE_OPENMP)
#ಅನ್ತರ್ಭಾವ <omp.h>
#ಇಲ್ಲದಿದ್ದರೆ
#ಅನ್ತರ್ಭಾವ <thread>
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು TILE_M 16
#ವ್ಯಾಖ್ಯಾನಿಸು TILE_N 16
#ವ್ಯಾಖ್ಯಾನಿಸು TILE_K 32
#ವ್ಯಾಖ್ಯಾನಿಸು VNNI_BLK 4

#ವ್ಯಾಖ್ಯಾನಿಸು AMX_BLK_ಗಾತ್ರ 32

#ವ್ಯಾಖ್ಯಾನಿಸು TMM0 0
#ವ್ಯಾಖ್ಯಾನಿಸು TMM1 1
#ವ್ಯಾಖ್ಯಾನಿಸು TMM2 2
#ವ್ಯಾಖ್ಯಾನಿಸು TMM3 3
#ವ್ಯಾಖ್ಯಾನಿಸು TMM4 4
#ವ್ಯಾಖ್ಯಾನಿಸು TMM5 5
#ವ್ಯಾಖ್ಯಾನಿಸು TMM6 6
#ವ್ಯಾಖ್ಯಾನಿಸು TMM7 7

// parallel routines
ಮಾದರಿ <typename T, typename std::enable_if<std::is_integral<T>::ಮೌಲ್ಯ, ಪೂರ್ಣಾಂಕ>::type = 0>
ಸಾಲೊಳಗೆ T div_ಮೇಲೆ(T x, T y) { ಹಿಂತಿರುಗಿ (x + y - 1) / y; }

ಮಾದರಿ <typename T>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ balance211(T n, T nth, T ith, T& n_ಆರಂಭ, T& n_ಕೊನೆ) {
#ಆದರೆ 0
    // onednn partition pattern
    T& n_my = n_ಕೊನೆ;
    ಆದರೆ (nth <= 1 || n == 0) {
        n_ಆರಂಭ = 0;
        n_my = n;
    } ಇಲ್ಲದಿದ್ದರೆ {
        T n1 = div_ಮೇಲೆ(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_ಆರಂಭ = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_ಕೊನೆ += n_ಆರಂಭ;
#ಇಲ್ಲದಿದ್ದರೆ
    // pytorch aten partition pattern
    T n_my = div_ಮೇಲೆ(n, nth);
    n_ಆರಂಭ = ith * n_my;
    n_ಕೊನೆ = std::min(n_ಆರಂಭ + n_my, n);
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <typename func_t>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ parallel_for(ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ func_t & f) {
    ಆದರೆ (n <= 0) {
        ಹಿಂತಿರುಗಿ;
    }
#ಆದರೆ defined(GGML_USE_OPENMP)
    #ಸೂಚನೆ omp parallel
    {
        ಪೂರ್ಣಾಂಕ nth = omp_get_ಅಂಕೆ_ಎಳೆಗಳು();
        ಪೂರ್ಣಾಂಕ ith = omp_get_thread_ಅಂಕೆ();
        ಪೂರ್ಣಾಂಕ tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಪೂರ್ಣಾಂಕ nth = std::thread::hardware_concurrency();
    ಆದರೆ (nth <= 1) {
        f(0, n);
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (nth > n) {
        nth = n;
    }
    std::ಸದಿಶ<std::thread> ಎಳೆಗಳು;
    ಎಳೆಗಳು.reserve(nth);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ith = 0; ith < nth; ++ith) {
        ಎಳೆಗಳು.emplace_ಹಿಂಭಾಗ([&f, n, ith, nth] {
            ಪೂರ್ಣಾಂಕ tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & t : ಎಳೆಗಳು) {
        t.join();
    }
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <typename func_t>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ parallel_for_ggml(ಸ್ಥಿರಾಂಕ ggml_compute_params * params, ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ func_t & f) {
    ಪೂರ್ಣಾಂಕ tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
ಸಾಲೊಳಗೆ bool qtype_has_amx_kernels(ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type) {
    // TODO: fix padding for vnni format
    ಹಿಂತಿರುಗಿ (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
