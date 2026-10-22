#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <type_traits>

#ਜੇ defined(GGML_USE_OPENMP)
#ਅਨ੍ਤਰ੍ਭਾਵ <omp.h>
#ਨਹੀਂਤਾਂ
#ਅਨ੍ਤਰ੍ਭਾਵ <thread>
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_M 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_N 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_K 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VNNI_BLK 4

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ AMX_BLK_ਮਾਪ 32

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TMM0 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TMM1 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TMM2 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TMM3 3
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TMM4 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TMM5 5
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TMM6 6
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TMM7 7

// parallel routines
ਸਾਂਚਾ <typename T, typename std::enable_if<std::is_integral<T>::ਮੁੱਲ, ਪੂਰਨ>::type = 0>
ਸਤਰਅੰਦਰ T div_ਉੱਪਰ(T x, T y) { ਵਾਪਸ (x + y - 1) / y; }

ਸਾਂਚਾ <typename T>
ਸਤਰਅੰਦਰ ਖਾਲੀ balance211(T n, T nth, T ith, T& n_ਸ਼ੁਰੂ, T& n_ਸਿਰਾ) {
#ਜੇ 0
    // onednn partition pattern
    T& n_my = n_ਸਿਰਾ;
    ਜੇ (nth <= 1 || n == 0) {
        n_ਸ਼ੁਰੂ = 0;
        n_my = n;
    } ਨਹੀਂਤਾਂ {
        T n1 = div_ਉੱਪਰ(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_ਸ਼ੁਰੂ = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_ਸਿਰਾ += n_ਸ਼ੁਰੂ;
#ਨਹੀਂਤਾਂ
    // pytorch aten partition pattern
    T n_my = div_ਉੱਪਰ(n, nth);
    n_ਸ਼ੁਰੂ = ith * n_my;
    n_ਸਿਰਾ = std::min(n_ਸ਼ੁਰੂ + n_my, n);
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <typename func_t>
ਸਤਰਅੰਦਰ ਖਾਲੀ parallel_for(ਪੂਰਨ n, ਅਚਲ func_t & f) {
    ਜੇ (n <= 0) {
        ਵਾਪਸ;
    }
#ਜੇ defined(GGML_USE_OPENMP)
    #ਹਦਾਇਤ omp parallel
    {
        ਪੂਰਨ nth = omp_get_ਅੰਕ_ਧਾਗੇ();
        ਪੂਰਨ ith = omp_get_thread_ਅੰਕ();
        ਪੂਰਨ tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#ਨਹੀਂਤਾਂ
    ਪੂਰਨ nth = std::thread::hardware_concurrency();
    ਜੇ (nth <= 1) {
        f(0, n);
        ਵਾਪਸ;
    }
    ਜੇ (nth > n) {
        nth = n;
    }
    std::ਸਦਿਸ਼<std::thread> ਧਾਗੇ;
    ਧਾਗੇ.reserve(nth);
    ਲਈ (ਪੂਰਨ ith = 0; ith < nth; ++ith) {
        ਧਾਗੇ.emplace_ਪਿੱਠ([&f, n, ith, nth] {
            ਪੂਰਨ tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    ਲਈ (ਸਵੈਚਾਲਿਤ & t : ਧਾਗੇ) {
        t.join();
    }
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <typename func_t>
ਸਤਰਅੰਦਰ ਖਾਲੀ parallel_for_ggml(ਅਚਲ ggml_compute_params * params, ਪੂਰਨ n, ਅਚਲ func_t & f) {
    ਪੂਰਨ tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
ਸਤਰਅੰਦਰ bool qtype_has_amx_kernels(ਅਚਲ ਗਿਣਤੀ ggml_type type) {
    // TODO: fix padding for vnni format
    ਵਾਪਸ (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
