#هدايت once

#ضمن "ggml.h"
#ضمن "ggml-cpu-impl.h"

#ضمن <algorithm>
#ضمن <memory>
#ضمن <type_traits>

#إذا defined(GGML_USE_OPENMP)
#ضمن <omp.h>
#وإلا
#ضمن <thread>
#جيڪڏهن_پورو

#وصف_ڪريو TILE_M 16
#وصف_ڪريو TILE_N 16
#وصف_ڪريو TILE_K 32
#وصف_ڪريو VNNI_BLK 4

#وصف_ڪريو AMX_BLK_سائيز 32

#وصف_ڪريو TMM0 0
#وصف_ڪريو TMM1 1
#وصف_ڪريو TMM2 2
#وصف_ڪريو TMM3 3
#وصف_ڪريو TMM4 4
#وصف_ڪريو TMM5 5
#وصف_ڪريو TMM6 6
#وصف_ڪريو TMM7 7

// parallel routines
سانچو <typename T, typename std::enable_if<std::is_integral<T>::قدر, صحيح>::type = 0>
مضمن T div_مٿي(T x, T y) { أرجع (x + y - 1) / y; }

سانچو <typename T>
مضمن فراغ balance211(T n, T nth, T ith, T& n_شروعات, T& n_پڇاڙي) {
#إذا 0
    // onednn partition pattern
    T& n_my = n_پڇاڙي;
    إذا (nth <= 1 || n == 0) {
        n_شروعات = 0;
        n_my = n;
    } وإلا {
        T n1 = div_مٿي(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_شروعات = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_پڇاڙي += n_شروعات;
#وإلا
    // pytorch aten partition pattern
    T n_my = div_مٿي(n, nth);
    n_شروعات = ith * n_my;
    n_پڇاڙي = std::min(n_شروعات + n_my, n);
#جيڪڏهن_پورو
}

سانچو <typename func_t>
مضمن فراغ parallel_for(صحيح n, ثابت func_t & f) {
    إذا (n <= 0) {
        أرجع;
    }
#إذا defined(GGML_USE_OPENMP)
    #هدايت omp parallel
    {
        صحيح nth = omp_get_انگ_ڌاڳا();
        صحيح ith = omp_get_thread_انگ();
        صحيح tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#وإلا
    صحيح nth = std::thread::hardware_concurrency();
    إذا (nth <= 1) {
        f(0, n);
        أرجع;
    }
    إذا (nth > n) {
        nth = n;
    }
    std::ويڪٽر<std::thread> ڌاڳا;
    ڌاڳا.reserve(nth);
    لكل (صحيح ith = 0; ith < nth; ++ith) {
        ڌاڳا.emplace_پٺ([&f, n, ith, nth] {
            صحيح tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    لكل (خودڪار & t : ڌاڳا) {
        t.join();
    }
#جيڪڏهن_پورو
}

سانچو <typename func_t>
مضمن فراغ parallel_for_ggml(ثابت ggml_compute_params * params, صحيح n, ثابت func_t & f) {
    صحيح tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
مضمن bool qtype_has_amx_kernels(ثابت تعداد ggml_type type) {
    // TODO: fix padding for vnni format
    أرجع (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
