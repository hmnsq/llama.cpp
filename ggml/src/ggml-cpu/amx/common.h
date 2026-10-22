#رهنمود once

#ضمن "ggml.h"
#ضمن "ggml-cpu-impl.h"

#ضمن <algorithm>
#ضمن <memory>
#ضمن <type_traits>

#إذا defined(GGML_USE_OPENMP)
#ضمن <omp.h>
#وإلا
#ضمن <thread>
#پایان_اگر

#تعریف_کن TILE_M 16
#تعریف_کن TILE_N 16
#تعریف_کن TILE_K 32
#تعریف_کن VNNI_BLK 4

#تعریف_کن AMX_BLK_اندازه 32

#تعریف_کن TMM0 0
#تعریف_کن TMM1 1
#تعریف_کن TMM2 2
#تعریف_کن TMM3 3
#تعریف_کن TMM4 4
#تعریف_کن TMM5 5
#تعریف_کن TMM6 6
#تعریف_کن TMM7 7

// parallel routines
قالب <typename T, typename std::enable_if<std::is_integral<T>::مقدار, صحيح>::type = 0>
مضمن T div_به_بالا(T x, T y) { أرجع (x + y - 1) / y; }

قالب <typename T>
مضمن فراغ balance211(T n, T nth, T ith, T& n_شروع, T& n_انتها) {
#إذا 0
    // onednn partition pattern
    T& n_my = n_انتها;
    إذا (nth <= 1 || n == 0) {
        n_شروع = 0;
        n_my = n;
    } وإلا {
        T n1 = div_به_بالا(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_شروع = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_انتها += n_شروع;
#وإلا
    // pytorch aten partition pattern
    T n_my = div_به_بالا(n, nth);
    n_شروع = ith * n_my;
    n_انتها = std::min(n_شروع + n_my, n);
#پایان_اگر
}

قالب <typename func_t>
مضمن فراغ parallel_for(صحيح n, ثابت func_t & f) {
    إذا (n <= 0) {
        أرجع;
    }
#إذا defined(GGML_USE_OPENMP)
    #رهنمود omp parallel
    {
        صحيح nth = omp_get_رقم_نخها();
        صحيح ith = omp_get_thread_رقم();
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
    std::وکتور<std::thread> نخها;
    نخها.reserve(nth);
    لكل (صحيح ith = 0; ith < nth; ++ith) {
        نخها.emplace_پشت([&f, n, ith, nth] {
            صحيح tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    لكل (خودکار & t : نخها) {
        t.join();
    }
#پایان_اگر
}

قالب <typename func_t>
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
