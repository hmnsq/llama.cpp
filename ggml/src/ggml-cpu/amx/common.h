#ہدایت once

#ضمن "ggml.h"
#ضمن "ggml-cpu-impl.h"

#ضمن <algorithm>
#ضمن <memory>
#ضمن <type_traits>

#إذا defined(GGML_USE_OPENMP)
#ضمن <omp.h>
#وإلا
#ضمن <thread>
#جے_انت

#تعریف_کرو TILE_M 16
#تعریف_کرو TILE_N 16
#تعریف_کرو TILE_K 32
#تعریف_کرو VNNI_BLK 4

#تعریف_کرو AMX_BLK_ناپ 32

#تعریف_کرو TMM0 0
#تعریف_کرو TMM1 1
#تعریف_کرو TMM2 2
#تعریف_کرو TMM3 3
#تعریف_کرو TMM4 4
#تعریف_کرو TMM5 5
#تعریف_کرو TMM6 6
#تعریف_کرو TMM7 7

// parallel routines
سانچا <typename T, typename std::enable_if<std::is_integral<T>::قدر, صحيح>::type = 0>
مضمن T div_اتے(T x, T y) { أرجع (x + y - 1) / y; }

سانچا <typename T>
مضمن فراغ balance211(T n, T nth, T ith, T& n_شروع, T& n_سرا) {
#إذا 0
    // onednn partition pattern
    T& n_my = n_سرا;
    إذا (nth <= 1 || n == 0) {
        n_شروع = 0;
        n_my = n;
    } وإلا {
        T n1 = div_اتے(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_شروع = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_سرا += n_شروع;
#وإلا
    // pytorch aten partition pattern
    T n_my = div_اتے(n, nth);
    n_شروع = ith * n_my;
    n_سرا = std::min(n_شروع + n_my, n);
#جے_انت
}

سانچا <typename func_t>
مضمن فراغ parallel_for(صحيح n, ثابت func_t & f) {
    إذا (n <= 0) {
        أرجع;
    }
#إذا defined(GGML_USE_OPENMP)
    #ہدایت omp parallel
    {
        صحيح nth = omp_get_عدد_دھاگے();
        صحيح ith = omp_get_thread_عدد();
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
    std::سمتیہ<std::thread> دھاگے;
    دھاگے.reserve(nth);
    لكل (صحيح ith = 0; ith < nth; ++ith) {
        دھاگے.emplace_پٹھ([&f, n, ith, nth] {
            صحيح tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    لكل (خودکار & t : دھاگے) {
        t.join();
    }
#جے_انت
}

سانچا <typename func_t>
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
