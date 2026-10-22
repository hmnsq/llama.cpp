#لارښوونه once

#ضمن "ggml.h"
#ضمن "ggml-cpu-impl.h"

#ضمن <algorithm>
#ضمن <memory>
#ضمن <type_traits>

#إذا defined(GGML_USE_OPENMP)
#ضمن <omp.h>
#وإلا
#ضمن <thread>
#د_که_پای

#تعریف_کړه TILE_M 16
#تعریف_کړه TILE_N 16
#تعریف_کړه TILE_K 32
#تعریف_کړه VNNI_BLK 4

#تعریف_کړه AMX_BLK_اندازه 32

#تعریف_کړه TMM0 0
#تعریف_کړه TMM1 1
#تعریف_کړه TMM2 2
#تعریف_کړه TMM3 3
#تعریف_کړه TMM4 4
#تعریف_کړه TMM5 5
#تعریف_کړه TMM6 6
#تعریف_کړه TMM7 7

// parallel routines
کينډۍ <typename T, typename std::enable_if<std::is_integral<T>::ارزښت, صحيح>::type = 0>
مضمن T div_پورته(T x, T y) { أرجع (x + y - 1) / y; }

کينډۍ <typename T>
مضمن فراغ balance211(T n, T nth, T ith, T& n_پیل, T& n_پای_ټکی) {
#إذا 0
    // onednn partition pattern
    T& n_my = n_پای_ټکی;
    إذا (nth <= 1 || n == 0) {
        n_پیل = 0;
        n_my = n;
    } وإلا {
        T n1 = div_پورته(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_پیل = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_پای_ټکی += n_پیل;
#وإلا
    // pytorch aten partition pattern
    T n_my = div_پورته(n, nth);
    n_پیل = ith * n_my;
    n_پای_ټکی = std::min(n_پیل + n_my, n);
#د_که_پای
}

کينډۍ <typename func_t>
مضمن فراغ parallel_for(صحيح n, ثابت func_t & f) {
    إذا (n <= 0) {
        أرجع;
    }
#إذا defined(GGML_USE_OPENMP)
    #لارښوونه omp parallel
    {
        صحيح nth = omp_get_عدد_تارونه();
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
    std::ویکټور<std::thread> تارونه;
    تارونه.reserve(nth);
    لكل (صحيح ith = 0; ith < nth; ++ith) {
        تارونه.emplace_شا([&f, n, ith, nth] {
            صحيح tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    لكل (خپلکاری & t : تارونه) {
        t.join();
    }
#د_که_پای
}

کينډۍ <typename func_t>
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
