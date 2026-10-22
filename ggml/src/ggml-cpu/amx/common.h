#ہدایت once

#شامل "ggml.h"
#شامل "ggml-cpu-impl.h"

#شامل <algorithm>
#شامل <memory>
#شامل <type_traits>

#اگر defined(GGML_USE_OPENMP)
#شامل <omp.h>
#ورنہ
#شامل <thread>
#اگر_ختم

#تعریف_کرو TILE_M 16
#تعریف_کرو TILE_N 16
#تعریف_کرو TILE_K 32
#تعریف_کرو VNNI_BLK 4

#تعریف_کرو AMX_BLK_سائز 32

#تعریف_کرو TMM0 0
#تعریف_کرو TMM1 1
#تعریف_کرو TMM2 2
#تعریف_کرو TMM3 3
#تعریف_کرو TMM4 4
#تعریف_کرو TMM5 5
#تعریف_کرو TMM6 6
#تعریف_کرو TMM7 7

// parallel routines
سانچہ <typename T, typename std::enable_if<std::is_integral<T>::قدر, صحیح>::type = 0>
اندرونی T div_اوپر(T x, T y) { واپس (x + y - 1) / y; }

سانچہ <typename T>
اندرونی خالی balance211(T n, T nth, T ith, T& n_آغاز, T& n_انتہا) {
#اگر 0
    // onednn partition pattern
    T& n_my = n_انتہا;
    اگر (nth <= 1 || n == 0) {
        n_آغاز = 0;
        n_my = n;
    } ورنہ {
        T n1 = div_اوپر(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_آغاز = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_انتہا += n_آغاز;
#ورنہ
    // pytorch aten partition pattern
    T n_my = div_اوپر(n, nth);
    n_آغاز = ith * n_my;
    n_انتہا = std::min(n_آغاز + n_my, n);
#اگر_ختم
}

سانچہ <typename func_t>
اندرونی خالی parallel_for(صحیح n, ثابت func_t & f) {
    اگر (n <= 0) {
        واپس;
    }
#اگر defined(GGML_USE_OPENMP)
    #ہدایت omp parallel
    {
        صحیح nth = omp_get_عدد_دھاگے();
        صحیح ith = omp_get_thread_عدد();
        صحیح tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#ورنہ
    صحیح nth = std::thread::hardware_concurrency();
    اگر (nth <= 1) {
        f(0, n);
        واپس;
    }
    اگر (nth > n) {
        nth = n;
    }
    std::سمتیہ<std::thread> دھاگے;
    دھاگے.reserve(nth);
    برائے (صحیح ith = 0; ith < nth; ++ith) {
        دھاگے.emplace_عقب([&f, n, ith, nth] {
            صحیح tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    برائے (خودکار & t : دھاگے) {
        t.join();
    }
#اگر_ختم
}

سانچہ <typename func_t>
اندرونی خالی parallel_for_ggml(ثابت ggml_compute_params * params, صحیح n, ثابت func_t & f) {
    صحیح tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
اندرونی bool qtype_has_amx_kernels(ثابت شمار ggml_type type) {
    // TODO: fix padding for vnni format
    واپس (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
