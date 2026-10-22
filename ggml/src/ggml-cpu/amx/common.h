#సూచన once

#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-cpu-impl.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <memory>
#అన్తర్భావ <type_traits>

#ఒకవేళ defined(GGML_USE_OPENMP)
#అన్తర్భావ <omp.h>
#లేకపోతే
#అన్తర్భావ <thread>
#అయితే_ముగింపు

#నిర్వచించు TILE_M 16
#నిర్వచించు TILE_N 16
#నిర్వచించు TILE_K 32
#నిర్వచించు VNNI_BLK 4

#నిర్వచించు AMX_BLK_కొలత 32

#నిర్వచించు TMM0 0
#నిర్వచించు TMM1 1
#నిర్వచించు TMM2 2
#నిర్వచించు TMM3 3
#నిర్వచించు TMM4 4
#నిర్వచించు TMM5 5
#నిర్వచించు TMM6 6
#నిర్వచించు TMM7 7

// parallel routines
మూస <typename T, typename std::enable_if<std::is_integral<T>::విలువ, పూర్ణాంకం>::type = 0>
అంతఃవరుస T div_పైకి(T x, T y) { తిరిగి (x + y - 1) / y; }

మూస <typename T>
అంతఃవరుస శూన్య balance211(T n, T nth, T ith, T& n_ప్రారంభం, T& n_చివర) {
#ఒకవేళ 0
    // onednn partition pattern
    T& n_my = n_చివర;
    ఒకవేళ (nth <= 1 || n == 0) {
        n_ప్రారంభం = 0;
        n_my = n;
    } లేకపోతే {
        T n1 = div_పైకి(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_ప్రారంభం = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_చివర += n_ప్రారంభం;
#లేకపోతే
    // pytorch aten partition pattern
    T n_my = div_పైకి(n, nth);
    n_ప్రారంభం = ith * n_my;
    n_చివర = std::min(n_ప్రారంభం + n_my, n);
#అయితే_ముగింపు
}

మూస <typename func_t>
అంతఃవరుస శూన్య parallel_for(పూర్ణాంకం n, స్థిరరాశి func_t & f) {
    ఒకవేళ (n <= 0) {
        తిరిగి;
    }
#ఒకవేళ defined(GGML_USE_OPENMP)
    #సూచన omp parallel
    {
        పూర్ణాంకం nth = omp_get_అంకె_దారాలు();
        పూర్ణాంకం ith = omp_get_thread_అంకె();
        పూర్ణాంకం tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#లేకపోతే
    పూర్ణాంకం nth = std::thread::hardware_concurrency();
    ఒకవేళ (nth <= 1) {
        f(0, n);
        తిరిగి;
    }
    ఒకవేళ (nth > n) {
        nth = n;
    }
    std::సదిశ<std::thread> దారాలు;
    దారాలు.reserve(nth);
    కొరకు (పూర్ణాంకం ith = 0; ith < nth; ++ith) {
        దారాలు.emplace_వెనుక([&f, n, ith, nth] {
            పూర్ణాంకం tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    కొరకు (స్వయంచాలక & t : దారాలు) {
        t.join();
    }
#అయితే_ముగింపు
}

మూస <typename func_t>
అంతఃవరుస శూన్య parallel_for_ggml(స్థిరరాశి ggml_compute_params * params, పూర్ణాంకం n, స్థిరరాశి func_t & f) {
    పూర్ణాంకం tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
అంతఃవరుస bool qtype_has_amx_kernels(స్థిరరాశి గణన ggml_type type) {
    // TODO: fix padding for vnni format
    తిరిగి (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
