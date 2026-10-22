#అన్తర్భావ "common.hpp"
#అన్తర్భావ "ggml-sycl/presets.hpp"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "element_wise.hpp"

#నిర్వచించు SYCL_GLOBAL_ID_LOOP(K, అంశం_2) \
    కొరకు (స్వయంచాలక i = అంశం_2.get_global_id(0); i < (size_t)K; i += అంశం_2.get_global_range(0))

#నిర్వచించు SYCL_LOCAL_ID_CALC(అంశం_2, IDX) \
    (అంశం_2.get_local_range(IDX) * అంశం_2.get_సమూహం(IDX) + అంశం_2.get_local_id(IDX))

స్థిర శూన్య acc_f32(స్థిరరాశి అక్షరం * x, స్థిరరాశి అక్షరం * y, దశాంశం * dst, స్థిరరాశి int64_t ne,
    స్థిరరాశి int64_t ne0, స్థిరరాశి int64_t ne1, స్థిరరాశి int64_t ne2,
    స్థిరరాశి int64_t nb00, స్థిరరాశి int64_t nb01, స్థిరరాశి int64_t nb02, స్థిరరాశి int64_t nb03,
    స్థిరరాశి int64_t ne10, స్థిరరాశి int64_t ne11, స్థిరరాశి int64_t ne12, స్థిరరాశి int64_t ne13,
    స్థిరరాశి int64_t nb10, స్థిరరాశి int64_t nb11, స్థిరరాశి int64_t nb12, స్థిరరాశి int64_t nb13,
    స్థిరరాశి int64_t s11, స్థిరరాశి int64_t s12, స్థిరరాశి int64_t s13, స్థిరరాశి int64_t స్థానభ్రంశం) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి int64_t i = SYCL_LOCAL_ID_CALC(అంశం_ct1, 2);

    ఒకవేళ (i >= ne) {
        తిరిగి;
    }

    int64_t src1_idx = i - స్థానభ్రంశం;

    int64_t tmp = src1_idx;
    స్థిరరాశి int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    స్థిరరాశి int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    స్థిరరాశి int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    స్థిరరాశి int64_t i10 = tmp;

    int64_t tmp_dst = i;
    స్థిరరాశి int64_t i3 = tmp_dst / (ne2*ne1*ne0);
    tmp_dst -= i3 * (ne2*ne1*ne0);
    స్థిరరాశి int64_t i2 = tmp_dst / (ne1*ne0);
    tmp_dst -= i2 * (ne1*ne0);
    స్థిరరాశి int64_t i1 = tmp_dst / ne0;
    tmp_dst -= i1 * ne0;
    స్థిరరాశి int64_t i0 = tmp_dst;

    దశాంశం val = *(స్థిరరాశి దశాంశం *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    ఒకవేళ (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(స్థిరరాశి దశాంశం *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_sgn(T x) {
    తిరిగి x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_abs(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } లేకపోతే {
        తిరిగి sycl::fabs(x);
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_expm1(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<దశాంశం>(x))
        );
    } లేకపోతే {
        తిరిగి sycl::expm1(x);
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_elu(T x) {
    తిరిగి (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_erf(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<దశాంశం>(x))
        );
    } లేకపోతే {
        తిరిగి sycl::erf(x);
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_gelu_erf(T x) {
    స్థిరరాశి T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    తిరిగి static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_gelu_శీఘ్ర(T x) {
    స్థిరరాశి T GELU_శీఘ్ర_COEF_LOCAL = static_cast<T>(-1.702f);
    తిరిగి x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_శీఘ్ర_COEF_LOCAL * x)));
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_relu(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } లేకపోతే {
        తిరిగి sycl::fmax(x, static_cast<T>(0));
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_sigmoid(T x) {
    తిరిగి static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_sqrt(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::experimental::వర్గమూలం(x);
    } లేకపోతే {
        తిరిగి sycl::వర్గమూలం(x);
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_sin(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::experimental::జ్యా(x);
    } లేకపోతే {
        తిరిగి sycl::జ్యా(x);
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_cos(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::experimental::కోజ్యా(x);
    } లేకపోతే {
        తిరిగి sycl::కోజ్యా(x);
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_hardsigmoid(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } లేకపోతే {
        తిరిగి sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_hardswish(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } లేకపోతే {
        తిరిగి x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_log(T x) {
    ఒకవేళ (x <= static_cast<T>(0)) {
        తిరిగి neg_infinity<T>();
    }
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::experimental::సంవర్గమానం(x);
    } లేకపోతే {
        తిరిగి sycl::సంవర్గమానం(x);
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_softplus(T x) {
    స్థిరరాశి దశాంశం xf = (దశాంశం) x;
    స్థిరరాశి దశాంశం ax = op_abs(xf);
    స్థిరరాశి దశాంశం m  = sycl::fmax(xf, 0.0f);
    స్థిరరాశి దశాంశం y  = m + sycl::log1p(sycl::ఘాతాంకం(-ax));
    తిరిగి (T) y;
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_neg(T x) {
    తిరిగి -x;
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_అడుగు(T x) {
    తిరిగి (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_leaky_relu(T x, దశాంశం negative_వాలు) {
    T neg_వాలు_T = static_cast<T>(negative_వాలు);
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_వాలు_T;

    } లేకపోతే {
        తిరిగి sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_వాలు_T;
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_xielu(T x, దశాంశం alpha_n, దశాంశం alpha_p, దశాంశం beta, దశాంశం eps) {
    స్థిరరాశి దశాంశం xi        = static_cast<దశాంశం>(x);
    స్థిరరాశి దశాంశం gate_pos  = (xi > 0.0f);
    స్థిరరాశి దశాంశం y_pos     = alpha_p * xi * xi + beta * xi;
    స్థిరరాశి దశాంశం min_v_eps = sycl::fmin(xi, eps);
    స్థిరరాశి దశాంశం y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    స్థిరరాశి దశాంశం బహిర్గతం       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    తిరిగి static_cast<T>(బహిర్గతం);
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_sqr(T x) {
    తిరిగి x * x;
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_clamp(T x, దశాంశం min_val, దశాంశం max_val) {
    తిరిగి x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_floor(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::experimental::floor(x);
    } లేకపోతే {
        తిరిగి sycl::floor(x);
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_ceil(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::experimental::ceil(x);
    } లేకపోతే {
        తిరిగి sycl::ceil(x);
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_విడత(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<దశాంశం>(x))
        );
    } లేకపోతే {
        తిరిగి sycl::round(x);
    }
}

మూస<రకం_పేరు T>
స్థిర __dpct_inline__ T op_trunc(T x) {
    ఒకవేళ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        తిరిగి sycl::ext::oneapi::experimental::trunc(x);
    } లేకపోతే {
        తిరిగి sycl::trunc(x);
    }
}

మూస<రకం_పేరు T, రకం_పేరు F>
స్థిర శూన్య unary_op_flat_kernel(స్థిరరాశి T * x, T * dst, స్థిరరాశి పూర్ణాంకం k, స్థిరరాశి sycl::nd_అంశం<1> & అంశం_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = func(x[i]);
    }
}

మూస<రకం_పేరు T, రకం_పేరు F>
స్థిర శూన్య unary_op_generic_kernel(
        స్థిరరాశి T * x,
        T * dst,
        స్థిరరాశి పూర్ణాంకం k,
        స్థిరరాశి sycl::uint3 ne0_fd, స్థిరరాశి sycl::uint3 ne1_fd, స్థిరరాశి sycl::uint3 ne2_fd,
        స్థిరరాశి size_t nb0,  స్థిరరాశి size_t nb1,  స్థిరరాశి size_t nb2,  స్థిరరాశి size_t nb3,
        స్థిరరాశి size_t nbd0, స్థిరరాశి size_t nbd1, స్థిరరాశి size_t nbd2, స్థిరరాశి size_t nbd3,
        స్థిరరాశి sycl::nd_అంశం<1> & అంశం_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        స్థిరరాశి uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        స్థిరరాశి uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        స్థిరరాశి uint32_t i2 = dm.y();
        స్థిరరాశి uint32_t i3 = dm.x();

        స్థిరరాశి అక్షరం * src_పునాది = (స్థిరరాశి అక్షరం *) x;
        అక్షరం       * dst_పునాది = (అక్షరం *) dst;

        స్థిరరాశి T * srcp = (స్థిరరాశి T *)(src_పునాది + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_పునాది + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

మూస<రకం_పేరు T>
స్థిర శూన్య unary_op_sqrt_kernel(స్థిరరాశి T * x, T * dst, స్థిరరాశి పూర్ణాంకం k, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

మూస<రకం_పేరు T>
స్థిర శూన్య unary_op_sin_kernel(స్థిరరాశి T * x, T * dst, స్థిరరాశి పూర్ణాంకం k, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

మూస<రకం_పేరు T>
స్థిర శూన్య unary_op_cos_kernel(స్థిరరాశి T * x, T * dst, స్థిరరాశి పూర్ణాంకం k, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

మూస<రకం_పేరు T>
స్థిర శూన్య unary_op_log_kernel(స్థిరరాశి T * x, T * dst, స్థిరరాశి పూర్ణాంకం k, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = op_log(x[i]);
    }
}


మూస<రకం_పేరు T>
స్థిర శూన్య unary_op_leaky_relu_kernel(స్థిరరాశి T * x, T * dst, స్థిరరాశి పూర్ణాంకం k, దశాంశం negative_వాలు, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_వాలు);
    }
}

మూస<రకం_పేరు T>
స్థిర శూన్య unary_op_xielu_kernel(స్థిరరాశి T * x, T * dst, స్థిరరాశి పూర్ణాంకం k, దశాంశం alpha_n, దశాంశం alpha_p, దశాంశం beta, దశాంశం eps, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

మూస<రకం_పేరు T>
స్థిర శూన్య unary_op_sqr_kernel(స్థిరరాశి T * x, T * dst, స్థిరరాశి పూర్ణాంకం k, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

మూస<రకం_పేరు T>
స్థిర శూన్య unary_op_clamp_kernel(స్థిరరాశి T * x, T * dst, స్థిరరాశి పూర్ణాంకం k, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1, దశాంశం min_val, దశాంశం max_val) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

మూస<రకం_పేరు T>
స్థిర శూన్య unary_op_ceil_kernel(స్థిరరాశి T * x, T * dst, స్థిరరాశి పూర్ణాంకం k, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

మూస<రకం_పేరు T>
స్థిర శూన్య clamp(స్థిరరాశి T * x, T * dst, స్థిరరాశి దశాంశం min, స్థిరరాశి దశాంశం max, స్థిరరాశి పూర్ణాంకం k,
                      స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

మూస<రకం_పేరు T, రకం_పేరు F>
స్థిర శూన్య unary_gated_op_flat_kernel(స్థిరరాశి T * x, స్థిరరాశి T * g, T * dst, స్థిరరాశి uint64_t k, స్థిరరాశి sycl::nd_అంశం<1> & అంశం_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

మూస<రకం_పేరు T, రకం_పేరు F>
స్థిర శూన్య unary_gated_op_generic_kernel(
        స్థిరరాశి T * x,
        స్థిరరాశి T * g,
        T * dst,
        స్థిరరాశి uint64_t k,
        స్థిరరాశి sycl::uint3 n_fd,
        స్థిరరాశి uint64_t o0,
        స్థిరరాశి uint64_t o1,
        స్థిరరాశి sycl::nd_అంశం<1> & అంశం_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        స్థిరరాశి sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        స్థిరరాశి int64_t j0 = rc.x() * o0 + rc.y();
        స్థిరరాశి int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
మూస<రకం_పేరు T, రకం_పేరు F>
స్థిర శూన్య unary_mul_flat_kernel(స్థిరరాశి T * x, స్థిరరాశి T * g, T * dst, స్థిరరాశి int64_t k, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = (T) (op((దశాంశం) x[i]) * (దశాంశం) g[i]);
    }
}

మూస<రకం_పేరు T, రకం_పేరు F>
స్థిర శూన్య unary_mul_strided_kernel(స్థిరరాశి T * x, స్థిరరాశి T * g, T * dst, స్థిరరాశి int64_t k, స్థిరరాశి sycl::uint3 n_fd, స్థిరరాశి int64_t o0, స్థిరరాశి int64_t o1, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        స్థిరరాశి sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        స్థిరరాశి int64_t j0 = rc.x() * o0 + rc.y();
        స్థిరరాశి int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((దశాంశం) x[j0]) * (దశాంశం) g[j1]);
    }
}

మూస<రకం_పేరు T, రకం_పేరు F>
స్థిర శూన్య unary_mul_sycl(స్థిరరాశి T * x, స్థిరరాశి T * g, T * dst, స్థిరరాశి int64_t k, స్థిరరాశి int64_t n, స్థిరరాశి int64_t o0, స్థిరరాశి int64_t o1, వరుస_ptr main_ప్రవాహం, F op) {
    స్థిరరాశి size_t            అంకె_ఖండాలు = ceil_div((size_t) k, (size_t) SYCL_GLU_ఖండం_కొలత);
    స్థిరరాశి sycl::nd_range<1> range(అంకె_ఖండాలు * sycl::range<1>(SYCL_GLU_ఖండం_కొలత), sycl::range<1>(SYCL_GLU_ఖండం_కొలత));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    ఒకవేళ (o0 == n && o1 == n) {
        main_ప్రవాహం->parallel_for(range, [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            unary_mul_flat_kernel(x, g, dst, k, అంశం_ct1, op);
        });
        తిరిగి;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    స్థిరరాశి sycl::uint3 n_fd = init_fastdiv_విలువలు((uint32_t) n);
    main_ప్రవాహం->parallel_for(range, [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, అంశం_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
మూస<రకం_పేరు F>
స్థిర శూన్య కూడు_unary_mul_flat_kernel(స్థిరరాశి దశాంశం * a, స్థిరరాశి దశాంశం * bias, స్థిరరాశి దశాంశం * scale, దశాంశం * dst,
                                      స్థిరరాశి int64_t k, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

మూస<రకం_పేరు F>
స్థిర శూన్య కూడు_unary_mul_bcast_kernel(స్థిరరాశి దశాంశం * a, స్థిరరాశి దశాంశం * bias, స్థిరరాశి దశాంశం * scale, దశాంశం * dst,
                                       స్థిరరాశి int64_t k, స్థిరరాశి sycl::uint3 ne0_fd, స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        స్థిరరాశి uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

మూస<రకం_పేరు F>
స్థిర శూన్య కూడు_unary_mul_sycl(స్థిరరాశి దశాంశం * a, స్థిరరాశి దశాంశం * bias, స్థిరరాశి దశాంశం * scale, దశాంశం * dst,
                               స్థిరరాశి int64_t k, స్థిరరాశి int64_t ne0, వరుస_ptr main_ప్రవాహం, F op) {
    స్థిరరాశి size_t            అంకె_ఖండాలు = ceil_div((size_t) k, (size_t) SYCL_GLU_ఖండం_కొలత);
    స్థిరరాశి sycl::nd_range<1> range(అంకె_ఖండాలు * sycl::range<1>(SYCL_GLU_ఖండం_కొలత), sycl::range<1>(SYCL_GLU_ఖండం_కొలత));

    ఒకవేళ (k == ne0) {
        main_ప్రవాహం->parallel_for(range, [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            కూడు_unary_mul_flat_kernel(a, bias, scale, dst, k, అంశం_ct1, op);
        });
        తిరిగి;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    స్థిరరాశి sycl::uint3 ne0_fd = init_fastdiv_విలువలు((uint32_t) ne0);
    main_ప్రవాహం->parallel_for(range, [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
        కూడు_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, అంశం_ct1, op);
    });
}

నామాకాశ ggml_sycl_detail {
స్థిర శూన్య acc_f32_sycl(స్థిరరాశి అక్షరం *x, స్థిరరాశి అక్షరం *y, దశాంశం *dst,
                         స్థిరరాశి int64_t n_elements,
                         స్థిరరాశి int64_t ne0, స్థిరరాశి int64_t ne1, స్థిరరాశి int64_t ne2,
                         స్థిరరాశి int64_t nb00, స్థిరరాశి int64_t nb01, స్థిరరాశి int64_t nb02, స్థిరరాశి int64_t nb03,
                         స్థిరరాశి int64_t ne10, స్థిరరాశి int64_t ne11, స్థిరరాశి int64_t ne12, స్థిరరాశి int64_t ne13,
                         స్థిరరాశి int64_t nb10, స్థిరరాశి int64_t nb11, స్థిరరాశి int64_t nb12, స్థిరరాశి int64_t nb13,
                         స్థిరరాశి int64_t s1, స్థిరరాశి int64_t s2, స్థిరరాశి int64_t s3,
                         స్థిరరాశి int64_t స్థానభ్రంశం, వరుస_ptr stream) {
    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = (n_elements + SYCL_ACC_ఖండం_కొలత - 1) / SYCL_ACC_ఖండం_కొలత;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, అంకె_ఖండాలు) * sycl::range<3>(1, 1, SYCL_ACC_ఖండం_కొలత),
                                           sycl::range<3>(1, 1, SYCL_ACC_ఖండం_కొలత)),
                         [=](sycl::nd_అంశం<3> /*item_ct1*/) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                            acc_f32(x, y, dst, n_elements,
                                ne0, ne1, ne2,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                s1, s2, s3, స్థానభ్రంశం);
                         });
}

మూస<రకం_పేరు T>
స్థిర శూన్య arange_kernel(T * dst, స్థిరరాశి పూర్ణాంకం k, T start, T అడుగు,
                         స్థిరరాశి sycl::nd_అంశం<1> &అంశం_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, అంశం_ct1) {
        dst[i] = start + static_cast<T>(i) * అడుగు;
    }
}

మూస<రకం_పేరు KernelInvoker, రకం_పేరు... వాదనలు>
స్థిర అంతఃవరుస శూన్య dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, వాదనలు&&... వాదనలు_2) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    ఎంపిక (dst->type) {
        స్థితి GGML_TYPE_F16:
            {
                స్వయంచాలక దత్తాంశం_pts = cast_దత్తాంశం<sycl::half>(dst);
                kernel_invoker(దత్తాంశం_pts.src, దత్తాంశం_pts.dst, (పూర్ణాంకం)ggml_nelements(dst->src[0]), main_ప్రవాహం, std::forward<వాదనలు>(వాదనలు_2)...);
                ఆపు;
            }
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
        స్థితి GGML_TYPE_BF16:
            {
                స్వయంచాలక దత్తాంశం_pts = cast_దత్తాంశం<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(దత్తాంశం_pts.src, దత్తాంశం_pts.dst, (పూర్ణాంకం)ggml_nelements(dst->src[0]), main_ప్రవాహం, std::forward<వాదనలు>(వాదనలు_2)...);
                ఆపు;
            }
#అయితే_ముగింపు
        స్థితి GGML_TYPE_F32:
            {
                స్వయంచాలక దత్తాంశం_pts = cast_దత్తాంశం<దశాంశం>(dst);
                kernel_invoker(దత్తాంశం_pts.src, దత్తాంశం_pts.dst, (పూర్ణాంకం)ggml_nelements(dst->src[0]), main_ప్రవాహం, std::forward<వాదనలు>(వాదనలు_2)...);
                ఆపు;
            }
        అప్రమేయ:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

మూస<రకం_పేరు KernelInvoker, రకం_పేరు... వాదనలు>
స్థిర అంతఃవరుస శూన్య dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, వాదనలు&&... వాదనలు_2) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    స్థిరరాశి int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // The fused GLU kernels index with 32-bit fastdiv, which is exact only for indices below
    // 2^31. A dst that large is ~8 GB at f32, and the grid sizing already narrows to 32 bits,
    // so assert the bound rather than carry a second code path for it.
    GGML_ASSERT(ggml_nelements(dst) < ((int64_t) 1 << 31));
    స్థిరరాశి int32_t swapped = ((స్థిరరాశి int32_t *) dst->op_params)[1];
    శూన్య * src0_d = src0->దత్తాంశం;
    శూన్య * src1_d = src1 ? src1->దత్తాంశం : src0->దత్తాంశం;
    స్థిరరాశి int64_t src0_o = src0->nb[1];
    స్థిరరాశి int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    శూన్య * dst_d = dst->దత్తాంశం;
    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_కొలత(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    ఎంపిక (dst->type) {
        స్థితి GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    ఒకవేళ (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / పరిమాణం(sycl::half),
                               src1_o / పరిమాణం(sycl::half),
                               main_ప్రవాహం,
                               std::forward<వాదనలు>(వాదనలు_2)...);
                ఆపు;
            }
        స్థితి GGML_TYPE_F32:
            {
                దశాంశం * src0_p = (దశాంశం *) src0_d;
                దశాంశం * src1_p = (దశాంశం *) src1_d;

                    ఒకవేళ (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (దశాంశం *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / పరిమాణం(దశాంశం),
                               src1_o / పరిమాణం(దశాంశం),
                               main_ప్రవాహం,
                               std::forward<వాదనలు>(వాదనలు_2)...);
                ఆపు;
            }
        అప్రమేయ:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

మూస<రకం_పేరు F>
స్థిర అంతఃవరుస శూన్య ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    స్థిరరాశి int64_t ne0  = dst->ne[0];
    స్థిరరాశి int64_t ne1  = dst->ne[1];
    స్థిరరాశి int64_t ne2  = dst->ne[2];

    స్థిరరాశి size_t  nb0  = src0->nb[0];
    స్థిరరాశి size_t  nb1  = src0->nb[1];
    స్థిరరాశి size_t  nb2  = src0->nb[2];
    స్థిరరాశి size_t  nb3  = src0->nb[3];

    స్థిరరాశి size_t  nbd0 = dst->nb[0];
    స్థిరరాశి size_t  nbd1 = dst->nb[1];
    స్థిరరాశి size_t  nbd2 = dst->nb[2];
    స్థిరరాశి size_t  nbd3 = dst->nb[3];

    // Hot unary ops (FFN/GDN silu, sigmoid, ...) run on contiguous tensors;
    // skip the strided index math entirely for them.
    స్థిరరాశి తార్కిక contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](స్థిరరాశి స్వయంచాలక* src, స్వయంచాలక* dst_ptr, పూర్ణాంకం k_elements, వరుస_ptr stream) {

            స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = ceil_div(k_elements, 256);

            ఒకవేళ (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, అంశం_ct1, func);
                    });
            } లేకపోతే {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                స్థిరరాశి sycl::uint3 ne0_fd = init_fastdiv_విలువలు((uint32_t) ne0);
                స్థిరరాశి sycl::uint3 ne1_fd = init_fastdiv_విలువలు((uint32_t) ne1);
                స్థిరరాశి sycl::uint3 ne2_fd = init_fastdiv_విలువలు((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        unary_op_generic_kernel(
                            src, dst_ptr, k_elements,
                            ne0_fd, ne1_fd, ne2_fd,
                            nb0, nb1, nb2, nb3,
                            nbd0, nbd1, nbd2, nbd3,
                            అంశం_ct1,
                            func
                        );
                    });
            }
        });
}

మూస<రకం_పేరు F>
స్థిర అంతఃవరుస శూన్య ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](స్థిరరాశి స్వయంచాలక * x_ptr, స్థిరరాశి స్వయంచాలక * g_ptr, స్వయంచాలక * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, వరుస_ptr main_ప్రవాహం) {

            స్థిరరాశి uint32_t అంకె_ఖండాలు = (uint32_t) ceil_div(k, SYCL_GLU_ఖండం_కొలత);
            స్థిరరాశి sycl::nd_range<1> launch_range(అంకె_ఖండాలు * sycl::range<1>(SYCL_GLU_ఖండం_కొలత),
                                                 sycl::range<1>(SYCL_GLU_ఖండం_కొలత));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            ఒకవేళ (o0 == n && o1 == n) {
                main_ప్రవాహం->parallel_for(launch_range,
                    [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, అంశం_ct1, func);
                    });
            } లేకపోతే {
                // launch-invariant divisor, and only this path needs it
                స్థిరరాశి sycl::uint3 n_fd = init_fastdiv_విలువలు((uint32_t) n);
                main_ప్రవాహం->parallel_for(launch_range,
                    [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, అంశం_ct1, func);
                    });
            }
        });
}


స్థిర అంతఃవరుస శూన్య ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    దశాంశం start, stop, అడుగు;
    జ్ఞాపకం_నకలు(&start, dst->op_params, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&stop, (దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&అడుగు, (దశాంశం *) dst->op_params + 2, పరిమాణం(దశాంశం));
    dpct::వరుస_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    దశాంశం * dst_ptr = (దశాంశం *)dst->దత్తాంశం;
    స్థిరరాశి పూర్ణాంకం k = (పూర్ణాంకం)ggml_nelements(dst);
    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = ceil_div(k, SYCL_ARANGE_ఖండం_కొలత);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు) * sycl::range<1>(SYCL_ARANGE_ఖండం_కొలత),
                          sycl::range<1>(SYCL_ARANGE_ఖండం_కొలత)),
        [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            arange_kernel(dst_ptr, k, start, అడుగు, అంశం_ct1);
        });
}

} // namespace ggml_sycl_detail



స్థిర అంతఃవరుస శూన్య ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_sgn(x);
    });
}


స్థిర అంతఃవరుస శూన్య ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_abs(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_elu(x);
    });
}
స్థిర అంతఃవరుస శూన్య ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_silu(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_gelu(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_gelu_శీఘ్ర(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_gelu_శీఘ్ర(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_gelu_erf(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_tanh(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_relu(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_hardsigmoid(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_hardswish(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_exp(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_expm1(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](స్థిరరాశి స్వయంచాలక* src, స్వయంచాలక* dst_ptr, పూర్ణాంకం k_elements, వరుస_ptr stream) {
            స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = ceil_div(k_elements, SYCL_EXP_ఖండం_కొలత); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు) * sycl::range<1>(SYCL_EXP_ఖండం_కొలత),
                                  sycl::range<1>(SYCL_EXP_ఖండం_కొలత)),
                [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, అంశం_ct1);
                });
        });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_softplus(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_neg(x);
    });
}


స్థిర అంతఃవరుస శూన్య ggml_sycl_op_అడుగు(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_అడుగు(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_sigmoid(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](స్థిరరాశి స్వయంచాలక* src, స్వయంచాలక* dst_ptr, పూర్ణాంకం k_elements, వరుస_ptr stream) {
            స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = ceil_div(k_elements, SYCL_SQRT_ఖండం_కొలత);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు) * sycl::range<1>(SYCL_SQRT_ఖండం_కొలత),
                                  sycl::range<1>(SYCL_SQRT_ఖండం_కొలత)),
                [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, అంశం_ct1);
                });
        });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](స్థిరరాశి స్వయంచాలక* src, స్వయంచాలక* dst_ptr, పూర్ణాంకం k_elements, వరుస_ptr stream) {
            స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = ceil_div(k_elements, SYCL_SIN_ఖండం_కొలత);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు) * sycl::range<1>(SYCL_SIN_ఖండం_కొలత),
                                  sycl::range<1>(SYCL_SIN_ఖండం_కొలత)),
                [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, అంశం_ct1);
                });
        });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](స్థిరరాశి స్వయంచాలక* src, స్వయంచాలక* dst_ptr, పూర్ణాంకం k_elements, వరుస_ptr stream) {
            స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = ceil_div(k_elements, SYCL_SIN_ఖండం_కొలత); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు) * sycl::range<1>(SYCL_SIN_ఖండం_కొలత),
                                  sycl::range<1>(SYCL_SIN_ఖండం_కొలత)),
                [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, అంశం_ct1);
                });
        });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    దశాంశం negative_వాలు;
    జ్ఞాపకం_నకలు(&negative_వాలు, dst->op_params, పరిమాణం(దశాంశం));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](స్థిరరాశి స్వయంచాలక* src, స్వయంచాలక* dst_ptr, పూర్ణాంకం k_elements, వరుస_ptr stream, దశాంశం వాలు) {
            స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = ceil_div(k_elements, SYCL_RELU_ఖండం_కొలత);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు) * sycl::range<1>(SYCL_RELU_ఖండం_కొలత),
                                  sycl::range<1>(SYCL_RELU_ఖండం_కొలత)),
                [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, వాలు, అంశం_ct1);
                });
        }, negative_వాలు);
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](స్థిరరాశి స్వయంచాలక* src, స్వయంచాలక* dst_ptr, పూర్ణాంకం k_elements, వరుస_ptr stream) {
            స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = ceil_div(k_elements, SYCL_SQR_ఖండం_కొలత);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు) * sycl::range<1>(SYCL_SQR_ఖండం_కొలత),
                                  sycl::range<1>(SYCL_SQR_ఖండం_కొలత)),
                [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, అంశం_ct1);
                });
        });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    దశాంశం min_val;
    దశాంశం max_val;
    జ్ఞాపకం_నకలు(&min_val, dst->op_params, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_val, (దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](స్థిరరాశి స్వయంచాలక* src, స్వయంచాలక* dst_ptr, పూర్ణాంకం k_elements, వరుస_ptr stream, దశాంశం min_arg, దశాంశం max_arg) {
            స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = ceil_div(k_elements, SYCL_CLAMP_ఖండం_కొలత);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు) * sycl::range<1>(SYCL_CLAMP_ఖండం_కొలత),
                                  sycl::range<1>(SYCL_CLAMP_ఖండం_కొలత)),
                [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, అంశం_ct1);
                });
        }, min_val, max_val);
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి దశాంశం alpha_n = ggml_get_op_params_f32(dst, 1);
    స్థిరరాశి దశాంశం alpha_p = ggml_get_op_params_f32(dst, 2);
    స్థిరరాశి దశాంశం beta    = ggml_get_op_params_f32(dst, 3);
    స్థిరరాశి దశాంశం eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](స్థిరరాశి స్వయంచాలక* src, స్వయంచాలక* dst_ptr, పూర్ణాంకం k_elements, వరుస_ptr stream, దశాంశం alpha_n_arg, దశాంశం alpha_p_arg, దశాంశం beta_arg, దశాంశం eps_arg) {
            స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = ceil_div(k_elements, SYCL_RELU_ఖండం_కొలత);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు) * sycl::range<1>(SYCL_RELU_ఖండం_కొలత),
                                  sycl::range<1>(SYCL_RELU_ఖండం_కొలత)),
                [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, అంశం_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_floor(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_ceil(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_విడత(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_విడత(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_trunc(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    స్థిరరాశి అక్షరం  * src0_d = (స్థిరరాశి అక్షరం  *) src0->దత్తాంశం;
    స్థిరరాశి అక్షరం  * src1_d = (స్థిరరాశి అక్షరం  *) src1->దత్తాంశం;
    దశాంశం       * dst_d  = (దశాంశం       *)  dst->దత్తాంశం;

    dpct::వరుస_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_కొలత(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    స్థిరరాశి int64_t s1     = (int64_t) ((స్థిరరాశి int32_t *) dst->op_params)[0] / (int64_t) పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t s2     = (int64_t) ((స్థిరరాశి int32_t *) dst->op_params)[1] / (int64_t) పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t s3     = (int64_t) ((స్థిరరాశి int32_t *) dst->op_params)[2] / (int64_t) పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t స్థానభ్రంశం = (int64_t) ((స్థిరరాశి int32_t *) dst->op_params)[3] / (int64_t) పరిమాణం(దశాంశం);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, స్థానభ్రంశం, stream);
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_gelu(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_relu(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
మూస<రకం_పేరు F>
స్థిర శూన్య dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
    ఎంపిక (uop) {
        స్థితి GGML_UNARY_OP_SILU:     launch([](దశాంశం v) { తిరిగి op_silu(v); });     ఆపు;
        స్థితి GGML_UNARY_OP_SIGMOID:  launch([](దశాంశం v) { తిరిగి op_sigmoid(v); });  ఆపు;
        స్థితి GGML_UNARY_OP_SOFTPLUS: launch([](దశాంశం v) { తిరిగి op_softplus(v); }); ఆపు;
        అప్రమేయ:
            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
    }
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
శూన్య ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_కణుపు, ggml_tensor * mul_కణుపు) {
    scope_op_debug_print scope_dbg_print(__func__, mul_కణుపు, /*num_src=*/2);

    స్థిరరాశి ggml_tensor * x = unary_కణుపు->src[0];
    స్థిరరాశి ggml_tensor * g = (mul_కణుపు->src[0] == unary_కణుపు) ? mul_కణుపు->src[1] : mul_కణుపు->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_కణుపు);
    GGML_ASSERT(x->type == g->type && x->type == mul_కణుపు->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_కణుపు));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_కణుపు));

    వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి int64_t k = ggml_nelements(mul_కణుపు);
    స్థిరరాశి int64_t n = mul_కణుపు->ne[0];

    స్థిరరాశి స్వయంచాలక dispatch_type = [&](స్వయంచాలక op) {
        ఎంపిక (mul_కణుపు->type) {
            స్థితి GGML_TYPE_F32:
                unary_mul_sycl((స్థిరరాశి దశాంశం *) x->దత్తాంశం, (స్థిరరాశి దశాంశం *) g->దత్తాంశం, (దశాంశం *) mul_కణుపు->దత్తాంశం,
                               k, n, x->nb[1] / పరిమాణం(దశాంశం), g->nb[1] / పరిమాణం(దశాంశం), main_ప్రవాహం, op);
                ఆపు;
            స్థితి GGML_TYPE_F16:
                unary_mul_sycl((స్థిరరాశి sycl::half *) x->దత్తాంశం, (స్థిరరాశి sycl::half *) g->దత్తాంశం, (sycl::half *) mul_కణుపు->దత్తాంశం,
                               k, n, x->nb[1] / పరిమాణం(sycl::half), g->nb[1] / పరిమాణం(sycl::half), main_ప్రవాహం, op);
                ఆపు;
            అప్రమేయ:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_కణుపు->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_కణుపు), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
శూన్య ggml_sycl_op_కూడు_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * కూడు_కణుపు,
                                      ggml_tensor * unary_కణుపు, ggml_tensor * mul_కణుపు) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_కణుపు, /*num_src=*/2);

    స్థిరరాశి ggml_tensor * a     = కూడు_కణుపు->src[0];
    స్థిరరాశి ggml_tensor * bias  = కూడు_కణుపు->src[1];
    స్థిరరాశి ggml_tensor * scale = (mul_కణుపు->src[0] == unary_కణుపు) ? mul_కణుపు->src[1] : mul_కణుపు->src[0];

    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(scale != unary_కణుపు);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_కణుపు->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_కణుపు));
    // a and dst are indexed flat
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_కణుపు));
    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి స్వయంచాలక dispatch_op = [&](స్వయంచాలక op) {
        కూడు_unary_mul_sycl((స్థిరరాశి దశాంశం *) a->దత్తాంశం, (స్థిరరాశి దశాంశం *) bias->దత్తాంశం, (స్థిరరాశి దశాంశం *) scale->దత్తాంశం,
                           (దశాంశం *) mul_కణుపు->దత్తాంశం, ggml_nelements(mul_కణుపు), mul_కణుపు->ne[0], main_ప్రవాహం, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_కణుపు), dispatch_op);
}

__dpct_inline__ దశాంశం ggml_sycl_op_swiglu_oai_single(దశాంశం x, దశాంశం g, దశాంశం alpha = 1.702f, దశాంశం హద్దు = 7.0f) {
    x = sycl::fmin(x, హద్దు);
    g = sycl::fmax(sycl::fmin(g, హద్దు), -హద్దు);

    దశాంశం బహిర్గతం_glu = x / (1.0f + sycl::native::ఘాతాంకం(-x * alpha));
    బహిర్గతం_glu = బహిర్గతం_glu * (1.0f + g);
    తిరిగి బహిర్గతం_glu;
}

మూస <రకం_పేరు T>
స్థిర శూన్య swiglu_oai_kernel(స్థిరరాశి T * x, స్థిరరాశి T * g, T * dst, స్థిరరాశి int64_t k,
                              స్థిరరాశి int64_t n, స్థిరరాశి int64_t o0, స్థిరరాశి int64_t o1,
                              దశాంశం alpha, దశాంశం హద్దు, sycl::nd_అంశం<3> అంశం_ct1) {
    స్థిరరాశి int64_t i = int64_t(అంశం_ct1.get_local_range(2)) * అంశం_ct1.get_సమూహం(2) + అంశం_ct1.get_local_id(2);

    ఒకవేళ (i >= k) {
        తిరిగి;
    }

    స్థిరరాశి int64_t j0 = (i / n) * o0 + (i % n);
    స్థిరరాశి int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    దశాంశం xi = x[j0];
    దశాంశం gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, హద్దు);
}

మూస <రకం_పేరు T>
స్థిర శూన్య swiglu_oai_sycl(స్థిరరాశి T *       x,
                            స్థిరరాశి T *       g,
                            T *             dst,
                            స్థిరరాశి int64_t   k,
                            స్థిరరాశి int64_t   n,
                            స్థిరరాశి int64_t   o0,
                            స్థిరరాశి int64_t   o1,
                            స్థిరరాశి దశాంశం     alpha,
                            స్థిరరాశి దశాంశం     హద్దు,
                            dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t అంకె_ఖండాలు = (k + SYCL_GLU_ఖండం_కొలత - 1) / SYCL_GLU_ఖండం_కొలత;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, అంకె_ఖండాలు) * sycl::range<3>(1, 1, SYCL_GLU_ఖండం_కొలత),
                                           sycl::range<3>(1, 1, SYCL_GLU_ఖండం_కొలత)),
                         [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, హద్దు, అంశం_ct1);
                         });
}

స్థిర శూన్య ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    శూన్య * src0_d = src0->దత్తాంశం;
    శూన్య * src1_d = src1 ? src1->దత్తాంశం : src0->దత్తాంశం;
    స్థిరరాశి int64_t src0_o = src0->nb[1];
    స్థిరరాశి int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    శూన్య * dst_d = dst->దత్తాంశం;
    స్థిరరాశి int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::వరుస_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_కొలత(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_కొలత(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);
    స్థిరరాశి దశాంశం alpha = ggml_get_op_params_f32(dst, 2);
    స్థిరరాశి దశాంశం హద్దు = ggml_get_op_params_f32(dst, 3);

    దశాంశం * src0_p = (దశాంశం *) src0_d;
    దశాంశం * src1_p = (దశాంశం *) src1_d;

    ఒకవేళ (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (దశాంశం *)dst_d, ggml_nelements(dst), nc, src0_o / పరిమాణం(దశాంశం), src1_o / పరిమాణం(దశాంశం), alpha, హద్దు, stream);
}

మూస <రకం_పేరు T>
స్థిర శూన్య swiglu_clamp_kernel(స్థిరరాశి T *        gate,
                                స్థిరరాశి T *        పైకి,
                                T *              dst,
                                స్థిరరాశి int64_t    k,
                                స్థిరరాశి int64_t    n,
                                స్థిరరాశి int64_t    o0,
                                స్థిరరాశి int64_t    o1,
                                దశాంశం            హద్దు,
                                sycl::nd_అంశం<3> అంశం_ct1) {
    స్థిరరాశి int64_t i = int64_t(అంశం_ct1.get_local_range(2)) * అంశం_ct1.get_సమూహం(2) + అంశం_ct1.get_local_id(2);

    ఒకవేళ (i >= k) {
        తిరిగి;
    }

    స్థిరరాశి int64_t j0 = (i / n) * o0 + (i % n);
    స్థిరరాశి int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    స్థిరరాశి దశాంశం gate_విలువ = sycl::fmin((దశాంశం) gate[j0], హద్దు);
    స్థిరరాశి దశాంశం పైకి_విలువ   = sycl::fmax(sycl::fmin((దశాంశం) పైకి[j1], హద్దు), -హద్దు);
    dst[i]                 = (T) (gate_విలువ / (1.0f + sycl::native::ఘాతాంకం(-gate_విలువ)) * పైకి_విలువ);
}

మూస <రకం_పేరు T>
స్థిర శూన్య swiglu_clamp_sycl(స్థిరరాశి T *       gate,
                              స్థిరరాశి T *       పైకి,
                              T *             dst,
                              స్థిరరాశి int64_t   k,
                              స్థిరరాశి int64_t   n,
                              స్థిరరాశి int64_t   o0,
                              స్థిరరాశి int64_t   o1,
                              దశాంశం           హద్దు,
                              dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t అంకె_ఖండాలు = (k + SYCL_GLU_ఖండం_కొలత - 1) / SYCL_GLU_ఖండం_కొలత;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, అంకె_ఖండాలు) * sycl::range<3>(1, 1, SYCL_GLU_ఖండం_కొలత),
                                           sycl::range<3>(1, 1, SYCL_GLU_ఖండం_కొలత)),
                         [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             swiglu_clamp_kernel(gate, పైకి, dst, k, n, o0, o1, హద్దు, అంశం_ct1);
                         });
}

స్థిర శూన్య ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0   = dst->src[0];
    స్థిరరాశి ggml_tensor * src1   = dst->src[1];
    శూన్య *              src0_d = src0->దత్తాంశం;
    శూన్య *              src1_d = src1 ? src1->దత్తాంశం : src0->దత్తాంశం;
    స్థిరరాశి int64_t       src0_o = src0->nb[1];
    స్థిరరాశి int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    శూన్య *              dst_d  = dst->దత్తాంశం;
    స్థిరరాశి int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::వరుస_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_కొలత(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_కొలత(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);
    స్థిరరాశి దశాంశం   హద్దు   = ggml_get_op_params_f32(dst, 3);

    ఒకవేళ (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / పరిమాణం(sycl::half),
                          src1_o / పరిమాణం(sycl::half), హద్దు, stream);
    } లేకపోతే {
        దశాంశం * src0_p = (దశాంశం *) src0_d;
        దశాంశం * src1_p = (దశాంశం *) src1_d;

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (దశాంశం *) dst_d, ggml_nelements(dst), nc, src0_o / పరిమాణం(దశాంశం),
                          src1_o / పరిమాణం(దశాంశం), హద్దు, stream);
    }
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_gelu_erf(x);
    });
}

స్థిర అంతఃవరుస శూన్య ggml_sycl_op_geglu_శీఘ్ర(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](స్వయంచాలక x) {
        తిరిగి op_gelu_శీఘ్ర(x);
    });
}


శూన్య ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

శూన్య ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

శూన్య ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

శూన్య ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

శూన్య ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

శూన్య ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

శూన్య ggml_sycl_gelu_శీఘ్ర(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_శీఘ్ర(ctx, dst);
}

శూన్య ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

శూన్య ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

శూన్య ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

శూన్య ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

శూన్య ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

శూన్య ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

శూన్య ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

శూన్య ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_expm1(ctx, dst);
}

శూన్య ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

శూన్య ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

శూన్య ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

శూన్య ggml_sycl_అడుగు(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_అడుగు(ctx, dst);
}

శూన్య ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

శూన్య ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

శూన్య ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

శూన్య ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_xielu(ctx, dst);
}

శూన్య ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

శూన్య ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

శూన్య ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

శూన్య ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

శూన్య ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

శూన్య ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

శూన్య ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

శూన్య ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_clamp(ctx, dst);
}

శూన్య ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

శూన్య ggml_sycl_geglu_శీఘ్ర(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_శీఘ్ర(ctx, dst);
}

శూన్య ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

శూన్య ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

శూన్య ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

శూన్య ggml_sycl_విడత(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_విడత(ctx, dst);
}

శూన్య ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
