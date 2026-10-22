#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/presets.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "element_wise.hpp"

#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_GLOBAL_ID_LOOP(K, ಅಂಶ_2) \
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ i = ಅಂಶ_2.get_global_id(0); i < (size_t)K; i += ಅಂಶ_2.get_global_range(0))

#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_LOCAL_ID_CALC(ಅಂಶ_2, IDX) \
    (ಅಂಶ_2.get_local_range(IDX) * ಅಂಶ_2.get_ಗುಂಪು(IDX) + ಅಂಶ_2.get_local_id(IDX))

ಸ್ಥಿರ ಶೂನ್ಯ acc_f32(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * x, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * y, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ int64_t ne,
    ಸ್ಥಿರಾಂಕ int64_t ne0, ಸ್ಥಿರಾಂಕ int64_t ne1, ಸ್ಥಿರಾಂಕ int64_t ne2,
    ಸ್ಥಿರಾಂಕ int64_t nb00, ಸ್ಥಿರಾಂಕ int64_t nb01, ಸ್ಥಿರಾಂಕ int64_t nb02, ಸ್ಥಿರಾಂಕ int64_t nb03,
    ಸ್ಥಿರಾಂಕ int64_t ne10, ಸ್ಥಿರಾಂಕ int64_t ne11, ಸ್ಥಿರಾಂಕ int64_t ne12, ಸ್ಥಿರಾಂಕ int64_t ne13,
    ಸ್ಥಿರಾಂಕ int64_t nb10, ಸ್ಥಿರಾಂಕ int64_t nb11, ಸ್ಥಿರಾಂಕ int64_t nb12, ಸ್ಥಿರಾಂಕ int64_t nb13,
    ಸ್ಥಿರಾಂಕ int64_t s11, ಸ್ಥಿರಾಂಕ int64_t s12, ಸ್ಥಿರಾಂಕ int64_t s13, ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ int64_t i = SYCL_LOCAL_ID_CALC(ಅಂಶ_ct1, 2);

    ಆದರೆ (i >= ne) {
        ಹಿಂತಿರುಗಿ;
    }

    int64_t src1_idx = i - ಸ್ಥಾನಾಂತರ;

    int64_t tmp = src1_idx;
    ಸ್ಥಿರಾಂಕ int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    ಸ್ಥಿರಾಂಕ int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    ಸ್ಥಿರಾಂಕ int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    ಸ್ಥಿರಾಂಕ int64_t i10 = tmp;

    int64_t tmp_dst = i;
    ಸ್ಥಿರಾಂಕ int64_t i3 = tmp_dst / (ne2*ne1*ne0);
    tmp_dst -= i3 * (ne2*ne1*ne0);
    ಸ್ಥಿರಾಂಕ int64_t i2 = tmp_dst / (ne1*ne0);
    tmp_dst -= i2 * (ne1*ne0);
    ಸ್ಥಿರಾಂಕ int64_t i1 = tmp_dst / ne0;
    tmp_dst -= i1 * ne0;
    ಸ್ಥಿರಾಂಕ int64_t i0 = tmp_dst;

    ದಶಮಾಂಶ val = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    ಆದರೆ (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_sgn(T x) {
    ಹಿಂತಿರುಗಿ x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_abs(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::fabs(x);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_expm1(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<ದಶಮಾಂಶ>(x))
        );
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::expm1(x);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_elu(T x) {
    ಹಿಂತಿರುಗಿ (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_erf(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<ದಶಮಾಂಶ>(x))
        );
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::erf(x);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_gelu_erf(T x) {
    ಸ್ಥಿರಾಂಕ T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    ಹಿಂತಿರುಗಿ static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_gelu_ಶೀಘ್ರ(T x) {
    ಸ್ಥಿರಾಂಕ T GELU_ಶೀಘ್ರ_COEF_LOCAL = static_cast<T>(-1.702f);
    ಹಿಂತಿರುಗಿ x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_ಶೀಘ್ರ_COEF_LOCAL * x)));
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_relu(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::fmax(x, static_cast<T>(0));
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_sigmoid(T x) {
    ಹಿಂತಿರುಗಿ static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_sqrt(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::ವರ್ಗಮೂಲ(x);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::ವರ್ಗಮೂಲ(x);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_sin(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::ಜ್ಯಾ(x);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::ಜ್ಯಾ(x);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_cos(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::ಕೋಜ್ಯಾ(x);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::ಕೋಜ್ಯಾ(x);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_hardsigmoid(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_hardswish(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_log(T x) {
    ಆದರೆ (x <= static_cast<T>(0)) {
        ಹಿಂತಿರುಗಿ neg_infinity<T>();
    }
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::ಲಘುಗಣಕ(x);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::ಲಘುಗಣಕ(x);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_softplus(T x) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xf = (ದಶಮಾಂಶ) x;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ax = op_abs(xf);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m  = sycl::fmax(xf, 0.0f);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y  = m + sycl::log1p(sycl::ಘಾತಾಂಕ(-ax));
    ಹಿಂತಿರುಗಿ (T) y;
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_neg(T x) {
    ಹಿಂತಿರುಗಿ -x;
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_ಹೆಜ್ಜೆ(T x) {
    ಹಿಂತಿರುಗಿ (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_leaky_relu(T x, ದಶಮಾಂಶ negative_ಇಳಿಜಾರು) {
    T neg_ಇಳಿಜಾರು_T = static_cast<T>(negative_ಇಳಿಜಾರು);
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_ಇಳಿಜಾರು_T;

    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_ಇಳಿಜಾರು_T;
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_xielu(T x, ದಶಮಾಂಶ alpha_n, ದಶಮಾಂಶ alpha_p, ದಶಮಾಂಶ beta, ದಶಮಾಂಶ eps) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xi        = static_cast<ದಶಮಾಂಶ>(x);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ gate_pos  = (xi > 0.0f);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y_pos     = alpha_p * xi * xi + beta * xi;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min_v_eps = sycl::fmin(xi, eps);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಹೊರಹೋಗು       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    ಹಿಂತಿರುಗಿ static_cast<T>(ಹೊರಹೋಗು);
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_sqr(T x) {
    ಹಿಂತಿರುಗಿ x * x;
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_clamp(T x, ದಶಮಾಂಶ min_val, ದಶಮಾಂಶ max_val) {
    ಹಿಂತಿರುಗಿ x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_floor(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::floor(x);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::floor(x);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_ceil(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::ceil(x);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::ceil(x);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_ಸುತ್ತು(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<ದಶಮಾಂಶ>(x))
        );
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::round(x);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ __dpct_inline__ T op_trunc(T x) {
    ಆದರೆ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಹಿಂತಿರುಗಿ sycl::ext::oneapi::experimental::trunc(x);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ sycl::trunc(x);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_flat_kernel(ಸ್ಥಿರಾಂಕ T * x, T * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = func(x[i]);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_generic_kernel(
        ಸ್ಥಿರಾಂಕ T * x,
        T * dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k,
        ಸ್ಥಿರಾಂಕ sycl::uint3 ne0_fd, ಸ್ಥಿರಾಂಕ sycl::uint3 ne1_fd, ಸ್ಥಿರಾಂಕ sycl::uint3 ne2_fd,
        ಸ್ಥಿರಾಂಕ size_t nb0,  ಸ್ಥಿರಾಂಕ size_t nb1,  ಸ್ಥಿರಾಂಕ size_t nb2,  ಸ್ಥಿರಾಂಕ size_t nb3,
        ಸ್ಥಿರಾಂಕ size_t nbd0, ಸ್ಥಿರಾಂಕ size_t nbd1, ಸ್ಥಿರಾಂಕ size_t nbd2, ಸ್ಥಿರಾಂಕ size_t nbd3,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        ಸ್ಥಿರಾಂಕ uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        ಸ್ಥಿರಾಂಕ uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        ಸ್ಥಿರಾಂಕ uint32_t i2 = dm.y();
        ಸ್ಥಿರಾಂಕ uint32_t i3 = dm.x();

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) x;
        ಅಕ್ಷರ       * dst_ತಳ = (ಅಕ್ಷರ *) dst;

        ಸ್ಥಿರಾಂಕ T * srcp = (ಸ್ಥಿರಾಂಕ T *)(src_ತಳ + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_ತಳ + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_sqrt_kernel(ಸ್ಥಿರಾಂಕ T * x, T * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_sin_kernel(ಸ್ಥಿರಾಂಕ T * x, T * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_cos_kernel(ಸ್ಥಿರಾಂಕ T * x, T * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_log_kernel(ಸ್ಥಿರಾಂಕ T * x, T * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = op_log(x[i]);
    }
}


ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_leaky_relu_kernel(ಸ್ಥಿರಾಂಕ T * x, T * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, ದಶಮಾಂಶ negative_ಇಳಿಜಾರು, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_ಇಳಿಜಾರು);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_xielu_kernel(ಸ್ಥಿರಾಂಕ T * x, T * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, ದಶಮಾಂಶ alpha_n, ದಶಮಾಂಶ alpha_p, ದಶಮಾಂಶ beta, ದಶಮಾಂಶ eps, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_sqr_kernel(ಸ್ಥಿರಾಂಕ T * x, T * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_clamp_kernel(ಸ್ಥಿರಾಂಕ T * x, T * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1, ದಶಮಾಂಶ min_val, ದಶಮಾಂಶ max_val) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_ceil_kernel(ಸ್ಥಿರಾಂಕ T * x, T * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ clamp(ಸ್ಥಿರಾಂಕ T * x, T * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ max, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k,
                      ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_gated_op_flat_kernel(ಸ್ಥಿರಾಂಕ T * x, ಸ್ಥಿರಾಂಕ T * g, T * dst, ಸ್ಥಿರಾಂಕ uint64_t k, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_gated_op_generic_kernel(
        ಸ್ಥಿರಾಂಕ T * x,
        ಸ್ಥಿರಾಂಕ T * g,
        T * dst,
        ಸ್ಥಿರಾಂಕ uint64_t k,
        ಸ್ಥಿರಾಂಕ sycl::uint3 n_fd,
        ಸ್ಥಿರಾಂಕ uint64_t o0,
        ಸ್ಥಿರಾಂಕ uint64_t o1,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        ಸ್ಥಿರಾಂಕ sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ಸ್ಥಿರಾಂಕ int64_t j0 = rc.x() * o0 + rc.y();
        ಸ್ಥಿರಾಂಕ int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_mul_flat_kernel(ಸ್ಥಿರಾಂಕ T * x, ಸ್ಥಿರಾಂಕ T * g, T * dst, ಸ್ಥಿರಾಂಕ int64_t k, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = (T) (op((ದಶಮಾಂಶ) x[i]) * (ದಶಮಾಂಶ) g[i]);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_mul_strided_kernel(ಸ್ಥಿರಾಂಕ T * x, ಸ್ಥಿರಾಂಕ T * g, T * dst, ಸ್ಥಿರಾಂಕ int64_t k, ಸ್ಥಿರಾಂಕ sycl::uint3 n_fd, ಸ್ಥಿರಾಂಕ int64_t o0, ಸ್ಥಿರಾಂಕ int64_t o1, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        ಸ್ಥಿರಾಂಕ sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ಸ್ಥಿರಾಂಕ int64_t j0 = rc.x() * o0 + rc.y();
        ಸ್ಥಿರಾಂಕ int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((ದಶಮಾಂಶ) x[j0]) * (ದಶಮಾಂಶ) g[j1]);
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_mul_sycl(ಸ್ಥಿರಾಂಕ T * x, ಸ್ಥಿರಾಂಕ T * g, T * dst, ಸ್ಥಿರಾಂಕ int64_t k, ಸ್ಥಿರಾಂಕ int64_t n, ಸ್ಥಿರಾಂಕ int64_t o0, ಸ್ಥಿರಾಂಕ int64_t o1, ಸರತಿ_ptr main_ಹರಿವು, F op) {
    ಸ್ಥಿರಾಂಕ size_t            ಅಂಕೆ_ಖಂಡಗಳು = ceil_div((size_t) k, (size_t) SYCL_GLU_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::nd_range<1> range(ಅಂಕೆ_ಖಂಡಗಳು * sycl::range<1>(SYCL_GLU_ಖಂಡ_ಗಾತ್ರ), sycl::range<1>(SYCL_GLU_ಖಂಡ_ಗಾತ್ರ));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    ಆದರೆ (o0 == n && o1 == n) {
        main_ಹರಿವು->parallel_for(range, [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            unary_mul_flat_kernel(x, g, dst, k, ಅಂಶ_ct1, op);
        });
        ಹಿಂತಿರುಗಿ;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ಸ್ಥಿರಾಂಕ sycl::uint3 n_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) n);
    main_ಹರಿವು->parallel_for(range, [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, ಅಂಶ_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
ಮಾದರಿ<ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ ಕೂಡಿಸು_unary_mul_flat_kernel(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * bias, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * scale, ದಶಮಾಂಶ * dst,
                                      ಸ್ಥಿರಾಂಕ int64_t k, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ ಕೂಡಿಸು_unary_mul_bcast_kernel(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * bias, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * scale, ದಶಮಾಂಶ * dst,
                                       ಸ್ಥಿರಾಂಕ int64_t k, ಸ್ಥಿರಾಂಕ sycl::uint3 ne0_fd, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        ಸ್ಥಿರಾಂಕ uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ ಕೂಡಿಸು_unary_mul_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * bias, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * scale, ದಶಮಾಂಶ * dst,
                               ಸ್ಥಿರಾಂಕ int64_t k, ಸ್ಥಿರಾಂಕ int64_t ne0, ಸರತಿ_ptr main_ಹರಿವು, F op) {
    ಸ್ಥಿರಾಂಕ size_t            ಅಂಕೆ_ಖಂಡಗಳು = ceil_div((size_t) k, (size_t) SYCL_GLU_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::nd_range<1> range(ಅಂಕೆ_ಖಂಡಗಳು * sycl::range<1>(SYCL_GLU_ಖಂಡ_ಗಾತ್ರ), sycl::range<1>(SYCL_GLU_ಖಂಡ_ಗಾತ್ರ));

    ಆದರೆ (k == ne0) {
        main_ಹರಿವು->parallel_for(range, [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            ಕೂಡಿಸು_unary_mul_flat_kernel(a, bias, scale, dst, k, ಅಂಶ_ct1, op);
        });
        ಹಿಂತಿರುಗಿ;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ಸ್ಥಿರಾಂಕ sycl::uint3 ne0_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) ne0);
    main_ಹರಿವು->parallel_for(range, [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
        ಕೂಡಿಸು_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, ಅಂಶ_ct1, op);
    });
}

ನಾಮಾಕಾಶ ggml_sycl_detail {
ಸ್ಥಿರ ಶೂನ್ಯ acc_f32_sycl(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *x, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *y, ದಶಮಾಂಶ *dst,
                         ಸ್ಥಿರಾಂಕ int64_t n_elements,
                         ಸ್ಥಿರಾಂಕ int64_t ne0, ಸ್ಥಿರಾಂಕ int64_t ne1, ಸ್ಥಿರಾಂಕ int64_t ne2,
                         ಸ್ಥಿರಾಂಕ int64_t nb00, ಸ್ಥಿರಾಂಕ int64_t nb01, ಸ್ಥಿರಾಂಕ int64_t nb02, ಸ್ಥಿರಾಂಕ int64_t nb03,
                         ಸ್ಥಿರಾಂಕ int64_t ne10, ಸ್ಥಿರಾಂಕ int64_t ne11, ಸ್ಥಿರಾಂಕ int64_t ne12, ಸ್ಥಿರಾಂಕ int64_t ne13,
                         ಸ್ಥಿರಾಂಕ int64_t nb10, ಸ್ಥಿರಾಂಕ int64_t nb11, ಸ್ಥಿರಾಂಕ int64_t nb12, ಸ್ಥಿರಾಂಕ int64_t nb13,
                         ಸ್ಥಿರಾಂಕ int64_t s1, ಸ್ಥಿರಾಂಕ int64_t s2, ಸ್ಥಿರಾಂಕ int64_t s3,
                         ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ, ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = (n_elements + SYCL_ACC_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_ACC_ಖಂಡ_ಗಾತ್ರ;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<3>(1, 1, SYCL_ACC_ಖಂಡ_ಗಾತ್ರ),
                                           sycl::range<3>(1, 1, SYCL_ACC_ಖಂಡ_ಗಾತ್ರ)),
                         [=](sycl::nd_ಅಂಶ<3> /*item_ct1*/) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                            acc_f32(x, y, dst, n_elements,
                                ne0, ne1, ne2,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                s1, s2, s3, ಸ್ಥಾನಾಂತರ);
                         });
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ arange_kernel(T * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, T start, T ಹೆಜ್ಜೆ,
                         ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &ಅಂಶ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ಅಂಶ_ct1) {
        dst[i] = start + static_cast<T>(i) * ಹೆಜ್ಜೆ;
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು KernelInvoker, ಬಗೆ_ಹೆಸರು... ವಾದಗಳು>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, ವಾದಗಳು&&... ವಾದಗಳು_2) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಆಯ್ಕೆ (dst->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_pts = cast_ದತ್ತಾಂಶ<sycl::half>(dst);
                kernel_invoker(ದತ್ತಾಂಶ_pts.src, ದತ್ತಾಂಶ_pts.dst, (ಪೂರ್ಣಾಂಕ)ggml_nelements(dst->src[0]), main_ಹರಿವು, std::forward<ವಾದಗಳು>(ವಾದಗಳು_2)...);
                ನಿಲ್ಲಿಸು;
            }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            {
                ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_pts = cast_ದತ್ತಾಂಶ<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(ದತ್ತಾಂಶ_pts.src, ದತ್ತಾಂಶ_pts.dst, (ಪೂರ್ಣಾಂಕ)ggml_nelements(dst->src[0]), main_ಹರಿವು, std::forward<ವಾದಗಳು>(ವಾದಗಳು_2)...);
                ನಿಲ್ಲಿಸು;
            }
#ಆದರೆ_ಅಂತ್ಯ
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_pts = cast_ದತ್ತಾಂಶ<ದಶಮಾಂಶ>(dst);
                kernel_invoker(ದತ್ತಾಂಶ_pts.src, ದತ್ತಾಂಶ_pts.dst, (ಪೂರ್ಣಾಂಕ)ggml_nelements(dst->src[0]), main_ಹರಿವು, std::forward<ವಾದಗಳು>(ವಾದಗಳು_2)...);
                ನಿಲ್ಲಿಸು;
            }
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು KernelInvoker, ಬಗೆ_ಹೆಸರು... ವಾದಗಳು>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, ವಾದಗಳು&&... ವಾದಗಳು_2) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // The fused GLU kernels index with 32-bit fastdiv, which is exact only for indices below
    // 2^31. A dst that large is ~8 GB at f32, and the grid sizing already narrows to 32 bits,
    // so assert the bound rather than carry a second code path for it.
    GGML_ASSERT(ggml_nelements(dst) < ((int64_t) 1 << 31));
    ಸ್ಥಿರಾಂಕ int32_t swapped = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[1];
    ಶೂನ್ಯ * src0_d = src0->ದತ್ತಾಂಶ;
    ಶೂನ್ಯ * src1_d = src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ಶೂನ್ಯ * dst_d = dst->ದತ್ತಾಂಶ;
    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_ಗಾತ್ರ(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    ಆಯ್ಕೆ (dst->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    ಆದರೆ (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / ಗಾತ್ರ(sycl::half),
                               src1_o / ಗಾತ್ರ(sycl::half),
                               main_ಹರಿವು,
                               std::forward<ವಾದಗಳು>(ವಾದಗಳು_2)...);
                ನಿಲ್ಲಿಸು;
            }
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            {
                ದಶಮಾಂಶ * src0_p = (ದಶಮಾಂಶ *) src0_d;
                ದಶಮಾಂಶ * src1_p = (ದಶಮಾಂಶ *) src1_d;

                    ಆದರೆ (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (ದಶಮಾಂಶ *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / ಗಾತ್ರ(ದಶಮಾಂಶ),
                               src1_o / ಗಾತ್ರ(ದಶಮಾಂಶ),
                               main_ಹರಿವು,
                               std::forward<ವಾದಗಳು>(ವಾದಗಳು_2)...);
                ನಿಲ್ಲಿಸು;
            }
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    ಸ್ಥಿರಾಂಕ int64_t ne0  = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1  = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2  = dst->ne[2];

    ಸ್ಥಿರಾಂಕ size_t  nb0  = src0->nb[0];
    ಸ್ಥಿರಾಂಕ size_t  nb1  = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t  nb2  = src0->nb[2];
    ಸ್ಥಿರಾಂಕ size_t  nb3  = src0->nb[3];

    ಸ್ಥಿರಾಂಕ size_t  nbd0 = dst->nb[0];
    ಸ್ಥಿರಾಂಕ size_t  nbd1 = dst->nb[1];
    ಸ್ಥಿರಾಂಕ size_t  nbd2 = dst->nb[2];
    ಸ್ಥಿರಾಂಕ size_t  nbd3 = dst->nb[3];

    // Hot unary ops (FFN/GDN silu, sigmoid, ...) run on contiguous tensors;
    // skip the strided index math entirely for them.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ* src, ಸ್ವಯಂಚಾಲಿತ* dst_ptr, ಪೂರ್ಣಾಂಕ k_elements, ಸರತಿ_ptr stream) {

            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = ceil_div(k_elements, 256);

            ಆದರೆ (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, ಅಂಶ_ct1, func);
                    });
            } ಇಲ್ಲದಿದ್ದರೆ {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                ಸ್ಥಿರಾಂಕ sycl::uint3 ne0_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) ne0);
                ಸ್ಥಿರಾಂಕ sycl::uint3 ne1_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) ne1);
                ಸ್ಥಿರಾಂಕ sycl::uint3 ne2_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        unary_op_generic_kernel(
                            src, dst_ptr, k_elements,
                            ne0_fd, ne1_fd, ne2_fd,
                            nb0, nb1, nb2, nb3,
                            nbd0, nbd1, nbd2, nbd3,
                            ಅಂಶ_ct1,
                            func
                        );
                    });
            }
        });
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * x_ptr, ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * g_ptr, ಸ್ವಯಂಚಾಲಿತ * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, ಸರತಿ_ptr main_ಹರಿವು) {

            ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಖಂಡಗಳು = (uint32_t) ceil_div(k, SYCL_GLU_ಖಂಡ_ಗಾತ್ರ);
            ಸ್ಥಿರಾಂಕ sycl::nd_range<1> launch_range(ಅಂಕೆ_ಖಂಡಗಳು * sycl::range<1>(SYCL_GLU_ಖಂಡ_ಗಾತ್ರ),
                                                 sycl::range<1>(SYCL_GLU_ಖಂಡ_ಗಾತ್ರ));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            ಆದರೆ (o0 == n && o1 == n) {
                main_ಹರಿವು->parallel_for(launch_range,
                    [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, ಅಂಶ_ct1, func);
                    });
            } ಇಲ್ಲದಿದ್ದರೆ {
                // launch-invariant divisor, and only this path needs it
                ಸ್ಥಿರಾಂಕ sycl::uint3 n_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) n);
                main_ಹರಿವು->parallel_for(launch_range,
                    [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, ಅಂಶ_ct1, func);
                    });
            }
        });
}


ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    ದಶಮಾಂಶ start, stop, ಹೆಜ್ಜೆ;
    ಸ್ಮರಣೆ_ನಕಲು(&start, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&stop, (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&ಹೆಜ್ಜೆ, (ದಶಮಾಂಶ *) dst->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));
    dpct::ಸರತಿ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ದಶಮಾಂಶ * dst_ptr = (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k = (ಪೂರ್ಣಾಂಕ)ggml_nelements(dst);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = ceil_div(k, SYCL_ARANGE_ಖಂಡ_ಗಾತ್ರ);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<1>(SYCL_ARANGE_ಖಂಡ_ಗಾತ್ರ),
                          sycl::range<1>(SYCL_ARANGE_ಖಂಡ_ಗಾತ್ರ)),
        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            arange_kernel(dst_ptr, k, start, ಹೆಜ್ಜೆ, ಅಂಶ_ct1);
        });
}

} // namespace ggml_sycl_detail



ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_sgn(x);
    });
}


ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_abs(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_elu(x);
    });
}
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_silu(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_gelu(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_gelu_ಶೀಘ್ರ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_gelu_ಶೀಘ್ರ(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_gelu_erf(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_tanh(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_relu(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_hardsigmoid(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_hardswish(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_exp(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_expm1(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ* src, ಸ್ವಯಂಚಾಲಿತ* dst_ptr, ಪೂರ್ಣಾಂಕ k_elements, ಸರತಿ_ptr stream) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = ceil_div(k_elements, SYCL_EXP_ಖಂಡ_ಗಾತ್ರ); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<1>(SYCL_EXP_ಖಂಡ_ಗಾತ್ರ),
                                  sycl::range<1>(SYCL_EXP_ಖಂಡ_ಗಾತ್ರ)),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, ಅಂಶ_ct1);
                });
        });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_softplus(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_neg(x);
    });
}


ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_ಹೆಜ್ಜೆ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_ಹೆಜ್ಜೆ(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_sigmoid(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ* src, ಸ್ವಯಂಚಾಲಿತ* dst_ptr, ಪೂರ್ಣಾಂಕ k_elements, ಸರತಿ_ptr stream) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = ceil_div(k_elements, SYCL_SQRT_ಖಂಡ_ಗಾತ್ರ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<1>(SYCL_SQRT_ಖಂಡ_ಗಾತ್ರ),
                                  sycl::range<1>(SYCL_SQRT_ಖಂಡ_ಗಾತ್ರ)),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, ಅಂಶ_ct1);
                });
        });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ* src, ಸ್ವಯಂಚಾಲಿತ* dst_ptr, ಪೂರ್ಣಾಂಕ k_elements, ಸರತಿ_ptr stream) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = ceil_div(k_elements, SYCL_SIN_ಖಂಡ_ಗಾತ್ರ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<1>(SYCL_SIN_ಖಂಡ_ಗಾತ್ರ),
                                  sycl::range<1>(SYCL_SIN_ಖಂಡ_ಗಾತ್ರ)),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, ಅಂಶ_ct1);
                });
        });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ* src, ಸ್ವಯಂಚಾಲಿತ* dst_ptr, ಪೂರ್ಣಾಂಕ k_elements, ಸರತಿ_ptr stream) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = ceil_div(k_elements, SYCL_SIN_ಖಂಡ_ಗಾತ್ರ); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<1>(SYCL_SIN_ಖಂಡ_ಗಾತ್ರ),
                                  sycl::range<1>(SYCL_SIN_ಖಂಡ_ಗಾತ್ರ)),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, ಅಂಶ_ct1);
                });
        });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ದಶಮಾಂಶ negative_ಇಳಿಜಾರು;
    ಸ್ಮರಣೆ_ನಕಲು(&negative_ಇಳಿಜಾರು, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ* src, ಸ್ವಯಂಚಾಲಿತ* dst_ptr, ಪೂರ್ಣಾಂಕ k_elements, ಸರತಿ_ptr stream, ದಶಮಾಂಶ ಇಳಿಜಾರು) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = ceil_div(k_elements, SYCL_RELU_ಖಂಡ_ಗಾತ್ರ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<1>(SYCL_RELU_ಖಂಡ_ಗಾತ್ರ),
                                  sycl::range<1>(SYCL_RELU_ಖಂಡ_ಗಾತ್ರ)),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, ಇಳಿಜಾರು, ಅಂಶ_ct1);
                });
        }, negative_ಇಳಿಜಾರು);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ* src, ಸ್ವಯಂಚಾಲಿತ* dst_ptr, ಪೂರ್ಣಾಂಕ k_elements, ಸರತಿ_ptr stream) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = ceil_div(k_elements, SYCL_SQR_ಖಂಡ_ಗಾತ್ರ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<1>(SYCL_SQR_ಖಂಡ_ಗಾತ್ರ),
                                  sycl::range<1>(SYCL_SQR_ಖಂಡ_ಗಾತ್ರ)),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, ಅಂಶ_ct1);
                });
        });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ದಶಮಾಂಶ min_val;
    ದಶಮಾಂಶ max_val;
    ಸ್ಮರಣೆ_ನಕಲು(&min_val, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_val, (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ* src, ಸ್ವಯಂಚಾಲಿತ* dst_ptr, ಪೂರ್ಣಾಂಕ k_elements, ಸರತಿ_ptr stream, ದಶಮಾಂಶ min_arg, ದಶಮಾಂಶ max_arg) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = ceil_div(k_elements, SYCL_CLAMP_ಖಂಡ_ಗಾತ್ರ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<1>(SYCL_CLAMP_ಖಂಡ_ಗಾತ್ರ),
                                  sycl::range<1>(SYCL_CLAMP_ಖಂಡ_ಗಾತ್ರ)),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, ಅಂಶ_ct1);
                });
        }, min_val, max_val);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha_n = ggml_get_op_params_f32(dst, 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha_p = ggml_get_op_params_f32(dst, 2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta    = ggml_get_op_params_f32(dst, 3);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ* src, ಸ್ವಯಂಚಾಲಿತ* dst_ptr, ಪೂರ್ಣಾಂಕ k_elements, ಸರತಿ_ptr stream, ದಶಮಾಂಶ alpha_n_arg, ದಶಮಾಂಶ alpha_p_arg, ದಶಮಾಂಶ beta_arg, ದಶಮಾಂಶ eps_arg) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = ceil_div(k_elements, SYCL_RELU_ಖಂಡ_ಗಾತ್ರ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<1>(SYCL_RELU_ಖಂಡ_ಗಾತ್ರ),
                                  sycl::range<1>(SYCL_RELU_ಖಂಡ_ಗಾತ್ರ)),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, ಅಂಶ_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_floor(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_ceil(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_ಸುತ್ತು(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_ಸುತ್ತು(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_trunc(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ  * src0_d = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ  *) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ  * src1_d = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ  *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ       * dst_d  = (ದಶಮಾಂಶ       *)  dst->ದತ್ತಾಂಶ;

    dpct::ಸರತಿ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_ಗಾತ್ರ(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ಸ್ಥಿರಾಂಕ int64_t s1     = (int64_t) ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[0] / (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t s2     = (int64_t) ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[1] / (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t s3     = (int64_t) ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[2] / (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ = (int64_t) ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[3] / (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, ಸ್ಥಾನಾಂತರ, stream);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_gelu(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_relu(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
ಮಾದರಿ<ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
    ಆಯ್ಕೆ (uop) {
        ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:     launch([](ದಶಮಾಂಶ v) { ಹಿಂತಿರುಗಿ op_silu(v); });     ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:  launch([](ದಶಮಾಂಶ v) { ಹಿಂತಿರುಗಿ op_sigmoid(v); });  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS: launch([](ದಶಮಾಂಶ v) { ಹಿಂತಿರುಗಿ op_softplus(v); }); ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
    }
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
ಶೂನ್ಯ ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_ಗಂಟು, ggml_tensor * mul_ಗಂಟು) {
    scope_op_debug_print scope_dbg_print(__func__, mul_ಗಂಟು, /*num_src=*/2);

    ಸ್ಥಿರಾಂಕ ggml_tensor * x = unary_ಗಂಟು->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * g = (mul_ಗಂಟು->src[0] == unary_ಗಂಟು) ? mul_ಗಂಟು->src[1] : mul_ಗಂಟು->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_ಗಂಟು);
    GGML_ASSERT(x->type == g->type && x->type == mul_ಗಂಟು->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_ಗಂಟು));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_ಗಂಟು));

    ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ int64_t k = ggml_nelements(mul_ಗಂಟು);
    ಸ್ಥಿರಾಂಕ int64_t n = mul_ಗಂಟು->ne[0];

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ dispatch_type = [&](ಸ್ವಯಂಚಾಲಿತ op) {
        ಆಯ್ಕೆ (mul_ಗಂಟು->type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:
                unary_mul_sycl((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) x->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) g->ದತ್ತಾಂಶ, (ದಶಮಾಂಶ *) mul_ಗಂಟು->ದತ್ತಾಂಶ,
                               k, n, x->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ), g->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ), main_ಹರಿವು, op);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                unary_mul_sycl((ಸ್ಥಿರಾಂಕ sycl::half *) x->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ sycl::half *) g->ದತ್ತಾಂಶ, (sycl::half *) mul_ಗಂಟು->ದತ್ತಾಂಶ,
                               k, n, x->nb[1] / ಗಾತ್ರ(sycl::half), g->nb[1] / ಗಾತ್ರ(sycl::half), main_ಹರಿವು, op);
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_ಗಂಟು->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_ಗಂಟು), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
ಶೂನ್ಯ ggml_sycl_op_ಕೂಡಿಸು_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * ಕೂಡಿಸು_ಗಂಟು,
                                      ggml_tensor * unary_ಗಂಟು, ggml_tensor * mul_ಗಂಟು) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_ಗಂಟು, /*num_src=*/2);

    ಸ್ಥಿರಾಂಕ ggml_tensor * a     = ಕೂಡಿಸು_ಗಂಟು->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * bias  = ಕೂಡಿಸು_ಗಂಟು->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * scale = (mul_ಗಂಟು->src[0] == unary_ಗಂಟು) ? mul_ಗಂಟು->src[1] : mul_ಗಂಟು->src[0];

    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(scale != unary_ಗಂಟು);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_ಗಂಟು->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_ಗಂಟು));
    // a and dst are indexed flat
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_ಗಂಟು));
    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ dispatch_op = [&](ಸ್ವಯಂಚಾಲಿತ op) {
        ಕೂಡಿಸು_unary_mul_sycl((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) a->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) bias->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) scale->ದತ್ತಾಂಶ,
                           (ದಶಮಾಂಶ *) mul_ಗಂಟು->ದತ್ತಾಂಶ, ggml_nelements(mul_ಗಂಟು), mul_ಗಂಟು->ne[0], main_ಹರಿವು, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_ಗಂಟು), dispatch_op);
}

__dpct_inline__ ದಶಮಾಂಶ ggml_sycl_op_swiglu_oai_single(ದಶಮಾಂಶ x, ದಶಮಾಂಶ g, ದಶಮಾಂಶ alpha = 1.702f, ದಶಮಾಂಶ ಎಲ್ಲೆ = 7.0f) {
    x = sycl::fmin(x, ಎಲ್ಲೆ);
    g = sycl::fmax(sycl::fmin(g, ಎಲ್ಲೆ), -ಎಲ್ಲೆ);

    ದಶಮಾಂಶ ಹೊರಹೋಗು_glu = x / (1.0f + sycl::native::ಘಾತಾಂಕ(-x * alpha));
    ಹೊರಹೋಗು_glu = ಹೊರಹೋಗು_glu * (1.0f + g);
    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು_glu;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ swiglu_oai_kernel(ಸ್ಥಿರಾಂಕ T * x, ಸ್ಥಿರಾಂಕ T * g, T * dst, ಸ್ಥಿರಾಂಕ int64_t k,
                              ಸ್ಥಿರಾಂಕ int64_t n, ಸ್ಥಿರಾಂಕ int64_t o0, ಸ್ಥಿರಾಂಕ int64_t o1,
                              ದಶಮಾಂಶ alpha, ದಶಮಾಂಶ ಎಲ್ಲೆ, sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ int64_t i = int64_t(ಅಂಶ_ct1.get_local_range(2)) * ಅಂಶ_ct1.get_ಗುಂಪು(2) + ಅಂಶ_ct1.get_local_id(2);

    ಆದರೆ (i >= k) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t j0 = (i / n) * o0 + (i % n);
    ಸ್ಥಿರಾಂಕ int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ದಶಮಾಂಶ xi = x[j0];
    ದಶಮಾಂಶ gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, ಎಲ್ಲೆ);
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ swiglu_oai_sycl(ಸ್ಥಿರಾಂಕ T *       x,
                            ಸ್ಥಿರಾಂಕ T *       g,
                            T *             dst,
                            ಸ್ಥಿರಾಂಕ int64_t   k,
                            ಸ್ಥಿರಾಂಕ int64_t   n,
                            ಸ್ಥಿರಾಂಕ int64_t   o0,
                            ಸ್ಥಿರಾಂಕ int64_t   o1,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     alpha,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     ಎಲ್ಲೆ,
                            dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (k + SYCL_GLU_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_GLU_ಖಂಡ_ಗಾತ್ರ;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<3>(1, 1, SYCL_GLU_ಖಂಡ_ಗಾತ್ರ),
                                           sycl::range<3>(1, 1, SYCL_GLU_ಖಂಡ_ಗಾತ್ರ)),
                         [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, ಎಲ್ಲೆ, ಅಂಶ_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಶೂನ್ಯ * src0_d = src0->ದತ್ತಾಂಶ;
    ಶೂನ್ಯ * src1_d = src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ಶೂನ್ಯ * dst_d = dst->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::ಸರತಿ_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_ಗಾತ್ರ(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_ಗಾತ್ರ(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha = ggml_get_op_params_f32(dst, 2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಎಲ್ಲೆ = ggml_get_op_params_f32(dst, 3);

    ದಶಮಾಂಶ * src0_p = (ದಶಮಾಂಶ *) src0_d;
    ದಶಮಾಂಶ * src1_p = (ದಶಮಾಂಶ *) src1_d;

    ಆದರೆ (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (ದಶಮಾಂಶ *)dst_d, ggml_nelements(dst), nc, src0_o / ಗಾತ್ರ(ದಶಮಾಂಶ), src1_o / ಗಾತ್ರ(ದಶಮಾಂಶ), alpha, ಎಲ್ಲೆ, stream);
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ swiglu_clamp_kernel(ಸ್ಥಿರಾಂಕ T *        gate,
                                ಸ್ಥಿರಾಂಕ T *        ಮೇಲೆ,
                                T *              dst,
                                ಸ್ಥಿರಾಂಕ int64_t    k,
                                ಸ್ಥಿರಾಂಕ int64_t    n,
                                ಸ್ಥಿರಾಂಕ int64_t    o0,
                                ಸ್ಥಿರಾಂಕ int64_t    o1,
                                ದಶಮಾಂಶ            ಎಲ್ಲೆ,
                                sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ int64_t i = int64_t(ಅಂಶ_ct1.get_local_range(2)) * ಅಂಶ_ct1.get_ಗುಂಪು(2) + ಅಂಶ_ct1.get_local_id(2);

    ಆದರೆ (i >= k) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t j0 = (i / n) * o0 + (i % n);
    ಸ್ಥಿರಾಂಕ int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ gate_ಮೌಲ್ಯ = sycl::fmin((ದಶಮಾಂಶ) gate[j0], ಎಲ್ಲೆ);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಮೇಲೆ_ಮೌಲ್ಯ   = sycl::fmax(sycl::fmin((ದಶಮಾಂಶ) ಮೇಲೆ[j1], ಎಲ್ಲೆ), -ಎಲ್ಲೆ);
    dst[i]                 = (T) (gate_ಮೌಲ್ಯ / (1.0f + sycl::native::ಘಾತಾಂಕ(-gate_ಮೌಲ್ಯ)) * ಮೇಲೆ_ಮೌಲ್ಯ);
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ swiglu_clamp_sycl(ಸ್ಥಿರಾಂಕ T *       gate,
                              ಸ್ಥಿರಾಂಕ T *       ಮೇಲೆ,
                              T *             dst,
                              ಸ್ಥಿರಾಂಕ int64_t   k,
                              ಸ್ಥಿರಾಂಕ int64_t   n,
                              ಸ್ಥಿರಾಂಕ int64_t   o0,
                              ಸ್ಥಿರಾಂಕ int64_t   o1,
                              ದಶಮಾಂಶ           ಎಲ್ಲೆ,
                              dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (k + SYCL_GLU_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_GLU_ಖಂಡ_ಗಾತ್ರ;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_ಖಂಡಗಳು) * sycl::range<3>(1, 1, SYCL_GLU_ಖಂಡ_ಗಾತ್ರ),
                                           sycl::range<3>(1, 1, SYCL_GLU_ಖಂಡ_ಗಾತ್ರ)),
                         [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             swiglu_clamp_kernel(gate, ಮೇಲೆ, dst, k, n, o0, o1, ಎಲ್ಲೆ, ಅಂಶ_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0   = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1   = dst->src[1];
    ಶೂನ್ಯ *              src0_d = src0->ದತ್ತಾಂಶ;
    ಶೂನ್ಯ *              src1_d = src1 ? src1->ದತ್ತಾಂಶ : src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t       src0_o = src0->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ಶೂನ್ಯ *              dst_d  = dst->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::ಸರತಿ_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_ಗಾತ್ರ(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    ಆದರೆ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_ಗಾತ್ರ(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   ಎಲ್ಲೆ   = ggml_get_op_params_f32(dst, 3);

    ಆದರೆ (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / ಗಾತ್ರ(sycl::half),
                          src1_o / ಗಾತ್ರ(sycl::half), ಎಲ್ಲೆ, stream);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ದಶಮಾಂಶ * src0_p = (ದಶಮಾಂಶ *) src0_d;
        ದಶಮಾಂಶ * src1_p = (ದಶಮಾಂಶ *) src1_d;

        ಆದರೆ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (ದಶಮಾಂಶ *) dst_d, ggml_nelements(dst), nc, src0_o / ಗಾತ್ರ(ದಶಮಾಂಶ),
                          src1_o / ಗಾತ್ರ(ದಶಮಾಂಶ), ಎಲ್ಲೆ, stream);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_gelu_erf(x);
    });
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_geglu_ಶೀಘ್ರ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](ಸ್ವಯಂಚಾಲಿತ x) {
        ಹಿಂತಿರುಗಿ op_gelu_ಶೀಘ್ರ(x);
    });
}


ಶೂನ್ಯ ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_gelu_ಶೀಘ್ರ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_ಶೀಘ್ರ(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_expm1(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_ಹೆಜ್ಜೆ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ಹೆಜ್ಜೆ(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_xielu(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_clamp(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_geglu_ಶೀಘ್ರ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_ಶೀಘ್ರ(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_ಸುತ್ತು(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ಸುತ್ತು(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
