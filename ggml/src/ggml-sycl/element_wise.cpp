#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/presets.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "element_wise.hpp"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_GLOBAL_ID_LOOP(K, ਵਸਤੂ_2) \
    ਲਈ (ਸਵੈਚਾਲਿਤ i = ਵਸਤੂ_2.get_global_id(0); i < (size_t)K; i += ਵਸਤੂ_2.get_global_range(0))

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_LOCAL_ID_CALC(ਵਸਤੂ_2, IDX) \
    (ਵਸਤੂ_2.get_local_range(IDX) * ਵਸਤੂ_2.get_ਸਮੂਹ(IDX) + ਵਸਤੂ_2.get_local_id(IDX))

ਸਥਿਰ ਖਾਲੀ acc_f32(ਅਚਲ ਅੱਖਰ * x, ਅਚਲ ਅੱਖਰ * y, ਦਸ਼ਮਲਵ * dst, ਅਚਲ int64_t ne,
    ਅਚਲ int64_t ne0, ਅਚਲ int64_t ne1, ਅਚਲ int64_t ne2,
    ਅਚਲ int64_t nb00, ਅਚਲ int64_t nb01, ਅਚਲ int64_t nb02, ਅਚਲ int64_t nb03,
    ਅਚਲ int64_t ne10, ਅਚਲ int64_t ne11, ਅਚਲ int64_t ne12, ਅਚਲ int64_t ne13,
    ਅਚਲ int64_t nb10, ਅਚਲ int64_t nb11, ਅਚਲ int64_t nb12, ਅਚਲ int64_t nb13,
    ਅਚਲ int64_t s11, ਅਚਲ int64_t s12, ਅਚਲ int64_t s13, ਅਚਲ int64_t ਵਿੱਥ) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ int64_t i = SYCL_LOCAL_ID_CALC(ਵਸਤੂ_ct1, 2);

    ਜੇ (i >= ne) {
        ਵਾਪਸ;
    }

    int64_t src1_idx = i - ਵਿੱਥ;

    int64_t tmp = src1_idx;
    ਅਚਲ int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    ਅਚਲ int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    ਅਚਲ int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    ਅਚਲ int64_t i10 = tmp;

    int64_t tmp_dst = i;
    ਅਚਲ int64_t i3 = tmp_dst / (ne2*ne1*ne0);
    tmp_dst -= i3 * (ne2*ne1*ne0);
    ਅਚਲ int64_t i2 = tmp_dst / (ne1*ne0);
    tmp_dst -= i2 * (ne1*ne0);
    ਅਚਲ int64_t i1 = tmp_dst / ne0;
    tmp_dst -= i1 * ne0;
    ਅਚਲ int64_t i0 = tmp_dst;

    ਦਸ਼ਮਲਵ val = *(ਅਚਲ ਦਸ਼ਮਲਵ *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    ਜੇ (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(ਅਚਲ ਦਸ਼ਮਲਵ *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_sgn(T x) {
    ਵਾਪਸ x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_abs(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::fabs(x);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_expm1(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<ਦਸ਼ਮਲਵ>(x))
        );
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::expm1(x);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_elu(T x) {
    ਵਾਪਸ (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_erf(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<ਦਸ਼ਮਲਵ>(x))
        );
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::erf(x);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_gelu_erf(T x) {
    ਅਚਲ T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    ਵਾਪਸ static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_gelu_ਤੇਜ਼(T x) {
    ਅਚਲ T GELU_ਤੇਜ਼_COEF_LOCAL = static_cast<T>(-1.702f);
    ਵਾਪਸ x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_ਤੇਜ਼_COEF_LOCAL * x)));
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_relu(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::fmax(x, static_cast<T>(0));
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_sigmoid(T x) {
    ਵਾਪਸ static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_sqrt(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::experimental::ਵਰਗਮੂਲ(x);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::ਵਰਗਮੂਲ(x);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_sin(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::experimental::ਜੀਵਾ(x);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::ਜੀਵਾ(x);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_cos(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::experimental::ਕੋਜੀਵਾ(x);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::ਕੋਜੀਵਾ(x);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_hardsigmoid(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_hardswish(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_log(T x) {
    ਜੇ (x <= static_cast<T>(0)) {
        ਵਾਪਸ neg_infinity<T>();
    }
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::experimental::ਲਘੂਗਣਕ(x);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::ਲਘੂਗਣਕ(x);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_softplus(T x) {
    ਅਚਲ ਦਸ਼ਮਲਵ xf = (ਦਸ਼ਮਲਵ) x;
    ਅਚਲ ਦਸ਼ਮਲਵ ax = op_abs(xf);
    ਅਚਲ ਦਸ਼ਮਲਵ m  = sycl::fmax(xf, 0.0f);
    ਅਚਲ ਦਸ਼ਮਲਵ y  = m + sycl::log1p(sycl::ਘਾਤਾਂਕ(-ax));
    ਵਾਪਸ (T) y;
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_neg(T x) {
    ਵਾਪਸ -x;
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_ਕਦਮ(T x) {
    ਵਾਪਸ (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_leaky_relu(T x, ਦਸ਼ਮਲਵ negative_ਢਲਾਨ) {
    T neg_ਢਲਾਨ_T = static_cast<T>(negative_ਢਲਾਨ);
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_ਢਲਾਨ_T;

    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_ਢਲਾਨ_T;
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_xielu(T x, ਦਸ਼ਮਲਵ alpha_n, ਦਸ਼ਮਲਵ alpha_p, ਦਸ਼ਮਲਵ beta, ਦਸ਼ਮਲਵ eps) {
    ਅਚਲ ਦਸ਼ਮਲਵ xi        = static_cast<ਦਸ਼ਮਲਵ>(x);
    ਅਚਲ ਦਸ਼ਮਲਵ gate_pos  = (xi > 0.0f);
    ਅਚਲ ਦਸ਼ਮਲਵ y_pos     = alpha_p * xi * xi + beta * xi;
    ਅਚਲ ਦਸ਼ਮਲਵ min_v_eps = sycl::fmin(xi, eps);
    ਅਚਲ ਦਸ਼ਮਲਵ y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    ਅਚਲ ਦਸ਼ਮਲਵ ਬਾਹਰਲਾ       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    ਵਾਪਸ static_cast<T>(ਬਾਹਰਲਾ);
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_sqr(T x) {
    ਵਾਪਸ x * x;
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_clamp(T x, ਦਸ਼ਮਲਵ min_val, ਦਸ਼ਮਲਵ max_val) {
    ਵਾਪਸ x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_floor(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::experimental::floor(x);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::floor(x);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_ceil(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::experimental::ceil(x);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::ceil(x);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_ਗੇੜਾ(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<ਦਸ਼ਮਲਵ>(x))
        );
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::round(x);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ __dpct_inline__ T op_trunc(T x) {
    ਜੇ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ਵਾਪਸ sycl::ext::oneapi::experimental::trunc(x);
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ sycl::trunc(x);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ unary_op_flat_kernel(ਅਚਲ T * x, T * dst, ਅਚਲ ਪੂਰਨ k, ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = func(x[i]);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ unary_op_generic_kernel(
        ਅਚਲ T * x,
        T * dst,
        ਅਚਲ ਪੂਰਨ k,
        ਅਚਲ sycl::uint3 ne0_fd, ਅਚਲ sycl::uint3 ne1_fd, ਅਚਲ sycl::uint3 ne2_fd,
        ਅਚਲ size_t nb0,  ਅਚਲ size_t nb1,  ਅਚਲ size_t nb2,  ਅਚਲ size_t nb3,
        ਅਚਲ size_t nbd0, ਅਚਲ size_t nbd1, ਅਚਲ size_t nbd2, ਅਚਲ size_t nbd3,
        ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        ਅਚਲ uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        ਅਚਲ uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        ਅਚਲ uint32_t i2 = dm.y();
        ਅਚਲ uint32_t i3 = dm.x();

        ਅਚਲ ਅੱਖਰ * src_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) x;
        ਅੱਖਰ       * dst_ਅਧਾਰ = (ਅੱਖਰ *) dst;

        ਅਚਲ T * srcp = (ਅਚਲ T *)(src_ਅਧਾਰ + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_ਅਧਾਰ + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ unary_op_sqrt_kernel(ਅਚਲ T * x, T * dst, ਅਚਲ ਪੂਰਨ k, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ unary_op_sin_kernel(ਅਚਲ T * x, T * dst, ਅਚਲ ਪੂਰਨ k, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ unary_op_cos_kernel(ਅਚਲ T * x, T * dst, ਅਚਲ ਪੂਰਨ k, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ unary_op_log_kernel(ਅਚਲ T * x, T * dst, ਅਚਲ ਪੂਰਨ k, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = op_log(x[i]);
    }
}


ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ unary_op_leaky_relu_kernel(ਅਚਲ T * x, T * dst, ਅਚਲ ਪੂਰਨ k, ਦਸ਼ਮਲਵ negative_ਢਲਾਨ, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_ਢਲਾਨ);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ unary_op_xielu_kernel(ਅਚਲ T * x, T * dst, ਅਚਲ ਪੂਰਨ k, ਦਸ਼ਮਲਵ alpha_n, ਦਸ਼ਮਲਵ alpha_p, ਦਸ਼ਮਲਵ beta, ਦਸ਼ਮਲਵ eps, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ unary_op_sqr_kernel(ਅਚਲ T * x, T * dst, ਅਚਲ ਪੂਰਨ k, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ unary_op_clamp_kernel(ਅਚਲ T * x, T * dst, ਅਚਲ ਪੂਰਨ k, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1, ਦਸ਼ਮਲਵ min_val, ਦਸ਼ਮਲਵ max_val) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ unary_op_ceil_kernel(ਅਚਲ T * x, T * dst, ਅਚਲ ਪੂਰਨ k, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ clamp(ਅਚਲ T * x, T * dst, ਅਚਲ ਦਸ਼ਮਲਵ min, ਅਚਲ ਦਸ਼ਮਲਵ max, ਅਚਲ ਪੂਰਨ k,
                      ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ unary_gated_op_flat_kernel(ਅਚਲ T * x, ਅਚਲ T * g, T * dst, ਅਚਲ uint64_t k, ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ unary_gated_op_generic_kernel(
        ਅਚਲ T * x,
        ਅਚਲ T * g,
        T * dst,
        ਅਚਲ uint64_t k,
        ਅਚਲ sycl::uint3 n_fd,
        ਅਚਲ uint64_t o0,
        ਅਚਲ uint64_t o1,
        ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        ਅਚਲ sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ਅਚਲ int64_t j0 = rc.x() * o0 + rc.y();
        ਅਚਲ int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ unary_mul_flat_kernel(ਅਚਲ T * x, ਅਚਲ T * g, T * dst, ਅਚਲ int64_t k, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = (T) (op((ਦਸ਼ਮਲਵ) x[i]) * (ਦਸ਼ਮਲਵ) g[i]);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ unary_mul_strided_kernel(ਅਚਲ T * x, ਅਚਲ T * g, T * dst, ਅਚਲ int64_t k, ਅਚਲ sycl::uint3 n_fd, ਅਚਲ int64_t o0, ਅਚਲ int64_t o1, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        ਅਚਲ sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ਅਚਲ int64_t j0 = rc.x() * o0 + rc.y();
        ਅਚਲ int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((ਦਸ਼ਮਲਵ) x[j0]) * (ਦਸ਼ਮਲਵ) g[j1]);
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ unary_mul_sycl(ਅਚਲ T * x, ਅਚਲ T * g, T * dst, ਅਚਲ int64_t k, ਅਚਲ int64_t n, ਅਚਲ int64_t o0, ਅਚਲ int64_t o1, ਕਤਾਰ_ptr main_ਧਾਰਾ, F op) {
    ਅਚਲ size_t            ਅੰਕ_ਖੰਡਾਂ = ceil_div((size_t) k, (size_t) SYCL_GLU_ਖੰਡ_ਮਾਪ);
    ਅਚਲ sycl::nd_range<1> range(ਅੰਕ_ਖੰਡਾਂ * sycl::range<1>(SYCL_GLU_ਖੰਡ_ਮਾਪ), sycl::range<1>(SYCL_GLU_ਖੰਡ_ਮਾਪ));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    ਜੇ (o0 == n && o1 == n) {
        main_ਧਾਰਾ->parallel_for(range, [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            unary_mul_flat_kernel(x, g, dst, k, ਵਸਤੂ_ct1, op);
        });
        ਵਾਪਸ;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ਅਚਲ sycl::uint3 n_fd = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) n);
    main_ਧਾਰਾ->parallel_for(range, [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, ਵਸਤੂ_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ ਜੋੜੋ_unary_mul_flat_kernel(ਅਚਲ ਦਸ਼ਮਲਵ * a, ਅਚਲ ਦਸ਼ਮਲਵ * bias, ਅਚਲ ਦਸ਼ਮਲਵ * scale, ਦਸ਼ਮਲਵ * dst,
                                      ਅਚਲ int64_t k, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ ਜੋੜੋ_unary_mul_bcast_kernel(ਅਚਲ ਦਸ਼ਮਲਵ * a, ਅਚਲ ਦਸ਼ਮਲਵ * bias, ਅਚਲ ਦਸ਼ਮਲਵ * scale, ਦਸ਼ਮਲਵ * dst,
                                       ਅਚਲ int64_t k, ਅਚਲ sycl::uint3 ne0_fd, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        ਅਚਲ uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ ਜੋੜੋ_unary_mul_sycl(ਅਚਲ ਦਸ਼ਮਲਵ * a, ਅਚਲ ਦਸ਼ਮਲਵ * bias, ਅਚਲ ਦਸ਼ਮਲਵ * scale, ਦਸ਼ਮਲਵ * dst,
                               ਅਚਲ int64_t k, ਅਚਲ int64_t ne0, ਕਤਾਰ_ptr main_ਧਾਰਾ, F op) {
    ਅਚਲ size_t            ਅੰਕ_ਖੰਡਾਂ = ceil_div((size_t) k, (size_t) SYCL_GLU_ਖੰਡ_ਮਾਪ);
    ਅਚਲ sycl::nd_range<1> range(ਅੰਕ_ਖੰਡਾਂ * sycl::range<1>(SYCL_GLU_ਖੰਡ_ਮਾਪ), sycl::range<1>(SYCL_GLU_ਖੰਡ_ਮਾਪ));

    ਜੇ (k == ne0) {
        main_ਧਾਰਾ->parallel_for(range, [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            ਜੋੜੋ_unary_mul_flat_kernel(a, bias, scale, dst, k, ਵਸਤੂ_ct1, op);
        });
        ਵਾਪਸ;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ਅਚਲ sycl::uint3 ne0_fd = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) ne0);
    main_ਧਾਰਾ->parallel_for(range, [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
        ਜੋੜੋ_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, ਵਸਤੂ_ct1, op);
    });
}

ਨਾਮਾਕਾਸ਼ ggml_sycl_detail {
ਸਥਿਰ ਖਾਲੀ acc_f32_sycl(ਅਚਲ ਅੱਖਰ *x, ਅਚਲ ਅੱਖਰ *y, ਦਸ਼ਮਲਵ *dst,
                         ਅਚਲ int64_t n_elements,
                         ਅਚਲ int64_t ne0, ਅਚਲ int64_t ne1, ਅਚਲ int64_t ne2,
                         ਅਚਲ int64_t nb00, ਅਚਲ int64_t nb01, ਅਚਲ int64_t nb02, ਅਚਲ int64_t nb03,
                         ਅਚਲ int64_t ne10, ਅਚਲ int64_t ne11, ਅਚਲ int64_t ne12, ਅਚਲ int64_t ne13,
                         ਅਚਲ int64_t nb10, ਅਚਲ int64_t nb11, ਅਚਲ int64_t nb12, ਅਚਲ int64_t nb13,
                         ਅਚਲ int64_t s1, ਅਚਲ int64_t s2, ਅਚਲ int64_t s3,
                         ਅਚਲ int64_t ਵਿੱਥ, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (n_elements + SYCL_ACC_ਖੰਡ_ਮਾਪ - 1) / SYCL_ACC_ਖੰਡ_ਮਾਪ;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_ACC_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_ACC_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> /*item_ct1*/) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                            acc_f32(x, y, dst, n_elements,
                                ne0, ne1, ne2,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                s1, s2, s3, ਵਿੱਥ);
                         });
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ arange_kernel(T * dst, ਅਚਲ ਪੂਰਨ k, T start, T ਕਦਮ,
                         ਅਚਲ sycl::nd_ਵਸਤੂ<1> &ਵਸਤੂ_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, ਵਸਤੂ_ct1) {
        dst[i] = start + static_cast<T>(i) * ਕਦਮ;
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ KernelInvoker, ਕਿਸਮ_ਨਾਂ... ਦਲੀਲਾਂ>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, ਦਲੀਲਾਂ&&... ਦਲੀਲਾਂ_2) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਚੋਣ (dst->type) {
        ਹਾਲਤ GGML_TYPE_F16:
            {
                ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_pts = cast_ਅੰਕੜੇ<sycl::half>(dst);
                kernel_invoker(ਅੰਕੜੇ_pts.src, ਅੰਕੜੇ_pts.dst, (ਪੂਰਨ)ggml_nelements(dst->src[0]), main_ਧਾਰਾ, std::forward<ਦਲੀਲਾਂ>(ਦਲੀਲਾਂ_2)...);
                ਰੋਕੋ;
            }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
        ਹਾਲਤ GGML_TYPE_BF16:
            {
                ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_pts = cast_ਅੰਕੜੇ<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(ਅੰਕੜੇ_pts.src, ਅੰਕੜੇ_pts.dst, (ਪੂਰਨ)ggml_nelements(dst->src[0]), main_ਧਾਰਾ, std::forward<ਦਲੀਲਾਂ>(ਦਲੀਲਾਂ_2)...);
                ਰੋਕੋ;
            }
#ਜੇ_ਅੰਤ
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_pts = cast_ਅੰਕੜੇ<ਦਸ਼ਮਲਵ>(dst);
                kernel_invoker(ਅੰਕੜੇ_pts.src, ਅੰਕੜੇ_pts.dst, (ਪੂਰਨ)ggml_nelements(dst->src[0]), main_ਧਾਰਾ, std::forward<ਦਲੀਲਾਂ>(ਦਲੀਲਾਂ_2)...);
                ਰੋਕੋ;
            }
        ਮੂਲ:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ KernelInvoker, ਕਿਸਮ_ਨਾਂ... ਦਲੀਲਾਂ>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, ਦਲੀਲਾਂ&&... ਦਲੀਲਾਂ_2) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅਚਲ int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // The fused GLU kernels index with 32-bit fastdiv, which is exact only for indices below
    // 2^31. A dst that large is ~8 GB at f32, and the grid sizing already narrows to 32 bits,
    // so assert the bound rather than carry a second code path for it.
    GGML_ASSERT(ggml_nelements(dst) < ((int64_t) 1 << 31));
    ਅਚਲ int32_t swapped = ((ਅਚਲ int32_t *) dst->op_params)[1];
    ਖਾਲੀ * src0_d = src0->ਅੰਕੜੇ;
    ਖਾਲੀ * src1_d = src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ;
    ਅਚਲ int64_t src0_o = src0->nb[1];
    ਅਚਲ int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ਖਾਲੀ * dst_d = dst->ਅੰਕੜੇ;
    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_ਮਾਪ(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    ਚੋਣ (dst->type) {
        ਹਾਲਤ GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    ਜੇ (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / ਆਕਾਰ(sycl::half),
                               src1_o / ਆਕਾਰ(sycl::half),
                               main_ਧਾਰਾ,
                               std::forward<ਦਲੀਲਾਂ>(ਦਲੀਲਾਂ_2)...);
                ਰੋਕੋ;
            }
        ਹਾਲਤ GGML_TYPE_F32:
            {
                ਦਸ਼ਮਲਵ * src0_p = (ਦਸ਼ਮਲਵ *) src0_d;
                ਦਸ਼ਮਲਵ * src1_p = (ਦਸ਼ਮਲਵ *) src1_d;

                    ਜੇ (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (ਦਸ਼ਮਲਵ *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                               src1_o / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                               main_ਧਾਰਾ,
                               std::forward<ਦਲੀਲਾਂ>(ਦਲੀਲਾਂ_2)...);
                ਰੋਕੋ;
            }
        ਮੂਲ:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    ਅਚਲ int64_t ne0  = dst->ne[0];
    ਅਚਲ int64_t ne1  = dst->ne[1];
    ਅਚਲ int64_t ne2  = dst->ne[2];

    ਅਚਲ size_t  nb0  = src0->nb[0];
    ਅਚਲ size_t  nb1  = src0->nb[1];
    ਅਚਲ size_t  nb2  = src0->nb[2];
    ਅਚਲ size_t  nb3  = src0->nb[3];

    ਅਚਲ size_t  nbd0 = dst->nb[0];
    ਅਚਲ size_t  nbd1 = dst->nb[1];
    ਅਚਲ size_t  nbd2 = dst->nb[2];
    ਅਚਲ size_t  nbd3 = dst->nb[3];

    // Hot unary ops (FFN/GDN silu, sigmoid, ...) run on contiguous tensors;
    // skip the strided index math entirely for them.
    ਅਚਲ ਤਾਰਕਿਕ contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](ਅਚਲ ਸਵੈਚਾਲਿਤ* src, ਸਵੈਚਾਲਿਤ* dst_ptr, ਪੂਰਨ k_elements, ਕਤਾਰ_ptr stream) {

            ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(k_elements, 256);

            ਜੇ (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, ਵਸਤੂ_ct1, func);
                    });
            } ਨਹੀਂਤਾਂ {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                ਅਚਲ sycl::uint3 ne0_fd = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) ne0);
                ਅਚਲ sycl::uint3 ne1_fd = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) ne1);
                ਅਚਲ sycl::uint3 ne2_fd = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        unary_op_generic_kernel(
                            src, dst_ptr, k_elements,
                            ne0_fd, ne1_fd, ne2_fd,
                            nb0, nb1, nb2, nb3,
                            nbd0, nbd1, nbd2, nbd3,
                            ਵਸਤੂ_ct1,
                            func
                        );
                    });
            }
        });
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](ਅਚਲ ਸਵੈਚਾਲਿਤ * x_ptr, ਅਚਲ ਸਵੈਚਾਲਿਤ * g_ptr, ਸਵੈਚਾਲਿਤ * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, ਕਤਾਰ_ptr main_ਧਾਰਾ) {

            ਅਚਲ uint32_t ਅੰਕ_ਖੰਡਾਂ = (uint32_t) ceil_div(k, SYCL_GLU_ਖੰਡ_ਮਾਪ);
            ਅਚਲ sycl::nd_range<1> launch_range(ਅੰਕ_ਖੰਡਾਂ * sycl::range<1>(SYCL_GLU_ਖੰਡ_ਮਾਪ),
                                                 sycl::range<1>(SYCL_GLU_ਖੰਡ_ਮਾਪ));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            ਜੇ (o0 == n && o1 == n) {
                main_ਧਾਰਾ->parallel_for(launch_range,
                    [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, ਵਸਤੂ_ct1, func);
                    });
            } ਨਹੀਂਤਾਂ {
                // launch-invariant divisor, and only this path needs it
                ਅਚਲ sycl::uint3 n_fd = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) n);
                main_ਧਾਰਾ->parallel_for(launch_range,
                    [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, ਵਸਤੂ_ct1, func);
                    });
            }
        });
}


ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    ਦਸ਼ਮਲਵ start, stop, ਕਦਮ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&start, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&stop, (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਕਦਮ, (ਦਸ਼ਮਲਵ *) dst->op_params + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    dpct::ਕਤਾਰ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਦਸ਼ਮਲਵ * dst_ptr = (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ;
    ਅਚਲ ਪੂਰਨ k = (ਪੂਰਨ)ggml_nelements(dst);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(k, SYCL_ARANGE_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ) * sycl::range<1>(SYCL_ARANGE_ਖੰਡ_ਮਾਪ),
                          sycl::range<1>(SYCL_ARANGE_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            arange_kernel(dst_ptr, k, start, ਕਦਮ, ਵਸਤੂ_ct1);
        });
}

} // namespace ggml_sycl_detail



ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_sgn(x);
    });
}


ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_abs(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_elu(x);
    });
}
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_silu(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_gelu(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_gelu_ਤੇਜ਼(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_gelu_ਤੇਜ਼(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_gelu_erf(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_tanh(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_relu(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_hardsigmoid(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_hardswish(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_exp(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_expm1(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ਅਚਲ ਸਵੈਚਾਲਿਤ* src, ਸਵੈਚਾਲਿਤ* dst_ptr, ਪੂਰਨ k_elements, ਕਤਾਰ_ptr stream) {
            ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(k_elements, SYCL_EXP_ਖੰਡ_ਮਾਪ); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ) * sycl::range<1>(SYCL_EXP_ਖੰਡ_ਮਾਪ),
                                  sycl::range<1>(SYCL_EXP_ਖੰਡ_ਮਾਪ)),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, ਵਸਤੂ_ct1);
                });
        });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_softplus(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_neg(x);
    });
}


ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_ਕਦਮ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_ਕਦਮ(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_sigmoid(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ਅਚਲ ਸਵੈਚਾਲਿਤ* src, ਸਵੈਚਾਲਿਤ* dst_ptr, ਪੂਰਨ k_elements, ਕਤਾਰ_ptr stream) {
            ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(k_elements, SYCL_SQRT_ਖੰਡ_ਮਾਪ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ) * sycl::range<1>(SYCL_SQRT_ਖੰਡ_ਮਾਪ),
                                  sycl::range<1>(SYCL_SQRT_ਖੰਡ_ਮਾਪ)),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, ਵਸਤੂ_ct1);
                });
        });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ਅਚਲ ਸਵੈਚਾਲਿਤ* src, ਸਵੈਚਾਲਿਤ* dst_ptr, ਪੂਰਨ k_elements, ਕਤਾਰ_ptr stream) {
            ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(k_elements, SYCL_SIN_ਖੰਡ_ਮਾਪ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ) * sycl::range<1>(SYCL_SIN_ਖੰਡ_ਮਾਪ),
                                  sycl::range<1>(SYCL_SIN_ਖੰਡ_ਮਾਪ)),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, ਵਸਤੂ_ct1);
                });
        });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ਅਚਲ ਸਵੈਚਾਲਿਤ* src, ਸਵੈਚਾਲਿਤ* dst_ptr, ਪੂਰਨ k_elements, ਕਤਾਰ_ptr stream) {
            ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(k_elements, SYCL_SIN_ਖੰਡ_ਮਾਪ); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ) * sycl::range<1>(SYCL_SIN_ਖੰਡ_ਮਾਪ),
                                  sycl::range<1>(SYCL_SIN_ਖੰਡ_ਮਾਪ)),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, ਵਸਤੂ_ct1);
                });
        });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਦਸ਼ਮਲਵ negative_ਢਲਾਨ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&negative_ਢਲਾਨ, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ਅਚਲ ਸਵੈਚਾਲਿਤ* src, ਸਵੈਚਾਲਿਤ* dst_ptr, ਪੂਰਨ k_elements, ਕਤਾਰ_ptr stream, ਦਸ਼ਮਲਵ ਢਲਾਨ) {
            ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(k_elements, SYCL_RELU_ਖੰਡ_ਮਾਪ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ) * sycl::range<1>(SYCL_RELU_ਖੰਡ_ਮਾਪ),
                                  sycl::range<1>(SYCL_RELU_ਖੰਡ_ਮਾਪ)),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, ਢਲਾਨ, ਵਸਤੂ_ct1);
                });
        }, negative_ਢਲਾਨ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ਅਚਲ ਸਵੈਚਾਲਿਤ* src, ਸਵੈਚਾਲਿਤ* dst_ptr, ਪੂਰਨ k_elements, ਕਤਾਰ_ptr stream) {
            ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(k_elements, SYCL_SQR_ਖੰਡ_ਮਾਪ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ) * sycl::range<1>(SYCL_SQR_ਖੰਡ_ਮਾਪ),
                                  sycl::range<1>(SYCL_SQR_ਖੰਡ_ਮਾਪ)),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, ਵਸਤੂ_ct1);
                });
        });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਦਸ਼ਮਲਵ min_val;
    ਦਸ਼ਮਲਵ max_val;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&min_val, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_val, (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ਅਚਲ ਸਵੈਚਾਲਿਤ* src, ਸਵੈਚਾਲਿਤ* dst_ptr, ਪੂਰਨ k_elements, ਕਤਾਰ_ptr stream, ਦਸ਼ਮਲਵ min_arg, ਦਸ਼ਮਲਵ max_arg) {
            ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(k_elements, SYCL_CLAMP_ਖੰਡ_ਮਾਪ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ) * sycl::range<1>(SYCL_CLAMP_ਖੰਡ_ਮਾਪ),
                                  sycl::range<1>(SYCL_CLAMP_ਖੰਡ_ਮਾਪ)),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, ਵਸਤੂ_ct1);
                });
        }, min_val, max_val);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ਦਸ਼ਮਲਵ alpha_n = ggml_get_op_params_f32(dst, 1);
    ਅਚਲ ਦਸ਼ਮਲਵ alpha_p = ggml_get_op_params_f32(dst, 2);
    ਅਚਲ ਦਸ਼ਮਲਵ beta    = ggml_get_op_params_f32(dst, 3);
    ਅਚਲ ਦਸ਼ਮਲਵ eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ਅਚਲ ਸਵੈਚਾਲਿਤ* src, ਸਵੈਚਾਲਿਤ* dst_ptr, ਪੂਰਨ k_elements, ਕਤਾਰ_ptr stream, ਦਸ਼ਮਲਵ alpha_n_arg, ਦਸ਼ਮਲਵ alpha_p_arg, ਦਸ਼ਮਲਵ beta_arg, ਦਸ਼ਮਲਵ eps_arg) {
            ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(k_elements, SYCL_RELU_ਖੰਡ_ਮਾਪ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ) * sycl::range<1>(SYCL_RELU_ਖੰਡ_ਮਾਪ),
                                  sycl::range<1>(SYCL_RELU_ਖੰਡ_ਮਾਪ)),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, ਵਸਤੂ_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_floor(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_ceil(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_ਗੇੜਾ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_ਗੇੜਾ(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_trunc(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    ਅਚਲ ਅੱਖਰ  * src0_d = (ਅਚਲ ਅੱਖਰ  *) src0->ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ  * src1_d = (ਅਚਲ ਅੱਖਰ  *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ       * dst_d  = (ਦਸ਼ਮਲਵ       *)  dst->ਅੰਕੜੇ;

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_ਮਾਪ(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ਅਚਲ int64_t s1     = (int64_t) ((ਅਚਲ int32_t *) dst->op_params)[0] / (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t s2     = (int64_t) ((ਅਚਲ int32_t *) dst->op_params)[1] / (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t s3     = (int64_t) ((ਅਚਲ int32_t *) dst->op_params)[2] / (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ int64_t ਵਿੱਥ = (int64_t) ((ਅਚਲ int32_t *) dst->op_params)[3] / (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, ਵਿੱਥ, stream);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_gelu(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_relu(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
    ਚੋਣ (uop) {
        ਹਾਲਤ GGML_UNARY_OP_SILU:     launch([](ਦਸ਼ਮਲਵ v) { ਵਾਪਸ op_silu(v); });     ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_SIGMOID:  launch([](ਦਸ਼ਮਲਵ v) { ਵਾਪਸ op_sigmoid(v); });  ਰੋਕੋ;
        ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS: launch([](ਦਸ਼ਮਲਵ v) { ਵਾਪਸ op_softplus(v); }); ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
    }
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
ਖਾਲੀ ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_ਗੰਢ, ggml_tensor * mul_ਗੰਢ) {
    scope_op_debug_print scope_dbg_print(__func__, mul_ਗੰਢ, /*num_src=*/2);

    ਅਚਲ ggml_tensor * x = unary_ਗੰਢ->src[0];
    ਅਚਲ ggml_tensor * g = (mul_ਗੰਢ->src[0] == unary_ਗੰਢ) ? mul_ਗੰਢ->src[1] : mul_ਗੰਢ->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_ਗੰਢ);
    GGML_ASSERT(x->type == g->type && x->type == mul_ਗੰਢ->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_ਗੰਢ));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_ਗੰਢ));

    ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ int64_t k = ggml_nelements(mul_ਗੰਢ);
    ਅਚਲ int64_t n = mul_ਗੰਢ->ne[0];

    ਅਚਲ ਸਵੈਚਾਲਿਤ dispatch_type = [&](ਸਵੈਚਾਲਿਤ op) {
        ਚੋਣ (mul_ਗੰਢ->type) {
            ਹਾਲਤ GGML_TYPE_F32:
                unary_mul_sycl((ਅਚਲ ਦਸ਼ਮਲਵ *) x->ਅੰਕੜੇ, (ਅਚਲ ਦਸ਼ਮਲਵ *) g->ਅੰਕੜੇ, (ਦਸ਼ਮਲਵ *) mul_ਗੰਢ->ਅੰਕੜੇ,
                               k, n, x->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ), g->nb[1] / ਆਕਾਰ(ਦਸ਼ਮਲਵ), main_ਧਾਰਾ, op);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                unary_mul_sycl((ਅਚਲ sycl::half *) x->ਅੰਕੜੇ, (ਅਚਲ sycl::half *) g->ਅੰਕੜੇ, (sycl::half *) mul_ਗੰਢ->ਅੰਕੜੇ,
                               k, n, x->nb[1] / ਆਕਾਰ(sycl::half), g->nb[1] / ਆਕਾਰ(sycl::half), main_ਧਾਰਾ, op);
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_ਗੰਢ->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_ਗੰਢ), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
ਖਾਲੀ ggml_sycl_op_ਜੋੜੋ_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * ਜੋੜੋ_ਗੰਢ,
                                      ggml_tensor * unary_ਗੰਢ, ggml_tensor * mul_ਗੰਢ) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_ਗੰਢ, /*num_src=*/2);

    ਅਚਲ ggml_tensor * a     = ਜੋੜੋ_ਗੰਢ->src[0];
    ਅਚਲ ggml_tensor * bias  = ਜੋੜੋ_ਗੰਢ->src[1];
    ਅਚਲ ggml_tensor * scale = (mul_ਗੰਢ->src[0] == unary_ਗੰਢ) ? mul_ਗੰਢ->src[1] : mul_ਗੰਢ->src[0];

    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(scale != unary_ਗੰਢ);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_ਗੰਢ->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_ਗੰਢ));
    // a and dst are indexed flat
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_ਗੰਢ));
    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ ਸਵੈਚਾਲਿਤ dispatch_op = [&](ਸਵੈਚਾਲਿਤ op) {
        ਜੋੜੋ_unary_mul_sycl((ਅਚਲ ਦਸ਼ਮਲਵ *) a->ਅੰਕੜੇ, (ਅਚਲ ਦਸ਼ਮਲਵ *) bias->ਅੰਕੜੇ, (ਅਚਲ ਦਸ਼ਮਲਵ *) scale->ਅੰਕੜੇ,
                           (ਦਸ਼ਮਲਵ *) mul_ਗੰਢ->ਅੰਕੜੇ, ggml_nelements(mul_ਗੰਢ), mul_ਗੰਢ->ne[0], main_ਧਾਰਾ, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_ਗੰਢ), dispatch_op);
}

__dpct_inline__ ਦਸ਼ਮਲਵ ggml_sycl_op_swiglu_oai_single(ਦਸ਼ਮਲਵ x, ਦਸ਼ਮਲਵ g, ਦਸ਼ਮਲਵ alpha = 1.702f, ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ = 7.0f) {
    x = sycl::fmin(x, ਹੱਦਬੰਦੀ);
    g = sycl::fmax(sycl::fmin(g, ਹੱਦਬੰਦੀ), -ਹੱਦਬੰਦੀ);

    ਦਸ਼ਮਲਵ ਬਾਹਰਲਾ_glu = x / (1.0f + sycl::native::ਘਾਤਾਂਕ(-x * alpha));
    ਬਾਹਰਲਾ_glu = ਬਾਹਰਲਾ_glu * (1.0f + g);
    ਵਾਪਸ ਬਾਹਰਲਾ_glu;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ swiglu_oai_kernel(ਅਚਲ T * x, ਅਚਲ T * g, T * dst, ਅਚਲ int64_t k,
                              ਅਚਲ int64_t n, ਅਚਲ int64_t o0, ਅਚਲ int64_t o1,
                              ਦਸ਼ਮਲਵ alpha, ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ, sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
    ਅਚਲ int64_t i = int64_t(ਵਸਤੂ_ct1.get_local_range(2)) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) + ਵਸਤੂ_ct1.get_local_id(2);

    ਜੇ (i >= k) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t j0 = (i / n) * o0 + (i % n);
    ਅਚਲ int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ਦਸ਼ਮਲਵ xi = x[j0];
    ਦਸ਼ਮਲਵ gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, ਹੱਦਬੰਦੀ);
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ swiglu_oai_sycl(ਅਚਲ T *       x,
                            ਅਚਲ T *       g,
                            T *             dst,
                            ਅਚਲ int64_t   k,
                            ਅਚਲ int64_t   n,
                            ਅਚਲ int64_t   o0,
                            ਅਚਲ int64_t   o1,
                            ਅਚਲ ਦਸ਼ਮਲਵ     alpha,
                            ਅਚਲ ਦਸ਼ਮਲਵ     ਹੱਦਬੰਦੀ,
                            dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (k + SYCL_GLU_ਖੰਡ_ਮਾਪ - 1) / SYCL_GLU_ਖੰਡ_ਮਾਪ;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_GLU_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_GLU_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, ਹੱਦਬੰਦੀ, ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਖਾਲੀ * src0_d = src0->ਅੰਕੜੇ;
    ਖਾਲੀ * src1_d = src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ;
    ਅਚਲ int64_t src0_o = src0->nb[1];
    ਅਚਲ int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ਖਾਲੀ * dst_d = dst->ਅੰਕੜੇ;
    ਅਚਲ int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::ਕਤਾਰ_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_ਮਾਪ(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_ਮਾਪ(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ ਦਸ਼ਮਲਵ alpha = ggml_get_op_params_f32(dst, 2);
    ਅਚਲ ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ = ggml_get_op_params_f32(dst, 3);

    ਦਸ਼ਮਲਵ * src0_p = (ਦਸ਼ਮਲਵ *) src0_d;
    ਦਸ਼ਮਲਵ * src1_p = (ਦਸ਼ਮਲਵ *) src1_d;

    ਜੇ (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (ਦਸ਼ਮਲਵ *)dst_d, ggml_nelements(dst), nc, src0_o / ਆਕਾਰ(ਦਸ਼ਮਲਵ), src1_o / ਆਕਾਰ(ਦਸ਼ਮਲਵ), alpha, ਹੱਦਬੰਦੀ, stream);
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ swiglu_clamp_kernel(ਅਚਲ T *        gate,
                                ਅਚਲ T *        ਉੱਪਰ,
                                T *              dst,
                                ਅਚਲ int64_t    k,
                                ਅਚਲ int64_t    n,
                                ਅਚਲ int64_t    o0,
                                ਅਚਲ int64_t    o1,
                                ਦਸ਼ਮਲਵ            ਹੱਦਬੰਦੀ,
                                sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
    ਅਚਲ int64_t i = int64_t(ਵਸਤੂ_ct1.get_local_range(2)) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) + ਵਸਤੂ_ct1.get_local_id(2);

    ਜੇ (i >= k) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t j0 = (i / n) * o0 + (i % n);
    ਅਚਲ int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ਅਚਲ ਦਸ਼ਮਲਵ gate_ਮੁੱਲ = sycl::fmin((ਦਸ਼ਮਲਵ) gate[j0], ਹੱਦਬੰਦੀ);
    ਅਚਲ ਦਸ਼ਮਲਵ ਉੱਪਰ_ਮੁੱਲ   = sycl::fmax(sycl::fmin((ਦਸ਼ਮਲਵ) ਉੱਪਰ[j1], ਹੱਦਬੰਦੀ), -ਹੱਦਬੰਦੀ);
    dst[i]                 = (T) (gate_ਮੁੱਲ / (1.0f + sycl::native::ਘਾਤਾਂਕ(-gate_ਮੁੱਲ)) * ਉੱਪਰ_ਮੁੱਲ);
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ swiglu_clamp_sycl(ਅਚਲ T *       gate,
                              ਅਚਲ T *       ਉੱਪਰ,
                              T *             dst,
                              ਅਚਲ int64_t   k,
                              ਅਚਲ int64_t   n,
                              ਅਚਲ int64_t   o0,
                              ਅਚਲ int64_t   o1,
                              ਦਸ਼ਮਲਵ           ਹੱਦਬੰਦੀ,
                              dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (k + SYCL_GLU_ਖੰਡ_ਮਾਪ - 1) / SYCL_GLU_ਖੰਡ_ਮਾਪ;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_GLU_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_GLU_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             swiglu_clamp_kernel(gate, ਉੱਪਰ, dst, k, n, o0, o1, ਹੱਦਬੰਦੀ, ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0   = dst->src[0];
    ਅਚਲ ggml_tensor * src1   = dst->src[1];
    ਖਾਲੀ *              src0_d = src0->ਅੰਕੜੇ;
    ਖਾਲੀ *              src1_d = src1 ? src1->ਅੰਕੜੇ : src0->ਅੰਕੜੇ;
    ਅਚਲ int64_t       src0_o = src0->nb[1];
    ਅਚਲ int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ਖਾਲੀ *              dst_d  = dst->ਅੰਕੜੇ;
    ਅਚਲ int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::ਕਤਾਰ_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_ਮਾਪ(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    ਜੇ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_ਮਾਪ(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    ਅਚਲ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ਅਚਲ ਦਸ਼ਮਲਵ   ਹੱਦਬੰਦੀ   = ggml_get_op_params_f32(dst, 3);

    ਜੇ (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / ਆਕਾਰ(sycl::half),
                          src1_o / ਆਕਾਰ(sycl::half), ਹੱਦਬੰਦੀ, stream);
    } ਨਹੀਂਤਾਂ {
        ਦਸ਼ਮਲਵ * src0_p = (ਦਸ਼ਮਲਵ *) src0_d;
        ਦਸ਼ਮਲਵ * src1_p = (ਦਸ਼ਮਲਵ *) src1_d;

        ਜੇ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (ਦਸ਼ਮਲਵ *) dst_d, ggml_nelements(dst), nc, src0_o / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                          src1_o / ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਹੱਦਬੰਦੀ, stream);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_gelu_erf(x);
    });
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_geglu_ਤੇਜ਼(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](ਸਵੈਚਾਲਿਤ x) {
        ਵਾਪਸ op_gelu_ਤੇਜ਼(x);
    });
}


ਖਾਲੀ ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

ਖਾਲੀ ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

ਖਾਲੀ ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

ਖਾਲੀ ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

ਖਾਲੀ ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

ਖਾਲੀ ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

ਖਾਲੀ ggml_sycl_gelu_ਤੇਜ਼(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_ਤੇਜ਼(ctx, dst);
}

ਖਾਲੀ ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

ਖਾਲੀ ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

ਖਾਲੀ ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

ਖਾਲੀ ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

ਖਾਲੀ ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

ਖਾਲੀ ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

ਖਾਲੀ ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

ਖਾਲੀ ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_expm1(ctx, dst);
}

ਖਾਲੀ ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

ਖਾਲੀ ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

ਖਾਲੀ ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

ਖਾਲੀ ggml_sycl_ਕਦਮ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ਕਦਮ(ctx, dst);
}

ਖਾਲੀ ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

ਖਾਲੀ ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

ਖਾਲੀ ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

ਖਾਲੀ ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_xielu(ctx, dst);
}

ਖਾਲੀ ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

ਖਾਲੀ ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

ਖਾਲੀ ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

ਖਾਲੀ ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

ਖਾਲੀ ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

ਖਾਲੀ ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

ਖਾਲੀ ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

ਖਾਲੀ ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_clamp(ctx, dst);
}

ਖਾਲੀ ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

ਖਾਲੀ ggml_sycl_geglu_ਤੇਜ਼(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_ਤੇਜ਼(ctx, dst);
}

ਖਾਲੀ ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

ਖਾਲੀ ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

ਖਾਲੀ ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

ਖਾਲੀ ggml_sycl_ਗੇੜਾ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ਗੇੜਾ(ctx, dst);
}

ਖਾਲੀ ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
