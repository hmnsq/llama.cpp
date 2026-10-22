#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/presets.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "element_wise.hpp"

#define SYCL_GLOBAL_ID_LOOP(K, ITEM) \
    ಪ್ರತಿ (auto i = ITEM.get_global_id(0); i < (size_t)K; i += ITEM.get_global_range(0))

#define SYCL_LOCAL_ID_CALC(ITEM, IDX) \
    (ITEM.get_local_range(IDX) * ITEM.get_group(IDX) + ITEM.get_local_id(IDX))

ಸ್ಥಿರ ಶೂನ್ಯ acc_f32(ಧ್ರುವ ವರ್ಣ * x, ಧ್ರುವ ವರ್ಣ * y, ಭಿನ್ನ * dst, ಧ್ರುವ int64_t ne,
    ಧ್ರುವ int64_t ne0, ಧ್ರುವ int64_t ne1, ಧ್ರುವ int64_t ne2,
    ಧ್ರುವ int64_t nb00, ಧ್ರುವ int64_t nb01, ಧ್ರುವ int64_t nb02, ಧ್ರುವ int64_t nb03,
    ಧ್ರುವ int64_t ne10, ಧ್ರುವ int64_t ne11, ಧ್ರುವ int64_t ne12, ಧ್ರುವ int64_t ne13,
    ಧ್ರುವ int64_t nb10, ಧ್ರುವ int64_t nb11, ಧ್ರುವ int64_t nb12, ಧ್ರುವ int64_t nb13,
    ಧ್ರುವ int64_t s11, ಧ್ರುವ int64_t s12, ಧ್ರುವ int64_t s13, ಧ್ರುವ int64_t offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ int64_t i = SYCL_LOCAL_ID_CALC(item_ct1, 2);

    ಯದಿ (i >= ne) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    ಧ್ರುವ int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    ಧ್ರುವ int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    ಧ್ರುವ int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    ಧ್ರುವ int64_t i10 = tmp;

    int64_t tmp_dst = i;
    ಧ್ರುವ int64_t i3 = tmp_dst / (ne2*ne1*ne0);
    tmp_dst -= i3 * (ne2*ne1*ne0);
    ಧ್ರುವ int64_t i2 = tmp_dst / (ne1*ne0);
    tmp_dst -= i2 * (ne1*ne0);
    ಧ್ರುವ int64_t i1 = tmp_dst / ne0;
    tmp_dst -= i1 * ne0;
    ಧ್ರುವ int64_t i0 = tmp_dst;

    ಭಿನ್ನ val = *(ಧ್ರುವ ಭಿನ್ನ *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    ಯದಿ (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(ಧ್ರುವ ಭಿನ್ನ *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_sgn(T x) {
    ಪ್ರತ್ಯಾಗಮ x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_abs(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::fabs(x);
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_expm1(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<ಭಿನ್ನ>(x))
        );
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::expm1(x);
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_elu(T x) {
    ಪ್ರತ್ಯಾಗಮ (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_erf(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<ಭಿನ್ನ>(x))
        );
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::erf(x);
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_gelu_erf(T x) {
    ಧ್ರುವ T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    ಪ್ರತ್ಯಾಗಮ static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_gelu_quick(T x) {
    ಧ್ರುವ T GELU_QUICK_COEF_LOCAL = static_cast<T>(-1.702f);
    ಪ್ರತ್ಯಾಗಮ x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_QUICK_COEF_LOCAL * x)));
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_relu(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::fmax(x, static_cast<T>(0));
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_sigmoid(T x) {
    ಪ್ರತ್ಯಾಗಮ static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_sqrt(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::sqrt(x);
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::sqrt(x);
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_sin(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::sin(x);
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::sin(x);
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_cos(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::cos(x);
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::cos(x);
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_hardsigmoid(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_hardswish(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_log(T x) {
    ಯದಿ (x <= static_cast<T>(0)) {
        ಪ್ರತ್ಯಾಗಮ neg_infinity<T>();
    }
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::log(x);
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::log(x);
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_softplus(T x) {
    ಧ್ರುವ ಭಿನ್ನ xf = (ಭಿನ್ನ) x;
    ಧ್ರುವ ಭಿನ್ನ ax = op_abs(xf);
    ಧ್ರುವ ಭಿನ್ನ m  = sycl::fmax(xf, 0.0f);
    ಧ್ರುವ ಭಿನ್ನ y  = m + sycl::log1p(sycl::exp(-ax));
    ಪ್ರತ್ಯಾಗಮ (T) y;
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_neg(T x) {
    ಪ್ರತ್ಯಾಗಮ -x;
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_step(T x) {
    ಪ್ರತ್ಯಾಗಮ (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_leaky_relu(T x, ಭಿನ್ನ negative_slope) {
    T neg_slope_T = static_cast<T>(negative_slope);
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;

    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_xielu(T x, ಭಿನ್ನ alpha_n, ಭಿನ್ನ alpha_p, ಭಿನ್ನ beta, ಭಿನ್ನ eps) {
    ಧ್ರುವ ಭಿನ್ನ xi        = static_cast<ಭಿನ್ನ>(x);
    ಧ್ರುವ ಭಿನ್ನ gate_pos  = (xi > 0.0f);
    ಧ್ರುವ ಭಿನ್ನ y_pos     = alpha_p * xi * xi + beta * xi;
    ಧ್ರುವ ಭಿನ್ನ min_v_eps = sycl::fmin(xi, eps);
    ಧ್ರುವ ಭಿನ್ನ y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    ಧ್ರುವ ಭಿನ್ನ out       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    ಪ್ರತ್ಯಾಗಮ static_cast<T>(out);
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_sqr(T x) {
    ಪ್ರತ್ಯಾಗಮ x * x;
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_clamp(T x, ಭಿನ್ನ min_val, ಭಿನ್ನ max_val) {
    ಪ್ರತ್ಯಾಗಮ x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_floor(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::floor(x);
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::floor(x);
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_ceil(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::ceil(x);
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::ceil(x);
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_round(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<ಭಿನ್ನ>(x))
        );
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::round(x);
    }
}

template<typename T>
ಸ್ಥಿರ __dpct_inline__ T op_trunc(T x) {
    ಯದಿ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ಪ್ರತ್ಯಾಗಮ sycl::ext::oneapi::experimental::trunc(x);
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ sycl::trunc(x);
    }
}

template<typename T, typename F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_flat_kernel(ಧ್ರುವ T * x, T * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]);
    }
}

template<typename T, typename F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_generic_kernel(
        ಧ್ರುವ T * x,
        T * dst,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k,
        ಧ್ರುವ sycl::uint3 ne0_fd, ಧ್ರುವ sycl::uint3 ne1_fd, ಧ್ರುವ sycl::uint3 ne2_fd,
        ಧ್ರುವ size_t nb0,  ಧ್ರುವ size_t nb1,  ಧ್ರುವ size_t nb2,  ಧ್ರುವ size_t nb3,
        ಧ್ರುವ size_t nbd0, ಧ್ರುವ size_t nbd1, ಧ್ರುವ size_t nbd2, ಧ್ರುವ size_t nbd3,
        ಧ್ರುವ sycl::nd_item<1> & item_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        ಧ್ರುವ uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        ಧ್ರುವ uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        ಧ್ರುವ uint32_t i2 = dm.y();
        ಧ್ರುವ uint32_t i3 = dm.x();

        ಧ್ರುವ ವರ್ಣ * src_base = (ಧ್ರುವ ವರ್ಣ *) x;
        ವರ್ಣ       * dst_base = (ವರ್ಣ *) dst;

        ಧ್ರುವ T * srcp = (ಧ್ರುವ T *)(src_base + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_base + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_sqrt_kernel(ಧ್ರುವ T * x, T * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_sin_kernel(ಧ್ರುವ T * x, T * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_cos_kernel(ಧ್ರುವ T * x, T * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_log_kernel(ಧ್ರುವ T * x, T * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_log(x[i]);
    }
}


template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_leaky_relu_kernel(ಧ್ರುವ T * x, T * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k, ಭಿನ್ನ negative_slope, ಧ್ರುವ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_slope);
    }
}

template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_xielu_kernel(ಧ್ರುವ T * x, T * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k, ಭಿನ್ನ alpha_n, ಭಿನ್ನ alpha_p, ಭಿನ್ನ beta, ಭಿನ್ನ eps, ಧ್ರುವ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_sqr_kernel(ಧ್ರುವ T * x, T * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_clamp_kernel(ಧ್ರುವ T * x, T * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ sycl::nd_item<1> &item_ct1, ಭಿನ್ನ min_val, ಭಿನ್ನ max_val) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ unary_op_ceil_kernel(ಧ್ರುವ T * x, T * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ clamp(ಧ್ರುವ T * x, T * dst, ಧ್ರುವ ಭಿನ್ನ min, ಧ್ರುವ ಭಿನ್ನ max, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k,
                      ಧ್ರುವ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

template<typename T, typename F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_gated_op_flat_kernel(ಧ್ರುವ T * x, ಧ್ರುವ T * g, T * dst, ಧ್ರುವ uint64_t k, ಧ್ರುವ sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

template<typename T, typename F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_gated_op_generic_kernel(
        ಧ್ರುವ T * x,
        ಧ್ರುವ T * g,
        T * dst,
        ಧ್ರುವ uint64_t k,
        ಧ್ರುವ sycl::uint3 n_fd,
        ಧ್ರುವ uint64_t o0,
        ಧ್ರುವ uint64_t o1,
        ಧ್ರುವ sycl::nd_item<1> & item_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ಧ್ರುವ sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ಧ್ರುವ int64_t j0 = rc.x() * o0 + rc.y();
        ಧ್ರುವ int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
template<typename T, typename F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_mul_flat_kernel(ಧ್ರುವ T * x, ಧ್ರುವ T * g, T * dst, ಧ್ರುವ int64_t k, ಧ್ರುವ sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = (T) (op((ಭಿನ್ನ) x[i]) * (ಭಿನ್ನ) g[i]);
    }
}

template<typename T, typename F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_mul_strided_kernel(ಧ್ರುವ T * x, ಧ್ರುವ T * g, T * dst, ಧ್ರುವ int64_t k, ಧ್ರುವ sycl::uint3 n_fd, ಧ್ರುವ int64_t o0, ಧ್ರುವ int64_t o1, ಧ್ರುವ sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ಧ್ರುವ sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ಧ್ರುವ int64_t j0 = rc.x() * o0 + rc.y();
        ಧ್ರುವ int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((ಭಿನ್ನ) x[j0]) * (ಭಿನ್ನ) g[j1]);
    }
}

template<typename T, typename F>
ಸ್ಥಿರ ಶೂನ್ಯ unary_mul_sycl(ಧ್ರುವ T * x, ಧ್ರುವ T * g, T * dst, ಧ್ರುವ int64_t k, ಧ್ರುವ int64_t n, ಧ್ರುವ int64_t o0, ಧ್ರುವ int64_t o1, queue_ptr main_stream, F op) {
    ಧ್ರುವ size_t            ಅಂಕೆ_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    ಧ್ರುವ sycl::nd_range<1> range(ಅಂಕೆ_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    ಯದಿ (o0 == n && o1 == n) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            unary_mul_flat_kernel(x, g, dst, k, item_ct1, op);
        });
        ಪ್ರತ್ಯಾಗಮ;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ಧ್ರುವ sycl::uint3 n_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) n);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, item_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
template<typename F>
ಸ್ಥಿರ ಶೂನ್ಯ add_unary_mul_flat_kernel(ಧ್ರುವ ಭಿನ್ನ * a, ಧ್ರುವ ಭಿನ್ನ * bias, ಧ್ರುವ ಭಿನ್ನ * scale, ಭಿನ್ನ * dst,
                                      ಧ್ರುವ int64_t k, ಧ್ರುವ sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

template<typename F>
ಸ್ಥಿರ ಶೂನ್ಯ add_unary_mul_bcast_kernel(ಧ್ರುವ ಭಿನ್ನ * a, ಧ್ರುವ ಭಿನ್ನ * bias, ಧ್ರುವ ಭಿನ್ನ * scale, ಭಿನ್ನ * dst,
                                       ಧ್ರುವ int64_t k, ಧ್ರುವ sycl::uint3 ne0_fd, ಧ್ರುವ sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ಧ್ರುವ uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

template<typename F>
ಸ್ಥಿರ ಶೂನ್ಯ add_unary_mul_sycl(ಧ್ರುವ ಭಿನ್ನ * a, ಧ್ರುವ ಭಿನ್ನ * bias, ಧ್ರುವ ಭಿನ್ನ * scale, ಭಿನ್ನ * dst,
                               ಧ್ರುವ int64_t k, ಧ್ರುವ int64_t ne0, queue_ptr main_stream, F op) {
    ಧ್ರುವ size_t            ಅಂಕೆ_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    ಧ್ರುವ sycl::nd_range<1> range(ಅಂಕೆ_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    ಯದಿ (k == ne0) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            add_unary_mul_flat_kernel(a, bias, scale, dst, k, item_ct1, op);
        });
        ಪ್ರತ್ಯಾಗಮ;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ಧ್ರುವ sycl::uint3 ne0_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) ne0);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        add_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, item_ct1, op);
    });
}

ನಾಮಾಕಾಶ ggml_sycl_detail {
ಸ್ಥಿರ ಶೂನ್ಯ acc_f32_sycl(ಧ್ರುವ ವರ್ಣ *x, ಧ್ರುವ ವರ್ಣ *y, ಭಿನ್ನ *dst,
                         ಧ್ರುವ int64_t n_elements,
                         ಧ್ರುವ int64_t ne0, ಧ್ರುವ int64_t ne1, ಧ್ರುವ int64_t ne2,
                         ಧ್ರುವ int64_t nb00, ಧ್ರುವ int64_t nb01, ಧ್ರುವ int64_t nb02, ಧ್ರುವ int64_t nb03,
                         ಧ್ರುವ int64_t ne10, ಧ್ರುವ int64_t ne11, ಧ್ರುವ int64_t ne12, ಧ್ರುವ int64_t ne13,
                         ಧ್ರುವ int64_t nb10, ಧ್ರುವ int64_t nb11, ಧ್ರುವ int64_t nb12, ಧ್ರುವ int64_t nb13,
                         ಧ್ರುವ int64_t s1, ಧ್ರುವ int64_t s2, ಧ್ರುವ int64_t s3,
                         ಧ್ರುವ int64_t offset, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                            acc_f32(x, y, dst, n_elements,
                                ne0, ne1, ne2,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                s1, s2, s3, offset);
                         });
}

template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ arange_kernel(T * dst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k, T start, T step,
                         ಧ್ರುವ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = start + static_cast<T>(i) * step;
    }
}

template<typename KernelInvoker, typename... Args>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ವಿಕಲ್ಪ (dst->type) {
        ಪಕ್ಷ GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (ಪೂರ್ಣಾಙ್ಕ)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                ಭಙ್ಗ;
            }
#ifdef GGML_SYCL_HAS_BF16
        ಪಕ್ಷ GGML_TYPE_BF16:
            {
                auto data_pts = cast_data<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (ಪೂರ್ಣಾಙ್ಕ)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                ಭಙ್ಗ;
            }
#endif
        ಪಕ್ಷ GGML_TYPE_F32:
            {
                auto data_pts = cast_data<ಭಿನ್ನ>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (ಪೂರ್ಣಾಙ್ಕ)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                ಭಙ್ಗ;
            }
        ಸ್ವಭಾವ:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename KernelInvoker, typename... Args>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];
    ಧ್ರುವ int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // The fused GLU kernels index with 32-bit fastdiv, which is exact only for indices below
    // 2^31. A dst that large is ~8 GB at f32, and the grid sizing already narrows to 32 bits,
    // so assert the bound rather than carry a second code path for it.
    GGML_ASSERT(ggml_nelements(dst) < ((int64_t) 1 << 31));
    ಧ್ರುವ int32_t swapped = ((ಧ್ರುವ int32_t *) dst->op_params)[1];
    ಶೂನ್ಯ * src0_d = src0->data;
    ಶೂನ್ಯ * src1_d = src1 ? src1->data : src0->data;
    ಧ್ರುವ int64_t src0_o = src0->nb[1];
    ಧ್ರುವ int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ಶೂನ್ಯ * dst_d = dst->data;
    ಯದಿ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    ವಿಕಲ್ಪ (dst->type) {
        ಪಕ್ಷ GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    ಯದಿ (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / ಪರಿಮಾಣ(sycl::half),
                               src1_o / ಪರಿಮಾಣ(sycl::half),
                               main_stream,
                               std::forward<Args>(args)...);
                ಭಙ್ಗ;
            }
        ಪಕ್ಷ GGML_TYPE_F32:
            {
                ಭಿನ್ನ * src0_p = (ಭಿನ್ನ *) src0_d;
                ಭಿನ್ನ * src1_p = (ಭಿನ್ನ *) src1_d;

                    ಯದಿ (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (ಭಿನ್ನ *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / ಪರಿಮಾಣ(ಭಿನ್ನ),
                               src1_o / ಪರಿಮಾಣ(ಭಿನ್ನ),
                               main_stream,
                               std::forward<Args>(args)...);
                ಭಙ್ಗ;
            }
        ಸ್ವಭಾವ:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename F>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    ಧ್ರುವ int64_t ne0  = dst->ne[0];
    ಧ್ರುವ int64_t ne1  = dst->ne[1];
    ಧ್ರುವ int64_t ne2  = dst->ne[2];

    ಧ್ರುವ size_t  nb0  = src0->nb[0];
    ಧ್ರುವ size_t  nb1  = src0->nb[1];
    ಧ್ರುವ size_t  nb2  = src0->nb[2];
    ಧ್ರುವ size_t  nb3  = src0->nb[3];

    ಧ್ರುವ size_t  nbd0 = dst->nb[0];
    ಧ್ರುವ size_t  nbd1 = dst->nb[1];
    ಧ್ರುವ size_t  nbd2 = dst->nb[2];
    ಧ್ರುವ size_t  nbd3 = dst->nb[3];

    // Hot unary ops (FFN/GDN silu, sigmoid, ...) run on contiguous tensors;
    // skip the strided index math entirely for them.
    ಧ್ರುವ ತಾರ್ಕಿಕ contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](ಧ್ರುವ auto* src, auto* dst_ptr, ಪೂರ್ಣಾಙ್ಕ k_elements, queue_ptr stream) {

            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(k_elements, 256);

            ಯದಿ (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, item_ct1, func);
                    });
            } ಅನ್ಯಥಾ {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                ಧ್ರುವ sycl::uint3 ne0_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) ne0);
                ಧ್ರುವ sycl::uint3 ne1_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) ne1);
                ಧ್ರುವ sycl::uint3 ne2_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_op_generic_kernel(
                            src, dst_ptr, k_elements,
                            ne0_fd, ne1_fd, ne2_fd,
                            nb0, nb1, nb2, nb3,
                            nbd0, nbd1, nbd2, nbd3,
                            item_ct1,
                            func
                        );
                    });
            }
        });
}

template<typename F>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](ಧ್ರುವ auto * x_ptr, ಧ್ರುವ auto * g_ptr, auto * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {

            ಧ್ರುವ uint32_t ಅಂಕೆ_blocks = (uint32_t) ceil_div(k, SYCL_GLU_BLOCK_SIZE);
            ಧ್ರುವ sycl::nd_range<1> launch_range(ಅಂಕೆ_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE),
                                                 sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            ಯದಿ (o0 == n && o1 == n) {
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, item_ct1, func);
                    });
            } ಅನ್ಯಥಾ {
                // launch-invariant divisor, and only this path needs it
                ಧ್ರುವ sycl::uint3 n_fd = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) n);
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, item_ct1, func);
                    });
            }
        });
}


ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    ಭಿನ್ನ start, stop, step;
    memcpy(&start, dst->op_params, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&stop, (ಭಿನ್ನ *) dst->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&step, (ಭಿನ್ನ *) dst->op_params + 2, ಪರಿಮಾಣ(ಭಿನ್ನ));
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ಭಿನ್ನ * dst_ptr = (ಭಿನ್ನ *)dst->data;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ k = (ಪೂರ್ಣಾಙ್ಕ)ggml_nelements(dst);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(k, SYCL_ARANGE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks) * sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE),
                          sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            arange_kernel(dst_ptr, k, start, step, item_ct1);
        });
}

} // namespace ggml_sycl_detail



ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_sgn(x);
    });
}


ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_abs(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_elu(x);
    });
}
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_silu(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_gelu(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_gelu_quick(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_gelu_erf(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_tanh(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_relu(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_hardsigmoid(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_hardswish(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_exp(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_expm1(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಧ್ರುವ auto* src, auto* dst_ptr, ಪೂರ್ಣಾಙ್ಕ k_elements, queue_ptr stream) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_softplus(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_neg(x);
    });
}


ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_step(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_sigmoid(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಧ್ರುವ auto* src, auto* dst_ptr, ಪೂರ್ಣಾಙ್ಕ k_elements, queue_ptr stream) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(k_elements, SYCL_SQRT_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks) * sycl::range<1>(SYCL_SQRT_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQRT_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಧ್ರುವ auto* src, auto* dst_ptr, ಪೂರ್ಣಾಙ್ಕ k_elements, queue_ptr stream) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಧ್ರುವ auto* src, auto* dst_ptr, ಪೂರ್ಣಾಙ್ಕ k_elements, queue_ptr stream) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಭಿನ್ನ negative_slope;
    memcpy(&negative_slope, dst->op_params, ಪರಿಮಾಣ(ಭಿನ್ನ));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಧ್ರುವ auto* src, auto* dst_ptr, ಪೂರ್ಣಾಙ್ಕ k_elements, queue_ptr stream, ಭಿನ್ನ slope) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, slope, item_ct1);
                });
        }, negative_slope);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಧ್ರುವ auto* src, auto* dst_ptr, ಪೂರ್ಣಾಙ್ಕ k_elements, queue_ptr stream) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(k_elements, SYCL_SQR_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks) * sycl::range<1>(SYCL_SQR_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQR_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಭಿನ್ನ min_val;
    ಭಿನ್ನ max_val;
    memcpy(&min_val, dst->op_params, ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&max_val, (ಭಿನ್ನ *) dst->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಧ್ರುವ auto* src, auto* dst_ptr, ಪೂರ್ಣಾಙ್ಕ k_elements, queue_ptr stream, ಭಿನ್ನ min_arg, ಭಿನ್ನ max_arg) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(k_elements, SYCL_CLAMP_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks) * sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, item_ct1);
                });
        }, min_val, max_val);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಧ್ರುವ ಭಿನ್ನ alpha_n = ggml_get_op_params_f32(dst, 1);
    ಧ್ರುವ ಭಿನ್ನ alpha_p = ggml_get_op_params_f32(dst, 2);
    ಧ್ರುವ ಭಿನ್ನ beta    = ggml_get_op_params_f32(dst, 3);
    ಧ್ರುವ ಭಿನ್ನ eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ಧ್ರುವ auto* src, auto* dst_ptr, ಪೂರ್ಣಾಙ್ಕ k_elements, queue_ptr stream, ಭಿನ್ನ alpha_n_arg, ಭಿನ್ನ alpha_p_arg, ಭಿನ್ನ beta_arg, ಭಿನ್ನ eps_arg) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, item_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_floor(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_ceil(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_round(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_trunc(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];

    ಧ್ರುವ ವರ್ಣ  * src0_d = (ಧ್ರುವ ವರ್ಣ  *) src0->data;
    ಧ್ರುವ ವರ್ಣ  * src1_d = (ಧ್ರುವ ವರ್ಣ  *) src1->data;
    ಭಿನ್ನ       * dst_d  = (ಭಿನ್ನ       *)  dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ಧ್ರುವ int64_t s1     = (int64_t) ((ಧ್ರುವ int32_t *) dst->op_params)[0] / (int64_t) ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t s2     = (int64_t) ((ಧ್ರುವ int32_t *) dst->op_params)[1] / (int64_t) ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t s3     = (int64_t) ((ಧ್ರುವ int32_t *) dst->op_params)[2] / (int64_t) ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t offset = (int64_t) ((ಧ್ರುವ int32_t *) dst->op_params)[3] / (int64_t) ಪರಿಮಾಣ(ಭಿನ್ನ);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, offset, stream);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_gelu(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_relu(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
template<typename F>
ಸ್ಥಿರ ಶೂನ್ಯ dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
    ವಿಕಲ್ಪ (uop) {
        ಪಕ್ಷ GGML_UNARY_OP_SILU:     launch([](ಭಿನ್ನ v) { ಪ್ರತ್ಯಾಗಮ op_silu(v); });     ಭಙ್ಗ;
        ಪಕ್ಷ GGML_UNARY_OP_SIGMOID:  launch([](ಭಿನ್ನ v) { ಪ್ರತ್ಯಾಗಮ op_sigmoid(v); });  ಭಙ್ಗ;
        ಪಕ್ಷ GGML_UNARY_OP_SOFTPLUS: launch([](ಭಿನ್ನ v) { ಪ್ರತ್ಯಾಗಮ op_softplus(v); }); ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
    }
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
ಶೂನ್ಯ ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_ಗಂಟು, ggml_tensor * mul_ಗಂಟು) {
    scope_op_debug_print scope_dbg_print(__func__, mul_ಗಂಟು, /*num_src=*/2);

    ಧ್ರುವ ggml_tensor * x = unary_ಗಂಟು->src[0];
    ಧ್ರುವ ggml_tensor * g = (mul_ಗಂಟು->src[0] == unary_ಗಂಟು) ? mul_ಗಂಟು->src[1] : mul_ಗಂಟು->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_ಗಂಟು);
    GGML_ASSERT(x->type == g->type && x->type == mul_ಗಂಟು->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_ಗಂಟು));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_ಗಂಟು));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ಧ್ರುವ int64_t k = ggml_nelements(mul_ಗಂಟು);
    ಧ್ರುವ int64_t n = mul_ಗಂಟು->ne[0];

    ಧ್ರುವ auto dispatch_type = [&](auto op) {
        ವಿಕಲ್ಪ (mul_ಗಂಟು->type) {
            ಪಕ್ಷ GGML_TYPE_F32:
                unary_mul_sycl((ಧ್ರುವ ಭಿನ್ನ *) x->data, (ಧ್ರುವ ಭಿನ್ನ *) g->data, (ಭಿನ್ನ *) mul_ಗಂಟು->data,
                               k, n, x->nb[1] / ಪರಿಮಾಣ(ಭಿನ್ನ), g->nb[1] / ಪರಿಮಾಣ(ಭಿನ್ನ), main_stream, op);
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                unary_mul_sycl((ಧ್ರುವ sycl::half *) x->data, (ಧ್ರುವ sycl::half *) g->data, (sycl::half *) mul_ಗಂಟು->data,
                               k, n, x->nb[1] / ಪರಿಮಾಣ(sycl::half), g->nb[1] / ಪರಿಮಾಣ(sycl::half), main_stream, op);
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_ಗಂಟು->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_ಗಂಟು), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
ಶೂನ್ಯ ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_ಗಂಟು,
                                      ggml_tensor * unary_ಗಂಟು, ggml_tensor * mul_ಗಂಟು) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_ಗಂಟು, /*num_src=*/2);

    ಧ್ರುವ ggml_tensor * a     = add_ಗಂಟು->src[0];
    ಧ್ರುವ ggml_tensor * bias  = add_ಗಂಟು->src[1];
    ಧ್ರುವ ggml_tensor * scale = (mul_ಗಂಟು->src[0] == unary_ಗಂಟು) ? mul_ಗಂಟು->src[1] : mul_ಗಂಟು->src[0];

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

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ಧ್ರುವ auto dispatch_op = [&](auto op) {
        add_unary_mul_sycl((ಧ್ರುವ ಭಿನ್ನ *) a->data, (ಧ್ರುವ ಭಿನ್ನ *) bias->data, (ಧ್ರುವ ಭಿನ್ನ *) scale->data,
                           (ಭಿನ್ನ *) mul_ಗಂಟು->data, ggml_nelements(mul_ಗಂಟು), mul_ಗಂಟು->ne[0], main_stream, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_ಗಂಟು), dispatch_op);
}

__dpct_inline__ ಭಿನ್ನ ggml_sycl_op_swiglu_oai_single(ಭಿನ್ನ x, ಭಿನ್ನ g, ಭಿನ್ನ alpha = 1.702f, ಭಿನ್ನ limit = 7.0f) {
    x = sycl::fmin(x, limit);
    g = sycl::fmax(sycl::fmin(g, limit), -limit);

    ಭಿನ್ನ out_glu = x / (1.0f + sycl::native::exp(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    ಪ್ರತ್ಯಾಗಮ out_glu;
}

template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ swiglu_oai_kernel(ಧ್ರುವ T * x, ಧ್ರುವ T * g, T * dst, ಧ್ರುವ int64_t k,
                              ಧ್ರುವ int64_t n, ಧ್ರುವ int64_t o0, ಧ್ರುವ int64_t o1,
                              ಭಿನ್ನ alpha, ಭಿನ್ನ limit, sycl::nd_item<3> item_ct1) {
    ಧ್ರುವ int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    ಯದಿ (i >= k) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ int64_t j0 = (i / n) * o0 + (i % n);
    ಧ್ರುವ int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ಭಿನ್ನ xi = x[j0];
    ಭಿನ್ನ gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ swiglu_oai_sycl(ಧ್ರುವ T *       x,
                            ಧ್ರುವ T *       g,
                            T *             dst,
                            ಧ್ರುವ int64_t   k,
                            ಧ್ರುವ int64_t   n,
                            ಧ್ರುವ int64_t   o0,
                            ಧ್ರುವ int64_t   o1,
                            ಧ್ರುವ ಭಿನ್ನ     alpha,
                            ಧ್ರುವ ಭಿನ್ನ     limit,
                            dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, limit, item_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];
    ಶೂನ್ಯ * src0_d = src0->data;
    ಶೂನ್ಯ * src1_d = src1 ? src1->data : src0->data;
    ಧ್ರುವ int64_t src0_o = src0->nb[1];
    ಧ್ರುವ int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ಶೂನ್ಯ * dst_d = dst->data;
    ಧ್ರುವ int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    ಯದಿ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    ಧ್ರುವ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ಧ್ರುವ ಭಿನ್ನ alpha = ggml_get_op_params_f32(dst, 2);
    ಧ್ರುವ ಭಿನ್ನ limit = ggml_get_op_params_f32(dst, 3);

    ಭಿನ್ನ * src0_p = (ಭಿನ್ನ *) src0_d;
    ಭಿನ್ನ * src1_p = (ಭಿನ್ನ *) src1_d;

    ಯದಿ (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (ಭಿನ್ನ *)dst_d, ggml_nelements(dst), nc, src0_o / ಪರಿಮಾಣ(ಭಿನ್ನ), src1_o / ಪರಿಮಾಣ(ಭಿನ್ನ), alpha, limit, stream);
}

template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ swiglu_clamp_kernel(ಧ್ರುವ T *        gate,
                                ಧ್ರುವ T *        up,
                                T *              dst,
                                ಧ್ರುವ int64_t    k,
                                ಧ್ರುವ int64_t    n,
                                ಧ್ರುವ int64_t    o0,
                                ಧ್ರುವ int64_t    o1,
                                ಭಿನ್ನ            limit,
                                sycl::nd_item<3> item_ct1) {
    ಧ್ರುವ int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    ಯದಿ (i >= k) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ int64_t j0 = (i / n) * o0 + (i % n);
    ಧ್ರುವ int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ಧ್ರುವ ಭಿನ್ನ gate_ಮೌಲ್ಯ = sycl::fmin((ಭಿನ್ನ) gate[j0], limit);
    ಧ್ರುವ ಭಿನ್ನ up_ಮೌಲ್ಯ   = sycl::fmax(sycl::fmin((ಭಿನ್ನ) up[j1], limit), -limit);
    dst[i]                 = (T) (gate_ಮೌಲ್ಯ / (1.0f + sycl::native::exp(-gate_ಮೌಲ್ಯ)) * up_ಮೌಲ್ಯ);
}

template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ swiglu_clamp_sycl(ಧ್ರುವ T *       gate,
                              ಧ್ರುವ T *       up,
                              T *             dst,
                              ಧ್ರುವ int64_t   k,
                              ಧ್ರುವ int64_t   n,
                              ಧ್ರುವ int64_t   o0,
                              ಧ್ರುವ int64_t   o1,
                              ಭಿನ್ನ           limit,
                              dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_clamp_kernel(gate, up, dst, k, n, o0, o1, limit, item_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0   = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1   = dst->src[1];
    ಶೂನ್ಯ *              src0_d = src0->data;
    ಶೂನ್ಯ *              src1_d = src1 ? src1->data : src0->data;
    ಧ್ರುವ int64_t       src0_o = src0->nb[1];
    ಧ್ರುವ int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ಶೂನ್ಯ *              dst_d  = dst->data;
    ಧ್ರುವ int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    ಯದಿ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    ಧ್ರುವ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ಧ್ರುವ ಭಿನ್ನ   limit   = ggml_get_op_params_f32(dst, 3);

    ಯದಿ (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        ಯದಿ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / ಪರಿಮಾಣ(sycl::half),
                          src1_o / ಪರಿಮಾಣ(sycl::half), limit, stream);
    } ಅನ್ಯಥಾ {
        ಭಿನ್ನ * src0_p = (ಭಿನ್ನ *) src0_d;
        ಭಿನ್ನ * src1_p = (ಭಿನ್ನ *) src1_d;

        ಯದಿ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (ಭಿನ್ನ *) dst_d, ggml_nelements(dst), nc, src0_o / ಪರಿಮಾಣ(ಭಿನ್ನ),
                          src1_o / ಪರಿಮಾಣ(ಭಿನ್ನ), limit, stream);
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_gelu_erf(x);
    });
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        ಪ್ರತ್ಯಾಗಮ op_gelu_quick(x);
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

ಶೂನ್ಯ ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_quick(ctx, dst);
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

ಶೂನ್ಯ ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_step(ctx, dst);
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

ಶೂನ್ಯ ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_quick(ctx, dst);
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

ಶೂನ್ಯ ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_round(ctx, dst);
}

ಶೂನ್ಯ ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
