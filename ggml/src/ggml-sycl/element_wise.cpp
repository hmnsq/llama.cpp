#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/presets.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "element_wise.hpp"

#define SYCL_GLOBAL_ID_LOOP(K, ITEM) \
    ପାଇଁ (auto i = ITEM.get_global_id(0); i < (size_t)K; i += ITEM.get_global_range(0))

#define SYCL_LOCAL_ID_CALC(ITEM, IDX) \
    (ITEM.get_local_range(IDX) * ITEM.get_group(IDX) + ITEM.get_local_id(IDX))

ସ୍ଥିର ଶୂନ୍ୟ acc_f32(ଅଚଳ ଅକ୍ଷର * x, ଅଚଳ ଅକ୍ଷର * y, ଦଶମିକ * dst, ଅଚଳ int64_t ne,
    ଅଚଳ int64_t ne0, ଅଚଳ int64_t ne1, ଅଚଳ int64_t ne2,
    ଅଚଳ int64_t nb00, ଅଚଳ int64_t nb01, ଅଚଳ int64_t nb02, ଅଚଳ int64_t nb03,
    ଅଚଳ int64_t ne10, ଅଚଳ int64_t ne11, ଅଚଳ int64_t ne12, ଅଚଳ int64_t ne13,
    ଅଚଳ int64_t nb10, ଅଚଳ int64_t nb11, ଅଚଳ int64_t nb12, ଅଚଳ int64_t nb13,
    ଅଚଳ int64_t s11, ଅଚଳ int64_t s12, ଅଚଳ int64_t s13, ଅଚଳ int64_t offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ int64_t i = SYCL_LOCAL_ID_CALC(item_ct1, 2);

    ଯଦି (i >= ne) {
        ଫେରନ୍ତୁ;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    ଅଚଳ int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    ଅଚଳ int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    ଅଚଳ int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    ଅଚଳ int64_t i10 = tmp;

    int64_t tmp_dst = i;
    ଅଚଳ int64_t i3 = tmp_dst / (ne2*ne1*ne0);
    tmp_dst -= i3 * (ne2*ne1*ne0);
    ଅଚଳ int64_t i2 = tmp_dst / (ne1*ne0);
    tmp_dst -= i2 * (ne1*ne0);
    ଅଚଳ int64_t i1 = tmp_dst / ne0;
    tmp_dst -= i1 * ne0;
    ଅଚଳ int64_t i0 = tmp_dst;

    ଦଶମିକ val = *(ଅଚଳ ଦଶମିକ *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    ଯଦି (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(ଅଚଳ ଦଶମିକ *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
template<typename T>
ସ୍ଥିର __dpct_inline__ T op_sgn(T x) {
    ଫେରନ୍ତୁ x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


template<typename T>
ସ୍ଥିର __dpct_inline__ T op_abs(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::fabs(x);
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_expm1(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<ଦଶମିକ>(x))
        );
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::expm1(x);
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_elu(T x) {
    ଫେରନ୍ତୁ (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_erf(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<ଦଶମିକ>(x))
        );
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::erf(x);
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_gelu_erf(T x) {
    ଅଚଳ T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    ଫେରନ୍ତୁ static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_gelu_quick(T x) {
    ଅଚଳ T GELU_QUICK_COEF_LOCAL = static_cast<T>(-1.702f);
    ଫେରନ୍ତୁ x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_QUICK_COEF_LOCAL * x)));
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_relu(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::fmax(x, static_cast<T>(0));
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_sigmoid(T x) {
    ଫେରନ୍ତୁ static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_sqrt(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::sqrt(x);
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::sqrt(x);
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_sin(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::sin(x);
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::sin(x);
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_cos(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::cos(x);
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::cos(x);
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_hardsigmoid(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_hardswish(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_log(T x) {
    ଯଦି (x <= static_cast<T>(0)) {
        ଫେରନ୍ତୁ neg_infinity<T>();
    }
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::log(x);
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::log(x);
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_softplus(T x) {
    ଅଚଳ ଦଶମିକ xf = (ଦଶମିକ) x;
    ଅଚଳ ଦଶମିକ ax = op_abs(xf);
    ଅଚଳ ଦଶମିକ m  = sycl::fmax(xf, 0.0f);
    ଅଚଳ ଦଶମିକ y  = m + sycl::log1p(sycl::exp(-ax));
    ଫେରନ୍ତୁ (T) y;
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_neg(T x) {
    ଫେରନ୍ତୁ -x;
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_step(T x) {
    ଫେରନ୍ତୁ (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_leaky_relu(T x, ଦଶମିକ negative_slope) {
    T neg_slope_T = static_cast<T>(negative_slope);
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;

    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_xielu(T x, ଦଶମିକ alpha_n, ଦଶମିକ alpha_p, ଦଶମିକ beta, ଦଶମିକ eps) {
    ଅଚଳ ଦଶମିକ xi        = static_cast<ଦଶମିକ>(x);
    ଅଚଳ ଦଶମିକ gate_pos  = (xi > 0.0f);
    ଅଚଳ ଦଶମିକ y_pos     = alpha_p * xi * xi + beta * xi;
    ଅଚଳ ଦଶମିକ min_v_eps = sycl::fmin(xi, eps);
    ଅଚଳ ଦଶମିକ y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    ଅଚଳ ଦଶମିକ out       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    ଫେରନ୍ତୁ static_cast<T>(out);
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_sqr(T x) {
    ଫେରନ୍ତୁ x * x;
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_clamp(T x, ଦଶମିକ min_val, ଦଶମିକ max_val) {
    ଫେରନ୍ତୁ x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_floor(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::floor(x);
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::floor(x);
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_ceil(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::ceil(x);
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::ceil(x);
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_round(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<ଦଶମିକ>(x))
        );
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::round(x);
    }
}

template<typename T>
ସ୍ଥିର __dpct_inline__ T op_trunc(T x) {
    ଯଦି constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        ଫେରନ୍ତୁ sycl::ext::oneapi::experimental::trunc(x);
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ sycl::trunc(x);
    }
}

template<typename T, typename F>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_flat_kernel(ଅଚଳ T * x, T * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]);
    }
}

template<typename T, typename F>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_generic_kernel(
        ଅଚଳ T * x,
        T * dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k,
        ଅଚଳ sycl::uint3 ne0_fd, ଅଚଳ sycl::uint3 ne1_fd, ଅଚଳ sycl::uint3 ne2_fd,
        ଅଚଳ size_t nb0,  ଅଚଳ size_t nb1,  ଅଚଳ size_t nb2,  ଅଚଳ size_t nb3,
        ଅଚଳ size_t nbd0, ଅଚଳ size_t nbd1, ଅଚଳ size_t nbd2, ଅଚଳ size_t nbd3,
        ଅଚଳ sycl::nd_item<1> & item_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        ଅଚଳ uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        ଅଚଳ uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        ଅଚଳ uint32_t i2 = dm.y();
        ଅଚଳ uint32_t i3 = dm.x();

        ଅଚଳ ଅକ୍ଷର * src_base = (ଅଚଳ ଅକ୍ଷର *) x;
        ଅକ୍ଷର       * dst_base = (ଅକ୍ଷର *) dst;

        ଅଚଳ T * srcp = (ଅଚଳ T *)(src_base + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_base + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

template<typename T>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_sqrt_kernel(ଅଚଳ T * x, T * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

template<typename T>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_sin_kernel(ଅଚଳ T * x, T * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

template<typename T>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_cos_kernel(ଅଚଳ T * x, T * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

template<typename T>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_log_kernel(ଅଚଳ T * x, T * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_log(x[i]);
    }
}


template<typename T>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_leaky_relu_kernel(ଅଚଳ T * x, T * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଦଶମିକ negative_slope, ଅଚଳ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_slope);
    }
}

template<typename T>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_xielu_kernel(ଅଚଳ T * x, T * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଦଶମିକ alpha_n, ଦଶମିକ alpha_p, ଦଶମିକ beta, ଦଶମିକ eps, ଅଚଳ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

template<typename T>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_sqr_kernel(ଅଚଳ T * x, T * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

template<typename T>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_clamp_kernel(ଅଚଳ T * x, T * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ sycl::nd_item<1> &item_ct1, ଦଶମିକ min_val, ଦଶମିକ max_val) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

template<typename T>
ସ୍ଥିର ଶୂନ୍ୟ unary_op_ceil_kernel(ଅଚଳ T * x, T * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

template<typename T>
ସ୍ଥିର ଶୂନ୍ୟ clamp(ଅଚଳ T * x, T * dst, ଅଚଳ ଦଶମିକ min, ଅଚଳ ଦଶମିକ max, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k,
                      ଅଚଳ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

template<typename T, typename F>
ସ୍ଥିର ଶୂନ୍ୟ unary_gated_op_flat_kernel(ଅଚଳ T * x, ଅଚଳ T * g, T * dst, ଅଚଳ uint64_t k, ଅଚଳ sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

template<typename T, typename F>
ସ୍ଥିର ଶୂନ୍ୟ unary_gated_op_generic_kernel(
        ଅଚଳ T * x,
        ଅଚଳ T * g,
        T * dst,
        ଅଚଳ uint64_t k,
        ଅଚଳ sycl::uint3 n_fd,
        ଅଚଳ uint64_t o0,
        ଅଚଳ uint64_t o1,
        ଅଚଳ sycl::nd_item<1> & item_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ଅଚଳ sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ଅଚଳ int64_t j0 = rc.x() * o0 + rc.y();
        ଅଚଳ int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
template<typename T, typename F>
ସ୍ଥିର ଶୂନ୍ୟ unary_mul_flat_kernel(ଅଚଳ T * x, ଅଚଳ T * g, T * dst, ଅଚଳ int64_t k, ଅଚଳ sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = (T) (op((ଦଶମିକ) x[i]) * (ଦଶମିକ) g[i]);
    }
}

template<typename T, typename F>
ସ୍ଥିର ଶୂନ୍ୟ unary_mul_strided_kernel(ଅଚଳ T * x, ଅଚଳ T * g, T * dst, ଅଚଳ int64_t k, ଅଚଳ sycl::uint3 n_fd, ଅଚଳ int64_t o0, ଅଚଳ int64_t o1, ଅଚଳ sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ଅଚଳ sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ଅଚଳ int64_t j0 = rc.x() * o0 + rc.y();
        ଅଚଳ int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((ଦଶମିକ) x[j0]) * (ଦଶମିକ) g[j1]);
    }
}

template<typename T, typename F>
ସ୍ଥିର ଶୂନ୍ୟ unary_mul_sycl(ଅଚଳ T * x, ଅଚଳ T * g, T * dst, ଅଚଳ int64_t k, ଅଚଳ int64_t n, ଅଚଳ int64_t o0, ଅଚଳ int64_t o1, queue_ptr main_stream, F op) {
    ଅଚଳ size_t            ଅଙ୍କ_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    ଅଚଳ sycl::nd_range<1> range(ଅଙ୍କ_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    ଯଦି (o0 == n && o1 == n) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            unary_mul_flat_kernel(x, g, dst, k, item_ct1, op);
        });
        ଫେରନ୍ତୁ;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ଅଚଳ sycl::uint3 n_fd = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) n);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, item_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
template<typename F>
ସ୍ଥିର ଶୂନ୍ୟ add_unary_mul_flat_kernel(ଅଚଳ ଦଶମିକ * a, ଅଚଳ ଦଶମିକ * bias, ଅଚଳ ଦଶମିକ * scale, ଦଶମିକ * dst,
                                      ଅଚଳ int64_t k, ଅଚଳ sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

template<typename F>
ସ୍ଥିର ଶୂନ୍ୟ add_unary_mul_bcast_kernel(ଅଚଳ ଦଶମିକ * a, ଅଚଳ ଦଶମିକ * bias, ଅଚଳ ଦଶମିକ * scale, ଦଶମିକ * dst,
                                       ଅଚଳ int64_t k, ଅଚଳ sycl::uint3 ne0_fd, ଅଚଳ sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ଅଚଳ uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

template<typename F>
ସ୍ଥିର ଶୂନ୍ୟ add_unary_mul_sycl(ଅଚଳ ଦଶମିକ * a, ଅଚଳ ଦଶମିକ * bias, ଅଚଳ ଦଶମିକ * scale, ଦଶମିକ * dst,
                               ଅଚଳ int64_t k, ଅଚଳ int64_t ne0, queue_ptr main_stream, F op) {
    ଅଚଳ size_t            ଅଙ୍କ_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    ଅଚଳ sycl::nd_range<1> range(ଅଙ୍କ_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    ଯଦି (k == ne0) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            add_unary_mul_flat_kernel(a, bias, scale, dst, k, item_ct1, op);
        });
        ଫେରନ୍ତୁ;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ଅଚଳ sycl::uint3 ne0_fd = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) ne0);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        add_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, item_ct1, op);
    });
}

ନାମାକାଶ ggml_sycl_detail {
ସ୍ଥିର ଶୂନ୍ୟ acc_f32_sycl(ଅଚଳ ଅକ୍ଷର *x, ଅଚଳ ଅକ୍ଷର *y, ଦଶମିକ *dst,
                         ଅଚଳ int64_t n_elements,
                         ଅଚଳ int64_t ne0, ଅଚଳ int64_t ne1, ଅଚଳ int64_t ne2,
                         ଅଚଳ int64_t nb00, ଅଚଳ int64_t nb01, ଅଚଳ int64_t nb02, ଅଚଳ int64_t nb03,
                         ଅଚଳ int64_t ne10, ଅଚଳ int64_t ne11, ଅଚଳ int64_t ne12, ଅଚଳ int64_t ne13,
                         ଅଚଳ int64_t nb10, ଅଚଳ int64_t nb11, ଅଚଳ int64_t nb12, ଅଚଳ int64_t nb13,
                         ଅଚଳ int64_t s1, ଅଚଳ int64_t s2, ଅଚଳ int64_t s3,
                         ଅଚଳ int64_t offset, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
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
ସ୍ଥିର ଶୂନ୍ୟ arange_kernel(T * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, T start, T step,
                         ଅଚଳ sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = start + static_cast<T>(i) * step;
    }
}

template<typename KernelInvoker, typename... Args>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ବାଛ (dst->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                ଭାଙ୍ଗ;
            }
#ifdef GGML_SYCL_HAS_BF16
        ସ୍ଥିତି GGML_TYPE_BF16:
            {
                auto data_pts = cast_data<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                ଭାଙ୍ଗ;
            }
#endif
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                auto data_pts = cast_data<ଦଶମିକ>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                ଭାଙ୍ଗ;
            }
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename KernelInvoker, typename... Args>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଅଚଳ int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // The fused GLU kernels index with 32-bit fastdiv, which is exact only for indices below
    // 2^31. A dst that large is ~8 GB at f32, and the grid sizing already narrows to 32 bits,
    // so assert the bound rather than carry a second code path for it.
    GGML_ASSERT(ggml_nelements(dst) < ((int64_t) 1 << 31));
    ଅଚଳ int32_t swapped = ((ଅଚଳ int32_t *) dst->op_params)[1];
    ଶୂନ୍ୟ * src0_d = src0->data;
    ଶୂନ୍ୟ * src1_d = src1 ? src1->data : src0->data;
    ଅଚଳ int64_t src0_o = src0->nb[1];
    ଅଚଳ int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ଶୂନ୍ୟ * dst_d = dst->data;
    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    ବାଛ (dst->type) {
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    ଯଦି (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / ଆକାର(sycl::half),
                               src1_o / ଆକାର(sycl::half),
                               main_stream,
                               std::forward<Args>(args)...);
                ଭାଙ୍ଗ;
            }
        ସ୍ଥିତି GGML_TYPE_F32:
            {
                ଦଶମିକ * src0_p = (ଦଶମିକ *) src0_d;
                ଦଶମିକ * src1_p = (ଦଶମିକ *) src1_d;

                    ଯଦି (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (ଦଶମିକ *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / ଆକାର(ଦଶମିକ),
                               src1_o / ଆକାର(ଦଶମିକ),
                               main_stream,
                               std::forward<Args>(args)...);
                ଭାଙ୍ଗ;
            }
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename F>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    ଅଚଳ int64_t ne0  = dst->ne[0];
    ଅଚଳ int64_t ne1  = dst->ne[1];
    ଅଚଳ int64_t ne2  = dst->ne[2];

    ଅଚଳ size_t  nb0  = src0->nb[0];
    ଅଚଳ size_t  nb1  = src0->nb[1];
    ଅଚଳ size_t  nb2  = src0->nb[2];
    ଅଚଳ size_t  nb3  = src0->nb[3];

    ଅଚଳ size_t  nbd0 = dst->nb[0];
    ଅଚଳ size_t  nbd1 = dst->nb[1];
    ଅଚଳ size_t  nbd2 = dst->nb[2];
    ଅଚଳ size_t  nbd3 = dst->nb[3];

    // Hot unary ops (FFN/GDN silu, sigmoid, ...) run on contiguous tensors;
    // skip the strided index math entirely for them.
    ଅଚଳ ତାର୍କିକ contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](ଅଚଳ auto* src, auto* dst_ptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_elements, queue_ptr stream) {

            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(k_elements, 256);

            ଯଦି (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(ଅଙ୍କ_blocks) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, item_ct1, func);
                    });
            } ଅନ୍ୟଥା {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                ଅଚଳ sycl::uint3 ne0_fd = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) ne0);
                ଅଚଳ sycl::uint3 ne1_fd = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) ne1);
                ଅଚଳ sycl::uint3 ne2_fd = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(ଅଙ୍କ_blocks) * sycl::range<1>(256),
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
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](ଅଚଳ auto * x_ptr, ଅଚଳ auto * g_ptr, auto * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {

            ଅଚଳ uint32_t ଅଙ୍କ_blocks = (uint32_t) ceil_div(k, SYCL_GLU_BLOCK_SIZE);
            ଅଚଳ sycl::nd_range<1> launch_range(ଅଙ୍କ_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE),
                                                 sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            ଯଦି (o0 == n && o1 == n) {
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, item_ct1, func);
                    });
            } ଅନ୍ୟଥା {
                // launch-invariant divisor, and only this path needs it
                ଅଚଳ sycl::uint3 n_fd = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) n);
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, item_ct1, func);
                    });
            }
        });
}


ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    ଦଶମିକ start, stop, step;
    memcpy(&start, dst->op_params, ଆକାର(ଦଶମିକ));
    memcpy(&stop, (ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));
    memcpy(&step, (ଦଶମିକ *) dst->op_params + 2, ଆକାର(ଦଶମିକ));
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ଦଶମିକ * dst_ptr = (ଦଶମିକ *)dst->data;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ggml_nelements(dst);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(k, SYCL_ARANGE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ଅଙ୍କ_blocks) * sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE),
                          sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            arange_kernel(dst_ptr, k, start, step, item_ct1);
        });
}

} // namespace ggml_sycl_detail



ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_sgn(x);
    });
}


ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_abs(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_elu(x);
    });
}
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_silu(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_gelu(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_gelu_quick(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_gelu_erf(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_tanh(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_relu(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_hardsigmoid(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_hardswish(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_exp(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_expm1(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ଅଚଳ auto* src, auto* dst_ptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_elements, queue_ptr stream) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ଅଙ୍କ_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_softplus(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_neg(x);
    });
}


ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_step(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_sigmoid(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ଅଚଳ auto* src, auto* dst_ptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_elements, queue_ptr stream) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(k_elements, SYCL_SQRT_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ଅଙ୍କ_blocks) * sycl::range<1>(SYCL_SQRT_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQRT_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ଅଚଳ auto* src, auto* dst_ptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_elements, queue_ptr stream) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ଅଙ୍କ_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ଅଚଳ auto* src, auto* dst_ptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_elements, queue_ptr stream) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ଅଙ୍କ_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଦଶମିକ negative_slope;
    memcpy(&negative_slope, dst->op_params, ଆକାର(ଦଶମିକ));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ଅଚଳ auto* src, auto* dst_ptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_elements, queue_ptr stream, ଦଶମିକ slope) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ଅଙ୍କ_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, slope, item_ct1);
                });
        }, negative_slope);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ଅଚଳ auto* src, auto* dst_ptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_elements, queue_ptr stream) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(k_elements, SYCL_SQR_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ଅଙ୍କ_blocks) * sycl::range<1>(SYCL_SQR_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQR_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଦଶମିକ min_val;
    ଦଶମିକ max_val;
    memcpy(&min_val, dst->op_params, ଆକାର(ଦଶମିକ));
    memcpy(&max_val, (ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ଅଚଳ auto* src, auto* dst_ptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_elements, queue_ptr stream, ଦଶମିକ min_arg, ଦଶମିକ max_arg) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(k_elements, SYCL_CLAMP_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ଅଙ୍କ_blocks) * sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, item_ct1);
                });
        }, min_val, max_val);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ଦଶମିକ alpha_n = ggml_get_op_params_f32(dst, 1);
    ଅଚଳ ଦଶମିକ alpha_p = ggml_get_op_params_f32(dst, 2);
    ଅଚଳ ଦଶମିକ beta    = ggml_get_op_params_f32(dst, 3);
    ଅଚଳ ଦଶମିକ eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ଅଚଳ auto* src, auto* dst_ptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_elements, queue_ptr stream, ଦଶମିକ alpha_n_arg, ଦଶମିକ alpha_p_arg, ଦଶମିକ beta_arg, ଦଶମିକ eps_arg) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(ଅଙ୍କ_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, item_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_floor(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_ceil(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_round(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_trunc(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    ଅଚଳ ଅକ୍ଷର  * src0_d = (ଅଚଳ ଅକ୍ଷର  *) src0->data;
    ଅଚଳ ଅକ୍ଷର  * src1_d = (ଅଚଳ ଅକ୍ଷର  *) src1->data;
    ଦଶମିକ       * dst_d  = (ଦଶମିକ       *)  dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ଅଚଳ int64_t s1     = (int64_t) ((ଅଚଳ int32_t *) dst->op_params)[0] / (int64_t) ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t s2     = (int64_t) ((ଅଚଳ int32_t *) dst->op_params)[1] / (int64_t) ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t s3     = (int64_t) ((ଅଚଳ int32_t *) dst->op_params)[2] / (int64_t) ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t offset = (int64_t) ((ଅଚଳ int32_t *) dst->op_params)[3] / (int64_t) ଆକାର(ଦଶମିକ);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, offset, stream);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_gelu(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_relu(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
template<typename F>
ସ୍ଥିର ଶୂନ୍ୟ dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
    ବାଛ (uop) {
        ସ୍ଥିତି GGML_UNARY_OP_SILU:     launch([](ଦଶମିକ v) { ଫେରନ୍ତୁ op_silu(v); });     ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_SIGMOID:  launch([](ଦଶମିକ v) { ଫେରନ୍ତୁ op_sigmoid(v); });  ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_UNARY_OP_SOFTPLUS: launch([](ଦଶମିକ v) { ଫେରନ୍ତୁ op_softplus(v); }); ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
    }
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
ଶୂନ୍ୟ ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_ଗଣ୍ଠି, ggml_tensor * mul_ଗଣ୍ଠି) {
    scope_op_debug_print scope_dbg_print(__func__, mul_ଗଣ୍ଠି, /*num_src=*/2);

    ଅଚଳ ggml_tensor * x = unary_ଗଣ୍ଠି->src[0];
    ଅଚଳ ggml_tensor * g = (mul_ଗଣ୍ଠି->src[0] == unary_ଗଣ୍ଠି) ? mul_ଗଣ୍ଠି->src[1] : mul_ଗଣ୍ଠି->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_ଗଣ୍ଠି);
    GGML_ASSERT(x->type == g->type && x->type == mul_ଗଣ୍ଠି->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_ଗଣ୍ଠି));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_ଗଣ୍ଠି));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ଅଚଳ int64_t k = ggml_nelements(mul_ଗଣ୍ଠି);
    ଅଚଳ int64_t n = mul_ଗଣ୍ଠି->ne[0];

    ଅଚଳ auto dispatch_type = [&](auto op) {
        ବାଛ (mul_ଗଣ୍ଠି->type) {
            ସ୍ଥିତି GGML_TYPE_F32:
                unary_mul_sycl((ଅଚଳ ଦଶମିକ *) x->data, (ଅଚଳ ଦଶମିକ *) g->data, (ଦଶମିକ *) mul_ଗଣ୍ଠି->data,
                               k, n, x->nb[1] / ଆକାର(ଦଶମିକ), g->nb[1] / ଆକାର(ଦଶମିକ), main_stream, op);
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TYPE_F16:
                unary_mul_sycl((ଅଚଳ sycl::half *) x->data, (ଅଚଳ sycl::half *) g->data, (sycl::half *) mul_ଗଣ୍ଠି->data,
                               k, n, x->nb[1] / ଆକାର(sycl::half), g->nb[1] / ଆକାର(sycl::half), main_stream, op);
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_ଗଣ୍ଠି->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_ଗଣ୍ଠି), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
ଶୂନ୍ୟ ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_ଗଣ୍ଠି,
                                      ggml_tensor * unary_ଗଣ୍ଠି, ggml_tensor * mul_ଗଣ୍ଠି) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_ଗଣ୍ଠି, /*num_src=*/2);

    ଅଚଳ ggml_tensor * a     = add_ଗଣ୍ଠି->src[0];
    ଅଚଳ ggml_tensor * bias  = add_ଗଣ୍ଠି->src[1];
    ଅଚଳ ggml_tensor * scale = (mul_ଗଣ୍ଠି->src[0] == unary_ଗଣ୍ଠି) ? mul_ଗଣ୍ଠି->src[1] : mul_ଗଣ୍ଠି->src[0];

    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(scale != unary_ଗଣ୍ଠି);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_ଗଣ୍ଠି->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_ଗଣ୍ଠି));
    // a and dst are indexed flat
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_ଗଣ୍ଠି));
    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ଅଚଳ auto dispatch_op = [&](auto op) {
        add_unary_mul_sycl((ଅଚଳ ଦଶମିକ *) a->data, (ଅଚଳ ଦଶମିକ *) bias->data, (ଅଚଳ ଦଶମିକ *) scale->data,
                           (ଦଶମିକ *) mul_ଗଣ୍ଠି->data, ggml_nelements(mul_ଗଣ୍ଠି), mul_ଗଣ୍ଠି->ne[0], main_stream, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_ଗଣ୍ଠି), dispatch_op);
}

__dpct_inline__ ଦଶମିକ ggml_sycl_op_swiglu_oai_single(ଦଶମିକ x, ଦଶମିକ g, ଦଶମିକ alpha = 1.702f, ଦଶମିକ limit = 7.0f) {
    x = sycl::fmin(x, limit);
    g = sycl::fmax(sycl::fmin(g, limit), -limit);

    ଦଶମିକ out_glu = x / (1.0f + sycl::native::exp(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    ଫେରନ୍ତୁ out_glu;
}

template <typename T>
ସ୍ଥିର ଶୂନ୍ୟ swiglu_oai_kernel(ଅଚଳ T * x, ଅଚଳ T * g, T * dst, ଅଚଳ int64_t k,
                              ଅଚଳ int64_t n, ଅଚଳ int64_t o0, ଅଚଳ int64_t o1,
                              ଦଶମିକ alpha, ଦଶମିକ limit, sycl::nd_item<3> item_ct1) {
    ଅଚଳ int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    ଯଦି (i >= k) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int64_t j0 = (i / n) * o0 + (i % n);
    ଅଚଳ int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ଦଶମିକ xi = x[j0];
    ଦଶମିକ gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
ସ୍ଥିର ଶୂନ୍ୟ swiglu_oai_sycl(ଅଚଳ T *       x,
                            ଅଚଳ T *       g,
                            T *             dst,
                            ଅଚଳ int64_t   k,
                            ଅଚଳ int64_t   n,
                            ଅଚଳ int64_t   o0,
                            ଅଚଳ int64_t   o1,
                            ଅଚଳ ଦଶମିକ     alpha,
                            ଅଚଳ ଦଶମିକ     limit,
                            dpct::queue_ptr stream) {
    ଅଚଳ int64_t ଅଙ୍କ_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, limit, item_ct1);
                         });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];
    ଶୂନ୍ୟ * src0_d = src0->data;
    ଶୂନ୍ୟ * src1_d = src1 ? src1->data : src0->data;
    ଅଚଳ int64_t src0_o = src0->nb[1];
    ଅଚଳ int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ଶୂନ୍ୟ * dst_d = dst->data;
    ଅଚଳ int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ଅଚଳ ଦଶମିକ alpha = ggml_get_op_params_f32(dst, 2);
    ଅଚଳ ଦଶମିକ limit = ggml_get_op_params_f32(dst, 3);

    ଦଶମିକ * src0_p = (ଦଶମିକ *) src0_d;
    ଦଶମିକ * src1_p = (ଦଶମିକ *) src1_d;

    ଯଦି (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (ଦଶମିକ *)dst_d, ggml_nelements(dst), nc, src0_o / ଆକାର(ଦଶମିକ), src1_o / ଆକାର(ଦଶମିକ), alpha, limit, stream);
}

template <typename T>
ସ୍ଥିର ଶୂନ୍ୟ swiglu_clamp_kernel(ଅଚଳ T *        gate,
                                ଅଚଳ T *        up,
                                T *              dst,
                                ଅଚଳ int64_t    k,
                                ଅଚଳ int64_t    n,
                                ଅଚଳ int64_t    o0,
                                ଅଚଳ int64_t    o1,
                                ଦଶମିକ            limit,
                                sycl::nd_item<3> item_ct1) {
    ଅଚଳ int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    ଯଦି (i >= k) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int64_t j0 = (i / n) * o0 + (i % n);
    ଅଚଳ int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ଅଚଳ ଦଶମିକ gate_ମୂଲ୍ୟ = sycl::fmin((ଦଶମିକ) gate[j0], limit);
    ଅଚଳ ଦଶମିକ up_ମୂଲ୍ୟ   = sycl::fmax(sycl::fmin((ଦଶମିକ) up[j1], limit), -limit);
    dst[i]                 = (T) (gate_ମୂଲ୍ୟ / (1.0f + sycl::native::exp(-gate_ମୂଲ୍ୟ)) * up_ମୂଲ୍ୟ);
}

template <typename T>
ସ୍ଥିର ଶୂନ୍ୟ swiglu_clamp_sycl(ଅଚଳ T *       gate,
                              ଅଚଳ T *       up,
                              T *             dst,
                              ଅଚଳ int64_t   k,
                              ଅଚଳ int64_t   n,
                              ଅଚଳ int64_t   o0,
                              ଅଚଳ int64_t   o1,
                              ଦଶମିକ           limit,
                              dpct::queue_ptr stream) {
    ଅଚଳ int64_t ଅଙ୍କ_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_clamp_kernel(gate, up, dst, k, n, o0, o1, limit, item_ct1);
                         });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0   = dst->src[0];
    ଅଚଳ ggml_tensor * src1   = dst->src[1];
    ଶୂନ୍ୟ *              src0_d = src0->data;
    ଶୂନ୍ୟ *              src1_d = src1 ? src1->data : src0->data;
    ଅଚଳ int64_t       src0_o = src0->nb[1];
    ଅଚଳ int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ଶୂନ୍ୟ *              dst_d  = dst->data;
    ଅଚଳ int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    ଯଦି (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    ଅଚଳ int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ଅଚଳ ଦଶମିକ   limit   = ggml_get_op_params_f32(dst, 3);

    ଯଦି (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / ଆକାର(sycl::half),
                          src1_o / ଆକାର(sycl::half), limit, stream);
    } ଅନ୍ୟଥା {
        ଦଶମିକ * src0_p = (ଦଶମିକ *) src0_d;
        ଦଶମିକ * src1_p = (ଦଶମିକ *) src1_d;

        ଯଦି (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (ଦଶମିକ *) dst_d, ggml_nelements(dst), nc, src0_o / ଆକାର(ଦଶମିକ),
                          src1_o / ଆକାର(ଦଶମିକ), limit, stream);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_gelu_erf(x);
    });
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        ଫେରନ୍ତୁ op_gelu_quick(x);
    });
}


ଶୂନ୍ୟ ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_quick(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_expm1(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_step(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_xielu(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_clamp(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_quick(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_round(ctx, dst);
}

ଶୂନ୍ୟ ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
