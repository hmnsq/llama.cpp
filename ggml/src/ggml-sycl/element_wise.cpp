#अन्तर्भाव "common.hpp"
#अन्तर्भाव "ggml-sycl/presets.hpp"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "element_wise.hpp"

#define SYCL_GLOBAL_ID_LOOP(K, ITEM) \
    अत्थाय (auto i = ITEM.get_global_id(0); i < (size_t)K; i += ITEM.get_global_range(0))

#define SYCL_LOCAL_ID_CALC(ITEM, IDX) \
    (ITEM.get_local_range(IDX) * ITEM.get_group(IDX) + ITEM.get_local_id(IDX))

स्थिर शून्य acc_f32(ध्रुव वर्ण * x, ध्रुव वर्ण * y, भिन्न * dst, ध्रुव int64_t ne,
    ध्रुव int64_t ne0, ध्रुव int64_t ne1, ध्रुव int64_t ne2,
    ध्रुव int64_t nb00, ध्रुव int64_t nb01, ध्रुव int64_t nb02, ध्रुव int64_t nb03,
    ध्रुव int64_t ne10, ध्रुव int64_t ne11, ध्रुव int64_t ne12, ध्रुव int64_t ne13,
    ध्रुव int64_t nb10, ध्रुव int64_t nb11, ध्रुव int64_t nb12, ध्रुव int64_t nb13,
    ध्रुव int64_t s11, ध्रुव int64_t s12, ध्रुव int64_t s13, ध्रुव int64_t offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव int64_t i = SYCL_LOCAL_ID_CALC(item_ct1, 2);

    सचे (i >= ne) {
        पटिनिवत्त;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    ध्रुव int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    ध्रुव int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    ध्रुव int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    ध्रुव int64_t i10 = tmp;

    int64_t tmp_dst = i;
    ध्रुव int64_t i3 = tmp_dst / (ne2*ne1*ne0);
    tmp_dst -= i3 * (ne2*ne1*ne0);
    ध्रुव int64_t i2 = tmp_dst / (ne1*ne0);
    tmp_dst -= i2 * (ne1*ne0);
    ध्रुव int64_t i1 = tmp_dst / ne0;
    tmp_dst -= i1 * ne0;
    ध्रुव int64_t i0 = tmp_dst;

    भिन्न val = *(ध्रुव भिन्न *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    सचे (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(ध्रुव भिन्न *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
template<typename T>
स्थिर __dpct_inline__ T op_sgn(T x) {
    पटिनिवत्त x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


template<typename T>
स्थिर __dpct_inline__ T op_abs(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } अञ्ञथा {
        पटिनिवत्त sycl::fabs(x);
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_expm1(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<भिन्न>(x))
        );
    } अञ्ञथा {
        पटिनिवत्त sycl::expm1(x);
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_elu(T x) {
    पटिनिवत्त (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

template<typename T>
स्थिर __dpct_inline__ T op_erf(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<भिन्न>(x))
        );
    } अञ्ञथा {
        पटिनिवत्त sycl::erf(x);
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_gelu_erf(T x) {
    ध्रुव T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    पटिनिवत्त static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

template<typename T>
स्थिर __dpct_inline__ T op_gelu_quick(T x) {
    ध्रुव T GELU_QUICK_COEF_LOCAL = static_cast<T>(-1.702f);
    पटिनिवत्त x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_QUICK_COEF_LOCAL * x)));
}

template<typename T>
स्थिर __dpct_inline__ T op_relu(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } अञ्ञथा {
        पटिनिवत्त sycl::fmax(x, static_cast<T>(0));
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_sigmoid(T x) {
    पटिनिवत्त static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

template<typename T>
स्थिर __dpct_inline__ T op_sqrt(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त sycl::ext::oneapi::experimental::sqrt(x);
    } अञ्ञथा {
        पटिनिवत्त sycl::sqrt(x);
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_sin(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त sycl::ext::oneapi::experimental::sin(x);
    } अञ्ञथा {
        पटिनिवत्त sycl::sin(x);
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_cos(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त sycl::ext::oneapi::experimental::cos(x);
    } अञ्ञथा {
        पटिनिवत्त sycl::cos(x);
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_hardsigmoid(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } अञ्ञथा {
        पटिनिवत्त sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_hardswish(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } अञ्ञथा {
        पटिनिवत्त x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_log(T x) {
    सचे (x <= static_cast<T>(0)) {
        पटिनिवत्त neg_infinity<T>();
    }
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त sycl::ext::oneapi::experimental::log(x);
    } अञ्ञथा {
        पटिनिवत्त sycl::log(x);
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_softplus(T x) {
    ध्रुव भिन्न xf = (भिन्न) x;
    ध्रुव भिन्न ax = op_abs(xf);
    ध्रुव भिन्न m  = sycl::fmax(xf, 0.0f);
    ध्रुव भिन्न y  = m + sycl::log1p(sycl::exp(-ax));
    पटिनिवत्त (T) y;
}

template<typename T>
स्थिर __dpct_inline__ T op_neg(T x) {
    पटिनिवत्त -x;
}

template<typename T>
स्थिर __dpct_inline__ T op_step(T x) {
    पटिनिवत्त (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

template<typename T>
स्थिर __dpct_inline__ T op_leaky_relu(T x, भिन्न negative_slope) {
    T neg_slope_T = static_cast<T>(negative_slope);
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;

    } अञ्ञथा {
        पटिनिवत्त sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_xielu(T x, भिन्न alpha_n, भिन्न alpha_p, भिन्न beta, भिन्न eps) {
    ध्रुव भिन्न xi        = static_cast<भिन्न>(x);
    ध्रुव भिन्न gate_pos  = (xi > 0.0f);
    ध्रुव भिन्न y_pos     = alpha_p * xi * xi + beta * xi;
    ध्रुव भिन्न min_v_eps = sycl::fmin(xi, eps);
    ध्रुव भिन्न y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    ध्रुव भिन्न out       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    पटिनिवत्त static_cast<T>(out);
}

template<typename T>
स्थिर __dpct_inline__ T op_sqr(T x) {
    पटिनिवत्त x * x;
}

template<typename T>
स्थिर __dpct_inline__ T op_clamp(T x, भिन्न min_val, भिन्न max_val) {
    पटिनिवत्त x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

template<typename T>
स्थिर __dpct_inline__ T op_floor(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त sycl::ext::oneapi::experimental::floor(x);
    } अञ्ञथा {
        पटिनिवत्त sycl::floor(x);
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_ceil(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त sycl::ext::oneapi::experimental::ceil(x);
    } अञ्ञथा {
        पटिनिवत्त sycl::ceil(x);
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_round(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<भिन्न>(x))
        );
    } अञ्ञथा {
        पटिनिवत्त sycl::round(x);
    }
}

template<typename T>
स्थिर __dpct_inline__ T op_trunc(T x) {
    सचे constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        पटिनिवत्त sycl::ext::oneapi::experimental::trunc(x);
    } अञ्ञथा {
        पटिनिवत्त sycl::trunc(x);
    }
}

template<typename T, typename F>
स्थिर शून्य unary_op_flat_kernel(ध्रुव T * x, T * dst, ध्रुव पूर्णाङ्क k, ध्रुव sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]);
    }
}

template<typename T, typename F>
स्थिर शून्य unary_op_generic_kernel(
        ध्रुव T * x,
        T * dst,
        ध्रुव पूर्णाङ्क k,
        ध्रुव sycl::uint3 ne0_fd, ध्रुव sycl::uint3 ne1_fd, ध्रुव sycl::uint3 ne2_fd,
        ध्रुव size_t nb0,  ध्रुव size_t nb1,  ध्रुव size_t nb2,  ध्रुव size_t nb3,
        ध्रुव size_t nbd0, ध्रुव size_t nbd1, ध्रुव size_t nbd2, ध्रुव size_t nbd3,
        ध्रुव sycl::nd_item<1> & item_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        ध्रुव uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        ध्रुव uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        ध्रुव uint32_t i2 = dm.y();
        ध्रुव uint32_t i3 = dm.x();

        ध्रुव वर्ण * src_base = (ध्रुव वर्ण *) x;
        वर्ण       * dst_base = (वर्ण *) dst;

        ध्रुव T * srcp = (ध्रुव T *)(src_base + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_base + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

template<typename T>
स्थिर शून्य unary_op_sqrt_kernel(ध्रुव T * x, T * dst, ध्रुव पूर्णाङ्क k, ध्रुव sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

template<typename T>
स्थिर शून्य unary_op_sin_kernel(ध्रुव T * x, T * dst, ध्रुव पूर्णाङ्क k, ध्रुव sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

template<typename T>
स्थिर शून्य unary_op_cos_kernel(ध्रुव T * x, T * dst, ध्रुव पूर्णाङ्क k, ध्रुव sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

template<typename T>
स्थिर शून्य unary_op_log_kernel(ध्रुव T * x, T * dst, ध्रुव पूर्णाङ्क k, ध्रुव sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_log(x[i]);
    }
}


template<typename T>
स्थिर शून्य unary_op_leaky_relu_kernel(ध्रुव T * x, T * dst, ध्रुव पूर्णाङ्क k, भिन्न negative_slope, ध्रुव sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_slope);
    }
}

template<typename T>
स्थिर शून्य unary_op_xielu_kernel(ध्रुव T * x, T * dst, ध्रुव पूर्णाङ्क k, भिन्न alpha_n, भिन्न alpha_p, भिन्न beta, भिन्न eps, ध्रुव sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

template<typename T>
स्थिर शून्य unary_op_sqr_kernel(ध्रुव T * x, T * dst, ध्रुव पूर्णाङ्क k, ध्रुव sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

template<typename T>
स्थिर शून्य unary_op_clamp_kernel(ध्रुव T * x, T * dst, ध्रुव पूर्णाङ्क k, ध्रुव sycl::nd_item<1> &item_ct1, भिन्न min_val, भिन्न max_val) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

template<typename T>
स्थिर शून्य unary_op_ceil_kernel(ध्रुव T * x, T * dst, ध्रुव पूर्णाङ्क k, ध्रुव sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

template<typename T>
स्थिर शून्य clamp(ध्रुव T * x, T * dst, ध्रुव भिन्न min, ध्रुव भिन्न max, ध्रुव पूर्णाङ्क k,
                      ध्रुव sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

template<typename T, typename F>
स्थिर शून्य unary_gated_op_flat_kernel(ध्रुव T * x, ध्रुव T * g, T * dst, ध्रुव uint64_t k, ध्रुव sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

template<typename T, typename F>
स्थिर शून्य unary_gated_op_generic_kernel(
        ध्रुव T * x,
        ध्रुव T * g,
        T * dst,
        ध्रुव uint64_t k,
        ध्रुव sycl::uint3 n_fd,
        ध्रुव uint64_t o0,
        ध्रुव uint64_t o1,
        ध्रुव sycl::nd_item<1> & item_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ध्रुव sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ध्रुव int64_t j0 = rc.x() * o0 + rc.y();
        ध्रुव int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
template<typename T, typename F>
स्थिर शून्य unary_mul_flat_kernel(ध्रुव T * x, ध्रुव T * g, T * dst, ध्रुव int64_t k, ध्रुव sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = (T) (op((भिन्न) x[i]) * (भिन्न) g[i]);
    }
}

template<typename T, typename F>
स्थिर शून्य unary_mul_strided_kernel(ध्रुव T * x, ध्रुव T * g, T * dst, ध्रुव int64_t k, ध्रुव sycl::uint3 n_fd, ध्रुव int64_t o0, ध्रुव int64_t o1, ध्रुव sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ध्रुव sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ध्रुव int64_t j0 = rc.x() * o0 + rc.y();
        ध्रुव int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((भिन्न) x[j0]) * (भिन्न) g[j1]);
    }
}

template<typename T, typename F>
स्थिर शून्य unary_mul_sycl(ध्रुव T * x, ध्रुव T * g, T * dst, ध्रुव int64_t k, ध्रुव int64_t n, ध्रुव int64_t o0, ध्रुव int64_t o1, queue_ptr main_stream, F op) {
    ध्रुव size_t            अङ्कः_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    ध्रुव sycl::nd_range<1> range(अङ्कः_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    सचे (o0 == n && o1 == n) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            unary_mul_flat_kernel(x, g, dst, k, item_ct1, op);
        });
        पटिनिवत्त;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ध्रुव sycl::uint3 n_fd = init_fastdiv_मानानि((uint32_t) n);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, item_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
template<typename F>
स्थिर शून्य add_unary_mul_flat_kernel(ध्रुव भिन्न * a, ध्रुव भिन्न * bias, ध्रुव भिन्न * scale, भिन्न * dst,
                                      ध्रुव int64_t k, ध्रुव sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

template<typename F>
स्थिर शून्य add_unary_mul_bcast_kernel(ध्रुव भिन्न * a, ध्रुव भिन्न * bias, ध्रुव भिन्न * scale, भिन्न * dst,
                                       ध्रुव int64_t k, ध्रुव sycl::uint3 ne0_fd, ध्रुव sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ध्रुव uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

template<typename F>
स्थिर शून्य add_unary_mul_sycl(ध्रुव भिन्न * a, ध्रुव भिन्न * bias, ध्रुव भिन्न * scale, भिन्न * dst,
                               ध्रुव int64_t k, ध्रुव int64_t ne0, queue_ptr main_stream, F op) {
    ध्रुव size_t            अङ्कः_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    ध्रुव sycl::nd_range<1> range(अङ्कः_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    सचे (k == ne0) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            add_unary_mul_flat_kernel(a, bias, scale, dst, k, item_ct1, op);
        });
        पटिनिवत्त;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ध्रुव sycl::uint3 ne0_fd = init_fastdiv_मानानि((uint32_t) ne0);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        add_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, item_ct1, op);
    });
}

नामाकाश ggml_sycl_detail {
स्थिर शून्य acc_f32_sycl(ध्रुव वर्ण *x, ध्रुव वर्ण *y, भिन्न *dst,
                         ध्रुव int64_t n_elements,
                         ध्रुव int64_t ne0, ध्रुव int64_t ne1, ध्रुव int64_t ne2,
                         ध्रुव int64_t nb00, ध्रुव int64_t nb01, ध्रुव int64_t nb02, ध्रुव int64_t nb03,
                         ध्रुव int64_t ne10, ध्रुव int64_t ne11, ध्रुव int64_t ne12, ध्रुव int64_t ne13,
                         ध्रुव int64_t nb10, ध्रुव int64_t nb11, ध्रुव int64_t nb12, ध्रुव int64_t nb13,
                         ध्रुव int64_t s1, ध्रुव int64_t s2, ध्रुव int64_t s3,
                         ध्रुव int64_t offset, queue_ptr stream) {
    ध्रुव पूर्णाङ्क अङ्कः_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अङ्कः_blocks) * sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
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
स्थिर शून्य arange_kernel(T * dst, ध्रुव पूर्णाङ्क k, T start, T step,
                         ध्रुव sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = start + static_cast<T>(i) * step;
    }
}

template<typename KernelInvoker, typename... Args>
स्थिर अन्तःस्थ शून्य dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    विकल्प (dst->type) {
        पक्ष GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (पूर्णाङ्क)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                भङ्ग;
            }
#ifdef GGML_SYCL_HAS_BF16
        पक्ष GGML_TYPE_BF16:
            {
                auto data_pts = cast_data<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (पूर्णाङ्क)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                भङ्ग;
            }
#endif
        पक्ष GGML_TYPE_F32:
            {
                auto data_pts = cast_data<भिन्न>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (पूर्णाङ्क)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                भङ्ग;
            }
        स्वभाव:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename KernelInvoker, typename... Args>
स्थिर अन्तःस्थ शून्य dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ध्रुव int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // The fused GLU kernels index with 32-bit fastdiv, which is exact only for indices below
    // 2^31. A dst that large is ~8 GB at f32, and the grid sizing already narrows to 32 bits,
    // so assert the bound rather than carry a second code path for it.
    GGML_ASSERT(ggml_nelements(dst) < ((int64_t) 1 << 31));
    ध्रुव int32_t swapped = ((ध्रुव int32_t *) dst->op_params)[1];
    शून्य * src0_d = src0->data;
    शून्य * src1_d = src1 ? src1->data : src0->data;
    ध्रुव int64_t src0_o = src0->nb[1];
    ध्रुव int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    शून्य * dst_d = dst->data;
    सचे (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    विकल्प (dst->type) {
        पक्ष GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    सचे (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / परिमाण(sycl::half),
                               src1_o / परिमाण(sycl::half),
                               main_stream,
                               std::forward<Args>(args)...);
                भङ्ग;
            }
        पक्ष GGML_TYPE_F32:
            {
                भिन्न * src0_p = (भिन्न *) src0_d;
                भिन्न * src1_p = (भिन्न *) src1_d;

                    सचे (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (भिन्न *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / परिमाण(भिन्न),
                               src1_o / परिमाण(भिन्न),
                               main_stream,
                               std::forward<Args>(args)...);
                भङ्ग;
            }
        स्वभाव:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename F>
स्थिर अन्तःस्थ शून्य ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    ध्रुव int64_t ne0  = dst->ne[0];
    ध्रुव int64_t ne1  = dst->ne[1];
    ध्रुव int64_t ne2  = dst->ne[2];

    ध्रुव size_t  nb0  = src0->nb[0];
    ध्रुव size_t  nb1  = src0->nb[1];
    ध्रुव size_t  nb2  = src0->nb[2];
    ध्रुव size_t  nb3  = src0->nb[3];

    ध्रुव size_t  nbd0 = dst->nb[0];
    ध्रुव size_t  nbd1 = dst->nb[1];
    ध्रुव size_t  nbd2 = dst->nb[2];
    ध्रुव size_t  nbd3 = dst->nb[3];

    // Hot unary ops (FFN/GDN silu, sigmoid, ...) run on contiguous tensors;
    // skip the strided index math entirely for them.
    ध्रुव तार्किक contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](ध्रुव auto* src, auto* dst_ptr, पूर्णाङ्क k_elements, queue_ptr stream) {

            ध्रुव पूर्णाङ्क अङ्कः_blocks = ceil_div(k_elements, 256);

            सचे (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(अङ्कः_blocks) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, item_ct1, func);
                    });
            } अञ्ञथा {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                ध्रुव sycl::uint3 ne0_fd = init_fastdiv_मानानि((uint32_t) ne0);
                ध्रुव sycl::uint3 ne1_fd = init_fastdiv_मानानि((uint32_t) ne1);
                ध्रुव sycl::uint3 ne2_fd = init_fastdiv_मानानि((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(अङ्कः_blocks) * sycl::range<1>(256),
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
स्थिर अन्तःस्थ शून्य ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](ध्रुव auto * x_ptr, ध्रुव auto * g_ptr, auto * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {

            ध्रुव uint32_t अङ्कः_blocks = (uint32_t) ceil_div(k, SYCL_GLU_BLOCK_SIZE);
            ध्रुव sycl::nd_range<1> launch_range(अङ्कः_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE),
                                                 sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            सचे (o0 == n && o1 == n) {
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, item_ct1, func);
                    });
            } अञ्ञथा {
                // launch-invariant divisor, and only this path needs it
                ध्रुव sycl::uint3 n_fd = init_fastdiv_मानानि((uint32_t) n);
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, item_ct1, func);
                    });
            }
        });
}


स्थिर अन्तःस्थ शून्य ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    भिन्न start, stop, step;
    memcpy(&start, dst->op_params, परिमाण(भिन्न));
    memcpy(&stop, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));
    memcpy(&step, (भिन्न *) dst->op_params + 2, परिमाण(भिन्न));
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    भिन्न * dst_ptr = (भिन्न *)dst->data;
    ध्रुव पूर्णाङ्क k = (पूर्णाङ्क)ggml_nelements(dst);
    ध्रुव पूर्णाङ्क अङ्कः_blocks = ceil_div(k, SYCL_ARANGE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(अङ्कः_blocks) * sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE),
                          sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            arange_kernel(dst_ptr, k, start, step, item_ct1);
        });
}

} // namespace ggml_sycl_detail



स्थिर अन्तःस्थ शून्य ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_sgn(x);
    });
}


स्थिर अन्तःस्थ शून्य ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_abs(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_elu(x);
    });
}
स्थिर अन्तःस्थ शून्य ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_silu(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_gelu(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_gelu_quick(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_gelu_erf(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_tanh(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_relu(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_hardsigmoid(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_hardswish(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_exp(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_expm1(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ध्रुव auto* src, auto* dst_ptr, पूर्णाङ्क k_elements, queue_ptr stream) {
            ध्रुव पूर्णाङ्क अङ्कः_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अङ्कः_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_softplus(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_neg(x);
    });
}


स्थिर अन्तःस्थ शून्य ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_step(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_sigmoid(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ध्रुव auto* src, auto* dst_ptr, पूर्णाङ्क k_elements, queue_ptr stream) {
            ध्रुव पूर्णाङ्क अङ्कः_blocks = ceil_div(k_elements, SYCL_SQRT_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अङ्कः_blocks) * sycl::range<1>(SYCL_SQRT_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQRT_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ध्रुव auto* src, auto* dst_ptr, पूर्णाङ्क k_elements, queue_ptr stream) {
            ध्रुव पूर्णाङ्क अङ्कः_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अङ्कः_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ध्रुव auto* src, auto* dst_ptr, पूर्णाङ्क k_elements, queue_ptr stream) {
            ध्रुव पूर्णाङ्क अङ्कः_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अङ्कः_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    भिन्न negative_slope;
    memcpy(&negative_slope, dst->op_params, परिमाण(भिन्न));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ध्रुव auto* src, auto* dst_ptr, पूर्णाङ्क k_elements, queue_ptr stream, भिन्न slope) {
            ध्रुव पूर्णाङ्क अङ्कः_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अङ्कः_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, slope, item_ct1);
                });
        }, negative_slope);
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ध्रुव auto* src, auto* dst_ptr, पूर्णाङ्क k_elements, queue_ptr stream) {
            ध्रुव पूर्णाङ्क अङ्कः_blocks = ceil_div(k_elements, SYCL_SQR_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अङ्कः_blocks) * sycl::range<1>(SYCL_SQR_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQR_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    भिन्न min_val;
    भिन्न max_val;
    memcpy(&min_val, dst->op_params, परिमाण(भिन्न));
    memcpy(&max_val, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ध्रुव auto* src, auto* dst_ptr, पूर्णाङ्क k_elements, queue_ptr stream, भिन्न min_arg, भिन्न max_arg) {
            ध्रुव पूर्णाङ्क अङ्कः_blocks = ceil_div(k_elements, SYCL_CLAMP_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अङ्कः_blocks) * sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, item_ct1);
                });
        }, min_val, max_val);
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव भिन्न alpha_n = ggml_get_op_params_f32(dst, 1);
    ध्रुव भिन्न alpha_p = ggml_get_op_params_f32(dst, 2);
    ध्रुव भिन्न beta    = ggml_get_op_params_f32(dst, 3);
    ध्रुव भिन्न eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ध्रुव auto* src, auto* dst_ptr, पूर्णाङ्क k_elements, queue_ptr stream, भिन्न alpha_n_arg, भिन्न alpha_p_arg, भिन्न beta_arg, भिन्न eps_arg) {
            ध्रुव पूर्णाङ्क अङ्कः_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अङ्कः_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, item_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_floor(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_ceil(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_round(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        पटिनिवत्त op_trunc(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    ध्रुव वर्ण  * src0_d = (ध्रुव वर्ण  *) src0->data;
    ध्रुव वर्ण  * src1_d = (ध्रुव वर्ण  *) src1->data;
    भिन्न       * dst_d  = (भिन्न       *)  dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ध्रुव int64_t s1     = (int64_t) ((ध्रुव int32_t *) dst->op_params)[0] / (int64_t) परिमाण(भिन्न);
    ध्रुव int64_t s2     = (int64_t) ((ध्रुव int32_t *) dst->op_params)[1] / (int64_t) परिमाण(भिन्न);
    ध्रुव int64_t s3     = (int64_t) ((ध्रुव int32_t *) dst->op_params)[2] / (int64_t) परिमाण(भिन्न);
    ध्रुव int64_t offset = (int64_t) ((ध्रुव int32_t *) dst->op_params)[3] / (int64_t) परिमाण(भिन्न);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, offset, stream);
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        पटिनिवत्त op_gelu(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        पटिनिवत्त op_relu(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        पटिनिवत्त op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
template<typename F>
स्थिर शून्य dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
    विकल्प (uop) {
        पक्ष GGML_UNARY_OP_SILU:     launch([](भिन्न v) { पटिनिवत्त op_silu(v); });     भङ्ग;
        पक्ष GGML_UNARY_OP_SIGMOID:  launch([](भिन्न v) { पटिनिवत्त op_sigmoid(v); });  भङ्ग;
        पक्ष GGML_UNARY_OP_SOFTPLUS: launch([](भिन्न v) { पटिनिवत्त op_softplus(v); }); भङ्ग;
        स्वभाव:
            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
    }
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
शून्य ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_ग्रन्थिः, ggml_tensor * mul_ग्रन्थिः) {
    scope_op_debug_print scope_dbg_print(__func__, mul_ग्रन्थिः, /*num_src=*/2);

    ध्रुव ggml_tensor * x = unary_ग्रन्थिः->src[0];
    ध्रुव ggml_tensor * g = (mul_ग्रन्थिः->src[0] == unary_ग्रन्थिः) ? mul_ग्रन्थिः->src[1] : mul_ग्रन्थिः->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_ग्रन्थिः);
    GGML_ASSERT(x->type == g->type && x->type == mul_ग्रन्थिः->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_ग्रन्थिः));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_ग्रन्थिः));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ध्रुव int64_t k = ggml_nelements(mul_ग्रन्थिः);
    ध्रुव int64_t n = mul_ग्रन्थिः->ne[0];

    ध्रुव auto dispatch_type = [&](auto op) {
        विकल्प (mul_ग्रन्थिः->type) {
            पक्ष GGML_TYPE_F32:
                unary_mul_sycl((ध्रुव भिन्न *) x->data, (ध्रुव भिन्न *) g->data, (भिन्न *) mul_ग्रन्थिः->data,
                               k, n, x->nb[1] / परिमाण(भिन्न), g->nb[1] / परिमाण(भिन्न), main_stream, op);
                भङ्ग;
            पक्ष GGML_TYPE_F16:
                unary_mul_sycl((ध्रुव sycl::half *) x->data, (ध्रुव sycl::half *) g->data, (sycl::half *) mul_ग्रन्थिः->data,
                               k, n, x->nb[1] / परिमाण(sycl::half), g->nb[1] / परिमाण(sycl::half), main_stream, op);
                भङ्ग;
            स्वभाव:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_ग्रन्थिः->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_ग्रन्थिः), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
शून्य ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_ग्रन्थिः,
                                      ggml_tensor * unary_ग्रन्थिः, ggml_tensor * mul_ग्रन्थिः) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_ग्रन्थिः, /*num_src=*/2);

    ध्रुव ggml_tensor * a     = add_ग्रन्थिः->src[0];
    ध्रुव ggml_tensor * bias  = add_ग्रन्थिः->src[1];
    ध्रुव ggml_tensor * scale = (mul_ग्रन्थिः->src[0] == unary_ग्रन्थिः) ? mul_ग्रन्थिः->src[1] : mul_ग्रन्थिः->src[0];

    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(scale != unary_ग्रन्थिः);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_ग्रन्थिः->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_ग्रन्थिः));
    // a and dst are indexed flat
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_ग्रन्थिः));
    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ध्रुव auto dispatch_op = [&](auto op) {
        add_unary_mul_sycl((ध्रुव भिन्न *) a->data, (ध्रुव भिन्न *) bias->data, (ध्रुव भिन्न *) scale->data,
                           (भिन्न *) mul_ग्रन्थिः->data, ggml_nelements(mul_ग्रन्थिः), mul_ग्रन्थिः->ne[0], main_stream, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_ग्रन्थिः), dispatch_op);
}

__dpct_inline__ भिन्न ggml_sycl_op_swiglu_oai_single(भिन्न x, भिन्न g, भिन्न alpha = 1.702f, भिन्न limit = 7.0f) {
    x = sycl::fmin(x, limit);
    g = sycl::fmax(sycl::fmin(g, limit), -limit);

    भिन्न out_glu = x / (1.0f + sycl::native::exp(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    पटिनिवत्त out_glu;
}

template <typename T>
स्थिर शून्य swiglu_oai_kernel(ध्रुव T * x, ध्रुव T * g, T * dst, ध्रुव int64_t k,
                              ध्रुव int64_t n, ध्रुव int64_t o0, ध्रुव int64_t o1,
                              भिन्न alpha, भिन्न limit, sycl::nd_item<3> item_ct1) {
    ध्रुव int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    सचे (i >= k) {
        पटिनिवत्त;
    }

    ध्रुव int64_t j0 = (i / n) * o0 + (i % n);
    ध्रुव int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    भिन्न xi = x[j0];
    भिन्न gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
स्थिर शून्य swiglu_oai_sycl(ध्रुव T *       x,
                            ध्रुव T *       g,
                            T *             dst,
                            ध्रुव int64_t   k,
                            ध्रुव int64_t   n,
                            ध्रुव int64_t   o0,
                            ध्रुव int64_t   o1,
                            ध्रुव भिन्न     alpha,
                            ध्रुव भिन्न     limit,
                            dpct::queue_ptr stream) {
    ध्रुव int64_t अङ्कः_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अङ्कः_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, limit, item_ct1);
                         });
}

स्थिर शून्य ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    शून्य * src0_d = src0->data;
    शून्य * src1_d = src1 ? src1->data : src0->data;
    ध्रुव int64_t src0_o = src0->nb[1];
    ध्रुव int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    शून्य * dst_d = dst->data;
    ध्रुव int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    सचे (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ध्रुव भिन्न alpha = ggml_get_op_params_f32(dst, 2);
    ध्रुव भिन्न limit = ggml_get_op_params_f32(dst, 3);

    भिन्न * src0_p = (भिन्न *) src0_d;
    भिन्न * src1_p = (भिन्न *) src1_d;

    सचे (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (भिन्न *)dst_d, ggml_nelements(dst), nc, src0_o / परिमाण(भिन्न), src1_o / परिमाण(भिन्न), alpha, limit, stream);
}

template <typename T>
स्थिर शून्य swiglu_clamp_kernel(ध्रुव T *        gate,
                                ध्रुव T *        up,
                                T *              dst,
                                ध्रुव int64_t    k,
                                ध्रुव int64_t    n,
                                ध्रुव int64_t    o0,
                                ध्रुव int64_t    o1,
                                भिन्न            limit,
                                sycl::nd_item<3> item_ct1) {
    ध्रुव int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    सचे (i >= k) {
        पटिनिवत्त;
    }

    ध्रुव int64_t j0 = (i / n) * o0 + (i % n);
    ध्रुव int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ध्रुव भिन्न gate_मानम् = sycl::fmin((भिन्न) gate[j0], limit);
    ध्रुव भिन्न up_मानम्   = sycl::fmax(sycl::fmin((भिन्न) up[j1], limit), -limit);
    dst[i]                 = (T) (gate_मानम् / (1.0f + sycl::native::exp(-gate_मानम्)) * up_मानम्);
}

template <typename T>
स्थिर शून्य swiglu_clamp_sycl(ध्रुव T *       gate,
                              ध्रुव T *       up,
                              T *             dst,
                              ध्रुव int64_t   k,
                              ध्रुव int64_t   n,
                              ध्रुव int64_t   o0,
                              ध्रुव int64_t   o1,
                              भिन्न           limit,
                              dpct::queue_ptr stream) {
    ध्रुव int64_t अङ्कः_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अङ्कः_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_clamp_kernel(gate, up, dst, k, n, o0, o1, limit, item_ct1);
                         });
}

स्थिर शून्य ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0   = dst->src[0];
    ध्रुव ggml_tensor * src1   = dst->src[1];
    शून्य *              src0_d = src0->data;
    शून्य *              src1_d = src1 ? src1->data : src0->data;
    ध्रुव int64_t       src0_o = src0->nb[1];
    ध्रुव int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    शून्य *              dst_d  = dst->data;
    ध्रुव int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    सचे (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ध्रुव भिन्न   limit   = ggml_get_op_params_f32(dst, 3);

    सचे (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        सचे (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / परिमाण(sycl::half),
                          src1_o / परिमाण(sycl::half), limit, stream);
    } अञ्ञथा {
        भिन्न * src0_p = (भिन्न *) src0_d;
        भिन्न * src1_p = (भिन्न *) src1_d;

        सचे (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (भिन्न *) dst_d, ggml_nelements(dst), nc, src0_o / परिमाण(भिन्न),
                          src1_o / परिमाण(भिन्न), limit, stream);
    }
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        पटिनिवत्त op_gelu_erf(x);
    });
}

स्थिर अन्तःस्थ शून्य ggml_sycl_op_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        पटिनिवत्त op_gelu_quick(x);
    });
}


शून्य ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

शून्य ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

शून्य ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

शून्य ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

शून्य ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

शून्य ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

शून्य ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_quick(ctx, dst);
}

शून्य ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

शून्य ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

शून्य ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

शून्य ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

शून्य ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

शून्य ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

शून्य ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

शून्य ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_expm1(ctx, dst);
}

शून्य ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

शून्य ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

शून्य ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

शून्य ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_step(ctx, dst);
}

शून्य ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

शून्य ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

शून्य ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

शून्य ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_xielu(ctx, dst);
}

शून्य ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

शून्य ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

शून्य ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

शून्य ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

शून्य ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

शून्य ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

शून्य ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

शून्य ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_clamp(ctx, dst);
}

शून्य ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

शून्य ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_quick(ctx, dst);
}

शून्य ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

शून्य ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

शून्य ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

शून्य ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_round(ctx, dst);
}

शून्य ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
