#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "ggml-sycl/presets.hpp"
#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "element_wise.hpp"

#define SYCL_GLOBAL_ID_LOOP(K, ITEM) \
    প্রতি (auto i = ITEM.get_global_id(0); i < (size_t)K; i += ITEM.get_global_range(0))

#define SYCL_LOCAL_ID_CALC(ITEM, IDX) \
    (ITEM.get_local_range(IDX) * ITEM.get_group(IDX) + ITEM.get_local_id(IDX))

স্থির শূন্য acc_f32(ধ্রুব বর্ণ * x, ধ্রুব বর্ণ * y, ভিন্ন * dst, ধ্রুব int64_t ne,
    ধ্রুব int64_t ne0, ধ্রুব int64_t ne1, ধ্রুব int64_t ne2,
    ধ্রুব int64_t nb00, ধ্রুব int64_t nb01, ধ্রুব int64_t nb02, ধ্রুব int64_t nb03,
    ধ্রুব int64_t ne10, ধ্রুব int64_t ne11, ধ্রুব int64_t ne12, ধ্রুব int64_t ne13,
    ধ্রুব int64_t nb10, ধ্রুব int64_t nb11, ধ্রুব int64_t nb12, ধ্রুব int64_t nb13,
    ধ্রুব int64_t s11, ধ্রুব int64_t s12, ধ্রুব int64_t s13, ধ্রুব int64_t offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুব int64_t i = SYCL_LOCAL_ID_CALC(item_ct1, 2);

    যদি (i >= ne) {
        প্রত্যাগম;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    ধ্রুব int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    ধ্রুব int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    ধ্রুব int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    ধ্রুব int64_t i10 = tmp;

    int64_t tmp_dst = i;
    ধ্রুব int64_t i3 = tmp_dst / (ne2*ne1*ne0);
    tmp_dst -= i3 * (ne2*ne1*ne0);
    ধ্রুব int64_t i2 = tmp_dst / (ne1*ne0);
    tmp_dst -= i2 * (ne1*ne0);
    ধ্রুব int64_t i1 = tmp_dst / ne0;
    tmp_dst -= i1 * ne0;
    ধ্রুব int64_t i0 = tmp_dst;

    ভিন্ন val = *(ধ্রুব ভিন্ন *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    যদি (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(ধ্রুব ভিন্ন *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
template<typename T>
স্থির __dpct_inline__ T op_sgn(T x) {
    প্রত্যাগম x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


template<typename T>
স্থির __dpct_inline__ T op_abs(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } অন্যথা {
        প্রত্যাগম sycl::fabs(x);
    }
}

template<typename T>
স্থির __dpct_inline__ T op_expm1(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<ভিন্ন>(x))
        );
    } অন্যথা {
        প্রত্যাগম sycl::expm1(x);
    }
}

template<typename T>
স্থির __dpct_inline__ T op_elu(T x) {
    প্রত্যাগম (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

template<typename T>
স্থির __dpct_inline__ T op_erf(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<ভিন্ন>(x))
        );
    } অন্যথা {
        প্রত্যাগম sycl::erf(x);
    }
}

template<typename T>
স্থির __dpct_inline__ T op_gelu_erf(T x) {
    ধ্রুব T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    প্রত্যাগম static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

template<typename T>
স্থির __dpct_inline__ T op_gelu_quick(T x) {
    ধ্রুব T GELU_QUICK_COEF_LOCAL = static_cast<T>(-1.702f);
    প্রত্যাগম x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_QUICK_COEF_LOCAL * x)));
}

template<typename T>
স্থির __dpct_inline__ T op_relu(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } অন্যথা {
        প্রত্যাগম sycl::fmax(x, static_cast<T>(0));
    }
}

template<typename T>
স্থির __dpct_inline__ T op_sigmoid(T x) {
    প্রত্যাগম static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

template<typename T>
স্থির __dpct_inline__ T op_sqrt(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম sycl::ext::oneapi::experimental::sqrt(x);
    } অন্যথা {
        প্রত্যাগম sycl::sqrt(x);
    }
}

template<typename T>
স্থির __dpct_inline__ T op_sin(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম sycl::ext::oneapi::experimental::sin(x);
    } অন্যথা {
        প্রত্যাগম sycl::sin(x);
    }
}

template<typename T>
স্থির __dpct_inline__ T op_cos(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম sycl::ext::oneapi::experimental::cos(x);
    } অন্যথা {
        প্রত্যাগম sycl::cos(x);
    }
}

template<typename T>
স্থির __dpct_inline__ T op_hardsigmoid(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } অন্যথা {
        প্রত্যাগম sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
স্থির __dpct_inline__ T op_hardswish(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } অন্যথা {
        প্রত্যাগম x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
স্থির __dpct_inline__ T op_log(T x) {
    যদি (x <= static_cast<T>(0)) {
        প্রত্যাগম neg_infinity<T>();
    }
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম sycl::ext::oneapi::experimental::log(x);
    } অন্যথা {
        প্রত্যাগম sycl::log(x);
    }
}

template<typename T>
স্থির __dpct_inline__ T op_softplus(T x) {
    ধ্রুব ভিন্ন xf = (ভিন্ন) x;
    ধ্রুব ভিন্ন ax = op_abs(xf);
    ধ্রুব ভিন্ন m  = sycl::fmax(xf, 0.0f);
    ধ্রুব ভিন্ন y  = m + sycl::log1p(sycl::exp(-ax));
    প্রত্যাগম (T) y;
}

template<typename T>
স্থির __dpct_inline__ T op_neg(T x) {
    প্রত্যাগম -x;
}

template<typename T>
স্থির __dpct_inline__ T op_step(T x) {
    প্রত্যাগম (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

template<typename T>
স্থির __dpct_inline__ T op_leaky_relu(T x, ভিন্ন negative_slope) {
    T neg_slope_T = static_cast<T>(negative_slope);
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;

    } অন্যথা {
        প্রত্যাগম sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;
    }
}

template<typename T>
স্থির __dpct_inline__ T op_xielu(T x, ভিন্ন alpha_n, ভিন্ন alpha_p, ভিন্ন beta, ভিন্ন eps) {
    ধ্রুব ভিন্ন xi        = static_cast<ভিন্ন>(x);
    ধ্রুব ভিন্ন gate_pos  = (xi > 0.0f);
    ধ্রুব ভিন্ন y_pos     = alpha_p * xi * xi + beta * xi;
    ধ্রুব ভিন্ন min_v_eps = sycl::fmin(xi, eps);
    ধ্রুব ভিন্ন y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    ধ্রুব ভিন্ন out       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    প্রত্যাগম static_cast<T>(out);
}

template<typename T>
স্থির __dpct_inline__ T op_sqr(T x) {
    প্রত্যাগম x * x;
}

template<typename T>
স্থির __dpct_inline__ T op_clamp(T x, ভিন্ন min_val, ভিন্ন max_val) {
    প্রত্যাগম x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

template<typename T>
স্থির __dpct_inline__ T op_floor(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম sycl::ext::oneapi::experimental::floor(x);
    } অন্যথা {
        প্রত্যাগম sycl::floor(x);
    }
}

template<typename T>
স্থির __dpct_inline__ T op_ceil(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম sycl::ext::oneapi::experimental::ceil(x);
    } অন্যথা {
        প্রত্যাগম sycl::ceil(x);
    }
}

template<typename T>
স্থির __dpct_inline__ T op_round(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<ভিন্ন>(x))
        );
    } অন্যথা {
        প্রত্যাগম sycl::round(x);
    }
}

template<typename T>
স্থির __dpct_inline__ T op_trunc(T x) {
    যদি constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        প্রত্যাগম sycl::ext::oneapi::experimental::trunc(x);
    } অন্যথা {
        প্রত্যাগম sycl::trunc(x);
    }
}

template<typename T, typename F>
স্থির শূন্য unary_op_flat_kernel(ধ্রুব T * x, T * dst, ধ্রুব পূর্ণাঙ্ক k, ধ্রুব sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]);
    }
}

template<typename T, typename F>
স্থির শূন্য unary_op_generic_kernel(
        ধ্রুব T * x,
        T * dst,
        ধ্রুব পূর্ণাঙ্ক k,
        ধ্রুব sycl::uint3 ne0_fd, ধ্রুব sycl::uint3 ne1_fd, ধ্রুব sycl::uint3 ne2_fd,
        ধ্রুব size_t nb0,  ধ্রুব size_t nb1,  ধ্রুব size_t nb2,  ধ্রুব size_t nb3,
        ধ্রুব size_t nbd0, ধ্রুব size_t nbd1, ধ্রুব size_t nbd2, ধ্রুব size_t nbd3,
        ধ্রুব sycl::nd_item<1> & item_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        ধ্রুব uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        ধ্রুব uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        ধ্রুব uint32_t i2 = dm.y();
        ধ্রুব uint32_t i3 = dm.x();

        ধ্রুব বর্ণ * src_base = (ধ্রুব বর্ণ *) x;
        বর্ণ       * dst_base = (বর্ণ *) dst;

        ধ্রুব T * srcp = (ধ্রুব T *)(src_base + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_base + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

template<typename T>
স্থির শূন্য unary_op_sqrt_kernel(ধ্রুব T * x, T * dst, ধ্রুব পূর্ণাঙ্ক k, ধ্রুব sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

template<typename T>
স্থির শূন্য unary_op_sin_kernel(ধ্রুব T * x, T * dst, ধ্রুব পূর্ণাঙ্ক k, ধ্রুব sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

template<typename T>
স্থির শূন্য unary_op_cos_kernel(ধ্রুব T * x, T * dst, ধ্রুব পূর্ণাঙ্ক k, ধ্রুব sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

template<typename T>
স্থির শূন্য unary_op_log_kernel(ধ্রুব T * x, T * dst, ধ্রুব পূর্ণাঙ্ক k, ধ্রুব sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_log(x[i]);
    }
}


template<typename T>
স্থির শূন্য unary_op_leaky_relu_kernel(ধ্রুব T * x, T * dst, ধ্রুব পূর্ণাঙ্ক k, ভিন্ন negative_slope, ধ্রুব sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_slope);
    }
}

template<typename T>
স্থির শূন্য unary_op_xielu_kernel(ধ্রুব T * x, T * dst, ধ্রুব পূর্ণাঙ্ক k, ভিন্ন alpha_n, ভিন্ন alpha_p, ভিন্ন beta, ভিন্ন eps, ধ্রুব sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

template<typename T>
স্থির শূন্য unary_op_sqr_kernel(ধ্রুব T * x, T * dst, ধ্রুব পূর্ণাঙ্ক k, ধ্রুব sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

template<typename T>
স্থির শূন্য unary_op_clamp_kernel(ধ্রুব T * x, T * dst, ধ্রুব পূর্ণাঙ্ক k, ধ্রুব sycl::nd_item<1> &item_ct1, ভিন্ন min_val, ভিন্ন max_val) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

template<typename T>
স্থির শূন্য unary_op_ceil_kernel(ধ্রুব T * x, T * dst, ধ্রুব পূর্ণাঙ্ক k, ধ্রুব sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

template<typename T>
স্থির শূন্য clamp(ধ্রুব T * x, T * dst, ধ্রুব ভিন্ন min, ধ্রুব ভিন্ন max, ধ্রুব পূর্ণাঙ্ক k,
                      ধ্রুব sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

template<typename T, typename F>
স্থির শূন্য unary_gated_op_flat_kernel(ধ্রুব T * x, ধ্রুব T * g, T * dst, ধ্রুব uint64_t k, ধ্রুব sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

template<typename T, typename F>
স্থির শূন্য unary_gated_op_generic_kernel(
        ধ্রুব T * x,
        ধ্রুব T * g,
        T * dst,
        ধ্রুব uint64_t k,
        ধ্রুব sycl::uint3 n_fd,
        ধ্রুব uint64_t o0,
        ধ্রুব uint64_t o1,
        ধ্রুব sycl::nd_item<1> & item_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ধ্রুব sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ধ্রুব int64_t j0 = rc.x() * o0 + rc.y();
        ধ্রুব int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
template<typename T, typename F>
স্থির শূন্য unary_mul_flat_kernel(ধ্রুব T * x, ধ্রুব T * g, T * dst, ধ্রুব int64_t k, ধ্রুব sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = (T) (op((ভিন্ন) x[i]) * (ভিন্ন) g[i]);
    }
}

template<typename T, typename F>
স্থির শূন্য unary_mul_strided_kernel(ধ্রুব T * x, ধ্রুব T * g, T * dst, ধ্রুব int64_t k, ধ্রুব sycl::uint3 n_fd, ধ্রুব int64_t o0, ধ্রুব int64_t o1, ধ্রুব sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ধ্রুব sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ধ্রুব int64_t j0 = rc.x() * o0 + rc.y();
        ধ্রুব int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((ভিন্ন) x[j0]) * (ভিন্ন) g[j1]);
    }
}

template<typename T, typename F>
স্থির শূন্য unary_mul_sycl(ধ্রুব T * x, ধ্রুব T * g, T * dst, ধ্রুব int64_t k, ধ্রুব int64_t n, ধ্রুব int64_t o0, ধ্রুব int64_t o1, queue_ptr main_stream, F op) {
    ধ্রুব size_t            অঙ্ক_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    ধ্রুব sycl::nd_range<1> range(অঙ্ক_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    যদি (o0 == n && o1 == n) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            unary_mul_flat_kernel(x, g, dst, k, item_ct1, op);
        });
        প্রত্যাগম;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ধ্রুব sycl::uint3 n_fd = init_fastdiv_মানসমূহ((uint32_t) n);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, item_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
template<typename F>
স্থির শূন্য add_unary_mul_flat_kernel(ধ্রুব ভিন্ন * a, ধ্রুব ভিন্ন * bias, ধ্রুব ভিন্ন * scale, ভিন্ন * dst,
                                      ধ্রুব int64_t k, ধ্রুব sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

template<typename F>
স্থির শূন্য add_unary_mul_bcast_kernel(ধ্রুব ভিন্ন * a, ধ্রুব ভিন্ন * bias, ধ্রুব ভিন্ন * scale, ভিন্ন * dst,
                                       ধ্রুব int64_t k, ধ্রুব sycl::uint3 ne0_fd, ধ্রুব sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ধ্রুব uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

template<typename F>
স্থির শূন্য add_unary_mul_sycl(ধ্রুব ভিন্ন * a, ধ্রুব ভিন্ন * bias, ধ্রুব ভিন্ন * scale, ভিন্ন * dst,
                               ধ্রুব int64_t k, ধ্রুব int64_t ne0, queue_ptr main_stream, F op) {
    ধ্রুব size_t            অঙ্ক_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    ধ্রুব sycl::nd_range<1> range(অঙ্ক_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    যদি (k == ne0) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            add_unary_mul_flat_kernel(a, bias, scale, dst, k, item_ct1, op);
        });
        প্রত্যাগম;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ধ্রুব sycl::uint3 ne0_fd = init_fastdiv_মানসমূহ((uint32_t) ne0);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        add_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, item_ct1, op);
    });
}

নামাকাশ ggml_sycl_detail {
স্থির শূন্য acc_f32_sycl(ধ্রুব বর্ণ *x, ধ্রুব বর্ণ *y, ভিন্ন *dst,
                         ধ্রুব int64_t n_elements,
                         ধ্রুব int64_t ne0, ধ্রুব int64_t ne1, ধ্রুব int64_t ne2,
                         ধ্রুব int64_t nb00, ধ্রুব int64_t nb01, ধ্রুব int64_t nb02, ধ্রুব int64_t nb03,
                         ধ্রুব int64_t ne10, ধ্রুব int64_t ne11, ধ্রুব int64_t ne12, ধ্রুব int64_t ne13,
                         ধ্রুব int64_t nb10, ধ্রুব int64_t nb11, ধ্রুব int64_t nb12, ধ্রুব int64_t nb13,
                         ধ্রুব int64_t s1, ধ্রুব int64_t s2, ধ্রুব int64_t s3,
                         ধ্রুব int64_t offset, queue_ptr stream) {
    ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
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
স্থির শূন্য arange_kernel(T * dst, ধ্রুব পূর্ণাঙ্ক k, T start, T step,
                         ধ্রুব sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = start + static_cast<T>(i) * step;
    }
}

template<typename KernelInvoker, typename... Args>
স্থির অন্তঃস্থ শূন্য dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    বিকল্প (dst->type) {
        পক্ষ GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (পূর্ণাঙ্ক)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                ভঙ্গ;
            }
#ifdef GGML_SYCL_HAS_BF16
        পক্ষ GGML_TYPE_BF16:
            {
                auto data_pts = cast_data<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (পূর্ণাঙ্ক)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                ভঙ্গ;
            }
#endif
        পক্ষ GGML_TYPE_F32:
            {
                auto data_pts = cast_data<ভিন্ন>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (পূর্ণাঙ্ক)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                ভঙ্গ;
            }
        স্বভাব:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename KernelInvoker, typename... Args>
স্থির অন্তঃস্থ শূন্য dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    ধ্রুব int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // The fused GLU kernels index with 32-bit fastdiv, which is exact only for indices below
    // 2^31. A dst that large is ~8 GB at f32, and the grid sizing already narrows to 32 bits,
    // so assert the bound rather than carry a second code path for it.
    GGML_ASSERT(ggml_nelements(dst) < ((int64_t) 1 << 31));
    ধ্রুব int32_t swapped = ((ধ্রুব int32_t *) dst->op_params)[1];
    শূন্য * src0_d = src0->data;
    শূন্য * src1_d = src1 ? src1->data : src0->data;
    ধ্রুব int64_t src0_o = src0->nb[1];
    ধ্রুব int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    শূন্য * dst_d = dst->data;
    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    বিকল্প (dst->type) {
        পক্ষ GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    যদি (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / পরিমাণ(sycl::half),
                               src1_o / পরিমাণ(sycl::half),
                               main_stream,
                               std::forward<Args>(args)...);
                ভঙ্গ;
            }
        পক্ষ GGML_TYPE_F32:
            {
                ভিন্ন * src0_p = (ভিন্ন *) src0_d;
                ভিন্ন * src1_p = (ভিন্ন *) src1_d;

                    যদি (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (ভিন্ন *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / পরিমাণ(ভিন্ন),
                               src1_o / পরিমাণ(ভিন্ন),
                               main_stream,
                               std::forward<Args>(args)...);
                ভঙ্গ;
            }
        স্বভাব:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename F>
স্থির অন্তঃস্থ শূন্য ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    ধ্রুব int64_t ne0  = dst->ne[0];
    ধ্রুব int64_t ne1  = dst->ne[1];
    ধ্রুব int64_t ne2  = dst->ne[2];

    ধ্রুব size_t  nb0  = src0->nb[0];
    ধ্রুব size_t  nb1  = src0->nb[1];
    ধ্রুব size_t  nb2  = src0->nb[2];
    ধ্রুব size_t  nb3  = src0->nb[3];

    ধ্রুব size_t  nbd0 = dst->nb[0];
    ধ্রুব size_t  nbd1 = dst->nb[1];
    ধ্রুব size_t  nbd2 = dst->nb[2];
    ধ্রুব size_t  nbd3 = dst->nb[3];

    // Hot unary ops (FFN/GDN silu, sigmoid, ...) run on contiguous tensors;
    // skip the strided index math entirely for them.
    ধ্রুব তার্কিক contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](ধ্রুব auto* src, auto* dst_ptr, পূর্ণাঙ্ক k_elements, queue_ptr stream) {

            ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = ceil_div(k_elements, 256);

            যদি (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, item_ct1, func);
                    });
            } অন্যথা {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                ধ্রুব sycl::uint3 ne0_fd = init_fastdiv_মানসমূহ((uint32_t) ne0);
                ধ্রুব sycl::uint3 ne1_fd = init_fastdiv_মানসমূহ((uint32_t) ne1);
                ধ্রুব sycl::uint3 ne2_fd = init_fastdiv_মানসমূহ((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks) * sycl::range<1>(256),
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
স্থির অন্তঃস্থ শূন্য ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](ধ্রুব auto * x_ptr, ধ্রুব auto * g_ptr, auto * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {

            ধ্রুব uint32_t অঙ্ক_blocks = (uint32_t) ceil_div(k, SYCL_GLU_BLOCK_SIZE);
            ধ্রুব sycl::nd_range<1> launch_range(অঙ্ক_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE),
                                                 sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            যদি (o0 == n && o1 == n) {
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, item_ct1, func);
                    });
            } অন্যথা {
                // launch-invariant divisor, and only this path needs it
                ধ্রুব sycl::uint3 n_fd = init_fastdiv_মানসমূহ((uint32_t) n);
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, item_ct1, func);
                    });
            }
        });
}


স্থির অন্তঃস্থ শূন্য ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    ভিন্ন start, stop, step;
    memcpy(&start, dst->op_params, পরিমাণ(ভিন্ন));
    memcpy(&stop, (ভিন্ন *) dst->op_params + 1, পরিমাণ(ভিন্ন));
    memcpy(&step, (ভিন্ন *) dst->op_params + 2, পরিমাণ(ভিন্ন));
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ভিন্ন * dst_ptr = (ভিন্ন *)dst->data;
    ধ্রুব পূর্ণাঙ্ক k = (পূর্ণাঙ্ক)ggml_nelements(dst);
    ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = ceil_div(k, SYCL_ARANGE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks) * sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE),
                          sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            arange_kernel(dst_ptr, k, start, step, item_ct1);
        });
}

} // namespace ggml_sycl_detail



স্থির অন্তঃস্থ শূন্য ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_sgn(x);
    });
}


স্থির অন্তঃস্থ শূন্য ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_abs(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_elu(x);
    });
}
স্থির অন্তঃস্থ শূন্য ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_silu(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_gelu(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_gelu_quick(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_gelu_erf(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_tanh(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_relu(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_hardsigmoid(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_hardswish(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_exp(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_expm1(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ধ্রুব auto* src, auto* dst_ptr, পূর্ণাঙ্ক k_elements, queue_ptr stream) {
            ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_softplus(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_neg(x);
    });
}


স্থির অন্তঃস্থ শূন্য ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_step(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_sigmoid(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ধ্রুব auto* src, auto* dst_ptr, পূর্ণাঙ্ক k_elements, queue_ptr stream) {
            ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = ceil_div(k_elements, SYCL_SQRT_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks) * sycl::range<1>(SYCL_SQRT_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQRT_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ধ্রুব auto* src, auto* dst_ptr, পূর্ণাঙ্ক k_elements, queue_ptr stream) {
            ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ধ্রুব auto* src, auto* dst_ptr, পূর্ণাঙ্ক k_elements, queue_ptr stream) {
            ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ভিন্ন negative_slope;
    memcpy(&negative_slope, dst->op_params, পরিমাণ(ভিন্ন));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ধ্রুব auto* src, auto* dst_ptr, পূর্ণাঙ্ক k_elements, queue_ptr stream, ভিন্ন slope) {
            ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, slope, item_ct1);
                });
        }, negative_slope);
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ধ্রুব auto* src, auto* dst_ptr, পূর্ণাঙ্ক k_elements, queue_ptr stream) {
            ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = ceil_div(k_elements, SYCL_SQR_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks) * sycl::range<1>(SYCL_SQR_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQR_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ভিন্ন min_val;
    ভিন্ন max_val;
    memcpy(&min_val, dst->op_params, পরিমাণ(ভিন্ন));
    memcpy(&max_val, (ভিন্ন *) dst->op_params + 1, পরিমাণ(ভিন্ন));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ধ্রুব auto* src, auto* dst_ptr, পূর্ণাঙ্ক k_elements, queue_ptr stream, ভিন্ন min_arg, ভিন্ন max_arg) {
            ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = ceil_div(k_elements, SYCL_CLAMP_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks) * sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, item_ct1);
                });
        }, min_val, max_val);
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুব ভিন্ন alpha_n = ggml_get_op_params_f32(dst, 1);
    ধ্রুব ভিন্ন alpha_p = ggml_get_op_params_f32(dst, 2);
    ধ্রুব ভিন্ন beta    = ggml_get_op_params_f32(dst, 3);
    ধ্রুব ভিন্ন eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ধ্রুব auto* src, auto* dst_ptr, পূর্ণাঙ্ক k_elements, queue_ptr stream, ভিন্ন alpha_n_arg, ভিন্ন alpha_p_arg, ভিন্ন beta_arg, ভিন্ন eps_arg) {
            ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, item_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_floor(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_ceil(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_round(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        প্রত্যাগম op_trunc(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    ধ্রুব বর্ণ  * src0_d = (ধ্রুব বর্ণ  *) src0->data;
    ধ্রুব বর্ণ  * src1_d = (ধ্রুব বর্ণ  *) src1->data;
    ভিন্ন       * dst_d  = (ভিন্ন       *)  dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ধ্রুব int64_t s1     = (int64_t) ((ধ্রুব int32_t *) dst->op_params)[0] / (int64_t) পরিমাণ(ভিন্ন);
    ধ্রুব int64_t s2     = (int64_t) ((ধ্রুব int32_t *) dst->op_params)[1] / (int64_t) পরিমাণ(ভিন্ন);
    ধ্রুব int64_t s3     = (int64_t) ((ধ্রুব int32_t *) dst->op_params)[2] / (int64_t) পরিমাণ(ভিন্ন);
    ধ্রুব int64_t offset = (int64_t) ((ধ্রুব int32_t *) dst->op_params)[3] / (int64_t) পরিমাণ(ভিন্ন);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, offset, stream);
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        প্রত্যাগম op_gelu(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        প্রত্যাগম op_relu(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        প্রত্যাগম op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
template<typename F>
স্থির শূন্য dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
    বিকল্প (uop) {
        পক্ষ GGML_UNARY_OP_SILU:     launch([](ভিন্ন v) { প্রত্যাগম op_silu(v); });     ভঙ্গ;
        পক্ষ GGML_UNARY_OP_SIGMOID:  launch([](ভিন্ন v) { প্রত্যাগম op_sigmoid(v); });  ভঙ্গ;
        পক্ষ GGML_UNARY_OP_SOFTPLUS: launch([](ভিন্ন v) { প্রত্যাগম op_softplus(v); }); ভঙ্গ;
        স্বভাব:
            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
    }
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
শূন্য ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_গ্রন্থি, ggml_tensor * mul_গ্রন্থি) {
    scope_op_debug_print scope_dbg_print(__func__, mul_গ্রন্থি, /*num_src=*/2);

    ধ্রুব ggml_tensor * x = unary_গ্রন্থি->src[0];
    ধ্রুব ggml_tensor * g = (mul_গ্রন্থি->src[0] == unary_গ্রন্থি) ? mul_গ্রন্থি->src[1] : mul_গ্রন্থি->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_গ্রন্থি);
    GGML_ASSERT(x->type == g->type && x->type == mul_গ্রন্থি->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_গ্রন্থি));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_গ্রন্থি));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুব int64_t k = ggml_nelements(mul_গ্রন্থি);
    ধ্রুব int64_t n = mul_গ্রন্থি->ne[0];

    ধ্রুব auto dispatch_type = [&](auto op) {
        বিকল্প (mul_গ্রন্থি->type) {
            পক্ষ GGML_TYPE_F32:
                unary_mul_sycl((ধ্রুব ভিন্ন *) x->data, (ধ্রুব ভিন্ন *) g->data, (ভিন্ন *) mul_গ্রন্থি->data,
                               k, n, x->nb[1] / পরিমাণ(ভিন্ন), g->nb[1] / পরিমাণ(ভিন্ন), main_stream, op);
                ভঙ্গ;
            পক্ষ GGML_TYPE_F16:
                unary_mul_sycl((ধ্রুব sycl::half *) x->data, (ধ্রুব sycl::half *) g->data, (sycl::half *) mul_গ্রন্থি->data,
                               k, n, x->nb[1] / পরিমাণ(sycl::half), g->nb[1] / পরিমাণ(sycl::half), main_stream, op);
                ভঙ্গ;
            স্বভাব:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_গ্রন্থি->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_গ্রন্থি), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
শূন্য ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_গ্রন্থি,
                                      ggml_tensor * unary_গ্রন্থি, ggml_tensor * mul_গ্রন্থি) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_গ্রন্থি, /*num_src=*/2);

    ধ্রুব ggml_tensor * a     = add_গ্রন্থি->src[0];
    ধ্রুব ggml_tensor * bias  = add_গ্রন্থি->src[1];
    ধ্রুব ggml_tensor * scale = (mul_গ্রন্থি->src[0] == unary_গ্রন্থি) ? mul_গ্রন্থি->src[1] : mul_গ্রন্থি->src[0];

    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(scale != unary_গ্রন্থি);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_গ্রন্থি->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_গ্রন্থি));
    // a and dst are indexed flat
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_গ্রন্থি));
    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুব auto dispatch_op = [&](auto op) {
        add_unary_mul_sycl((ধ্রুব ভিন্ন *) a->data, (ধ্রুব ভিন্ন *) bias->data, (ধ্রুব ভিন্ন *) scale->data,
                           (ভিন্ন *) mul_গ্রন্থি->data, ggml_nelements(mul_গ্রন্থি), mul_গ্রন্থি->ne[0], main_stream, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_গ্রন্থি), dispatch_op);
}

__dpct_inline__ ভিন্ন ggml_sycl_op_swiglu_oai_single(ভিন্ন x, ভিন্ন g, ভিন্ন alpha = 1.702f, ভিন্ন limit = 7.0f) {
    x = sycl::fmin(x, limit);
    g = sycl::fmax(sycl::fmin(g, limit), -limit);

    ভিন্ন out_glu = x / (1.0f + sycl::native::exp(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    প্রত্যাগম out_glu;
}

template <typename T>
স্থির শূন্য swiglu_oai_kernel(ধ্রুব T * x, ধ্রুব T * g, T * dst, ধ্রুব int64_t k,
                              ধ্রুব int64_t n, ধ্রুব int64_t o0, ধ্রুব int64_t o1,
                              ভিন্ন alpha, ভিন্ন limit, sycl::nd_item<3> item_ct1) {
    ধ্রুব int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    যদি (i >= k) {
        প্রত্যাগম;
    }

    ধ্রুব int64_t j0 = (i / n) * o0 + (i % n);
    ধ্রুব int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ভিন্ন xi = x[j0];
    ভিন্ন gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
স্থির শূন্য swiglu_oai_sycl(ধ্রুব T *       x,
                            ধ্রুব T *       g,
                            T *             dst,
                            ধ্রুব int64_t   k,
                            ধ্রুব int64_t   n,
                            ধ্রুব int64_t   o0,
                            ধ্রুব int64_t   o1,
                            ধ্রুব ভিন্ন     alpha,
                            ধ্রুব ভিন্ন     limit,
                            dpct::queue_ptr stream) {
    ধ্রুব int64_t অঙ্ক_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, limit, item_ct1);
                         });
}

স্থির শূন্য ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];
    শূন্য * src0_d = src0->data;
    শূন্য * src1_d = src1 ? src1->data : src0->data;
    ধ্রুব int64_t src0_o = src0->nb[1];
    ধ্রুব int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    শূন্য * dst_d = dst->data;
    ধ্রুব int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ধ্রুব ভিন্ন alpha = ggml_get_op_params_f32(dst, 2);
    ধ্রুব ভিন্ন limit = ggml_get_op_params_f32(dst, 3);

    ভিন্ন * src0_p = (ভিন্ন *) src0_d;
    ভিন্ন * src1_p = (ভিন্ন *) src1_d;

    যদি (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (ভিন্ন *)dst_d, ggml_nelements(dst), nc, src0_o / পরিমাণ(ভিন্ন), src1_o / পরিমাণ(ভিন্ন), alpha, limit, stream);
}

template <typename T>
স্থির শূন্য swiglu_clamp_kernel(ধ্রুব T *        gate,
                                ধ্রুব T *        up,
                                T *              dst,
                                ধ্রুব int64_t    k,
                                ধ্রুব int64_t    n,
                                ধ্রুব int64_t    o0,
                                ধ্রুব int64_t    o1,
                                ভিন্ন            limit,
                                sycl::nd_item<3> item_ct1) {
    ধ্রুব int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    যদি (i >= k) {
        প্রত্যাগম;
    }

    ধ্রুব int64_t j0 = (i / n) * o0 + (i % n);
    ধ্রুব int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ধ্রুব ভিন্ন gate_মান = sycl::fmin((ভিন্ন) gate[j0], limit);
    ধ্রুব ভিন্ন up_মান   = sycl::fmax(sycl::fmin((ভিন্ন) up[j1], limit), -limit);
    dst[i]                 = (T) (gate_মান / (1.0f + sycl::native::exp(-gate_মান)) * up_মান);
}

template <typename T>
স্থির শূন্য swiglu_clamp_sycl(ধ্রুব T *       gate,
                              ধ্রুব T *       up,
                              T *             dst,
                              ধ্রুব int64_t   k,
                              ধ্রুব int64_t   n,
                              ধ্রুব int64_t   o0,
                              ধ্রুব int64_t   o1,
                              ভিন্ন           limit,
                              dpct::queue_ptr stream) {
    ধ্রুব int64_t অঙ্ক_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_clamp_kernel(gate, up, dst, k, n, o0, o1, limit, item_ct1);
                         });
}

স্থির শূন্য ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুব ggml_tensor * src0   = dst->src[0];
    ধ্রুব ggml_tensor * src1   = dst->src[1];
    শূন্য *              src0_d = src0->data;
    শূন্য *              src1_d = src1 ? src1->data : src0->data;
    ধ্রুব int64_t       src0_o = src0->nb[1];
    ধ্রুব int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    শূন্য *              dst_d  = dst->data;
    ধ্রুব int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    যদি (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    ধ্রুব int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ধ্রুব ভিন্ন   limit   = ggml_get_op_params_f32(dst, 3);

    যদি (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / পরিমাণ(sycl::half),
                          src1_o / পরিমাণ(sycl::half), limit, stream);
    } অন্যথা {
        ভিন্ন * src0_p = (ভিন্ন *) src0_d;
        ভিন্ন * src1_p = (ভিন্ন *) src1_d;

        যদি (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (ভিন্ন *) dst_d, ggml_nelements(dst), nc, src0_o / পরিমাণ(ভিন্ন),
                          src1_o / পরিমাণ(ভিন্ন), limit, stream);
    }
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        প্রত্যাগম op_gelu_erf(x);
    });
}

স্থির অন্তঃস্থ শূন্য ggml_sycl_op_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        প্রত্যাগম op_gelu_quick(x);
    });
}


শূন্য ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

শূন্য ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

শূন্য ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

শূন্য ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

শূন্য ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

শূন্য ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

শূন্য ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_quick(ctx, dst);
}

শূন্য ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

শূন্য ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

শূন্য ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

শূন্য ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

শূন্য ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

শূন্য ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

শূন্য ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

শূন্য ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_expm1(ctx, dst);
}

শূন্য ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

শূন্য ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

শূন্য ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

শূন্য ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_step(ctx, dst);
}

শূন্য ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

শূন্য ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

শূন্য ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

শূন্য ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_xielu(ctx, dst);
}

শূন্য ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

শূন্য ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

শূন্য ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

শূন্য ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

শূন্য ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

শূন্য ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

শূন্য ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

শূন্য ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_clamp(ctx, dst);
}

শূন্য ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

শূন্য ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_quick(ctx, dst);
}

শূন্য ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

শূন্য ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

শূন্য ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

শূন্য ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_round(ctx, dst);
}

শূন্য ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
