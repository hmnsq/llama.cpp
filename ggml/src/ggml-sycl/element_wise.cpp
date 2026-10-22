#അന്തര്ഭാവ "common.hpp"
#അന്തര്ഭാവ "ggml-sycl/presets.hpp"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "element_wise.hpp"

#define SYCL_GLOBAL_ID_LOOP(K, ITEM) \
    വേണ്ടി (auto i = ITEM.get_global_id(0); i < (size_t)K; i += ITEM.get_global_range(0))

#define SYCL_LOCAL_ID_CALC(ITEM, IDX) \
    (ITEM.get_local_range(IDX) * ITEM.get_group(IDX) + ITEM.get_local_id(IDX))

സ്ഥിതം ശൂന്യം acc_f32(സ്ഥിരം അക്ഷരം * x, സ്ഥിരം അക്ഷരം * y, ദശാംശം * dst, സ്ഥിരം int64_t ne,
    സ്ഥിരം int64_t ne0, സ്ഥിരം int64_t ne1, സ്ഥിരം int64_t ne2,
    സ്ഥിരം int64_t nb00, സ്ഥിരം int64_t nb01, സ്ഥിരം int64_t nb02, സ്ഥിരം int64_t nb03,
    സ്ഥിരം int64_t ne10, സ്ഥിരം int64_t ne11, സ്ഥിരം int64_t ne12, സ്ഥിരം int64_t ne13,
    സ്ഥിരം int64_t nb10, സ്ഥിരം int64_t nb11, സ്ഥിരം int64_t nb12, സ്ഥിരം int64_t nb13,
    സ്ഥിരം int64_t s11, സ്ഥിരം int64_t s12, സ്ഥിരം int64_t s13, സ്ഥിരം int64_t offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം int64_t i = SYCL_LOCAL_ID_CALC(item_ct1, 2);

    എങ്കിൽ (i >= ne) {
        തിരികെ;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    സ്ഥിരം int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    സ്ഥിരം int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    സ്ഥിരം int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    സ്ഥിരം int64_t i10 = tmp;

    int64_t tmp_dst = i;
    സ്ഥിരം int64_t i3 = tmp_dst / (ne2*ne1*ne0);
    tmp_dst -= i3 * (ne2*ne1*ne0);
    സ്ഥിരം int64_t i2 = tmp_dst / (ne1*ne0);
    tmp_dst -= i2 * (ne1*ne0);
    സ്ഥിരം int64_t i1 = tmp_dst / ne0;
    tmp_dst -= i1 * ne0;
    സ്ഥിരം int64_t i0 = tmp_dst;

    ദശാംശം val = *(സ്ഥിരം ദശാംശം *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    എങ്കിൽ (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(സ്ഥിരം ദശാംശം *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
template<typename T>
സ്ഥിതം __dpct_inline__ T op_sgn(T x) {
    തിരികെ x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


template<typename T>
സ്ഥിതം __dpct_inline__ T op_abs(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::fabs(x);
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_expm1(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<ദശാംശം>(x))
        );
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::expm1(x);
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_elu(T x) {
    തിരികെ (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_erf(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<ദശാംശം>(x))
        );
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::erf(x);
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_gelu_erf(T x) {
    സ്ഥിരം T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    തിരികെ static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_gelu_quick(T x) {
    സ്ഥിരം T GELU_QUICK_COEF_LOCAL = static_cast<T>(-1.702f);
    തിരികെ x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_QUICK_COEF_LOCAL * x)));
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_relu(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::fmax(x, static_cast<T>(0));
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_sigmoid(T x) {
    തിരികെ static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_sqrt(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ sycl::ext::oneapi::experimental::sqrt(x);
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::sqrt(x);
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_sin(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ sycl::ext::oneapi::experimental::sin(x);
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::sin(x);
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_cos(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ sycl::ext::oneapi::experimental::cos(x);
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::cos(x);
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_hardsigmoid(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_hardswish(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } അല്ലെങ്കിൽ {
        തിരികെ x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_log(T x) {
    എങ്കിൽ (x <= static_cast<T>(0)) {
        തിരികെ neg_infinity<T>();
    }
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ sycl::ext::oneapi::experimental::log(x);
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::log(x);
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_softplus(T x) {
    സ്ഥിരം ദശാംശം xf = (ദശാംശം) x;
    സ്ഥിരം ദശാംശം ax = op_abs(xf);
    സ്ഥിരം ദശാംശം m  = sycl::fmax(xf, 0.0f);
    സ്ഥിരം ദശാംശം y  = m + sycl::log1p(sycl::exp(-ax));
    തിരികെ (T) y;
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_neg(T x) {
    തിരികെ -x;
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_step(T x) {
    തിരികെ (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_leaky_relu(T x, ദശാംശം negative_slope) {
    T neg_slope_T = static_cast<T>(negative_slope);
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;

    } അല്ലെങ്കിൽ {
        തിരികെ sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_xielu(T x, ദശാംശം alpha_n, ദശാംശം alpha_p, ദശാംശം beta, ദശാംശം eps) {
    സ്ഥിരം ദശാംശം xi        = static_cast<ദശാംശം>(x);
    സ്ഥിരം ദശാംശം gate_pos  = (xi > 0.0f);
    സ്ഥിരം ദശാംശം y_pos     = alpha_p * xi * xi + beta * xi;
    സ്ഥിരം ദശാംശം min_v_eps = sycl::fmin(xi, eps);
    സ്ഥിരം ദശാംശം y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    സ്ഥിരം ദശാംശം out       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    തിരികെ static_cast<T>(out);
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_sqr(T x) {
    തിരികെ x * x;
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_clamp(T x, ദശാംശം min_val, ദശാംശം max_val) {
    തിരികെ x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_floor(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ sycl::ext::oneapi::experimental::floor(x);
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::floor(x);
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_ceil(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ sycl::ext::oneapi::experimental::ceil(x);
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::ceil(x);
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_round(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<ദശാംശം>(x))
        );
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::round(x);
    }
}

template<typename T>
സ്ഥിതം __dpct_inline__ T op_trunc(T x) {
    എങ്കിൽ constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        തിരികെ sycl::ext::oneapi::experimental::trunc(x);
    } അല്ലെങ്കിൽ {
        തിരികെ sycl::trunc(x);
    }
}

template<typename T, typename F>
സ്ഥിതം ശൂന്യം unary_op_flat_kernel(സ്ഥിരം T * x, T * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ k, സ്ഥിരം sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]);
    }
}

template<typename T, typename F>
സ്ഥിതം ശൂന്യം unary_op_generic_kernel(
        സ്ഥിരം T * x,
        T * dst,
        സ്ഥിരം പൂർണ്ണസംഖ്യ k,
        സ്ഥിരം sycl::uint3 ne0_fd, സ്ഥിരം sycl::uint3 ne1_fd, സ്ഥിരം sycl::uint3 ne2_fd,
        സ്ഥിരം size_t nb0,  സ്ഥിരം size_t nb1,  സ്ഥിരം size_t nb2,  സ്ഥിരം size_t nb3,
        സ്ഥിരം size_t nbd0, സ്ഥിരം size_t nbd1, സ്ഥിരം size_t nbd2, സ്ഥിരം size_t nbd3,
        സ്ഥിരം sycl::nd_item<1> & item_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        സ്ഥിരം uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        സ്ഥിരം uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        സ്ഥിരം uint32_t i2 = dm.y();
        സ്ഥിരം uint32_t i3 = dm.x();

        സ്ഥിരം അക്ഷരം * src_base = (സ്ഥിരം അക്ഷരം *) x;
        അക്ഷരം       * dst_base = (അക്ഷരം *) dst;

        സ്ഥിരം T * srcp = (സ്ഥിരം T *)(src_base + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_base + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

template<typename T>
സ്ഥിതം ശൂന്യം unary_op_sqrt_kernel(സ്ഥിരം T * x, T * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ k, സ്ഥിരം sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

template<typename T>
സ്ഥിതം ശൂന്യം unary_op_sin_kernel(സ്ഥിരം T * x, T * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ k, സ്ഥിരം sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

template<typename T>
സ്ഥിതം ശൂന്യം unary_op_cos_kernel(സ്ഥിരം T * x, T * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ k, സ്ഥിരം sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

template<typename T>
സ്ഥിതം ശൂന്യം unary_op_log_kernel(സ്ഥിരം T * x, T * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ k, സ്ഥിരം sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_log(x[i]);
    }
}


template<typename T>
സ്ഥിതം ശൂന്യം unary_op_leaky_relu_kernel(സ്ഥിരം T * x, T * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ k, ദശാംശം negative_slope, സ്ഥിരം sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_slope);
    }
}

template<typename T>
സ്ഥിതം ശൂന്യം unary_op_xielu_kernel(സ്ഥിരം T * x, T * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ k, ദശാംശം alpha_n, ദശാംശം alpha_p, ദശാംശം beta, ദശാംശം eps, സ്ഥിരം sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

template<typename T>
സ്ഥിതം ശൂന്യം unary_op_sqr_kernel(സ്ഥിരം T * x, T * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ k, സ്ഥിരം sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

template<typename T>
സ്ഥിതം ശൂന്യം unary_op_clamp_kernel(സ്ഥിരം T * x, T * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ k, സ്ഥിരം sycl::nd_item<1> &item_ct1, ദശാംശം min_val, ദശാംശം max_val) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

template<typename T>
സ്ഥിതം ശൂന്യം unary_op_ceil_kernel(സ്ഥിരം T * x, T * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ k, സ്ഥിരം sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

template<typename T>
സ്ഥിതം ശൂന്യം clamp(സ്ഥിരം T * x, T * dst, സ്ഥിരം ദശാംശം min, സ്ഥിരം ദശാംശം max, സ്ഥിരം പൂർണ്ണസംഖ്യ k,
                      സ്ഥിരം sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

template<typename T, typename F>
സ്ഥിതം ശൂന്യം unary_gated_op_flat_kernel(സ്ഥിരം T * x, സ്ഥിരം T * g, T * dst, സ്ഥിരം uint64_t k, സ്ഥിരം sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

template<typename T, typename F>
സ്ഥിതം ശൂന്യം unary_gated_op_generic_kernel(
        സ്ഥിരം T * x,
        സ്ഥിരം T * g,
        T * dst,
        സ്ഥിരം uint64_t k,
        സ്ഥിരം sycl::uint3 n_fd,
        സ്ഥിരം uint64_t o0,
        സ്ഥിരം uint64_t o1,
        സ്ഥിരം sycl::nd_item<1> & item_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        സ്ഥിരം sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        സ്ഥിരം int64_t j0 = rc.x() * o0 + rc.y();
        സ്ഥിരം int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
template<typename T, typename F>
സ്ഥിതം ശൂന്യം unary_mul_flat_kernel(സ്ഥിരം T * x, സ്ഥിരം T * g, T * dst, സ്ഥിരം int64_t k, സ്ഥിരം sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = (T) (op((ദശാംശം) x[i]) * (ദശാംശം) g[i]);
    }
}

template<typename T, typename F>
സ്ഥിതം ശൂന്യം unary_mul_strided_kernel(സ്ഥിരം T * x, സ്ഥിരം T * g, T * dst, സ്ഥിരം int64_t k, സ്ഥിരം sycl::uint3 n_fd, സ്ഥിരം int64_t o0, സ്ഥിരം int64_t o1, സ്ഥിരം sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        സ്ഥിരം sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        സ്ഥിരം int64_t j0 = rc.x() * o0 + rc.y();
        സ്ഥിരം int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((ദശാംശം) x[j0]) * (ദശാംശം) g[j1]);
    }
}

template<typename T, typename F>
സ്ഥിതം ശൂന്യം unary_mul_sycl(സ്ഥിരം T * x, സ്ഥിരം T * g, T * dst, സ്ഥിരം int64_t k, സ്ഥിരം int64_t n, സ്ഥിരം int64_t o0, സ്ഥിരം int64_t o1, queue_ptr main_stream, F op) {
    സ്ഥിരം size_t            അക്കം_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    സ്ഥിരം sycl::nd_range<1> range(അക്കം_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    എങ്കിൽ (o0 == n && o1 == n) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            unary_mul_flat_kernel(x, g, dst, k, item_ct1, op);
        });
        തിരികെ;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    സ്ഥിരം sycl::uint3 n_fd = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) n);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, item_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
template<typename F>
സ്ഥിതം ശൂന്യം add_unary_mul_flat_kernel(സ്ഥിരം ദശാംശം * a, സ്ഥിരം ദശാംശം * bias, സ്ഥിരം ദശാംശം * scale, ദശാംശം * dst,
                                      സ്ഥിരം int64_t k, സ്ഥിരം sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

template<typename F>
സ്ഥിതം ശൂന്യം add_unary_mul_bcast_kernel(സ്ഥിരം ദശാംശം * a, സ്ഥിരം ദശാംശം * bias, സ്ഥിരം ദശാംശം * scale, ദശാംശം * dst,
                                       സ്ഥിരം int64_t k, സ്ഥിരം sycl::uint3 ne0_fd, സ്ഥിരം sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        സ്ഥിരം uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

template<typename F>
സ്ഥിതം ശൂന്യം add_unary_mul_sycl(സ്ഥിരം ദശാംശം * a, സ്ഥിരം ദശാംശം * bias, സ്ഥിരം ദശാംശം * scale, ദശാംശം * dst,
                               സ്ഥിരം int64_t k, സ്ഥിരം int64_t ne0, queue_ptr main_stream, F op) {
    സ്ഥിരം size_t            അക്കം_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    സ്ഥിരം sycl::nd_range<1> range(അക്കം_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    എങ്കിൽ (k == ne0) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            add_unary_mul_flat_kernel(a, bias, scale, dst, k, item_ct1, op);
        });
        തിരികെ;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    സ്ഥിരം sycl::uint3 ne0_fd = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) ne0);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        add_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, item_ct1, op);
    });
}

നാമാകാശ ggml_sycl_detail {
സ്ഥിതം ശൂന്യം acc_f32_sycl(സ്ഥിരം അക്ഷരം *x, സ്ഥിരം അക്ഷരം *y, ദശാംശം *dst,
                         സ്ഥിരം int64_t n_elements,
                         സ്ഥിരം int64_t ne0, സ്ഥിരം int64_t ne1, സ്ഥിരം int64_t ne2,
                         സ്ഥിരം int64_t nb00, സ്ഥിരം int64_t nb01, സ്ഥിരം int64_t nb02, സ്ഥിരം int64_t nb03,
                         സ്ഥിരം int64_t ne10, സ്ഥിരം int64_t ne11, സ്ഥിരം int64_t ne12, സ്ഥിരം int64_t ne13,
                         സ്ഥിരം int64_t nb10, സ്ഥിരം int64_t nb11, സ്ഥിരം int64_t nb12, സ്ഥിരം int64_t nb13,
                         സ്ഥിരം int64_t s1, സ്ഥിരം int64_t s2, സ്ഥിരം int64_t s3,
                         സ്ഥിരം int64_t offset, queue_ptr stream) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, അക്കം_blocks) * sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
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
സ്ഥിതം ശൂന്യം arange_kernel(T * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ k, T start, T step,
                         സ്ഥിരം sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = start + static_cast<T>(i) * step;
    }
}

template<typename KernelInvoker, typename... Args>
സ്ഥിതം വരിയിൽ ശൂന്യം dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    തിരഞ്ഞെടുപ്പ് (dst->type) {
        അവസ്ഥ GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (പൂർണ്ണസംഖ്യ)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                നിർത്തുക;
            }
#ifdef GGML_SYCL_HAS_BF16
        അവസ്ഥ GGML_TYPE_BF16:
            {
                auto data_pts = cast_data<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (പൂർണ്ണസംഖ്യ)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                നിർത്തുക;
            }
#endif
        അവസ്ഥ GGML_TYPE_F32:
            {
                auto data_pts = cast_data<ദശാംശം>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (പൂർണ്ണസംഖ്യ)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                നിർത്തുക;
            }
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename KernelInvoker, typename... Args>
സ്ഥിതം വരിയിൽ ശൂന്യം dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    സ്ഥിരം int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // The fused GLU kernels index with 32-bit fastdiv, which is exact only for indices below
    // 2^31. A dst that large is ~8 GB at f32, and the grid sizing already narrows to 32 bits,
    // so assert the bound rather than carry a second code path for it.
    GGML_ASSERT(ggml_nelements(dst) < ((int64_t) 1 << 31));
    സ്ഥിരം int32_t swapped = ((സ്ഥിരം int32_t *) dst->op_params)[1];
    ശൂന്യം * src0_d = src0->data;
    ശൂന്യം * src1_d = src1 ? src1->data : src0->data;
    സ്ഥിരം int64_t src0_o = src0->nb[1];
    സ്ഥിരം int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ശൂന്യം * dst_d = dst->data;
    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    തിരഞ്ഞെടുപ്പ് (dst->type) {
        അവസ്ഥ GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    എങ്കിൽ (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / വലുപ്പം(sycl::half),
                               src1_o / വലുപ്പം(sycl::half),
                               main_stream,
                               std::forward<Args>(args)...);
                നിർത്തുക;
            }
        അവസ്ഥ GGML_TYPE_F32:
            {
                ദശാംശം * src0_p = (ദശാംശം *) src0_d;
                ദശാംശം * src1_p = (ദശാംശം *) src1_d;

                    എങ്കിൽ (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (ദശാംശം *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / വലുപ്പം(ദശാംശം),
                               src1_o / വലുപ്പം(ദശാംശം),
                               main_stream,
                               std::forward<Args>(args)...);
                നിർത്തുക;
            }
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename F>
സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    സ്ഥിരം int64_t ne0  = dst->ne[0];
    സ്ഥിരം int64_t ne1  = dst->ne[1];
    സ്ഥിരം int64_t ne2  = dst->ne[2];

    സ്ഥിരം size_t  nb0  = src0->nb[0];
    സ്ഥിരം size_t  nb1  = src0->nb[1];
    സ്ഥിരം size_t  nb2  = src0->nb[2];
    സ്ഥിരം size_t  nb3  = src0->nb[3];

    സ്ഥിരം size_t  nbd0 = dst->nb[0];
    സ്ഥിരം size_t  nbd1 = dst->nb[1];
    സ്ഥിരം size_t  nbd2 = dst->nb[2];
    സ്ഥിരം size_t  nbd3 = dst->nb[3];

    // Hot unary ops (FFN/GDN silu, sigmoid, ...) run on contiguous tensors;
    // skip the strided index math entirely for them.
    സ്ഥിരം തർക്കം contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](സ്ഥിരം auto* src, auto* dst_ptr, പൂർണ്ണസംഖ്യ k_elements, queue_ptr stream) {

            സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = ceil_div(k_elements, 256);

            എങ്കിൽ (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, item_ct1, func);
                    });
            } അല്ലെങ്കിൽ {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                സ്ഥിരം sycl::uint3 ne0_fd = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) ne0);
                സ്ഥിരം sycl::uint3 ne1_fd = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) ne1);
                സ്ഥിരം sycl::uint3 ne2_fd = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks) * sycl::range<1>(256),
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
സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](സ്ഥിരം auto * x_ptr, സ്ഥിരം auto * g_ptr, auto * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {

            സ്ഥിരം uint32_t അക്കം_blocks = (uint32_t) ceil_div(k, SYCL_GLU_BLOCK_SIZE);
            സ്ഥിരം sycl::nd_range<1> launch_range(അക്കം_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE),
                                                 sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            എങ്കിൽ (o0 == n && o1 == n) {
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, item_ct1, func);
                    });
            } അല്ലെങ്കിൽ {
                // launch-invariant divisor, and only this path needs it
                സ്ഥിരം sycl::uint3 n_fd = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) n);
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, item_ct1, func);
                    });
            }
        });
}


സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    ദശാംശം start, stop, step;
    memcpy(&start, dst->op_params, വലുപ്പം(ദശാംശം));
    memcpy(&stop, (ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));
    memcpy(&step, (ദശാംശം *) dst->op_params + 2, വലുപ്പം(ദശാംശം));
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ദശാംശം * dst_ptr = (ദശാംശം *)dst->data;
    സ്ഥിരം പൂർണ്ണസംഖ്യ k = (പൂർണ്ണസംഖ്യ)ggml_nelements(dst);
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = ceil_div(k, SYCL_ARANGE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks) * sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE),
                          sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            arange_kernel(dst_ptr, k, start, step, item_ct1);
        });
}

} // namespace ggml_sycl_detail



സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_sgn(x);
    });
}


സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_abs(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_elu(x);
    });
}
സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_silu(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_gelu(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_gelu_quick(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_gelu_erf(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_tanh(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_relu(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_hardsigmoid(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_hardswish(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_exp(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_expm1(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](സ്ഥിരം auto* src, auto* dst_ptr, പൂർണ്ണസംഖ്യ k_elements, queue_ptr stream) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_softplus(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_neg(x);
    });
}


സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_step(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_sigmoid(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](സ്ഥിരം auto* src, auto* dst_ptr, പൂർണ്ണസംഖ്യ k_elements, queue_ptr stream) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = ceil_div(k_elements, SYCL_SQRT_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks) * sycl::range<1>(SYCL_SQRT_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQRT_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](സ്ഥിരം auto* src, auto* dst_ptr, പൂർണ്ണസംഖ്യ k_elements, queue_ptr stream) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](സ്ഥിരം auto* src, auto* dst_ptr, പൂർണ്ണസംഖ്യ k_elements, queue_ptr stream) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ദശാംശം negative_slope;
    memcpy(&negative_slope, dst->op_params, വലുപ്പം(ദശാംശം));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](സ്ഥിരം auto* src, auto* dst_ptr, പൂർണ്ണസംഖ്യ k_elements, queue_ptr stream, ദശാംശം slope) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, slope, item_ct1);
                });
        }, negative_slope);
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](സ്ഥിരം auto* src, auto* dst_ptr, പൂർണ്ണസംഖ്യ k_elements, queue_ptr stream) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = ceil_div(k_elements, SYCL_SQR_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks) * sycl::range<1>(SYCL_SQR_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQR_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ദശാംശം min_val;
    ദശാംശം max_val;
    memcpy(&min_val, dst->op_params, വലുപ്പം(ദശാംശം));
    memcpy(&max_val, (ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](സ്ഥിരം auto* src, auto* dst_ptr, പൂർണ്ണസംഖ്യ k_elements, queue_ptr stream, ദശാംശം min_arg, ദശാംശം max_arg) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = ceil_div(k_elements, SYCL_CLAMP_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks) * sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, item_ct1);
                });
        }, min_val, max_val);
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    സ്ഥിരം ദശാംശം alpha_n = ggml_get_op_params_f32(dst, 1);
    സ്ഥിരം ദശാംശം alpha_p = ggml_get_op_params_f32(dst, 2);
    സ്ഥിരം ദശാംശം beta    = ggml_get_op_params_f32(dst, 3);
    സ്ഥിരം ദശാംശം eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](സ്ഥിരം auto* src, auto* dst_ptr, പൂർണ്ണസംഖ്യ k_elements, queue_ptr stream, ദശാംശം alpha_n_arg, ദശാംശം alpha_p_arg, ദശാംശം beta_arg, ദശാംശം eps_arg) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, item_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_floor(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_ceil(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_round(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        തിരികെ op_trunc(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    സ്ഥിരം അക്ഷരം  * src0_d = (സ്ഥിരം അക്ഷരം  *) src0->data;
    സ്ഥിരം അക്ഷരം  * src1_d = (സ്ഥിരം അക്ഷരം  *) src1->data;
    ദശാംശം       * dst_d  = (ദശാംശം       *)  dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    സ്ഥിരം int64_t s1     = (int64_t) ((സ്ഥിരം int32_t *) dst->op_params)[0] / (int64_t) വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t s2     = (int64_t) ((സ്ഥിരം int32_t *) dst->op_params)[1] / (int64_t) വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t s3     = (int64_t) ((സ്ഥിരം int32_t *) dst->op_params)[2] / (int64_t) വലുപ്പം(ദശാംശം);
    സ്ഥിരം int64_t offset = (int64_t) ((സ്ഥിരം int32_t *) dst->op_params)[3] / (int64_t) വലുപ്പം(ദശാംശം);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, offset, stream);
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        തിരികെ op_gelu(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        തിരികെ op_relu(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        തിരികെ op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
template<typename F>
സ്ഥിതം ശൂന്യം dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
    തിരഞ്ഞെടുപ്പ് (uop) {
        അവസ്ഥ GGML_UNARY_OP_SILU:     launch([](ദശാംശം v) { തിരികെ op_silu(v); });     നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_SIGMOID:  launch([](ദശാംശം v) { തിരികെ op_sigmoid(v); });  നിർത്തുക;
        അവസ്ഥ GGML_UNARY_OP_SOFTPLUS: launch([](ദശാംശം v) { തിരികെ op_softplus(v); }); നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
    }
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
ശൂന്യം ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_കണ്ണി, ggml_tensor * mul_കണ്ണി) {
    scope_op_debug_print scope_dbg_print(__func__, mul_കണ്ണി, /*num_src=*/2);

    സ്ഥിരം ggml_tensor * x = unary_കണ്ണി->src[0];
    സ്ഥിരം ggml_tensor * g = (mul_കണ്ണി->src[0] == unary_കണ്ണി) ? mul_കണ്ണി->src[1] : mul_കണ്ണി->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_കണ്ണി);
    GGML_ASSERT(x->type == g->type && x->type == mul_കണ്ണി->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_കണ്ണി));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_കണ്ണി));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    സ്ഥിരം int64_t k = ggml_nelements(mul_കണ്ണി);
    സ്ഥിരം int64_t n = mul_കണ്ണി->ne[0];

    സ്ഥിരം auto dispatch_type = [&](auto op) {
        തിരഞ്ഞെടുപ്പ് (mul_കണ്ണി->type) {
            അവസ്ഥ GGML_TYPE_F32:
                unary_mul_sycl((സ്ഥിരം ദശാംശം *) x->data, (സ്ഥിരം ദശാംശം *) g->data, (ദശാംശം *) mul_കണ്ണി->data,
                               k, n, x->nb[1] / വലുപ്പം(ദശാംശം), g->nb[1] / വലുപ്പം(ദശാംശം), main_stream, op);
                നിർത്തുക;
            അവസ്ഥ GGML_TYPE_F16:
                unary_mul_sycl((സ്ഥിരം sycl::half *) x->data, (സ്ഥിരം sycl::half *) g->data, (sycl::half *) mul_കണ്ണി->data,
                               k, n, x->nb[1] / വലുപ്പം(sycl::half), g->nb[1] / വലുപ്പം(sycl::half), main_stream, op);
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_കണ്ണി->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_കണ്ണി), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
ശൂന്യം ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_കണ്ണി,
                                      ggml_tensor * unary_കണ്ണി, ggml_tensor * mul_കണ്ണി) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_കണ്ണി, /*num_src=*/2);

    സ്ഥിരം ggml_tensor * a     = add_കണ്ണി->src[0];
    സ്ഥിരം ggml_tensor * bias  = add_കണ്ണി->src[1];
    സ്ഥിരം ggml_tensor * scale = (mul_കണ്ണി->src[0] == unary_കണ്ണി) ? mul_കണ്ണി->src[1] : mul_കണ്ണി->src[0];

    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(scale != unary_കണ്ണി);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_കണ്ണി->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_കണ്ണി));
    // a and dst are indexed flat
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_കണ്ണി));
    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    സ്ഥിരം auto dispatch_op = [&](auto op) {
        add_unary_mul_sycl((സ്ഥിരം ദശാംശം *) a->data, (സ്ഥിരം ദശാംശം *) bias->data, (സ്ഥിരം ദശാംശം *) scale->data,
                           (ദശാംശം *) mul_കണ്ണി->data, ggml_nelements(mul_കണ്ണി), mul_കണ്ണി->ne[0], main_stream, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_കണ്ണി), dispatch_op);
}

__dpct_inline__ ദശാംശം ggml_sycl_op_swiglu_oai_single(ദശാംശം x, ദശാംശം g, ദശാംശം alpha = 1.702f, ദശാംശം limit = 7.0f) {
    x = sycl::fmin(x, limit);
    g = sycl::fmax(sycl::fmin(g, limit), -limit);

    ദശാംശം out_glu = x / (1.0f + sycl::native::exp(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    തിരികെ out_glu;
}

template <typename T>
സ്ഥിതം ശൂന്യം swiglu_oai_kernel(സ്ഥിരം T * x, സ്ഥിരം T * g, T * dst, സ്ഥിരം int64_t k,
                              സ്ഥിരം int64_t n, സ്ഥിരം int64_t o0, സ്ഥിരം int64_t o1,
                              ദശാംശം alpha, ദശാംശം limit, sycl::nd_item<3> item_ct1) {
    സ്ഥിരം int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    എങ്കിൽ (i >= k) {
        തിരികെ;
    }

    സ്ഥിരം int64_t j0 = (i / n) * o0 + (i % n);
    സ്ഥിരം int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ദശാംശം xi = x[j0];
    ദശാംശം gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
സ്ഥിതം ശൂന്യം swiglu_oai_sycl(സ്ഥിരം T *       x,
                            സ്ഥിരം T *       g,
                            T *             dst,
                            സ്ഥിരം int64_t   k,
                            സ്ഥിരം int64_t   n,
                            സ്ഥിരം int64_t   o0,
                            സ്ഥിരം int64_t   o1,
                            സ്ഥിരം ദശാംശം     alpha,
                            സ്ഥിരം ദശാംശം     limit,
                            dpct::queue_ptr stream) {
    സ്ഥിരം int64_t അക്കം_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, അക്കം_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, limit, item_ct1);
                         });
}

സ്ഥിതം ശൂന്യം ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];
    ശൂന്യം * src0_d = src0->data;
    ശൂന്യം * src1_d = src1 ? src1->data : src0->data;
    സ്ഥിരം int64_t src0_o = src0->nb[1];
    സ്ഥിരം int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ശൂന്യം * dst_d = dst->data;
    സ്ഥിരം int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);
    സ്ഥിരം ദശാംശം alpha = ggml_get_op_params_f32(dst, 2);
    സ്ഥിരം ദശാംശം limit = ggml_get_op_params_f32(dst, 3);

    ദശാംശം * src0_p = (ദശാംശം *) src0_d;
    ദശാംശം * src1_p = (ദശാംശം *) src1_d;

    എങ്കിൽ (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (ദശാംശം *)dst_d, ggml_nelements(dst), nc, src0_o / വലുപ്പം(ദശാംശം), src1_o / വലുപ്പം(ദശാംശം), alpha, limit, stream);
}

template <typename T>
സ്ഥിതം ശൂന്യം swiglu_clamp_kernel(സ്ഥിരം T *        gate,
                                സ്ഥിരം T *        up,
                                T *              dst,
                                സ്ഥിരം int64_t    k,
                                സ്ഥിരം int64_t    n,
                                സ്ഥിരം int64_t    o0,
                                സ്ഥിരം int64_t    o1,
                                ദശാംശം            limit,
                                sycl::nd_item<3> item_ct1) {
    സ്ഥിരം int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    എങ്കിൽ (i >= k) {
        തിരികെ;
    }

    സ്ഥിരം int64_t j0 = (i / n) * o0 + (i % n);
    സ്ഥിരം int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    സ്ഥിരം ദശാംശം gate_മൂല്യം = sycl::fmin((ദശാംശം) gate[j0], limit);
    സ്ഥിരം ദശാംശം up_മൂല്യം   = sycl::fmax(sycl::fmin((ദശാംശം) up[j1], limit), -limit);
    dst[i]                 = (T) (gate_മൂല്യം / (1.0f + sycl::native::exp(-gate_മൂല്യം)) * up_മൂല്യം);
}

template <typename T>
സ്ഥിതം ശൂന്യം swiglu_clamp_sycl(സ്ഥിരം T *       gate,
                              സ്ഥിരം T *       up,
                              T *             dst,
                              സ്ഥിരം int64_t   k,
                              സ്ഥിരം int64_t   n,
                              സ്ഥിരം int64_t   o0,
                              സ്ഥിരം int64_t   o1,
                              ദശാംശം           limit,
                              dpct::queue_ptr stream) {
    സ്ഥിരം int64_t അക്കം_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, അക്കം_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_clamp_kernel(gate, up, dst, k, n, o0, o1, limit, item_ct1);
                         });
}

സ്ഥിതം ശൂന്യം ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0   = dst->src[0];
    സ്ഥിരം ggml_tensor * src1   = dst->src[1];
    ശൂന്യം *              src0_d = src0->data;
    ശൂന്യം *              src1_d = src1 ? src1->data : src0->data;
    സ്ഥിരം int64_t       src0_o = src0->nb[1];
    സ്ഥിരം int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    ശൂന്യം *              dst_d  = dst->data;
    സ്ഥിരം int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    എങ്കിൽ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    സ്ഥിരം int32_t swapped = ggml_get_op_params_i32(dst, 1);
    സ്ഥിരം ദശാംശം   limit   = ggml_get_op_params_f32(dst, 3);

    എങ്കിൽ (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / വലുപ്പം(sycl::half),
                          src1_o / വലുപ്പം(sycl::half), limit, stream);
    } അല്ലെങ്കിൽ {
        ദശാംശം * src0_p = (ദശാംശം *) src0_d;
        ദശാംശം * src1_p = (ദശാംശം *) src1_d;

        എങ്കിൽ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (ദശാംശം *) dst_d, ggml_nelements(dst), nc, src0_o / വലുപ്പം(ദശാംശം),
                          src1_o / വലുപ്പം(ദശാംശം), limit, stream);
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        തിരികെ op_gelu_erf(x);
    });
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_op_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        തിരികെ op_gelu_quick(x);
    });
}


ശൂന്യം ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

ശൂന്യം ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

ശൂന്യം ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

ശൂന്യം ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

ശൂന്യം ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

ശൂന്യം ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

ശൂന്യം ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_quick(ctx, dst);
}

ശൂന്യം ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

ശൂന്യം ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

ശൂന്യം ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

ശൂന്യം ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

ശൂന്യം ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

ശൂന്യം ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

ശൂന്യം ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

ശൂന്യം ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_expm1(ctx, dst);
}

ശൂന്യം ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

ശൂന്യം ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

ശൂന്യം ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

ശൂന്യം ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_step(ctx, dst);
}

ശൂന്യം ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

ശൂന്യം ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

ശൂന്യം ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

ശൂന്യം ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_xielu(ctx, dst);
}

ശൂന്യം ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

ശൂന്യം ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

ശൂന്യം ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

ശൂന്യം ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

ശൂന്യം ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

ശൂന്യം ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

ശൂന്യം ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

ശൂന്യം ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_clamp(ctx, dst);
}

ശൂന്യം ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

ശൂന്യം ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_quick(ctx, dst);
}

ശൂന്യം ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

ശൂന്യം ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

ശൂന്യം ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

ശൂന്യം ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_round(ctx, dst);
}

ശൂന്യം ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
