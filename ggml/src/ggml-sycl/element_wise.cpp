#समावेश "common.hpp"
#समावेश "ggml-sycl/presets.hpp"
#समावेश "ggml.h"
#समावेश "element_wise.hpp"

#घोषणा SYCL_GLOBAL_ID_LOOP(K, ITEM) \
    क्रम (स्वतः i = ITEM.get_global_id(0); i < (माप_प्रकार)K; i += ITEM.get_global_range(0))

#घोषणा SYCL_LOCAL_ID_CALC(ITEM, IDX) \
    (ITEM.get_local_range(IDX) * ITEM.get_group(IDX) + ITEM.get_local_id(IDX))

अटल व्योम acc_f32(स्थिर अक्षर * x, स्थिर अक्षर * y, भग्न * dst, स्थिर int64_t ne,
    स्थिर int64_t ne0, स्थिर int64_t ne1, स्थिर int64_t ne2,
    स्थिर int64_t nb00, स्थिर int64_t nb01, स्थिर int64_t nb02, स्थिर int64_t nb03,
    स्थिर int64_t ne10, स्थिर int64_t ne11, स्थिर int64_t ne12, स्थिर int64_t ne13,
    स्थिर int64_t nb10, स्थिर int64_t nb11, स्थिर int64_t nb12, स्थिर int64_t nb13,
    स्थिर int64_t s11, स्थिर int64_t s12, स्थिर int64_t s13, स्थिर int64_t offset) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर int64_t i = SYCL_LOCAL_ID_CALC(item_ct1, 2);

    अगर (i >= ne) {
        वापस;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    स्थिर int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    स्थिर int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    स्थिर int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    स्थिर int64_t i10 = tmp;

    int64_t tmp_dst = i;
    स्थिर int64_t i3 = tmp_dst / (ne2*ne1*ne0);
    tmp_dst -= i3 * (ne2*ne1*ne0);
    स्थिर int64_t i2 = tmp_dst / (ne1*ne0);
    tmp_dst -= i2 * (ne1*ne0);
    स्थिर int64_t i1 = tmp_dst / ne0;
    tmp_dst -= i1 * ne0;
    स्थिर int64_t i0 = tmp_dst;

    भग्न val = *(स्थिर भग्न *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    अगर (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(स्थिर भग्न *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
ढाँचा<typename T>
अटल __dpct_inline__ T op_sgn(T x) {
    वापस x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


ढाँचा<typename T>
अटल __dpct_inline__ T op_abs(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::experimental::भ_असल(x);  // or experimental namespace if needed
    } अन्यथा {
        वापस sycl::भ_असल(x);
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_expm1(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<भग्न>(x))
        );
    } अन्यथा {
        वापस sycl::expm1(x);
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_elu(T x) {
    वापस (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_erf(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<भग्न>(x))
        );
    } अन्यथा {
        वापस sycl::erf(x);
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_gelu_erf(T x) {
    स्थिर T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    वापस static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_gelu_quick(T x) {
    स्थिर T GELU_QUICK_COEF_LOCAL = static_cast<T>(-1.702f);
    वापस x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_QUICK_COEF_LOCAL * x)));
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_relu(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } अन्यथा {
        वापस sycl::fmax(x, static_cast<T>(0));
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_sigmoid(T x) {
    वापस static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_sqrt(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::experimental::वर्ग_मूल(x);
    } अन्यथा {
        वापस sycl::वर्ग_मूल(x);
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_sin(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::experimental::sin(x);
    } अन्यथा {
        वापस sycl::sin(x);
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_cos(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::experimental::cos(x);
    } अन्यथा {
        वापस sycl::cos(x);
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_hardsigmoid(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } अन्यथा {
        वापस sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_hardswish(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } अन्यथा {
        वापस x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_log(T x) {
    अगर (x <= static_cast<T>(0)) {
        वापस neg_infinity<T>();
    }
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::experimental::log(x);
    } अन्यथा {
        वापस sycl::log(x);
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_softplus(T x) {
    स्थिर भग्न xf = (भग्न) x;
    स्थिर भग्न ax = op_abs(xf);
    स्थिर भग्न m  = sycl::fmax(xf, 0.0f);
    स्थिर भग्न y  = m + sycl::log1p(sycl::exp(-ax));
    वापस (T) y;
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_neg(T x) {
    वापस -x;
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_step(T x) {
    वापस (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_leaky_relu(T x, भग्न negative_slope) {
    T neg_slope_T = static_cast<T>(negative_slope);
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;

    } अन्यथा {
        वापस sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_xielu(T x, भग्न alpha_n, भग्न alpha_p, भग्न beta, भग्न eps) {
    स्थिर भग्न xi        = static_cast<भग्न>(x);
    स्थिर भग्न gate_pos  = (xi > 0.0f);
    स्थिर भग्न y_pos     = alpha_p * xi * xi + beta * xi;
    स्थिर भग्न min_v_eps = sycl::fmin(xi, eps);
    स्थिर भग्न y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    स्थिर भग्न out       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    वापस static_cast<T>(out);
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_sqr(T x) {
    वापस x * x;
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_clamp(T x, भग्न min_val, भग्न max_val) {
    वापस x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_floor(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::experimental::न्यूनमान(x);
    } अन्यथा {
        वापस sycl::न्यूनमान(x);
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_ceil(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::experimental::उच्चमान(x);
    } अन्यथा {
        वापस sycl::उच्चमान(x);
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_round(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<भग्न>(x))
        );
    } अन्यथा {
        वापस sycl::round(x);
    }
}

ढाँचा<typename T>
अटल __dpct_inline__ T op_trunc(T x) {
    अगर constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        वापस sycl::ext::oneapi::experimental::छाँटो(x);
    } अन्यथा {
        वापस sycl::छाँटो(x);
    }
}

ढाँचा<typename T, typename F>
अटल व्योम unary_op_flat_kernel(स्थिर T * x, T * dst, स्थिर पूर्णांक k, स्थिर sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]);
    }
}

ढाँचा<typename T, typename F>
अटल व्योम unary_op_generic_kernel(
        स्थिर T * x,
        T * dst,
        स्थिर पूर्णांक k,
        स्थिर sycl::uint3 ne0_fd, स्थिर sycl::uint3 ne1_fd, स्थिर sycl::uint3 ne2_fd,
        स्थिर माप_प्रकार nb0,  स्थिर माप_प्रकार nb1,  स्थिर माप_प्रकार nb2,  स्थिर माप_प्रकार nb3,
        स्थिर माप_प्रकार nbd0, स्थिर माप_प्रकार nbd1, स्थिर माप_प्रकार nbd2, स्थिर माप_प्रकार nbd3,
        स्थिर sycl::nd_item<1> & item_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        स्थिर uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        स्थिर uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        स्थिर uint32_t i2 = dm.y();
        स्थिर uint32_t i3 = dm.x();

        स्थिर अक्षर * src_base = (स्थिर अक्षर *) x;
        अक्षर       * dst_base = (अक्षर *) dst;

        स्थिर T * srcp = (स्थिर T *)(src_base + (माप_प्रकार) i0*nb0  + (माप_प्रकार) i1*nb1  + (माप_प्रकार) i2*nb2  + (माप_प्रकार) i3*nb3 );
        T *       dstp = (T *)(dst_base + (माप_प्रकार) i0*nbd0 + (माप_प्रकार) i1*nbd1 + (माप_प्रकार) i2*nbd2 + (माप_प्रकार) i3*nbd3);

        *dstp = func(*srcp);
    }
}

ढाँचा<typename T>
अटल व्योम unary_op_sqrt_kernel(स्थिर T * x, T * dst, स्थिर पूर्णांक k, स्थिर sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

ढाँचा<typename T>
अटल व्योम unary_op_sin_kernel(स्थिर T * x, T * dst, स्थिर पूर्णांक k, स्थिर sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

ढाँचा<typename T>
अटल व्योम unary_op_cos_kernel(स्थिर T * x, T * dst, स्थिर पूर्णांक k, स्थिर sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

ढाँचा<typename T>
अटल व्योम unary_op_log_kernel(स्थिर T * x, T * dst, स्थिर पूर्णांक k, स्थिर sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_log(x[i]);
    }
}


ढाँचा<typename T>
अटल व्योम unary_op_leaky_relu_kernel(स्थिर T * x, T * dst, स्थिर पूर्णांक k, भग्न negative_slope, स्थिर sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_slope);
    }
}

ढाँचा<typename T>
अटल व्योम unary_op_xielu_kernel(स्थिर T * x, T * dst, स्थिर पूर्णांक k, भग्न alpha_n, भग्न alpha_p, भग्न beta, भग्न eps, स्थिर sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

ढाँचा<typename T>
अटल व्योम unary_op_sqr_kernel(स्थिर T * x, T * dst, स्थिर पूर्णांक k, स्थिर sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

ढाँचा<typename T>
अटल व्योम unary_op_clamp_kernel(स्थिर T * x, T * dst, स्थिर पूर्णांक k, स्थिर sycl::nd_item<1> &item_ct1, भग्न min_val, भग्न max_val) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

ढाँचा<typename T>
अटल व्योम unary_op_ceil_kernel(स्थिर T * x, T * dst, स्थिर पूर्णांक k, स्थिर sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

ढाँचा<typename T>
अटल व्योम clamp(स्थिर T * x, T * dst, स्थिर भग्न min, स्थिर भग्न max, स्थिर पूर्णांक k,
                      स्थिर sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

ढाँचा<typename T, typename F>
अटल व्योम unary_gated_op_flat_kernel(स्थिर T * x, स्थिर T * g, T * dst, स्थिर uint64_t k, स्थिर sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

ढाँचा<typename T, typename F>
अटल व्योम unary_gated_op_generic_kernel(
        स्थिर T * x,
        स्थिर T * g,
        T * dst,
        स्थिर uint64_t k,
        स्थिर sycl::uint3 n_fd,
        स्थिर uint64_t o0,
        स्थिर uint64_t o1,
        स्थिर sycl::nd_item<1> & item_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        स्थिर sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        स्थिर int64_t j0 = rc.x() * o0 + rc.y();
        स्थिर int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
ढाँचा<typename T, typename F>
अटल व्योम unary_mul_flat_kernel(स्थिर T * x, स्थिर T * g, T * dst, स्थिर int64_t k, स्थिर sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = (T) (op((भग्न) x[i]) * (भग्न) g[i]);
    }
}

ढाँचा<typename T, typename F>
अटल व्योम unary_mul_strided_kernel(स्थिर T * x, स्थिर T * g, T * dst, स्थिर int64_t k, स्थिर sycl::uint3 n_fd, स्थिर int64_t o0, स्थिर int64_t o1, स्थिर sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        स्थिर sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        स्थिर int64_t j0 = rc.x() * o0 + rc.y();
        स्थिर int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((भग्न) x[j0]) * (भग्न) g[j1]);
    }
}

ढाँचा<typename T, typename F>
अटल व्योम unary_mul_sycl(स्थिर T * x, स्थिर T * g, T * dst, स्थिर int64_t k, स्थिर int64_t n, स्थिर int64_t o0, स्थिर int64_t o1, queue_ptr main_stream, F op) {
    स्थिर माप_प्रकार            अंक_blocks = ceil_div((माप_प्रकार) k, (माप_प्रकार) SYCL_GLU_BLOCK_आकार);
    स्थिर sycl::nd_range<1> range(अंक_blocks * sycl::range<1>(SYCL_GLU_BLOCK_आकार), sycl::range<1>(SYCL_GLU_BLOCK_आकार));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    अगर (o0 == n && o1 == n) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
            unary_mul_flat_kernel(x, g, dst, k, item_ct1, op);
        });
        वापस;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    स्थिर sycl::uint3 n_fd = init_fastdiv_मूल्यावली((uint32_t) n);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, item_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
ढाँचा<typename F>
अटल व्योम add_unary_mul_flat_kernel(स्थिर भग्न * a, स्थिर भग्न * bias, स्थिर भग्न * scale, भग्न * dst,
                                      स्थिर int64_t k, स्थिर sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

ढाँचा<typename F>
अटल व्योम add_unary_mul_bcast_kernel(स्थिर भग्न * a, स्थिर भग्न * bias, स्थिर भग्न * scale, भग्न * dst,
                                       स्थिर int64_t k, स्थिर sycl::uint3 ne0_fd, स्थिर sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        स्थिर uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

ढाँचा<typename F>
अटल व्योम add_unary_mul_sycl(स्थिर भग्न * a, स्थिर भग्न * bias, स्थिर भग्न * scale, भग्न * dst,
                               स्थिर int64_t k, स्थिर int64_t ne0, queue_ptr main_stream, F op) {
    स्थिर माप_प्रकार            अंक_blocks = ceil_div((माप_प्रकार) k, (माप_प्रकार) SYCL_GLU_BLOCK_आकार);
    स्थिर sycl::nd_range<1> range(अंक_blocks * sycl::range<1>(SYCL_GLU_BLOCK_आकार), sycl::range<1>(SYCL_GLU_BLOCK_आकार));

    अगर (k == ne0) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
            add_unary_mul_flat_kernel(a, bias, scale, dst, k, item_ct1, op);
        });
        वापस;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    स्थिर sycl::uint3 ne0_fd = init_fastdiv_मूल्यावली((uint32_t) ne0);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
        add_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, item_ct1, op);
    });
}

namespace ggml_sycl_detail {
अटल व्योम acc_f32_sycl(स्थिर अक्षर *x, स्थिर अक्षर *y, भग्न *dst,
                         स्थिर int64_t n_elements,
                         स्थिर int64_t ne0, स्थिर int64_t ne1, स्थिर int64_t ne2,
                         स्थिर int64_t nb00, स्थिर int64_t nb01, स्थिर int64_t nb02, स्थिर int64_t nb03,
                         स्थिर int64_t ne10, स्थिर int64_t ne11, स्थिर int64_t ne12, स्थिर int64_t ne13,
                         स्थिर int64_t nb10, स्थिर int64_t nb11, स्थिर int64_t nb12, स्थिर int64_t nb13,
                         स्थिर int64_t s1, स्थिर int64_t s2, स्थिर int64_t s3,
                         स्थिर int64_t offset, queue_ptr stream) {
    स्थिर पूर्णांक अंक_blocks = (n_elements + SYCL_ACC_BLOCK_आकार - 1) / SYCL_ACC_BLOCK_आकार;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_ACC_BLOCK_आकार),
                                           sycl::range<3>(1, 1, SYCL_ACC_BLOCK_आकार)),
                         [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                            acc_f32(x, y, dst, n_elements,
                                ne0, ne1, ne2,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                s1, s2, s3, offset);
                         });
}

ढाँचा<typename T>
अटल व्योम arange_kernel(T * dst, स्थिर पूर्णांक k, T start, T step,
                         स्थिर sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = start + static_cast<T>(i) * step;
    }
}

ढाँचा<typename KernelInvoker, typename... Args>
अटल अंतरभूत व्योम dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    चयन (dst->type) {
        हाल GGML_TYPE_F16:
            {
                स्वतः तथ्य_pts = cast_तथ्य<sycl::half>(dst);
                kernel_invoker(तथ्य_pts.src, तथ्य_pts.dst, (पूर्णांक)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                अवरोध;
            }
#अगर_घोषित GGML_SYCL_HAS_BF16
        हाल GGML_TYPE_BF16:
            {
                स्वतः तथ्य_pts = cast_तथ्य<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(तथ्य_pts.src, तथ्य_pts.dst, (पूर्णांक)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                अवरोध;
            }
#पूर्ण_अगर
        हाल GGML_TYPE_F32:
            {
                स्वतः तथ्य_pts = cast_तथ्य<भग्न>(dst);
                kernel_invoker(तथ्य_pts.src, तथ्य_pts.dst, (पूर्णांक)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                अवरोध;
            }
        शेष:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

ढाँचा<typename KernelInvoker, typename... Args>
अटल अंतरभूत व्योम dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    स्थिर int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // The fused GLU kernels index with 32-bit fastdiv, which is exact only for indices below
    // 2^31. A dst that large is ~8 GB at f32, and the grid sizing already narrows to 32 bits,
    // so assert the bound rather than carry a second code path for it.
    GGML_ASSERT(ggml_nelements(dst) < ((int64_t) 1 << 31));
    स्थिर int32_t swapped = ((स्थिर int32_t *) dst->op_params)[1];
    व्योम * src0_d = src0->तथ्य;
    व्योम * src1_d = src1 ? src1->तथ्य : src0->तथ्य;
    स्थिर int64_t src0_o = src0->nb[1];
    स्थिर int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    व्योम * dst_d = dst->तथ्य;
    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_आकार(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    चयन (dst->type) {
        हाल GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    अगर (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / माप(sycl::half),
                               src1_o / माप(sycl::half),
                               main_stream,
                               std::forward<Args>(args)...);
                अवरोध;
            }
        हाल GGML_TYPE_F32:
            {
                भग्न * src0_p = (भग्न *) src0_d;
                भग्न * src1_p = (भग्न *) src1_d;

                    अगर (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (भग्न *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / माप(भग्न),
                               src1_o / माप(भग्न),
                               main_stream,
                               std::forward<Args>(args)...);
                अवरोध;
            }
        शेष:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

ढाँचा<typename F>
अटल अंतरभूत व्योम ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    स्थिर int64_t ne0  = dst->ne[0];
    स्थिर int64_t ne1  = dst->ne[1];
    स्थिर int64_t ne2  = dst->ne[2];

    स्थिर माप_प्रकार  nb0  = src0->nb[0];
    स्थिर माप_प्रकार  nb1  = src0->nb[1];
    स्थिर माप_प्रकार  nb2  = src0->nb[2];
    स्थिर माप_प्रकार  nb3  = src0->nb[3];

    स्थिर माप_प्रकार  nbd0 = dst->nb[0];
    स्थिर माप_प्रकार  nbd1 = dst->nb[1];
    स्थिर माप_प्रकार  nbd2 = dst->nb[2];
    स्थिर माप_प्रकार  nbd3 = dst->nb[3];

    // Hot unary ops (FFN/GDN silu, sigmoid, ...) run on contiguous tensors;
    // skip the strided index math entirely for them.
    स्थिर bool contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](स्थिर स्वतः* src, स्वतः* dst_ptr, पूर्णांक k_elements, queue_ptr stream) {

            स्थिर पूर्णांक अंक_blocks = ceil_div(k_elements, 256);

            अगर (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(अंक_blocks) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, item_ct1, func);
                    });
            } अन्यथा {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                स्थिर sycl::uint3 ne0_fd = init_fastdiv_मूल्यावली((uint32_t) ne0);
                स्थिर sycl::uint3 ne1_fd = init_fastdiv_मूल्यावली((uint32_t) ne1);
                स्थिर sycl::uint3 ne2_fd = init_fastdiv_मूल्यावली((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(अंक_blocks) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
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

ढाँचा<typename F>
अटल अंतरभूत व्योम ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](स्थिर स्वतः * x_ptr, स्थिर स्वतः * g_ptr, स्वतः * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {

            स्थिर uint32_t अंक_blocks = (uint32_t) ceil_div(k, SYCL_GLU_BLOCK_आकार);
            स्थिर sycl::nd_range<1> launch_range(अंक_blocks * sycl::range<1>(SYCL_GLU_BLOCK_आकार),
                                                 sycl::range<1>(SYCL_GLU_BLOCK_आकार));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            अगर (o0 == n && o1 == n) {
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, item_ct1, func);
                    });
            } अन्यथा {
                // launch-invariant divisor, and only this path needs it
                स्थिर sycl::uint3 n_fd = init_fastdiv_मूल्यावली((uint32_t) n);
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, item_ct1, func);
                    });
            }
        });
}


अटल अंतरभूत व्योम ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    भग्न start, stop, step;
    स_नकल(&start, dst->op_params, माप(भग्न));
    स_नकल(&stop, (भग्न *) dst->op_params + 1, माप(भग्न));
    स_नकल(&step, (भग्न *) dst->op_params + 2, माप(भग्न));
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    भग्न * dst_ptr = (भग्न *)dst->तथ्य;
    स्थिर पूर्णांक k = (पूर्णांक)ggml_nelements(dst);
    स्थिर पूर्णांक अंक_blocks = ceil_div(k, SYCL_ARANGE_BLOCK_आकार);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(अंक_blocks) * sycl::range<1>(SYCL_ARANGE_BLOCK_आकार),
                          sycl::range<1>(SYCL_ARANGE_BLOCK_आकार)),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
            arange_kernel(dst_ptr, k, start, step, item_ct1);
        });
}

} // namespace ggml_sycl_detail



अटल अंतरभूत व्योम ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_sgn(x);
    });
}


अटल अंतरभूत व्योम ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_abs(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_elu(x);
    });
}
अटल अंतरभूत व्योम ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_silu(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_gelu(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_gelu_quick(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_gelu_erf(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_tanh(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_relu(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_hardsigmoid(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_hardswish(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_exp(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_expm1(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](स्थिर स्वतः* src, स्वतः* dst_ptr, पूर्णांक k_elements, queue_ptr stream) {
            स्थिर पूर्णांक अंक_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_आकार); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अंक_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_आकार),
                                  sycl::range<1>(SYCL_EXP_BLOCK_आकार)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

अटल अंतरभूत व्योम ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_softplus(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_neg(x);
    });
}


अटल अंतरभूत व्योम ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_step(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_sigmoid(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](स्थिर स्वतः* src, स्वतः* dst_ptr, पूर्णांक k_elements, queue_ptr stream) {
            स्थिर पूर्णांक अंक_blocks = ceil_div(k_elements, SYCL_SQRT_BLOCK_आकार);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अंक_blocks) * sycl::range<1>(SYCL_SQRT_BLOCK_आकार),
                                  sycl::range<1>(SYCL_SQRT_BLOCK_आकार)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

अटल अंतरभूत व्योम ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](स्थिर स्वतः* src, स्वतः* dst_ptr, पूर्णांक k_elements, queue_ptr stream) {
            स्थिर पूर्णांक अंक_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_आकार);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अंक_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_आकार),
                                  sycl::range<1>(SYCL_SIN_BLOCK_आकार)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

अटल अंतरभूत व्योम ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](स्थिर स्वतः* src, स्वतः* dst_ptr, पूर्णांक k_elements, queue_ptr stream) {
            स्थिर पूर्णांक अंक_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_आकार); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अंक_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_आकार),
                                  sycl::range<1>(SYCL_SIN_BLOCK_आकार)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

अटल अंतरभूत व्योम ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    भग्न negative_slope;
    स_नकल(&negative_slope, dst->op_params, माप(भग्न));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](स्थिर स्वतः* src, स्वतः* dst_ptr, पूर्णांक k_elements, queue_ptr stream, भग्न slope) {
            स्थिर पूर्णांक अंक_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_आकार);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अंक_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_आकार),
                                  sycl::range<1>(SYCL_RELU_BLOCK_आकार)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, slope, item_ct1);
                });
        }, negative_slope);
}

अटल अंतरभूत व्योम ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](स्थिर स्वतः* src, स्वतः* dst_ptr, पूर्णांक k_elements, queue_ptr stream) {
            स्थिर पूर्णांक अंक_blocks = ceil_div(k_elements, SYCL_SQR_BLOCK_आकार);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अंक_blocks) * sycl::range<1>(SYCL_SQR_BLOCK_आकार),
                                  sycl::range<1>(SYCL_SQR_BLOCK_आकार)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

अटल अंतरभूत व्योम ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    भग्न min_val;
    भग्न max_val;
    स_नकल(&min_val, dst->op_params, माप(भग्न));
    स_नकल(&max_val, (भग्न *) dst->op_params + 1, माप(भग्न));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](स्थिर स्वतः* src, स्वतः* dst_ptr, पूर्णांक k_elements, queue_ptr stream, भग्न min_arg, भग्न max_arg) {
            स्थिर पूर्णांक अंक_blocks = ceil_div(k_elements, SYCL_CLAMP_BLOCK_आकार);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अंक_blocks) * sycl::range<1>(SYCL_CLAMP_BLOCK_आकार),
                                  sycl::range<1>(SYCL_CLAMP_BLOCK_आकार)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, item_ct1);
                });
        }, min_val, max_val);
}

अटल अंतरभूत व्योम ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर भग्न alpha_n = ggml_get_op_params_f32(dst, 1);
    स्थिर भग्न alpha_p = ggml_get_op_params_f32(dst, 2);
    स्थिर भग्न beta    = ggml_get_op_params_f32(dst, 3);
    स्थिर भग्न eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](स्थिर स्वतः* src, स्वतः* dst_ptr, पूर्णांक k_elements, queue_ptr stream, भग्न alpha_n_arg, भग्न alpha_p_arg, भग्न beta_arg, भग्न eps_arg) {
            स्थिर पूर्णांक अंक_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_आकार);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(अंक_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_आकार),
                                  sycl::range<1>(SYCL_RELU_BLOCK_आकार)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, item_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

अटल अंतरभूत व्योम ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_floor(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_ceil(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_round(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](स्वतः x) {
        वापस op_trunc(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    स्थिर अक्षर  * src0_d = (स्थिर अक्षर  *) src0->तथ्य;
    स्थिर अक्षर  * src1_d = (स्थिर अक्षर  *) src1->तथ्य;
    भग्न       * dst_d  = (भग्न       *)  dst->तथ्य;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_आकार(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    स्थिर int64_t s1     = (int64_t) ((स्थिर int32_t *) dst->op_params)[0] / (int64_t) माप(भग्न);
    स्थिर int64_t s2     = (int64_t) ((स्थिर int32_t *) dst->op_params)[1] / (int64_t) माप(भग्न);
    स्थिर int64_t s3     = (int64_t) ((स्थिर int32_t *) dst->op_params)[2] / (int64_t) माप(भग्न);
    स्थिर int64_t offset = (int64_t) ((स्थिर int32_t *) dst->op_params)[3] / (int64_t) माप(भग्न);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, offset, stream);
}

अटल अंतरभूत व्योम ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](स्वतः x) {
        वापस op_gelu(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](स्वतः x) {
        वापस op_relu(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](स्वतः x) {
        वापस op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
ढाँचा<typename F>
अटल व्योम dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
    चयन (uop) {
        हाल GGML_UNARY_OP_SILU:     launch([](भग्न v) { वापस op_silu(v); });     अवरोध;
        हाल GGML_UNARY_OP_SIGMOID:  launch([](भग्न v) { वापस op_sigmoid(v); });  अवरोध;
        हाल GGML_UNARY_OP_SOFTPLUS: launch([](भग्न v) { वापस op_softplus(v); }); अवरोध;
        शेष:
            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
    }
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
व्योम ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_गाँठ, ggml_tensor * mul_गाँठ) {
    scope_op_debug_print scope_dbg_print(__func__, mul_गाँठ, /*num_src=*/2);

    स्थिर ggml_tensor * x = unary_गाँठ->src[0];
    स्थिर ggml_tensor * g = (mul_गाँठ->src[0] == unary_गाँठ) ? mul_गाँठ->src[1] : mul_गाँठ->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_गाँठ);
    GGML_ASSERT(x->type == g->type && x->type == mul_गाँठ->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_गाँठ));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_गाँठ));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिर int64_t k = ggml_nelements(mul_गाँठ);
    स्थिर int64_t n = mul_गाँठ->ne[0];

    स्थिर स्वतः dispatch_type = [&](स्वतः op) {
        चयन (mul_गाँठ->type) {
            हाल GGML_TYPE_F32:
                unary_mul_sycl((स्थिर भग्न *) x->तथ्य, (स्थिर भग्न *) g->तथ्य, (भग्न *) mul_गाँठ->तथ्य,
                               k, n, x->nb[1] / माप(भग्न), g->nb[1] / माप(भग्न), main_stream, op);
                अवरोध;
            हाल GGML_TYPE_F16:
                unary_mul_sycl((स्थिर sycl::half *) x->तथ्य, (स्थिर sycl::half *) g->तथ्य, (sycl::half *) mul_गाँठ->तथ्य,
                               k, n, x->nb[1] / माप(sycl::half), g->nb[1] / माप(sycl::half), main_stream, op);
                अवरोध;
            शेष:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_गाँठ->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_गाँठ), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
व्योम ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_गाँठ,
                                      ggml_tensor * unary_गाँठ, ggml_tensor * mul_गाँठ) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_गाँठ, /*num_src=*/2);

    स्थिर ggml_tensor * a     = add_गाँठ->src[0];
    स्थिर ggml_tensor * bias  = add_गाँठ->src[1];
    स्थिर ggml_tensor * scale = (mul_गाँठ->src[0] == unary_गाँठ) ? mul_गाँठ->src[1] : mul_गाँठ->src[0];

    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(scale != unary_गाँठ);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_गाँठ->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_गाँठ));
    // a and dst are indexed flat
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_गाँठ));
    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिर स्वतः dispatch_op = [&](स्वतः op) {
        add_unary_mul_sycl((स्थिर भग्न *) a->तथ्य, (स्थिर भग्न *) bias->तथ्य, (स्थिर भग्न *) scale->तथ्य,
                           (भग्न *) mul_गाँठ->तथ्य, ggml_nelements(mul_गाँठ), mul_गाँठ->ne[0], main_stream, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_गाँठ), dispatch_op);
}

__dpct_inline__ भग्न ggml_sycl_op_swiglu_oai_single(भग्न x, भग्न g, भग्न alpha = 1.702f, भग्न limit = 7.0f) {
    x = sycl::fmin(x, limit);
    g = sycl::fmax(sycl::fmin(g, limit), -limit);

    भग्न out_glu = x / (1.0f + sycl::native::exp(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    वापस out_glu;
}

ढाँचा <typename T>
अटल व्योम swiglu_oai_kernel(स्थिर T * x, स्थिर T * g, T * dst, स्थिर int64_t k,
                              स्थिर int64_t n, स्थिर int64_t o0, स्थिर int64_t o1,
                              भग्न alpha, भग्न limit, sycl::nd_item<3> item_ct1) {
    स्थिर int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    अगर (i >= k) {
        वापस;
    }

    स्थिर int64_t j0 = (i / n) * o0 + (i % n);
    स्थिर int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    भग्न xi = x[j0];
    भग्न gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, limit);
}

ढाँचा <typename T>
अटल व्योम swiglu_oai_sycl(स्थिर T *       x,
                            स्थिर T *       g,
                            T *             dst,
                            स्थिर int64_t   k,
                            स्थिर int64_t   n,
                            स्थिर int64_t   o0,
                            स्थिर int64_t   o1,
                            स्थिर भग्न     alpha,
                            स्थिर भग्न     limit,
                            dpct::queue_ptr stream) {
    स्थिर int64_t अंक_blocks = (k + SYCL_GLU_BLOCK_आकार - 1) / SYCL_GLU_BLOCK_आकार;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_आकार),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_आकार)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, limit, item_ct1);
                         });
}

अटल व्योम ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    व्योम * src0_d = src0->तथ्य;
    व्योम * src1_d = src1 ? src1->तथ्य : src0->तथ्य;
    स्थिर int64_t src0_o = src0->nb[1];
    स्थिर int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    व्योम * dst_d = dst->तथ्य;
    स्थिर int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_आकार(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_आकार(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);
    स्थिर भग्न alpha = ggml_get_op_params_f32(dst, 2);
    स्थिर भग्न limit = ggml_get_op_params_f32(dst, 3);

    भग्न * src0_p = (भग्न *) src0_d;
    भग्न * src1_p = (भग्न *) src1_d;

    अगर (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (भग्न *)dst_d, ggml_nelements(dst), nc, src0_o / माप(भग्न), src1_o / माप(भग्न), alpha, limit, stream);
}

ढाँचा <typename T>
अटल व्योम swiglu_clamp_kernel(स्थिर T *        gate,
                                स्थिर T *        up,
                                T *              dst,
                                स्थिर int64_t    k,
                                स्थिर int64_t    n,
                                स्थिर int64_t    o0,
                                स्थिर int64_t    o1,
                                भग्न            limit,
                                sycl::nd_item<3> item_ct1) {
    स्थिर int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    अगर (i >= k) {
        वापस;
    }

    स्थिर int64_t j0 = (i / n) * o0 + (i % n);
    स्थिर int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    स्थिर भग्न gate_मूल्य = sycl::fmin((भग्न) gate[j0], limit);
    स्थिर भग्न up_मूल्य   = sycl::fmax(sycl::fmin((भग्न) up[j1], limit), -limit);
    dst[i]                 = (T) (gate_मूल्य / (1.0f + sycl::native::exp(-gate_मूल्य)) * up_मूल्य);
}

ढाँचा <typename T>
अटल व्योम swiglu_clamp_sycl(स्थिर T *       gate,
                              स्थिर T *       up,
                              T *             dst,
                              स्थिर int64_t   k,
                              स्थिर int64_t   n,
                              स्थिर int64_t   o0,
                              स्थिर int64_t   o1,
                              भग्न           limit,
                              dpct::queue_ptr stream) {
    स्थिर int64_t अंक_blocks = (k + SYCL_GLU_BLOCK_आकार - 1) / SYCL_GLU_BLOCK_आकार;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_आकार),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_आकार)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                             swiglu_clamp_kernel(gate, up, dst, k, n, o0, o1, limit, item_ct1);
                         });
}

अटल व्योम ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0   = dst->src[0];
    स्थिर ggml_tensor * src1   = dst->src[1];
    व्योम *              src0_d = src0->तथ्य;
    व्योम *              src1_d = src1 ? src1->तथ्य : src0->तथ्य;
    स्थिर int64_t       src0_o = src0->nb[1];
    स्थिर int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    व्योम *              dst_d  = dst->तथ्य;
    स्थिर int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_आकार(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_आकार(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);
    स्थिर भग्न   limit   = ggml_get_op_params_f32(dst, 3);

    अगर (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / माप(sycl::half),
                          src1_o / माप(sycl::half), limit, stream);
    } अन्यथा {
        भग्न * src0_p = (भग्न *) src0_d;
        भग्न * src1_p = (भग्न *) src1_d;

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (भग्न *) dst_d, ggml_nelements(dst), nc, src0_o / माप(भग्न),
                          src1_o / माप(भग्न), limit, stream);
    }
}

अटल अंतरभूत व्योम ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](स्वतः x) {
        वापस op_gelu_erf(x);
    });
}

अटल अंतरभूत व्योम ggml_sycl_op_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](स्वतः x) {
        वापस op_gelu_quick(x);
    });
}


व्योम ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

व्योम ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

व्योम ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

व्योम ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

व्योम ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

व्योम ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

व्योम ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_quick(ctx, dst);
}

व्योम ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

व्योम ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

व्योम ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

व्योम ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

व्योम ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

व्योम ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

व्योम ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

व्योम ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_expm1(ctx, dst);
}

व्योम ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

व्योम ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

व्योम ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

व्योम ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_step(ctx, dst);
}

व्योम ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

व्योम ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

व्योम ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

व्योम ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_xielu(ctx, dst);
}

व्योम ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

व्योम ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

व्योम ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

व्योम ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

व्योम ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

व्योम ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

व्योम ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

व्योम ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_clamp(ctx, dst);
}

व्योम ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

व्योम ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_quick(ctx, dst);
}

व्योम ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

व्योम ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

व्योम ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

व्योम ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_round(ctx, dst);
}

व्योम ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
