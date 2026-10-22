#அந்தர்பாவ "common.hpp"
#அந்தர்பாவ "ggml-sycl/presets.hpp"
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "element_wise.hpp"

#define SYCL_GLOBAL_ID_LOOP(K, ITEM) \
    க்கு (auto i = ITEM.get_global_id(0); i < (size_t)K; i += ITEM.get_global_range(0))

#define SYCL_LOCAL_ID_CALC(ITEM, IDX) \
    (ITEM.get_local_range(IDX) * ITEM.get_group(IDX) + ITEM.get_local_id(IDX))

நிலையான வெறுமை acc_f32(மாறிலி எழுத்து * x, மாறிலி எழுத்து * y, தசம * dst, மாறிலி int64_t ne,
    மாறிலி int64_t ne0, மாறிலி int64_t ne1, மாறிலி int64_t ne2,
    மாறிலி int64_t nb00, மாறிலி int64_t nb01, மாறிலி int64_t nb02, மாறிலி int64_t nb03,
    மாறிலி int64_t ne10, மாறிலி int64_t ne11, மாறிலி int64_t ne12, மாறிலி int64_t ne13,
    மாறிலி int64_t nb10, மாறிலி int64_t nb11, மாறிலி int64_t nb12, மாறிலி int64_t nb13,
    மாறிலி int64_t s11, மாறிலி int64_t s12, மாறிலி int64_t s13, மாறிலி int64_t offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி int64_t i = SYCL_LOCAL_ID_CALC(item_ct1, 2);

    எனில் (i >= ne) {
        திரும்பு;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    மாறிலி int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    மாறிலி int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    மாறிலி int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    மாறிலி int64_t i10 = tmp;

    int64_t tmp_dst = i;
    மாறிலி int64_t i3 = tmp_dst / (ne2*ne1*ne0);
    tmp_dst -= i3 * (ne2*ne1*ne0);
    மாறிலி int64_t i2 = tmp_dst / (ne1*ne0);
    tmp_dst -= i2 * (ne1*ne0);
    மாறிலி int64_t i1 = tmp_dst / ne0;
    tmp_dst -= i1 * ne0;
    மாறிலி int64_t i0 = tmp_dst;

    தசம val = *(மாறிலி தசம *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    எனில் (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(மாறிலி தசம *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
template<typename T>
நிலையான __dpct_inline__ T op_sgn(T x) {
    திரும்பு x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


template<typename T>
நிலையான __dpct_inline__ T op_abs(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } இல்லையெனில் {
        திரும்பு sycl::fabs(x);
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_expm1(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<தசம>(x))
        );
    } இல்லையெனில் {
        திரும்பு sycl::expm1(x);
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_elu(T x) {
    திரும்பு (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

template<typename T>
நிலையான __dpct_inline__ T op_erf(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<தசம>(x))
        );
    } இல்லையெனில் {
        திரும்பு sycl::erf(x);
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_gelu_erf(T x) {
    மாறிலி T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    திரும்பு static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

template<typename T>
நிலையான __dpct_inline__ T op_gelu_quick(T x) {
    மாறிலி T GELU_QUICK_COEF_LOCAL = static_cast<T>(-1.702f);
    திரும்பு x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_QUICK_COEF_LOCAL * x)));
}

template<typename T>
நிலையான __dpct_inline__ T op_relu(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } இல்லையெனில் {
        திரும்பு sycl::fmax(x, static_cast<T>(0));
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_sigmoid(T x) {
    திரும்பு static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

template<typename T>
நிலையான __dpct_inline__ T op_sqrt(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு sycl::ext::oneapi::experimental::sqrt(x);
    } இல்லையெனில் {
        திரும்பு sycl::sqrt(x);
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_sin(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு sycl::ext::oneapi::experimental::sin(x);
    } இல்லையெனில் {
        திரும்பு sycl::sin(x);
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_cos(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு sycl::ext::oneapi::experimental::cos(x);
    } இல்லையெனில் {
        திரும்பு sycl::cos(x);
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_hardsigmoid(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } இல்லையெனில் {
        திரும்பு sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_hardswish(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } இல்லையெனில் {
        திரும்பு x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_log(T x) {
    எனில் (x <= static_cast<T>(0)) {
        திரும்பு neg_infinity<T>();
    }
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு sycl::ext::oneapi::experimental::log(x);
    } இல்லையெனில் {
        திரும்பு sycl::log(x);
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_softplus(T x) {
    மாறிலி தசம xf = (தசம) x;
    மாறிலி தசம ax = op_abs(xf);
    மாறிலி தசம m  = sycl::fmax(xf, 0.0f);
    மாறிலி தசம y  = m + sycl::log1p(sycl::exp(-ax));
    திரும்பு (T) y;
}

template<typename T>
நிலையான __dpct_inline__ T op_neg(T x) {
    திரும்பு -x;
}

template<typename T>
நிலையான __dpct_inline__ T op_step(T x) {
    திரும்பு (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

template<typename T>
நிலையான __dpct_inline__ T op_leaky_relu(T x, தசம negative_slope) {
    T neg_slope_T = static_cast<T>(negative_slope);
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;

    } இல்லையெனில் {
        திரும்பு sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_xielu(T x, தசம alpha_n, தசம alpha_p, தசம beta, தசம eps) {
    மாறிலி தசம xi        = static_cast<தசம>(x);
    மாறிலி தசம gate_pos  = (xi > 0.0f);
    மாறிலி தசம y_pos     = alpha_p * xi * xi + beta * xi;
    மாறிலி தசம min_v_eps = sycl::fmin(xi, eps);
    மாறிலி தசம y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    மாறிலி தசம out       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    திரும்பு static_cast<T>(out);
}

template<typename T>
நிலையான __dpct_inline__ T op_sqr(T x) {
    திரும்பு x * x;
}

template<typename T>
நிலையான __dpct_inline__ T op_clamp(T x, தசம min_val, தசம max_val) {
    திரும்பு x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

template<typename T>
நிலையான __dpct_inline__ T op_floor(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு sycl::ext::oneapi::experimental::floor(x);
    } இல்லையெனில் {
        திரும்பு sycl::floor(x);
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_ceil(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு sycl::ext::oneapi::experimental::ceil(x);
    } இல்லையெனில் {
        திரும்பு sycl::ceil(x);
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_round(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<தசம>(x))
        );
    } இல்லையெனில் {
        திரும்பு sycl::round(x);
    }
}

template<typename T>
நிலையான __dpct_inline__ T op_trunc(T x) {
    எனில் constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        திரும்பு sycl::ext::oneapi::experimental::trunc(x);
    } இல்லையெனில் {
        திரும்பு sycl::trunc(x);
    }
}

template<typename T, typename F>
நிலையான வெறுமை unary_op_flat_kernel(மாறிலி T * x, T * dst, மாறிலி முழு k, மாறிலி sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]);
    }
}

template<typename T, typename F>
நிலையான வெறுமை unary_op_generic_kernel(
        மாறிலி T * x,
        T * dst,
        மாறிலி முழு k,
        மாறிலி sycl::uint3 ne0_fd, மாறிலி sycl::uint3 ne1_fd, மாறிலி sycl::uint3 ne2_fd,
        மாறிலி size_t nb0,  மாறிலி size_t nb1,  மாறிலி size_t nb2,  மாறிலி size_t nb3,
        மாறிலி size_t nbd0, மாறிலி size_t nbd1, மாறிலி size_t nbd2, மாறிலி size_t nbd3,
        மாறிலி sycl::nd_item<1> & item_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        மாறிலி uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        மாறிலி uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        மாறிலி uint32_t i2 = dm.y();
        மாறிலி uint32_t i3 = dm.x();

        மாறிலி எழுத்து * src_base = (மாறிலி எழுத்து *) x;
        எழுத்து       * dst_base = (எழுத்து *) dst;

        மாறிலி T * srcp = (மாறிலி T *)(src_base + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_base + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

template<typename T>
நிலையான வெறுமை unary_op_sqrt_kernel(மாறிலி T * x, T * dst, மாறிலி முழு k, மாறிலி sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

template<typename T>
நிலையான வெறுமை unary_op_sin_kernel(மாறிலி T * x, T * dst, மாறிலி முழு k, மாறிலி sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

template<typename T>
நிலையான வெறுமை unary_op_cos_kernel(மாறிலி T * x, T * dst, மாறிலி முழு k, மாறிலி sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

template<typename T>
நிலையான வெறுமை unary_op_log_kernel(மாறிலி T * x, T * dst, மாறிலி முழு k, மாறிலி sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_log(x[i]);
    }
}


template<typename T>
நிலையான வெறுமை unary_op_leaky_relu_kernel(மாறிலி T * x, T * dst, மாறிலி முழு k, தசம negative_slope, மாறிலி sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_slope);
    }
}

template<typename T>
நிலையான வெறுமை unary_op_xielu_kernel(மாறிலி T * x, T * dst, மாறிலி முழு k, தசம alpha_n, தசம alpha_p, தசம beta, தசம eps, மாறிலி sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

template<typename T>
நிலையான வெறுமை unary_op_sqr_kernel(மாறிலி T * x, T * dst, மாறிலி முழு k, மாறிலி sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

template<typename T>
நிலையான வெறுமை unary_op_clamp_kernel(மாறிலி T * x, T * dst, மாறிலி முழு k, மாறிலி sycl::nd_item<1> &item_ct1, தசம min_val, தசம max_val) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

template<typename T>
நிலையான வெறுமை unary_op_ceil_kernel(மாறிலி T * x, T * dst, மாறிலி முழு k, மாறிலி sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

template<typename T>
நிலையான வெறுமை clamp(மாறிலி T * x, T * dst, மாறிலி தசம min, மாறிலி தசம max, மாறிலி முழு k,
                      மாறிலி sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

template<typename T, typename F>
நிலையான வெறுமை unary_gated_op_flat_kernel(மாறிலி T * x, மாறிலி T * g, T * dst, மாறிலி uint64_t k, மாறிலி sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

template<typename T, typename F>
நிலையான வெறுமை unary_gated_op_generic_kernel(
        மாறிலி T * x,
        மாறிலி T * g,
        T * dst,
        மாறிலி uint64_t k,
        மாறிலி sycl::uint3 n_fd,
        மாறிலி uint64_t o0,
        மாறிலி uint64_t o1,
        மாறிலி sycl::nd_item<1> & item_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        மாறிலி sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        மாறிலி int64_t j0 = rc.x() * o0 + rc.y();
        மாறிலி int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
template<typename T, typename F>
நிலையான வெறுமை unary_mul_flat_kernel(மாறிலி T * x, மாறிலி T * g, T * dst, மாறிலி int64_t k, மாறிலி sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = (T) (op((தசம) x[i]) * (தசம) g[i]);
    }
}

template<typename T, typename F>
நிலையான வெறுமை unary_mul_strided_kernel(மாறிலி T * x, மாறிலி T * g, T * dst, மாறிலி int64_t k, மாறிலி sycl::uint3 n_fd, மாறிலி int64_t o0, மாறிலி int64_t o1, மாறிலி sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        மாறிலி sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        மாறிலி int64_t j0 = rc.x() * o0 + rc.y();
        மாறிலி int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((தசம) x[j0]) * (தசம) g[j1]);
    }
}

template<typename T, typename F>
நிலையான வெறுமை unary_mul_sycl(மாறிலி T * x, மாறிலி T * g, T * dst, மாறிலி int64_t k, மாறிலி int64_t n, மாறிலி int64_t o0, மாறிலி int64_t o1, queue_ptr main_stream, F op) {
    மாறிலி size_t            இலக்கம்_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    மாறிலி sycl::nd_range<1> range(இலக்கம்_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    எனில் (o0 == n && o1 == n) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            unary_mul_flat_kernel(x, g, dst, k, item_ct1, op);
        });
        திரும்பு;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    மாறிலி sycl::uint3 n_fd = init_fastdiv_மதிப்புகள்((uint32_t) n);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, item_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
template<typename F>
நிலையான வெறுமை add_unary_mul_flat_kernel(மாறிலி தசம * a, மாறிலி தசம * bias, மாறிலி தசம * scale, தசம * dst,
                                      மாறிலி int64_t k, மாறிலி sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

template<typename F>
நிலையான வெறுமை add_unary_mul_bcast_kernel(மாறிலி தசம * a, மாறிலி தசம * bias, மாறிலி தசம * scale, தசம * dst,
                                       மாறிலி int64_t k, மாறிலி sycl::uint3 ne0_fd, மாறிலி sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        மாறிலி uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

template<typename F>
நிலையான வெறுமை add_unary_mul_sycl(மாறிலி தசம * a, மாறிலி தசம * bias, மாறிலி தசம * scale, தசம * dst,
                               மாறிலி int64_t k, மாறிலி int64_t ne0, queue_ptr main_stream, F op) {
    மாறிலி size_t            இலக்கம்_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    மாறிலி sycl::nd_range<1> range(இலக்கம்_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    எனில் (k == ne0) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            add_unary_mul_flat_kernel(a, bias, scale, dst, k, item_ct1, op);
        });
        திரும்பு;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    மாறிலி sycl::uint3 ne0_fd = init_fastdiv_மதிப்புகள்((uint32_t) ne0);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        add_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, item_ct1, op);
    });
}

நாமாகாஶ ggml_sycl_detail {
நிலையான வெறுமை acc_f32_sycl(மாறிலி எழுத்து *x, மாறிலி எழுத்து *y, தசம *dst,
                         மாறிலி int64_t n_elements,
                         மாறிலி int64_t ne0, மாறிலி int64_t ne1, மாறிலி int64_t ne2,
                         மாறிலி int64_t nb00, மாறிலி int64_t nb01, மாறிலி int64_t nb02, மாறிலி int64_t nb03,
                         மாறிலி int64_t ne10, மாறிலி int64_t ne11, மாறிலி int64_t ne12, மாறிலி int64_t ne13,
                         மாறிலி int64_t nb10, மாறிலி int64_t nb11, மாறிலி int64_t nb12, மாறிலி int64_t nb13,
                         மாறிலி int64_t s1, மாறிலி int64_t s2, மாறிலி int64_t s3,
                         மாறிலி int64_t offset, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
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
நிலையான வெறுமை arange_kernel(T * dst, மாறிலி முழு k, T start, T step,
                         மாறிலி sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = start + static_cast<T>(i) * step;
    }
}

template<typename KernelInvoker, typename... Args>
நிலையான உள்வரி வெறுமை dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    தேர்வு (dst->type) {
        நிலை GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (முழு)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                நிறுத்து;
            }
#ifdef GGML_SYCL_HAS_BF16
        நிலை GGML_TYPE_BF16:
            {
                auto data_pts = cast_data<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (முழு)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                நிறுத்து;
            }
#endif
        நிலை GGML_TYPE_F32:
            {
                auto data_pts = cast_data<தசம>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (முழு)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                நிறுத்து;
            }
        இயல்பு:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename KernelInvoker, typename... Args>
நிலையான உள்வரி வெறுமை dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    மாறிலி int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // The fused GLU kernels index with 32-bit fastdiv, which is exact only for indices below
    // 2^31. A dst that large is ~8 GB at f32, and the grid sizing already narrows to 32 bits,
    // so assert the bound rather than carry a second code path for it.
    GGML_ASSERT(ggml_nelements(dst) < ((int64_t) 1 << 31));
    மாறிலி int32_t swapped = ((மாறிலி int32_t *) dst->op_params)[1];
    வெறுமை * src0_d = src0->data;
    வெறுமை * src1_d = src1 ? src1->data : src0->data;
    மாறிலி int64_t src0_o = src0->nb[1];
    மாறிலி int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    வெறுமை * dst_d = dst->data;
    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    தேர்வு (dst->type) {
        நிலை GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    எனில் (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / அளவு(sycl::half),
                               src1_o / அளவு(sycl::half),
                               main_stream,
                               std::forward<Args>(args)...);
                நிறுத்து;
            }
        நிலை GGML_TYPE_F32:
            {
                தசம * src0_p = (தசம *) src0_d;
                தசம * src1_p = (தசம *) src1_d;

                    எனில் (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (தசம *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / அளவு(தசம),
                               src1_o / அளவு(தசம),
                               main_stream,
                               std::forward<Args>(args)...);
                நிறுத்து;
            }
        இயல்பு:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename F>
நிலையான உள்வரி வெறுமை ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    மாறிலி int64_t ne0  = dst->ne[0];
    மாறிலி int64_t ne1  = dst->ne[1];
    மாறிலி int64_t ne2  = dst->ne[2];

    மாறிலி size_t  nb0  = src0->nb[0];
    மாறிலி size_t  nb1  = src0->nb[1];
    மாறிலி size_t  nb2  = src0->nb[2];
    மாறிலி size_t  nb3  = src0->nb[3];

    மாறிலி size_t  nbd0 = dst->nb[0];
    மாறிலி size_t  nbd1 = dst->nb[1];
    மாறிலி size_t  nbd2 = dst->nb[2];
    மாறிலி size_t  nbd3 = dst->nb[3];

    // Hot unary ops (FFN/GDN silu, sigmoid, ...) run on contiguous tensors;
    // skip the strided index math entirely for them.
    மாறிலி தருக்க contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](மாறிலி auto* src, auto* dst_ptr, முழு k_elements, queue_ptr stream) {

            மாறிலி முழு இலக்கம்_blocks = ceil_div(k_elements, 256);

            எனில் (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, item_ct1, func);
                    });
            } இல்லையெனில் {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                மாறிலி sycl::uint3 ne0_fd = init_fastdiv_மதிப்புகள்((uint32_t) ne0);
                மாறிலி sycl::uint3 ne1_fd = init_fastdiv_மதிப்புகள்((uint32_t) ne1);
                மாறிலி sycl::uint3 ne2_fd = init_fastdiv_மதிப்புகள்((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks) * sycl::range<1>(256),
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
நிலையான உள்வரி வெறுமை ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](மாறிலி auto * x_ptr, மாறிலி auto * g_ptr, auto * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {

            மாறிலி uint32_t இலக்கம்_blocks = (uint32_t) ceil_div(k, SYCL_GLU_BLOCK_SIZE);
            மாறிலி sycl::nd_range<1> launch_range(இலக்கம்_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE),
                                                 sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            எனில் (o0 == n && o1 == n) {
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, item_ct1, func);
                    });
            } இல்லையெனில் {
                // launch-invariant divisor, and only this path needs it
                மாறிலி sycl::uint3 n_fd = init_fastdiv_மதிப்புகள்((uint32_t) n);
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, item_ct1, func);
                    });
            }
        });
}


நிலையான உள்வரி வெறுமை ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    தசம start, stop, step;
    memcpy(&start, dst->op_params, அளவு(தசம));
    memcpy(&stop, (தசம *) dst->op_params + 1, அளவு(தசம));
    memcpy(&step, (தசம *) dst->op_params + 2, அளவு(தசம));
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    தசம * dst_ptr = (தசம *)dst->data;
    மாறிலி முழு k = (முழு)ggml_nelements(dst);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(k, SYCL_ARANGE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks) * sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE),
                          sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            arange_kernel(dst_ptr, k, start, step, item_ct1);
        });
}

} // namespace ggml_sycl_detail



நிலையான உள்வரி வெறுமை ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_sgn(x);
    });
}


நிலையான உள்வரி வெறுமை ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_abs(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_elu(x);
    });
}
நிலையான உள்வரி வெறுமை ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_silu(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_gelu(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_gelu_quick(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_gelu_erf(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_tanh(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_relu(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_hardsigmoid(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_hardswish(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_exp(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_expm1(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](மாறிலி auto* src, auto* dst_ptr, முழு k_elements, queue_ptr stream) {
            மாறிலி முழு இலக்கம்_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_softplus(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_neg(x);
    });
}


நிலையான உள்வரி வெறுமை ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_step(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_sigmoid(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](மாறிலி auto* src, auto* dst_ptr, முழு k_elements, queue_ptr stream) {
            மாறிலி முழு இலக்கம்_blocks = ceil_div(k_elements, SYCL_SQRT_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks) * sycl::range<1>(SYCL_SQRT_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQRT_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](மாறிலி auto* src, auto* dst_ptr, முழு k_elements, queue_ptr stream) {
            மாறிலி முழு இலக்கம்_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](மாறிலி auto* src, auto* dst_ptr, முழு k_elements, queue_ptr stream) {
            மாறிலி முழு இலக்கம்_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    தசம negative_slope;
    memcpy(&negative_slope, dst->op_params, அளவு(தசம));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](மாறிலி auto* src, auto* dst_ptr, முழு k_elements, queue_ptr stream, தசம slope) {
            மாறிலி முழு இலக்கம்_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, slope, item_ct1);
                });
        }, negative_slope);
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](மாறிலி auto* src, auto* dst_ptr, முழு k_elements, queue_ptr stream) {
            மாறிலி முழு இலக்கம்_blocks = ceil_div(k_elements, SYCL_SQR_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks) * sycl::range<1>(SYCL_SQR_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQR_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    தசம min_val;
    தசம max_val;
    memcpy(&min_val, dst->op_params, அளவு(தசம));
    memcpy(&max_val, (தசம *) dst->op_params + 1, அளவு(தசம));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](மாறிலி auto* src, auto* dst_ptr, முழு k_elements, queue_ptr stream, தசம min_arg, தசம max_arg) {
            மாறிலி முழு இலக்கம்_blocks = ceil_div(k_elements, SYCL_CLAMP_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks) * sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, item_ct1);
                });
        }, min_val, max_val);
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    மாறிலி தசம alpha_n = ggml_get_op_params_f32(dst, 1);
    மாறிலி தசம alpha_p = ggml_get_op_params_f32(dst, 2);
    மாறிலி தசம beta    = ggml_get_op_params_f32(dst, 3);
    மாறிலி தசம eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](மாறிலி auto* src, auto* dst_ptr, முழு k_elements, queue_ptr stream, தசம alpha_n_arg, தசம alpha_p_arg, தசம beta_arg, தசம eps_arg) {
            மாறிலி முழு இலக்கம்_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, item_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_floor(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_ceil(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_round(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        திரும்பு op_trunc(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    மாறிலி எழுத்து  * src0_d = (மாறிலி எழுத்து  *) src0->data;
    மாறிலி எழுத்து  * src1_d = (மாறிலி எழுத்து  *) src1->data;
    தசம       * dst_d  = (தசம       *)  dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    மாறிலி int64_t s1     = (int64_t) ((மாறிலி int32_t *) dst->op_params)[0] / (int64_t) அளவு(தசம);
    மாறிலி int64_t s2     = (int64_t) ((மாறிலி int32_t *) dst->op_params)[1] / (int64_t) அளவு(தசம);
    மாறிலி int64_t s3     = (int64_t) ((மாறிலி int32_t *) dst->op_params)[2] / (int64_t) அளவு(தசம);
    மாறிலி int64_t offset = (int64_t) ((மாறிலி int32_t *) dst->op_params)[3] / (int64_t) அளவு(தசம);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, offset, stream);
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        திரும்பு op_gelu(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        திரும்பு op_relu(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        திரும்பு op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
template<typename F>
நிலையான வெறுமை dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
    தேர்வு (uop) {
        நிலை GGML_UNARY_OP_SILU:     launch([](தசம v) { திரும்பு op_silu(v); });     நிறுத்து;
        நிலை GGML_UNARY_OP_SIGMOID:  launch([](தசம v) { திரும்பு op_sigmoid(v); });  நிறுத்து;
        நிலை GGML_UNARY_OP_SOFTPLUS: launch([](தசம v) { திரும்பு op_softplus(v); }); நிறுத்து;
        இயல்பு:
            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
    }
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
வெறுமை ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_முனை, ggml_tensor * mul_முனை) {
    scope_op_debug_print scope_dbg_print(__func__, mul_முனை, /*num_src=*/2);

    மாறிலி ggml_tensor * x = unary_முனை->src[0];
    மாறிலி ggml_tensor * g = (mul_முனை->src[0] == unary_முனை) ? mul_முனை->src[1] : mul_முனை->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_முனை);
    GGML_ASSERT(x->type == g->type && x->type == mul_முனை->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_முனை));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_முனை));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    மாறிலி int64_t k = ggml_nelements(mul_முனை);
    மாறிலி int64_t n = mul_முனை->ne[0];

    மாறிலி auto dispatch_type = [&](auto op) {
        தேர்வு (mul_முனை->type) {
            நிலை GGML_TYPE_F32:
                unary_mul_sycl((மாறிலி தசம *) x->data, (மாறிலி தசம *) g->data, (தசம *) mul_முனை->data,
                               k, n, x->nb[1] / அளவு(தசம), g->nb[1] / அளவு(தசம), main_stream, op);
                நிறுத்து;
            நிலை GGML_TYPE_F16:
                unary_mul_sycl((மாறிலி sycl::half *) x->data, (மாறிலி sycl::half *) g->data, (sycl::half *) mul_முனை->data,
                               k, n, x->nb[1] / அளவு(sycl::half), g->nb[1] / அளவு(sycl::half), main_stream, op);
                நிறுத்து;
            இயல்பு:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_முனை->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_முனை), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
வெறுமை ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_முனை,
                                      ggml_tensor * unary_முனை, ggml_tensor * mul_முனை) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_முனை, /*num_src=*/2);

    மாறிலி ggml_tensor * a     = add_முனை->src[0];
    மாறிலி ggml_tensor * bias  = add_முனை->src[1];
    மாறிலி ggml_tensor * scale = (mul_முனை->src[0] == unary_முனை) ? mul_முனை->src[1] : mul_முனை->src[0];

    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(scale != unary_முனை);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_முனை->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_முனை));
    // a and dst are indexed flat
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_முனை));
    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    மாறிலி auto dispatch_op = [&](auto op) {
        add_unary_mul_sycl((மாறிலி தசம *) a->data, (மாறிலி தசம *) bias->data, (மாறிலி தசம *) scale->data,
                           (தசம *) mul_முனை->data, ggml_nelements(mul_முனை), mul_முனை->ne[0], main_stream, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_முனை), dispatch_op);
}

__dpct_inline__ தசம ggml_sycl_op_swiglu_oai_single(தசம x, தசம g, தசம alpha = 1.702f, தசம limit = 7.0f) {
    x = sycl::fmin(x, limit);
    g = sycl::fmax(sycl::fmin(g, limit), -limit);

    தசம out_glu = x / (1.0f + sycl::native::exp(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    திரும்பு out_glu;
}

template <typename T>
நிலையான வெறுமை swiglu_oai_kernel(மாறிலி T * x, மாறிலி T * g, T * dst, மாறிலி int64_t k,
                              மாறிலி int64_t n, மாறிலி int64_t o0, மாறிலி int64_t o1,
                              தசம alpha, தசம limit, sycl::nd_item<3> item_ct1) {
    மாறிலி int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    எனில் (i >= k) {
        திரும்பு;
    }

    மாறிலி int64_t j0 = (i / n) * o0 + (i % n);
    மாறிலி int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    தசம xi = x[j0];
    தசம gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
நிலையான வெறுமை swiglu_oai_sycl(மாறிலி T *       x,
                            மாறிலி T *       g,
                            T *             dst,
                            மாறிலி int64_t   k,
                            மாறிலி int64_t   n,
                            மாறிலி int64_t   o0,
                            மாறிலி int64_t   o1,
                            மாறிலி தசம     alpha,
                            மாறிலி தசம     limit,
                            dpct::queue_ptr stream) {
    மாறிலி int64_t இலக்கம்_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, limit, item_ct1);
                         });
}

நிலையான வெறுமை ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    வெறுமை * src0_d = src0->data;
    வெறுமை * src1_d = src1 ? src1->data : src0->data;
    மாறிலி int64_t src0_o = src0->nb[1];
    மாறிலி int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    வெறுமை * dst_d = dst->data;
    மாறிலி int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);
    மாறிலி தசம alpha = ggml_get_op_params_f32(dst, 2);
    மாறிலி தசம limit = ggml_get_op_params_f32(dst, 3);

    தசம * src0_p = (தசம *) src0_d;
    தசம * src1_p = (தசம *) src1_d;

    எனில் (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (தசம *)dst_d, ggml_nelements(dst), nc, src0_o / அளவு(தசம), src1_o / அளவு(தசம), alpha, limit, stream);
}

template <typename T>
நிலையான வெறுமை swiglu_clamp_kernel(மாறிலி T *        gate,
                                மாறிலி T *        up,
                                T *              dst,
                                மாறிலி int64_t    k,
                                மாறிலி int64_t    n,
                                மாறிலி int64_t    o0,
                                மாறிலி int64_t    o1,
                                தசம            limit,
                                sycl::nd_item<3> item_ct1) {
    மாறிலி int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    எனில் (i >= k) {
        திரும்பு;
    }

    மாறிலி int64_t j0 = (i / n) * o0 + (i % n);
    மாறிலி int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    மாறிலி தசம gate_மதிப்பு = sycl::fmin((தசம) gate[j0], limit);
    மாறிலி தசம up_மதிப்பு   = sycl::fmax(sycl::fmin((தசம) up[j1], limit), -limit);
    dst[i]                 = (T) (gate_மதிப்பு / (1.0f + sycl::native::exp(-gate_மதிப்பு)) * up_மதிப்பு);
}

template <typename T>
நிலையான வெறுமை swiglu_clamp_sycl(மாறிலி T *       gate,
                              மாறிலி T *       up,
                              T *             dst,
                              மாறிலி int64_t   k,
                              மாறிலி int64_t   n,
                              மாறிலி int64_t   o0,
                              மாறிலி int64_t   o1,
                              தசம           limit,
                              dpct::queue_ptr stream) {
    மாறிலி int64_t இலக்கம்_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_clamp_kernel(gate, up, dst, k, n, o0, o1, limit, item_ct1);
                         });
}

நிலையான வெறுமை ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0   = dst->src[0];
    மாறிலி ggml_tensor * src1   = dst->src[1];
    வெறுமை *              src0_d = src0->data;
    வெறுமை *              src1_d = src1 ? src1->data : src0->data;
    மாறிலி int64_t       src0_o = src0->nb[1];
    மாறிலி int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    வெறுமை *              dst_d  = dst->data;
    மாறிலி int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);
    மாறிலி தசம   limit   = ggml_get_op_params_f32(dst, 3);

    எனில் (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / அளவு(sycl::half),
                          src1_o / அளவு(sycl::half), limit, stream);
    } இல்லையெனில் {
        தசம * src0_p = (தசம *) src0_d;
        தசம * src1_p = (தசம *) src1_d;

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (தசம *) dst_d, ggml_nelements(dst), nc, src0_o / அளவு(தசம),
                          src1_o / அளவு(தசம), limit, stream);
    }
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        திரும்பு op_gelu_erf(x);
    });
}

நிலையான உள்வரி வெறுமை ggml_sycl_op_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        திரும்பு op_gelu_quick(x);
    });
}


வெறுமை ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

வெறுமை ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

வெறுமை ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

வெறுமை ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

வெறுமை ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

வெறுமை ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

வெறுமை ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_quick(ctx, dst);
}

வெறுமை ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

வெறுமை ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

வெறுமை ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

வெறுமை ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

வெறுமை ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

வெறுமை ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

வெறுமை ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

வெறுமை ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_expm1(ctx, dst);
}

வெறுமை ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

வெறுமை ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

வெறுமை ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

வெறுமை ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_step(ctx, dst);
}

வெறுமை ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

வெறுமை ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

வெறுமை ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

வெறுமை ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_xielu(ctx, dst);
}

வெறுமை ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

வெறுமை ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

வெறுமை ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

வெறுமை ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

வெறுமை ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

வெறுமை ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

வெறுமை ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

வெறுமை ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_clamp(ctx, dst);
}

வெறுமை ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

வெறுமை ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_quick(ctx, dst);
}

வெறுமை ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

வெறுமை ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

வெறுமை ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

வெறுமை ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_round(ctx, dst);
}

வெறுமை ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
