#شامل "common.hpp"
#شامل "ggml-sycl/presets.hpp"
#شامل "ggml.h"
#شامل "element_wise.hpp"

#define SYCL_GLOBAL_ID_LOOP(K, ITEM) \
    برای (auto i = ITEM.get_global_id(0); i < (size_t)K; i += ITEM.get_global_range(0))

#define SYCL_LOCAL_ID_CALC(ITEM, IDX) \
    (ITEM.get_local_range(IDX) * ITEM.get_group(IDX) + ITEM.get_local_id(IDX))

ایستا تهی acc_f32(ثابت نویسه * x, ثابت نویسه * y, اعشاری * dst, ثابت int64_t ne,
    ثابت int64_t ne0, ثابت int64_t ne1, ثابت int64_t ne2,
    ثابت int64_t nb00, ثابت int64_t nb01, ثابت int64_t nb02, ثابت int64_t nb03,
    ثابت int64_t ne10, ثابت int64_t ne11, ثابت int64_t ne12, ثابت int64_t ne13,
    ثابت int64_t nb10, ثابت int64_t nb11, ثابت int64_t nb12, ثابت int64_t nb13,
    ثابت int64_t s11, ثابت int64_t s12, ثابت int64_t s13, ثابت int64_t offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت int64_t i = SYCL_LOCAL_ID_CALC(item_ct1, 2);

    اگر (i >= ne) {
        بازگشت;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    ثابت int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    ثابت int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    ثابت int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    ثابت int64_t i10 = tmp;

    int64_t tmp_dst = i;
    ثابت int64_t i3 = tmp_dst / (ne2*ne1*ne0);
    tmp_dst -= i3 * (ne2*ne1*ne0);
    ثابت int64_t i2 = tmp_dst / (ne1*ne0);
    tmp_dst -= i2 * (ne1*ne0);
    ثابت int64_t i1 = tmp_dst / ne0;
    tmp_dst -= i1 * ne0;
    ثابت int64_t i0 = tmp_dst;

    اعشاری val = *(ثابت اعشاری *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    اگر (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(ثابت اعشاری *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
template<typename T>
ایستا __dpct_inline__ T op_sgn(T x) {
    بازگشت x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


template<typename T>
ایستا __dpct_inline__ T op_abs(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } وگرنه {
        بازگشت sycl::fabs(x);
    }
}

template<typename T>
ایستا __dpct_inline__ T op_expm1(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<اعشاری>(x))
        );
    } وگرنه {
        بازگشت sycl::expm1(x);
    }
}

template<typename T>
ایستا __dpct_inline__ T op_elu(T x) {
    بازگشت (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

template<typename T>
ایستا __dpct_inline__ T op_erf(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<اعشاری>(x))
        );
    } وگرنه {
        بازگشت sycl::erf(x);
    }
}

template<typename T>
ایستا __dpct_inline__ T op_gelu_erf(T x) {
    ثابت T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    بازگشت static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

template<typename T>
ایستا __dpct_inline__ T op_gelu_quick(T x) {
    ثابت T GELU_QUICK_COEF_LOCAL = static_cast<T>(-1.702f);
    بازگشت x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_QUICK_COEF_LOCAL * x)));
}

template<typename T>
ایستا __dpct_inline__ T op_relu(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } وگرنه {
        بازگشت sycl::fmax(x, static_cast<T>(0));
    }
}

template<typename T>
ایستا __dpct_inline__ T op_sigmoid(T x) {
    بازگشت static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

template<typename T>
ایستا __dpct_inline__ T op_sqrt(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت sycl::ext::oneapi::experimental::sqrt(x);
    } وگرنه {
        بازگشت sycl::sqrt(x);
    }
}

template<typename T>
ایستا __dpct_inline__ T op_sin(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت sycl::ext::oneapi::experimental::sin(x);
    } وگرنه {
        بازگشت sycl::sin(x);
    }
}

template<typename T>
ایستا __dpct_inline__ T op_cos(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت sycl::ext::oneapi::experimental::cos(x);
    } وگرنه {
        بازگشت sycl::cos(x);
    }
}

template<typename T>
ایستا __dpct_inline__ T op_hardsigmoid(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } وگرنه {
        بازگشت sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
ایستا __dpct_inline__ T op_hardswish(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } وگرنه {
        بازگشت x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
ایستا __dpct_inline__ T op_log(T x) {
    اگر (x <= static_cast<T>(0)) {
        بازگشت neg_infinity<T>();
    }
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت sycl::ext::oneapi::experimental::log(x);
    } وگرنه {
        بازگشت sycl::log(x);
    }
}

template<typename T>
ایستا __dpct_inline__ T op_softplus(T x) {
    ثابت اعشاری xf = (اعشاری) x;
    ثابت اعشاری ax = op_abs(xf);
    ثابت اعشاری m  = sycl::fmax(xf, 0.0f);
    ثابت اعشاری y  = m + sycl::log1p(sycl::exp(-ax));
    بازگشت (T) y;
}

template<typename T>
ایستا __dpct_inline__ T op_neg(T x) {
    بازگشت -x;
}

template<typename T>
ایستا __dpct_inline__ T op_step(T x) {
    بازگشت (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

template<typename T>
ایستا __dpct_inline__ T op_leaky_relu(T x, اعشاری negative_slope) {
    T neg_slope_T = static_cast<T>(negative_slope);
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;

    } وگرنه {
        بازگشت sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;
    }
}

template<typename T>
ایستا __dpct_inline__ T op_xielu(T x, اعشاری alpha_n, اعشاری alpha_p, اعشاری beta, اعشاری eps) {
    ثابت اعشاری xi        = static_cast<اعشاری>(x);
    ثابت اعشاری gate_pos  = (xi > 0.0f);
    ثابت اعشاری y_pos     = alpha_p * xi * xi + beta * xi;
    ثابت اعشاری min_v_eps = sycl::fmin(xi, eps);
    ثابت اعشاری y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    ثابت اعشاری out       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    بازگشت static_cast<T>(out);
}

template<typename T>
ایستا __dpct_inline__ T op_sqr(T x) {
    بازگشت x * x;
}

template<typename T>
ایستا __dpct_inline__ T op_clamp(T x, اعشاری min_val, اعشاری max_val) {
    بازگشت x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

template<typename T>
ایستا __dpct_inline__ T op_floor(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت sycl::ext::oneapi::experimental::floor(x);
    } وگرنه {
        بازگشت sycl::floor(x);
    }
}

template<typename T>
ایستا __dpct_inline__ T op_ceil(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت sycl::ext::oneapi::experimental::ceil(x);
    } وگرنه {
        بازگشت sycl::ceil(x);
    }
}

template<typename T>
ایستا __dpct_inline__ T op_round(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<اعشاری>(x))
        );
    } وگرنه {
        بازگشت sycl::round(x);
    }
}

template<typename T>
ایستا __dpct_inline__ T op_trunc(T x) {
    اگر constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        بازگشت sycl::ext::oneapi::experimental::trunc(x);
    } وگرنه {
        بازگشت sycl::trunc(x);
    }
}

template<typename T, typename F>
ایستا تهی unary_op_flat_kernel(ثابت T * x, T * dst, ثابت صحیح k, ثابت sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]);
    }
}

template<typename T, typename F>
ایستا تهی unary_op_generic_kernel(
        ثابت T * x,
        T * dst,
        ثابت صحیح k,
        ثابت sycl::uint3 ne0_fd, ثابت sycl::uint3 ne1_fd, ثابت sycl::uint3 ne2_fd,
        ثابت size_t nb0,  ثابت size_t nb1,  ثابت size_t nb2,  ثابت size_t nb3,
        ثابت size_t nbd0, ثابت size_t nbd1, ثابت size_t nbd2, ثابت size_t nbd3,
        ثابت sycl::nd_item<1> & item_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        ثابت uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        ثابت uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        ثابت uint32_t i2 = dm.y();
        ثابت uint32_t i3 = dm.x();

        ثابت نویسه * src_base = (ثابت نویسه *) x;
        نویسه       * dst_base = (نویسه *) dst;

        ثابت T * srcp = (ثابت T *)(src_base + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_base + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

template<typename T>
ایستا تهی unary_op_sqrt_kernel(ثابت T * x, T * dst, ثابت صحیح k, ثابت sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

template<typename T>
ایستا تهی unary_op_sin_kernel(ثابت T * x, T * dst, ثابت صحیح k, ثابت sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

template<typename T>
ایستا تهی unary_op_cos_kernel(ثابت T * x, T * dst, ثابت صحیح k, ثابت sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

template<typename T>
ایستا تهی unary_op_log_kernel(ثابت T * x, T * dst, ثابت صحیح k, ثابت sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_log(x[i]);
    }
}


template<typename T>
ایستا تهی unary_op_leaky_relu_kernel(ثابت T * x, T * dst, ثابت صحیح k, اعشاری negative_slope, ثابت sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_slope);
    }
}

template<typename T>
ایستا تهی unary_op_xielu_kernel(ثابت T * x, T * dst, ثابت صحیح k, اعشاری alpha_n, اعشاری alpha_p, اعشاری beta, اعشاری eps, ثابت sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

template<typename T>
ایستا تهی unary_op_sqr_kernel(ثابت T * x, T * dst, ثابت صحیح k, ثابت sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

template<typename T>
ایستا تهی unary_op_clamp_kernel(ثابت T * x, T * dst, ثابت صحیح k, ثابت sycl::nd_item<1> &item_ct1, اعشاری min_val, اعشاری max_val) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

template<typename T>
ایستا تهی unary_op_ceil_kernel(ثابت T * x, T * dst, ثابت صحیح k, ثابت sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

template<typename T>
ایستا تهی clamp(ثابت T * x, T * dst, ثابت اعشاری min, ثابت اعشاری max, ثابت صحیح k,
                      ثابت sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

template<typename T, typename F>
ایستا تهی unary_gated_op_flat_kernel(ثابت T * x, ثابت T * g, T * dst, ثابت uint64_t k, ثابت sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

template<typename T, typename F>
ایستا تهی unary_gated_op_generic_kernel(
        ثابت T * x,
        ثابت T * g,
        T * dst,
        ثابت uint64_t k,
        ثابت sycl::uint3 n_fd,
        ثابت uint64_t o0,
        ثابت uint64_t o1,
        ثابت sycl::nd_item<1> & item_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ثابت sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ثابت int64_t j0 = rc.x() * o0 + rc.y();
        ثابت int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
template<typename T, typename F>
ایستا تهی unary_mul_flat_kernel(ثابت T * x, ثابت T * g, T * dst, ثابت int64_t k, ثابت sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = (T) (op((اعشاری) x[i]) * (اعشاری) g[i]);
    }
}

template<typename T, typename F>
ایستا تهی unary_mul_strided_kernel(ثابت T * x, ثابت T * g, T * dst, ثابت int64_t k, ثابت sycl::uint3 n_fd, ثابت int64_t o0, ثابت int64_t o1, ثابت sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ثابت sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ثابت int64_t j0 = rc.x() * o0 + rc.y();
        ثابت int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((اعشاری) x[j0]) * (اعشاری) g[j1]);
    }
}

template<typename T, typename F>
ایستا تهی unary_mul_sycl(ثابت T * x, ثابت T * g, T * dst, ثابت int64_t k, ثابت int64_t n, ثابت int64_t o0, ثابت int64_t o1, queue_ptr main_stream, F op) {
    ثابت size_t            رقم_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    ثابت sycl::nd_range<1> range(رقم_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    اگر (o0 == n && o1 == n) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            unary_mul_flat_kernel(x, g, dst, k, item_ct1, op);
        });
        بازگشت;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ثابت sycl::uint3 n_fd = init_fastdiv_مقادیر((uint32_t) n);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, item_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
template<typename F>
ایستا تهی add_unary_mul_flat_kernel(ثابت اعشاری * a, ثابت اعشاری * bias, ثابت اعشاری * scale, اعشاری * dst,
                                      ثابت int64_t k, ثابت sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

template<typename F>
ایستا تهی add_unary_mul_bcast_kernel(ثابت اعشاری * a, ثابت اعشاری * bias, ثابت اعشاری * scale, اعشاری * dst,
                                       ثابت int64_t k, ثابت sycl::uint3 ne0_fd, ثابت sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        ثابت uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

template<typename F>
ایستا تهی add_unary_mul_sycl(ثابت اعشاری * a, ثابت اعشاری * bias, ثابت اعشاری * scale, اعشاری * dst,
                               ثابت int64_t k, ثابت int64_t ne0, queue_ptr main_stream, F op) {
    ثابت size_t            رقم_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    ثابت sycl::nd_range<1> range(رقم_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    اگر (k == ne0) {
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            add_unary_mul_flat_kernel(a, bias, scale, dst, k, item_ct1, op);
        });
        بازگشت;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ثابت sycl::uint3 ne0_fd = init_fastdiv_مقادیر((uint32_t) ne0);
    main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        add_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, item_ct1, op);
    });
}

فضاینام ggml_sycl_detail {
ایستا تهی acc_f32_sycl(ثابت نویسه *x, ثابت نویسه *y, اعشاری *dst,
                         ثابت int64_t n_elements,
                         ثابت int64_t ne0, ثابت int64_t ne1, ثابت int64_t ne2,
                         ثابت int64_t nb00, ثابت int64_t nb01, ثابت int64_t nb02, ثابت int64_t nb03,
                         ثابت int64_t ne10, ثابت int64_t ne11, ثابت int64_t ne12, ثابت int64_t ne13,
                         ثابت int64_t nb10, ثابت int64_t nb11, ثابت int64_t nb12, ثابت int64_t nb13,
                         ثابت int64_t s1, ثابت int64_t s2, ثابت int64_t s3,
                         ثابت int64_t offset, queue_ptr stream) {
    ثابت صحیح رقم_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, رقم_blocks) * sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
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
ایستا تهی arange_kernel(T * dst, ثابت صحیح k, T start, T step,
                         ثابت sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = start + static_cast<T>(i) * step;
    }
}

template<typename KernelInvoker, typename... Args>
ایستا درونخطی تهی dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    انتخاب (dst->type) {
        حالت GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (صحیح)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                بشکن;
            }
#ifdef GGML_SYCL_HAS_BF16
        حالت GGML_TYPE_BF16:
            {
                auto data_pts = cast_data<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (صحیح)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                بشکن;
            }
#endif
        حالت GGML_TYPE_F32:
            {
                auto data_pts = cast_data<اعشاری>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (صحیح)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                بشکن;
            }
        پیشفرض:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename KernelInvoker, typename... Args>
ایستا درونخطی تهی dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    ثابت int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // The fused GLU kernels index with 32-bit fastdiv, which is exact only for indices below
    // 2^31. A dst that large is ~8 GB at f32, and the grid sizing already narrows to 32 bits,
    // so assert the bound rather than carry a second code path for it.
    GGML_ASSERT(ggml_nelements(dst) < ((int64_t) 1 << 31));
    ثابت int32_t swapped = ((ثابت int32_t *) dst->op_params)[1];
    تهی * src0_d = src0->data;
    تهی * src1_d = src1 ? src1->data : src0->data;
    ثابت int64_t src0_o = src0->nb[1];
    ثابت int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    تهی * dst_d = dst->data;
    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    انتخاب (dst->type) {
        حالت GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    اگر (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / اندازه(sycl::half),
                               src1_o / اندازه(sycl::half),
                               main_stream,
                               std::forward<Args>(args)...);
                بشکن;
            }
        حالت GGML_TYPE_F32:
            {
                اعشاری * src0_p = (اعشاری *) src0_d;
                اعشاری * src1_p = (اعشاری *) src1_d;

                    اگر (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (اعشاری *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / اندازه(اعشاری),
                               src1_o / اندازه(اعشاری),
                               main_stream,
                               std::forward<Args>(args)...);
                بشکن;
            }
        پیشفرض:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename F>
ایستا درونخطی تهی ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    ثابت int64_t ne0  = dst->ne[0];
    ثابت int64_t ne1  = dst->ne[1];
    ثابت int64_t ne2  = dst->ne[2];

    ثابت size_t  nb0  = src0->nb[0];
    ثابت size_t  nb1  = src0->nb[1];
    ثابت size_t  nb2  = src0->nb[2];
    ثابت size_t  nb3  = src0->nb[3];

    ثابت size_t  nbd0 = dst->nb[0];
    ثابت size_t  nbd1 = dst->nb[1];
    ثابت size_t  nbd2 = dst->nb[2];
    ثابت size_t  nbd3 = dst->nb[3];

    // Hot unary ops (FFN/GDN silu, sigmoid, ...) run on contiguous tensors;
    // skip the strided index math entirely for them.
    ثابت منطقی contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](ثابت auto* src, auto* dst_ptr, صحیح k_elements, queue_ptr stream) {

            ثابت صحیح رقم_blocks = ceil_div(k_elements, 256);

            اگر (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(رقم_blocks) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, item_ct1, func);
                    });
            } وگرنه {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                ثابت sycl::uint3 ne0_fd = init_fastdiv_مقادیر((uint32_t) ne0);
                ثابت sycl::uint3 ne1_fd = init_fastdiv_مقادیر((uint32_t) ne1);
                ثابت sycl::uint3 ne2_fd = init_fastdiv_مقادیر((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(رقم_blocks) * sycl::range<1>(256),
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
ایستا درونخطی تهی ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](ثابت auto * x_ptr, ثابت auto * g_ptr, auto * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {

            ثابت uint32_t رقم_blocks = (uint32_t) ceil_div(k, SYCL_GLU_BLOCK_SIZE);
            ثابت sycl::nd_range<1> launch_range(رقم_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE),
                                                 sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            اگر (o0 == n && o1 == n) {
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, item_ct1, func);
                    });
            } وگرنه {
                // launch-invariant divisor, and only this path needs it
                ثابت sycl::uint3 n_fd = init_fastdiv_مقادیر((uint32_t) n);
                main_stream->parallel_for(launch_range,
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, item_ct1, func);
                    });
            }
        });
}


ایستا درونخطی تهی ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    اعشاری start, stop, step;
    memcpy(&start, dst->op_params, اندازه(اعشاری));
    memcpy(&stop, (اعشاری *) dst->op_params + 1, اندازه(اعشاری));
    memcpy(&step, (اعشاری *) dst->op_params + 2, اندازه(اعشاری));
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    اعشاری * dst_ptr = (اعشاری *)dst->data;
    ثابت صحیح k = (صحیح)ggml_nelements(dst);
    ثابت صحیح رقم_blocks = ceil_div(k, SYCL_ARANGE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(رقم_blocks) * sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE),
                          sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            arange_kernel(dst_ptr, k, start, step, item_ct1);
        });
}

} // namespace ggml_sycl_detail



ایستا درونخطی تهی ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_sgn(x);
    });
}


ایستا درونخطی تهی ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_abs(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_elu(x);
    });
}
ایستا درونخطی تهی ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_silu(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_gelu(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_gelu_quick(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_gelu_erf(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_tanh(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_relu(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_hardsigmoid(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_hardswish(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_exp(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_expm1(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت auto* src, auto* dst_ptr, صحیح k_elements, queue_ptr stream) {
            ثابت صحیح رقم_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(رقم_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ایستا درونخطی تهی ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_softplus(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_neg(x);
    });
}


ایستا درونخطی تهی ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_step(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_sigmoid(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت auto* src, auto* dst_ptr, صحیح k_elements, queue_ptr stream) {
            ثابت صحیح رقم_blocks = ceil_div(k_elements, SYCL_SQRT_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(رقم_blocks) * sycl::range<1>(SYCL_SQRT_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQRT_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ایستا درونخطی تهی ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت auto* src, auto* dst_ptr, صحیح k_elements, queue_ptr stream) {
            ثابت صحیح رقم_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(رقم_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ایستا درونخطی تهی ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت auto* src, auto* dst_ptr, صحیح k_elements, queue_ptr stream) {
            ثابت صحیح رقم_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(رقم_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ایستا درونخطی تهی ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    اعشاری negative_slope;
    memcpy(&negative_slope, dst->op_params, اندازه(اعشاری));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت auto* src, auto* dst_ptr, صحیح k_elements, queue_ptr stream, اعشاری slope) {
            ثابت صحیح رقم_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(رقم_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, slope, item_ct1);
                });
        }, negative_slope);
}

ایستا درونخطی تهی ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت auto* src, auto* dst_ptr, صحیح k_elements, queue_ptr stream) {
            ثابت صحیح رقم_blocks = ceil_div(k_elements, SYCL_SQR_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(رقم_blocks) * sycl::range<1>(SYCL_SQR_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQR_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

ایستا درونخطی تهی ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    اعشاری min_val;
    اعشاری max_val;
    memcpy(&min_val, dst->op_params, اندازه(اعشاری));
    memcpy(&max_val, (اعشاری *) dst->op_params + 1, اندازه(اعشاری));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت auto* src, auto* dst_ptr, صحیح k_elements, queue_ptr stream, اعشاری min_arg, اعشاری max_arg) {
            ثابت صحیح رقم_blocks = ceil_div(k_elements, SYCL_CLAMP_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(رقم_blocks) * sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, item_ct1);
                });
        }, min_val, max_val);
}

ایستا درونخطی تهی ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت اعشاری alpha_n = ggml_get_op_params_f32(dst, 1);
    ثابت اعشاری alpha_p = ggml_get_op_params_f32(dst, 2);
    ثابت اعشاری beta    = ggml_get_op_params_f32(dst, 3);
    ثابت اعشاری eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت auto* src, auto* dst_ptr, صحیح k_elements, queue_ptr stream, اعشاری alpha_n_arg, اعشاری alpha_p_arg, اعشاری beta_arg, اعشاری eps_arg) {
            ثابت صحیح رقم_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(رقم_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, item_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

ایستا درونخطی تهی ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_floor(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_ceil(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_round(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        بازگشت op_trunc(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    ثابت نویسه  * src0_d = (ثابت نویسه  *) src0->data;
    ثابت نویسه  * src1_d = (ثابت نویسه  *) src1->data;
    اعشاری       * dst_d  = (اعشاری       *)  dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ثابت int64_t s1     = (int64_t) ((ثابت int32_t *) dst->op_params)[0] / (int64_t) اندازه(اعشاری);
    ثابت int64_t s2     = (int64_t) ((ثابت int32_t *) dst->op_params)[1] / (int64_t) اندازه(اعشاری);
    ثابت int64_t s3     = (int64_t) ((ثابت int32_t *) dst->op_params)[2] / (int64_t) اندازه(اعشاری);
    ثابت int64_t offset = (int64_t) ((ثابت int32_t *) dst->op_params)[3] / (int64_t) اندازه(اعشاری);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, offset, stream);
}

ایستا درونخطی تهی ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        بازگشت op_gelu(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        بازگشت op_relu(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        بازگشت op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
template<typename F>
ایستا تهی dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
    انتخاب (uop) {
        حالت GGML_UNARY_OP_SILU:     launch([](اعشاری v) { بازگشت op_silu(v); });     بشکن;
        حالت GGML_UNARY_OP_SIGMOID:  launch([](اعشاری v) { بازگشت op_sigmoid(v); });  بشکن;
        حالت GGML_UNARY_OP_SOFTPLUS: launch([](اعشاری v) { بازگشت op_softplus(v); }); بشکن;
        پیشفرض:
            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
    }
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
تهی ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_گره, ggml_tensor * mul_گره) {
    scope_op_debug_print scope_dbg_print(__func__, mul_گره, /*num_src=*/2);

    ثابت ggml_tensor * x = unary_گره->src[0];
    ثابت ggml_tensor * g = (mul_گره->src[0] == unary_گره) ? mul_گره->src[1] : mul_گره->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_گره);
    GGML_ASSERT(x->type == g->type && x->type == mul_گره->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_گره));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_گره));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ثابت int64_t k = ggml_nelements(mul_گره);
    ثابت int64_t n = mul_گره->ne[0];

    ثابت auto dispatch_type = [&](auto op) {
        انتخاب (mul_گره->type) {
            حالت GGML_TYPE_F32:
                unary_mul_sycl((ثابت اعشاری *) x->data, (ثابت اعشاری *) g->data, (اعشاری *) mul_گره->data,
                               k, n, x->nb[1] / اندازه(اعشاری), g->nb[1] / اندازه(اعشاری), main_stream, op);
                بشکن;
            حالت GGML_TYPE_F16:
                unary_mul_sycl((ثابت sycl::half *) x->data, (ثابت sycl::half *) g->data, (sycl::half *) mul_گره->data,
                               k, n, x->nb[1] / اندازه(sycl::half), g->nb[1] / اندازه(sycl::half), main_stream, op);
                بشکن;
            پیشفرض:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_گره->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_گره), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
تهی ggml_sycl_op_add_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add_گره,
                                      ggml_tensor * unary_گره, ggml_tensor * mul_گره) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_گره, /*num_src=*/2);

    ثابت ggml_tensor * a     = add_گره->src[0];
    ثابت ggml_tensor * bias  = add_گره->src[1];
    ثابت ggml_tensor * scale = (mul_گره->src[0] == unary_گره) ? mul_گره->src[1] : mul_گره->src[0];

    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(scale != unary_گره);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_گره->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_گره));
    // a and dst are indexed flat
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_گره));
    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ثابت auto dispatch_op = [&](auto op) {
        add_unary_mul_sycl((ثابت اعشاری *) a->data, (ثابت اعشاری *) bias->data, (ثابت اعشاری *) scale->data,
                           (اعشاری *) mul_گره->data, ggml_nelements(mul_گره), mul_گره->ne[0], main_stream, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_گره), dispatch_op);
}

__dpct_inline__ اعشاری ggml_sycl_op_swiglu_oai_single(اعشاری x, اعشاری g, اعشاری alpha = 1.702f, اعشاری limit = 7.0f) {
    x = sycl::fmin(x, limit);
    g = sycl::fmax(sycl::fmin(g, limit), -limit);

    اعشاری out_glu = x / (1.0f + sycl::native::exp(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    بازگشت out_glu;
}

template <typename T>
ایستا تهی swiglu_oai_kernel(ثابت T * x, ثابت T * g, T * dst, ثابت int64_t k,
                              ثابت int64_t n, ثابت int64_t o0, ثابت int64_t o1,
                              اعشاری alpha, اعشاری limit, sycl::nd_item<3> item_ct1) {
    ثابت int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    اگر (i >= k) {
        بازگشت;
    }

    ثابت int64_t j0 = (i / n) * o0 + (i % n);
    ثابت int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    اعشاری xi = x[j0];
    اعشاری gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
ایستا تهی swiglu_oai_sycl(ثابت T *       x,
                            ثابت T *       g,
                            T *             dst,
                            ثابت int64_t   k,
                            ثابت int64_t   n,
                            ثابت int64_t   o0,
                            ثابت int64_t   o1,
                            ثابت اعشاری     alpha,
                            ثابت اعشاری     limit,
                            dpct::queue_ptr stream) {
    ثابت int64_t رقم_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, رقم_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, limit, item_ct1);
                         });
}

ایستا تهی ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    تهی * src0_d = src0->data;
    تهی * src1_d = src1 ? src1->data : src0->data;
    ثابت int64_t src0_o = src0->nb[1];
    ثابت int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    تهی * dst_d = dst->data;
    ثابت int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت اعشاری alpha = ggml_get_op_params_f32(dst, 2);
    ثابت اعشاری limit = ggml_get_op_params_f32(dst, 3);

    اعشاری * src0_p = (اعشاری *) src0_d;
    اعشاری * src1_p = (اعشاری *) src1_d;

    اگر (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (اعشاری *)dst_d, ggml_nelements(dst), nc, src0_o / اندازه(اعشاری), src1_o / اندازه(اعشاری), alpha, limit, stream);
}

template <typename T>
ایستا تهی swiglu_clamp_kernel(ثابت T *        gate,
                                ثابت T *        up,
                                T *              dst,
                                ثابت int64_t    k,
                                ثابت int64_t    n,
                                ثابت int64_t    o0,
                                ثابت int64_t    o1,
                                اعشاری            limit,
                                sycl::nd_item<3> item_ct1) {
    ثابت int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    اگر (i >= k) {
        بازگشت;
    }

    ثابت int64_t j0 = (i / n) * o0 + (i % n);
    ثابت int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ثابت اعشاری gate_مقدار = sycl::fmin((اعشاری) gate[j0], limit);
    ثابت اعشاری up_مقدار   = sycl::fmax(sycl::fmin((اعشاری) up[j1], limit), -limit);
    dst[i]                 = (T) (gate_مقدار / (1.0f + sycl::native::exp(-gate_مقدار)) * up_مقدار);
}

template <typename T>
ایستا تهی swiglu_clamp_sycl(ثابت T *       gate,
                              ثابت T *       up,
                              T *             dst,
                              ثابت int64_t   k,
                              ثابت int64_t   n,
                              ثابت int64_t   o0,
                              ثابت int64_t   o1,
                              اعشاری           limit,
                              dpct::queue_ptr stream) {
    ثابت int64_t رقم_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, رقم_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_clamp_kernel(gate, up, dst, k, n, o0, o1, limit, item_ct1);
                         });
}

ایستا تهی ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0   = dst->src[0];
    ثابت ggml_tensor * src1   = dst->src[1];
    تهی *              src0_d = src0->data;
    تهی *              src1_d = src1 ? src1->data : src0->data;
    ثابت int64_t       src0_o = src0->nb[1];
    ثابت int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    تهی *              dst_d  = dst->data;
    ثابت int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    اگر (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت اعشاری   limit   = ggml_get_op_params_f32(dst, 3);

    اگر (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / اندازه(sycl::half),
                          src1_o / اندازه(sycl::half), limit, stream);
    } وگرنه {
        اعشاری * src0_p = (اعشاری *) src0_d;
        اعشاری * src1_p = (اعشاری *) src1_d;

        اگر (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (اعشاری *) dst_d, ggml_nelements(dst), nc, src0_o / اندازه(اعشاری),
                          src1_o / اندازه(اعشاری), limit, stream);
    }
}

ایستا درونخطی تهی ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        بازگشت op_gelu_erf(x);
    });
}

ایستا درونخطی تهی ggml_sycl_op_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        بازگشت op_gelu_quick(x);
    });
}


تهی ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

تهی ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

تهی ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

تهی ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

تهی ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

تهی ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

تهی ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_quick(ctx, dst);
}

تهی ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

تهی ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

تهی ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

تهی ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

تهی ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

تهی ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

تهی ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

تهی ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_expm1(ctx, dst);
}

تهی ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

تهی ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

تهی ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

تهی ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_step(ctx, dst);
}

تهی ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

تهی ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

تهی ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

تهی ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_xielu(ctx, dst);
}

تهی ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

تهی ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

تهی ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

تهی ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

تهی ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

تهی ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

تهی ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

تهی ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_clamp(ctx, dst);
}

تهی ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

تهی ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_quick(ctx, dst);
}

تهی ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

تهی ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

تهی ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

تهی ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_round(ctx, dst);
}

تهی ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
