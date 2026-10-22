#ضمن "common.hpp"
#ضمن "ggml-sycl/presets.hpp"
#ضمن "ggml.h"
#ضمن "element_wise.hpp"

#تعریف_کړه SYCL_GLOBAL_ID_LOOP(K, توکی_2) \
    لكل (خپلکاری i = توکی_2.get_global_id(0); i < (size_t)K; i += توکی_2.get_global_range(0))

#تعریف_کړه SYCL_LOCAL_ID_CALC(توکی_2, IDX) \
    (توکی_2.get_local_range(IDX) * توکی_2.get_ډله(IDX) + توکی_2.get_local_id(IDX))

ساكن فراغ acc_f32(ثابت حرف * x, ثابت حرف * y, عائم * dst, ثابت int64_t ne,
    ثابت int64_t ne0, ثابت int64_t ne1, ثابت int64_t ne2,
    ثابت int64_t nb00, ثابت int64_t nb01, ثابت int64_t nb02, ثابت int64_t nb03,
    ثابت int64_t ne10, ثابت int64_t ne11, ثابت int64_t ne12, ثابت int64_t ne13,
    ثابت int64_t nb10, ثابت int64_t nb11, ثابت int64_t nb12, ثابت int64_t nb13,
    ثابت int64_t s11, ثابت int64_t s12, ثابت int64_t s13, ثابت int64_t دننه_واټن) {
    خپلکاری توکی_ct1 = sycl::ext::oneapi::this_work_توکی::get_nd_توکی<3>();
    ثابت int64_t i = SYCL_LOCAL_ID_CALC(توکی_ct1, 2);

    إذا (i >= ne) {
        أرجع;
    }

    int64_t src1_idx = i - دننه_واټن;

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

    عائم val = *(ثابت عائم *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    إذا (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(ثابت عائم *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_sgn(T x) {
    أرجع x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_abs(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } وإلا {
        أرجع sycl::fabs(x);
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_expm1(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<عائم>(x))
        );
    } وإلا {
        أرجع sycl::expm1(x);
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_elu(T x) {
    أرجع (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_erf(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<عائم>(x))
        );
    } وإلا {
        أرجع sycl::erf(x);
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_gelu_erf(T x) {
    ثابت T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    أرجع static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_gelu_چټک(T x) {
    ثابت T GELU_چټک_COEF_LOCAL = static_cast<T>(-1.702f);
    أرجع x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_چټک_COEF_LOCAL * x)));
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_relu(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } وإلا {
        أرجع sycl::fmax(x, static_cast<T>(0));
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_sigmoid(T x) {
    أرجع static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_sqrt(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::جذر(x);
    } وإلا {
        أرجع sycl::جذر(x);
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_sin(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::ساین(x);
    } وإلا {
        أرجع sycl::ساین(x);
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_cos(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::کوساین(x);
    } وإلا {
        أرجع sycl::کوساین(x);
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_hardsigmoid(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } وإلا {
        أرجع sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_hardswish(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } وإلا {
        أرجع x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_log(T x) {
    إذا (x <= static_cast<T>(0)) {
        أرجع neg_infinity<T>();
    }
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::لوګاریتم(x);
    } وإلا {
        أرجع sycl::لوګاریتم(x);
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_softplus(T x) {
    ثابت عائم xf = (عائم) x;
    ثابت عائم ax = op_abs(xf);
    ثابت عائم m  = sycl::fmax(xf, 0.0f);
    ثابت عائم y  = m + sycl::log1p(sycl::توانیز_تابع(-ax));
    أرجع (T) y;
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_neg(T x) {
    أرجع -x;
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_ګام(T x) {
    أرجع (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_leaky_relu(T x, عائم negative_میلان) {
    T neg_میلان_T = static_cast<T>(negative_میلان);
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_میلان_T;

    } وإلا {
        أرجع sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_میلان_T;
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_xielu(T x, عائم alpha_n, عائم alpha_p, عائم beta, عائم eps) {
    ثابت عائم xi        = static_cast<عائم>(x);
    ثابت عائم gate_pos  = (xi > 0.0f);
    ثابت عائم y_pos     = alpha_p * xi * xi + beta * xi;
    ثابت عائم min_v_eps = sycl::fmin(xi, eps);
    ثابت عائم y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    ثابت عائم بهر       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    أرجع static_cast<T>(بهر);
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_sqr(T x) {
    أرجع x * x;
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_clamp(T x, عائم min_val, عائم max_val) {
    أرجع x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_floor(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::floor(x);
    } وإلا {
        أرجع sycl::floor(x);
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_ceil(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::ceil(x);
    } وإلا {
        أرجع sycl::ceil(x);
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_پړاو(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<عائم>(x))
        );
    } وإلا {
        أرجع sycl::round(x);
    }
}

کينډۍ<ډول_نوم T>
ساكن __dpct_inline__ T op_trunc(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::trunc(x);
    } وإلا {
        أرجع sycl::trunc(x);
    }
}

کينډۍ<ډول_نوم T, ډول_نوم F>
ساكن فراغ unary_op_flat_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_توکی<1> & توکی_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = func(x[i]);
    }
}

کينډۍ<ډول_نوم T, ډول_نوم F>
ساكن فراغ unary_op_generic_kernel(
        ثابت T * x,
        T * dst,
        ثابت صحيح k,
        ثابت sycl::uint3 ne0_fd, ثابت sycl::uint3 ne1_fd, ثابت sycl::uint3 ne2_fd,
        ثابت size_t nb0,  ثابت size_t nb1,  ثابت size_t nb2,  ثابت size_t nb3,
        ثابت size_t nbd0, ثابت size_t nbd1, ثابت size_t nbd2, ثابت size_t nbd3,
        ثابت sycl::nd_توکی<1> & توکی_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        ثابت uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        ثابت uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        ثابت uint32_t i2 = dm.y();
        ثابت uint32_t i3 = dm.x();

        ثابت حرف * src_بنسټ = (ثابت حرف *) x;
        حرف       * dst_بنسټ = (حرف *) dst;

        ثابت T * srcp = (ثابت T *)(src_بنسټ + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_بنسټ + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

کينډۍ<ډول_نوم T>
ساكن فراغ unary_op_sqrt_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_توکی<1> &توکی_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

کينډۍ<ډول_نوم T>
ساكن فراغ unary_op_sin_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_توکی<1> &توکی_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

کينډۍ<ډول_نوم T>
ساكن فراغ unary_op_cos_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_توکی<1> &توکی_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

کينډۍ<ډول_نوم T>
ساكن فراغ unary_op_log_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_توکی<1> &توکی_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = op_log(x[i]);
    }
}


کينډۍ<ډول_نوم T>
ساكن فراغ unary_op_leaky_relu_kernel(ثابت T * x, T * dst, ثابت صحيح k, عائم negative_میلان, ثابت sycl::nd_توکی<1> &توکی_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_میلان);
    }
}

کينډۍ<ډول_نوم T>
ساكن فراغ unary_op_xielu_kernel(ثابت T * x, T * dst, ثابت صحيح k, عائم alpha_n, عائم alpha_p, عائم beta, عائم eps, ثابت sycl::nd_توکی<1> &توکی_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

کينډۍ<ډول_نوم T>
ساكن فراغ unary_op_sqr_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_توکی<1> &توکی_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

کينډۍ<ډول_نوم T>
ساكن فراغ unary_op_clamp_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_توکی<1> &توکی_ct1, عائم min_val, عائم max_val) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

کينډۍ<ډول_نوم T>
ساكن فراغ unary_op_ceil_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_توکی<1> &توکی_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

کينډۍ<ډول_نوم T>
ساكن فراغ clamp(ثابت T * x, T * dst, ثابت عائم min, ثابت عائم max, ثابت صحيح k,
                      ثابت sycl::nd_توکی<1> &توکی_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

کينډۍ<ډول_نوم T, ډول_نوم F>
ساكن فراغ unary_gated_op_flat_kernel(ثابت T * x, ثابت T * g, T * dst, ثابت uint64_t k, ثابت sycl::nd_توکی<1> & توکی_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

کينډۍ<ډول_نوم T, ډول_نوم F>
ساكن فراغ unary_gated_op_generic_kernel(
        ثابت T * x,
        ثابت T * g,
        T * dst,
        ثابت uint64_t k,
        ثابت sycl::uint3 n_fd,
        ثابت uint64_t o0,
        ثابت uint64_t o1,
        ثابت sycl::nd_توکی<1> & توکی_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        ثابت sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ثابت int64_t j0 = rc.x() * o0 + rc.y();
        ثابت int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
کينډۍ<ډول_نوم T, ډول_نوم F>
ساكن فراغ unary_mul_flat_kernel(ثابت T * x, ثابت T * g, T * dst, ثابت int64_t k, ثابت sycl::nd_توکی<1> &توکی_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = (T) (op((عائم) x[i]) * (عائم) g[i]);
    }
}

کينډۍ<ډول_نوم T, ډول_نوم F>
ساكن فراغ unary_mul_strided_kernel(ثابت T * x, ثابت T * g, T * dst, ثابت int64_t k, ثابت sycl::uint3 n_fd, ثابت int64_t o0, ثابت int64_t o1, ثابت sycl::nd_توکی<1> &توکی_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        ثابت sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ثابت int64_t j0 = rc.x() * o0 + rc.y();
        ثابت int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((عائم) x[j0]) * (عائم) g[j1]);
    }
}

کينډۍ<ډول_نوم T, ډول_نوم F>
ساكن فراغ unary_mul_sycl(ثابت T * x, ثابت T * g, T * dst, ثابت int64_t k, ثابت int64_t n, ثابت int64_t o0, ثابت int64_t o1, کتار_ptr main_بهیر, F op) {
    ثابت size_t            عدد_بلاکونه = ceil_div((size_t) k, (size_t) SYCL_GLU_بلاک_اندازه);
    ثابت sycl::nd_range<1> range(عدد_بلاکونه * sycl::range<1>(SYCL_GLU_بلاک_اندازه), sycl::range<1>(SYCL_GLU_بلاک_اندازه));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    إذا (o0 == n && o1 == n) {
        main_بهیر->parallel_for(range, [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
            unary_mul_flat_kernel(x, g, dst, k, توکی_ct1, op);
        });
        أرجع;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ثابت sycl::uint3 n_fd = init_fastdiv_ارزښتونه((uint32_t) n);
    main_بهیر->parallel_for(range, [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, توکی_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
کينډۍ<ډول_نوم F>
ساكن فراغ جمع_کړه_unary_mul_flat_kernel(ثابت عائم * a, ثابت عائم * bias, ثابت عائم * scale, عائم * dst,
                                      ثابت int64_t k, ثابت sycl::nd_توکی<1> &توکی_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

کينډۍ<ډول_نوم F>
ساكن فراغ جمع_کړه_unary_mul_bcast_kernel(ثابت عائم * a, ثابت عائم * bias, ثابت عائم * scale, عائم * dst,
                                       ثابت int64_t k, ثابت sycl::uint3 ne0_fd, ثابت sycl::nd_توکی<1> &توکی_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        ثابت uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

کينډۍ<ډول_نوم F>
ساكن فراغ جمع_کړه_unary_mul_sycl(ثابت عائم * a, ثابت عائم * bias, ثابت عائم * scale, عائم * dst,
                               ثابت int64_t k, ثابت int64_t ne0, کتار_ptr main_بهیر, F op) {
    ثابت size_t            عدد_بلاکونه = ceil_div((size_t) k, (size_t) SYCL_GLU_بلاک_اندازه);
    ثابت sycl::nd_range<1> range(عدد_بلاکونه * sycl::range<1>(SYCL_GLU_بلاک_اندازه), sycl::range<1>(SYCL_GLU_بلاک_اندازه));

    إذا (k == ne0) {
        main_بهیر->parallel_for(range, [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
            جمع_کړه_unary_mul_flat_kernel(a, bias, scale, dst, k, توکی_ct1, op);
        });
        أرجع;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ثابت sycl::uint3 ne0_fd = init_fastdiv_ارزښتونه((uint32_t) ne0);
    main_بهیر->parallel_for(range, [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
        جمع_کړه_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, توکی_ct1, op);
    });
}

نطاق ggml_sycl_detail {
ساكن فراغ acc_f32_sycl(ثابت حرف *x, ثابت حرف *y, عائم *dst,
                         ثابت int64_t n_elements,
                         ثابت int64_t ne0, ثابت int64_t ne1, ثابت int64_t ne2,
                         ثابت int64_t nb00, ثابت int64_t nb01, ثابت int64_t nb02, ثابت int64_t nb03,
                         ثابت int64_t ne10, ثابت int64_t ne11, ثابت int64_t ne12, ثابت int64_t ne13,
                         ثابت int64_t nb10, ثابت int64_t nb11, ثابت int64_t nb12, ثابت int64_t nb13,
                         ثابت int64_t s1, ثابت int64_t s2, ثابت int64_t s3,
                         ثابت int64_t دننه_واټن, کتار_ptr stream) {
    ثابت صحيح عدد_بلاکونه = (n_elements + SYCL_ACC_بلاک_اندازه - 1) / SYCL_ACC_بلاک_اندازه;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکونه) * sycl::range<3>(1, 1, SYCL_ACC_بلاک_اندازه),
                                           sycl::range<3>(1, 1, SYCL_ACC_بلاک_اندازه)),
                         [=](sycl::nd_توکی<3> /*item_ct1*/) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                            acc_f32(x, y, dst, n_elements,
                                ne0, ne1, ne2,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                s1, s2, s3, دننه_واټن);
                         });
}

کينډۍ<ډول_نوم T>
ساكن فراغ arange_kernel(T * dst, ثابت صحيح k, T start, T ګام,
                         ثابت sycl::nd_توکی<1> &توکی_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, توکی_ct1) {
        dst[i] = start + static_cast<T>(i) * ګام;
    }
}

کينډۍ<ډول_نوم KernelInvoker, ډول_نوم... دلیلونه>
ساكن مضمن فراغ dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, دلیلونه&&... دلیلونه_2) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::کتار_ptr main_بهیر = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));
    بدل (dst->type) {
        حالة GGML_TYPE_F16:
            {
                خپلکاری معلومات_pts = cast_معلومات<sycl::half>(dst);
                kernel_invoker(معلومات_pts.src, معلومات_pts.dst, (صحيح)ggml_nelements(dst->src[0]), main_بهیر, std::forward<دلیلونه>(دلیلونه_2)...);
                اكسر;
            }
#که_تعریف_شوی GGML_SYCL_HAS_BF16
        حالة GGML_TYPE_BF16:
            {
                خپلکاری معلومات_pts = cast_معلومات<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(معلومات_pts.src, معلومات_pts.dst, (صحيح)ggml_nelements(dst->src[0]), main_بهیر, std::forward<دلیلونه>(دلیلونه_2)...);
                اكسر;
            }
#د_که_پای
        حالة GGML_TYPE_F32:
            {
                خپلکاری معلومات_pts = cast_معلومات<عائم>(dst);
                kernel_invoker(معلومات_pts.src, معلومات_pts.dst, (صحيح)ggml_nelements(dst->src[0]), main_بهیر, std::forward<دلیلونه>(دلیلونه_2)...);
                اكسر;
            }
        افتراضي:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

کينډۍ<ډول_نوم KernelInvoker, ډول_نوم... دلیلونه>
ساكن مضمن فراغ dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, دلیلونه&&... دلیلونه_2) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::کتار_ptr main_بهیر = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));
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
    فراغ * src0_d = src0->معلومات;
    فراغ * src1_d = src1 ? src1->معلومات : src0->معلومات;
    ثابت int64_t src0_o = src0->nb[1];
    ثابت int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    فراغ * dst_d = dst->معلومات;
    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_اندازه(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    بدل (dst->type) {
        حالة GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    إذا (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / حجم(sycl::half),
                               src1_o / حجم(sycl::half),
                               main_بهیر,
                               std::forward<دلیلونه>(دلیلونه_2)...);
                اكسر;
            }
        حالة GGML_TYPE_F32:
            {
                عائم * src0_p = (عائم *) src0_d;
                عائم * src1_p = (عائم *) src1_d;

                    إذا (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (عائم *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / حجم(عائم),
                               src1_o / حجم(عائم),
                               main_بهیر,
                               std::forward<دلیلونه>(دلیلونه_2)...);
                اكسر;
            }
        افتراضي:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

کينډۍ<ډول_نوم F>
ساكن مضمن فراغ ggml_sycl_op_unary(
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
    ثابت منطقي contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](ثابت خپلکاری* src, خپلکاری* dst_ptr, صحيح k_elements, کتار_ptr stream) {

            ثابت صحيح عدد_بلاکونه = ceil_div(k_elements, 256);

            إذا (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(عدد_بلاکونه) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, توکی_ct1, func);
                    });
            } وإلا {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                ثابت sycl::uint3 ne0_fd = init_fastdiv_ارزښتونه((uint32_t) ne0);
                ثابت sycl::uint3 ne1_fd = init_fastdiv_ارزښتونه((uint32_t) ne1);
                ثابت sycl::uint3 ne2_fd = init_fastdiv_ارزښتونه((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(عدد_بلاکونه) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                        unary_op_generic_kernel(
                            src, dst_ptr, k_elements,
                            ne0_fd, ne1_fd, ne2_fd,
                            nb0, nb1, nb2, nb3,
                            nbd0, nbd1, nbd2, nbd3,
                            توکی_ct1,
                            func
                        );
                    });
            }
        });
}

کينډۍ<ډول_نوم F>
ساكن مضمن فراغ ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](ثابت خپلکاری * x_ptr, ثابت خپلکاری * g_ptr, خپلکاری * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, کتار_ptr main_بهیر) {

            ثابت uint32_t عدد_بلاکونه = (uint32_t) ceil_div(k, SYCL_GLU_بلاک_اندازه);
            ثابت sycl::nd_range<1> launch_range(عدد_بلاکونه * sycl::range<1>(SYCL_GLU_بلاک_اندازه),
                                                 sycl::range<1>(SYCL_GLU_بلاک_اندازه));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            إذا (o0 == n && o1 == n) {
                main_بهیر->parallel_for(launch_range,
                    [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, توکی_ct1, func);
                    });
            } وإلا {
                // launch-invariant divisor, and only this path needs it
                ثابت sycl::uint3 n_fd = init_fastdiv_ارزښتونه((uint32_t) n);
                main_بهیر->parallel_for(launch_range,
                    [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, توکی_ct1, func);
                    });
            }
        });
}


ساكن مضمن فراغ ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    عائم start, stop, ګام;
    حافظه_نقل(&start, dst->op_params, حجم(عائم));
    حافظه_نقل(&stop, (عائم *) dst->op_params + 1, حجم(عائم));
    حافظه_نقل(&ګام, (عائم *) dst->op_params + 2, حجم(عائم));
    dpct::کتار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));
    عائم * dst_ptr = (عائم *)dst->معلومات;
    ثابت صحيح k = (صحيح)ggml_nelements(dst);
    ثابت صحيح عدد_بلاکونه = ceil_div(k, SYCL_ARANGE_بلاک_اندازه);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(عدد_بلاکونه) * sycl::range<1>(SYCL_ARANGE_بلاک_اندازه),
                          sycl::range<1>(SYCL_ARANGE_بلاک_اندازه)),
        [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
            arange_kernel(dst_ptr, k, start, ګام, توکی_ct1);
        });
}

} // namespace ggml_sycl_detail



ساكن مضمن فراغ ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_sgn(x);
    });
}


ساكن مضمن فراغ ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_abs(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_elu(x);
    });
}
ساكن مضمن فراغ ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_silu(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_gelu(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_gelu_چټک(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_gelu_چټک(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_gelu_erf(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_tanh(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_relu(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_hardsigmoid(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_hardswish(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_exp(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_expm1(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خپلکاری* src, خپلکاری* dst_ptr, صحيح k_elements, کتار_ptr stream) {
            ثابت صحيح عدد_بلاکونه = ceil_div(k_elements, SYCL_EXP_بلاک_اندازه); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکونه) * sycl::range<1>(SYCL_EXP_بلاک_اندازه),
                                  sycl::range<1>(SYCL_EXP_بلاک_اندازه)),
                [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, توکی_ct1);
                });
        });
}

ساكن مضمن فراغ ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_softplus(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_neg(x);
    });
}


ساكن مضمن فراغ ggml_sycl_op_ګام(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_ګام(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_sigmoid(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خپلکاری* src, خپلکاری* dst_ptr, صحيح k_elements, کتار_ptr stream) {
            ثابت صحيح عدد_بلاکونه = ceil_div(k_elements, SYCL_SQRT_بلاک_اندازه);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکونه) * sycl::range<1>(SYCL_SQRT_بلاک_اندازه),
                                  sycl::range<1>(SYCL_SQRT_بلاک_اندازه)),
                [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, توکی_ct1);
                });
        });
}

ساكن مضمن فراغ ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خپلکاری* src, خپلکاری* dst_ptr, صحيح k_elements, کتار_ptr stream) {
            ثابت صحيح عدد_بلاکونه = ceil_div(k_elements, SYCL_SIN_بلاک_اندازه);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکونه) * sycl::range<1>(SYCL_SIN_بلاک_اندازه),
                                  sycl::range<1>(SYCL_SIN_بلاک_اندازه)),
                [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, توکی_ct1);
                });
        });
}

ساكن مضمن فراغ ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خپلکاری* src, خپلکاری* dst_ptr, صحيح k_elements, کتار_ptr stream) {
            ثابت صحيح عدد_بلاکونه = ceil_div(k_elements, SYCL_SIN_بلاک_اندازه); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکونه) * sycl::range<1>(SYCL_SIN_بلاک_اندازه),
                                  sycl::range<1>(SYCL_SIN_بلاک_اندازه)),
                [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, توکی_ct1);
                });
        });
}

ساكن مضمن فراغ ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    عائم negative_میلان;
    حافظه_نقل(&negative_میلان, dst->op_params, حجم(عائم));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خپلکاری* src, خپلکاری* dst_ptr, صحيح k_elements, کتار_ptr stream, عائم میلان) {
            ثابت صحيح عدد_بلاکونه = ceil_div(k_elements, SYCL_RELU_بلاک_اندازه);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکونه) * sycl::range<1>(SYCL_RELU_بلاک_اندازه),
                                  sycl::range<1>(SYCL_RELU_بلاک_اندازه)),
                [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, میلان, توکی_ct1);
                });
        }, negative_میلان);
}

ساكن مضمن فراغ ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خپلکاری* src, خپلکاری* dst_ptr, صحيح k_elements, کتار_ptr stream) {
            ثابت صحيح عدد_بلاکونه = ceil_div(k_elements, SYCL_SQR_بلاک_اندازه);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکونه) * sycl::range<1>(SYCL_SQR_بلاک_اندازه),
                                  sycl::range<1>(SYCL_SQR_بلاک_اندازه)),
                [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, توکی_ct1);
                });
        });
}

ساكن مضمن فراغ ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    عائم min_val;
    عائم max_val;
    حافظه_نقل(&min_val, dst->op_params, حجم(عائم));
    حافظه_نقل(&max_val, (عائم *) dst->op_params + 1, حجم(عائم));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خپلکاری* src, خپلکاری* dst_ptr, صحيح k_elements, کتار_ptr stream, عائم min_arg, عائم max_arg) {
            ثابت صحيح عدد_بلاکونه = ceil_div(k_elements, SYCL_CLAMP_بلاک_اندازه);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکونه) * sycl::range<1>(SYCL_CLAMP_بلاک_اندازه),
                                  sycl::range<1>(SYCL_CLAMP_بلاک_اندازه)),
                [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, توکی_ct1);
                });
        }, min_val, max_val);
}

ساكن مضمن فراغ ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت عائم alpha_n = ggml_get_op_params_f32(dst, 1);
    ثابت عائم alpha_p = ggml_get_op_params_f32(dst, 2);
    ثابت عائم beta    = ggml_get_op_params_f32(dst, 3);
    ثابت عائم eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خپلکاری* src, خپلکاری* dst_ptr, صحيح k_elements, کتار_ptr stream, عائم alpha_n_arg, عائم alpha_p_arg, عائم beta_arg, عائم eps_arg) {
            ثابت صحيح عدد_بلاکونه = ceil_div(k_elements, SYCL_RELU_بلاک_اندازه);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکونه) * sycl::range<1>(SYCL_RELU_بلاک_اندازه),
                                  sycl::range<1>(SYCL_RELU_بلاک_اندازه)),
                [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, توکی_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

ساكن مضمن فراغ ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_floor(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_ceil(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_پړاو(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_پړاو(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خپلکاری x) {
        أرجع op_trunc(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    ثابت حرف  * src0_d = (ثابت حرف  *) src0->معلومات;
    ثابت حرف  * src1_d = (ثابت حرف  *) src1->معلومات;
    عائم       * dst_d  = (عائم       *)  dst->معلومات;

    dpct::کتار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_اندازه(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ثابت int64_t s1     = (int64_t) ((ثابت int32_t *) dst->op_params)[0] / (int64_t) حجم(عائم);
    ثابت int64_t s2     = (int64_t) ((ثابت int32_t *) dst->op_params)[1] / (int64_t) حجم(عائم);
    ثابت int64_t s3     = (int64_t) ((ثابت int32_t *) dst->op_params)[2] / (int64_t) حجم(عائم);
    ثابت int64_t دننه_واټن = (int64_t) ((ثابت int32_t *) dst->op_params)[3] / (int64_t) حجم(عائم);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, دننه_واټن, stream);
}

ساكن مضمن فراغ ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](خپلکاری x) {
        أرجع op_gelu(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](خپلکاری x) {
        أرجع op_relu(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](خپلکاری x) {
        أرجع op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
کينډۍ<ډول_نوم F>
ساكن فراغ dispatch_fused_unary_op(ggml_unary_op uop, F && launch) {
    بدل (uop) {
        حالة GGML_UNARY_OP_SILU:     launch([](عائم v) { أرجع op_silu(v); });     اكسر;
        حالة GGML_UNARY_OP_SIGMOID:  launch([](عائم v) { أرجع op_sigmoid(v); });  اكسر;
        حالة GGML_UNARY_OP_SOFTPLUS: launch([](عائم v) { أرجع op_softplus(v); }); اكسر;
        افتراضي:
            GGML_ABORT("fused unary chain: unsupported unary op %s", ggml_unary_op_name(uop));
    }
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
فراغ ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_غوټه, ggml_tensor * mul_غوټه) {
    scope_op_debug_print scope_dbg_print(__func__, mul_غوټه, /*num_src=*/2);

    ثابت ggml_tensor * x = unary_غوټه->src[0];
    ثابت ggml_tensor * g = (mul_غوټه->src[0] == unary_غوټه) ? mul_غوټه->src[1] : mul_غوټه->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_غوټه);
    GGML_ASSERT(x->type == g->type && x->type == mul_غوټه->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_غوټه));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_غوټه));

    کتار_ptr main_بهیر = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));

    ثابت int64_t k = ggml_nelements(mul_غوټه);
    ثابت int64_t n = mul_غوټه->ne[0];

    ثابت خپلکاری dispatch_type = [&](خپلکاری op) {
        بدل (mul_غوټه->type) {
            حالة GGML_TYPE_F32:
                unary_mul_sycl((ثابت عائم *) x->معلومات, (ثابت عائم *) g->معلومات, (عائم *) mul_غوټه->معلومات,
                               k, n, x->nb[1] / حجم(عائم), g->nb[1] / حجم(عائم), main_بهیر, op);
                اكسر;
            حالة GGML_TYPE_F16:
                unary_mul_sycl((ثابت sycl::half *) x->معلومات, (ثابت sycl::half *) g->معلومات, (sycl::half *) mul_غوټه->معلومات,
                               k, n, x->nb[1] / حجم(sycl::half), g->nb[1] / حجم(sycl::half), main_بهیر, op);
                اكسر;
            افتراضي:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_غوټه->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_غوټه), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
فراغ ggml_sycl_op_جمع_کړه_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * جمع_کړه_غوټه,
                                      ggml_tensor * unary_غوټه, ggml_tensor * mul_غوټه) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_غوټه, /*num_src=*/2);

    ثابت ggml_tensor * a     = جمع_کړه_غوټه->src[0];
    ثابت ggml_tensor * bias  = جمع_کړه_غوټه->src[1];
    ثابت ggml_tensor * scale = (mul_غوټه->src[0] == unary_غوټه) ? mul_غوټه->src[1] : mul_غوټه->src[0];

    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(scale != unary_غوټه);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_غوټه->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_غوټه));
    // a and dst are indexed flat
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_غوټه));
    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    کتار_ptr main_بهیر = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));

    ثابت خپلکاری dispatch_op = [&](خپلکاری op) {
        جمع_کړه_unary_mul_sycl((ثابت عائم *) a->معلومات, (ثابت عائم *) bias->معلومات, (ثابت عائم *) scale->معلومات,
                           (عائم *) mul_غوټه->معلومات, ggml_nelements(mul_غوټه), mul_غوټه->ne[0], main_بهیر, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_غوټه), dispatch_op);
}

__dpct_inline__ عائم ggml_sycl_op_swiglu_oai_single(عائم x, عائم g, عائم alpha = 1.702f, عائم حد = 7.0f) {
    x = sycl::fmin(x, حد);
    g = sycl::fmax(sycl::fmin(g, حد), -حد);

    عائم بهر_glu = x / (1.0f + sycl::native::توانیز_تابع(-x * alpha));
    بهر_glu = بهر_glu * (1.0f + g);
    أرجع بهر_glu;
}

کينډۍ <ډول_نوم T>
ساكن فراغ swiglu_oai_kernel(ثابت T * x, ثابت T * g, T * dst, ثابت int64_t k,
                              ثابت int64_t n, ثابت int64_t o0, ثابت int64_t o1,
                              عائم alpha, عائم حد, sycl::nd_توکی<3> توکی_ct1) {
    ثابت int64_t i = int64_t(توکی_ct1.get_local_range(2)) * توکی_ct1.get_ډله(2) + توکی_ct1.get_local_id(2);

    إذا (i >= k) {
        أرجع;
    }

    ثابت int64_t j0 = (i / n) * o0 + (i % n);
    ثابت int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    عائم xi = x[j0];
    عائم gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, حد);
}

کينډۍ <ډول_نوم T>
ساكن فراغ swiglu_oai_sycl(ثابت T *       x,
                            ثابت T *       g,
                            T *             dst,
                            ثابت int64_t   k,
                            ثابت int64_t   n,
                            ثابت int64_t   o0,
                            ثابت int64_t   o1,
                            ثابت عائم     alpha,
                            ثابت عائم     حد,
                            dpct::کتار_ptr stream) {
    ثابت int64_t عدد_بلاکونه = (k + SYCL_GLU_بلاک_اندازه - 1) / SYCL_GLU_بلاک_اندازه;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکونه) * sycl::range<3>(1, 1, SYCL_GLU_بلاک_اندازه),
                                           sycl::range<3>(1, 1, SYCL_GLU_بلاک_اندازه)),
                         [=](sycl::nd_توکی<3> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, حد, توکی_ct1);
                         });
}

ساكن فراغ ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    فراغ * src0_d = src0->معلومات;
    فراغ * src1_d = src1 ? src1->معلومات : src0->معلومات;
    ثابت int64_t src0_o = src0->nb[1];
    ثابت int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    فراغ * dst_d = dst->معلومات;
    ثابت int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::کتار_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_اندازه(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_اندازه(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت عائم alpha = ggml_get_op_params_f32(dst, 2);
    ثابت عائم حد = ggml_get_op_params_f32(dst, 3);

    عائم * src0_p = (عائم *) src0_d;
    عائم * src1_p = (عائم *) src1_d;

    إذا (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (عائم *)dst_d, ggml_nelements(dst), nc, src0_o / حجم(عائم), src1_o / حجم(عائم), alpha, حد, stream);
}

کينډۍ <ډول_نوم T>
ساكن فراغ swiglu_clamp_kernel(ثابت T *        gate,
                                ثابت T *        پورته,
                                T *              dst,
                                ثابت int64_t    k,
                                ثابت int64_t    n,
                                ثابت int64_t    o0,
                                ثابت int64_t    o1,
                                عائم            حد,
                                sycl::nd_توکی<3> توکی_ct1) {
    ثابت int64_t i = int64_t(توکی_ct1.get_local_range(2)) * توکی_ct1.get_ډله(2) + توکی_ct1.get_local_id(2);

    إذا (i >= k) {
        أرجع;
    }

    ثابت int64_t j0 = (i / n) * o0 + (i % n);
    ثابت int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ثابت عائم gate_ارزښت = sycl::fmin((عائم) gate[j0], حد);
    ثابت عائم پورته_ارزښت   = sycl::fmax(sycl::fmin((عائم) پورته[j1], حد), -حد);
    dst[i]                 = (T) (gate_ارزښت / (1.0f + sycl::native::توانیز_تابع(-gate_ارزښت)) * پورته_ارزښت);
}

کينډۍ <ډول_نوم T>
ساكن فراغ swiglu_clamp_sycl(ثابت T *       gate,
                              ثابت T *       پورته,
                              T *             dst,
                              ثابت int64_t   k,
                              ثابت int64_t   n,
                              ثابت int64_t   o0,
                              ثابت int64_t   o1,
                              عائم           حد,
                              dpct::کتار_ptr stream) {
    ثابت int64_t عدد_بلاکونه = (k + SYCL_GLU_بلاک_اندازه - 1) / SYCL_GLU_بلاک_اندازه;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکونه) * sycl::range<3>(1, 1, SYCL_GLU_بلاک_اندازه),
                                           sycl::range<3>(1, 1, SYCL_GLU_بلاک_اندازه)),
                         [=](sycl::nd_توکی<3> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                             swiglu_clamp_kernel(gate, پورته, dst, k, n, o0, o1, حد, توکی_ct1);
                         });
}

ساكن فراغ ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0   = dst->src[0];
    ثابت ggml_tensor * src1   = dst->src[1];
    فراغ *              src0_d = src0->معلومات;
    فراغ *              src1_d = src1 ? src1->معلومات : src0->معلومات;
    ثابت int64_t       src0_o = src0->nb[1];
    ثابت int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    فراغ *              dst_d  = dst->معلومات;
    ثابت int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::کتار_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_اندازه(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_اندازه(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت عائم   حد   = ggml_get_op_params_f32(dst, 3);

    إذا (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / حجم(sycl::half),
                          src1_o / حجم(sycl::half), حد, stream);
    } وإلا {
        عائم * src0_p = (عائم *) src0_d;
        عائم * src1_p = (عائم *) src1_d;

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (عائم *) dst_d, ggml_nelements(dst), nc, src0_o / حجم(عائم),
                          src1_o / حجم(عائم), حد, stream);
    }
}

ساكن مضمن فراغ ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](خپلکاری x) {
        أرجع op_gelu_erf(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_geglu_چټک(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](خپلکاری x) {
        أرجع op_gelu_چټک(x);
    });
}


فراغ ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

فراغ ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

فراغ ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

فراغ ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

فراغ ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

فراغ ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

فراغ ggml_sycl_gelu_چټک(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_چټک(ctx, dst);
}

فراغ ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

فراغ ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

فراغ ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

فراغ ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

فراغ ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

فراغ ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

فراغ ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

فراغ ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_expm1(ctx, dst);
}

فراغ ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

فراغ ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

فراغ ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

فراغ ggml_sycl_ګام(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ګام(ctx, dst);
}

فراغ ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

فراغ ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

فراغ ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

فراغ ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_xielu(ctx, dst);
}

فراغ ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

فراغ ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

فراغ ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

فراغ ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

فراغ ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

فراغ ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

فراغ ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

فراغ ggml_sycl_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_clamp(ctx, dst);
}

فراغ ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

فراغ ggml_sycl_geglu_چټک(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_چټک(ctx, dst);
}

فراغ ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

فراغ ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

فراغ ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

فراغ ggml_sycl_پړاو(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_پړاو(ctx, dst);
}

فراغ ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
