#ضمن "common.hpp"
#ضمن "ggml-sycl/presets.hpp"
#ضمن "ggml.h"
#ضمن "element_wise.hpp"

#تعریف_کرو SYCL_GLOBAL_ID_LOOP(K, شے_2) \
    لكل (خودکار i = شے_2.get_global_id(0); i < (size_t)K; i += شے_2.get_global_range(0))

#تعریف_کرو SYCL_LOCAL_ID_CALC(شے_2, IDX) \
    (شے_2.get_local_range(IDX) * شے_2.get_گروہ(IDX) + شے_2.get_local_id(IDX))

ساكن فراغ acc_f32(ثابت حرف * x, ثابت حرف * y, عائم * dst, ثابت int64_t ne,
    ثابت int64_t ne0, ثابت int64_t ne1, ثابت int64_t ne2,
    ثابت int64_t nb00, ثابت int64_t nb01, ثابت int64_t nb02, ثابت int64_t nb03,
    ثابت int64_t ne10, ثابت int64_t ne11, ثابت int64_t ne12, ثابت int64_t ne13,
    ثابت int64_t nb10, ثابت int64_t nb11, ثابت int64_t nb12, ثابت int64_t nb13,
    ثابت int64_t s11, ثابت int64_t s12, ثابت int64_t s13, ثابت int64_t وتھ) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت int64_t i = SYCL_LOCAL_ID_CALC(شے_ct1, 2);

    إذا (i >= ne) {
        أرجع;
    }

    int64_t src1_idx = i - وتھ;

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
سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_sgn(T x) {
    أرجع x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_abs(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } وإلا {
        أرجع sycl::fabs(x);
    }
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_expm1(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<عائم>(x))
        );
    } وإلا {
        أرجع sycl::expm1(x);
    }
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_elu(T x) {
    أرجع (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_erf(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<عائم>(x))
        );
    } وإلا {
        أرجع sycl::erf(x);
    }
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_gelu_erf(T x) {
    ثابت T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    أرجع static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_gelu_تیز(T x) {
    ثابت T GELU_تیز_COEF_LOCAL = static_cast<T>(-1.702f);
    أرجع x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_تیز_COEF_LOCAL * x)));
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_relu(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } وإلا {
        أرجع sycl::fmax(x, static_cast<T>(0));
    }
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_sigmoid(T x) {
    أرجع static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_sqrt(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::جذر(x);
    } وإلا {
        أرجع sycl::جذر(x);
    }
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_sin(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::جیب(x);
    } وإلا {
        أرجع sycl::جیب(x);
    }
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_cos(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::جیب_تمام(x);
    } وإلا {
        أرجع sycl::جیب_تمام(x);
    }
}

سانچا<قسم_ناں T>
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

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_hardswish(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } وإلا {
        أرجع x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_log(T x) {
    إذا (x <= static_cast<T>(0)) {
        أرجع neg_infinity<T>();
    }
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::لوگارتھم(x);
    } وإلا {
        أرجع sycl::لوگارتھم(x);
    }
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_softplus(T x) {
    ثابت عائم xf = (عائم) x;
    ثابت عائم ax = op_abs(xf);
    ثابت عائم m  = sycl::fmax(xf, 0.0f);
    ثابت عائم y  = m + sycl::log1p(sycl::قوت_نما(-ax));
    أرجع (T) y;
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_neg(T x) {
    أرجع -x;
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_قدم(T x) {
    أرجع (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_leaky_relu(T x, عائم negative_ڈھلوان) {
    T neg_ڈھلوان_T = static_cast<T>(negative_ڈھلوان);
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_ڈھلوان_T;

    } وإلا {
        أرجع sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_ڈھلوان_T;
    }
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_xielu(T x, عائم alpha_n, عائم alpha_p, عائم beta, عائم eps) {
    ثابت عائم xi        = static_cast<عائم>(x);
    ثابت عائم gate_pos  = (xi > 0.0f);
    ثابت عائم y_pos     = alpha_p * xi * xi + beta * xi;
    ثابت عائم min_v_eps = sycl::fmin(xi, eps);
    ثابت عائم y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    ثابت عائم باہرلا       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    أرجع static_cast<T>(باہرلا);
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_sqr(T x) {
    أرجع x * x;
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_clamp(T x, عائم min_val, عائم max_val) {
    أرجع x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_floor(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::floor(x);
    } وإلا {
        أرجع sycl::floor(x);
    }
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_ceil(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::ceil(x);
    } وإلا {
        أرجع sycl::ceil(x);
    }
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_گیڑا(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<عائم>(x))
        );
    } وإلا {
        أرجع sycl::round(x);
    }
}

سانچا<قسم_ناں T>
ساكن __dpct_inline__ T op_trunc(T x) {
    إذا constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        أرجع sycl::ext::oneapi::experimental::trunc(x);
    } وإلا {
        أرجع sycl::trunc(x);
    }
}

سانچا<قسم_ناں T, قسم_ناں F>
ساكن فراغ unary_op_flat_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_شے<1> & شے_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = func(x[i]);
    }
}

سانچا<قسم_ناں T, قسم_ناں F>
ساكن فراغ unary_op_generic_kernel(
        ثابت T * x,
        T * dst,
        ثابت صحيح k,
        ثابت sycl::uint3 ne0_fd, ثابت sycl::uint3 ne1_fd, ثابت sycl::uint3 ne2_fd,
        ثابت size_t nb0,  ثابت size_t nb1,  ثابت size_t nb2,  ثابت size_t nb3,
        ثابت size_t nbd0, ثابت size_t nbd1, ثابت size_t nbd2, ثابت size_t nbd3,
        ثابت sycl::nd_شے<1> & شے_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        ثابت uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        ثابت uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        ثابت uint32_t i2 = dm.y();
        ثابت uint32_t i3 = dm.x();

        ثابت حرف * src_ادھار = (ثابت حرف *) x;
        حرف       * dst_ادھار = (حرف *) dst;

        ثابت T * srcp = (ثابت T *)(src_ادھار + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_ادھار + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

سانچا<قسم_ناں T>
ساكن فراغ unary_op_sqrt_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_شے<1> &شے_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

سانچا<قسم_ناں T>
ساكن فراغ unary_op_sin_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_شے<1> &شے_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

سانچا<قسم_ناں T>
ساكن فراغ unary_op_cos_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_شے<1> &شے_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

سانچا<قسم_ناں T>
ساكن فراغ unary_op_log_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_شے<1> &شے_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = op_log(x[i]);
    }
}


سانچا<قسم_ناں T>
ساكن فراغ unary_op_leaky_relu_kernel(ثابت T * x, T * dst, ثابت صحيح k, عائم negative_ڈھلوان, ثابت sycl::nd_شے<1> &شے_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_ڈھلوان);
    }
}

سانچا<قسم_ناں T>
ساكن فراغ unary_op_xielu_kernel(ثابت T * x, T * dst, ثابت صحيح k, عائم alpha_n, عائم alpha_p, عائم beta, عائم eps, ثابت sycl::nd_شے<1> &شے_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

سانچا<قسم_ناں T>
ساكن فراغ unary_op_sqr_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_شے<1> &شے_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

سانچا<قسم_ناں T>
ساكن فراغ unary_op_clamp_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_شے<1> &شے_ct1, عائم min_val, عائم max_val) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

سانچا<قسم_ناں T>
ساكن فراغ unary_op_ceil_kernel(ثابت T * x, T * dst, ثابت صحيح k, ثابت sycl::nd_شے<1> &شے_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

سانچا<قسم_ناں T>
ساكن فراغ clamp(ثابت T * x, T * dst, ثابت عائم min, ثابت عائم max, ثابت صحيح k,
                      ثابت sycl::nd_شے<1> &شے_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

سانچا<قسم_ناں T, قسم_ناں F>
ساكن فراغ unary_gated_op_flat_kernel(ثابت T * x, ثابت T * g, T * dst, ثابت uint64_t k, ثابت sycl::nd_شے<1> & شے_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = func(x[i]) * g[i];
    }
}

سانچا<قسم_ناں T, قسم_ناں F>
ساكن فراغ unary_gated_op_generic_kernel(
        ثابت T * x,
        ثابت T * g,
        T * dst,
        ثابت uint64_t k,
        ثابت sycl::uint3 n_fd,
        ثابت uint64_t o0,
        ثابت uint64_t o1,
        ثابت sycl::nd_شے<1> & شے_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        ثابت sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ثابت int64_t j0 = rc.x() * o0 + rc.y();
        ثابت int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = func(x[j0]) * g[j1];
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
سانچا<قسم_ناں T, قسم_ناں F>
ساكن فراغ unary_mul_flat_kernel(ثابت T * x, ثابت T * g, T * dst, ثابت int64_t k, ثابت sycl::nd_شے<1> &شے_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = (T) (op((عائم) x[i]) * (عائم) g[i]);
    }
}

سانچا<قسم_ناں T, قسم_ناں F>
ساكن فراغ unary_mul_strided_kernel(ثابت T * x, ثابت T * g, T * dst, ثابت int64_t k, ثابت sycl::uint3 n_fd, ثابت int64_t o0, ثابت int64_t o1, ثابت sycl::nd_شے<1> &شے_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        ثابت sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        ثابت int64_t j0 = rc.x() * o0 + rc.y();
        ثابت int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((عائم) x[j0]) * (عائم) g[j1]);
    }
}

سانچا<قسم_ناں T, قسم_ناں F>
ساكن فراغ unary_mul_sycl(ثابت T * x, ثابت T * g, T * dst, ثابت int64_t k, ثابت int64_t n, ثابت int64_t o0, ثابت int64_t o1, قطار_ptr main_دھارا, F op) {
    ثابت size_t            عدد_بلاکاں = ceil_div((size_t) k, (size_t) SYCL_GLU_بلاک_ناپ);
    ثابت sycl::nd_range<1> range(عدد_بلاکاں * sycl::range<1>(SYCL_GLU_بلاک_ناپ), sycl::range<1>(SYCL_GLU_بلاک_ناپ));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    إذا (o0 == n && o1 == n) {
        main_دھارا->parallel_for(range, [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
            unary_mul_flat_kernel(x, g, dst, k, شے_ct1, op);
        });
        أرجع;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ثابت sycl::uint3 n_fd = init_fastdiv_اقدار((uint32_t) n);
    main_دھارا->parallel_for(range, [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
        unary_mul_strided_kernel(x, g, dst, k, n_fd, o0, o1, شے_ct1, op);
    });
}

// ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the delta-net alpha gate:
// dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]. k == ne0 makes that the flat index.
سانچا<قسم_ناں F>
ساكن فراغ جوڑو_unary_mul_flat_kernel(ثابت عائم * a, ثابت عائم * bias, ثابت عائم * scale, عائم * dst,
                                      ثابت int64_t k, ثابت sycl::nd_شے<1> &شے_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = op(a[i] + bias[i]) * scale[i];
    }
}

سانچا<قسم_ناں F>
ساكن فراغ جوڑو_unary_mul_bcast_kernel(ثابت عائم * a, ثابت عائم * bias, ثابت عائم * scale, عائم * dst,
                                       ثابت int64_t k, ثابت sycl::uint3 ne0_fd, ثابت sycl::nd_شے<1> &شے_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        ثابت uint32_t h = fastmodulo((uint32_t) i, ne0_fd);
        dst[i] = op(a[i] + bias[h]) * scale[h];
    }
}

سانچا<قسم_ناں F>
ساكن فراغ جوڑو_unary_mul_sycl(ثابت عائم * a, ثابت عائم * bias, ثابت عائم * scale, عائم * dst,
                               ثابت int64_t k, ثابت int64_t ne0, قطار_ptr main_دھارا, F op) {
    ثابت size_t            عدد_بلاکاں = ceil_div((size_t) k, (size_t) SYCL_GLU_بلاک_ناپ);
    ثابت sycl::nd_range<1> range(عدد_بلاکاں * sycl::range<1>(SYCL_GLU_بلاک_ناپ), sycl::range<1>(SYCL_GLU_بلاک_ناپ));

    إذا (k == ne0) {
        main_دھارا->parallel_for(range, [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
            جوڑو_unary_mul_flat_kernel(a, bias, scale, dst, k, شے_ct1, op);
        });
        أرجع;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    ثابت sycl::uint3 ne0_fd = init_fastdiv_اقدار((uint32_t) ne0);
    main_دھارا->parallel_for(range, [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
        جوڑو_unary_mul_bcast_kernel(a, bias, scale, dst, k, ne0_fd, شے_ct1, op);
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
                         ثابت int64_t وتھ, قطار_ptr stream) {
    ثابت صحيح عدد_بلاکاں = (n_elements + SYCL_ACC_بلاک_ناپ - 1) / SYCL_ACC_بلاک_ناپ;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکاں) * sycl::range<3>(1, 1, SYCL_ACC_بلاک_ناپ),
                                           sycl::range<3>(1, 1, SYCL_ACC_بلاک_ناپ)),
                         [=](sycl::nd_شے<3> /*item_ct1*/) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                            acc_f32(x, y, dst, n_elements,
                                ne0, ne1, ne2,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                s1, s2, s3, وتھ);
                         });
}

سانچا<قسم_ناں T>
ساكن فراغ arange_kernel(T * dst, ثابت صحيح k, T start, T قدم,
                         ثابت sycl::nd_شے<1> &شے_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, شے_ct1) {
        dst[i] = start + static_cast<T>(i) * قدم;
    }
}

سانچا<قسم_ناں KernelInvoker, قسم_ناں... دلیلاں>
ساكن مضمن فراغ dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, دلیلاں&&... دلیلاں_2) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    بدل (dst->type) {
        حالة GGML_TYPE_F16:
            {
                خودکار انکڑے_pts = cast_انکڑے<sycl::half>(dst);
                kernel_invoker(انکڑے_pts.src, انکڑے_pts.dst, (صحيح)ggml_nelements(dst->src[0]), main_دھارا, std::forward<دلیلاں>(دلیلاں_2)...);
                اكسر;
            }
#جے_تعریف_شدہ GGML_SYCL_HAS_BF16
        حالة GGML_TYPE_BF16:
            {
                خودکار انکڑے_pts = cast_انکڑے<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(انکڑے_pts.src, انکڑے_pts.dst, (صحيح)ggml_nelements(dst->src[0]), main_دھارا, std::forward<دلیلاں>(دلیلاں_2)...);
                اكسر;
            }
#جے_انت
        حالة GGML_TYPE_F32:
            {
                خودکار انکڑے_pts = cast_انکڑے<عائم>(dst);
                kernel_invoker(انکڑے_pts.src, انکڑے_pts.dst, (صحيح)ggml_nelements(dst->src[0]), main_دھارا, std::forward<دلیلاں>(دلیلاں_2)...);
                اكسر;
            }
        افتراضي:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

سانچا<قسم_ناں KernelInvoker, قسم_ناں... دلیلاں>
ساكن مضمن فراغ dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, دلیلاں&&... دلیلاں_2) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
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
    فراغ * src0_d = src0->انکڑے;
    فراغ * src1_d = src1 ? src1->انکڑے : src0->انکڑے;
    ثابت int64_t src0_o = src0->nb[1];
    ثابت int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    فراغ * dst_d = dst->انکڑے;
    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_ناپ(src1));
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
                               main_دھارا,
                               std::forward<دلیلاں>(دلیلاں_2)...);
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
                               main_دھارا,
                               std::forward<دلیلاں>(دلیلاں_2)...);
                اكسر;
            }
        افتراضي:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

سانچا<قسم_ناں F>
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
        [=](ثابت خودکار* src, خودکار* dst_ptr, صحيح k_elements, قطار_ptr stream) {

            ثابت صحيح عدد_بلاکاں = ceil_div(k_elements, 256);

            إذا (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(عدد_بلاکاں) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, شے_ct1, func);
                    });
            } وإلا {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                ثابت sycl::uint3 ne0_fd = init_fastdiv_اقدار((uint32_t) ne0);
                ثابت sycl::uint3 ne1_fd = init_fastdiv_اقدار((uint32_t) ne1);
                ثابت sycl::uint3 ne2_fd = init_fastdiv_اقدار((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(عدد_بلاکاں) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        unary_op_generic_kernel(
                            src, dst_ptr, k_elements,
                            ne0_fd, ne1_fd, ne2_fd,
                            nb0, nb1, nb2, nb3,
                            nbd0, nbd1, nbd2, nbd3,
                            شے_ct1,
                            func
                        );
                    });
            }
        });
}

سانچا<قسم_ناں F>
ساكن مضمن فراغ ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func](ثابت خودکار * x_ptr, ثابت خودکار * g_ptr, خودکار * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, قطار_ptr main_دھارا) {

            ثابت uint32_t عدد_بلاکاں = (uint32_t) ceil_div(k, SYCL_GLU_بلاک_ناپ);
            ثابت sycl::nd_range<1> launch_range(عدد_بلاکاں * sycl::range<1>(SYCL_GLU_بلاک_ناپ),
                                                 sycl::range<1>(SYCL_GLU_بلاک_ناپ));

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            إذا (o0 == n && o1 == n) {
                main_دھارا->parallel_for(launch_range,
                    [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        unary_gated_op_flat_kernel(x_ptr, g_ptr, dst_ptr, k, شے_ct1, func);
                    });
            } وإلا {
                // launch-invariant divisor, and only this path needs it
                ثابت sycl::uint3 n_fd = init_fastdiv_اقدار((uint32_t) n);
                main_دھارا->parallel_for(launch_range,
                    [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                        unary_gated_op_generic_kernel(x_ptr, g_ptr, dst_ptr, k, n_fd, o0, o1, شے_ct1, func);
                    });
            }
        });
}


ساكن مضمن فراغ ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    عائم start, stop, قدم;
    حافظہ_نقل(&start, dst->op_params, حجم(عائم));
    حافظہ_نقل(&stop, (عائم *) dst->op_params + 1, حجم(عائم));
    حافظہ_نقل(&قدم, (عائم *) dst->op_params + 2, حجم(عائم));
    dpct::قطار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    عائم * dst_ptr = (عائم *)dst->انکڑے;
    ثابت صحيح k = (صحيح)ggml_nelements(dst);
    ثابت صحيح عدد_بلاکاں = ceil_div(k, SYCL_ARANGE_بلاک_ناپ);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(عدد_بلاکاں) * sycl::range<1>(SYCL_ARANGE_بلاک_ناپ),
                          sycl::range<1>(SYCL_ARANGE_بلاک_ناپ)),
        [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
            arange_kernel(dst_ptr, k, start, قدم, شے_ct1);
        });
}

} // namespace ggml_sycl_detail



ساكن مضمن فراغ ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_sgn(x);
    });
}


ساكن مضمن فراغ ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_abs(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_elu(x);
    });
}
ساكن مضمن فراغ ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_silu(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_gelu(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_gelu_تیز(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_gelu_تیز(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_gelu_erf(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_tanh(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_relu(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_hardsigmoid(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_hardswish(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_exp(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_expm1(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خودکار* src, خودکار* dst_ptr, صحيح k_elements, قطار_ptr stream) {
            ثابت صحيح عدد_بلاکاں = ceil_div(k_elements, SYCL_EXP_بلاک_ناپ); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکاں) * sycl::range<1>(SYCL_EXP_بلاک_ناپ),
                                  sycl::range<1>(SYCL_EXP_بلاک_ناپ)),
                [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, شے_ct1);
                });
        });
}

ساكن مضمن فراغ ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_softplus(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_neg(x);
    });
}


ساكن مضمن فراغ ggml_sycl_op_قدم(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_قدم(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_sigmoid(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خودکار* src, خودکار* dst_ptr, صحيح k_elements, قطار_ptr stream) {
            ثابت صحيح عدد_بلاکاں = ceil_div(k_elements, SYCL_SQRT_بلاک_ناپ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکاں) * sycl::range<1>(SYCL_SQRT_بلاک_ناپ),
                                  sycl::range<1>(SYCL_SQRT_بلاک_ناپ)),
                [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, شے_ct1);
                });
        });
}

ساكن مضمن فراغ ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خودکار* src, خودکار* dst_ptr, صحيح k_elements, قطار_ptr stream) {
            ثابت صحيح عدد_بلاکاں = ceil_div(k_elements, SYCL_SIN_بلاک_ناپ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکاں) * sycl::range<1>(SYCL_SIN_بلاک_ناپ),
                                  sycl::range<1>(SYCL_SIN_بلاک_ناپ)),
                [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, شے_ct1);
                });
        });
}

ساكن مضمن فراغ ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خودکار* src, خودکار* dst_ptr, صحيح k_elements, قطار_ptr stream) {
            ثابت صحيح عدد_بلاکاں = ceil_div(k_elements, SYCL_SIN_بلاک_ناپ); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکاں) * sycl::range<1>(SYCL_SIN_بلاک_ناپ),
                                  sycl::range<1>(SYCL_SIN_بلاک_ناپ)),
                [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, شے_ct1);
                });
        });
}

ساكن مضمن فراغ ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    عائم negative_ڈھلوان;
    حافظہ_نقل(&negative_ڈھلوان, dst->op_params, حجم(عائم));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خودکار* src, خودکار* dst_ptr, صحيح k_elements, قطار_ptr stream, عائم ڈھلوان) {
            ثابت صحيح عدد_بلاکاں = ceil_div(k_elements, SYCL_RELU_بلاک_ناپ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکاں) * sycl::range<1>(SYCL_RELU_بلاک_ناپ),
                                  sycl::range<1>(SYCL_RELU_بلاک_ناپ)),
                [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, ڈھلوان, شے_ct1);
                });
        }, negative_ڈھلوان);
}

ساكن مضمن فراغ ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خودکار* src, خودکار* dst_ptr, صحيح k_elements, قطار_ptr stream) {
            ثابت صحيح عدد_بلاکاں = ceil_div(k_elements, SYCL_SQR_بلاک_ناپ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکاں) * sycl::range<1>(SYCL_SQR_بلاک_ناپ),
                                  sycl::range<1>(SYCL_SQR_بلاک_ناپ)),
                [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, شے_ct1);
                });
        });
}

ساكن مضمن فراغ ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    عائم min_val;
    عائم max_val;
    حافظہ_نقل(&min_val, dst->op_params, حجم(عائم));
    حافظہ_نقل(&max_val, (عائم *) dst->op_params + 1, حجم(عائم));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خودکار* src, خودکار* dst_ptr, صحيح k_elements, قطار_ptr stream, عائم min_arg, عائم max_arg) {
            ثابت صحيح عدد_بلاکاں = ceil_div(k_elements, SYCL_CLAMP_بلاک_ناپ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکاں) * sycl::range<1>(SYCL_CLAMP_بلاک_ناپ),
                                  sycl::range<1>(SYCL_CLAMP_بلاک_ناپ)),
                [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, شے_ct1);
                });
        }, min_val, max_val);
}

ساكن مضمن فراغ ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت عائم alpha_n = ggml_get_op_params_f32(dst, 1);
    ثابت عائم alpha_p = ggml_get_op_params_f32(dst, 2);
    ثابت عائم beta    = ggml_get_op_params_f32(dst, 3);
    ثابت عائم eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](ثابت خودکار* src, خودکار* dst_ptr, صحيح k_elements, قطار_ptr stream, عائم alpha_n_arg, عائم alpha_p_arg, عائم beta_arg, عائم eps_arg) {
            ثابت صحيح عدد_بلاکاں = ceil_div(k_elements, SYCL_RELU_بلاک_ناپ);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(عدد_بلاکاں) * sycl::range<1>(SYCL_RELU_بلاک_ناپ),
                                  sycl::range<1>(SYCL_RELU_بلاک_ناپ)),
                [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, شے_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

ساكن مضمن فراغ ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_floor(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_ceil(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_گیڑا(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_گیڑا(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](خودکار x) {
        أرجع op_trunc(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    ثابت حرف  * src0_d = (ثابت حرف  *) src0->انکڑے;
    ثابت حرف  * src1_d = (ثابت حرف  *) src1->انکڑے;
    عائم       * dst_d  = (عائم       *)  dst->انکڑے;

    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_ناپ(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ثابت int64_t s1     = (int64_t) ((ثابت int32_t *) dst->op_params)[0] / (int64_t) حجم(عائم);
    ثابت int64_t s2     = (int64_t) ((ثابت int32_t *) dst->op_params)[1] / (int64_t) حجم(عائم);
    ثابت int64_t s3     = (int64_t) ((ثابت int32_t *) dst->op_params)[2] / (int64_t) حجم(عائم);
    ثابت int64_t وتھ = (int64_t) ((ثابت int32_t *) dst->op_params)[3] / (int64_t) حجم(عائم);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, وتھ, stream);
}

ساكن مضمن فراغ ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](خودکار x) {
        أرجع op_gelu(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](خودکار x) {
        أرجع op_relu(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](خودکار x) {
        أرجع op_silu(x);
    });
}

// Hands `launch` the functor for the unary op of a fused unary chain. Anything else
// ggml_sycl_can_fuse() has already declined, so the default is a dispatcher bug.
سانچا<قسم_ناں F>
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
فراغ ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_گرہ, ggml_tensor * mul_گرہ) {
    scope_op_debug_print scope_dbg_print(__func__, mul_گرہ, /*num_src=*/2);

    ثابت ggml_tensor * x = unary_گرہ->src[0];
    ثابت ggml_tensor * g = (mul_گرہ->src[0] == unary_گرہ) ? mul_گرہ->src[1] : mul_گرہ->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_گرہ);
    GGML_ASSERT(x->type == g->type && x->type == mul_گرہ->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_گرہ));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_گرہ));

    قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت int64_t k = ggml_nelements(mul_گرہ);
    ثابت int64_t n = mul_گرہ->ne[0];

    ثابت خودکار dispatch_type = [&](خودکار op) {
        بدل (mul_گرہ->type) {
            حالة GGML_TYPE_F32:
                unary_mul_sycl((ثابت عائم *) x->انکڑے, (ثابت عائم *) g->انکڑے, (عائم *) mul_گرہ->انکڑے,
                               k, n, x->nb[1] / حجم(عائم), g->nb[1] / حجم(عائم), main_دھارا, op);
                اكسر;
            حالة GGML_TYPE_F16:
                unary_mul_sycl((ثابت sycl::half *) x->انکڑے, (ثابت sycl::half *) g->انکڑے, (sycl::half *) mul_گرہ->انکڑے,
                               k, n, x->nb[1] / حجم(sycl::half), g->nb[1] / حجم(sycl::half), main_دھارا, op);
                اكسر;
            افتراضي:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_گرہ->type));
        }
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_گرہ), dispatch_type);
}

// dst = op(a + bias) * scale for an ADD + UNARY + MUL chain whose bias and scale broadcast
// over dim 0. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
فراغ ggml_sycl_op_جوڑو_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * جوڑو_گرہ,
                                      ggml_tensor * unary_گرہ, ggml_tensor * mul_گرہ) {
    // the dst-arity convention the other fusions follow; a and bias live on add_node
    scope_op_debug_print scope_dbg_print(__func__, mul_گرہ, /*num_src=*/2);

    ثابت ggml_tensor * a     = جوڑو_گرہ->src[0];
    ثابت ggml_tensor * bias  = جوڑو_گرہ->src[1];
    ثابت ggml_tensor * scale = (mul_گرہ->src[0] == unary_گرہ) ? mul_گرہ->src[1] : mul_گرہ->src[0];

    // scale is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(scale != unary_گرہ);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && mul_گرہ->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, mul_گرہ));
    // a and dst are indexed flat
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_گرہ));
    // bias and scale are one contiguous ne0-length row each, broadcast over the outer dims
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1 && ggml_nrows(scale) == 1);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(scale));

    قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت خودکار dispatch_op = [&](خودکار op) {
        جوڑو_unary_mul_sycl((ثابت عائم *) a->انکڑے, (ثابت عائم *) bias->انکڑے, (ثابت عائم *) scale->انکڑے,
                           (عائم *) mul_گرہ->انکڑے, ggml_nelements(mul_گرہ), mul_گرہ->ne[0], main_دھارا, op);
    };

    dispatch_fused_unary_op(ggml_get_unary_op(unary_گرہ), dispatch_op);
}

__dpct_inline__ عائم ggml_sycl_op_swiglu_oai_single(عائم x, عائم g, عائم alpha = 1.702f, عائم حد_بندی = 7.0f) {
    x = sycl::fmin(x, حد_بندی);
    g = sycl::fmax(sycl::fmin(g, حد_بندی), -حد_بندی);

    عائم باہرلا_glu = x / (1.0f + sycl::native::قوت_نما(-x * alpha));
    باہرلا_glu = باہرلا_glu * (1.0f + g);
    أرجع باہرلا_glu;
}

سانچا <قسم_ناں T>
ساكن فراغ swiglu_oai_kernel(ثابت T * x, ثابت T * g, T * dst, ثابت int64_t k,
                              ثابت int64_t n, ثابت int64_t o0, ثابت int64_t o1,
                              عائم alpha, عائم حد_بندی, sycl::nd_شے<3> شے_ct1) {
    ثابت int64_t i = int64_t(شے_ct1.get_local_range(2)) * شے_ct1.get_گروہ(2) + شے_ct1.get_local_id(2);

    إذا (i >= k) {
        أرجع;
    }

    ثابت int64_t j0 = (i / n) * o0 + (i % n);
    ثابت int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    عائم xi = x[j0];
    عائم gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, حد_بندی);
}

سانچا <قسم_ناں T>
ساكن فراغ swiglu_oai_sycl(ثابت T *       x,
                            ثابت T *       g,
                            T *             dst,
                            ثابت int64_t   k,
                            ثابت int64_t   n,
                            ثابت int64_t   o0,
                            ثابت int64_t   o1,
                            ثابت عائم     alpha,
                            ثابت عائم     حد_بندی,
                            dpct::قطار_ptr stream) {
    ثابت int64_t عدد_بلاکاں = (k + SYCL_GLU_بلاک_ناپ - 1) / SYCL_GLU_بلاک_ناپ;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکاں) * sycl::range<3>(1, 1, SYCL_GLU_بلاک_ناپ),
                                           sycl::range<3>(1, 1, SYCL_GLU_بلاک_ناپ)),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, حد_بندی, شے_ct1);
                         });
}

ساكن فراغ ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];
    فراغ * src0_d = src0->انکڑے;
    فراغ * src1_d = src1 ? src1->انکڑے : src0->انکڑے;
    ثابت int64_t src0_o = src0->nb[1];
    ثابت int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    فراغ * dst_d = dst->انکڑے;
    ثابت int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::قطار_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_ناپ(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_ناپ(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت عائم alpha = ggml_get_op_params_f32(dst, 2);
    ثابت عائم حد_بندی = ggml_get_op_params_f32(dst, 3);

    عائم * src0_p = (عائم *) src0_d;
    عائم * src1_p = (عائم *) src1_d;

    إذا (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (عائم *)dst_d, ggml_nelements(dst), nc, src0_o / حجم(عائم), src1_o / حجم(عائم), alpha, حد_بندی, stream);
}

سانچا <قسم_ناں T>
ساكن فراغ swiglu_clamp_kernel(ثابت T *        gate,
                                ثابت T *        اتے,
                                T *              dst,
                                ثابت int64_t    k,
                                ثابت int64_t    n,
                                ثابت int64_t    o0,
                                ثابت int64_t    o1,
                                عائم            حد_بندی,
                                sycl::nd_شے<3> شے_ct1) {
    ثابت int64_t i = int64_t(شے_ct1.get_local_range(2)) * شے_ct1.get_گروہ(2) + شے_ct1.get_local_id(2);

    إذا (i >= k) {
        أرجع;
    }

    ثابت int64_t j0 = (i / n) * o0 + (i % n);
    ثابت int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ثابت عائم gate_قدر = sycl::fmin((عائم) gate[j0], حد_بندی);
    ثابت عائم اتے_قدر   = sycl::fmax(sycl::fmin((عائم) اتے[j1], حد_بندی), -حد_بندی);
    dst[i]                 = (T) (gate_قدر / (1.0f + sycl::native::قوت_نما(-gate_قدر)) * اتے_قدر);
}

سانچا <قسم_ناں T>
ساكن فراغ swiglu_clamp_sycl(ثابت T *       gate,
                              ثابت T *       اتے,
                              T *             dst,
                              ثابت int64_t   k,
                              ثابت int64_t   n,
                              ثابت int64_t   o0,
                              ثابت int64_t   o1,
                              عائم           حد_بندی,
                              dpct::قطار_ptr stream) {
    ثابت int64_t عدد_بلاکاں = (k + SYCL_GLU_بلاک_ناپ - 1) / SYCL_GLU_بلاک_ناپ;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکاں) * sycl::range<3>(1, 1, SYCL_GLU_بلاک_ناپ),
                                           sycl::range<3>(1, 1, SYCL_GLU_بلاک_ناپ)),
                         [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             swiglu_clamp_kernel(gate, اتے, dst, k, n, o0, o1, حد_بندی, شے_ct1);
                         });
}

ساكن فراغ ggml_sycl_op_swiglu_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0   = dst->src[0];
    ثابت ggml_tensor * src1   = dst->src[1];
    فراغ *              src0_d = src0->انکڑے;
    فراغ *              src1_d = src1 ? src1->انکڑے : src0->انکڑے;
    ثابت int64_t       src0_o = src0->nb[1];
    ثابت int64_t       src1_o = src1 ? src1->nb[1] : src0->nb[1];
    فراغ *              dst_d  = dst->انکڑے;
    ثابت int64_t       nc     = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::قطار_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_ناپ(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    إذا (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_ناپ(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    ثابت int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ثابت عائم   حد_بندی   = ggml_get_op_params_f32(dst, 3);

    إذا (src0->type == GGML_TYPE_F16) {
        sycl::half * src0_p = (sycl::half *) src0_d;
        sycl::half * src1_p = (sycl::half *) src1_d;

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (sycl::half *) dst_d, ggml_nelements(dst), nc, src0_o / حجم(sycl::half),
                          src1_o / حجم(sycl::half), حد_بندی, stream);
    } وإلا {
        عائم * src0_p = (عائم *) src0_d;
        عائم * src1_p = (عائم *) src1_d;

        إذا (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_sycl(src0_p, src1_p, (عائم *) dst_d, ggml_nelements(dst), nc, src0_o / حجم(عائم),
                          src1_o / حجم(عائم), حد_بندی, stream);
    }
}

ساكن مضمن فراغ ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](خودکار x) {
        أرجع op_gelu_erf(x);
    });
}

ساكن مضمن فراغ ggml_sycl_op_geglu_تیز(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](خودکار x) {
        أرجع op_gelu_تیز(x);
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

فراغ ggml_sycl_gelu_تیز(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_تیز(ctx, dst);
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

فراغ ggml_sycl_قدم(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_قدم(ctx, dst);
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

فراغ ggml_sycl_geglu_تیز(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_تیز(ctx, dst);
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

فراغ ggml_sycl_گیڑا(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_گیڑا(ctx, dst);
}

فراغ ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
