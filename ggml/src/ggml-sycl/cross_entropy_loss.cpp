#ضمن "cross_entropy_loss.hpp"

#ضمن <cstdint>
#ضمن <cmath>

قالب <منطقي has_shared>
ساكن __dpct_inline__ فراغ cross_entropy_loss_f32_kernel(
        ثابت عائم * __restrict__ logits,
        ثابت عائم * __restrict__ labels,
        عائم * __restrict__ ردیف_loss,
        ثابت صحيح nclasses,
        ثابت صحيح nrows,
        عائم * __restrict__ smem,
        ثابت sycl::nd_قلم<3> & قلم) {

    ثابت صحيح ردیف = قلم.get_گروه(2);
    ثابت صحيح tid = قلم.get_local_id(2);

    logits += (int64_t) ردیف * nclasses;
    labels += (int64_t) ردیف * nclasses;

    عائم max_logit = -INFINITY;
    لكل (صحيح i = tid; i < nclasses; i += WARP_اندازه) {
        ثابت عائم v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        إذا (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_اندازه>(max_logit);

    عائم جمع_exp = 0.0f;
    لكل (صحيح i = tid; i < nclasses; i += WARP_اندازه) {
        ثابت عائم v = has_shared ? smem[i] : logits[i];
        جمع_exp += sycl::نمایی_تابع(v - max_logit);
    }
    جمع_exp = warp_reduce_جمع<WARP_اندازه>(جمع_exp);
    ثابت عائم log_جمع = sycl::لگاریتم(جمع_exp);

    عائم loss = 0.0f;
    لكل (صحيح i = tid; i < nclasses; i += WARP_اندازه) {
        ثابت عائم v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_جمع) * labels[i];
    }
    loss = -warp_reduce_جمع<WARP_اندازه>(loss) / (عائم) nrows;

    إذا (tid == 0) {
        ردیف_loss[ردیف] = loss;
    }
}

قالب <منطقي has_shared>
ساكن __dpct_inline__ فراغ cross_entropy_loss_پشت_f32_kernel(
        ثابت عائم * __restrict__ grad,
        ثابت عائم * __restrict__ logits,
        ثابت عائم * __restrict__ labels,
        عائم * __restrict__ dst,
        ثابت صحيح nclasses,
        ثابت صحيح nrows,
        عائم * __restrict__ smem,
        ثابت sycl::nd_قلم<3> & قلم) {

    ثابت صحيح ردیف = قلم.get_گروه(2);
    ثابت صحيح tid = قلم.get_local_id(2);

    logits += (int64_t) ردیف * nclasses;
    labels += (int64_t) ردیف * nclasses;
    dst    += (int64_t) ردیف * nclasses;

    عائم max_logit = -INFINITY;
    لكل (صحيح i = tid; i < nclasses; i += WARP_اندازه) {
        ثابت عائم v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        إذا (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_اندازه>(max_logit);

    عائم جمع_exp = 0.0f;
    لكل (صحيح i = tid; i < nclasses; i += WARP_اندازه) {
        ثابت عائم v = sycl::نمایی_تابع((has_shared ? smem[i] : logits[i]) - max_logit);
        جمع_exp += v;
        إذا (has_shared) {
            smem[i] = v;
        } وإلا {
            dst[i] = v;
        }
    }
    جمع_exp = warp_reduce_جمع<WARP_اندازه>(جمع_exp);
    ثابت عائم inv_جمع = 1.0f / جمع_exp;

    ثابت عائم d_by_nrows = grad[0] / (عائم) nrows;
    لكل (صحيح i = tid; i < nclasses; i += WARP_اندازه) {
        ثابت عائم sm_رقم = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_رقم * inv_جمع - labels[i]) * d_by_nrows;
    }
}

ساكن فراغ cross_entropy_reduce_ردیفها(
        ggml_backend_sycl_context & ctx,
        ثابت عائم * ردیف_loss,
        عائم * dst,
        ثابت int64_t nrows) {
    إذا (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->کپی_حافظه(dst, ردیف_loss, حجم(عائم))));
        أرجع;
    }

    ggml_sycl_pool_alloc<عائم> tmp_alloc(ctx.pool(), nrows);
    عائم * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->کپی_حافظه(tmp, ردیف_loss, nrows * حجم(عائم))));

    int64_t cur = nrows;
    طالما (cur > 1) {
        ثابت int64_t بیرون = (cur + WARP_اندازه - 1) / WARP_اندازه;
        ثابت sycl::range<3> بلوک(1, 1, WARP_اندازه);
        ثابت sycl::range<3> grid(1, 1, بیرون);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * بلوک, بلوک),
            [=](sycl::nd_قلم<3> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                ثابت صحيح ردیف = قلم.get_گروه(2);
                ثابت صحيح tid = قلم.get_local_id(2);
                ثابت int64_t i = (int64_t) ردیف * WARP_اندازه + tid;
                عائم v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_جمع<WARP_اندازه>(v);
                إذا (tid == 0) {
                    tmp[ردیف] = v;
                }
            });
        cur = بیرون;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->کپی_حافظه(dst, tmp, حجم(عائم))));
}

فراغ ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    ثابت int64_t nclasses = src0->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0);

    ثابت عائم * logits_d = (ثابت عائم *) src0->داده;
    ثابت عائم * labels_d = (ثابت عائم *) src1->داده;
    عائم * dst_d = (عائم *) dst->داده;

    ggml_sycl_pool_alloc<عائم> ردیف_loss_alloc(ctx.pool(), nrows);
    عائم * ردیف_loss = ردیف_loss_alloc.get();

    ثابت sycl::range<3> بلوک(1, 1, WARP_اندازه);
    ثابت sycl::range<3> grid(1, 1, nrows);
    ثابت size_t nbytes_shared = (size_t) nclasses * حجم(عائم);
    ثابت size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    إذا (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<عائم, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * بلوک, بلوک),
                [=](sycl::nd_قلم<3> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    cross_entropy_loss_f32_kernel<درست>(
                        logits_d, labels_d, ردیف_loss,
                        (صحيح) nclasses, (صحيح) nrows,
                        get_اشاره_گر(smem), قلم);
                });
        });
    } وإلا {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * بلوک, بلوک),
            [=](sycl::nd_قلم<3> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                cross_entropy_loss_f32_kernel<خطأ>(
                    logits_d, labels_d, ردیف_loss,
                    (صحيح) nclasses, (صحيح) nrows,
                    اشاره_تهی, قلم);
            });
    }

    cross_entropy_reduce_ردیفها(ctx, ردیف_loss, dst_d, nrows);
}

فراغ ggml_sycl_cross_entropy_loss_پشت(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ثابت ggml_tensor * grad  = dst->src[0];
    ثابت ggml_tensor * src0f = dst->src[1];
    ثابت ggml_tensor * src1f = dst->src[2];

    GGML_ASSERT(grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(src1f->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f));
    GGML_ASSERT(ggml_are_same_shape(src0f, dst));

    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    ثابت int64_t nclasses = src0f->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0f);

    ثابت عائم * grad_d  = (ثابت عائم *) grad->داده;
    ثابت عائم * logits_d = (ثابت عائم *) src0f->داده;
    ثابت عائم * labels_d = (ثابت عائم *) src1f->داده;
    عائم * dst_d = (عائم *) dst->داده;

    ثابت sycl::range<3> بلوک(1, 1, WARP_اندازه);
    ثابت sycl::range<3> grid(1, 1, nrows);
    ثابت size_t nbytes_shared = (size_t) nclasses * حجم(عائم);
    ثابت size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    إذا (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<عائم, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * بلوک, بلوک),
                [=](sycl::nd_قلم<3> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                    cross_entropy_loss_پشت_f32_kernel<درست>(
                        grad_d, logits_d, labels_d, dst_d,
                        (صحيح) nclasses, (صحيح) nrows,
                        get_اشاره_گر(smem), قلم);
                });
        });
    } وإلا {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * بلوک, بلوک),
            [=](sycl::nd_قلم<3> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                cross_entropy_loss_پشت_f32_kernel<خطأ>(
                    grad_d, logits_d, labels_d, dst_d,
                    (صحيح) nclasses, (صحيح) nrows,
                    اشاره_تهی, قلم);
            });
    }
}
