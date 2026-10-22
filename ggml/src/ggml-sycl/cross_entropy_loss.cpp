#ضمن "cross_entropy_loss.hpp"

#ضمن <cstdint>
#ضمن <cmath>

سانچو <منطقي has_shared>
ساكن __dpct_inline__ فراغ cross_entropy_loss_f32_kernel(
        ثابت عائم * __restrict__ logits,
        ثابت عائم * __restrict__ labels,
        عائم * __restrict__ صف_loss,
        ثابت صحيح nclasses,
        ثابت صحيح nrows,
        عائم * __restrict__ smem,
        ثابت sycl::nd_شي<3> & شي) {

    ثابت صحيح صف = شي.get_گروهه(2);
    ثابت صحيح tid = شي.get_local_id(2);

    logits += (int64_t) صف * nclasses;
    labels += (int64_t) صف * nclasses;

    عائم max_logit = -INFINITY;
    لكل (صحيح i = tid; i < nclasses; i += WARP_سائيز) {
        ثابت عائم v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        إذا (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_سائيز>(max_logit);

    عائم جوڙ_exp = 0.0f;
    لكل (صحيح i = tid; i < nclasses; i += WARP_سائيز) {
        ثابت عائم v = has_shared ? smem[i] : logits[i];
        جوڙ_exp += sycl::قوت_نما(v - max_logit);
    }
    جوڙ_exp = warp_reduce_جوڙ<WARP_سائيز>(جوڙ_exp);
    ثابت عائم log_جوڙ = sycl::لاگرٿم(جوڙ_exp);

    عائم loss = 0.0f;
    لكل (صحيح i = tid; i < nclasses; i += WARP_سائيز) {
        ثابت عائم v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_جوڙ) * labels[i];
    }
    loss = -warp_reduce_جوڙ<WARP_سائيز>(loss) / (عائم) nrows;

    إذا (tid == 0) {
        صف_loss[صف] = loss;
    }
}

سانچو <منطقي has_shared>
ساكن __dpct_inline__ فراغ cross_entropy_loss_پٺ_f32_kernel(
        ثابت عائم * __restrict__ grad,
        ثابت عائم * __restrict__ logits,
        ثابت عائم * __restrict__ labels,
        عائم * __restrict__ dst,
        ثابت صحيح nclasses,
        ثابت صحيح nrows,
        عائم * __restrict__ smem,
        ثابت sycl::nd_شي<3> & شي) {

    ثابت صحيح صف = شي.get_گروهه(2);
    ثابت صحيح tid = شي.get_local_id(2);

    logits += (int64_t) صف * nclasses;
    labels += (int64_t) صف * nclasses;
    dst    += (int64_t) صف * nclasses;

    عائم max_logit = -INFINITY;
    لكل (صحيح i = tid; i < nclasses; i += WARP_سائيز) {
        ثابت عائم v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        إذا (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_سائيز>(max_logit);

    عائم جوڙ_exp = 0.0f;
    لكل (صحيح i = tid; i < nclasses; i += WARP_سائيز) {
        ثابت عائم v = sycl::قوت_نما((has_shared ? smem[i] : logits[i]) - max_logit);
        جوڙ_exp += v;
        إذا (has_shared) {
            smem[i] = v;
        } وإلا {
            dst[i] = v;
        }
    }
    جوڙ_exp = warp_reduce_جوڙ<WARP_سائيز>(جوڙ_exp);
    ثابت عائم inv_جوڙ = 1.0f / جوڙ_exp;

    ثابت عائم d_by_nrows = grad[0] / (عائم) nrows;
    لكل (صحيح i = tid; i < nclasses; i += WARP_سائيز) {
        ثابت عائم sm_انگ = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_انگ * inv_جوڙ - labels[i]) * d_by_nrows;
    }
}

ساكن فراغ cross_entropy_reduce_صفون(
        ggml_backend_sycl_context & ctx,
        ثابت عائم * صف_loss,
        عائم * dst,
        ثابت int64_t nrows) {
    إذا (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->يادگيري_نقل(dst, صف_loss, حجم(عائم))));
        أرجع;
    }

    ggml_sycl_pool_alloc<عائم> tmp_alloc(ctx.pool(), nrows);
    عائم * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->يادگيري_نقل(tmp, صف_loss, nrows * حجم(عائم))));

    int64_t cur = nrows;
    طالما (cur > 1) {
        ثابت int64_t ٻاهريون = (cur + WARP_سائيز - 1) / WARP_سائيز;
        ثابت sycl::range<3> بلاڪ(1, 1, WARP_سائيز);
        ثابت sycl::range<3> grid(1, 1, ٻاهريون);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * بلاڪ, بلاڪ),
            [=](sycl::nd_شي<3> شي) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                ثابت صحيح صف = شي.get_گروهه(2);
                ثابت صحيح tid = شي.get_local_id(2);
                ثابت int64_t i = (int64_t) صف * WARP_سائيز + tid;
                عائم v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_جوڙ<WARP_سائيز>(v);
                إذا (tid == 0) {
                    tmp[صف] = v;
                }
            });
        cur = ٻاهريون;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->يادگيري_نقل(dst, tmp, حجم(عائم))));
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

    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));

    ثابت int64_t nclasses = src0->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0);

    ثابت عائم * logits_d = (ثابت عائم *) src0->معلومات;
    ثابت عائم * labels_d = (ثابت عائم *) src1->معلومات;
    عائم * dst_d = (عائم *) dst->معلومات;

    ggml_sycl_pool_alloc<عائم> صف_loss_alloc(ctx.pool(), nrows);
    عائم * صف_loss = صف_loss_alloc.get();

    ثابت sycl::range<3> بلاڪ(1, 1, WARP_سائيز);
    ثابت sycl::range<3> grid(1, 1, nrows);
    ثابت size_t nbytes_shared = (size_t) nclasses * حجم(عائم);
    ثابت size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    إذا (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<عائم, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * بلاڪ, بلاڪ),
                [=](sycl::nd_شي<3> شي) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    cross_entropy_loss_f32_kernel<سچ>(
                        logits_d, labels_d, صف_loss,
                        (صحيح) nclasses, (صحيح) nrows,
                        get_اشارو(smem), شي);
                });
        });
    } وإلا {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * بلاڪ, بلاڪ),
            [=](sycl::nd_شي<3> شي) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                cross_entropy_loss_f32_kernel<خطأ>(
                    logits_d, labels_d, صف_loss,
                    (صحيح) nclasses, (صحيح) nrows,
                    خالي_اشارو, شي);
            });
    }

    cross_entropy_reduce_صفون(ctx, صف_loss, dst_d, nrows);
}

فراغ ggml_sycl_cross_entropy_loss_پٺ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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

    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));

    ثابت int64_t nclasses = src0f->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0f);

    ثابت عائم * grad_d  = (ثابت عائم *) grad->معلومات;
    ثابت عائم * logits_d = (ثابت عائم *) src0f->معلومات;
    ثابت عائم * labels_d = (ثابت عائم *) src1f->معلومات;
    عائم * dst_d = (عائم *) dst->معلومات;

    ثابت sycl::range<3> بلاڪ(1, 1, WARP_سائيز);
    ثابت sycl::range<3> grid(1, 1, nrows);
    ثابت size_t nbytes_shared = (size_t) nclasses * حجم(عائم);
    ثابت size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    إذا (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<عائم, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * بلاڪ, بلاڪ),
                [=](sycl::nd_شي<3> شي) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                    cross_entropy_loss_پٺ_f32_kernel<سچ>(
                        grad_d, logits_d, labels_d, dst_d,
                        (صحيح) nclasses, (صحيح) nrows,
                        get_اشارو(smem), شي);
                });
        });
    } وإلا {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * بلاڪ, بلاڪ),
            [=](sycl::nd_شي<3> شي) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                cross_entropy_loss_پٺ_f32_kernel<خطأ>(
                    grad_d, logits_d, labels_d, dst_d,
                    (صحيح) nclasses, (صحيح) nrows,
                    خالي_اشارو, شي);
            });
    }
}
