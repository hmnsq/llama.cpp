#شامل "cross_entropy_loss.hpp"

#شامل <cstdint>
#شامل <cmath>

سانچہ <منطقی has_shared>
ساکن __dpct_inline__ خالی cross_entropy_loss_f32_kernel(
        ثابت اعشاری * __restrict__ logits,
        ثابت اعشاری * __restrict__ labels,
        اعشاری * __restrict__ صف_loss,
        ثابت صحیح nclasses,
        ثابت صحیح nrows,
        اعشاری * __restrict__ smem,
        ثابت sycl::nd_شے<3> & شے) {

    ثابت صحیح صف = شے.get_گروہ(2);
    ثابت صحیح tid = شے.get_local_id(2);

    logits += (int64_t) صف * nclasses;
    labels += (int64_t) صف * nclasses;

    اعشاری max_logit = -INFINITY;
    برائے (صحیح i = tid; i < nclasses; i += WARP_سائز) {
        ثابت اعشاری v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        اگر (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_سائز>(max_logit);

    اعشاری مجموع_exp = 0.0f;
    برائے (صحیح i = tid; i < nclasses; i += WARP_سائز) {
        ثابت اعشاری v = has_shared ? smem[i] : logits[i];
        مجموع_exp += sycl::قوت_نما(v - max_logit);
    }
    مجموع_exp = warp_reduce_مجموع<WARP_سائز>(مجموع_exp);
    ثابت اعشاری log_مجموع = sycl::لوگارتھم(مجموع_exp);

    اعشاری loss = 0.0f;
    برائے (صحیح i = tid; i < nclasses; i += WARP_سائز) {
        ثابت اعشاری v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_مجموع) * labels[i];
    }
    loss = -warp_reduce_مجموع<WARP_سائز>(loss) / (اعشاری) nrows;

    اگر (tid == 0) {
        صف_loss[صف] = loss;
    }
}

سانچہ <منطقی has_shared>
ساکن __dpct_inline__ خالی cross_entropy_loss_عقب_f32_kernel(
        ثابت اعشاری * __restrict__ grad,
        ثابت اعشاری * __restrict__ logits,
        ثابت اعشاری * __restrict__ labels,
        اعشاری * __restrict__ dst,
        ثابت صحیح nclasses,
        ثابت صحیح nrows,
        اعشاری * __restrict__ smem,
        ثابت sycl::nd_شے<3> & شے) {

    ثابت صحیح صف = شے.get_گروہ(2);
    ثابت صحیح tid = شے.get_local_id(2);

    logits += (int64_t) صف * nclasses;
    labels += (int64_t) صف * nclasses;
    dst    += (int64_t) صف * nclasses;

    اعشاری max_logit = -INFINITY;
    برائے (صحیح i = tid; i < nclasses; i += WARP_سائز) {
        ثابت اعشاری v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        اگر (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_سائز>(max_logit);

    اعشاری مجموع_exp = 0.0f;
    برائے (صحیح i = tid; i < nclasses; i += WARP_سائز) {
        ثابت اعشاری v = sycl::قوت_نما((has_shared ? smem[i] : logits[i]) - max_logit);
        مجموع_exp += v;
        اگر (has_shared) {
            smem[i] = v;
        } ورنہ {
            dst[i] = v;
        }
    }
    مجموع_exp = warp_reduce_مجموع<WARP_سائز>(مجموع_exp);
    ثابت اعشاری inv_مجموع = 1.0f / مجموع_exp;

    ثابت اعشاری d_by_nrows = grad[0] / (اعشاری) nrows;
    برائے (صحیح i = tid; i < nclasses; i += WARP_سائز) {
        ثابت اعشاری sm_عدد = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_عدد * inv_مجموع - labels[i]) * d_by_nrows;
    }
}

ساکن خالی cross_entropy_reduce_صفیں(
        ggml_backend_sycl_context & ctx,
        ثابت اعشاری * صف_loss,
        اعشاری * dst,
        ثابت int64_t nrows) {
    اگر (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->حافظہ_نقل(dst, صف_loss, حجم(اعشاری))));
        واپس;
    }

    ggml_sycl_pool_alloc<اعشاری> tmp_alloc(ctx.pool(), nrows);
    اعشاری * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->حافظہ_نقل(tmp, صف_loss, nrows * حجم(اعشاری))));

    int64_t cur = nrows;
    جب (cur > 1) {
        ثابت int64_t برآمد = (cur + WARP_سائز - 1) / WARP_سائز;
        ثابت sycl::range<3> بلاک(1, 1, WARP_سائز);
        ثابت sycl::range<3> grid(1, 1, برآمد);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * بلاک, بلاک),
            [=](sycl::nd_شے<3> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                ثابت صحیح صف = شے.get_گروہ(2);
                ثابت صحیح tid = شے.get_local_id(2);
                ثابت int64_t i = (int64_t) صف * WARP_سائز + tid;
                اعشاری v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_مجموع<WARP_سائز>(v);
                اگر (tid == 0) {
                    tmp[صف] = v;
                }
            });
        cur = برآمد;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->حافظہ_نقل(dst, tmp, حجم(اعشاری))));
}

خالی ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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

    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت int64_t nclasses = src0->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0);

    ثابت اعشاری * logits_d = (ثابت اعشاری *) src0->ڈیٹا;
    ثابت اعشاری * labels_d = (ثابت اعشاری *) src1->ڈیٹا;
    اعشاری * dst_d = (اعشاری *) dst->ڈیٹا;

    ggml_sycl_pool_alloc<اعشاری> صف_loss_alloc(ctx.pool(), nrows);
    اعشاری * صف_loss = صف_loss_alloc.get();

    ثابت sycl::range<3> بلاک(1, 1, WARP_سائز);
    ثابت sycl::range<3> grid(1, 1, nrows);
    ثابت size_t nbytes_shared = (size_t) nclasses * حجم(اعشاری);
    ثابت size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    اگر (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<اعشاری, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * بلاک, بلاک),
                [=](sycl::nd_شے<3> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    cross_entropy_loss_f32_kernel<سچ>(
                        logits_d, labels_d, صف_loss,
                        (صحیح) nclasses, (صحیح) nrows,
                        get_اشارہ_گر(smem), شے);
                });
        });
    } ورنہ {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * بلاک, بلاک),
            [=](sycl::nd_شے<3> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                cross_entropy_loss_f32_kernel<جھوٹ>(
                    logits_d, labels_d, صف_loss,
                    (صحیح) nclasses, (صحیح) nrows,
                    خالی_اشاریہ, شے);
            });
    }

    cross_entropy_reduce_صفیں(ctx, صف_loss, dst_d, nrows);
}

خالی ggml_sycl_cross_entropy_loss_عقب(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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

    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت int64_t nclasses = src0f->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0f);

    ثابت اعشاری * grad_d  = (ثابت اعشاری *) grad->ڈیٹا;
    ثابت اعشاری * logits_d = (ثابت اعشاری *) src0f->ڈیٹا;
    ثابت اعشاری * labels_d = (ثابت اعشاری *) src1f->ڈیٹا;
    اعشاری * dst_d = (اعشاری *) dst->ڈیٹا;

    ثابت sycl::range<3> بلاک(1, 1, WARP_سائز);
    ثابت sycl::range<3> grid(1, 1, nrows);
    ثابت size_t nbytes_shared = (size_t) nclasses * حجم(اعشاری);
    ثابت size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    اگر (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<اعشاری, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * بلاک, بلاک),
                [=](sycl::nd_شے<3> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                    cross_entropy_loss_عقب_f32_kernel<سچ>(
                        grad_d, logits_d, labels_d, dst_d,
                        (صحیح) nclasses, (صحیح) nrows,
                        get_اشارہ_گر(smem), شے);
                });
        });
    } ورنہ {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * بلاک, بلاک),
            [=](sycl::nd_شے<3> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                cross_entropy_loss_عقب_f32_kernel<جھوٹ>(
                    grad_d, logits_d, labels_d, dst_d,
                    (صحیح) nclasses, (صحیح) nrows,
                    خالی_اشاریہ, شے);
            });
    }
}
