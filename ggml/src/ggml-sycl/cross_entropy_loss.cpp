#అన్తర్భావ "cross_entropy_loss.hpp"

#అన్తర్భావ <cstdint>
#అన్తర్భావ <cmath>

మూస <తార్కిక has_shared>
స్థిర __dpct_inline__ శూన్య cross_entropy_loss_f32_kernel(
        స్థిరరాశి దశాంశం * __restrict__ logits,
        స్థిరరాశి దశాంశం * __restrict__ labels,
        దశాంశం * __restrict__ అడ్డువరుస_loss,
        స్థిరరాశి పూర్ణాంకం nclasses,
        స్థిరరాశి పూర్ణాంకం nrows,
        దశాంశం * __restrict__ smem,
        స్థిరరాశి sycl::nd_అంశం<3> & అంశం) {

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం.get_సమూహం(2);
    స్థిరరాశి పూర్ణాంకం tid = అంశం.get_local_id(2);

    logits += (int64_t) అడ్డువరుస * nclasses;
    labels += (int64_t) అడ్డువరుస * nclasses;

    దశాంశం max_logit = -INFINITY;
    కొరకు (పూర్ణాంకం i = tid; i < nclasses; i += WARP_కొలత) {
        స్థిరరాశి దశాంశం v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        ఒకవేళ (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_కొలత>(max_logit);

    దశాంశం కూడిక_exp = 0.0f;
    కొరకు (పూర్ణాంకం i = tid; i < nclasses; i += WARP_కొలత) {
        స్థిరరాశి దశాంశం v = has_shared ? smem[i] : logits[i];
        కూడిక_exp += sycl::ఘాతాంకం(v - max_logit);
    }
    కూడిక_exp = warp_reduce_కూడిక<WARP_కొలత>(కూడిక_exp);
    స్థిరరాశి దశాంశం log_కూడిక = sycl::సంవర్గమానం(కూడిక_exp);

    దశాంశం loss = 0.0f;
    కొరకు (పూర్ణాంకం i = tid; i < nclasses; i += WARP_కొలత) {
        స్థిరరాశి దశాంశం v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_కూడిక) * labels[i];
    }
    loss = -warp_reduce_కూడిక<WARP_కొలత>(loss) / (దశాంశం) nrows;

    ఒకవేళ (tid == 0) {
        అడ్డువరుస_loss[అడ్డువరుస] = loss;
    }
}

మూస <తార్కిక has_shared>
స్థిర __dpct_inline__ శూన్య cross_entropy_loss_వెనుక_f32_kernel(
        స్థిరరాశి దశాంశం * __restrict__ grad,
        స్థిరరాశి దశాంశం * __restrict__ logits,
        స్థిరరాశి దశాంశం * __restrict__ labels,
        దశాంశం * __restrict__ dst,
        స్థిరరాశి పూర్ణాంకం nclasses,
        స్థిరరాశి పూర్ణాంకం nrows,
        దశాంశం * __restrict__ smem,
        స్థిరరాశి sycl::nd_అంశం<3> & అంశం) {

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం.get_సమూహం(2);
    స్థిరరాశి పూర్ణాంకం tid = అంశం.get_local_id(2);

    logits += (int64_t) అడ్డువరుస * nclasses;
    labels += (int64_t) అడ్డువరుస * nclasses;
    dst    += (int64_t) అడ్డువరుస * nclasses;

    దశాంశం max_logit = -INFINITY;
    కొరకు (పూర్ణాంకం i = tid; i < nclasses; i += WARP_కొలత) {
        స్థిరరాశి దశాంశం v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        ఒకవేళ (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_కొలత>(max_logit);

    దశాంశం కూడిక_exp = 0.0f;
    కొరకు (పూర్ణాంకం i = tid; i < nclasses; i += WARP_కొలత) {
        స్థిరరాశి దశాంశం v = sycl::ఘాతాంకం((has_shared ? smem[i] : logits[i]) - max_logit);
        కూడిక_exp += v;
        ఒకవేళ (has_shared) {
            smem[i] = v;
        } లేకపోతే {
            dst[i] = v;
        }
    }
    కూడిక_exp = warp_reduce_కూడిక<WARP_కొలత>(కూడిక_exp);
    స్థిరరాశి దశాంశం inv_కూడిక = 1.0f / కూడిక_exp;

    స్థిరరాశి దశాంశం d_by_nrows = grad[0] / (దశాంశం) nrows;
    కొరకు (పూర్ణాంకం i = tid; i < nclasses; i += WARP_కొలత) {
        స్థిరరాశి దశాంశం sm_అంకె = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_అంకె * inv_కూడిక - labels[i]) * d_by_nrows;
    }
}

స్థిర శూన్య cross_entropy_reduce_అడ్డువరుసలు(
        ggml_backend_sycl_context & ctx,
        స్థిరరాశి దశాంశం * అడ్డువరుస_loss,
        దశాంశం * dst,
        స్థిరరాశి int64_t nrows) {
    ఒకవేళ (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->జ్ఞాపకం_నకలు(dst, అడ్డువరుస_loss, పరిమాణం(దశాంశం))));
        తిరిగి;
    }

    ggml_sycl_pool_alloc<దశాంశం> tmp_alloc(ctx.pool(), nrows);
    దశాంశం * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->జ్ఞాపకం_నకలు(tmp, అడ్డువరుస_loss, nrows * పరిమాణం(దశాంశం))));

    int64_t cur = nrows;
    అయితే (cur > 1) {
        స్థిరరాశి int64_t బహిర్గతం = (cur + WARP_కొలత - 1) / WARP_కొలత;
        స్థిరరాశి sycl::range<3> ఖండం(1, 1, WARP_కొలత);
        స్థిరరాశి sycl::range<3> grid(1, 1, బహిర్గతం);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * ఖండం, ఖండం),
            [=](sycl::nd_అంశం<3> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం.get_సమూహం(2);
                స్థిరరాశి పూర్ణాంకం tid = అంశం.get_local_id(2);
                స్థిరరాశి int64_t i = (int64_t) అడ్డువరుస * WARP_కొలత + tid;
                దశాంశం v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_కూడిక<WARP_కొలత>(v);
                ఒకవేళ (tid == 0) {
                    tmp[అడ్డువరుస] = v;
                }
            });
        cur = బహిర్గతం;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->జ్ఞాపకం_నకలు(dst, tmp, పరిమాణం(దశాంశం))));
}

శూన్య ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి int64_t nclasses = src0->ne[0];
    స్థిరరాశి int64_t nrows = ggml_nrows(src0);

    స్థిరరాశి దశాంశం * logits_d = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;
    స్థిరరాశి దశాంశం * labels_d = (స్థిరరాశి దశాంశం *) src1->దత్తాంశం;
    దశాంశం * dst_d = (దశాంశం *) dst->దత్తాంశం;

    ggml_sycl_pool_alloc<దశాంశం> అడ్డువరుస_loss_alloc(ctx.pool(), nrows);
    దశాంశం * అడ్డువరుస_loss = అడ్డువరుస_loss_alloc.get();

    స్థిరరాశి sycl::range<3> ఖండం(1, 1, WARP_కొలత);
    స్థిరరాశి sycl::range<3> grid(1, 1, nrows);
    స్థిరరాశి size_t nbytes_shared = (size_t) nclasses * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    ఒకవేళ (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<దశాంశం, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * ఖండం, ఖండం),
                [=](sycl::nd_అంశం<3> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    cross_entropy_loss_f32_kernel<నిజం>(
                        logits_d, labels_d, అడ్డువరుస_loss,
                        (పూర్ణాంకం) nclasses, (పూర్ణాంకం) nrows,
                        get_సూచిక(smem), అంశం);
                });
        });
    } లేకపోతే {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * ఖండం, ఖండం),
            [=](sycl::nd_అంశం<3> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                cross_entropy_loss_f32_kernel<అబద్ధం>(
                    logits_d, labels_d, అడ్డువరుస_loss,
                    (పూర్ణాంకం) nclasses, (పూర్ణాంకం) nrows,
                    శూన్యసూచిక, అంశం);
            });
    }

    cross_entropy_reduce_అడ్డువరుసలు(ctx, అడ్డువరుస_loss, dst_d, nrows);
}

శూన్య ggml_sycl_cross_entropy_loss_వెనుక(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    స్థిరరాశి ggml_tensor * grad  = dst->src[0];
    స్థిరరాశి ggml_tensor * src0f = dst->src[1];
    స్థిరరాశి ggml_tensor * src1f = dst->src[2];

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

    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి int64_t nclasses = src0f->ne[0];
    స్థిరరాశి int64_t nrows = ggml_nrows(src0f);

    స్థిరరాశి దశాంశం * grad_d  = (స్థిరరాశి దశాంశం *) grad->దత్తాంశం;
    స్థిరరాశి దశాంశం * logits_d = (స్థిరరాశి దశాంశం *) src0f->దత్తాంశం;
    స్థిరరాశి దశాంశం * labels_d = (స్థిరరాశి దశాంశం *) src1f->దత్తాంశం;
    దశాంశం * dst_d = (దశాంశం *) dst->దత్తాంశం;

    స్థిరరాశి sycl::range<3> ఖండం(1, 1, WARP_కొలత);
    స్థిరరాశి sycl::range<3> grid(1, 1, nrows);
    స్థిరరాశి size_t nbytes_shared = (size_t) nclasses * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    ఒకవేళ (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<దశాంశం, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * ఖండం, ఖండం),
                [=](sycl::nd_అంశం<3> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    cross_entropy_loss_వెనుక_f32_kernel<నిజం>(
                        grad_d, logits_d, labels_d, dst_d,
                        (పూర్ణాంకం) nclasses, (పూర్ణాంకం) nrows,
                        get_సూచిక(smem), అంశం);
                });
        });
    } లేకపోతే {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * ఖండం, ఖండం),
            [=](sycl::nd_అంశం<3> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                cross_entropy_loss_వెనుక_f32_kernel<అబద్ధం>(
                    grad_d, logits_d, labels_d, dst_d,
                    (పూర్ణాంకం) nclasses, (పూర్ణాంకం) nrows,
                    శూన్యసూచిక, అంశం);
            });
    }
}
