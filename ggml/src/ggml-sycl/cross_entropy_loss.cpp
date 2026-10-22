#ಅನ್ತರ್ಭಾವ "cross_entropy_loss.hpp"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cmath>

ಮಾದರಿ <ತಾರ್ಕಿಕ has_shared>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ cross_entropy_loss_f32_kernel(
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ logits,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ labels,
        ದಶಮಾಂಶ * __restrict__ ಅಡ್ಡಸಾಲು_loss,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nclasses,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ದಶಮಾಂಶ * __restrict__ smem,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ.get_local_id(2);

    logits += (int64_t) ಅಡ್ಡಸಾಲು * nclasses;
    labels += (int64_t) ಅಡ್ಡಸಾಲು * nclasses;

    ದಶಮಾಂಶ max_logit = -INFINITY;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = tid; i < nclasses; i += WARP_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        ಆದರೆ (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_ಗಾತ್ರ>(max_logit);

    ದಶಮಾಂಶ ಮೊತ್ತ_exp = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = tid; i < nclasses; i += WARP_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = has_shared ? smem[i] : logits[i];
        ಮೊತ್ತ_exp += sycl::ಘಾತಾಂಕ(v - max_logit);
    }
    ಮೊತ್ತ_exp = warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(ಮೊತ್ತ_exp);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ log_ಮೊತ್ತ = sycl::ಲಘುಗಣಕ(ಮೊತ್ತ_exp);

    ದಶಮಾಂಶ loss = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = tid; i < nclasses; i += WARP_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_ಮೊತ್ತ) * labels[i];
    }
    loss = -warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(loss) / (ದಶಮಾಂಶ) nrows;

    ಆದರೆ (tid == 0) {
        ಅಡ್ಡಸಾಲು_loss[ಅಡ್ಡಸಾಲು] = loss;
    }
}

ಮಾದರಿ <ತಾರ್ಕಿಕ has_shared>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ cross_entropy_loss_ಹಿಂಭಾಗ_f32_kernel(
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ grad,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ logits,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ labels,
        ದಶಮಾಂಶ * __restrict__ dst,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nclasses,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ದಶಮಾಂಶ * __restrict__ smem,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ.get_local_id(2);

    logits += (int64_t) ಅಡ್ಡಸಾಲು * nclasses;
    labels += (int64_t) ಅಡ್ಡಸಾಲು * nclasses;
    dst    += (int64_t) ಅಡ್ಡಸಾಲು * nclasses;

    ದಶಮಾಂಶ max_logit = -INFINITY;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = tid; i < nclasses; i += WARP_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        ಆದರೆ (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_ಗಾತ್ರ>(max_logit);

    ದಶಮಾಂಶ ಮೊತ್ತ_exp = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = tid; i < nclasses; i += WARP_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = sycl::ಘಾತಾಂಕ((has_shared ? smem[i] : logits[i]) - max_logit);
        ಮೊತ್ತ_exp += v;
        ಆದರೆ (has_shared) {
            smem[i] = v;
        } ಇಲ್ಲದಿದ್ದರೆ {
            dst[i] = v;
        }
    }
    ಮೊತ್ತ_exp = warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(ಮೊತ್ತ_exp);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ಮೊತ್ತ = 1.0f / ಮೊತ್ತ_exp;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d_by_nrows = grad[0] / (ದಶಮಾಂಶ) nrows;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = tid; i < nclasses; i += WARP_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sm_ಅಂಕೆ = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_ಅಂಕೆ * inv_ಮೊತ್ತ - labels[i]) * d_by_nrows;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ cross_entropy_reduce_ಅಡ್ಡಸಾಲುಗಳು(
        ggml_backend_sycl_context & ctx,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಅಡ್ಡಸಾಲು_loss,
        ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ int64_t nrows) {
    ಆದರೆ (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->ಸ್ಮರಣೆ_ನಕಲು(dst, ಅಡ್ಡಸಾಲು_loss, ಗಾತ್ರ(ದಶಮಾಂಶ))));
        ಹಿಂತಿರುಗಿ;
    }

    ggml_sycl_pool_alloc<ದಶಮಾಂಶ> tmp_alloc(ctx.pool(), nrows);
    ದಶಮಾಂಶ * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->ಸ್ಮರಣೆ_ನಕಲು(tmp, ಅಡ್ಡಸಾಲು_loss, nrows * ಗಾತ್ರ(ದಶಮಾಂಶ))));

    int64_t cur = nrows;
    ಆಗ (cur > 1) {
        ಸ್ಥಿರಾಂಕ int64_t ಹೊರಹೋಗು = (cur + WARP_ಗಾತ್ರ - 1) / WARP_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ(1, 1, WARP_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ sycl::range<3> grid(1, 1, ಹೊರಹೋಗು);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * ಖಂಡ, ಖಂಡ),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ.get_ಗುಂಪು(2);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ.get_local_id(2);
                ಸ್ಥಿರಾಂಕ int64_t i = (int64_t) ಅಡ್ಡಸಾಲು * WARP_ಗಾತ್ರ + tid;
                ದಶಮಾಂಶ v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(v);
                ಆದರೆ (tid == 0) {
                    tmp[ಅಡ್ಡಸಾಲು] = v;
                }
            });
        cur = ಹೊರಹೋಗು;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->ಸ್ಮರಣೆ_ನಕಲು(dst, tmp, ಗಾತ್ರ(ದಶಮಾಂಶ))));
}

ಶೂನ್ಯ ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ int64_t nclasses = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nrows = ggml_nrows(src0);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * logits_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * labels_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_d = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ggml_sycl_pool_alloc<ದಶಮಾಂಶ> ಅಡ್ಡಸಾಲು_loss_alloc(ctx.pool(), nrows);
    ದಶಮಾಂಶ * ಅಡ್ಡಸಾಲು_loss = ಅಡ್ಡಸಾಲು_loss_alloc.get();

    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ(1, 1, WARP_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<3> grid(1, 1, nrows);
    ಸ್ಥಿರಾಂಕ size_t nbytes_shared = (size_t) nclasses * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    ಆದರೆ (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * ಖಂಡ, ಖಂಡ),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    cross_entropy_loss_f32_kernel<ನಿಜ>(
                        logits_d, labels_d, ಅಡ್ಡಸಾಲು_loss,
                        (ಪೂರ್ಣಾಂಕ) nclasses, (ಪೂರ್ಣಾಂಕ) nrows,
                        get_ಸೂಚಕ(smem), ಅಂಶ);
                });
        });
    } ಇಲ್ಲದಿದ್ದರೆ {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * ಖಂಡ, ಖಂಡ),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                cross_entropy_loss_f32_kernel<ಸುಳ್ಳು>(
                    logits_d, labels_d, ಅಡ್ಡಸಾಲು_loss,
                    (ಪೂರ್ಣಾಂಕ) nclasses, (ಪೂರ್ಣಾಂಕ) nrows,
                    ಶೂನ್ಯಸೂಚಕ, ಅಂಶ);
            });
    }

    cross_entropy_reduce_ಅಡ್ಡಸಾಲುಗಳು(ctx, ಅಡ್ಡಸಾಲು_loss, dst_d, nrows);
}

ಶೂನ್ಯ ggml_sycl_cross_entropy_loss_ಹಿಂಭಾಗ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ಸ್ಥಿರಾಂಕ ggml_tensor * grad  = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0f = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1f = dst->src[2];

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

    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ int64_t nclasses = src0f->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nrows = ggml_nrows(src0f);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * grad_d  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) grad->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * logits_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0f->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * labels_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1f->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_d = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ(1, 1, WARP_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<3> grid(1, 1, nrows);
    ಸ್ಥಿರಾಂಕ size_t nbytes_shared = (size_t) nclasses * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    ಆದರೆ (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * ಖಂಡ, ಖಂಡ),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                    cross_entropy_loss_ಹಿಂಭಾಗ_f32_kernel<ನಿಜ>(
                        grad_d, logits_d, labels_d, dst_d,
                        (ಪೂರ್ಣಾಂಕ) nclasses, (ಪೂರ್ಣಾಂಕ) nrows,
                        get_ಸೂಚಕ(smem), ಅಂಶ);
                });
        });
    } ಇಲ್ಲದಿದ್ದರೆ {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * ಖಂಡ, ಖಂಡ),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                cross_entropy_loss_ಹಿಂಭಾಗ_f32_kernel<ಸುಳ್ಳು>(
                    grad_d, logits_d, labels_d, dst_d,
                    (ಪೂರ್ಣಾಂಕ) nclasses, (ಪೂರ್ಣಾಂಕ) nrows,
                    ಶೂನ್ಯಸೂಚಕ, ಅಂಶ);
            });
    }
}
