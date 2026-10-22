#ਅਨ੍ਤਰ੍ਭਾਵ "cross_entropy_loss.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>

ਸਾਂਚਾ <ਤਾਰਕਿਕ has_shared>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ cross_entropy_loss_f32_kernel(
        ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ logits,
        ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ labels,
        ਦਸ਼ਮਲਵ * __restrict__ ਪੰਕਤੀ_loss,
        ਅਚਲ ਪੂਰਨ nclasses,
        ਅਚਲ ਪੂਰਨ nrows,
        ਦਸ਼ਮਲਵ * __restrict__ smem,
        ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ) {

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ.get_ਸਮੂਹ(2);
    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ.get_local_id(2);

    logits += (int64_t) ਪੰਕਤੀ * nclasses;
    labels += (int64_t) ਪੰਕਤੀ * nclasses;

    ਦਸ਼ਮਲਵ max_logit = -INFINITY;
    ਲਈ (ਪੂਰਨ i = tid; i < nclasses; i += WARP_ਮਾਪ) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        ਜੇ (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_ਮਾਪ>(max_logit);

    ਦਸ਼ਮਲਵ ਜੋੜ_exp = 0.0f;
    ਲਈ (ਪੂਰਨ i = tid; i < nclasses; i += WARP_ਮਾਪ) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = has_shared ? smem[i] : logits[i];
        ਜੋੜ_exp += sycl::ਘਾਤਾਂਕ(v - max_logit);
    }
    ਜੋੜ_exp = warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(ਜੋੜ_exp);
    ਅਚਲ ਦਸ਼ਮਲਵ log_ਜੋੜ = sycl::ਲਘੂਗਣਕ(ਜੋੜ_exp);

    ਦਸ਼ਮਲਵ loss = 0.0f;
    ਲਈ (ਪੂਰਨ i = tid; i < nclasses; i += WARP_ਮਾਪ) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_ਜੋੜ) * labels[i];
    }
    loss = -warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(loss) / (ਦਸ਼ਮਲਵ) nrows;

    ਜੇ (tid == 0) {
        ਪੰਕਤੀ_loss[ਪੰਕਤੀ] = loss;
    }
}

ਸਾਂਚਾ <ਤਾਰਕਿਕ has_shared>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ cross_entropy_loss_ਪਿੱਠ_f32_kernel(
        ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ grad,
        ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ logits,
        ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ labels,
        ਦਸ਼ਮਲਵ * __restrict__ dst,
        ਅਚਲ ਪੂਰਨ nclasses,
        ਅਚਲ ਪੂਰਨ nrows,
        ਦਸ਼ਮਲਵ * __restrict__ smem,
        ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ) {

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ.get_ਸਮੂਹ(2);
    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ.get_local_id(2);

    logits += (int64_t) ਪੰਕਤੀ * nclasses;
    labels += (int64_t) ਪੰਕਤੀ * nclasses;
    dst    += (int64_t) ਪੰਕਤੀ * nclasses;

    ਦਸ਼ਮਲਵ max_logit = -INFINITY;
    ਲਈ (ਪੂਰਨ i = tid; i < nclasses; i += WARP_ਮਾਪ) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        ਜੇ (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_ਮਾਪ>(max_logit);

    ਦਸ਼ਮਲਵ ਜੋੜ_exp = 0.0f;
    ਲਈ (ਪੂਰਨ i = tid; i < nclasses; i += WARP_ਮਾਪ) {
        ਅਚਲ ਦਸ਼ਮਲਵ v = sycl::ਘਾਤਾਂਕ((has_shared ? smem[i] : logits[i]) - max_logit);
        ਜੋੜ_exp += v;
        ਜੇ (has_shared) {
            smem[i] = v;
        } ਨਹੀਂਤਾਂ {
            dst[i] = v;
        }
    }
    ਜੋੜ_exp = warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(ਜੋੜ_exp);
    ਅਚਲ ਦਸ਼ਮਲਵ inv_ਜੋੜ = 1.0f / ਜੋੜ_exp;

    ਅਚਲ ਦਸ਼ਮਲਵ d_by_nrows = grad[0] / (ਦਸ਼ਮਲਵ) nrows;
    ਲਈ (ਪੂਰਨ i = tid; i < nclasses; i += WARP_ਮਾਪ) {
        ਅਚਲ ਦਸ਼ਮਲਵ sm_ਅੰਕ = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_ਅੰਕ * inv_ਜੋੜ - labels[i]) * d_by_nrows;
    }
}

ਸਥਿਰ ਖਾਲੀ cross_entropy_reduce_ਪੰਕਤੀਆਂ(
        ggml_backend_sycl_context & ctx,
        ਅਚਲ ਦਸ਼ਮਲਵ * ਪੰਕਤੀ_loss,
        ਦਸ਼ਮਲਵ * dst,
        ਅਚਲ int64_t nrows) {
    ਜੇ (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst, ਪੰਕਤੀ_loss, ਆਕਾਰ(ਦਸ਼ਮਲਵ))));
        ਵਾਪਸ;
    }

    ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ> tmp_alloc(ctx.pool(), nrows);
    ਦਸ਼ਮਲਵ * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp, ਪੰਕਤੀ_loss, nrows * ਆਕਾਰ(ਦਸ਼ਮਲਵ))));

    int64_t cur = nrows;
    ਜਦੋਂ (cur > 1) {
        ਅਚਲ int64_t ਬਾਹਰਲਾ = (cur + WARP_ਮਾਪ - 1) / WARP_ਮਾਪ;
        ਅਚਲ sycl::range<3> ਖੰਡ(1, 1, WARP_ਮਾਪ);
        ਅਚਲ sycl::range<3> grid(1, 1, ਬਾਹਰਲਾ);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * ਖੰਡ, ਖੰਡ),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ.get_ਸਮੂਹ(2);
                ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ.get_local_id(2);
                ਅਚਲ int64_t i = (int64_t) ਪੰਕਤੀ * WARP_ਮਾਪ + tid;
                ਦਸ਼ਮਲਵ v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(v);
                ਜੇ (tid == 0) {
                    tmp[ਪੰਕਤੀ] = v;
                }
            });
        cur = ਬਾਹਰਲਾ;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst, tmp, ਆਕਾਰ(ਦਸ਼ਮਲਵ))));
}

ਖਾਲੀ ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ int64_t nclasses = src0->ne[0];
    ਅਚਲ int64_t nrows = ggml_nrows(src0);

    ਅਚਲ ਦਸ਼ਮਲਵ * logits_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * labels_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_d = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ> ਪੰਕਤੀ_loss_alloc(ctx.pool(), nrows);
    ਦਸ਼ਮਲਵ * ਪੰਕਤੀ_loss = ਪੰਕਤੀ_loss_alloc.get();

    ਅਚਲ sycl::range<3> ਖੰਡ(1, 1, WARP_ਮਾਪ);
    ਅਚਲ sycl::range<3> grid(1, 1, nrows);
    ਅਚਲ size_t nbytes_shared = (size_t) nclasses * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    ਜੇ (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * ਖੰਡ, ਖੰਡ),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    cross_entropy_loss_f32_kernel<ਸੱਚ>(
                        logits_d, labels_d, ਪੰਕਤੀ_loss,
                        (ਪੂਰਨ) nclasses, (ਪੂਰਨ) nrows,
                        get_ਸੂਚਕ(smem), ਵਸਤੂ);
                });
        });
    } ਨਹੀਂਤਾਂ {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * ਖੰਡ, ਖੰਡ),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                cross_entropy_loss_f32_kernel<ਝੂਠ>(
                    logits_d, labels_d, ਪੰਕਤੀ_loss,
                    (ਪੂਰਨ) nclasses, (ਪੂਰਨ) nrows,
                    ਖਾਲੀ_ਸੂਚਕ, ਵਸਤੂ);
            });
    }

    cross_entropy_reduce_ਪੰਕਤੀਆਂ(ctx, ਪੰਕਤੀ_loss, dst_d, nrows);
}

ਖਾਲੀ ggml_sycl_cross_entropy_loss_ਪਿੱਠ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ਅਚਲ ggml_tensor * grad  = dst->src[0];
    ਅਚਲ ggml_tensor * src0f = dst->src[1];
    ਅਚਲ ggml_tensor * src1f = dst->src[2];

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

    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ int64_t nclasses = src0f->ne[0];
    ਅਚਲ int64_t nrows = ggml_nrows(src0f);

    ਅਚਲ ਦਸ਼ਮਲਵ * grad_d  = (ਅਚਲ ਦਸ਼ਮਲਵ *) grad->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * logits_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0f->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * labels_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1f->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_d = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਅਚਲ sycl::range<3> ਖੰਡ(1, 1, WARP_ਮਾਪ);
    ਅਚਲ sycl::range<3> grid(1, 1, nrows);
    ਅਚਲ size_t nbytes_shared = (size_t) nclasses * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    ਜੇ (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * ਖੰਡ, ਖੰਡ),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                    cross_entropy_loss_ਪਿੱਠ_f32_kernel<ਸੱਚ>(
                        grad_d, logits_d, labels_d, dst_d,
                        (ਪੂਰਨ) nclasses, (ਪੂਰਨ) nrows,
                        get_ਸੂਚਕ(smem), ਵਸਤੂ);
                });
        });
    } ਨਹੀਂਤਾਂ {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * ਖੰਡ, ਖੰਡ),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                cross_entropy_loss_ਪਿੱਠ_f32_kernel<ਝੂਠ>(
                    grad_d, logits_d, labels_d, dst_d,
                    (ਪੂਰਨ) nclasses, (ਪੂਰਨ) nrows,
                    ਖਾਲੀ_ਸੂਚਕ, ਵਸਤੂ);
            });
    }
}
