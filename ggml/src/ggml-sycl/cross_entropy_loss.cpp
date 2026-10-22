#ಅನ್ತರ್ಭಾವ "cross_entropy_loss.hpp"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cmath>

template <ತಾರ್ಕಿಕ has_shared>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ cross_entropy_loss_f32_kernel(
        ಧ್ರುವ ಭಿನ್ನ * __restrict__ logits,
        ಧ್ರುವ ಭಿನ್ನ * __restrict__ labels,
        ಭಿನ್ನ * __restrict__ row_loss,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nclasses,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nrows,
        ಭಿನ್ನ * __restrict__ smem,
        ಧ್ರುವ sycl::nd_item<3> & item) {

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ row = item.get_group(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    ಭಿನ್ನ max_logit = -INFINITY;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = tid; i < nclasses; i += WARP_SIZE) {
        ಧ್ರುವ ಭಿನ್ನ v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        ಯದಿ (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    ಭಿನ್ನ sum_exp = 0.0f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = tid; i < nclasses; i += WARP_SIZE) {
        ಧ್ರುವ ಭಿನ್ನ v = has_shared ? smem[i] : logits[i];
        sum_exp += sycl::exp(v - max_logit);
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    ಧ್ರುವ ಭಿನ್ನ log_sum = sycl::log(sum_exp);

    ಭಿನ್ನ loss = 0.0f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = tid; i < nclasses; i += WARP_SIZE) {
        ಧ್ರುವ ಭಿನ್ನ v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_sum) * labels[i];
    }
    loss = -warp_reduce_sum<WARP_SIZE>(loss) / (ಭಿನ್ನ) nrows;

    ಯದಿ (tid == 0) {
        row_loss[row] = loss;
    }
}

template <ತಾರ್ಕಿಕ has_shared>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ cross_entropy_loss_back_f32_kernel(
        ಧ್ರುವ ಭಿನ್ನ * __restrict__ grad,
        ಧ್ರುವ ಭಿನ್ನ * __restrict__ logits,
        ಧ್ರುವ ಭಿನ್ನ * __restrict__ labels,
        ಭಿನ್ನ * __restrict__ dst,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nclasses,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nrows,
        ಭಿನ್ನ * __restrict__ smem,
        ಧ್ರುವ sycl::nd_item<3> & item) {

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ row = item.get_group(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    ಭಿನ್ನ max_logit = -INFINITY;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = tid; i < nclasses; i += WARP_SIZE) {
        ಧ್ರುವ ಭಿನ್ನ v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        ಯದಿ (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    ಭಿನ್ನ sum_exp = 0.0f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = tid; i < nclasses; i += WARP_SIZE) {
        ಧ್ರುವ ಭಿನ್ನ v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        sum_exp += v;
        ಯದಿ (has_shared) {
            smem[i] = v;
        } ಅನ್ಯಥಾ {
            dst[i] = v;
        }
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    ಧ್ರುವ ಭಿನ್ನ inv_sum = 1.0f / sum_exp;

    ಧ್ರುವ ಭಿನ್ನ d_by_nrows = grad[0] / (ಭಿನ್ನ) nrows;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = tid; i < nclasses; i += WARP_SIZE) {
        ಧ್ರುವ ಭಿನ್ನ sm_ಅಂಕೆ = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_ಅಂಕೆ * inv_sum - labels[i]) * d_by_nrows;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        ಧ್ರುವ ಭಿನ್ನ * row_loss,
        ಭಿನ್ನ * dst,
        ಧ್ರುವ int64_t nrows) {
    ಯದಿ (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->memcpy(dst, row_loss, ಪರಿಮಾಣ(ಭಿನ್ನ))));
        ಪ್ರತ್ಯಾಗಮ;
    }

    ggml_sycl_pool_alloc<ಭಿನ್ನ> tmp_alloc(ctx.pool(), nrows);
    ಭಿನ್ನ * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(tmp, row_loss, nrows * ಪರಿಮಾಣ(ಭಿನ್ನ))));

    int64_t cur = nrows;
    ಯಾವತ್ (cur > 1) {
        ಧ್ರುವ int64_t out = (cur + WARP_SIZE - 1) / WARP_SIZE;
        ಧ್ರುವ sycl::range<3> block(1, 1, WARP_SIZE);
        ಧ್ರುವ sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ row = item.get_group(2);
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tid = item.get_local_id(2);
                ಧ್ರುವ int64_t i = (int64_t) row * WARP_SIZE + tid;
                ಭಿನ್ನ v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_sum<WARP_SIZE>(v);
                ಯದಿ (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(dst, tmp, ಪರಿಮಾಣ(ಭಿನ್ನ))));
}

ಶೂನ್ಯ ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ಧ್ರುವ int64_t nclasses = src0->ne[0];
    ಧ್ರುವ int64_t nrows = ggml_nrows(src0);

    ಧ್ರುವ ಭಿನ್ನ * logits_d = (ಧ್ರುವ ಭಿನ್ನ *) src0->data;
    ಧ್ರುವ ಭಿನ್ನ * labels_d = (ಧ್ರುವ ಭಿನ್ನ *) src1->data;
    ಭಿನ್ನ * dst_d = (ಭಿನ್ನ *) dst->data;

    ggml_sycl_pool_alloc<ಭಿನ್ನ> row_loss_alloc(ctx.pool(), nrows);
    ಭಿನ್ನ * row_loss = row_loss_alloc.get();

    ಧ್ರುವ sycl::range<3> block(1, 1, WARP_SIZE);
    ಧ್ರುವ sycl::range<3> grid(1, 1, nrows);
    ಧ್ರುವ size_t nbytes_shared = (size_t) nclasses * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    ಯದಿ (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ಭಿನ್ನ, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32_kernel<ಸತ್ಯ>(
                        logits_d, labels_d, row_loss,
                        (ಪೂರ್ಣಾಙ್ಕ) nclasses, (ಪೂರ್ಣಾಙ್ಕ) nrows,
                        get_pointer(smem), item);
                });
        });
    } ಅನ್ಯಥಾ {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_f32_kernel<ಅಸತ್ಯ>(
                    logits_d, labels_d, row_loss,
                    (ಪೂರ್ಣಾಙ್ಕ) nclasses, (ಪೂರ್ಣಾಙ್ಕ) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

ಶೂನ್ಯ ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ಧ್ರುವ ggml_tensor * grad  = dst->src[0];
    ಧ್ರುವ ggml_tensor * src0f = dst->src[1];
    ಧ್ರುವ ggml_tensor * src1f = dst->src[2];

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

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ಧ್ರುವ int64_t nclasses = src0f->ne[0];
    ಧ್ರುವ int64_t nrows = ggml_nrows(src0f);

    ಧ್ರುವ ಭಿನ್ನ * grad_d  = (ಧ್ರುವ ಭಿನ್ನ *) grad->data;
    ಧ್ರುವ ಭಿನ್ನ * logits_d = (ಧ್ರುವ ಭಿನ್ನ *) src0f->data;
    ಧ್ರುವ ಭಿನ್ನ * labels_d = (ಧ್ರುವ ಭಿನ್ನ *) src1f->data;
    ಭಿನ್ನ * dst_d = (ಭಿನ್ನ *) dst->data;

    ಧ್ರುವ sycl::range<3> block(1, 1, WARP_SIZE);
    ಧ್ರುವ sycl::range<3> grid(1, 1, nrows);
    ಧ್ರುವ size_t nbytes_shared = (size_t) nclasses * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    ಯದಿ (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ಭಿನ್ನ, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32_kernel<ಸತ್ಯ>(
                        grad_d, logits_d, labels_d, dst_d,
                        (ಪೂರ್ಣಾಙ್ಕ) nclasses, (ಪೂರ್ಣಾಙ್ಕ) nrows,
                        get_pointer(smem), item);
                });
        });
    } ಅನ್ಯಥಾ {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_back_f32_kernel<ಅಸತ್ಯ>(
                    grad_d, logits_d, labels_d, dst_d,
                    (ಪೂರ್ಣಾಙ್ಕ) nclasses, (ಪೂರ್ಣಾಙ್ಕ) nrows,
                    nullptr, item);
            });
    }
}
