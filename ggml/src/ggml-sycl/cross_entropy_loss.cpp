#અન્તર્ભાવ "cross_entropy_loss.hpp"

#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cmath>

template <તાર્કિક has_shared>
સ્થિર __dpct_inline__ શૂન્ય cross_entropy_loss_f32_kernel(
        અચલ દશાંશ * __restrict__ logits,
        અચલ દશાંશ * __restrict__ labels,
        દશાંશ * __restrict__ row_loss,
        અચલ પૂર્ણાંક nclasses,
        અચલ પૂર્ણાંક nrows,
        દશાંશ * __restrict__ smem,
        અચલ sycl::nd_item<3> & item) {

    અચલ પૂર્ણાંક row = item.get_group(2);
    અચલ પૂર્ણાંક tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    દશાંશ max_logit = -INFINITY;
    માટે (પૂર્ણાંક i = tid; i < nclasses; i += WARP_SIZE) {
        અચલ દશાંશ v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        જો (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    દશાંશ sum_exp = 0.0f;
    માટે (પૂર્ણાંક i = tid; i < nclasses; i += WARP_SIZE) {
        અચલ દશાંશ v = has_shared ? smem[i] : logits[i];
        sum_exp += sycl::exp(v - max_logit);
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    અચલ દશાંશ log_sum = sycl::log(sum_exp);

    દશાંશ loss = 0.0f;
    માટે (પૂર્ણાંક i = tid; i < nclasses; i += WARP_SIZE) {
        અચલ દશાંશ v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_sum) * labels[i];
    }
    loss = -warp_reduce_sum<WARP_SIZE>(loss) / (દશાંશ) nrows;

    જો (tid == 0) {
        row_loss[row] = loss;
    }
}

template <તાર્કિક has_shared>
સ્થિર __dpct_inline__ શૂન્ય cross_entropy_loss_back_f32_kernel(
        અચલ દશાંશ * __restrict__ grad,
        અચલ દશાંશ * __restrict__ logits,
        અચલ દશાંશ * __restrict__ labels,
        દશાંશ * __restrict__ dst,
        અચલ પૂર્ણાંક nclasses,
        અચલ પૂર્ણાંક nrows,
        દશાંશ * __restrict__ smem,
        અચલ sycl::nd_item<3> & item) {

    અચલ પૂર્ણાંક row = item.get_group(2);
    અચલ પૂર્ણાંક tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    દશાંશ max_logit = -INFINITY;
    માટે (પૂર્ણાંક i = tid; i < nclasses; i += WARP_SIZE) {
        અચલ દશાંશ v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        જો (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    દશાંશ sum_exp = 0.0f;
    માટે (પૂર્ણાંક i = tid; i < nclasses; i += WARP_SIZE) {
        અચલ દશાંશ v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        sum_exp += v;
        જો (has_shared) {
            smem[i] = v;
        } નહીંતર {
            dst[i] = v;
        }
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    અચલ દશાંશ inv_sum = 1.0f / sum_exp;

    અચલ દશાંશ d_by_nrows = grad[0] / (દશાંશ) nrows;
    માટે (પૂર્ણાંક i = tid; i < nclasses; i += WARP_SIZE) {
        અચલ દશાંશ sm_અંક = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_અંક * inv_sum - labels[i]) * d_by_nrows;
    }
}

સ્થિર શૂન્ય cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        અચલ દશાંશ * row_loss,
        દશાંશ * dst,
        અચલ int64_t nrows) {
    જો (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->memcpy(dst, row_loss, કદ(દશાંશ))));
        પાછા;
    }

    ggml_sycl_pool_alloc<દશાંશ> tmp_alloc(ctx.pool(), nrows);
    દશાંશ * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(tmp, row_loss, nrows * કદ(દશાંશ))));

    int64_t cur = nrows;
    જ્યારે (cur > 1) {
        અચલ int64_t out = (cur + WARP_SIZE - 1) / WARP_SIZE;
        અચલ sycl::range<3> block(1, 1, WARP_SIZE);
        અચલ sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                અચલ પૂર્ણાંક row = item.get_group(2);
                અચલ પૂર્ણાંક tid = item.get_local_id(2);
                અચલ int64_t i = (int64_t) row * WARP_SIZE + tid;
                દશાંશ v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_sum<WARP_SIZE>(v);
                જો (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(dst, tmp, કદ(દશાંશ))));
}

શૂન્ય ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    અચલ int64_t nclasses = src0->ne[0];
    અચલ int64_t nrows = ggml_nrows(src0);

    અચલ દશાંશ * logits_d = (અચલ દશાંશ *) src0->data;
    અચલ દશાંશ * labels_d = (અચલ દશાંશ *) src1->data;
    દશાંશ * dst_d = (દશાંશ *) dst->data;

    ggml_sycl_pool_alloc<દશાંશ> row_loss_alloc(ctx.pool(), nrows);
    દશાંશ * row_loss = row_loss_alloc.get();

    અચલ sycl::range<3> block(1, 1, WARP_SIZE);
    અચલ sycl::range<3> grid(1, 1, nrows);
    અચલ size_t nbytes_shared = (size_t) nclasses * કદ(દશાંશ);
    અચલ size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    જો (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<દશાંશ, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32_kernel<સાચું>(
                        logits_d, labels_d, row_loss,
                        (પૂર્ણાંક) nclasses, (પૂર્ણાંક) nrows,
                        get_pointer(smem), item);
                });
        });
    } નહીંતર {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_f32_kernel<ખોટું>(
                    logits_d, labels_d, row_loss,
                    (પૂર્ણાંક) nclasses, (પૂર્ણાંક) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

શૂન્ય ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    અચલ ggml_tensor * grad  = dst->src[0];
    અચલ ggml_tensor * src0f = dst->src[1];
    અચલ ggml_tensor * src1f = dst->src[2];

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

    અચલ int64_t nclasses = src0f->ne[0];
    અચલ int64_t nrows = ggml_nrows(src0f);

    અચલ દશાંશ * grad_d  = (અચલ દશાંશ *) grad->data;
    અચલ દશાંશ * logits_d = (અચલ દશાંશ *) src0f->data;
    અચલ દશાંશ * labels_d = (અચલ દશાંશ *) src1f->data;
    દશાંશ * dst_d = (દશાંશ *) dst->data;

    અચલ sycl::range<3> block(1, 1, WARP_SIZE);
    અચલ sycl::range<3> grid(1, 1, nrows);
    અચલ size_t nbytes_shared = (size_t) nclasses * કદ(દશાંશ);
    અચલ size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    જો (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<દશાંશ, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32_kernel<સાચું>(
                        grad_d, logits_d, labels_d, dst_d,
                        (પૂર્ણાંક) nclasses, (પૂર્ણાંક) nrows,
                        get_pointer(smem), item);
                });
        });
    } નહીંતર {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_back_f32_kernel<ખોટું>(
                    grad_d, logits_d, labels_d, dst_d,
                    (પૂર્ણાંક) nclasses, (પૂર્ણાંક) nrows,
                    nullptr, item);
            });
    }
}
