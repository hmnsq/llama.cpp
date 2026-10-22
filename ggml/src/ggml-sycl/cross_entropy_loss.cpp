#ଅନ୍ତର୍ଭାଵ "cross_entropy_loss.hpp"

#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <cmath>

template <ତାର୍କିକ has_shared>
ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ cross_entropy_loss_f32_kernel(
        ଅଚଳ ଦଶମିକ * __restrict__ logits,
        ଅଚଳ ଦଶମିକ * __restrict__ labels,
        ଦଶମିକ * __restrict__ row_loss,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nclasses,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଦଶମିକ * __restrict__ smem,
        ଅଚଳ sycl::nd_item<3> & item) {

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item.get_group(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    ଦଶମିକ max_logit = -INFINITY;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = tid; i < nclasses; i += WARP_SIZE) {
        ଅଚଳ ଦଶମିକ v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        ଯଦି (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    ଦଶମିକ sum_exp = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = tid; i < nclasses; i += WARP_SIZE) {
        ଅଚଳ ଦଶମିକ v = has_shared ? smem[i] : logits[i];
        sum_exp += sycl::exp(v - max_logit);
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    ଅଚଳ ଦଶମିକ log_sum = sycl::log(sum_exp);

    ଦଶମିକ loss = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = tid; i < nclasses; i += WARP_SIZE) {
        ଅଚଳ ଦଶମିକ v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_sum) * labels[i];
    }
    loss = -warp_reduce_sum<WARP_SIZE>(loss) / (ଦଶମିକ) nrows;

    ଯଦି (tid == 0) {
        row_loss[row] = loss;
    }
}

template <ତାର୍କିକ has_shared>
ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ cross_entropy_loss_back_f32_kernel(
        ଅଚଳ ଦଶମିକ * __restrict__ grad,
        ଅଚଳ ଦଶମିକ * __restrict__ logits,
        ଅଚଳ ଦଶମିକ * __restrict__ labels,
        ଦଶମିକ * __restrict__ dst,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nclasses,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଦଶମିକ * __restrict__ smem,
        ଅଚଳ sycl::nd_item<3> & item) {

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item.get_group(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    ଦଶମିକ max_logit = -INFINITY;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = tid; i < nclasses; i += WARP_SIZE) {
        ଅଚଳ ଦଶମିକ v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        ଯଦି (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    ଦଶମିକ sum_exp = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = tid; i < nclasses; i += WARP_SIZE) {
        ଅଚଳ ଦଶମିକ v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        sum_exp += v;
        ଯଦି (has_shared) {
            smem[i] = v;
        } ଅନ୍ୟଥା {
            dst[i] = v;
        }
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    ଅଚଳ ଦଶମିକ inv_sum = 1.0f / sum_exp;

    ଅଚଳ ଦଶମିକ d_by_nrows = grad[0] / (ଦଶମିକ) nrows;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = tid; i < nclasses; i += WARP_SIZE) {
        ଅଚଳ ଦଶମିକ sm_ଅଙ୍କ = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_ଅଙ୍କ * inv_sum - labels[i]) * d_by_nrows;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        ଅଚଳ ଦଶମିକ * row_loss,
        ଦଶମିକ * dst,
        ଅଚଳ int64_t nrows) {
    ଯଦି (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->memcpy(dst, row_loss, ଆକାର(ଦଶମିକ))));
        ଫେରନ୍ତୁ;
    }

    ggml_sycl_pool_alloc<ଦଶମିକ> tmp_alloc(ctx.pool(), nrows);
    ଦଶମିକ * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(tmp, row_loss, nrows * ଆକାର(ଦଶମିକ))));

    int64_t cur = nrows;
    ଯେତେବେଳେ (cur > 1) {
        ଅଚଳ int64_t out = (cur + WARP_SIZE - 1) / WARP_SIZE;
        ଅଚଳ sycl::range<3> block(1, 1, WARP_SIZE);
        ଅଚଳ sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item.get_group(2);
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid = item.get_local_id(2);
                ଅଚଳ int64_t i = (int64_t) row * WARP_SIZE + tid;
                ଦଶମିକ v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_sum<WARP_SIZE>(v);
                ଯଦି (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(dst, tmp, ଆକାର(ଦଶମିକ))));
}

ଶୂନ୍ୟ ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ଅଚଳ int64_t nclasses = src0->ne[0];
    ଅଚଳ int64_t nrows = ggml_nrows(src0);

    ଅଚଳ ଦଶମିକ * logits_d = (ଅଚଳ ଦଶମିକ *) src0->data;
    ଅଚଳ ଦଶମିକ * labels_d = (ଅଚଳ ଦଶମିକ *) src1->data;
    ଦଶମିକ * dst_d = (ଦଶମିକ *) dst->data;

    ggml_sycl_pool_alloc<ଦଶମିକ> row_loss_alloc(ctx.pool(), nrows);
    ଦଶମିକ * row_loss = row_loss_alloc.get();

    ଅଚଳ sycl::range<3> block(1, 1, WARP_SIZE);
    ଅଚଳ sycl::range<3> grid(1, 1, nrows);
    ଅଚଳ size_t nbytes_shared = (size_t) nclasses * ଆକାର(ଦଶମିକ);
    ଅଚଳ size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    ଯଦି (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ଦଶମିକ, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32_kernel<ସତ୍ୟ>(
                        logits_d, labels_d, row_loss,
                        (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) nclasses, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) nrows,
                        get_pointer(smem), item);
                });
        });
    } ଅନ୍ୟଥା {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_f32_kernel<ମିଥ୍ୟା>(
                    logits_d, labels_d, row_loss,
                    (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) nclasses, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

ଶୂନ୍ୟ ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ଅଚଳ ggml_tensor * grad  = dst->src[0];
    ଅଚଳ ggml_tensor * src0f = dst->src[1];
    ଅଚଳ ggml_tensor * src1f = dst->src[2];

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

    ଅଚଳ int64_t nclasses = src0f->ne[0];
    ଅଚଳ int64_t nrows = ggml_nrows(src0f);

    ଅଚଳ ଦଶମିକ * grad_d  = (ଅଚଳ ଦଶମିକ *) grad->data;
    ଅଚଳ ଦଶମିକ * logits_d = (ଅଚଳ ଦଶମିକ *) src0f->data;
    ଅଚଳ ଦଶମିକ * labels_d = (ଅଚଳ ଦଶମିକ *) src1f->data;
    ଦଶମିକ * dst_d = (ଦଶମିକ *) dst->data;

    ଅଚଳ sycl::range<3> block(1, 1, WARP_SIZE);
    ଅଚଳ sycl::range<3> grid(1, 1, nrows);
    ଅଚଳ size_t nbytes_shared = (size_t) nclasses * ଆକାର(ଦଶମିକ);
    ଅଚଳ size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    ଯଦି (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ଦଶମିକ, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32_kernel<ସତ୍ୟ>(
                        grad_d, logits_d, labels_d, dst_d,
                        (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) nclasses, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) nrows,
                        get_pointer(smem), item);
                });
        });
    } ଅନ୍ୟଥା {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_back_f32_kernel<ମିଥ୍ୟା>(
                    grad_d, logits_d, labels_d, dst_d,
                    (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) nclasses, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) nrows,
                    nullptr, item);
            });
    }
}
