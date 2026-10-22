#அந்தர்பாவ "cross_entropy_loss.hpp"

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cmath>

template <தருக்க has_shared>
நிலையான __dpct_inline__ வெறுமை cross_entropy_loss_f32_kernel(
        மாறிலி தசம * __restrict__ logits,
        மாறிலி தசம * __restrict__ labels,
        தசம * __restrict__ row_loss,
        மாறிலி முழு nclasses,
        மாறிலி முழு nrows,
        தசம * __restrict__ smem,
        மாறிலி sycl::nd_item<3> & item) {

    மாறிலி முழு row = item.get_group(2);
    மாறிலி முழு tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    தசம max_logit = -INFINITY;
    க்கு (முழு i = tid; i < nclasses; i += WARP_SIZE) {
        மாறிலி தசம v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        எனில் (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    தசம sum_exp = 0.0f;
    க்கு (முழு i = tid; i < nclasses; i += WARP_SIZE) {
        மாறிலி தசம v = has_shared ? smem[i] : logits[i];
        sum_exp += sycl::exp(v - max_logit);
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    மாறிலி தசம log_sum = sycl::log(sum_exp);

    தசம loss = 0.0f;
    க்கு (முழு i = tid; i < nclasses; i += WARP_SIZE) {
        மாறிலி தசம v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_sum) * labels[i];
    }
    loss = -warp_reduce_sum<WARP_SIZE>(loss) / (தசம) nrows;

    எனில் (tid == 0) {
        row_loss[row] = loss;
    }
}

template <தருக்க has_shared>
நிலையான __dpct_inline__ வெறுமை cross_entropy_loss_back_f32_kernel(
        மாறிலி தசம * __restrict__ grad,
        மாறிலி தசம * __restrict__ logits,
        மாறிலி தசம * __restrict__ labels,
        தசம * __restrict__ dst,
        மாறிலி முழு nclasses,
        மாறிலி முழு nrows,
        தசம * __restrict__ smem,
        மாறிலி sycl::nd_item<3> & item) {

    மாறிலி முழு row = item.get_group(2);
    மாறிலி முழு tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    தசம max_logit = -INFINITY;
    க்கு (முழு i = tid; i < nclasses; i += WARP_SIZE) {
        மாறிலி தசம v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        எனில் (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    தசம sum_exp = 0.0f;
    க்கு (முழு i = tid; i < nclasses; i += WARP_SIZE) {
        மாறிலி தசம v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        sum_exp += v;
        எனில் (has_shared) {
            smem[i] = v;
        } இல்லையெனில் {
            dst[i] = v;
        }
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    மாறிலி தசம inv_sum = 1.0f / sum_exp;

    மாறிலி தசம d_by_nrows = grad[0] / (தசம) nrows;
    க்கு (முழு i = tid; i < nclasses; i += WARP_SIZE) {
        மாறிலி தசம sm_இலக்கம் = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_இலக்கம் * inv_sum - labels[i]) * d_by_nrows;
    }
}

நிலையான வெறுமை cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        மாறிலி தசம * row_loss,
        தசம * dst,
        மாறிலி int64_t nrows) {
    எனில் (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->memcpy(dst, row_loss, அளவு(தசம))));
        திரும்பு;
    }

    ggml_sycl_pool_alloc<தசம> tmp_alloc(ctx.pool(), nrows);
    தசம * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(tmp, row_loss, nrows * அளவு(தசம))));

    int64_t cur = nrows;
    போது (cur > 1) {
        மாறிலி int64_t out = (cur + WARP_SIZE - 1) / WARP_SIZE;
        மாறிலி sycl::range<3> block(1, 1, WARP_SIZE);
        மாறிலி sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                மாறிலி முழு row = item.get_group(2);
                மாறிலி முழு tid = item.get_local_id(2);
                மாறிலி int64_t i = (int64_t) row * WARP_SIZE + tid;
                தசம v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_sum<WARP_SIZE>(v);
                எனில் (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(dst, tmp, அளவு(தசம))));
}

வெறுமை ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    மாறிலி int64_t nclasses = src0->ne[0];
    மாறிலி int64_t nrows = ggml_nrows(src0);

    மாறிலி தசம * logits_d = (மாறிலி தசம *) src0->data;
    மாறிலி தசம * labels_d = (மாறிலி தசம *) src1->data;
    தசம * dst_d = (தசம *) dst->data;

    ggml_sycl_pool_alloc<தசம> row_loss_alloc(ctx.pool(), nrows);
    தசம * row_loss = row_loss_alloc.get();

    மாறிலி sycl::range<3> block(1, 1, WARP_SIZE);
    மாறிலி sycl::range<3> grid(1, 1, nrows);
    மாறிலி size_t nbytes_shared = (size_t) nclasses * அளவு(தசம);
    மாறிலி size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    எனில் (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<தசம, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32_kernel<உண்மை>(
                        logits_d, labels_d, row_loss,
                        (முழு) nclasses, (முழு) nrows,
                        get_pointer(smem), item);
                });
        });
    } இல்லையெனில் {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_f32_kernel<பொய்>(
                    logits_d, labels_d, row_loss,
                    (முழு) nclasses, (முழு) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

வெறுமை ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    மாறிலி ggml_tensor * grad  = dst->src[0];
    மாறிலி ggml_tensor * src0f = dst->src[1];
    மாறிலி ggml_tensor * src1f = dst->src[2];

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

    மாறிலி int64_t nclasses = src0f->ne[0];
    மாறிலி int64_t nrows = ggml_nrows(src0f);

    மாறிலி தசம * grad_d  = (மாறிலி தசம *) grad->data;
    மாறிலி தசம * logits_d = (மாறிலி தசம *) src0f->data;
    மாறிலி தசம * labels_d = (மாறிலி தசம *) src1f->data;
    தசம * dst_d = (தசம *) dst->data;

    மாறிலி sycl::range<3> block(1, 1, WARP_SIZE);
    மாறிலி sycl::range<3> grid(1, 1, nrows);
    மாறிலி size_t nbytes_shared = (size_t) nclasses * அளவு(தசம);
    மாறிலி size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    எனில் (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<தசம, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32_kernel<உண்மை>(
                        grad_d, logits_d, labels_d, dst_d,
                        (முழு) nclasses, (முழு) nrows,
                        get_pointer(smem), item);
                });
        });
    } இல்லையெனில் {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_back_f32_kernel<பொய்>(
                    grad_d, logits_d, labels_d, dst_d,
                    (முழு) nclasses, (முழு) nrows,
                    nullptr, item);
            });
    }
}
