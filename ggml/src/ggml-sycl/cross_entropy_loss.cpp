#अन्तर्भाव "cross_entropy_loss.hpp"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <cmath>

ढाँचा <तार्किक has_shared>
स्थिर __dpct_inline__ शून्य cross_entropy_loss_f32_kernel(
        अचर भिन्न * __restrict__ logits,
        अचर भिन्न * __restrict__ labels,
        भिन्न * __restrict__ row_loss,
        अचर पूर्णाङ्क nclasses,
        अचर पूर्णाङ्क nrows,
        भिन्न * __restrict__ smem,
        अचर sycl::nd_item<3> & item) {

    अचर पूर्णाङ्क row = item.get_group(2);
    अचर पूर्णाङ्क tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    भिन्न max_logit = -अनन्त;
    लागि (पूर्णाङ्क i = tid; i < nclasses; i += WARP_SIZE) {
        अचर भिन्न v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        यदि (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    भिन्न sum_exp = 0.0f;
    लागि (पूर्णाङ्क i = tid; i < nclasses; i += WARP_SIZE) {
        अचर भिन्न v = has_shared ? smem[i] : logits[i];
        sum_exp += sycl::exp(v - max_logit);
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    अचर भिन्न log_sum = sycl::log(sum_exp);

    भिन्न loss = 0.0f;
    लागि (पूर्णाङ्क i = tid; i < nclasses; i += WARP_SIZE) {
        अचर भिन्न v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_sum) * labels[i];
    }
    loss = -warp_reduce_sum<WARP_SIZE>(loss) / (भिन्न) nrows;

    यदि (tid == 0) {
        row_loss[row] = loss;
    }
}

ढाँचा <तार्किक has_shared>
स्थिर __dpct_inline__ शून्य cross_entropy_loss_back_f32_kernel(
        अचर भिन्न * __restrict__ grad,
        अचर भिन्न * __restrict__ logits,
        अचर भिन्न * __restrict__ labels,
        भिन्न * __restrict__ dst,
        अचर पूर्णाङ्क nclasses,
        अचर पूर्णाङ्क nrows,
        भिन्न * __restrict__ smem,
        अचर sycl::nd_item<3> & item) {

    अचर पूर्णाङ्क row = item.get_group(2);
    अचर पूर्णाङ्क tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    भिन्न max_logit = -अनन्त;
    लागि (पूर्णाङ्क i = tid; i < nclasses; i += WARP_SIZE) {
        अचर भिन्न v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        यदि (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    भिन्न sum_exp = 0.0f;
    लागि (पूर्णाङ्क i = tid; i < nclasses; i += WARP_SIZE) {
        अचर भिन्न v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        sum_exp += v;
        यदि (has_shared) {
            smem[i] = v;
        } अन्यथा {
            dst[i] = v;
        }
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    अचर भिन्न inv_sum = 1.0f / sum_exp;

    अचर भिन्न d_by_nrows = grad[0] / (भिन्न) nrows;
    लागि (पूर्णाङ्क i = tid; i < nclasses; i += WARP_SIZE) {
        अचर भिन्न sm_अङ्क = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_अङ्क * inv_sum - labels[i]) * d_by_nrows;
    }
}

स्थिर शून्य cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        अचर भिन्न * row_loss,
        भिन्न * dst,
        अचर int64_t nrows) {
    यदि (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->स_नकल(dst, row_loss, परिमाण(भिन्न))));
        फर्काऊ;
    }

    ggml_sycl_pool_alloc<भिन्न> tmp_alloc(ctx.pool(), nrows);
    भिन्न * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->स_नकल(tmp, row_loss, nrows * परिमाण(भिन्न))));

    int64_t cur = nrows;
    जबसम्म (cur > 1) {
        अचर int64_t out = (cur + WARP_SIZE - 1) / WARP_SIZE;
        अचर sycl::range<3> block(1, 1, WARP_SIZE);
        अचर sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                अचर पूर्णाङ्क row = item.get_group(2);
                अचर पूर्णाङ्क tid = item.get_local_id(2);
                अचर int64_t i = (int64_t) row * WARP_SIZE + tid;
                भिन्न v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_sum<WARP_SIZE>(v);
                यदि (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->स_नकल(dst, tmp, परिमाण(भिन्न))));
}

शून्य ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    अचर ggml_tensor * src0 = dst->src[0];
    अचर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    अचर int64_t nclasses = src0->ne[0];
    अचर int64_t nrows = ggml_nrows(src0);

    अचर भिन्न * logits_d = (अचर भिन्न *) src0->data;
    अचर भिन्न * labels_d = (अचर भिन्न *) src1->data;
    भिन्न * dst_d = (भिन्न *) dst->data;

    ggml_sycl_pool_alloc<भिन्न> row_loss_alloc(ctx.pool(), nrows);
    भिन्न * row_loss = row_loss_alloc.get();

    अचर sycl::range<3> block(1, 1, WARP_SIZE);
    अचर sycl::range<3> grid(1, 1, nrows);
    अचर माप_प्रकार nbytes_shared = (माप_प्रकार) nclasses * परिमाण(भिन्न);
    अचर माप_प्रकार smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    यदि (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<भिन्न, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32_kernel<सत्य>(
                        logits_d, labels_d, row_loss,
                        (पूर्णाङ्क) nclasses, (पूर्णाङ्क) nrows,
                        get_pointer(smem), item);
                });
        });
    } अन्यथा {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_f32_kernel<असत्य>(
                    logits_d, labels_d, row_loss,
                    (पूर्णाङ्क) nclasses, (पूर्णाङ्क) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

शून्य ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    अचर ggml_tensor * grad  = dst->src[0];
    अचर ggml_tensor * src0f = dst->src[1];
    अचर ggml_tensor * src1f = dst->src[2];

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

    अचर int64_t nclasses = src0f->ne[0];
    अचर int64_t nrows = ggml_nrows(src0f);

    अचर भिन्न * grad_d  = (अचर भिन्न *) grad->data;
    अचर भिन्न * logits_d = (अचर भिन्न *) src0f->data;
    अचर भिन्न * labels_d = (अचर भिन्न *) src1f->data;
    भिन्न * dst_d = (भिन्न *) dst->data;

    अचर sycl::range<3> block(1, 1, WARP_SIZE);
    अचर sycl::range<3> grid(1, 1, nrows);
    अचर माप_प्रकार nbytes_shared = (माप_प्रकार) nclasses * परिमाण(भिन्न);
    अचर माप_प्रकार smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    यदि (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<भिन्न, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32_kernel<सत्य>(
                        grad_d, logits_d, labels_d, dst_d,
                        (पूर्णाङ्क) nclasses, (पूर्णाङ्क) nrows,
                        get_pointer(smem), item);
                });
        });
    } अन्यथा {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_back_f32_kernel<असत्य>(
                    grad_d, logits_d, labels_d, dst_d,
                    (पूर्णाङ्क) nclasses, (पूर्णाङ्क) nrows,
                    nullptr, item);
            });
    }
}
