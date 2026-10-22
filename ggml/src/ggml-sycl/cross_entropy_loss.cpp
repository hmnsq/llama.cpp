#समावेश "cross_entropy_loss.hpp"

#समावेश <cstdint>
#समावेश <cmath>

ढाँचा <bool has_shared>
अटल __dpct_inline__ व्योम cross_entropy_loss_f32_kernel(
        स्थिर भग्न * __restrict__ logits,
        स्थिर भग्न * __restrict__ labels,
        भग्न * __restrict__ row_loss,
        स्थिर पूर्णांक nclasses,
        स्थिर पूर्णांक nrows,
        भग्न * __restrict__ smem,
        स्थिर sycl::nd_item<3> & item) {

    स्थिर पूर्णांक row = item.get_group(2);
    स्थिर पूर्णांक tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    भग्न max_logit = -अनन्त;
    क्रम (पूर्णांक i = tid; i < nclasses; i += WARP_आकार) {
        स्थिर भग्न v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        अगर (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_आकार>(max_logit);

    भग्न योग_exp = 0.0f;
    क्रम (पूर्णांक i = tid; i < nclasses; i += WARP_आकार) {
        स्थिर भग्न v = has_shared ? smem[i] : logits[i];
        योग_exp += sycl::exp(v - max_logit);
    }
    योग_exp = warp_reduce_योग<WARP_आकार>(योग_exp);
    स्थिर भग्न log_योग = sycl::log(योग_exp);

    भग्न loss = 0.0f;
    क्रम (पूर्णांक i = tid; i < nclasses; i += WARP_आकार) {
        स्थिर भग्न v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_योग) * labels[i];
    }
    loss = -warp_reduce_योग<WARP_आकार>(loss) / (भग्न) nrows;

    अगर (tid == 0) {
        row_loss[row] = loss;
    }
}

ढाँचा <bool has_shared>
अटल __dpct_inline__ व्योम cross_entropy_loss_back_f32_kernel(
        स्थिर भग्न * __restrict__ grad,
        स्थिर भग्न * __restrict__ logits,
        स्थिर भग्न * __restrict__ labels,
        भग्न * __restrict__ dst,
        स्थिर पूर्णांक nclasses,
        स्थिर पूर्णांक nrows,
        भग्न * __restrict__ smem,
        स्थिर sycl::nd_item<3> & item) {

    स्थिर पूर्णांक row = item.get_group(2);
    स्थिर पूर्णांक tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    भग्न max_logit = -अनन्त;
    क्रम (पूर्णांक i = tid; i < nclasses; i += WARP_आकार) {
        स्थिर भग्न v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        अगर (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_आकार>(max_logit);

    भग्न योग_exp = 0.0f;
    क्रम (पूर्णांक i = tid; i < nclasses; i += WARP_आकार) {
        स्थिर भग्न v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        योग_exp += v;
        अगर (has_shared) {
            smem[i] = v;
        } अन्यथा {
            dst[i] = v;
        }
    }
    योग_exp = warp_reduce_योग<WARP_आकार>(योग_exp);
    स्थिर भग्न inv_योग = 1.0f / योग_exp;

    स्थिर भग्न d_by_nrows = grad[0] / (भग्न) nrows;
    क्रम (पूर्णांक i = tid; i < nclasses; i += WARP_आकार) {
        स्थिर भग्न sm_अंक = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_अंक * inv_योग - labels[i]) * d_by_nrows;
    }
}

अटल व्योम cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        स्थिर भग्न * row_loss,
        भग्न * dst,
        स्थिर int64_t nrows) {
    अगर (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->स_नकल(dst, row_loss, माप(भग्न))));
        वापस;
    }

    ggml_sycl_pool_alloc<भग्न> tmp_alloc(ctx.pool(), nrows);
    भग्न * tmp = tmp_alloc.लो();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->स_नकल(tmp, row_loss, nrows * माप(भग्न))));

    int64_t cur = nrows;
    जबतक (cur > 1) {
        स्थिर int64_t out = (cur + WARP_आकार - 1) / WARP_आकार;
        स्थिर sycl::range<3> block(1, 1, WARP_आकार);
        स्थिर sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                स्थिर पूर्णांक row = item.get_group(2);
                स्थिर पूर्णांक tid = item.get_local_id(2);
                स्थिर int64_t i = (int64_t) row * WARP_आकार + tid;
                भग्न v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_योग<WARP_आकार>(v);
                अगर (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->स_नकल(dst, tmp, माप(भग्न))));
}

व्योम ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिर int64_t nclasses = src0->ne[0];
    स्थिर int64_t nrows = ggml_nrows(src0);

    स्थिर भग्न * logits_d = (स्थिर भग्न *) src0->तथ्य;
    स्थिर भग्न * labels_d = (स्थिर भग्न *) src1->तथ्य;
    भग्न * dst_d = (भग्न *) dst->तथ्य;

    ggml_sycl_pool_alloc<भग्न> row_loss_alloc(ctx.pool(), nrows);
    भग्न * row_loss = row_loss_alloc.लो();

    स्थिर sycl::range<3> block(1, 1, WARP_आकार);
    स्थिर sycl::range<3> grid(1, 1, nrows);
    स्थिर माप_प्रकार nbytes_shared = (माप_प्रकार) nclasses * माप(भग्न);
    स्थिर माप_प्रकार smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    अगर (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<भग्न, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    cross_entropy_loss_f32_kernel<true>(
                        logits_d, labels_d, row_loss,
                        (पूर्णांक) nclasses, (पूर्णांक) nrows,
                        get_pointer(smem), item);
                });
        });
    } अन्यथा {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                cross_entropy_loss_f32_kernel<false>(
                    logits_d, labels_d, row_loss,
                    (पूर्णांक) nclasses, (पूर्णांक) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

व्योम ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    स्थिर ggml_tensor * grad  = dst->src[0];
    स्थिर ggml_tensor * src0f = dst->src[1];
    स्थिर ggml_tensor * src1f = dst->src[2];

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

    स्थिर int64_t nclasses = src0f->ne[0];
    स्थिर int64_t nrows = ggml_nrows(src0f);

    स्थिर भग्न * grad_d  = (स्थिर भग्न *) grad->तथ्य;
    स्थिर भग्न * logits_d = (स्थिर भग्न *) src0f->तथ्य;
    स्थिर भग्न * labels_d = (स्थिर भग्न *) src1f->तथ्य;
    भग्न * dst_d = (भग्न *) dst->तथ्य;

    स्थिर sycl::range<3> block(1, 1, WARP_आकार);
    स्थिर sycl::range<3> grid(1, 1, nrows);
    स्थिर माप_प्रकार nbytes_shared = (माप_प्रकार) nclasses * माप(भग्न);
    स्थिर माप_प्रकार smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    अगर (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<भग्न, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                    cross_entropy_loss_back_f32_kernel<true>(
                        grad_d, logits_d, labels_d, dst_d,
                        (पूर्णांक) nclasses, (पूर्णांक) nrows,
                        get_pointer(smem), item);
                });
        });
    } अन्यथा {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                cross_entropy_loss_back_f32_kernel<false>(
                    grad_d, logits_d, labels_d, dst_d,
                    (पूर्णांक) nclasses, (पूर्णांक) nrows,
                    nullptr, item);
            });
    }
}
