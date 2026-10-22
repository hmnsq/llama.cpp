#अन्तर्भाव "cross_entropy_loss.hpp"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <cmath>

ढाँचा <तार्किक has_shared>
स्थिर __dpct_inline__ शून्य cross_entropy_loss_f32_kernel(
        स्थिरांक दशांश * __restrict__ logits,
        स्थिरांक दशांश * __restrict__ labels,
        दशांश * __restrict__ row_loss,
        स्थिरांक पूर्णांक nclasses,
        स्थिरांक पूर्णांक nrows,
        दशांश * __restrict__ smem,
        स्थिरांक sycl::nd_item<3> & item) {

    स्थिरांक पूर्णांक row = item.get_group(2);
    स्थिरांक पूर्णांक tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    दशांश max_logit = -अनन्त;
    साठी (पूर्णांक i = tid; i < nclasses; i += WARP_SIZE) {
        स्थिरांक दशांश v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        जर (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    दशांश sum_exp = 0.0f;
    साठी (पूर्णांक i = tid; i < nclasses; i += WARP_SIZE) {
        स्थिरांक दशांश v = has_shared ? smem[i] : logits[i];
        sum_exp += sycl::exp(v - max_logit);
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    स्थिरांक दशांश log_sum = sycl::log(sum_exp);

    दशांश loss = 0.0f;
    साठी (पूर्णांक i = tid; i < nclasses; i += WARP_SIZE) {
        स्थिरांक दशांश v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_sum) * labels[i];
    }
    loss = -warp_reduce_sum<WARP_SIZE>(loss) / (दशांश) nrows;

    जर (tid == 0) {
        row_loss[row] = loss;
    }
}

ढाँचा <तार्किक has_shared>
स्थिर __dpct_inline__ शून्य cross_entropy_loss_back_f32_kernel(
        स्थिरांक दशांश * __restrict__ grad,
        स्थिरांक दशांश * __restrict__ logits,
        स्थिरांक दशांश * __restrict__ labels,
        दशांश * __restrict__ dst,
        स्थिरांक पूर्णांक nclasses,
        स्थिरांक पूर्णांक nrows,
        दशांश * __restrict__ smem,
        स्थिरांक sycl::nd_item<3> & item) {

    स्थिरांक पूर्णांक row = item.get_group(2);
    स्थिरांक पूर्णांक tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    दशांश max_logit = -अनन्त;
    साठी (पूर्णांक i = tid; i < nclasses; i += WARP_SIZE) {
        स्थिरांक दशांश v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        जर (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    दशांश sum_exp = 0.0f;
    साठी (पूर्णांक i = tid; i < nclasses; i += WARP_SIZE) {
        स्थिरांक दशांश v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        sum_exp += v;
        जर (has_shared) {
            smem[i] = v;
        } अन्यथा {
            dst[i] = v;
        }
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    स्थिरांक दशांश inv_sum = 1.0f / sum_exp;

    स्थिरांक दशांश d_by_nrows = grad[0] / (दशांश) nrows;
    साठी (पूर्णांक i = tid; i < nclasses; i += WARP_SIZE) {
        स्थिरांक दशांश sm_अंक = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_अंक * inv_sum - labels[i]) * d_by_nrows;
    }
}

स्थिर शून्य cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        स्थिरांक दशांश * row_loss,
        दशांश * dst,
        स्थिरांक int64_t nrows) {
    जर (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->स_नकल(dst, row_loss, आकार(दशांश))));
        परत;
    }

    ggml_sycl_pool_alloc<दशांश> tmp_alloc(ctx.pool(), nrows);
    दशांश * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->स_नकल(tmp, row_loss, nrows * आकार(दशांश))));

    int64_t cur = nrows;
    पर्यंत (cur > 1) {
        स्थिरांक int64_t out = (cur + WARP_SIZE - 1) / WARP_SIZE;
        स्थिरांक sycl::range<3> block(1, 1, WARP_SIZE);
        स्थिरांक sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                स्थिरांक पूर्णांक row = item.get_group(2);
                स्थिरांक पूर्णांक tid = item.get_local_id(2);
                स्थिरांक int64_t i = (int64_t) row * WARP_SIZE + tid;
                दशांश v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_sum<WARP_SIZE>(v);
                जर (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->स_नकल(dst, tmp, आकार(दशांश))));
}

शून्य ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिरांक int64_t nclasses = src0->ne[0];
    स्थिरांक int64_t nrows = ggml_nrows(src0);

    स्थिरांक दशांश * logits_d = (स्थिरांक दशांश *) src0->data;
    स्थिरांक दशांश * labels_d = (स्थिरांक दशांश *) src1->data;
    दशांश * dst_d = (दशांश *) dst->data;

    ggml_sycl_pool_alloc<दशांश> row_loss_alloc(ctx.pool(), nrows);
    दशांश * row_loss = row_loss_alloc.get();

    स्थिरांक sycl::range<3> block(1, 1, WARP_SIZE);
    स्थिरांक sycl::range<3> grid(1, 1, nrows);
    स्थिरांक माप_प्रकार nbytes_shared = (माप_प्रकार) nclasses * आकार(दशांश);
    स्थिरांक माप_प्रकार smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    जर (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<दशांश, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32_kernel<सत्य>(
                        logits_d, labels_d, row_loss,
                        (पूर्णांक) nclasses, (पूर्णांक) nrows,
                        get_pointer(smem), item);
                });
        });
    } अन्यथा {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_f32_kernel<असत्य>(
                    logits_d, labels_d, row_loss,
                    (पूर्णांक) nclasses, (पूर्णांक) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

शून्य ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    स्थिरांक ggml_tensor * grad  = dst->src[0];
    स्थिरांक ggml_tensor * src0f = dst->src[1];
    स्थिरांक ggml_tensor * src1f = dst->src[2];

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

    स्थिरांक int64_t nclasses = src0f->ne[0];
    स्थिरांक int64_t nrows = ggml_nrows(src0f);

    स्थिरांक दशांश * grad_d  = (स्थिरांक दशांश *) grad->data;
    स्थिरांक दशांश * logits_d = (स्थिरांक दशांश *) src0f->data;
    स्थिरांक दशांश * labels_d = (स्थिरांक दशांश *) src1f->data;
    दशांश * dst_d = (दशांश *) dst->data;

    स्थिरांक sycl::range<3> block(1, 1, WARP_SIZE);
    स्थिरांक sycl::range<3> grid(1, 1, nrows);
    स्थिरांक माप_प्रकार nbytes_shared = (माप_प्रकार) nclasses * आकार(दशांश);
    स्थिरांक माप_प्रकार smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    जर (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<दशांश, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32_kernel<सत्य>(
                        grad_d, logits_d, labels_d, dst_d,
                        (पूर्णांक) nclasses, (पूर्णांक) nrows,
                        get_pointer(smem), item);
                });
        });
    } अन्यथा {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_back_f32_kernel<असत्य>(
                    grad_d, logits_d, labels_d, dst_d,
                    (पूर्णांक) nclasses, (पूर्णांक) nrows,
                    nullptr, item);
            });
    }
}
