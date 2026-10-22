#ضمن "cross_entropy_loss.hpp"

#ضمن <cstdint>
#ضمن <cmath>

template <منطقي has_shared>
ساكن __dpct_inline__ فراغ cross_entropy_loss_f32_kernel(
        ثابت عائم * __restrict__ logits,
        ثابت عائم * __restrict__ labels,
        عائم * __restrict__ row_loss,
        ثابت صحيح nclasses,
        ثابت صحيح nrows,
        عائم * __restrict__ smem,
        ثابت sycl::nd_item<3> & item) {

    ثابت صحيح row = item.get_group(2);
    ثابت صحيح tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    عائم max_logit = -INFINITY;
    لكل (صحيح i = tid; i < nclasses; i += WARP_SIZE) {
        ثابت عائم v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        إذا (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    عائم sum_exp = 0.0f;
    لكل (صحيح i = tid; i < nclasses; i += WARP_SIZE) {
        ثابت عائم v = has_shared ? smem[i] : logits[i];
        sum_exp += sycl::exp(v - max_logit);
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    ثابت عائم log_sum = sycl::log(sum_exp);

    عائم loss = 0.0f;
    لكل (صحيح i = tid; i < nclasses; i += WARP_SIZE) {
        ثابت عائم v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_sum) * labels[i];
    }
    loss = -warp_reduce_sum<WARP_SIZE>(loss) / (عائم) nrows;

    إذا (tid == 0) {
        row_loss[row] = loss;
    }
}

template <منطقي has_shared>
ساكن __dpct_inline__ فراغ cross_entropy_loss_back_f32_kernel(
        ثابت عائم * __restrict__ grad,
        ثابت عائم * __restrict__ logits,
        ثابت عائم * __restrict__ labels,
        عائم * __restrict__ dst,
        ثابت صحيح nclasses,
        ثابت صحيح nrows,
        عائم * __restrict__ smem,
        ثابت sycl::nd_item<3> & item) {

    ثابت صحيح row = item.get_group(2);
    ثابت صحيح tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    عائم max_logit = -INFINITY;
    لكل (صحيح i = tid; i < nclasses; i += WARP_SIZE) {
        ثابت عائم v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        إذا (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    عائم sum_exp = 0.0f;
    لكل (صحيح i = tid; i < nclasses; i += WARP_SIZE) {
        ثابت عائم v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        sum_exp += v;
        إذا (has_shared) {
            smem[i] = v;
        } وإلا {
            dst[i] = v;
        }
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    ثابت عائم inv_sum = 1.0f / sum_exp;

    ثابت عائم d_by_nrows = grad[0] / (عائم) nrows;
    لكل (صحيح i = tid; i < nclasses; i += WARP_SIZE) {
        ثابت عائم sm_اڠک = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_اڠک * inv_sum - labels[i]) * d_by_nrows;
    }
}

ساكن فراغ cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        ثابت عائم * row_loss,
        عائم * dst,
        ثابت int64_t nrows) {
    إذا (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->memcpy(dst, row_loss, حجم(عائم))));
        أرجع;
    }

    ggml_sycl_pool_alloc<عائم> tmp_alloc(ctx.pool(), nrows);
    عائم * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(tmp, row_loss, nrows * حجم(عائم))));

    int64_t cur = nrows;
    طالما (cur > 1) {
        ثابت int64_t out = (cur + WARP_SIZE - 1) / WARP_SIZE;
        ثابت sycl::range<3> block(1, 1, WARP_SIZE);
        ثابت sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ثابت صحيح row = item.get_group(2);
                ثابت صحيح tid = item.get_local_id(2);
                ثابت int64_t i = (int64_t) row * WARP_SIZE + tid;
                عائم v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_sum<WARP_SIZE>(v);
                إذا (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(dst, tmp, حجم(عائم))));
}

فراغ ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ثابت int64_t nclasses = src0->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0);

    ثابت عائم * logits_d = (ثابت عائم *) src0->data;
    ثابت عائم * labels_d = (ثابت عائم *) src1->data;
    عائم * dst_d = (عائم *) dst->data;

    ggml_sycl_pool_alloc<عائم> row_loss_alloc(ctx.pool(), nrows);
    عائم * row_loss = row_loss_alloc.get();

    ثابت sycl::range<3> block(1, 1, WARP_SIZE);
    ثابت sycl::range<3> grid(1, 1, nrows);
    ثابت size_t nbytes_shared = (size_t) nclasses * حجم(عائم);
    ثابت size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    إذا (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<عائم, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32_kernel<true>(
                        logits_d, labels_d, row_loss,
                        (صحيح) nclasses, (صحيح) nrows,
                        get_pointer(smem), item);
                });
        });
    } وإلا {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_f32_kernel<خطأ>(
                    logits_d, labels_d, row_loss,
                    (صحيح) nclasses, (صحيح) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

فراغ ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ثابت ggml_tensor * grad  = dst->src[0];
    ثابت ggml_tensor * src0f = dst->src[1];
    ثابت ggml_tensor * src1f = dst->src[2];

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

    ثابت int64_t nclasses = src0f->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0f);

    ثابت عائم * grad_d  = (ثابت عائم *) grad->data;
    ثابت عائم * logits_d = (ثابت عائم *) src0f->data;
    ثابت عائم * labels_d = (ثابت عائم *) src1f->data;
    عائم * dst_d = (عائم *) dst->data;

    ثابت sycl::range<3> block(1, 1, WARP_SIZE);
    ثابت sycl::range<3> grid(1, 1, nrows);
    ثابت size_t nbytes_shared = (size_t) nclasses * حجم(عائم);
    ثابت size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    إذا (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<عائم, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32_kernel<true>(
                        grad_d, logits_d, labels_d, dst_d,
                        (صحيح) nclasses, (صحيح) nrows,
                        get_pointer(smem), item);
                });
        });
    } وإلا {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_back_f32_kernel<خطأ>(
                    grad_d, logits_d, labels_d, dst_d,
                    (صحيح) nclasses, (صحيح) nrows,
                    nullptr, item);
            });
    }
}
