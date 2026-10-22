#شامل "cross_entropy_loss.hpp"

#شامل <cstdint>
#شامل <cmath>

template <منطقی has_shared>
ایستا __dpct_inline__ تهی cross_entropy_loss_f32_kernel(
        ثابت اعشاری * __restrict__ logits,
        ثابت اعشاری * __restrict__ labels,
        اعشاری * __restrict__ row_loss,
        ثابت صحیح nclasses,
        ثابت صحیح nrows,
        اعشاری * __restrict__ smem,
        ثابت sycl::nd_item<3> & item) {

    ثابت صحیح row = item.get_group(2);
    ثابت صحیح tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    اعشاری max_logit = -INFINITY;
    برای (صحیح i = tid; i < nclasses; i += WARP_SIZE) {
        ثابت اعشاری v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        اگر (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    اعشاری sum_exp = 0.0f;
    برای (صحیح i = tid; i < nclasses; i += WARP_SIZE) {
        ثابت اعشاری v = has_shared ? smem[i] : logits[i];
        sum_exp += sycl::exp(v - max_logit);
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    ثابت اعشاری log_sum = sycl::log(sum_exp);

    اعشاری loss = 0.0f;
    برای (صحیح i = tid; i < nclasses; i += WARP_SIZE) {
        ثابت اعشاری v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_sum) * labels[i];
    }
    loss = -warp_reduce_sum<WARP_SIZE>(loss) / (اعشاری) nrows;

    اگر (tid == 0) {
        row_loss[row] = loss;
    }
}

template <منطقی has_shared>
ایستا __dpct_inline__ تهی cross_entropy_loss_back_f32_kernel(
        ثابت اعشاری * __restrict__ grad,
        ثابت اعشاری * __restrict__ logits,
        ثابت اعشاری * __restrict__ labels,
        اعشاری * __restrict__ dst,
        ثابت صحیح nclasses,
        ثابت صحیح nrows,
        اعشاری * __restrict__ smem,
        ثابت sycl::nd_item<3> & item) {

    ثابت صحیح row = item.get_group(2);
    ثابت صحیح tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    اعشاری max_logit = -INFINITY;
    برای (صحیح i = tid; i < nclasses; i += WARP_SIZE) {
        ثابت اعشاری v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        اگر (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    اعشاری sum_exp = 0.0f;
    برای (صحیح i = tid; i < nclasses; i += WARP_SIZE) {
        ثابت اعشاری v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        sum_exp += v;
        اگر (has_shared) {
            smem[i] = v;
        } وگرنه {
            dst[i] = v;
        }
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    ثابت اعشاری inv_sum = 1.0f / sum_exp;

    ثابت اعشاری d_by_nrows = grad[0] / (اعشاری) nrows;
    برای (صحیح i = tid; i < nclasses; i += WARP_SIZE) {
        ثابت اعشاری sm_رقم = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_رقم * inv_sum - labels[i]) * d_by_nrows;
    }
}

ایستا تهی cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        ثابت اعشاری * row_loss,
        اعشاری * dst,
        ثابت int64_t nrows) {
    اگر (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->memcpy(dst, row_loss, اندازه(اعشاری))));
        بازگشت;
    }

    ggml_sycl_pool_alloc<اعشاری> tmp_alloc(ctx.pool(), nrows);
    اعشاری * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(tmp, row_loss, nrows * اندازه(اعشاری))));

    int64_t cur = nrows;
    وقتی (cur > 1) {
        ثابت int64_t out = (cur + WARP_SIZE - 1) / WARP_SIZE;
        ثابت sycl::range<3> block(1, 1, WARP_SIZE);
        ثابت sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ثابت صحیح row = item.get_group(2);
                ثابت صحیح tid = item.get_local_id(2);
                ثابت int64_t i = (int64_t) row * WARP_SIZE + tid;
                اعشاری v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_sum<WARP_SIZE>(v);
                اگر (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(dst, tmp, اندازه(اعشاری))));
}

تهی ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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

    ثابت اعشاری * logits_d = (ثابت اعشاری *) src0->data;
    ثابت اعشاری * labels_d = (ثابت اعشاری *) src1->data;
    اعشاری * dst_d = (اعشاری *) dst->data;

    ggml_sycl_pool_alloc<اعشاری> row_loss_alloc(ctx.pool(), nrows);
    اعشاری * row_loss = row_loss_alloc.get();

    ثابت sycl::range<3> block(1, 1, WARP_SIZE);
    ثابت sycl::range<3> grid(1, 1, nrows);
    ثابت size_t nbytes_shared = (size_t) nclasses * اندازه(اعشاری);
    ثابت size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    اگر (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<اعشاری, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32_kernel<درست>(
                        logits_d, labels_d, row_loss,
                        (صحیح) nclasses, (صحیح) nrows,
                        get_pointer(smem), item);
                });
        });
    } وگرنه {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_f32_kernel<نادرست>(
                    logits_d, labels_d, row_loss,
                    (صحیح) nclasses, (صحیح) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

تهی ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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

    ثابت اعشاری * grad_d  = (ثابت اعشاری *) grad->data;
    ثابت اعشاری * logits_d = (ثابت اعشاری *) src0f->data;
    ثابت اعشاری * labels_d = (ثابت اعشاری *) src1f->data;
    اعشاری * dst_d = (اعشاری *) dst->data;

    ثابت sycl::range<3> block(1, 1, WARP_SIZE);
    ثابت sycl::range<3> grid(1, 1, nrows);
    ثابت size_t nbytes_shared = (size_t) nclasses * اندازه(اعشاری);
    ثابت size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    اگر (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<اعشاری, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32_kernel<درست>(
                        grad_d, logits_d, labels_d, dst_d,
                        (صحیح) nclasses, (صحیح) nrows,
                        get_pointer(smem), item);
                });
        });
    } وگرنه {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_back_f32_kernel<نادرست>(
                    grad_d, logits_d, labels_d, dst_d,
                    (صحیح) nclasses, (صحیح) nrows,
                    nullptr, item);
            });
    }
}
