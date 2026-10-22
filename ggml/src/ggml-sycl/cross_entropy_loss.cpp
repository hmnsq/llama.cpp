#অন্তর্ভাব "cross_entropy_loss.hpp"

#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cmath>

template <যৌক্তিক has_shared>
স্থির __dpct_inline__ শূন্য cross_entropy_loss_f32_kernel(
        ধ্রুবক দশমিক * __restrict__ logits,
        ধ্রুবক দশমিক * __restrict__ labels,
        দশমিক * __restrict__ row_loss,
        ধ্রুবক পূর্ণসংখ্যা nclasses,
        ধ্রুবক পূর্ণসংখ্যা nrows,
        দশমিক * __restrict__ smem,
        ধ্রুবক sycl::nd_item<3> & item) {

    ধ্রুবক পূর্ণসংখ্যা row = item.get_group(2);
    ধ্রুবক পূর্ণসংখ্যা tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    দশমিক max_logit = -INFINITY;
    জন্য (পূর্ণসংখ্যা i = tid; i < nclasses; i += WARP_আকার) {
        ধ্রুবক দশমিক v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        যদি (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_আকার>(max_logit);

    দশমিক যোগ_exp = 0.0f;
    জন্য (পূর্ণসংখ্যা i = tid; i < nclasses; i += WARP_আকার) {
        ধ্রুবক দশমিক v = has_shared ? smem[i] : logits[i];
        যোগ_exp += sycl::exp(v - max_logit);
    }
    যোগ_exp = warp_reduce_যোগ<WARP_আকার>(যোগ_exp);
    ধ্রুবক দশমিক log_যোগ = sycl::log(যোগ_exp);

    দশমিক loss = 0.0f;
    জন্য (পূর্ণসংখ্যা i = tid; i < nclasses; i += WARP_আকার) {
        ধ্রুবক দশমিক v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_যোগ) * labels[i];
    }
    loss = -warp_reduce_যোগ<WARP_আকার>(loss) / (দশমিক) nrows;

    যদি (tid == 0) {
        row_loss[row] = loss;
    }
}

template <যৌক্তিক has_shared>
স্থির __dpct_inline__ শূন্য cross_entropy_loss_back_f32_kernel(
        ধ্রুবক দশমিক * __restrict__ grad,
        ধ্রুবক দশমিক * __restrict__ logits,
        ধ্রুবক দশমিক * __restrict__ labels,
        দশমিক * __restrict__ dst,
        ধ্রুবক পূর্ণসংখ্যা nclasses,
        ধ্রুবক পূর্ণসংখ্যা nrows,
        দশমিক * __restrict__ smem,
        ধ্রুবক sycl::nd_item<3> & item) {

    ধ্রুবক পূর্ণসংখ্যা row = item.get_group(2);
    ধ্রুবক পূর্ণসংখ্যা tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    দশমিক max_logit = -INFINITY;
    জন্য (পূর্ণসংখ্যা i = tid; i < nclasses; i += WARP_আকার) {
        ধ্রুবক দশমিক v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        যদি (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_আকার>(max_logit);

    দশমিক যোগ_exp = 0.0f;
    জন্য (পূর্ণসংখ্যা i = tid; i < nclasses; i += WARP_আকার) {
        ধ্রুবক দশমিক v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        যোগ_exp += v;
        যদি (has_shared) {
            smem[i] = v;
        } অন্যথা {
            dst[i] = v;
        }
    }
    যোগ_exp = warp_reduce_যোগ<WARP_আকার>(যোগ_exp);
    ধ্রুবক দশমিক inv_যোগ = 1.0f / যোগ_exp;

    ধ্রুবক দশমিক d_by_nrows = grad[0] / (দশমিক) nrows;
    জন্য (পূর্ণসংখ্যা i = tid; i < nclasses; i += WARP_আকার) {
        ধ্রুবক দশমিক sm_অঙ্ক = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_অঙ্ক * inv_যোগ - labels[i]) * d_by_nrows;
    }
}

স্থির শূন্য cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        ধ্রুবক দশমিক * row_loss,
        দশমিক * dst,
        ধ্রুবক int64_t nrows) {
    যদি (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->memcpy(dst, row_loss, আকার(দশমিক))));
        ফেরত;
    }

    ggml_sycl_pool_alloc<দশমিক> tmp_alloc(ctx.pool(), nrows);
    দশমিক * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(tmp, row_loss, nrows * আকার(দশমিক))));

    int64_t cur = nrows;
    যখন (cur > 1) {
        ধ্রুবক int64_t out = (cur + WARP_আকার - 1) / WARP_আকার;
        ধ্রুবক sycl::range<3> block(1, 1, WARP_আকার);
        ধ্রুবক sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                ধ্রুবক পূর্ণসংখ্যা row = item.get_group(2);
                ধ্রুবক পূর্ণসংখ্যা tid = item.get_local_id(2);
                ধ্রুবক int64_t i = (int64_t) row * WARP_আকার + tid;
                দশমিক v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_যোগ<WARP_আকার>(v);
                যদি (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(dst, tmp, আকার(দশমিক))));
}

শূন্য ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ধ্রুবক ggml_tensor * src0 = dst->src[0];
    ধ্রুবক ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুবক int64_t nclasses = src0->ne[0];
    ধ্রুবক int64_t nrows = ggml_nrows(src0);

    ধ্রুবক দশমিক * logits_d = (ধ্রুবক দশমিক *) src0->তথ্য;
    ধ্রুবক দশমিক * labels_d = (ধ্রুবক দশমিক *) src1->তথ্য;
    দশমিক * dst_d = (দশমিক *) dst->তথ্য;

    ggml_sycl_pool_alloc<দশমিক> row_loss_alloc(ctx.pool(), nrows);
    দশমিক * row_loss = row_loss_alloc.get();

    ধ্রুবক sycl::range<3> block(1, 1, WARP_আকার);
    ধ্রুবক sycl::range<3> grid(1, 1, nrows);
    ধ্রুবক size_t nbytes_shared = (size_t) nclasses * আকার(দশমিক);
    ধ্রুবক size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    যদি (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<দশমিক, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    cross_entropy_loss_f32_kernel<সত্য>(
                        logits_d, labels_d, row_loss,
                        (পূর্ণসংখ্যা) nclasses, (পূর্ণসংখ্যা) nrows,
                        get_pointer(smem), item);
                });
        });
    } অন্যথা {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                cross_entropy_loss_f32_kernel<মিথ্যা>(
                    logits_d, labels_d, row_loss,
                    (পূর্ণসংখ্যা) nclasses, (পূর্ণসংখ্যা) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

শূন্য ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ধ্রুবক ggml_tensor * grad  = dst->src[0];
    ধ্রুবক ggml_tensor * src0f = dst->src[1];
    ধ্রুবক ggml_tensor * src1f = dst->src[2];

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

    ধ্রুবক int64_t nclasses = src0f->ne[0];
    ধ্রুবক int64_t nrows = ggml_nrows(src0f);

    ধ্রুবক দশমিক * grad_d  = (ধ্রুবক দশমিক *) grad->তথ্য;
    ধ্রুবক দশমিক * logits_d = (ধ্রুবক দশমিক *) src0f->তথ্য;
    ধ্রুবক দশমিক * labels_d = (ধ্রুবক দশমিক *) src1f->তথ্য;
    দশমিক * dst_d = (দশমিক *) dst->তথ্য;

    ধ্রুবক sycl::range<3> block(1, 1, WARP_আকার);
    ধ্রুবক sycl::range<3> grid(1, 1, nrows);
    ধ্রুবক size_t nbytes_shared = (size_t) nclasses * আকার(দশমিক);
    ধ্রুবক size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    যদি (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<দশমিক, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    cross_entropy_loss_back_f32_kernel<সত্য>(
                        grad_d, logits_d, labels_d, dst_d,
                        (পূর্ণসংখ্যা) nclasses, (পূর্ণসংখ্যা) nrows,
                        get_pointer(smem), item);
                });
        });
    } অন্যথা {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                cross_entropy_loss_back_f32_kernel<মিথ্যা>(
                    grad_d, logits_d, labels_d, dst_d,
                    (পূর্ণসংখ্যা) nclasses, (পূর্ণসংখ্যা) nrows,
                    nullptr, item);
            });
    }
}
