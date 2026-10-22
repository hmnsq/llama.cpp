#অন্তর্ভাব "cross_entropy_loss.hpp"

#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cmath>

template <তার্কিক has_shared>
স্থির __dpct_inline__ শূন্য cross_entropy_loss_f32_kernel(
        ধ্রুব ভিন্ন * __restrict__ logits,
        ধ্রুব ভিন্ন * __restrict__ labels,
        ভিন্ন * __restrict__ row_loss,
        ধ্রুব পূর্ণাঙ্ক nclasses,
        ধ্রুব পূর্ণাঙ্ক nrows,
        ভিন্ন * __restrict__ smem,
        ধ্রুব sycl::nd_item<3> & item) {

    ধ্রুব পূর্ণাঙ্ক row = item.get_group(2);
    ধ্রুব পূর্ণাঙ্ক tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    ভিন্ন max_logit = -INFINITY;
    প্রতি (পূর্ণাঙ্ক i = tid; i < nclasses; i += WARP_SIZE) {
        ধ্রুব ভিন্ন v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        যদি (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    ভিন্ন sum_exp = 0.0f;
    প্রতি (পূর্ণাঙ্ক i = tid; i < nclasses; i += WARP_SIZE) {
        ধ্রুব ভিন্ন v = has_shared ? smem[i] : logits[i];
        sum_exp += sycl::exp(v - max_logit);
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    ধ্রুব ভিন্ন log_sum = sycl::log(sum_exp);

    ভিন্ন loss = 0.0f;
    প্রতি (পূর্ণাঙ্ক i = tid; i < nclasses; i += WARP_SIZE) {
        ধ্রুব ভিন্ন v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_sum) * labels[i];
    }
    loss = -warp_reduce_sum<WARP_SIZE>(loss) / (ভিন্ন) nrows;

    যদি (tid == 0) {
        row_loss[row] = loss;
    }
}

template <তার্কিক has_shared>
স্থির __dpct_inline__ শূন্য cross_entropy_loss_back_f32_kernel(
        ধ্রুব ভিন্ন * __restrict__ grad,
        ধ্রুব ভিন্ন * __restrict__ logits,
        ধ্রুব ভিন্ন * __restrict__ labels,
        ভিন্ন * __restrict__ dst,
        ধ্রুব পূর্ণাঙ্ক nclasses,
        ধ্রুব পূর্ণাঙ্ক nrows,
        ভিন্ন * __restrict__ smem,
        ধ্রুব sycl::nd_item<3> & item) {

    ধ্রুব পূর্ণাঙ্ক row = item.get_group(2);
    ধ্রুব পূর্ণাঙ্ক tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    ভিন্ন max_logit = -INFINITY;
    প্রতি (পূর্ণাঙ্ক i = tid; i < nclasses; i += WARP_SIZE) {
        ধ্রুব ভিন্ন v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        যদি (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    ভিন্ন sum_exp = 0.0f;
    প্রতি (পূর্ণাঙ্ক i = tid; i < nclasses; i += WARP_SIZE) {
        ধ্রুব ভিন্ন v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        sum_exp += v;
        যদি (has_shared) {
            smem[i] = v;
        } অন্যথা {
            dst[i] = v;
        }
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    ধ্রুব ভিন্ন inv_sum = 1.0f / sum_exp;

    ধ্রুব ভিন্ন d_by_nrows = grad[0] / (ভিন্ন) nrows;
    প্রতি (পূর্ণাঙ্ক i = tid; i < nclasses; i += WARP_SIZE) {
        ধ্রুব ভিন্ন sm_অঙ্ক = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_অঙ্ক * inv_sum - labels[i]) * d_by_nrows;
    }
}

স্থির শূন্য cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        ধ্রুব ভিন্ন * row_loss,
        ভিন্ন * dst,
        ধ্রুব int64_t nrows) {
    যদি (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->memcpy(dst, row_loss, পরিমাণ(ভিন্ন))));
        প্রত্যাগম;
    }

    ggml_sycl_pool_alloc<ভিন্ন> tmp_alloc(ctx.pool(), nrows);
    ভিন্ন * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(tmp, row_loss, nrows * পরিমাণ(ভিন্ন))));

    int64_t cur = nrows;
    যাবত্ (cur > 1) {
        ধ্রুব int64_t out = (cur + WARP_SIZE - 1) / WARP_SIZE;
        ধ্রুব sycl::range<3> block(1, 1, WARP_SIZE);
        ধ্রুব sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ধ্রুব পূর্ণাঙ্ক row = item.get_group(2);
                ধ্রুব পূর্ণাঙ্ক tid = item.get_local_id(2);
                ধ্রুব int64_t i = (int64_t) row * WARP_SIZE + tid;
                ভিন্ন v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_sum<WARP_SIZE>(v);
                যদি (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(dst, tmp, পরিমাণ(ভিন্ন))));
}

শূন্য ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুব int64_t nclasses = src0->ne[0];
    ধ্রুব int64_t nrows = ggml_nrows(src0);

    ধ্রুব ভিন্ন * logits_d = (ধ্রুব ভিন্ন *) src0->data;
    ধ্রুব ভিন্ন * labels_d = (ধ্রুব ভিন্ন *) src1->data;
    ভিন্ন * dst_d = (ভিন্ন *) dst->data;

    ggml_sycl_pool_alloc<ভিন্ন> row_loss_alloc(ctx.pool(), nrows);
    ভিন্ন * row_loss = row_loss_alloc.get();

    ধ্রুব sycl::range<3> block(1, 1, WARP_SIZE);
    ধ্রুব sycl::range<3> grid(1, 1, nrows);
    ধ্রুব size_t nbytes_shared = (size_t) nclasses * পরিমাণ(ভিন্ন);
    ধ্রুব size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    যদি (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ভিন্ন, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32_kernel<সত্য>(
                        logits_d, labels_d, row_loss,
                        (পূর্ণাঙ্ক) nclasses, (পূর্ণাঙ্ক) nrows,
                        get_pointer(smem), item);
                });
        });
    } অন্যথা {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_f32_kernel<অসত্য>(
                    logits_d, labels_d, row_loss,
                    (পূর্ণাঙ্ক) nclasses, (পূর্ণাঙ্ক) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

শূন্য ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ধ্রুব ggml_tensor * grad  = dst->src[0];
    ধ্রুব ggml_tensor * src0f = dst->src[1];
    ধ্রুব ggml_tensor * src1f = dst->src[2];

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

    ধ্রুব int64_t nclasses = src0f->ne[0];
    ধ্রুব int64_t nrows = ggml_nrows(src0f);

    ধ্রুব ভিন্ন * grad_d  = (ধ্রুব ভিন্ন *) grad->data;
    ধ্রুব ভিন্ন * logits_d = (ধ্রুব ভিন্ন *) src0f->data;
    ধ্রুব ভিন্ন * labels_d = (ধ্রুব ভিন্ন *) src1f->data;
    ভিন্ন * dst_d = (ভিন্ন *) dst->data;

    ধ্রুব sycl::range<3> block(1, 1, WARP_SIZE);
    ধ্রুব sycl::range<3> grid(1, 1, nrows);
    ধ্রুব size_t nbytes_shared = (size_t) nclasses * পরিমাণ(ভিন্ন);
    ধ্রুব size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    যদি (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ভিন্ন, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32_kernel<সত্য>(
                        grad_d, logits_d, labels_d, dst_d,
                        (পূর্ণাঙ্ক) nclasses, (পূর্ণাঙ্ক) nrows,
                        get_pointer(smem), item);
                });
        });
    } অন্যথা {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_back_f32_kernel<অসত্য>(
                    grad_d, logits_d, labels_d, dst_d,
                    (পূর্ণাঙ্ক) nclasses, (পূর্ণাঙ্ক) nrows,
                    nullptr, item);
            });
    }
}
