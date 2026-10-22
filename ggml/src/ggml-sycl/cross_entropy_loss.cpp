#അന്തര്ഭാവ "cross_entropy_loss.hpp"

#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cmath>

template <തർക്കം has_shared>
സ്ഥിതം __dpct_inline__ ശൂന്യം cross_entropy_loss_f32_kernel(
        സ്ഥിരം ദശാംശം * __restrict__ logits,
        സ്ഥിരം ദശാംശം * __restrict__ labels,
        ദശാംശം * __restrict__ row_loss,
        സ്ഥിരം പൂർണ്ണസംഖ്യ nclasses,
        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
        ദശാംശം * __restrict__ smem,
        സ്ഥിരം sycl::nd_item<3> & item) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item.get_group(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;

    ദശാംശം max_logit = -INFINITY;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = tid; i < nclasses; i += WARP_SIZE) {
        സ്ഥിരം ദശാംശം v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        എങ്കിൽ (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    ദശാംശം sum_exp = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = tid; i < nclasses; i += WARP_SIZE) {
        സ്ഥിരം ദശാംശം v = has_shared ? smem[i] : logits[i];
        sum_exp += sycl::exp(v - max_logit);
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    സ്ഥിരം ദശാംശം log_sum = sycl::log(sum_exp);

    ദശാംശം loss = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = tid; i < nclasses; i += WARP_SIZE) {
        സ്ഥിരം ദശാംശം v = has_shared ? smem[i] : logits[i];
        loss += (v - max_logit - log_sum) * labels[i];
    }
    loss = -warp_reduce_sum<WARP_SIZE>(loss) / (ദശാംശം) nrows;

    എങ്കിൽ (tid == 0) {
        row_loss[row] = loss;
    }
}

template <തർക്കം has_shared>
സ്ഥിതം __dpct_inline__ ശൂന്യം cross_entropy_loss_back_f32_kernel(
        സ്ഥിരം ദശാംശം * __restrict__ grad,
        സ്ഥിരം ദശാംശം * __restrict__ logits,
        സ്ഥിരം ദശാംശം * __restrict__ labels,
        ദശാംശം * __restrict__ dst,
        സ്ഥിരം പൂർണ്ണസംഖ്യ nclasses,
        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
        ദശാംശം * __restrict__ smem,
        സ്ഥിരം sycl::nd_item<3> & item) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item.get_group(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item.get_local_id(2);

    logits += (int64_t) row * nclasses;
    labels += (int64_t) row * nclasses;
    dst    += (int64_t) row * nclasses;

    ദശാംശം max_logit = -INFINITY;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = tid; i < nclasses; i += WARP_SIZE) {
        സ്ഥിരം ദശാംശം v = logits[i];
        max_logit = sycl::fmax(max_logit, v);
        എങ്കിൽ (has_shared) {
            smem[i] = v;
        }
    }
    max_logit = warp_reduce_max<WARP_SIZE>(max_logit);

    ദശാംശം sum_exp = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = tid; i < nclasses; i += WARP_SIZE) {
        സ്ഥിരം ദശാംശം v = sycl::exp((has_shared ? smem[i] : logits[i]) - max_logit);
        sum_exp += v;
        എങ്കിൽ (has_shared) {
            smem[i] = v;
        } അല്ലെങ്കിൽ {
            dst[i] = v;
        }
    }
    sum_exp = warp_reduce_sum<WARP_SIZE>(sum_exp);
    സ്ഥിരം ദശാംശം inv_sum = 1.0f / sum_exp;

    സ്ഥിരം ദശാംശം d_by_nrows = grad[0] / (ദശാംശം) nrows;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = tid; i < nclasses; i += WARP_SIZE) {
        സ്ഥിരം ദശാംശം sm_അക്കം = has_shared ? smem[i] : dst[i];
        dst[i] = (sm_അക്കം * inv_sum - labels[i]) * d_by_nrows;
    }
}

സ്ഥിതം ശൂന്യം cross_entropy_reduce_rows(
        ggml_backend_sycl_context & ctx,
        സ്ഥിരം ദശാംശം * row_loss,
        ദശാംശം * dst,
        സ്ഥിരം int64_t nrows) {
    എങ്കിൽ (nrows == 1) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            ctx.stream()->memcpy(dst, row_loss, വലുപ്പം(ദശാംശം))));
        തിരികെ;
    }

    ggml_sycl_pool_alloc<ദശാംശം> tmp_alloc(ctx.pool(), nrows);
    ദശാംശം * tmp = tmp_alloc.get();
    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(tmp, row_loss, nrows * വലുപ്പം(ദശാംശം))));

    int64_t cur = nrows;
    ആകുമ്പോൾ (cur > 1) {
        സ്ഥിരം int64_t out = (cur + WARP_SIZE - 1) / WARP_SIZE;
        സ്ഥിരം sycl::range<3> block(1, 1, WARP_SIZE);
        സ്ഥിരം sycl::range<3> grid(1, 1, out);
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                സ്ഥിരം പൂർണ്ണസംഖ്യ row = item.get_group(2);
                സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item.get_local_id(2);
                സ്ഥിരം int64_t i = (int64_t) row * WARP_SIZE + tid;
                ദശാംശം v = i < cur ? tmp[i] : 0.0f;
                v = warp_reduce_sum<WARP_SIZE>(v);
                എങ്കിൽ (tid == 0) {
                    tmp[row] = v;
                }
            });
        cur = out;
    }

    SYCL_CHECK(CHECK_TRY_ERROR(
        ctx.stream()->memcpy(dst, tmp, വലുപ്പം(ദശാംശം))));
}

ശൂന്യം ggml_sycl_cross_entropy_loss(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    സ്ഥിരം int64_t nclasses = src0->ne[0];
    സ്ഥിരം int64_t nrows = ggml_nrows(src0);

    സ്ഥിരം ദശാംശം * logits_d = (സ്ഥിരം ദശാംശം *) src0->data;
    സ്ഥിരം ദശാംശം * labels_d = (സ്ഥിരം ദശാംശം *) src1->data;
    ദശാംശം * dst_d = (ദശാംശം *) dst->data;

    ggml_sycl_pool_alloc<ദശാംശം> row_loss_alloc(ctx.pool(), nrows);
    ദശാംശം * row_loss = row_loss_alloc.get();

    സ്ഥിരം sycl::range<3> block(1, 1, WARP_SIZE);
    സ്ഥിരം sycl::range<3> grid(1, 1, nrows);
    സ്ഥിരം size_t nbytes_shared = (size_t) nclasses * വലുപ്പം(ദശാംശം);
    സ്ഥിരം size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    എങ്കിൽ (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ദശാംശം, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_f32_kernel<സത്യം>(
                        logits_d, labels_d, row_loss,
                        (പൂർണ്ണസംഖ്യ) nclasses, (പൂർണ്ണസംഖ്യ) nrows,
                        get_pointer(smem), item);
                });
        });
    } അല്ലെങ്കിൽ {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_f32_kernel<അസത്യം>(
                    logits_d, labels_d, row_loss,
                    (പൂർണ്ണസംഖ്യ) nclasses, (പൂർണ്ണസംഖ്യ) nrows,
                    nullptr, item);
            });
    }

    cross_entropy_reduce_rows(ctx, row_loss, dst_d, nrows);
}

ശൂന്യം ggml_sycl_cross_entropy_loss_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    സ്ഥിരം ggml_tensor * grad  = dst->src[0];
    സ്ഥിരം ggml_tensor * src0f = dst->src[1];
    സ്ഥിരം ggml_tensor * src1f = dst->src[2];

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

    സ്ഥിരം int64_t nclasses = src0f->ne[0];
    സ്ഥിരം int64_t nrows = ggml_nrows(src0f);

    സ്ഥിരം ദശാംശം * grad_d  = (സ്ഥിരം ദശാംശം *) grad->data;
    സ്ഥിരം ദശാംശം * logits_d = (സ്ഥിരം ദശാംശം *) src0f->data;
    സ്ഥിരം ദശാംശം * labels_d = (സ്ഥിരം ദശാംശം *) src1f->data;
    ദശാംശം * dst_d = (ദശാംശം *) dst->data;

    സ്ഥിരം sycl::range<3> block(1, 1, WARP_SIZE);
    സ്ഥിരം sycl::range<3> grid(1, 1, nrows);
    സ്ഥിരം size_t nbytes_shared = (size_t) nclasses * വലുപ്പം(ദശാംശം);
    സ്ഥിരം size_t smpbo = ggml_sycl_info().devices[ctx.device].smpbo;

    എങ്കിൽ (nbytes_shared <= smpbo) {
        ctx.stream()->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<ദശാംശം, 1> smem(sycl::range<1>(nclasses), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    cross_entropy_loss_back_f32_kernel<സത്യം>(
                        grad_d, logits_d, labels_d, dst_d,
                        (പൂർണ്ണസംഖ്യ) nclasses, (പൂർണ്ണസംഖ്യ) nrows,
                        get_pointer(smem), item);
                });
        });
    } അല്ലെങ്കിൽ {
        ctx.stream()->parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cross_entropy_loss_back_f32_kernel<അസത്യം>(
                    grad_d, logits_d, labels_d, dst_d,
                    (പൂർണ്ണസംഖ്യ) nclasses, (പൂർണ്ണസംഖ്യ) nrows,
                    nullptr, item);
            });
    }
}
