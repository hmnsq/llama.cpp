#ଅନ୍ତର୍ଭାଵ "cumsum.hpp"
#ଅନ୍ତର୍ଭାଵ "common.hpp"

#ଅନ୍ତର୍ଭାଵ <algorithm>

#define SYCL_CUMSUM_BLOCK_SIZE 256

ସ୍ଥିର __dpct_inline__ ଦଶମିକ warp_prefix_inclusive_sum_f32(ଦଶମିକ x, ଅଚଳ sycl::nd_item<3> & item) {
    ଫେରନ୍ତୁ sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<ଦଶମିକ>());
}

ସ୍ଥିର ଶୂନ୍ୟ cumsum_f32_kernel(
        ଅଚଳ ଦଶମିକ * __restrict__ src, ଦଶମିକ * __restrict__ dst,
        ଅଚଳ int64_t ne00, ଅଚଳ int64_t ne01, ଅଚଳ int64_t ne02, ଅଚଳ int64_t ne03,
        ଅଚଳ int64_t s01, ଅଚଳ int64_t s02, ଅଚଳ int64_t s03,
        ଅଚଳ int64_t  d1, ଅଚଳ int64_t  d2, ଅଚଳ int64_t  d3,
        ଅଚଳ sycl::nd_item<3> & item, ଦଶମିକ * smem) {

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid = item.get_local_id(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size = item.get_local_range(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane = tid % WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp = tid / WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା warps_per_block = block_size / WARP_SIZE;

    ଦଶମିକ * s_vals      = smem;
    ଦଶମିକ * s_warp_sums = smem + block_size;
    ଦଶମିକ * s_carry     = smem + block_size + warps_per_block;

    ଯଦି (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    ଅଚଳ int64_t i3 = item.get_group(0);
    ଅଚଳ int64_t i2 = item.get_group(1);
    ଅଚଳ int64_t i1 = item.get_group(2);
    ଯଦି (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ଦଶମିକ * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    ଦଶମିକ       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_unroll = 4;
    ଦଶମିକ temp[ଅଙ୍କ_unroll];

    ପାଇଁ (int64_t i = 0; i < ne00; i += ଅଙ୍କ_unroll * block_size) {
        int64_t idx = i + tid * ଅଙ୍କ_unroll;

        temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 1; j < ଅଙ୍କ_unroll; j++) {
            temp[j] = temp[j - 1];
            ଯଦି (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        ଦଶମିକ val = (idx < ne00) ? temp[ଅଙ୍କ_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_sum_f32(val, item);
        s_vals[tid] = val;

        ଯଦି (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        ଯଦି (warp == 0) {
            ଦଶମିକ w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            ଦଶମିକ inc = warp_prefix_inclusive_sum_f32(w, item);
            ଯଦି (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            ଯଦି (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        ଦଶମିକ carry = s_carry[0];
        ଦଶମିକ final_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[ଅଙ୍କ_unroll - 1];

#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ଅଙ୍କ_unroll; j++) {
            ଯଦି (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        ଯଦି (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ଅଚଳ ଦଶମିକ * src_d = static_cast<ଅଚଳ ଦଶମିକ *>(src0->data);
    ଦଶମିକ       * dst_d = static_cast<ଦଶମିକ *>(dst->data);

    ଅଚଳ int64_t ne00 = src0->ne[0];
    ଅଚଳ int64_t ne01 = src0->ne[1];
    ଅଚଳ int64_t ne02 = src0->ne[2];
    ଅଚଳ int64_t ne03 = src0->ne[3];

    ଅଚଳ size_t ts = ଆକାର(ଦଶମିକ);
    ଅଚଳ int64_t s01 = src0->nb[1] / ts;
    ଅଚଳ int64_t s02 = src0->nb[2] / ts;
    ଅଚଳ int64_t s03 = src0->nb[3] / ts;
    ଅଚଳ int64_t d1  = dst->nb[1] / ts;
    ଅଚଳ int64_t d2  = dst->nb[2] / ts;
    ଅଚଳ int64_t d3  = dst->nb[3] / ts;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size = ଅଙ୍କ_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା warps_per_block = block_size / WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା smem_size = block_size + warps_per_block + 2;

    ଅଚଳ sycl::range<3> grid(ne03, ne02, ne01);
    ଅଚଳ sycl::range<3> block(1, 1, block_size);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ଦଶମିକ, 1> smem_acc(sycl::range<1>(smem_size), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

ଶୂନ୍ୟ ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
