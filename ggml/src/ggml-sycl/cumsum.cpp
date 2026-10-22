#ಅನ್ತರ್ಭಾವ "cumsum.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>

#define SYCL_CUMSUM_BLOCK_SIZE 256

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ warp_prefix_inclusive_sum_f32(ಭಿನ್ನ x, ಧ್ರುವ sycl::nd_item<3> & item) {
    ಪ್ರತ್ಯಾಗಮ sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<ಭಿನ್ನ>());
}

ಸ್ಥಿರ ಶೂನ್ಯ cumsum_f32_kernel(
        ಧ್ರುವ ಭಿನ್ನ * __restrict__ src, ಭಿನ್ನ * __restrict__ dst,
        ಧ್ರುವ int64_t ne00, ಧ್ರುವ int64_t ne01, ಧ್ರುವ int64_t ne02, ಧ್ರುವ int64_t ne03,
        ಧ್ರುವ int64_t s01, ಧ್ರುವ int64_t s02, ಧ್ರುವ int64_t s03,
        ಧ್ರುವ int64_t  d1, ಧ್ರುವ int64_t  d2, ಧ್ರುವ int64_t  d3,
        ಧ್ರುವ sycl::nd_item<3> & item, ಭಿನ್ನ * smem) {

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tid = item.get_local_id(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_size = item.get_local_range(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lane = tid % WARP_SIZE;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ warp = tid / WARP_SIZE;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ warps_per_block = block_size / WARP_SIZE;

    ಭಿನ್ನ * s_vals      = smem;
    ಭಿನ್ನ * s_warp_sums = smem + block_size;
    ಭಿನ್ನ * s_carry     = smem + block_size + warps_per_block;

    ಯದಿ (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    ಧ್ರುವ int64_t i3 = item.get_group(0);
    ಧ್ರುವ int64_t i2 = item.get_group(1);
    ಧ್ರುವ int64_t i1 = item.get_group(2);
    ಯದಿ (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಭಿನ್ನ * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    ಭಿನ್ನ       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_unroll = 4;
    ಭಿನ್ನ temp[ಅಂಕೆ_unroll];

    ಪ್ರತಿ (int64_t i = 0; i < ne00; i += ಅಂಕೆ_unroll * block_size) {
        int64_t idx = i + tid * ಅಂಕೆ_unroll;

        temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 1; j < ಅಂಕೆ_unroll; j++) {
            temp[j] = temp[j - 1];
            ಯದಿ (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        ಭಿನ್ನ val = (idx < ne00) ? temp[ಅಂಕೆ_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_sum_f32(val, item);
        s_vals[tid] = val;

        ಯದಿ (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        ಯದಿ (warp == 0) {
            ಭಿನ್ನ w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            ಭಿನ್ನ inc = warp_prefix_inclusive_sum_f32(w, item);
            ಯದಿ (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            ಯದಿ (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        ಭಿನ್ನ carry = s_carry[0];
        ಭಿನ್ನ final_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[ಅಂಕೆ_unroll - 1];

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ಅಂಕೆ_unroll; j++) {
            ಯದಿ (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        ಯದಿ (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ಧ್ರುವ ಭಿನ್ನ * src_d = static_cast<ಧ್ರುವ ಭಿನ್ನ *>(src0->data);
    ಭಿನ್ನ       * dst_d = static_cast<ಭಿನ್ನ *>(dst->data);

    ಧ್ರುವ int64_t ne00 = src0->ne[0];
    ಧ್ರುವ int64_t ne01 = src0->ne[1];
    ಧ್ರುವ int64_t ne02 = src0->ne[2];
    ಧ್ರುವ int64_t ne03 = src0->ne[3];

    ಧ್ರುವ size_t ts = ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ int64_t s01 = src0->nb[1] / ts;
    ಧ್ರುವ int64_t s02 = src0->nb[2] / ts;
    ಧ್ರುವ int64_t s03 = src0->nb[3] / ts;
    ಧ್ರುವ int64_t d1  = dst->nb[1] / ts;
    ಧ್ರುವ int64_t d2  = dst->nb[2] / ts;
    ಧ್ರುವ int64_t d3  = dst->nb[3] / ts;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    ಪೂರ್ಣಾಙ್ಕ block_size = ಅಂಕೆ_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ warps_per_block = block_size / WARP_SIZE;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ smem_size = block_size + warps_per_block + 2;

    ಧ್ರುವ sycl::range<3> grid(ne03, ne02, ne01);
    ಧ್ರುವ sycl::range<3> block(1, 1, block_size);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<ಭಿನ್ನ, 1> smem_acc(sycl::range<1>(smem_size), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

ಶೂನ್ಯ ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
