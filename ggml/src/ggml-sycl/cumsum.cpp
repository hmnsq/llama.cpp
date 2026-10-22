#अन्तर्भाव "cumsum.hpp"
#अन्तर्भाव "common.hpp"

#अन्तर्भाव <algorithm>

#define SYCL_CUMSUM_BLOCK_SIZE 256

स्थिर __dpct_inline__ दशांश warp_prefix_inclusive_sum_f32(दशांश x, स्थिरांक sycl::nd_item<3> & item) {
    परत sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<दशांश>());
}

स्थिर शून्य cumsum_f32_kernel(
        स्थिरांक दशांश * __restrict__ src, दशांश * __restrict__ dst,
        स्थिरांक int64_t ne00, स्थिरांक int64_t ne01, स्थिरांक int64_t ne02, स्थिरांक int64_t ne03,
        स्थिरांक int64_t s01, स्थिरांक int64_t s02, स्थिरांक int64_t s03,
        स्थिरांक int64_t  d1, स्थिरांक int64_t  d2, स्थिरांक int64_t  d3,
        स्थिरांक sycl::nd_item<3> & item, दशांश * smem) {

    स्थिरांक पूर्णांक tid = item.get_local_id(2);
    स्थिरांक पूर्णांक block_size = item.get_local_range(2);
    स्थिरांक पूर्णांक lane = tid % WARP_SIZE;
    स्थिरांक पूर्णांक warp = tid / WARP_SIZE;
    स्थिरांक पूर्णांक warps_per_block = block_size / WARP_SIZE;

    दशांश * s_vals      = smem;
    दशांश * s_warp_sums = smem + block_size;
    दशांश * s_carry     = smem + block_size + warps_per_block;

    जर (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    स्थिरांक int64_t i3 = item.get_group(0);
    स्थिरांक int64_t i2 = item.get_group(1);
    स्थिरांक int64_t i1 = item.get_group(2);
    जर (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        परत;
    }

    स्थिरांक दशांश * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    दशांश       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr पूर्णांक अंक_unroll = 4;
    दशांश temp[अंक_unroll];

    साठी (int64_t i = 0; i < ne00; i += अंक_unroll * block_size) {
        int64_t idx = i + tid * अंक_unroll;

        temp[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        साठी (पूर्णांक j = 1; j < अंक_unroll; j++) {
            temp[j] = temp[j - 1];
            जर (idx + j < ne00) {
                temp[j] += src_row[idx + j];
            }
        }

        दशांश val = (idx < ne00) ? temp[अंक_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_sum_f32(val, item);
        s_vals[tid] = val;

        जर (lane == WARP_SIZE - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        जर (warp == 0) {
            दशांश w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            दशांश inc = warp_prefix_inclusive_sum_f32(w, item);
            जर (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            जर (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        दशांश carry = s_carry[0];
        दशांश final_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[अंक_unroll - 1];

#pragma unroll
        साठी (पूर्णांक j = 0; j < अंक_unroll; j++) {
            जर (idx + j < ne00) {
                dst_row[idx + j] = temp[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        जर (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

ओळीत शून्य ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिरांक दशांश * src_d = static_cast<स्थिरांक दशांश *>(src0->data);
    दशांश       * dst_d = static_cast<दशांश *>(dst->data);

    स्थिरांक int64_t ne00 = src0->ne[0];
    स्थिरांक int64_t ne01 = src0->ne[1];
    स्थिरांक int64_t ne02 = src0->ne[2];
    स्थिरांक int64_t ne03 = src0->ne[3];

    स्थिरांक माप_प्रकार ts = आकार(दशांश);
    स्थिरांक int64_t s01 = src0->nb[1] / ts;
    स्थिरांक int64_t s02 = src0->nb[2] / ts;
    स्थिरांक int64_t s03 = src0->nb[3] / ts;
    स्थिरांक int64_t d1  = dst->nb[1] / ts;
    स्थिरांक int64_t d2  = dst->nb[2] / ts;
    स्थिरांक int64_t d3  = dst->nb[3] / ts;

    स्थिरांक पूर्णांक अंक_warps = (ne00 + WARP_SIZE - 1) / WARP_SIZE;
    पूर्णांक block_size = अंक_warps * WARP_SIZE;
    block_size = std::min(block_size, SYCL_CUMSUM_BLOCK_SIZE);
    स्थिरांक पूर्णांक warps_per_block = block_size / WARP_SIZE;
    स्थिरांक पूर्णांक smem_size = block_size + warps_per_block + 2;

    स्थिरांक sycl::range<3> grid(ne03, ne02, ne01);
    स्थिरांक sycl::range<3> block(1, 1, block_size);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<दशांश, 1> smem_acc(sycl::range<1>(smem_size), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

शून्य ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
