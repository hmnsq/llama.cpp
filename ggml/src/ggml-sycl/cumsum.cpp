#অন্তর্ভাব "cumsum.hpp"
#অন্তর্ভাব "common.hpp"

#অন্তর্ভাব <algorithm>

#define SYCL_CUMSUM_BLOCK_আকার 256

স্থির __dpct_inline__ দশমিক warp_prefix_inclusive_যোগ_f32(দশমিক x, ধ্রুবক sycl::nd_item<3> & item) {
    ফেরত sycl::inclusive_scan_over_group(item.get_sub_group(), x, sycl::plus<দশমিক>());
}

স্থির শূন্য cumsum_f32_kernel(
        ধ্রুবক দশমিক * __restrict__ src, দশমিক * __restrict__ dst,
        ধ্রুবক int64_t ne00, ধ্রুবক int64_t ne01, ধ্রুবক int64_t ne02, ধ্রুবক int64_t ne03,
        ধ্রুবক int64_t s01, ধ্রুবক int64_t s02, ধ্রুবক int64_t s03,
        ধ্রুবক int64_t  d1, ধ্রুবক int64_t  d2, ধ্রুবক int64_t  d3,
        ধ্রুবক sycl::nd_item<3> & item, দশমিক * smem) {

    ধ্রুবক পূর্ণসংখ্যা tid = item.get_local_id(2);
    ধ্রুবক পূর্ণসংখ্যা block_আকার_2 = item.get_local_range(2);
    ধ্রুবক পূর্ণসংখ্যা lane = tid % WARP_আকার;
    ধ্রুবক পূর্ণসংখ্যা warp = tid / WARP_আকার;
    ধ্রুবক পূর্ণসংখ্যা warps_per_block = block_আকার_2 / WARP_আকার;

    দশমিক * s_vals      = smem;
    দশমিক * s_warp_sums = smem + block_আকার_2;
    দশমিক * s_carry     = smem + block_আকার_2 + warps_per_block;

    যদি (tid == 0) {
        s_carry[0] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    ধ্রুবক int64_t i3 = item.get_group(0);
    ধ্রুবক int64_t i2 = item.get_group(1);
    ধ্রুবক int64_t i1 = item.get_group(2);
    যদি (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        ফেরত;
    }

    ধ্রুবক দশমিক * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    দশমিক       * dst_row = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr পূর্ণসংখ্যা অঙ্ক_unroll = 4;
    দশমিক অস্থায়ী[অঙ্ক_unroll];

    জন্য (int64_t i = 0; i < ne00; i += অঙ্ক_unroll * block_আকার_2) {
        int64_t idx = i + tid * অঙ্ক_unroll;

        অস্থায়ী[0] = (idx < ne00 ? src_row[idx] : 0.0f);
#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 1; j < অঙ্ক_unroll; j++) {
            অস্থায়ী[j] = অস্থায়ী[j - 1];
            যদি (idx + j < ne00) {
                অস্থায়ী[j] += src_row[idx + j];
            }
        }

        দশমিক val = (idx < ne00) ? অস্থায়ী[অঙ্ক_unroll - 1] : 0.0f;

        val = warp_prefix_inclusive_যোগ_f32(val, item);
        s_vals[tid] = val;

        যদি (lane == WARP_আকার - 1) {
            s_warp_sums[warp] = val;
        }
        item.barrier(sycl::access::fence_space::local_space);

        যদি (warp == 0) {
            দশমিক w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            দশমিক inc = warp_prefix_inclusive_যোগ_f32(w, item);
            যদি (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;
            }
            যদি (tid == warps_per_block - 1) {
                s_carry[1] = inc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        দশমিক carry = s_carry[0];
        দশমিক final_offset = s_vals[tid] + s_warp_sums[warp] + carry - অস্থায়ী[অঙ্ক_unroll - 1];

#pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < অঙ্ক_unroll; j++) {
            যদি (idx + j < ne00) {
                dst_row[idx + j] = অস্থায়ী[j] + final_offset;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        যদি (tid == 0) {
            s_carry[0] += s_carry[1];
        }
    }
}

অন্তঃরেখ শূন্য ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুবক দশমিক * src_d = static_cast<ধ্রুবক দশমিক *>(src0->তথ্য);
    দশমিক       * dst_d = static_cast<দশমিক *>(dst->তথ্য);

    ধ্রুবক int64_t ne00 = src0->ne[0];
    ধ্রুবক int64_t ne01 = src0->ne[1];
    ধ্রুবক int64_t ne02 = src0->ne[2];
    ধ্রুবক int64_t ne03 = src0->ne[3];

    ধ্রুবক size_t ts = আকার(দশমিক);
    ধ্রুবক int64_t s01 = src0->nb[1] / ts;
    ধ্রুবক int64_t s02 = src0->nb[2] / ts;
    ধ্রুবক int64_t s03 = src0->nb[3] / ts;
    ধ্রুবক int64_t d1  = dst->nb[1] / ts;
    ধ্রুবক int64_t d2  = dst->nb[2] / ts;
    ধ্রুবক int64_t d3  = dst->nb[3] / ts;

    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_warps = (ne00 + WARP_আকার - 1) / WARP_আকার;
    পূর্ণসংখ্যা block_আকার_2 = অঙ্ক_warps * WARP_আকার;
    block_আকার_2 = std::min(block_আকার_2, SYCL_CUMSUM_BLOCK_আকার);
    ধ্রুবক পূর্ণসংখ্যা warps_per_block = block_আকার_2 / WARP_আকার;
    ধ্রুবক পূর্ণসংখ্যা smem_আকার = block_আকার_2 + warps_per_block + 2;

    ধ্রুবক sycl::range<3> grid(ne03, ne02, ne01);
    ধ্রুবক sycl::range<3> block(1, 1, block_আকার_2);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<দশমিক, 1> smem_acc(sycl::range<1>(smem_আকার), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  item, get_pointer(smem_acc));
            });
    });
}

শূন্য ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
