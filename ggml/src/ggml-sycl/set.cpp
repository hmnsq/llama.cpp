#অন্তর্ভাব "presets.hpp"
#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "set.hpp"
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <sycl/sycl.hpp>
using নামাকাশ sycl;

// Internal function: perform element-wise set operation for each thread
অন্তঃরেখ শূন্য set_f32(ধ্রুবক দশমিক* src, দশমিক* dst,
                    ধ্রুবক int64_t ne0, ধ্রুবক int64_t ne1,
                    ধ্রুবক int64_t ne2, ধ্রুবক int64_t ne3,
                    ধ্রুবক int64_t nb[3], ধ্রুবক int64_t src_nb[3],
                    ধ্রুবক int64_t offset_elem,
                    ধ্রুবক nd_item<1>& item)
{
    ধ্রুবক size_t idx = item.get_global_id(0);
    ধ্রুবক size_t যোগ = ne0 * ne1 * ne2 * ne3;
    যদি (idx >= যোগ) ফেরত;

    // Convert linear index to 4D indices
    ধ্রুবক size_t i3 = idx / (ne2 * ne1 * ne0);
    ধ্রুবক size_t rem = idx % (ne2 * ne1 * ne0);
    ধ্রুবক size_t i2 = rem / (ne1 * ne0);
    ধ্রুবক size_t rem2 = rem % (ne1 * ne0);
    ধ্রুবক size_t i1 = rem2 / ne0;
    ধ্রুবক size_t i0 = rem2 % ne0;

    // Compute source and destination indices and copy
    dst[i0 + i1*nb[0] + i2*nb[1] + i3*nb[2] + offset_elem] =
        src[i0 + i1*src_nb[0] + i2*src_nb[1] + i3*src_nb[2]];
}

// Main function: prepare GPU queue and launch parallel_for
শূন্য ggml_sycl_op_set(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ধ্রুবক ggml_tensor* src0 = dst->src[0];
    ধ্রুবক ggml_tensor* src1 = dst->src[1];

    // Ensure shapes and types are compatible
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type && src0->type == src1->type && dst->type == GGML_TYPE_F32);

    ধ্রুবক int32_t* opts = (ধ্রুবক int32_t*) dst->op_params;
    ধ্রুবক int64_t nb[3]     = {opts[0]/আকার(দশমিক), opts[1]/আকার(দশমিক), opts[2]/আকার(দশমিক)};
    ধ্রুবক int64_t offset_elem = opts[3] / আকার(দশমিক);
    ধ্রুবক যৌক্তিক inplace = opts[4];

    দশমিক* dst_ptr = (দশমিক*) dst->তথ্য;
    ধ্রুবক দশমিক* src0_ptr = (ধ্রুবক দশমিক*) src0->তথ্য;
    ধ্রুবক দশমিক* src1_ptr = (ধ্রুবক দশমিক*) src1->তথ্য;

    queue_ptr stream = ctx.stream();

    // Copy src0 to dst if not inplace
    যদি (!inplace)
        stream->memcpy(dst_ptr, src0_ptr, ggml_nbytes(dst));

    ধ্রুবক int64_t ne[4] = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    ধ্রুবক int64_t src_nb[3] = {src1->nb[1]/আকার(দশমিক), src1->nb[2]/আকার(দশমিক), src1->nb[3]/আকার(দশমিক)};

    ধ্রুবক size_t যোগ_threads = ne[0]*ne[1]*ne[2]*ne[3];
    ধ্রুবক size_t grid_আকার = ((যোগ_threads + SYCL_SET_BLOCK_আকার - 1) / SYCL_SET_BLOCK_আকার) * SYCL_SET_BLOCK_আকার;

    // Copy src0 to dst if not inplace
    stream->parallel_for(
        nd_range<1>(range<1>(grid_আকার), range<1>(SYCL_SET_BLOCK_আকার)),
        [=](nd_item<1> item) {
            set_f32(src1_ptr, dst_ptr,
                ne[0], ne[1], ne[2], ne[3],
                nb, src_nb, offset_elem, item); }
    );
}
