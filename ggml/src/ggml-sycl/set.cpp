#அந்தர்பாவ "presets.hpp"
#அந்தர்பாவ "common.hpp"
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "set.hpp"
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <sycl/sycl.hpp>
using நாமாகாஶ sycl;

// Internal function: perform element-wise set operation for each thread
உள்வரி வெறுமை set_f32(மாறிலி தசம* src, தசம* dst,
                    மாறிலி int64_t ne0, மாறிலி int64_t ne1,
                    மாறிலி int64_t ne2, மாறிலி int64_t ne3,
                    மாறிலி int64_t nb[3], மாறிலி int64_t src_nb[3],
                    மாறிலி int64_t offset_elem,
                    மாறிலி nd_item<1>& item)
{
    மாறிலி size_t idx = item.get_global_id(0);
    மாறிலி size_t total = ne0 * ne1 * ne2 * ne3;
    எனில் (idx >= total) திரும்பு;

    // Convert linear index to 4D indices
    மாறிலி size_t i3 = idx / (ne2 * ne1 * ne0);
    மாறிலி size_t rem = idx % (ne2 * ne1 * ne0);
    மாறிலி size_t i2 = rem / (ne1 * ne0);
    மாறிலி size_t rem2 = rem % (ne1 * ne0);
    மாறிலி size_t i1 = rem2 / ne0;
    மாறிலி size_t i0 = rem2 % ne0;

    // Compute source and destination indices and copy
    dst[i0 + i1*nb[0] + i2*nb[1] + i3*nb[2] + offset_elem] =
        src[i0 + i1*src_nb[0] + i2*src_nb[1] + i3*src_nb[2]];
}

// Main function: prepare GPU queue and launch parallel_for
வெறுமை ggml_sycl_op_set(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    மாறிலி ggml_tensor* src0 = dst->src[0];
    மாறிலி ggml_tensor* src1 = dst->src[1];

    // Ensure shapes and types are compatible
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type && src0->type == src1->type && dst->type == GGML_TYPE_F32);

    மாறிலி int32_t* opts = (மாறிலி int32_t*) dst->op_params;
    மாறிலி int64_t nb[3]     = {opts[0]/அளவு(தசம), opts[1]/அளவு(தசம), opts[2]/அளவு(தசம)};
    மாறிலி int64_t offset_elem = opts[3] / அளவு(தசம);
    மாறிலி தருக்க inplace = opts[4];

    தசம* dst_ptr = (தசம*) dst->data;
    மாறிலி தசம* src0_ptr = (மாறிலி தசம*) src0->data;
    மாறிலி தசம* src1_ptr = (மாறிலி தசம*) src1->data;

    queue_ptr stream = ctx.stream();

    // Copy src0 to dst if not inplace
    எனில் (!inplace)
        stream->memcpy(dst_ptr, src0_ptr, ggml_nbytes(dst));

    மாறிலி int64_t ne[4] = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    மாறிலி int64_t src_nb[3] = {src1->nb[1]/அளவு(தசம), src1->nb[2]/அளவு(தசம), src1->nb[3]/அளவு(தசம)};

    மாறிலி size_t total_threads = ne[0]*ne[1]*ne[2]*ne[3];
    மாறிலி size_t grid_size = ((total_threads + SYCL_SET_BLOCK_SIZE - 1) / SYCL_SET_BLOCK_SIZE) * SYCL_SET_BLOCK_SIZE;

    // Copy src0 to dst if not inplace
    stream->parallel_for(
        nd_range<1>(range<1>(grid_size), range<1>(SYCL_SET_BLOCK_SIZE)),
        [=](nd_item<1> item) {
            set_f32(src1_ptr, dst_ptr,
                ne[0], ne[1], ne[2], ne[3],
                nb, src_nb, offset_elem, item); }
    );
}
