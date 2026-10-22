#अन्तर्भाव "presets.hpp"
#अन्तर्भाव "common.hpp"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "set.hpp"
#अन्तर्भाव <cstdint>
#अन्तर्भाव <sycl/sycl.hpp>
using नामाकाश sycl;

// Internal function: perform element-wise set operation for each thread
अन्तःस्थ शून्य set_f32(ध्रुव भिन्न* src, भिन्न* dst,
                    ध्रुव int64_t ne0, ध्रुव int64_t ne1,
                    ध्रुव int64_t ne2, ध्रुव int64_t ne3,
                    ध्रुव int64_t nb[3], ध्रुव int64_t src_nb[3],
                    ध्रुव int64_t offset_elem,
                    ध्रुव nd_item<1>& item)
{
    ध्रुव माप_प्रकार idx = item.get_global_id(0);
    ध्रुव माप_प्रकार total = ne0 * ne1 * ne2 * ne3;
    यदि (idx >= total) प्रत्यागम;

    // Convert linear index to 4D indices
    ध्रुव माप_प्रकार i3 = idx / (ne2 * ne1 * ne0);
    ध्रुव माप_प्रकार rem = idx % (ne2 * ne1 * ne0);
    ध्रुव माप_प्रकार i2 = rem / (ne1 * ne0);
    ध्रुव माप_प्रकार rem2 = rem % (ne1 * ne0);
    ध्रुव माप_प्रकार i1 = rem2 / ne0;
    ध्रुव माप_प्रकार i0 = rem2 % ne0;

    // Compute source and destination indices and copy
    dst[i0 + i1*nb[0] + i2*nb[1] + i3*nb[2] + offset_elem] =
        src[i0 + i1*src_nb[0] + i2*src_nb[1] + i3*src_nb[2]];
}

// Main function: prepare GPU queue and launch parallel_for
शून्य ggml_sycl_op_set(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ध्रुव ggml_tensor* src0 = dst->src[0];
    ध्रुव ggml_tensor* src1 = dst->src[1];

    // Ensure shapes and types are compatible
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type && src0->type == src1->type && dst->type == GGML_TYPE_F32);

    ध्रुव int32_t* opts = (ध्रुव int32_t*) dst->op_params;
    ध्रुव int64_t nb[3]     = {opts[0]/परिमाण(भिन्न), opts[1]/परिमाण(भिन्न), opts[2]/परिमाण(भिन्न)};
    ध्रुव int64_t offset_elem = opts[3] / परिमाण(भिन्न);
    ध्रुव तार्किक inplace = opts[4];

    भिन्न* dst_ptr = (भिन्न*) dst->data;
    ध्रुव भिन्न* src0_ptr = (ध्रुव भिन्न*) src0->data;
    ध्रुव भिन्न* src1_ptr = (ध्रुव भिन्न*) src1->data;

    queue_ptr stream = ctx.stream();

    // Copy src0 to dst if not inplace
    यदि (!inplace)
        stream->स_नकल(dst_ptr, src0_ptr, ggml_nbytes(dst));

    ध्रुव int64_t ne[4] = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    ध्रुव int64_t src_nb[3] = {src1->nb[1]/परिमाण(भिन्न), src1->nb[2]/परिमाण(भिन्न), src1->nb[3]/परिमाण(भिन्न)};

    ध्रुव माप_प्रकार total_threads = ne[0]*ne[1]*ne[2]*ne[3];
    ध्रुव माप_प्रकार grid_size = ((total_threads + SYCL_SET_BLOCK_SIZE - 1) / SYCL_SET_BLOCK_SIZE) * SYCL_SET_BLOCK_SIZE;

    // Copy src0 to dst if not inplace
    stream->parallel_for(
        nd_range<1>(range<1>(grid_size), range<1>(SYCL_SET_BLOCK_SIZE)),
        [=](nd_item<1> item) {
            set_f32(src1_ptr, dst_ptr,
                ne[0], ne[1], ne[2], ne[3],
                nb, src_nb, offset_elem, item); }
    );
}
