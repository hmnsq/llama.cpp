#ಅನ್ತರ್ಭಾವ "presets.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "set.hpp"
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
using ನಾಮಾಕಾಶ sycl;

// Internal function: perform element-wise set operation for each thread
ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ set_f32(ಧ್ರುವ ಭಿನ್ನ* src, ಭಿನ್ನ* dst,
                    ಧ್ರುವ int64_t ne0, ಧ್ರುವ int64_t ne1,
                    ಧ್ರುವ int64_t ne2, ಧ್ರುವ int64_t ne3,
                    ಧ್ರುವ int64_t nb[3], ಧ್ರುವ int64_t src_nb[3],
                    ಧ್ರುವ int64_t offset_elem,
                    ಧ್ರುವ nd_item<1>& item)
{
    ಧ್ರುವ size_t idx = item.get_global_id(0);
    ಧ್ರುವ size_t total = ne0 * ne1 * ne2 * ne3;
    ಯದಿ (idx >= total) ಪ್ರತ್ಯಾಗಮ;

    // Convert linear index to 4D indices
    ಧ್ರುವ size_t i3 = idx / (ne2 * ne1 * ne0);
    ಧ್ರುವ size_t rem = idx % (ne2 * ne1 * ne0);
    ಧ್ರುವ size_t i2 = rem / (ne1 * ne0);
    ಧ್ರುವ size_t rem2 = rem % (ne1 * ne0);
    ಧ್ರುವ size_t i1 = rem2 / ne0;
    ಧ್ರುವ size_t i0 = rem2 % ne0;

    // Compute source and destination indices and copy
    dst[i0 + i1*nb[0] + i2*nb[1] + i3*nb[2] + offset_elem] =
        src[i0 + i1*src_nb[0] + i2*src_nb[1] + i3*src_nb[2]];
}

// Main function: prepare GPU queue and launch parallel_for
ಶೂನ್ಯ ggml_sycl_op_set(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ಧ್ರುವ ggml_tensor* src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor* src1 = dst->src[1];

    // Ensure shapes and types are compatible
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type && src0->type == src1->type && dst->type == GGML_TYPE_F32);

    ಧ್ರುವ int32_t* opts = (ಧ್ರುವ int32_t*) dst->op_params;
    ಧ್ರುವ int64_t nb[3]     = {opts[0]/ಪರಿಮಾಣ(ಭಿನ್ನ), opts[1]/ಪರಿಮಾಣ(ಭಿನ್ನ), opts[2]/ಪರಿಮಾಣ(ಭಿನ್ನ)};
    ಧ್ರುವ int64_t offset_elem = opts[3] / ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ ತಾರ್ಕಿಕ inplace = opts[4];

    ಭಿನ್ನ* dst_ptr = (ಭಿನ್ನ*) dst->data;
    ಧ್ರುವ ಭಿನ್ನ* src0_ptr = (ಧ್ರುವ ಭಿನ್ನ*) src0->data;
    ಧ್ರುವ ಭಿನ್ನ* src1_ptr = (ಧ್ರುವ ಭಿನ್ನ*) src1->data;

    queue_ptr stream = ctx.stream();

    // Copy src0 to dst if not inplace
    ಯದಿ (!inplace)
        stream->memcpy(dst_ptr, src0_ptr, ggml_nbytes(dst));

    ಧ್ರುವ int64_t ne[4] = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    ಧ್ರುವ int64_t src_nb[3] = {src1->nb[1]/ಪರಿಮಾಣ(ಭಿನ್ನ), src1->nb[2]/ಪರಿಮಾಣ(ಭಿನ್ನ), src1->nb[3]/ಪರಿಮಾಣ(ಭಿನ್ನ)};

    ಧ್ರುವ size_t total_threads = ne[0]*ne[1]*ne[2]*ne[3];
    ಧ್ರುವ size_t grid_size = ((total_threads + SYCL_SET_BLOCK_SIZE - 1) / SYCL_SET_BLOCK_SIZE) * SYCL_SET_BLOCK_SIZE;

    // Copy src0 to dst if not inplace
    stream->parallel_for(
        nd_range<1>(range<1>(grid_size), range<1>(SYCL_SET_BLOCK_SIZE)),
        [=](nd_item<1> item) {
            set_f32(src1_ptr, dst_ptr,
                ne[0], ne[1], ne[2], ne[3],
                nb, src_nb, offset_elem, item); }
    );
}
