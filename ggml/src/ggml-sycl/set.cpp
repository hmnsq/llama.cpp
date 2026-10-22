#ଅନ୍ତର୍ଭାଵ "presets.hpp"
#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "set.hpp"
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <sycl/sycl.hpp>
using ନାମାକାଶ sycl;

// Internal function: perform element-wise set operation for each thread
ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ set_f32(ଅଚଳ ଦଶମିକ* src, ଦଶମିକ* dst,
                    ଅଚଳ int64_t ne0, ଅଚଳ int64_t ne1,
                    ଅଚଳ int64_t ne2, ଅଚଳ int64_t ne3,
                    ଅଚଳ int64_t nb[3], ଅଚଳ int64_t src_nb[3],
                    ଅଚଳ int64_t offset_elem,
                    ଅଚଳ nd_item<1>& item)
{
    ଅଚଳ size_t idx = item.get_global_id(0);
    ଅଚଳ size_t total = ne0 * ne1 * ne2 * ne3;
    ଯଦି (idx >= total) ଫେରନ୍ତୁ;

    // Convert linear index to 4D indices
    ଅଚଳ size_t i3 = idx / (ne2 * ne1 * ne0);
    ଅଚଳ size_t rem = idx % (ne2 * ne1 * ne0);
    ଅଚଳ size_t i2 = rem / (ne1 * ne0);
    ଅଚଳ size_t rem2 = rem % (ne1 * ne0);
    ଅଚଳ size_t i1 = rem2 / ne0;
    ଅଚଳ size_t i0 = rem2 % ne0;

    // Compute source and destination indices and copy
    dst[i0 + i1*nb[0] + i2*nb[1] + i3*nb[2] + offset_elem] =
        src[i0 + i1*src_nb[0] + i2*src_nb[1] + i3*src_nb[2]];
}

// Main function: prepare GPU queue and launch parallel_for
ଶୂନ୍ୟ ggml_sycl_op_set(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ଅଚଳ ggml_tensor* src0 = dst->src[0];
    ଅଚଳ ggml_tensor* src1 = dst->src[1];

    // Ensure shapes and types are compatible
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type && src0->type == src1->type && dst->type == GGML_TYPE_F32);

    ଅଚଳ int32_t* opts = (ଅଚଳ int32_t*) dst->op_params;
    ଅଚଳ int64_t nb[3]     = {opts[0]/ଆକାର(ଦଶମିକ), opts[1]/ଆକାର(ଦଶମିକ), opts[2]/ଆକାର(ଦଶମିକ)};
    ଅଚଳ int64_t offset_elem = opts[3] / ଆକାର(ଦଶମିକ);
    ଅଚଳ ତାର୍କିକ inplace = opts[4];

    ଦଶମିକ* dst_ptr = (ଦଶମିକ*) dst->data;
    ଅଚଳ ଦଶମିକ* src0_ptr = (ଅଚଳ ଦଶମିକ*) src0->data;
    ଅଚଳ ଦଶମିକ* src1_ptr = (ଅଚଳ ଦଶମିକ*) src1->data;

    queue_ptr stream = ctx.stream();

    // Copy src0 to dst if not inplace
    ଯଦି (!inplace)
        stream->memcpy(dst_ptr, src0_ptr, ggml_nbytes(dst));

    ଅଚଳ int64_t ne[4] = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    ଅଚଳ int64_t src_nb[3] = {src1->nb[1]/ଆକାର(ଦଶମିକ), src1->nb[2]/ଆକାର(ଦଶମିକ), src1->nb[3]/ଆକାର(ଦଶମିକ)};

    ଅଚଳ size_t total_threads = ne[0]*ne[1]*ne[2]*ne[3];
    ଅଚଳ size_t grid_size = ((total_threads + SYCL_SET_BLOCK_SIZE - 1) / SYCL_SET_BLOCK_SIZE) * SYCL_SET_BLOCK_SIZE;

    // Copy src0 to dst if not inplace
    stream->parallel_for(
        nd_range<1>(range<1>(grid_size), range<1>(SYCL_SET_BLOCK_SIZE)),
        [=](nd_item<1> item) {
            set_f32(src1_ptr, dst_ptr,
                ne[0], ne[1], ne[2], ne[3],
                nb, src_nb, offset_elem, item); }
    );
}
