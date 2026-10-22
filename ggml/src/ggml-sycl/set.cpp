#અન્તર્ભાવ "presets.hpp"
#અન્તર્ભાવ "common.hpp"
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "set.hpp"
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <sycl/sycl.hpp>
using નામાકાશ sycl;

// Internal function: perform element-wise set operation for each thread
અંતઃરેખ શૂન્ય set_f32(અચલ દશાંશ* src, દશાંશ* dst,
                    અચલ int64_t ne0, અચલ int64_t ne1,
                    અચલ int64_t ne2, અચલ int64_t ne3,
                    અચલ int64_t nb[3], અચલ int64_t src_nb[3],
                    અચલ int64_t offset_elem,
                    અચલ nd_item<1>& item)
{
    અચલ size_t idx = item.get_global_id(0);
    અચલ size_t total = ne0 * ne1 * ne2 * ne3;
    જો (idx >= total) પાછા;

    // Convert linear index to 4D indices
    અચલ size_t i3 = idx / (ne2 * ne1 * ne0);
    અચલ size_t rem = idx % (ne2 * ne1 * ne0);
    અચલ size_t i2 = rem / (ne1 * ne0);
    અચલ size_t rem2 = rem % (ne1 * ne0);
    અચલ size_t i1 = rem2 / ne0;
    અચલ size_t i0 = rem2 % ne0;

    // Compute source and destination indices and copy
    dst[i0 + i1*nb[0] + i2*nb[1] + i3*nb[2] + offset_elem] =
        src[i0 + i1*src_nb[0] + i2*src_nb[1] + i3*src_nb[2]];
}

// Main function: prepare GPU queue and launch parallel_for
શૂન્ય ggml_sycl_op_set(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    અચલ ggml_tensor* src0 = dst->src[0];
    અચલ ggml_tensor* src1 = dst->src[1];

    // Ensure shapes and types are compatible
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type && src0->type == src1->type && dst->type == GGML_TYPE_F32);

    અચલ int32_t* opts = (અચલ int32_t*) dst->op_params;
    અચલ int64_t nb[3]     = {opts[0]/કદ(દશાંશ), opts[1]/કદ(દશાંશ), opts[2]/કદ(દશાંશ)};
    અચલ int64_t offset_elem = opts[3] / કદ(દશાંશ);
    અચલ તાર્કિક inplace = opts[4];

    દશાંશ* dst_ptr = (દશાંશ*) dst->data;
    અચલ દશાંશ* src0_ptr = (અચલ દશાંશ*) src0->data;
    અચલ દશાંશ* src1_ptr = (અચલ દશાંશ*) src1->data;

    queue_ptr stream = ctx.stream();

    // Copy src0 to dst if not inplace
    જો (!inplace)
        stream->memcpy(dst_ptr, src0_ptr, ggml_nbytes(dst));

    અચલ int64_t ne[4] = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    અચલ int64_t src_nb[3] = {src1->nb[1]/કદ(દશાંશ), src1->nb[2]/કદ(દશાંશ), src1->nb[3]/કદ(દશાંશ)};

    અચલ size_t total_threads = ne[0]*ne[1]*ne[2]*ne[3];
    અચલ size_t grid_size = ((total_threads + SYCL_SET_BLOCK_SIZE - 1) / SYCL_SET_BLOCK_SIZE) * SYCL_SET_BLOCK_SIZE;

    // Copy src0 to dst if not inplace
    stream->parallel_for(
        nd_range<1>(range<1>(grid_size), range<1>(SYCL_SET_BLOCK_SIZE)),
        [=](nd_item<1> item) {
            set_f32(src1_ptr, dst_ptr,
                ne[0], ne[1], ne[2], ne[3],
                nb, src_nb, offset_elem, item); }
    );
}
