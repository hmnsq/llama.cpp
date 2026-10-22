#அந்தர்பாவ "diag.hpp"
#அந்தர்பாவ "common.hpp"

#define SYCL_DIAG_BLOCK_SIZE 256

template <typename T>
நிலையான வெறுமை diag_kernel(T * __restrict__ dst, மாறிலி T * __restrict__ src,
                        மாறிலி int64_t ne0, மாறிலி int64_t ne1,
                        மாறிலி int64_t ne2, மாறிலி int64_t ne3,
                        மாறிலி int64_t total_elements,
                        மாறிலி sycl::nd_item<1> & item) {
    மாறிலி int64_t i = item.get_global_id(0);
    எனில் (i >= total_elements) {
        திரும்பு;
    }

    மாறிலி int64_t i0 = i % ne0;
    மாறிலி int64_t i1 = (i / ne0) % ne1;
    மாறிலி int64_t i2 = (i / (ne0 * ne1)) % ne2;
    மாறிலி int64_t i3 = i / (ne0 * ne1 * ne2);

    மாறிலி int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    எனில் (i0 == i1) {
        மாறிலி int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } இல்லையெனில் {
        dst[dst_idx] = T(0);
    }

    (வெறுமை)ne3;
}

உள்வரி வெறுமை ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    மாறிலி வெறுமை * src0_d = src0->data;
    வெறுமை * dst_d = dst->data;

    மாறிலி int64_t ne0 = dst->ne[0];
    மாறிலி int64_t ne1 = dst->ne[1];
    மாறிலி int64_t ne2 = dst->ne[2];
    மாறிலி int64_t ne3 = dst->ne[3];
    மாறிலி int64_t n_elems = ggml_nelements(dst);
    மாறிலி int64_t இலக்கம்_blocks = (n_elems + SYCL_DIAG_BLOCK_SIZE - 1) / SYCL_DIAG_BLOCK_SIZE;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(இலக்கம்_blocks * SYCL_DIAG_BLOCK_SIZE, SYCL_DIAG_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            diag_kernel(static_cast<தசம *>(dst_d),
                        static_cast<மாறிலி தசம *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, item);
        });
}

வெறுமை ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
