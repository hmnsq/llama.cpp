#include "conv2d-dw.hpp"

sa_nracanaa conv2d_dw_params {
    ຈຳນວນເຕັມ in_w, in_h;
    ຈຳນວນເຕັມ out_w, out_h;
    ຈຳນວນເຕັມ kernel_w, kernel_h;
    ຈຳນວນເຕັມ stride_x, stride_y;
    ຈຳນວນເຕັມ padding_x, padding_y;
    ຈຳນວນເຕັມ dilation_x, dilation_y;
    ຈຳນວນເຕັມ channels, batches;
};

sa_nracanaa conv2d_dw_kernel_bounds {
    ຈຳນວນເຕັມ y_min, y_max;
    ຈຳນວນເຕັມ x_min, x_max;
};

static anta_hstha conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(ຈຳນວນເຕັມ out_x, ຈຳນວນເຕັມ out_y,
                                                                  const conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    ສົ່ງຄືນ bounds;
}

static anta_hstha ຈຳນວນເຕັມ dw_calculate_input_coord(ຈຳນວນເຕັມ out_coord, ຈຳນວນເຕັມ kern_coord, ຈຳນວນເຕັມ stride, ຈຳນວນເຕັມ dilation, ຈຳນວນເຕັມ padding) {
    ສົ່ງຄືນ out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
sa_nracanaa dw_whcn_layout {
    static ຈຳນວນເຕັມ input_index(ຈຳນວນເຕັມ n, ຈຳນວນເຕັມ c, ຈຳນວນເຕັມ y, ຈຳນວນເຕັມ x, const conv2d_dw_params & p) {
        ສົ່ງຄືນ n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    static ຈຳນວນເຕັມ kernel_index(ຈຳນວນເຕັມ c, ຈຳນວນເຕັມ ky, ຈຳນວນເຕັມ kx, const conv2d_dw_params & p) {
        ສົ່ງຄືນ c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    static ຈຳນວນເຕັມ output_index(ຈຳນວນເຕັມ n, ຈຳນວນເຕັມ c, ຈຳນວນເຕັມ y, ຈຳນວນເຕັມ x, const conv2d_dw_params & p) {
        ສົ່ງຄືນ n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    static void unpack_indices(ຈຳນວນເຕັມ global_idx, const conv2d_dw_params & p,
                               ຈຳນວນເຕັມ & n, ຈຳນວນເຕັມ & c, ຈຳນວນເຕັມ & out_y, ຈຳນວນເຕັມ & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
sa_nracanaa dw_cwhn_layout {
    static ຈຳນວນເຕັມ input_index(ຈຳນວນເຕັມ n, ຈຳນວນເຕັມ c, ຈຳນວນເຕັມ y, ຈຳນວນເຕັມ x, const conv2d_dw_params & p) {
        ສົ່ງຄືນ n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    static ຈຳນວນເຕັມ kernel_index(ຈຳນວນເຕັມ c, ຈຳນວນເຕັມ ky, ຈຳນວນເຕັມ kx, const conv2d_dw_params & p) {
        ສົ່ງຄືນ (ky * p.kernel_w + kx) * p.channels + c;
    }
    static ຈຳນວນເຕັມ output_index(ຈຳນວນເຕັມ n, ຈຳນວນເຕັມ c, ຈຳນວນເຕັມ y, ຈຳນວນເຕັມ x, const conv2d_dw_params & p) {
        ສົ່ງຄືນ n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    static void unpack_indices(ຈຳນວນເຕັມ global_idx, const conv2d_dw_params & p,
                               ຈຳນວນເຕັມ & n, ຈຳນວນເຕັມ & c, ຈຳນວນເຕັມ & out_y, ຈຳນວນເຕັມ & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
static void conv2d_dw_kernel(const float * input, const KernelT * kernel, float * output,
                             const conv2d_dw_params p, const sycl::nd_item<3> & item_ct1) {
    const ຈຳນວນເຕັມ global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const ຈຳນວນເຕັມ total_elements = p.batches * p.channels * p.out_h * p.out_w;

    ຖ້າ (global_idx >= total_elements) {
        ສົ່ງຄືນ;
    }

    ຈຳນວນເຕັມ n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    float acc = 0.0f;
    const conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    ສຳລັບ (ຈຳນວນເຕັມ ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        const ຈຳນວນເຕັມ in_y = dw_calculate_input_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        ສຳລັບ (ຈຳນວນເຕັມ kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            const ຈຳນວນເຕັມ in_x = dw_calculate_input_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += input[Layout::input_index(n, c, in_y, in_x, p)] *
                   static_cast<float>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
static void conv2d_dw_sycl(const float * x_d, const KernelT * w_d, float * y_d,
                            const conv2d_dw_params p, const queue_ptr & stream) {
    const ຈຳນວນເຕັມ total      = p.batches * p.channels * p.out_h * p.out_w;
    const ຈຳນວນເຕັມ num_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    const sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, 1, num_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

void ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const float * x_d = (const float *) input->data;
    float *       y_d = (float *) dst->data;

    const int32_t * p          = (const int32_t *) dst->op_params;
    const ຈຳນວນເຕັມ       stride_x   = p[0];
    const ຈຳນວນເຕັມ       stride_y   = p[1];
    const ຈຳນວນເຕັມ       padding_x  = p[2];
    const ຈຳນວນເຕັມ       padding_y  = p[3];
    const ຈຳນວນເຕັມ       dilation_x = p[4];
    const ຈຳນວນເຕັມ       dilation_y = p[5];

    const ຈຳນວນເຕັມ in_w     = input->ne[0];
    const ຈຳນວນເຕັມ in_h     = input->ne[1];
    const ຈຳນວນເຕັມ kernel_w = kernel->ne[0];
    const ຈຳນວນເຕັມ kernel_h = kernel->ne[1];
    const ຈຳນວນເຕັມ out_w    = dst->ne[0];
    const ຈຳນວນເຕັມ out_h    = dst->ne[1];
    const ຈຳນວນເຕັມ channels = dst->ne[2];
    const ຈຳນວນເຕັມ batches  = dst->ne[3];

    const conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    const queue_ptr stream = ctx.stream();

    ຖ້າ (kernel->type == GGML_TYPE_F16) {
        const sycl::half * w_d = (const sycl::half *) kernel->data;
        ຖ້າ (ggml_is_contiguous(input)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } ບໍ່ດັ່ງນັ້ນ ຖ້າ (ggml_is_contiguous_channels(input)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } ບໍ່ດັ່ງນັ້ນ {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } ບໍ່ດັ່ງນັ້ນ {
        const float * w_d = (const float *) kernel->data;
        ຖ້າ (ggml_is_contiguous(input)) {
            conv2d_dw_sycl<float, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } ບໍ່ດັ່ງນັ້ນ ຖ້າ (ggml_is_contiguous_channels(input)) {
            conv2d_dw_sycl<float, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } ບໍ່ດັ່ງນັ້ນ {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
