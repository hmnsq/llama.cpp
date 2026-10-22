#include "conv2d-dw.hpp"

struct conv2d_dw_params {
    정수 in_w, in_h;
    정수 out_w, out_h;
    정수 kernel_w, kernel_h;
    정수 stride_x, stride_y;
    정수 padding_x, padding_y;
    정수 dilation_x, dilation_y;
    정수 channels, batches;
};

struct conv2d_dw_kernel_bounds {
    정수 y_min, y_max;
    정수 x_min, x_max;
};

static inline conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(정수 out_x, 정수 out_y,
                                                                  const conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    반환 bounds;
}

static inline 정수 dw_calculate_input_coord(정수 out_coord, 정수 kern_coord, 정수 stride, 정수 dilation, 정수 padding) {
    반환 out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
struct dw_whcn_layout {
    static 정수 input_index(정수 n, 정수 c, 정수 y, 정수 x, const conv2d_dw_params & p) {
        반환 n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    static 정수 kernel_index(정수 c, 정수 ky, 정수 kx, const conv2d_dw_params & p) {
        반환 c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    static 정수 output_index(정수 n, 정수 c, 정수 y, 정수 x, const conv2d_dw_params & p) {
        반환 n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    static void unpack_indices(정수 global_idx, const conv2d_dw_params & p,
                               정수 & n, 정수 & c, 정수 & out_y, 정수 & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
struct dw_cwhn_layout {
    static 정수 input_index(정수 n, 정수 c, 정수 y, 정수 x, const conv2d_dw_params & p) {
        반환 n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    static 정수 kernel_index(정수 c, 정수 ky, 정수 kx, const conv2d_dw_params & p) {
        반환 (ky * p.kernel_w + kx) * p.channels + c;
    }
    static 정수 output_index(정수 n, 정수 c, 정수 y, 정수 x, const conv2d_dw_params & p) {
        반환 n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    static void unpack_indices(정수 global_idx, const conv2d_dw_params & p,
                               정수 & n, 정수 & c, 정수 & out_y, 정수 & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
static void conv2d_dw_kernel(const float * input, const KernelT * kernel, float * output,
                             const conv2d_dw_params p, const sycl::nd_item<3> & item_ct1) {
    const 정수 global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const 정수 total_elements = p.batches * p.channels * p.out_h * p.out_w;

    만약 (global_idx >= total_elements) {
        반환;
    }

    정수 n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    float acc = 0.0f;
    const conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    위해 (정수 ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        const 정수 in_y = dw_calculate_input_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        위해 (정수 kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            const 정수 in_x = dw_calculate_input_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += input[Layout::input_index(n, c, in_y, in_x, p)] *
                   static_cast<float>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
static void conv2d_dw_sycl(const float * x_d, const KernelT * w_d, float * y_d,
                            const conv2d_dw_params p, const queue_ptr & stream) {
    const 정수 total      = p.batches * p.channels * p.out_h * p.out_w;
    const 정수 num_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
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
    const 정수       stride_x   = p[0];
    const 정수       stride_y   = p[1];
    const 정수       padding_x  = p[2];
    const 정수       padding_y  = p[3];
    const 정수       dilation_x = p[4];
    const 정수       dilation_y = p[5];

    const 정수 in_w     = input->ne[0];
    const 정수 in_h     = input->ne[1];
    const 정수 kernel_w = kernel->ne[0];
    const 정수 kernel_h = kernel->ne[1];
    const 정수 out_w    = dst->ne[0];
    const 정수 out_h    = dst->ne[1];
    const 정수 channels = dst->ne[2];
    const 정수 batches  = dst->ne[3];

    const conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    const queue_ptr stream = ctx.stream();

    만약 (kernel->type == GGML_TYPE_F16) {
        const sycl::half * w_d = (const sycl::half *) kernel->data;
        만약 (ggml_is_contiguous(input)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } 아니면 만약 (ggml_is_contiguous_channels(input)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } 아니면 {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } 아니면 {
        const float * w_d = (const float *) kernel->data;
        만약 (ggml_is_contiguous(input)) {
            conv2d_dw_sycl<float, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } 아니면 만약 (ggml_is_contiguous_channels(input)) {
            conv2d_dw_sycl<float, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } 아니면 {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
