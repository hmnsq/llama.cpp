#include "conv2d-dw.hpp"

struct conv2d_dw_params {
    цео in_w, in_h;
    цео out_w, out_h;
    цео kernel_w, kernel_h;
    цео stride_x, stride_y;
    цео padding_x, padding_y;
    цео dilation_x, dilation_y;
    цео channels, batches;
};

struct conv2d_dw_kernel_bounds {
    цео y_min, y_max;
    цео x_min, x_max;
};

static inline conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(цео out_x, цео out_y,
                                                                  const conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    врати bounds;
}

static inline цео dw_calculate_input_coord(цео out_coord, цео kern_coord, цео stride, цео dilation, цео padding) {
    врати out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
struct dw_whcn_layout {
    static цео input_index(цео n, цео c, цео y, цео x, const conv2d_dw_params & p) {
        врати n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    static цео kernel_index(цео c, цео ky, цео kx, const conv2d_dw_params & p) {
        врати c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    static цео output_index(цео n, цео c, цео y, цео x, const conv2d_dw_params & p) {
        врати n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    static void unpack_indices(цео global_idx, const conv2d_dw_params & p,
                               цео & n, цео & c, цео & out_y, цео & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
struct dw_cwhn_layout {
    static цео input_index(цео n, цео c, цео y, цео x, const conv2d_dw_params & p) {
        врати n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    static цео kernel_index(цео c, цео ky, цео kx, const conv2d_dw_params & p) {
        врати (ky * p.kernel_w + kx) * p.channels + c;
    }
    static цео output_index(цео n, цео c, цео y, цео x, const conv2d_dw_params & p) {
        врати n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    static void unpack_indices(цео global_idx, const conv2d_dw_params & p,
                               цео & n, цео & c, цео & out_y, цео & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
static void conv2d_dw_kernel(const float * input, const KernelT * kernel, float * output,
                             const conv2d_dw_params p, const sycl::nd_item<3> & item_ct1) {
    const цео global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const цео total_elements = p.batches * p.channels * p.out_h * p.out_w;

    ако (global_idx >= total_elements) {
        врати;
    }

    цео n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    float acc = 0.0f;
    const conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    за (цео ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        const цео in_y = dw_calculate_input_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        за (цео kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            const цео in_x = dw_calculate_input_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += input[Layout::input_index(n, c, in_y, in_x, p)] *
                   static_cast<float>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
static void conv2d_dw_sycl(const float * x_d, const KernelT * w_d, float * y_d,
                            const conv2d_dw_params p, const queue_ptr & stream) {
    const цео total      = p.batches * p.channels * p.out_h * p.out_w;
    const цео num_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
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
    const цео       stride_x   = p[0];
    const цео       stride_y   = p[1];
    const цео       padding_x  = p[2];
    const цео       padding_y  = p[3];
    const цео       dilation_x = p[4];
    const цео       dilation_y = p[5];

    const цео in_w     = input->ne[0];
    const цео in_h     = input->ne[1];
    const цео kernel_w = kernel->ne[0];
    const цео kernel_h = kernel->ne[1];
    const цео out_w    = dst->ne[0];
    const цео out_h    = dst->ne[1];
    const цео channels = dst->ne[2];
    const цео batches  = dst->ne[3];

    const conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    const queue_ptr stream = ctx.stream();

    ако (kernel->type == GGML_TYPE_F16) {
        const sycl::half * w_d = (const sycl::half *) kernel->data;
        ако (ggml_is_contiguous(input)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } инако ако (ggml_is_contiguous_channels(input)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } инако {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } инако {
        const float * w_d = (const float *) kernel->data;
        ако (ggml_is_contiguous(input)) {
            conv2d_dw_sycl<float, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } инако ако (ggml_is_contiguous_channels(input)) {
            conv2d_dw_sycl<float, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } инако {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
