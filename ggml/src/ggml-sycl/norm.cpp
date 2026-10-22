#شامل "norm.hpp"
#شامل "ggml-sycl/common.hpp"
#شامل "ggml-sycl/presets.hpp"

ایستا تهی norm_f32(ثابت اعشاری* x, اعشاری* dst, ثابت صحیح ncols,
    ثابت int64_t src_stride_col, ثابت int64_t src_stride_row, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_col, ثابت int64_t dst_stride_row, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت اعشاری eps, ثابت sycl::nd_item<3>& item_ct1, sycl::float2* s_sum, صحیح block_size) {

    ثابت صحیح nthreads = item_ct1.get_local_range(2);
    ثابت صحیح sample  = item_ct1.get_group(0);
    ثابت صحیح channel = item_ct1.get_group(1);
    ثابت صحیح row     = item_ct1.get_group(2);

    ثابت صحیح tid = item_ct1.get_local_id(2);
    ثابت صحیح nwarps = nthreads / WARP_SIZE;

    ثابت auto src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    ثابت auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x += src_offset;
    dst += dst_offset;

    sycl::float2 mean_var = sycl::float2(0.f, 0.f);

    برای (صحیح col = tid; col < ncols; col += block_size) {
        ثابت اعشاری xi = x[col * src_stride_col];
        mean_var.x() += xi;
        mean_var.y() += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var, item_ct1);
    اگر  (block_size > WARP_SIZE) {
        ثابت auto sub_group = item_ct1.get_sub_group();
        ثابت auto sg_id = sub_group.get_group_linear_id();
        ثابت auto wi_in_sg = sub_group.get_local_linear_id();
        اگر (wi_in_sg == 0) {
            s_sum[sg_id] = mean_var;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        mean_var = 0.f;
        ثابت size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        برای (size_t i = 0; i < nreduce; i += 1)
        {
            mean_var += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        mean_var = warp_reduce_sum(mean_var, item_ct1);
    }

    ثابت اعشاری mean = mean_var.x() / ncols;
    ثابت اعشاری var = mean_var.y() / ncols - mean * mean;
    ثابت اعشاری inv_std = sycl::rsqrt(var + eps);

    برای (صحیح col = tid; col < ncols; col += block_size) {
        dst[col * dst_stride_col] = (x[col * src_stride_col] - mean) * inv_std;
    }
}

ایستا تهی group_norm_f32(ثابت اعشاری* x, اعشاری* dst, ثابت صحیح group_size, ثابت صحیح ne_elements, ثابت اعشاری eps,
    ثابت sycl::nd_item<3>& item_ct1, اعشاری* s_sum, صحیح block_size) {
    صحیح start = item_ct1.get_group(2) * group_size;
    صحیح end = start + group_size;
    ثابت صحیح nthreads = item_ct1.get_local_range(2);
    ثابت صحیح nwarps = nthreads / WARP_SIZE;
    start += item_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_SIZE;

    اگر (end >= ne_elements) {
        end = ne_elements;
    }

    اعشاری tmp = 0.0f; // partial sum for thread in warp

    برای (صحیح j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    اگر (block_size > WARP_SIZE) {

        صحیح warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        صحیح lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        اگر (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        /*
        DPCT1118:1: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        tmp = 0.f;
        برای (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    اعشاری mean = tmp / group_size;
    tmp = 0.0f;

    برای (صحیح j = start; j < end; j += block_size) {
        اعشاری xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    اگر (block_size > WARP_SIZE) {

        صحیح warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        صحیح lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        اگر (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        /*
        DPCT1118:2: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:55: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        tmp = 0.f;
        برای (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    اعشاری variance = tmp / group_size;
    اعشاری scale = sycl::rsqrt(variance + eps);
    برای (صحیح j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <منطقی do_multiply = نادرست, منطقی do_add = نادرست>
ایستا تهی rms_norm_f32(ثابت اعشاری* x, اعشاری* dst, ثابت صحیح ncols,
    ثابت int64_t src_stride_col, ثابت int64_t src_stride_row, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_col, ثابت int64_t dst_stride_row, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت اعشاری eps, ثابت sycl::nd_item<3>& item_ct1, اعشاری* s_sum, صحیح block_size,
    ثابت اعشاری* mul = nullptr, ثابت int64_t mul_stride_row = 0, ثابت int64_t mul_stride_channel = 0,
    ثابت int64_t mul_stride_sample = 0, ثابت صحیح mul_nrows = 0, ثابت صحیح mul_nchannels = 0, ثابت صحیح mul_nsamples = 0,
    ثابت اعشاری* add = nullptr, ثابت int64_t add_stride_row = 0, ثابت int64_t add_stride_channel = 0,
    ثابت int64_t add_stride_sample = 0, ثابت صحیح add_nrows = 0, ثابت صحیح add_nchannels = 0, ثابت صحیح add_nsamples = 0,
    ثابت اعشاری scale_mul = 1.0f) {

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    ثابت صحیح sample  = item_ct1.get_group(0);
    ثابت صحیح channel = item_ct1.get_group(1);
    ثابت صحیح row     = item_ct1.get_group(2);

    ثابت صحیح nthreads = item_ct1.get_local_range(2);

    ثابت صحیح tid = item_ct1.get_local_id(2);
    ثابت صحیح nwarps = nthreads / WARP_SIZE;

    ثابت auto src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    ثابت auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x   += src_offset;
    dst += dst_offset;

    اگر constexpr (do_multiply) {
        ثابت صحیح mul_row     = row     % mul_nrows;
        ثابت صحیح mul_channel = channel % mul_nchannels;
        ثابت صحیح mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    اگر constexpr (do_add) {
        ثابت صحیح add_row     = row     % add_nrows;
        ثابت صحیح add_channel = channel % add_nchannels;
        ثابت صحیح add_sample  = sample  % add_nsamples;
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    اعشاری tmp = 0.0f; // partial sum for thread in warp

    برای (صحیح col = tid; col < ncols; col += block_size) {
        ثابت اعشاری xi = x[col * src_stride_col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    اگر (block_size > WARP_SIZE) {
        ثابت auto sub_group = item_ct1.get_sub_group();
        ثابت auto sg_id = sub_group.get_group_linear_id();
        ثابت auto wi_in_sg = sub_group.get_local_linear_id();
        اگر (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        ثابت size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        برای (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    ثابت اعشاری mean = tmp / ncols;
    ثابت اعشاری scale = sycl::rsqrt(mean + eps);

    برای (صحیح col = tid; col < ncols; col += block_size) {
        اگر constexpr (do_multiply && do_add) {
            dst[col * dst_stride_col] = scale * x[col * src_stride_col] * mul[col] + add[col];
        } وگرنه اگر constexpr (do_multiply) {
            dst[col * dst_stride_col] = scale * x[col * src_stride_col] * mul[col];
        } وگرنه {
            // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
            // the explicit temporary keeps the float evaluation order identical to
            // running rms_norm and scale as two separate kernels
            ثابت اعشاری v = scale * x[col * src_stride_col];
            dst[col * dst_stride_col] = v * scale_mul;
        }
    }
}

template<صحیح warp_size>
ایستا تهی l2_norm_f32(ثابت اعشاری * x, اعشاری * dst, ثابت صحیح ncols,
    ثابت int64_t src_stride_col, ثابت int64_t src_stride_row, ثابت int64_t src_stride_channel,
    ثابت int64_t src_stride_sample, ثابت int64_t dst_stride_col, ثابت int64_t dst_stride_row,
    ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample, ثابت اعشاری eps,
    ثابت sycl::nd_item<3>& item_ct1, اعشاری* s_sum, ثابت صحیح block_size) {

    ثابت صحیح row     = item_ct1.get_group(2);
    ثابت صحیح channel = item_ct1.get_group(1);
    ثابت صحیح sample  = item_ct1.get_group(0);
    ثابت صحیح tid     = item_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + row*src_stride_row;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + row*dst_stride_row;

    اعشاری tmp = 0.0f; // partial sum for thread in warp

    برای (صحیح col = tid; col < ncols; col += block_size) {
        ثابت اعشاری xi = x[col * src_stride_col];
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, s_sum, block_size);
    ثابت اعشاری scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    برای (صحیح col = tid; col < ncols; col += block_size) {
        dst[col * dst_stride_col] = scale * x[col * src_stride_col];
    }
}

ایستا تهی norm_f32_sycl(ثابت اعشاری * x, اعشاری * dst, ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح nchannels, ثابت صحیح nsamples,
    ثابت int64_t src_stride_col, ثابت int64_t src_stride_row, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_col, ثابت int64_t dst_stride_row, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت اعشاری eps, queue_ptr stream, صحیح device) {

    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    اگر (ncols < 1024) {
        ثابت sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE);
                });
            });
    }
    وگرنه {
        ثابت صحیح work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        أكد(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ثابت sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_sum_acc_ct1(
                            sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

ایستا تهی group_norm_f32_sycl(ثابت اعشاری* x, اعشاری* dst,
    ثابت صحیح رقم_groups, ثابت اعشاری eps, ثابت صحیح group_size,
    ثابت صحیح ne_elements, queue_ptr stream, صحیح device) {
    اگر (group_size < 1024) {
        ثابت sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            ثابت اعشاری eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, رقم_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(
                        x, dst, group_size, ne_elements, eps_ct4, item_ct1,
                        nullptr, WARP_SIZE);
                });
            });
    }
    وگرنه {
        ثابت صحیح work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        أكد(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ثابت sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);

            ثابت اعشاری eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, رقم_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(x, dst, group_size, ne_elements,
                        eps_ct4, item_ct1,
                        get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

ایستا تهی rms_norm_f32_sycl(ثابت اعشاری* x, اعشاری* dst, ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح nchannels, ثابت صحیح nsamples,
    ثابت int64_t src_stride_col, ثابت int64_t src_stride_row, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_col, ثابت int64_t dst_stride_row, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت اعشاری eps, queue_ptr stream, صحیح device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    اگر (ncols < 1024) {
        ثابت sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE);
                });
            });
    }
    وگرنه {
        ثابت صحیح work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        أكد(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ثابت sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

ایستا تهی rms_norm_scale_f32_sycl(ثابت اعشاری* x, اعشاری* dst, ثابت صحیح ncols, ثابت صحیح nrows,
    ثابت صحیح nchannels, ثابت صحیح nsamples,
    ثابت int64_t src_stride_col, ثابت int64_t src_stride_row, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
    ثابت int64_t dst_stride_col, ثابت int64_t dst_stride_row, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
    ثابت اعشاری eps, ثابت اعشاری scale_mul, queue_ptr stream, صحیح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    اگر (ncols < 1024) {
        ثابت sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE,
                        nullptr, 0, 0, 0, 0, 0, 0,
                        nullptr, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
    وگرنه {
        ثابت صحیح work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        أكد(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ثابت sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size,
                        nullptr, 0, 0, 0, 0, 0, 0,
                        nullptr, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
}

ایستا تهی rms_norm_mul_f32_sycl(ثابت اعشاری* x, ثابت اعشاری* mul, اعشاری* dst, ثابت صحیح ncols, ثابت صحیح nrows,
        ثابت صحیح nchannels, ثابت صحیح nsamples,
        ثابت int64_t src_stride_col, ثابت int64_t src_stride_row, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
        ثابت int64_t dst_stride_col, ثابت int64_t dst_stride_row, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت int64_t mul_stride_row, ثابت int64_t mul_stride_channel, ثابت int64_t mul_stride_sample,
        ثابت صحیح mul_nrows, ثابت صحیح mul_nchannels, ثابت صحیح mul_nsamples,
        ثابت اعشاری eps, queue_ptr stream, صحیح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    اگر (ncols < 1024) {
        ثابت sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<درست>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    وگرنه {
        ثابت صحیح work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        أكد(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ثابت sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<درست>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

ایستا تهی rms_norm_mul_add_f32_sycl(ثابت اعشاری* x, ثابت اعشاری* mul, ثابت اعشاری* add, اعشاری* dst,
        ثابت صحیح ncols, ثابت صحیح nrows, ثابت صحیح nchannels, ثابت صحیح nsamples,
        ثابت int64_t src_stride_col, ثابت int64_t src_stride_row, ثابت int64_t src_stride_channel, ثابت int64_t src_stride_sample,
        ثابت int64_t dst_stride_col, ثابت int64_t dst_stride_row, ثابت int64_t dst_stride_channel, ثابت int64_t dst_stride_sample,
        ثابت int64_t mul_stride_row, ثابت int64_t mul_stride_channel, ثابت int64_t mul_stride_sample,
        ثابت صحیح mul_nrows, ثابت صحیح mul_nchannels, ثابت صحیح mul_nsamples,
        ثابت int64_t add_stride_row, ثابت int64_t add_stride_channel, ثابت int64_t add_stride_sample,
        ثابت صحیح add_nrows, ثابت صحیح add_nchannels, ثابت صحیح add_nsamples,
        ثابت اعشاری eps, queue_ptr stream, صحیح device) {
    ثابت sycl::range<3> global_dims(nsamples, nchannels, nrows);
    اگر (ncols < 1024) {
        ثابت sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<درست, درست>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add, add_stride_row, add_stride_channel, add_stride_sample, add_nrows, add_nchannels, add_nsamples);
                });
            });
    }
    وگرنه {
        ثابت صحیح work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        أكد(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ثابت sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<درست, درست>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add, add_stride_row, add_stride_channel, add_stride_sample, add_nrows, add_nchannels, add_nsamples);
                });
            });
    }
}

template<صحیح warp_size>
ایستا تهی l2_norm_f32_sycl(ثابت اعشاری *   x,
                             اعشاری *         dst,
                             ثابت صحیح       ncols,
                             ثابت صحیح       nrows,
                             ثابت صحیح       nchannels,
                             ثابت صحیح       nsamples,
                             ثابت int64_t   src_stride_col,
                             ثابت int64_t   src_stride_row,
                             ثابت int64_t   src_stride_channel,
                             ثابت int64_t   src_stride_sample,
                             ثابت int64_t   dst_stride_col,
                             ثابت int64_t   dst_stride_row,
                             ثابت int64_t   dst_stride_channel,
                             ثابت int64_t   dst_stride_sample,
                             ثابت اعشاری     eps,
                             queue_ptr       stream,
                             صحیح             device) {
    ثابت dpct::dim3 blocks_رقم(nrows, nchannels, nsamples);

    اگر (ncols < 1024) {
        ثابت dpct::dim3 block_dims(warp_size, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(blocks_رقم * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(warp_size)]] {
                    l2_norm_f32<warp_size>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1,
                        nullptr, warp_size);
                });
            });
    }
    وگرنه {
        ثابت صحیح work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        أكد(work_group_size % (warp_size * warp_size) == 0);
        ثابت sycl::range<3> block_dims(1, 1, work_group_size);
        صحیح lsm_size =  block_dims[2] > warp_size ? work_group_size / warp_size * اندازه(اعشاری): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> s_sum_acc_ct1(sycl::range<1>(lsm_size),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(blocks_رقم * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(warp_size)]] {
                    l2_norm_f32<warp_size>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

// Batched L2 norm: N independent same-shape F32 tensors in one launch; the tensor
// index is folded into grid dim0 and each row's reduction is identical to the
// single-tensor kernel, so the result is bit-exact.
ساختار l2_batch_ptrs {
    ثابت اعشاری * src[GGML_SYCL_L2_BATCH_MAX];
    اعشاری *       dst[GGML_SYCL_L2_BATCH_MAX];
};

// One stride set shared by the whole batch: the caller only groups tensors whose nb[]
// all match, so per-tensor state stays two pointers.
ساختار l2_batch_strides {
    صحیح     ne1, ne2;
    int64_t ss0, ss1, ss2, ss3;
    int64_t ds0, ds1, ds2, ds3;
};

template <صحیح warp_size>
ایستا تهی l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, ثابت صحیح ncols, ثابت اعشاری eps,
                              ثابت sycl::nd_item<3> & item_ct1) {
    ثابت صحیح t   = item_ct1.get_group(0);  // tensor index
    ثابت صحیح r   = item_ct1.get_group(2);  // flattened row over ne1*ne2*ne3
    ثابت صحیح tid = item_ct1.get_local_id(2);

    ثابت صحیح i1 = r % st.ne1;
    ثابت صحیح i2 = (r / st.ne1) % st.ne2;
    ثابت صحیح i3 = r / (st.ne1 * st.ne2);

    ثابت اعشاری * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    اعشاری *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    اعشاری tmp = 0.0f;
    برای (صحیح col = tid; col < ncols; col += warp_size) {
        ثابت اعشاری xi = x[col * st.ss0];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, (اعشاری *) nullptr, warp_size);
    ثابت اعشاری scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    برای (صحیح col = tid; col < ncols; col += warp_size) {
        dst[col * st.ds0] = scale * x[col * st.ss0];
    }
}

template <صحیح warp_size>
ایستا تهی l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, ثابت صحیح n_tensors,
                                   ثابت صحیح ncols, ثابت صحیح nrows_total, ثابت اعشاری eps,
                                   queue_ptr stream) {
    ثابت dpct::dim3 blocks_رقم(nrows_total, 1, n_tensors);
    ثابت dpct::dim3 block_dims(warp_size, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(blocks_رقم * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                l2_norm_f32_batch<warp_size>(p, st, ncols, eps, item_ct1);
            });
    });
}

تهی ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->data);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->data);

    اعشاری eps;
    memcpy(&eps, dst->op_params, اندازه(اعشاری));
    GGML_ASSERT(eps >= 0.0f);
    ثابت size_t ts0 = ggml_type_size(src0->type);
    ثابت size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ثابت int64_t ss0 = nb00 / ts0;
    ثابت int64_t ss1 = nb01 / ts0;
    ثابت int64_t ss2 = nb02 / ts0;
    ثابت int64_t ss3 = nb03 / ts0;
    ثابت int64_t ds0 = nb0 / tdst;
    ثابت int64_t ds1 = nb1 / tdst;
    ثابت int64_t ds2 = nb2 / tdst;
    ثابت int64_t ds3 = nb3 / tdst;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
}

تهی ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    صحیح رقم_groups = dst->op_params[0];
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->data);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->data);

    اعشاری eps;
    memcpy(&eps, dst->op_params + 1, اندازه(اعشاری));

    صحیح group_size = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + رقم_groups - 1) / رقم_groups);
    group_norm_f32_sycl(src0_dd, dst_dd, رقم_groups, eps, group_size, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_stream, ctx.device);
}

تهی ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ثابت ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->data);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->data);

    اعشاری eps;
    memcpy(&eps, dst->op_params, اندازه(اعشاری));

    GGML_TENSOR_UNARY_OP_LOCALS
    ثابت size_t ts0 = ggml_type_size(src0->type);
    ثابت size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ثابت int64_t ss0 = nb00 / ts0;
    ثابت int64_t ss1 = nb01 / ts0;
    ثابت int64_t ss2 = nb02 / ts0;
    ثابت int64_t ss3 = nb03 / ts0;
    ثابت int64_t ds0 = nb0 / tdst;
    ثابت int64_t ds1 = nb1 / tdst;
    ثابت int64_t ds2 = nb2 / tdst;
    ثابت int64_t ds3 = nb3 / tdst;
    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
}

// Fused rms_norm + scale (the qwen35 GDN l2-norm pair build_gdn_l2_norm emits):
// the scale factor is a host scalar in the GGML_OP_SCALE node's op_params, so
// unlike the mul variants there is no second device tensor to wire up.
تهی ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       ggml_tensor * scale_tensor) {
    ثابت ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(src0->data);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(scale_tensor->data);

    اعشاری eps;
    memcpy(&eps, dst->op_params, اندازه(اعشاری));
    اعشاری scale_mul;
    memcpy(&scale_mul, scale_tensor->op_params, اندازه(اعشاری));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    ثابت size_t ts0 = ggml_type_size(src0->type);
    ثابت size_t tdst = ggml_type_size(scale_tensor->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(scale_tensor->nb[0] % tdst == 0 && scale_tensor->nb[1] % tdst == 0 &&
                scale_tensor->nb[2] % tdst == 0 && scale_tensor->nb[3] % tdst == 0);
    ثابت int64_t ss0 = nb00 / ts0;
    ثابت int64_t ss1 = nb01 / ts0;
    ثابت int64_t ss2 = nb02 / ts0;
    ثابت int64_t ss3 = nb03 / ts0;
    ثابت int64_t ds0 = scale_tensor->nb[0] / tdst;
    ثابت int64_t ds1 = scale_tensor->nb[1] / tdst;
    ثابت int64_t ds2 = scale_tensor->nb[2] / tdst;
    ثابت int64_t ds3 = scale_tensor->nb[3] / tdst;
    rms_norm_scale_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_stream, ctx.device);
}

تهی ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    ثابت ggml_tensor * rms_norm_src = dst->src[0];
    اعشاری eps = 0.0f;
    memcpy(&eps, dst->op_params, اندازه(اعشاری));

    ثابت اعشاری *       src0_dd = static_cast<ثابت اعشاری *>(rms_norm_src->data);
    ثابت اعشاری *       mul_dd  = nullptr;
    ثابت ggml_tensor * mul_src = nullptr;
    اگر (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ثابت اعشاری *>(mul_tensor->src[1]->data);
        mul_src = mul_tensor->src[1];
    } وگرنه اگر (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ثابت اعشاری *>(mul_tensor->src[0]->data);
        mul_src = mul_tensor->src[0];
    } وگرنه {
        GGML_ASSERT(نادرست);
    }
    اعشاری * dst_dd = static_cast<اعشاری *>(mul_tensor->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ثابت int64_t ne00 = rms_norm_src->ne[0];
    ثابت int64_t ne01 = rms_norm_src->ne[1];
    ثابت int64_t ne02 = rms_norm_src->ne[2];
    ثابت int64_t ne03 = rms_norm_src->ne[3];

    ثابت size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ثابت int64_t s00 = rms_norm_src->nb[0] / ts0;
    ثابت int64_t s01 = rms_norm_src->nb[1] / ts0;
    ثابت int64_t s02 = rms_norm_src->nb[2] / ts0;
    ثابت int64_t s03 = rms_norm_src->nb[3] / ts0;

    ثابت size_t tdst = ggml_type_size(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    ثابت int64_t d00 = mul_tensor->nb[0] / tdst;
    ثابت int64_t d01 = mul_tensor->nb[1] / tdst;
    ثابت int64_t d02 = mul_tensor->nb[2] / tdst;
    ثابت int64_t d03 = mul_tensor->nb[3] / tdst;

    ثابت size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ثابت int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ثابت int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ثابت int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ثابت صحیح mul_nrows     = mul_src->ne[1];
    ثابت صحیح mul_nchannels = mul_src->ne[2];
    ثابت صحیح mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_stream, ctx.device);
}

تهی ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    ثابت ggml_tensor * rms_norm_src = dst->src[0];
    اعشاری eps = 0.0f;
    memcpy(&eps, dst->op_params, اندازه(اعشاری));

    ثابت اعشاری *       src0_dd = static_cast<ثابت اعشاری *>(rms_norm_src->data);
    ثابت اعشاری *       mul_dd  = nullptr;
    ثابت ggml_tensor * mul_src = nullptr;
    اگر (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ثابت اعشاری *>(mul_tensor->src[1]->data);
        mul_src = mul_tensor->src[1];
    } وگرنه اگر (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ثابت اعشاری *>(mul_tensor->src[0]->data);
        mul_src = mul_tensor->src[0];
    } وگرنه {
        GGML_ASSERT(نادرست);
    }

    ثابت اعشاری *       add_dd  = nullptr;
    ثابت ggml_tensor * add_src = nullptr;
    اگر (add_tensor->src[0] == mul_tensor) {
        add_dd  = static_cast<ثابت اعشاری *>(add_tensor->src[1]->data);
        add_src = add_tensor->src[1];
    } وگرنه اگر (add_tensor->src[1] == mul_tensor) {
        add_dd  = static_cast<ثابت اعشاری *>(add_tensor->src[0]->data);
        add_src = add_tensor->src[0];
    } وگرنه {
        GGML_ASSERT(نادرست);
    }

    اعشاری * dst_dd = static_cast<اعشاری *>(add_tensor->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ثابت int64_t ne00 = rms_norm_src->ne[0];
    ثابت int64_t ne01 = rms_norm_src->ne[1];
    ثابت int64_t ne02 = rms_norm_src->ne[2];
    ثابت int64_t ne03 = rms_norm_src->ne[3];

    ثابت size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ثابت int64_t s00 = rms_norm_src->nb[0] / ts0;
    ثابت int64_t s01 = rms_norm_src->nb[1] / ts0;
    ثابت int64_t s02 = rms_norm_src->nb[2] / ts0;
    ثابت int64_t s03 = rms_norm_src->nb[3] / ts0;

    ثابت size_t tdst = ggml_type_size(add_tensor->type);
    GGML_ASSERT(add_tensor->nb[0] == tdst);
    ثابت int64_t d00 = add_tensor->nb[0] / tdst;
    ثابت int64_t d01 = add_tensor->nb[1] / tdst;
    ثابت int64_t d02 = add_tensor->nb[2] / tdst;
    ثابت int64_t d03 = add_tensor->nb[3] / tdst;

    ثابت size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ثابت int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ثابت int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ثابت int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ثابت صحیح mul_nrows     = mul_src->ne[1];
    ثابت صحیح mul_nchannels = mul_src->ne[2];
    ثابت صحیح mul_nsamples  = mul_src->ne[3];

    ثابت size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    ثابت int64_t add_s01 = add_src->nb[1] / ts_add;
    ثابت int64_t add_s02 = add_src->nb[2] / ts_add;
    ثابت int64_t add_s03 = add_src->nb[3] / ts_add;
    ثابت صحیح add_nrows     = add_src->ne[1];
    ثابت صحیح add_nchannels = add_src->ne[2];
    ثابت صحیح add_nsamples  = add_src->ne[3];

    rms_norm_mul_add_f32_sycl(src0_dd, mul_dd, add_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        add_s01, add_s02, add_s03, add_nrows, add_nchannels, add_nsamples, eps, main_stream, ctx.device);
}

تهی ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    اعشاری eps = 1e-5f;
    std::memcpy(&eps, dst->op_params, اندازه(اعشاری));
    اگر (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    ثابت اعشاری * g_base  = static_cast<ثابت اعشاری *>(dst->src[0]->data); // dz
    ثابت اعشاری * x_base  = static_cast<ثابت اعشاری *>(dst->src[1]->data); // x
          اعشاری * dx_base = static_cast<      اعشاری *>(dst->data);

    ثابت int64_t D  = dst->ne[0];
    ثابت int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (تهی) n3;
    ثابت int64_t N  = ggml_nrows(dst);
    اگر (D == 0 || N == 0) بازگشت;

    ثابت ggml_tensor *G = dst->src[0];
    ثابت ggml_tensor *X = dst->src[1];
    ثابت صحیح ts = (صحیح) ggml_type_size(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    ثابت int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    ثابت int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    ثابت int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::queue_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    ثابت صحیح device_max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
    auto roundup = [](صحیح v, صحیح m) { بازگشت ((v + m - 1) / m) * m; };
    صحیح wg_cap = 256;
    اگر (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    صحیح WG = std::max(WARP_SIZE, std::min(roundup((صحیح)std::min<int64_t>(D, wg_cap), WARP_SIZE), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        ثابت صحیح nwarps_loc = std::max(1, WG / WARP_SIZE);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        auto l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        auto l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ثابت صحیح row = item_ct1.get_group(2);
                ثابت صحیح tid = item_ct1.get_local_id(2);

                ثابت int64_t i1 = row % n1;
                ثابت int64_t i2 = (row / n1) % n2;
                ثابت int64_t i3 = row / (n1 * n2);

                ثابت اعشاری *__restrict x_row = x_base + i3 * xs3 + i2 * xs2 + i1 * xs1;
                ثابت اعشاری *__restrict g_row = g_base + i3 * gs3 + i2 * gs2 + i1 * gs1;
                اعشاری *__restrict d_row       = dx_base + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                اعشاری sum_xx = 0.f, sum_xg = 0.f;
#ifndef GGML_SYCL_RMS_BACK_FAST
                اعشاری c_xx = 0.f, c_xg = 0.f;
#endif
                برای (int64_t col = tid; col < D; col += WG) {
                    ثابت اعشاری xv = x_row[col];
                    ثابت اعشاری gv = g_row[col];
#ifdef GGML_SYCL_RMS_BACK_FAST
                    sum_xx += xv * xv;
                    sum_xg += xv * gv;
#وگرنه
                    اعشاری y1 = xv * xv - c_xx;
                    اعشاری t1 = sum_xx + y1;
                    c_xx = (t1 - sum_xx) - y1;
                    sum_xx = t1;

                    اعشاری y2 = xv * gv - c_xg;
                    اعشاری t2 = sum_xg + y2;
                    c_xg = (t2 - sum_xg) - y2;
                    sum_xg = t2;
#endif
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(sum_xx,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xx
#وگرنه
                    0.f
#endif
                );
                sycl::float2 xg = sycl::float2(sum_xg,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xg
#وگرنه
                    0.f
#endif
                );
                xx = warp_reduce_sum(xx, item_ct1);
                xg = warp_reduce_sum(xg, item_ct1);

                // cross-warp reduction using local memory (single barrier)
                ثابت auto sub_group = item_ct1.get_sub_group();
                ثابت auto sg_id     = sub_group.get_group_linear_id();
                ثابت auto wi_in_sg  = sub_group.get_local_linear_id();
                ثابت صحیح nthreads   = item_ct1.get_local_range(2);
                ثابت صحیح nwarps     = nthreads / WARP_SIZE;

                sycl::float2 xx_total = xx;
                sycl::float2 xg_total = xg;
                اگر (nwarps > 1) {
                    اگر (wi_in_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);

                    اگر (sg_id == 0) {
                        ثابت unsigned wi_u = wi_in_sg;
                        sycl::float2 xx_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_total = warp_reduce_sum(xx_first, item_ct1);
                        xg_total = warp_reduce_sum(xg_first, item_ct1);
                    } وگرنه {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_total = xx;
                        xg_total = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                اعشاری inv_r = 0.f;
                اعشاری coeff = 0.f;
                اگر (tid == 0) {
                    ثابت اعشاری sum_xx_f  = xx_total.x() + xx_total.y();
                    ثابت اعشاری sum_xdz_f = xg_total.x() + xg_total.y();
                    ثابت اعشاری mean_eps  = sum_xx_f / (اعشاری) D + eps;
                    ثابت اعشاری sum_eps   = sum_xx_f + eps * (اعشاری) D;
                    inv_r = sycl::rsqrt(mean_eps);
                    coeff = -sum_xdz_f / sum_eps;
                }
                inv_r = sycl::group_broadcast(item_ct1.get_group(), inv_r);
                coeff = sycl::group_broadcast(item_ct1.get_group(), coeff);

                برای (int64_t col = tid; col < D; col += WG) {
                    d_row[col] = (g_row[col] + coeff * x_row[col]) * inv_r;
                }
            });
    });

}

تهی ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت اعشاری * src0_d = (ثابت اعشاری *) src0->data;
    اعشاری * dst_d = (اعشاری *) dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    اعشاری eps;
    memcpy(&eps, dst->op_params, اندازه(اعشاری));
    GGML_ASSERT(eps >= 0.0f);

    ثابت size_t ts0 = ggml_type_size(src0->type);
    ثابت size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ثابت int64_t ss0 = nb00 / ts0;
    ثابت int64_t ss1 = nb01 / ts0;
    ثابت int64_t ss2 = nb02 / ts0;
    ثابت int64_t ss3 = nb03 / ts0;
    ثابت int64_t ds0 = nb0 / tdst;
    ثابت int64_t ds1 = nb1 / tdst;
    ثابت int64_t ds2 = nb2 / tdst;
    ثابت int64_t ds3 = nb3 / tdst;

    /*support both WARP_SIZE or WARP_32_SIZE in code
      choose by hardware for better performance
    */
    l2_norm_f32_sycl<WARP_SIZE>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
تهی ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, صحیح count) {
    ثابت ggml_tensor * s0 = nodes[0]->src[0];
    ثابت صحیح ncols       = (صحیح) s0->ne[0];
    ثابت صحیح nrows_total = (صحیح) ggml_nrows(s0);
    اعشاری eps;
    memcpy(&eps, nodes[0]->op_params, اندازه(اعشاری));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    برای (صحیح t = 0; t < count; ++t) {
        p.src[t] = (ثابت اعشاری *) nodes[t]->src[0]->data;
        p.dst[t] = (اعشاری *) nodes[t]->data;
    }

    ثابت ggml_tensor * d0 = nodes[0];
    ثابت size_t        ts = ggml_type_size(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (صحیح) s0->ne[1];
    st.ne2 = (صحیح) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_SIZE>(p, st, count, ncols, nrows_total, eps, ctx.stream());
}
