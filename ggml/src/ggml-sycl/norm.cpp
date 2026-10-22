#அந்தர்பாவ "norm.hpp"
#அந்தர்பாவ "ggml-sycl/common.hpp"
#அந்தர்பாவ "ggml-sycl/presets.hpp"

நிலையான வெறுமை norm_f32(மாறிலி தசம* x, தசம* dst, மாறிலி முழு ncols,
    மாறிலி int64_t src_stride_col, மாறிலி int64_t src_stride_row, மாறிலி int64_t src_stride_channel, மாறிலி int64_t src_stride_sample,
    மாறிலி int64_t dst_stride_col, மாறிலி int64_t dst_stride_row, மாறிலி int64_t dst_stride_channel, மாறிலி int64_t dst_stride_sample,
    மாறிலி தசம eps, மாறிலி sycl::nd_item<3>& item_ct1, sycl::float2* s_sum, முழு block_size) {

    மாறிலி முழு nthreads = item_ct1.get_local_range(2);
    மாறிலி முழு sample  = item_ct1.get_group(0);
    மாறிலி முழு channel = item_ct1.get_group(1);
    மாறிலி முழு row     = item_ct1.get_group(2);

    மாறிலி முழு tid = item_ct1.get_local_id(2);
    மாறிலி முழு nwarps = nthreads / WARP_SIZE;

    மாறிலி auto src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    மாறிலி auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x += src_offset;
    dst += dst_offset;

    sycl::float2 mean_var = sycl::float2(0.f, 0.f);

    க்கு (முழு col = tid; col < ncols; col += block_size) {
        மாறிலி தசம xi = x[col * src_stride_col];
        mean_var.x() += xi;
        mean_var.y() += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var, item_ct1);
    எனில்  (block_size > WARP_SIZE) {
        மாறிலி auto sub_group = item_ct1.get_sub_group();
        மாறிலி auto sg_id = sub_group.get_group_linear_id();
        மாறிலி auto wi_in_sg = sub_group.get_local_linear_id();
        எனில் (wi_in_sg == 0) {
            s_sum[sg_id] = mean_var;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        mean_var = 0.f;
        மாறிலி size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        க்கு (size_t i = 0; i < nreduce; i += 1)
        {
            mean_var += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        mean_var = warp_reduce_sum(mean_var, item_ct1);
    }

    மாறிலி தசம mean = mean_var.x() / ncols;
    மாறிலி தசம var = mean_var.y() / ncols - mean * mean;
    மாறிலி தசம inv_std = sycl::rsqrt(var + eps);

    க்கு (முழு col = tid; col < ncols; col += block_size) {
        dst[col * dst_stride_col] = (x[col * src_stride_col] - mean) * inv_std;
    }
}

நிலையான வெறுமை group_norm_f32(மாறிலி தசம* x, தசம* dst, மாறிலி முழு group_size, மாறிலி முழு ne_elements, மாறிலி தசம eps,
    மாறிலி sycl::nd_item<3>& item_ct1, தசம* s_sum, முழு block_size) {
    முழு start = item_ct1.get_group(2) * group_size;
    முழு end = start + group_size;
    மாறிலி முழு nthreads = item_ct1.get_local_range(2);
    மாறிலி முழு nwarps = nthreads / WARP_SIZE;
    start += item_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_SIZE;

    எனில் (end >= ne_elements) {
        end = ne_elements;
    }

    தசம tmp = 0.0f; // partial sum for thread in warp

    க்கு (முழு j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    எனில் (block_size > WARP_SIZE) {

        முழு warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        முழு lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        எனில் (lane_id == 0) {
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
        க்கு (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    தசம mean = tmp / group_size;
    tmp = 0.0f;

    க்கு (முழு j = start; j < end; j += block_size) {
        தசம xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    எனில் (block_size > WARP_SIZE) {

        முழு warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        முழு lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        எனில் (lane_id == 0) {
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
        க்கு (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    தசம variance = tmp / group_size;
    தசம scale = sycl::rsqrt(variance + eps);
    க்கு (முழு j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <தருக்க do_multiply = பொய், தருக்க do_add = பொய்>
நிலையான வெறுமை rms_norm_f32(மாறிலி தசம* x, தசம* dst, மாறிலி முழு ncols,
    மாறிலி int64_t src_stride_col, மாறிலி int64_t src_stride_row, மாறிலி int64_t src_stride_channel, மாறிலி int64_t src_stride_sample,
    மாறிலி int64_t dst_stride_col, மாறிலி int64_t dst_stride_row, மாறிலி int64_t dst_stride_channel, மாறிலி int64_t dst_stride_sample,
    மாறிலி தசம eps, மாறிலி sycl::nd_item<3>& item_ct1, தசம* s_sum, முழு block_size,
    மாறிலி தசம* mul = nullptr, மாறிலி int64_t mul_stride_row = 0, மாறிலி int64_t mul_stride_channel = 0,
    மாறிலி int64_t mul_stride_sample = 0, மாறிலி முழு mul_nrows = 0, மாறிலி முழு mul_nchannels = 0, மாறிலி முழு mul_nsamples = 0,
    மாறிலி தசம* add = nullptr, மாறிலி int64_t add_stride_row = 0, மாறிலி int64_t add_stride_channel = 0,
    மாறிலி int64_t add_stride_sample = 0, மாறிலி முழு add_nrows = 0, மாறிலி முழு add_nchannels = 0, மாறிலி முழு add_nsamples = 0,
    மாறிலி தசம scale_mul = 1.0f) {

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    மாறிலி முழு sample  = item_ct1.get_group(0);
    மாறிலி முழு channel = item_ct1.get_group(1);
    மாறிலி முழு row     = item_ct1.get_group(2);

    மாறிலி முழு nthreads = item_ct1.get_local_range(2);

    மாறிலி முழு tid = item_ct1.get_local_id(2);
    மாறிலி முழு nwarps = nthreads / WARP_SIZE;

    மாறிலி auto src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    மாறிலி auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x   += src_offset;
    dst += dst_offset;

    எனில் constexpr (do_multiply) {
        மாறிலி முழு mul_row     = row     % mul_nrows;
        மாறிலி முழு mul_channel = channel % mul_nchannels;
        மாறிலி முழு mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    எனில் constexpr (do_add) {
        மாறிலி முழு add_row     = row     % add_nrows;
        மாறிலி முழு add_channel = channel % add_nchannels;
        மாறிலி முழு add_sample  = sample  % add_nsamples;
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    தசம tmp = 0.0f; // partial sum for thread in warp

    க்கு (முழு col = tid; col < ncols; col += block_size) {
        மாறிலி தசம xi = x[col * src_stride_col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    எனில் (block_size > WARP_SIZE) {
        மாறிலி auto sub_group = item_ct1.get_sub_group();
        மாறிலி auto sg_id = sub_group.get_group_linear_id();
        மாறிலி auto wi_in_sg = sub_group.get_local_linear_id();
        எனில் (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        மாறிலி size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        க்கு (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    மாறிலி தசம mean = tmp / ncols;
    மாறிலி தசம scale = sycl::rsqrt(mean + eps);

    க்கு (முழு col = tid; col < ncols; col += block_size) {
        எனில் constexpr (do_multiply && do_add) {
            dst[col * dst_stride_col] = scale * x[col * src_stride_col] * mul[col] + add[col];
        } இல்லையெனில் எனில் constexpr (do_multiply) {
            dst[col * dst_stride_col] = scale * x[col * src_stride_col] * mul[col];
        } இல்லையெனில் {
            // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
            // the explicit temporary keeps the float evaluation order identical to
            // running rms_norm and scale as two separate kernels
            மாறிலி தசம v = scale * x[col * src_stride_col];
            dst[col * dst_stride_col] = v * scale_mul;
        }
    }
}

template<முழு warp_size>
நிலையான வெறுமை l2_norm_f32(மாறிலி தசம * x, தசம * dst, மாறிலி முழு ncols,
    மாறிலி int64_t src_stride_col, மாறிலி int64_t src_stride_row, மாறிலி int64_t src_stride_channel,
    மாறிலி int64_t src_stride_sample, மாறிலி int64_t dst_stride_col, மாறிலி int64_t dst_stride_row,
    மாறிலி int64_t dst_stride_channel, மாறிலி int64_t dst_stride_sample, மாறிலி தசம eps,
    மாறிலி sycl::nd_item<3>& item_ct1, தசம* s_sum, மாறிலி முழு block_size) {

    மாறிலி முழு row     = item_ct1.get_group(2);
    மாறிலி முழு channel = item_ct1.get_group(1);
    மாறிலி முழு sample  = item_ct1.get_group(0);
    மாறிலி முழு tid     = item_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + row*src_stride_row;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + row*dst_stride_row;

    தசம tmp = 0.0f; // partial sum for thread in warp

    க்கு (முழு col = tid; col < ncols; col += block_size) {
        மாறிலி தசம xi = x[col * src_stride_col];
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, s_sum, block_size);
    மாறிலி தசம scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    க்கு (முழு col = tid; col < ncols; col += block_size) {
        dst[col * dst_stride_col] = scale * x[col * src_stride_col];
    }
}

நிலையான வெறுமை norm_f32_sycl(மாறிலி தசம * x, தசம * dst, மாறிலி முழு ncols, மாறிலி முழு nrows, மாறிலி முழு nchannels, மாறிலி முழு nsamples,
    மாறிலி int64_t src_stride_col, மாறிலி int64_t src_stride_row, மாறிலி int64_t src_stride_channel, மாறிலி int64_t src_stride_sample,
    மாறிலி int64_t dst_stride_col, மாறிலி int64_t dst_stride_row, மாறிலி int64_t dst_stride_channel, மாறிலி int64_t dst_stride_sample,
        மாறிலி தசம eps, queue_ptr stream, முழு device) {

    மாறிலி sycl::range<3> global_dims(nsamples, nchannels, nrows);
    எனில் (ncols < 1024) {
        மாறிலி sycl::range<3> block_dims(1, 1, WARP_SIZE);
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
    இல்லையெனில் {
        மாறிலி முழு work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        ப்ரதிஜ்ஞா(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        மாறிலி sycl::range<3> block_dims(1, 1, work_group_size);
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

நிலையான வெறுமை group_norm_f32_sycl(மாறிலி தசம* x, தசம* dst,
    மாறிலி முழு இலக்கம்_groups, மாறிலி தசம eps, மாறிலி முழு group_size,
    மாறிலி முழு ne_elements, queue_ptr stream, முழு device) {
    எனில் (group_size < 1024) {
        மாறிலி sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            மாறிலி தசம eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(
                        x, dst, group_size, ne_elements, eps_ct4, item_ct1,
                        nullptr, WARP_SIZE);
                });
            });
    }
    இல்லையெனில் {
        மாறிலி முழு work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        ப்ரதிஜ்ஞா(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        மாறிலி sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<தசம, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);

            மாறிலி தசம eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_groups) * block_dims,
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

நிலையான வெறுமை rms_norm_f32_sycl(மாறிலி தசம* x, தசம* dst, மாறிலி முழு ncols, மாறிலி முழு nrows, மாறிலி முழு nchannels, மாறிலி முழு nsamples,
    மாறிலி int64_t src_stride_col, மாறிலி int64_t src_stride_row, மாறிலி int64_t src_stride_channel, மாறிலி int64_t src_stride_sample,
    மாறிலி int64_t dst_stride_col, மாறிலி int64_t dst_stride_row, மாறிலி int64_t dst_stride_channel, மாறிலி int64_t dst_stride_sample,
    மாறிலி தசம eps, queue_ptr stream, முழு device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    மாறிலி sycl::range<3> global_dims(nsamples, nchannels, nrows);
    எனில் (ncols < 1024) {
        மாறிலி sycl::range<3> block_dims(1, 1, WARP_SIZE);
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
    இல்லையெனில் {
        மாறிலி முழு work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        ப்ரதிஜ்ஞா(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        மாறிலி sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<தசம, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
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

நிலையான வெறுமை rms_norm_scale_f32_sycl(மாறிலி தசம* x, தசம* dst, மாறிலி முழு ncols, மாறிலி முழு nrows,
    மாறிலி முழு nchannels, மாறிலி முழு nsamples,
    மாறிலி int64_t src_stride_col, மாறிலி int64_t src_stride_row, மாறிலி int64_t src_stride_channel, மாறிலி int64_t src_stride_sample,
    மாறிலி int64_t dst_stride_col, மாறிலி int64_t dst_stride_row, மாறிலி int64_t dst_stride_channel, மாறிலி int64_t dst_stride_sample,
    மாறிலி தசம eps, மாறிலி தசம scale_mul, queue_ptr stream, முழு device) {
    மாறிலி sycl::range<3> global_dims(nsamples, nchannels, nrows);
    எனில் (ncols < 1024) {
        மாறிலி sycl::range<3> block_dims(1, 1, WARP_SIZE);
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
    இல்லையெனில் {
        மாறிலி முழு work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        ப்ரதிஜ்ஞா(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        மாறிலி sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<தசம, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
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

நிலையான வெறுமை rms_norm_mul_f32_sycl(மாறிலி தசம* x, மாறிலி தசம* mul, தசம* dst, மாறிலி முழு ncols, மாறிலி முழு nrows,
        மாறிலி முழு nchannels, மாறிலி முழு nsamples,
        மாறிலி int64_t src_stride_col, மாறிலி int64_t src_stride_row, மாறிலி int64_t src_stride_channel, மாறிலி int64_t src_stride_sample,
        மாறிலி int64_t dst_stride_col, மாறிலி int64_t dst_stride_row, மாறிலி int64_t dst_stride_channel, மாறிலி int64_t dst_stride_sample,
        மாறிலி int64_t mul_stride_row, மாறிலி int64_t mul_stride_channel, மாறிலி int64_t mul_stride_sample,
        மாறிலி முழு mul_nrows, மாறிலி முழு mul_nchannels, மாறிலி முழு mul_nsamples,
        மாறிலி தசம eps, queue_ptr stream, முழு device) {
    மாறிலி sycl::range<3> global_dims(nsamples, nchannels, nrows);
    எனில் (ncols < 1024) {
        மாறிலி sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<உண்மை>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    இல்லையெனில் {
        மாறிலி முழு work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        ப்ரதிஜ்ஞா(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        மாறிலி sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<தசம, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<உண்மை>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

நிலையான வெறுமை rms_norm_mul_add_f32_sycl(மாறிலி தசம* x, மாறிலி தசம* mul, மாறிலி தசம* add, தசம* dst,
        மாறிலி முழு ncols, மாறிலி முழு nrows, மாறிலி முழு nchannels, மாறிலி முழு nsamples,
        மாறிலி int64_t src_stride_col, மாறிலி int64_t src_stride_row, மாறிலி int64_t src_stride_channel, மாறிலி int64_t src_stride_sample,
        மாறிலி int64_t dst_stride_col, மாறிலி int64_t dst_stride_row, மாறிலி int64_t dst_stride_channel, மாறிலி int64_t dst_stride_sample,
        மாறிலி int64_t mul_stride_row, மாறிலி int64_t mul_stride_channel, மாறிலி int64_t mul_stride_sample,
        மாறிலி முழு mul_nrows, மாறிலி முழு mul_nchannels, மாறிலி முழு mul_nsamples,
        மாறிலி int64_t add_stride_row, மாறிலி int64_t add_stride_channel, மாறிலி int64_t add_stride_sample,
        மாறிலி முழு add_nrows, மாறிலி முழு add_nchannels, மாறிலி முழு add_nsamples,
        மாறிலி தசம eps, queue_ptr stream, முழு device) {
    மாறிலி sycl::range<3> global_dims(nsamples, nchannels, nrows);
    எனில் (ncols < 1024) {
        மாறிலி sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<உண்மை, உண்மை>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add, add_stride_row, add_stride_channel, add_stride_sample, add_nrows, add_nchannels, add_nsamples);
                });
            });
    }
    இல்லையெனில் {
        மாறிலி முழு work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        ப்ரதிஜ்ஞா(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        மாறிலி sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<தசம, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<உண்மை, உண்மை>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add, add_stride_row, add_stride_channel, add_stride_sample, add_nrows, add_nchannels, add_nsamples);
                });
            });
    }
}

template<முழு warp_size>
நிலையான வெறுமை l2_norm_f32_sycl(மாறிலி தசம *   x,
                             தசம *         dst,
                             மாறிலி முழு       ncols,
                             மாறிலி முழு       nrows,
                             மாறிலி முழு       nchannels,
                             மாறிலி முழு       nsamples,
                             மாறிலி int64_t   src_stride_col,
                             மாறிலி int64_t   src_stride_row,
                             மாறிலி int64_t   src_stride_channel,
                             மாறிலி int64_t   src_stride_sample,
                             மாறிலி int64_t   dst_stride_col,
                             மாறிலி int64_t   dst_stride_row,
                             மாறிலி int64_t   dst_stride_channel,
                             மாறிலி int64_t   dst_stride_sample,
                             மாறிலி தசம     eps,
                             queue_ptr       stream,
                             முழு             device) {
    மாறிலி dpct::dim3 blocks_இலக்கம்(nrows, nchannels, nsamples);

    எனில் (ncols < 1024) {
        மாறிலி dpct::dim3 block_dims(warp_size, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(blocks_இலக்கம் * block_dims,
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
    இல்லையெனில் {
        மாறிலி முழு work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        ப்ரதிஜ்ஞா(work_group_size % (warp_size * warp_size) == 0);
        மாறிலி sycl::range<3> block_dims(1, 1, work_group_size);
        முழு lsm_size =  block_dims[2] > warp_size ? work_group_size / warp_size * அளவு(தசம): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<தசம, 1> s_sum_acc_ct1(sycl::range<1>(lsm_size),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(blocks_இலக்கம் * block_dims,
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
கட்டமைப்பு l2_batch_ptrs {
    மாறிலி தசம * src[GGML_SYCL_L2_BATCH_MAX];
    தசம *       dst[GGML_SYCL_L2_BATCH_MAX];
};

// One stride set shared by the whole batch: the caller only groups tensors whose nb[]
// all match, so per-tensor state stays two pointers.
கட்டமைப்பு l2_batch_strides {
    முழு     ne1, ne2;
    int64_t ss0, ss1, ss2, ss3;
    int64_t ds0, ds1, ds2, ds3;
};

template <முழு warp_size>
நிலையான வெறுமை l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, மாறிலி முழு ncols, மாறிலி தசம eps,
                              மாறிலி sycl::nd_item<3> & item_ct1) {
    மாறிலி முழு t   = item_ct1.get_group(0);  // tensor index
    மாறிலி முழு r   = item_ct1.get_group(2);  // flattened row over ne1*ne2*ne3
    மாறிலி முழு tid = item_ct1.get_local_id(2);

    மாறிலி முழு i1 = r % st.ne1;
    மாறிலி முழு i2 = (r / st.ne1) % st.ne2;
    மாறிலி முழு i3 = r / (st.ne1 * st.ne2);

    மாறிலி தசம * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    தசம *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    தசம tmp = 0.0f;
    க்கு (முழு col = tid; col < ncols; col += warp_size) {
        மாறிலி தசம xi = x[col * st.ss0];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, (தசம *) nullptr, warp_size);
    மாறிலி தசம scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    க்கு (முழு col = tid; col < ncols; col += warp_size) {
        dst[col * st.ds0] = scale * x[col * st.ss0];
    }
}

template <முழு warp_size>
நிலையான வெறுமை l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, மாறிலி முழு n_tensors,
                                   மாறிலி முழு ncols, மாறிலி முழு nrows_total, மாறிலி தசம eps,
                                   queue_ptr stream) {
    மாறிலி dpct::dim3 blocks_இலக்கம்(nrows_total, 1, n_tensors);
    மாறிலி dpct::dim3 block_dims(warp_size, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(blocks_இலக்கம் * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                l2_norm_f32_batch<warp_size>(p, st, ncols, eps, item_ct1);
            });
    });
}

வெறுமை ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    மாறிலி தசம * src0_dd = static_cast<மாறிலி தசம *>(dst->src[0]->data);
    தசம *       dst_dd  = static_cast<தசம *>(dst->data);

    தசம eps;
    memcpy(&eps, dst->op_params, அளவு(தசம));
    GGML_ASSERT(eps >= 0.0f);
    மாறிலி size_t ts0 = ggml_type_size(src0->type);
    மாறிலி size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    மாறிலி int64_t ss0 = nb00 / ts0;
    மாறிலி int64_t ss1 = nb01 / ts0;
    மாறிலி int64_t ss2 = nb02 / ts0;
    மாறிலி int64_t ss3 = nb03 / ts0;
    மாறிலி int64_t ds0 = nb0 / tdst;
    மாறிலி int64_t ds1 = nb1 / tdst;
    மாறிலி int64_t ds2 = nb2 / tdst;
    மாறிலி int64_t ds3 = nb3 / tdst;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
}

வெறுமை ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    முழு இலக்கம்_groups = dst->op_params[0];
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    மாறிலி தசம * src0_dd = static_cast<மாறிலி தசம *>(dst->src[0]->data);
    தசம *       dst_dd  = static_cast<தசம *>(dst->data);

    தசம eps;
    memcpy(&eps, dst->op_params + 1, அளவு(தசம));

    முழு group_size = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + இலக்கம்_groups - 1) / இலக்கம்_groups);
    group_norm_f32_sycl(src0_dd, dst_dd, இலக்கம்_groups, eps, group_size, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_stream, ctx.device);
}

வெறுமை ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    மாறிலி தசம * src0_dd = static_cast<மாறிலி தசம *>(dst->src[0]->data);
    தசம *       dst_dd  = static_cast<தசம *>(dst->data);

    தசம eps;
    memcpy(&eps, dst->op_params, அளவு(தசம));

    GGML_TENSOR_UNARY_OP_LOCALS
    மாறிலி size_t ts0 = ggml_type_size(src0->type);
    மாறிலி size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    மாறிலி int64_t ss0 = nb00 / ts0;
    மாறிலி int64_t ss1 = nb01 / ts0;
    மாறிலி int64_t ss2 = nb02 / ts0;
    மாறிலி int64_t ss3 = nb03 / ts0;
    மாறிலி int64_t ds0 = nb0 / tdst;
    மாறிலி int64_t ds1 = nb1 / tdst;
    மாறிலி int64_t ds2 = nb2 / tdst;
    மாறிலி int64_t ds3 = nb3 / tdst;
    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
}

// Fused rms_norm + scale (the qwen35 GDN l2-norm pair build_gdn_l2_norm emits):
// the scale factor is a host scalar in the GGML_OP_SCALE node's op_params, so
// unlike the mul variants there is no second device tensor to wire up.
வெறுமை ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       ggml_tensor * scale_tensor) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    மாறிலி தசம * src0_dd = static_cast<மாறிலி தசம *>(src0->data);
    தசம *       dst_dd  = static_cast<தசம *>(scale_tensor->data);

    தசம eps;
    memcpy(&eps, dst->op_params, அளவு(தசம));
    தசம scale_mul;
    memcpy(&scale_mul, scale_tensor->op_params, அளவு(தசம));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    மாறிலி size_t ts0 = ggml_type_size(src0->type);
    மாறிலி size_t tdst = ggml_type_size(scale_tensor->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(scale_tensor->nb[0] % tdst == 0 && scale_tensor->nb[1] % tdst == 0 &&
                scale_tensor->nb[2] % tdst == 0 && scale_tensor->nb[3] % tdst == 0);
    மாறிலி int64_t ss0 = nb00 / ts0;
    மாறிலி int64_t ss1 = nb01 / ts0;
    மாறிலி int64_t ss2 = nb02 / ts0;
    மாறிலி int64_t ss3 = nb03 / ts0;
    மாறிலி int64_t ds0 = scale_tensor->nb[0] / tdst;
    மாறிலி int64_t ds1 = scale_tensor->nb[1] / tdst;
    மாறிலி int64_t ds2 = scale_tensor->nb[2] / tdst;
    மாறிலி int64_t ds3 = scale_tensor->nb[3] / tdst;
    rms_norm_scale_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_stream, ctx.device);
}

வெறுமை ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    மாறிலி ggml_tensor * rms_norm_src = dst->src[0];
    தசம eps = 0.0f;
    memcpy(&eps, dst->op_params, அளவு(தசம));

    மாறிலி தசம *       src0_dd = static_cast<மாறிலி தசம *>(rms_norm_src->data);
    மாறிலி தசம *       mul_dd  = nullptr;
    மாறிலி ggml_tensor * mul_src = nullptr;
    எனில் (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<மாறிலி தசம *>(mul_tensor->src[1]->data);
        mul_src = mul_tensor->src[1];
    } இல்லையெனில் எனில் (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<மாறிலி தசம *>(mul_tensor->src[0]->data);
        mul_src = mul_tensor->src[0];
    } இல்லையெனில் {
        GGML_ASSERT(பொய்);
    }
    தசம * dst_dd = static_cast<தசம *>(mul_tensor->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    மாறிலி int64_t ne00 = rms_norm_src->ne[0];
    மாறிலி int64_t ne01 = rms_norm_src->ne[1];
    மாறிலி int64_t ne02 = rms_norm_src->ne[2];
    மாறிலி int64_t ne03 = rms_norm_src->ne[3];

    மாறிலி size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    மாறிலி int64_t s00 = rms_norm_src->nb[0] / ts0;
    மாறிலி int64_t s01 = rms_norm_src->nb[1] / ts0;
    மாறிலி int64_t s02 = rms_norm_src->nb[2] / ts0;
    மாறிலி int64_t s03 = rms_norm_src->nb[3] / ts0;

    மாறிலி size_t tdst = ggml_type_size(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    மாறிலி int64_t d00 = mul_tensor->nb[0] / tdst;
    மாறிலி int64_t d01 = mul_tensor->nb[1] / tdst;
    மாறிலி int64_t d02 = mul_tensor->nb[2] / tdst;
    மாறிலி int64_t d03 = mul_tensor->nb[3] / tdst;

    மாறிலி size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    மாறிலி int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    மாறிலி int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    மாறிலி int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    மாறிலி முழு mul_nrows     = mul_src->ne[1];
    மாறிலி முழு mul_nchannels = mul_src->ne[2];
    மாறிலி முழு mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_stream, ctx.device);
}

வெறுமை ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    மாறிலி ggml_tensor * rms_norm_src = dst->src[0];
    தசம eps = 0.0f;
    memcpy(&eps, dst->op_params, அளவு(தசம));

    மாறிலி தசம *       src0_dd = static_cast<மாறிலி தசம *>(rms_norm_src->data);
    மாறிலி தசம *       mul_dd  = nullptr;
    மாறிலி ggml_tensor * mul_src = nullptr;
    எனில் (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<மாறிலி தசம *>(mul_tensor->src[1]->data);
        mul_src = mul_tensor->src[1];
    } இல்லையெனில் எனில் (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<மாறிலி தசம *>(mul_tensor->src[0]->data);
        mul_src = mul_tensor->src[0];
    } இல்லையெனில் {
        GGML_ASSERT(பொய்);
    }

    மாறிலி தசம *       add_dd  = nullptr;
    மாறிலி ggml_tensor * add_src = nullptr;
    எனில் (add_tensor->src[0] == mul_tensor) {
        add_dd  = static_cast<மாறிலி தசம *>(add_tensor->src[1]->data);
        add_src = add_tensor->src[1];
    } இல்லையெனில் எனில் (add_tensor->src[1] == mul_tensor) {
        add_dd  = static_cast<மாறிலி தசம *>(add_tensor->src[0]->data);
        add_src = add_tensor->src[0];
    } இல்லையெனில் {
        GGML_ASSERT(பொய்);
    }

    தசம * dst_dd = static_cast<தசம *>(add_tensor->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    மாறிலி int64_t ne00 = rms_norm_src->ne[0];
    மாறிலி int64_t ne01 = rms_norm_src->ne[1];
    மாறிலி int64_t ne02 = rms_norm_src->ne[2];
    மாறிலி int64_t ne03 = rms_norm_src->ne[3];

    மாறிலி size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    மாறிலி int64_t s00 = rms_norm_src->nb[0] / ts0;
    மாறிலி int64_t s01 = rms_norm_src->nb[1] / ts0;
    மாறிலி int64_t s02 = rms_norm_src->nb[2] / ts0;
    மாறிலி int64_t s03 = rms_norm_src->nb[3] / ts0;

    மாறிலி size_t tdst = ggml_type_size(add_tensor->type);
    GGML_ASSERT(add_tensor->nb[0] == tdst);
    மாறிலி int64_t d00 = add_tensor->nb[0] / tdst;
    மாறிலி int64_t d01 = add_tensor->nb[1] / tdst;
    மாறிலி int64_t d02 = add_tensor->nb[2] / tdst;
    மாறிலி int64_t d03 = add_tensor->nb[3] / tdst;

    மாறிலி size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    மாறிலி int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    மாறிலி int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    மாறிலி int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    மாறிலி முழு mul_nrows     = mul_src->ne[1];
    மாறிலி முழு mul_nchannels = mul_src->ne[2];
    மாறிலி முழு mul_nsamples  = mul_src->ne[3];

    மாறிலி size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    மாறிலி int64_t add_s01 = add_src->nb[1] / ts_add;
    மாறிலி int64_t add_s02 = add_src->nb[2] / ts_add;
    மாறிலி int64_t add_s03 = add_src->nb[3] / ts_add;
    மாறிலி முழு add_nrows     = add_src->ne[1];
    மாறிலி முழு add_nchannels = add_src->ne[2];
    மாறிலி முழு add_nsamples  = add_src->ne[3];

    rms_norm_mul_add_f32_sycl(src0_dd, mul_dd, add_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        add_s01, add_s02, add_s03, add_nrows, add_nchannels, add_nsamples, eps, main_stream, ctx.device);
}

வெறுமை ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    தசம eps = 1e-5f;
    std::memcpy(&eps, dst->op_params, அளவு(தசம));
    எனில் (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    மாறிலி தசம * g_base  = static_cast<மாறிலி தசம *>(dst->src[0]->data); // dz
    மாறிலி தசம * x_base  = static_cast<மாறிலி தசம *>(dst->src[1]->data); // x
          தசம * dx_base = static_cast<      தசம *>(dst->data);

    மாறிலி int64_t D  = dst->ne[0];
    மாறிலி int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (வெறுமை) n3;
    மாறிலி int64_t N  = ggml_nrows(dst);
    எனில் (D == 0 || N == 0) திரும்பு;

    மாறிலி ggml_tensor *G = dst->src[0];
    மாறிலி ggml_tensor *X = dst->src[1];
    மாறிலி முழு ts = (முழு) ggml_type_size(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    மாறிலி int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    மாறிலி int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    மாறிலி int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::queue_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    மாறிலி முழு device_max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
    auto roundup = [](முழு v, முழு m) { திரும்பு ((v + m - 1) / m) * m; };
    முழு wg_cap = 256;
    எனில் (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    முழு WG = std::max(WARP_SIZE, std::min(roundup((முழு)std::min<int64_t>(D, wg_cap), WARP_SIZE), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        மாறிலி முழு nwarps_loc = std::max(1, WG / WARP_SIZE);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        auto l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        auto l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                மாறிலி முழு row = item_ct1.get_group(2);
                மாறிலி முழு tid = item_ct1.get_local_id(2);

                மாறிலி int64_t i1 = row % n1;
                மாறிலி int64_t i2 = (row / n1) % n2;
                மாறிலி int64_t i3 = row / (n1 * n2);

                மாறிலி தசம *__restrict x_row = x_base + i3 * xs3 + i2 * xs2 + i1 * xs1;
                மாறிலி தசம *__restrict g_row = g_base + i3 * gs3 + i2 * gs2 + i1 * gs1;
                தசம *__restrict d_row       = dx_base + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                தசம sum_xx = 0.f, sum_xg = 0.f;
#ifndef GGML_SYCL_RMS_BACK_FAST
                தசம c_xx = 0.f, c_xg = 0.f;
#endif
                க்கு (int64_t col = tid; col < D; col += WG) {
                    மாறிலி தசம xv = x_row[col];
                    மாறிலி தசம gv = g_row[col];
#ifdef GGML_SYCL_RMS_BACK_FAST
                    sum_xx += xv * xv;
                    sum_xg += xv * gv;
#இல்லையெனில்
                    தசம y1 = xv * xv - c_xx;
                    தசம t1 = sum_xx + y1;
                    c_xx = (t1 - sum_xx) - y1;
                    sum_xx = t1;

                    தசம y2 = xv * gv - c_xg;
                    தசம t2 = sum_xg + y2;
                    c_xg = (t2 - sum_xg) - y2;
                    sum_xg = t2;
#endif
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(sum_xx,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xx
#இல்லையெனில்
                    0.f
#endif
                );
                sycl::float2 xg = sycl::float2(sum_xg,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xg
#இல்லையெனில்
                    0.f
#endif
                );
                xx = warp_reduce_sum(xx, item_ct1);
                xg = warp_reduce_sum(xg, item_ct1);

                // cross-warp reduction using local memory (single barrier)
                மாறிலி auto sub_group = item_ct1.get_sub_group();
                மாறிலி auto sg_id     = sub_group.get_group_linear_id();
                மாறிலி auto wi_in_sg  = sub_group.get_local_linear_id();
                மாறிலி முழு nthreads   = item_ct1.get_local_range(2);
                மாறிலி முழு nwarps     = nthreads / WARP_SIZE;

                sycl::float2 xx_total = xx;
                sycl::float2 xg_total = xg;
                எனில் (nwarps > 1) {
                    எனில் (wi_in_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);

                    எனில் (sg_id == 0) {
                        மாறிலி unsigned wi_u = wi_in_sg;
                        sycl::float2 xx_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_total = warp_reduce_sum(xx_first, item_ct1);
                        xg_total = warp_reduce_sum(xg_first, item_ct1);
                    } இல்லையெனில் {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_total = xx;
                        xg_total = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                தசம inv_r = 0.f;
                தசம coeff = 0.f;
                எனில் (tid == 0) {
                    மாறிலி தசம sum_xx_f  = xx_total.x() + xx_total.y();
                    மாறிலி தசம sum_xdz_f = xg_total.x() + xg_total.y();
                    மாறிலி தசம mean_eps  = sum_xx_f / (தசம) D + eps;
                    மாறிலி தசம sum_eps   = sum_xx_f + eps * (தசம) D;
                    inv_r = sycl::rsqrt(mean_eps);
                    coeff = -sum_xdz_f / sum_eps;
                }
                inv_r = sycl::group_broadcast(item_ct1.get_group(), inv_r);
                coeff = sycl::group_broadcast(item_ct1.get_group(), coeff);

                க்கு (int64_t col = tid; col < D; col += WG) {
                    d_row[col] = (g_row[col] + coeff * x_row[col]) * inv_r;
                }
            });
    });

}

வெறுமை ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி தசம * src0_d = (மாறிலி தசம *) src0->data;
    தசம * dst_d = (தசம *) dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    தசம eps;
    memcpy(&eps, dst->op_params, அளவு(தசம));
    GGML_ASSERT(eps >= 0.0f);

    மாறிலி size_t ts0 = ggml_type_size(src0->type);
    மாறிலி size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    மாறிலி int64_t ss0 = nb00 / ts0;
    மாறிலி int64_t ss1 = nb01 / ts0;
    மாறிலி int64_t ss2 = nb02 / ts0;
    மாறிலி int64_t ss3 = nb03 / ts0;
    மாறிலி int64_t ds0 = nb0 / tdst;
    மாறிலி int64_t ds1 = nb1 / tdst;
    மாறிலி int64_t ds2 = nb2 / tdst;
    மாறிலி int64_t ds3 = nb3 / tdst;

    /*support both WARP_SIZE or WARP_32_SIZE in code
      choose by hardware for better performance
    */
    l2_norm_f32_sycl<WARP_SIZE>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
வெறுமை ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, முழு count) {
    மாறிலி ggml_tensor * s0 = nodes[0]->src[0];
    மாறிலி முழு ncols       = (முழு) s0->ne[0];
    மாறிலி முழு nrows_total = (முழு) ggml_nrows(s0);
    தசம eps;
    memcpy(&eps, nodes[0]->op_params, அளவு(தசம));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    க்கு (முழு t = 0; t < count; ++t) {
        p.src[t] = (மாறிலி தசம *) nodes[t]->src[0]->data;
        p.dst[t] = (தசம *) nodes[t]->data;
    }

    மாறிலி ggml_tensor * d0 = nodes[0];
    மாறிலி size_t        ts = ggml_type_size(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (முழு) s0->ne[1];
    st.ne2 = (முழு) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_SIZE>(p, st, count, ncols, nrows_total, eps, ctx.stream());
}
