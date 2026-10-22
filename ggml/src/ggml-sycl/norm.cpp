#अन्तर्भाव "norm.hpp"
#अन्तर्भाव "ggml-sycl/common.hpp"
#अन्तर्भाव "ggml-sycl/presets.hpp"

स्थिर शून्य norm_f32(ध्रुव भिन्न* x, भिन्न* dst, ध्रुव पूर्णाङ्क ncols,
    ध्रुव int64_t src_stride_col, ध्रुव int64_t src_stride_row, ध्रुव int64_t src_stride_channel, ध्रुव int64_t src_stride_sample,
    ध्रुव int64_t dst_stride_col, ध्रुव int64_t dst_stride_row, ध्रुव int64_t dst_stride_channel, ध्रुव int64_t dst_stride_sample,
    ध्रुव भिन्न eps, ध्रुव sycl::nd_item<3>& item_ct1, sycl::float2* s_sum, पूर्णाङ्क block_size) {

    ध्रुव पूर्णाङ्क nthreads = item_ct1.get_local_range(2);
    ध्रुव पूर्णाङ्क sample  = item_ct1.get_group(0);
    ध्रुव पूर्णाङ्क channel = item_ct1.get_group(1);
    ध्रुव पूर्णाङ्क row     = item_ct1.get_group(2);

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);
    ध्रुव पूर्णाङ्क nwarps = nthreads / WARP_SIZE;

    ध्रुव स्वतः src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    ध्रुव स्वतः dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x += src_offset;
    dst += dst_offset;

    sycl::float2 mean_var = sycl::float2(0.f, 0.f);

    प्रति (पूर्णाङ्क col = tid; col < ncols; col += block_size) {
        ध्रुव भिन्न xi = x[col * src_stride_col];
        mean_var.x() += xi;
        mean_var.y() += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var, item_ct1);
    यदि  (block_size > WARP_SIZE) {
        ध्रुव स्वतः sub_group = item_ct1.get_sub_group();
        ध्रुव स्वतः sg_id = sub_group.get_group_linear_id();
        ध्रुव स्वतः wi_in_sg = sub_group.get_local_linear_id();
        यदि (wi_in_sg == 0) {
            s_sum[sg_id] = mean_var;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        mean_var = 0.f;
        ध्रुव माप_प्रकार nreduce = ceil_div(nwarps, WARP_SIZE);
        प्रति (माप_प्रकार i = 0; i < nreduce; i += 1)
        {
            mean_var += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        mean_var = warp_reduce_sum(mean_var, item_ct1);
    }

    ध्रुव भिन्न mean = mean_var.x() / ncols;
    ध्रुव भिन्न var = mean_var.y() / ncols - mean * mean;
    ध्रुव भिन्न inv_std = sycl::rsqrt(var + eps);

    प्रति (पूर्णाङ्क col = tid; col < ncols; col += block_size) {
        dst[col * dst_stride_col] = (x[col * src_stride_col] - mean) * inv_std;
    }
}

स्थिर शून्य group_norm_f32(ध्रुव भिन्न* x, भिन्न* dst, ध्रुव पूर्णाङ्क group_size, ध्रुव पूर्णाङ्क ne_elements, ध्रुव भिन्न eps,
    ध्रुव sycl::nd_item<3>& item_ct1, भिन्न* s_sum, पूर्णाङ्क block_size) {
    पूर्णाङ्क start = item_ct1.get_group(2) * group_size;
    पूर्णाङ्क end = start + group_size;
    ध्रुव पूर्णाङ्क nthreads = item_ct1.get_local_range(2);
    ध्रुव पूर्णाङ्क nwarps = nthreads / WARP_SIZE;
    start += item_ct1.get_local_id(2);
    माप_प्रकार nreduce = nwarps / WARP_SIZE;

    यदि (end >= ne_elements) {
        end = ne_elements;
    }

    भिन्न tmp = 0.0f; // partial sum for thread in warp

    प्रति (पूर्णाङ्क j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    यदि (block_size > WARP_SIZE) {

        पूर्णाङ्क warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        पूर्णाङ्क lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        यदि (lane_id == 0) {
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
        प्रति (माप_प्रकार i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    भिन्न mean = tmp / group_size;
    tmp = 0.0f;

    प्रति (पूर्णाङ्क j = start; j < end; j += block_size) {
        भिन्न xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    यदि (block_size > WARP_SIZE) {

        पूर्णाङ्क warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        पूर्णाङ्क lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        यदि (lane_id == 0) {
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
        प्रति (माप_प्रकार i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    भिन्न variance = tmp / group_size;
    भिन्न scale = sycl::rsqrt(variance + eps);
    प्रति (पूर्णाङ्क j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

ढाँचा <तार्किक do_multiply = असत्य, तार्किक do_add = असत्य>
स्थिर शून्य rms_norm_f32(ध्रुव भिन्न* x, भिन्न* dst, ध्रुव पूर्णाङ्क ncols,
    ध्रुव int64_t src_stride_col, ध्रुव int64_t src_stride_row, ध्रुव int64_t src_stride_channel, ध्रुव int64_t src_stride_sample,
    ध्रुव int64_t dst_stride_col, ध्रुव int64_t dst_stride_row, ध्रुव int64_t dst_stride_channel, ध्रुव int64_t dst_stride_sample,
    ध्रुव भिन्न eps, ध्रुव sycl::nd_item<3>& item_ct1, भिन्न* s_sum, पूर्णाङ्क block_size,
    ध्रुव भिन्न* mul = nullptr, ध्रुव int64_t mul_stride_row = 0, ध्रुव int64_t mul_stride_channel = 0,
    ध्रुव int64_t mul_stride_sample = 0, ध्रुव पूर्णाङ्क mul_nrows = 0, ध्रुव पूर्णाङ्क mul_nchannels = 0, ध्रुव पूर्णाङ्क mul_nsamples = 0,
    ध्रुव भिन्न* add = nullptr, ध्रुव int64_t add_stride_row = 0, ध्रुव int64_t add_stride_channel = 0,
    ध्रुव int64_t add_stride_sample = 0, ध्रुव पूर्णाङ्क add_nrows = 0, ध्रुव पूर्णाङ्क add_nchannels = 0, ध्रुव पूर्णाङ्क add_nsamples = 0,
    ध्रुव भिन्न scale_mul = 1.0f) {

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    ध्रुव पूर्णाङ्क sample  = item_ct1.get_group(0);
    ध्रुव पूर्णाङ्क channel = item_ct1.get_group(1);
    ध्रुव पूर्णाङ्क row     = item_ct1.get_group(2);

    ध्रुव पूर्णाङ्क nthreads = item_ct1.get_local_range(2);

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);
    ध्रुव पूर्णाङ्क nwarps = nthreads / WARP_SIZE;

    ध्रुव स्वतः src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    ध्रुव स्वतः dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x   += src_offset;
    dst += dst_offset;

    यदि constexpr (do_multiply) {
        ध्रुव पूर्णाङ्क mul_row     = row     % mul_nrows;
        ध्रुव पूर्णाङ्क mul_channel = channel % mul_nchannels;
        ध्रुव पूर्णाङ्क mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    यदि constexpr (do_add) {
        ध्रुव पूर्णाङ्क add_row     = row     % add_nrows;
        ध्रुव पूर्णाङ्क add_channel = channel % add_nchannels;
        ध्रुव पूर्णाङ्क add_sample  = sample  % add_nsamples;
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    भिन्न tmp = 0.0f; // partial sum for thread in warp

    प्रति (पूर्णाङ्क col = tid; col < ncols; col += block_size) {
        ध्रुव भिन्न xi = x[col * src_stride_col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    यदि (block_size > WARP_SIZE) {
        ध्रुव स्वतः sub_group = item_ct1.get_sub_group();
        ध्रुव स्वतः sg_id = sub_group.get_group_linear_id();
        ध्रुव स्वतः wi_in_sg = sub_group.get_local_linear_id();
        यदि (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        ध्रुव माप_प्रकार nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        प्रति (माप_प्रकार i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    ध्रुव भिन्न mean = tmp / ncols;
    ध्रुव भिन्न scale = sycl::rsqrt(mean + eps);

    प्रति (पूर्णाङ्क col = tid; col < ncols; col += block_size) {
        यदि constexpr (do_multiply && do_add) {
            dst[col * dst_stride_col] = scale * x[col * src_stride_col] * mul[col] + add[col];
        } अन्यथा यदि constexpr (do_multiply) {
            dst[col * dst_stride_col] = scale * x[col * src_stride_col] * mul[col];
        } अन्यथा {
            // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
            // the explicit temporary keeps the float evaluation order identical to
            // running rms_norm and scale as two separate kernels
            ध्रुव भिन्न v = scale * x[col * src_stride_col];
            dst[col * dst_stride_col] = v * scale_mul;
        }
    }
}

ढाँचा<पूर्णाङ्क warp_size>
स्थिर शून्य l2_norm_f32(ध्रुव भिन्न * x, भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
    ध्रुव int64_t src_stride_col, ध्रुव int64_t src_stride_row, ध्रुव int64_t src_stride_channel,
    ध्रुव int64_t src_stride_sample, ध्रुव int64_t dst_stride_col, ध्रुव int64_t dst_stride_row,
    ध्रुव int64_t dst_stride_channel, ध्रुव int64_t dst_stride_sample, ध्रुव भिन्न eps,
    ध्रुव sycl::nd_item<3>& item_ct1, भिन्न* s_sum, ध्रुव पूर्णाङ्क block_size) {

    ध्रुव पूर्णाङ्क row     = item_ct1.get_group(2);
    ध्रुव पूर्णाङ्क channel = item_ct1.get_group(1);
    ध्रुव पूर्णाङ्क sample  = item_ct1.get_group(0);
    ध्रुव पूर्णाङ्क tid     = item_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + row*src_stride_row;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + row*dst_stride_row;

    भिन्न tmp = 0.0f; // partial sum for thread in warp

    प्रति (पूर्णाङ्क col = tid; col < ncols; col += block_size) {
        ध्रुव भिन्न xi = x[col * src_stride_col];
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, s_sum, block_size);
    ध्रुव भिन्न scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    प्रति (पूर्णाङ्क col = tid; col < ncols; col += block_size) {
        dst[col * dst_stride_col] = scale * x[col * src_stride_col];
    }
}

स्थिर शून्य norm_f32_sycl(ध्रुव भिन्न * x, भिन्न * dst, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क nchannels, ध्रुव पूर्णाङ्क nsamples,
    ध्रुव int64_t src_stride_col, ध्रुव int64_t src_stride_row, ध्रुव int64_t src_stride_channel, ध्रुव int64_t src_stride_sample,
    ध्रुव int64_t dst_stride_col, ध्रुव int64_t dst_stride_row, ध्रुव int64_t dst_stride_channel, ध्रुव int64_t dst_stride_sample,
        ध्रुव भिन्न eps, queue_ptr stream, पूर्णाङ्क device) {

    ध्रुव sycl::range<3> global_dims(nsamples, nchannels, nrows);
    यदि (ncols < 1024) {
        ध्रुव sycl::range<3> block_dims(1, 1, WARP_SIZE);
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
    अन्यथा {
        ध्रुव पूर्णाङ्क work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        प्रतिज्ञा(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ध्रुव sycl::range<3> block_dims(1, 1, work_group_size);
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

स्थिर शून्य group_norm_f32_sycl(ध्रुव भिन्न* x, भिन्न* dst,
    ध्रुव पूर्णाङ्क अंक_groups, ध्रुव भिन्न eps, ध्रुव पूर्णाङ्क group_size,
    ध्रुव पूर्णाङ्क ne_elements, queue_ptr stream, पूर्णाङ्क device) {
    यदि (group_size < 1024) {
        ध्रुव sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            ध्रुव भिन्न eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(
                        x, dst, group_size, ne_elements, eps_ct4, item_ct1,
                        nullptr, WARP_SIZE);
                });
            });
    }
    अन्यथा {
        ध्रुव पूर्णाङ्क work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        प्रतिज्ञा(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ध्रुव sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भिन्न, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);

            ध्रुव भिन्न eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_groups) * block_dims,
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

स्थिर शून्य rms_norm_f32_sycl(ध्रुव भिन्न* x, भिन्न* dst, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क nchannels, ध्रुव पूर्णाङ्क nsamples,
    ध्रुव int64_t src_stride_col, ध्रुव int64_t src_stride_row, ध्रुव int64_t src_stride_channel, ध्रुव int64_t src_stride_sample,
    ध्रुव int64_t dst_stride_col, ध्रुव int64_t dst_stride_row, ध्रुव int64_t dst_stride_channel, ध्रुव int64_t dst_stride_sample,
    ध्रुव भिन्न eps, queue_ptr stream, पूर्णाङ्क device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    ध्रुव sycl::range<3> global_dims(nsamples, nchannels, nrows);
    यदि (ncols < 1024) {
        ध्रुव sycl::range<3> block_dims(1, 1, WARP_SIZE);
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
    अन्यथा {
        ध्रुव पूर्णाङ्क work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        प्रतिज्ञा(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ध्रुव sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भिन्न, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
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

स्थिर शून्य rms_norm_scale_f32_sycl(ध्रुव भिन्न* x, भिन्न* dst, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
    ध्रुव पूर्णाङ्क nchannels, ध्रुव पूर्णाङ्क nsamples,
    ध्रुव int64_t src_stride_col, ध्रुव int64_t src_stride_row, ध्रुव int64_t src_stride_channel, ध्रुव int64_t src_stride_sample,
    ध्रुव int64_t dst_stride_col, ध्रुव int64_t dst_stride_row, ध्रुव int64_t dst_stride_channel, ध्रुव int64_t dst_stride_sample,
    ध्रुव भिन्न eps, ध्रुव भिन्न scale_mul, queue_ptr stream, पूर्णाङ्क device) {
    ध्रुव sycl::range<3> global_dims(nsamples, nchannels, nrows);
    यदि (ncols < 1024) {
        ध्रुव sycl::range<3> block_dims(1, 1, WARP_SIZE);
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
    अन्यथा {
        ध्रुव पूर्णाङ्क work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        प्रतिज्ञा(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ध्रुव sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भिन्न, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
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

स्थिर शून्य rms_norm_mul_f32_sycl(ध्रुव भिन्न* x, ध्रुव भिन्न* mul, भिन्न* dst, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क nchannels, ध्रुव पूर्णाङ्क nsamples,
        ध्रुव int64_t src_stride_col, ध्रुव int64_t src_stride_row, ध्रुव int64_t src_stride_channel, ध्रुव int64_t src_stride_sample,
        ध्रुव int64_t dst_stride_col, ध्रुव int64_t dst_stride_row, ध्रुव int64_t dst_stride_channel, ध्रुव int64_t dst_stride_sample,
        ध्रुव int64_t mul_stride_row, ध्रुव int64_t mul_stride_channel, ध्रुव int64_t mul_stride_sample,
        ध्रुव पूर्णाङ्क mul_nrows, ध्रुव पूर्णाङ्क mul_nchannels, ध्रुव पूर्णाङ्क mul_nsamples,
        ध्रुव भिन्न eps, queue_ptr stream, पूर्णाङ्क device) {
    ध्रुव sycl::range<3> global_dims(nsamples, nchannels, nrows);
    यदि (ncols < 1024) {
        ध्रुव sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<सत्य>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    अन्यथा {
        ध्रुव पूर्णाङ्क work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        प्रतिज्ञा(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ध्रुव sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भिन्न, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<सत्य>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

स्थिर शून्य rms_norm_mul_add_f32_sycl(ध्रुव भिन्न* x, ध्रुव भिन्न* mul, ध्रुव भिन्न* add, भिन्न* dst,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, ध्रुव पूर्णाङ्क nchannels, ध्रुव पूर्णाङ्क nsamples,
        ध्रुव int64_t src_stride_col, ध्रुव int64_t src_stride_row, ध्रुव int64_t src_stride_channel, ध्रुव int64_t src_stride_sample,
        ध्रुव int64_t dst_stride_col, ध्रुव int64_t dst_stride_row, ध्रुव int64_t dst_stride_channel, ध्रुव int64_t dst_stride_sample,
        ध्रुव int64_t mul_stride_row, ध्रुव int64_t mul_stride_channel, ध्रुव int64_t mul_stride_sample,
        ध्रुव पूर्णाङ्क mul_nrows, ध्रुव पूर्णाङ्क mul_nchannels, ध्रुव पूर्णाङ्क mul_nsamples,
        ध्रुव int64_t add_stride_row, ध्रुव int64_t add_stride_channel, ध्रुव int64_t add_stride_sample,
        ध्रुव पूर्णाङ्क add_nrows, ध्रुव पूर्णाङ्क add_nchannels, ध्रुव पूर्णाङ्क add_nsamples,
        ध्रुव भिन्न eps, queue_ptr stream, पूर्णाङ्क device) {
    ध्रुव sycl::range<3> global_dims(nsamples, nchannels, nrows);
    यदि (ncols < 1024) {
        ध्रुव sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<सत्य, सत्य>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add, add_stride_row, add_stride_channel, add_stride_sample, add_nrows, add_nchannels, add_nsamples);
                });
            });
    }
    अन्यथा {
        ध्रुव पूर्णाङ्क work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        प्रतिज्ञा(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ध्रुव sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भिन्न, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<सत्य, सत्य>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add, add_stride_row, add_stride_channel, add_stride_sample, add_nrows, add_nchannels, add_nsamples);
                });
            });
    }
}

ढाँचा<पूर्णाङ्क warp_size>
स्थिर शून्य l2_norm_f32_sycl(ध्रुव भिन्न *   x,
                             भिन्न *         dst,
                             ध्रुव पूर्णाङ्क       ncols,
                             ध्रुव पूर्णाङ्क       nrows,
                             ध्रुव पूर्णाङ्क       nchannels,
                             ध्रुव पूर्णाङ्क       nsamples,
                             ध्रुव int64_t   src_stride_col,
                             ध्रुव int64_t   src_stride_row,
                             ध्रुव int64_t   src_stride_channel,
                             ध्रुव int64_t   src_stride_sample,
                             ध्रुव int64_t   dst_stride_col,
                             ध्रुव int64_t   dst_stride_row,
                             ध्रुव int64_t   dst_stride_channel,
                             ध्रुव int64_t   dst_stride_sample,
                             ध्रुव भिन्न     eps,
                             queue_ptr       stream,
                             पूर्णाङ्क             device) {
    ध्रुव dpct::dim3 blocks_अंक(nrows, nchannels, nsamples);

    यदि (ncols < 1024) {
        ध्रुव dpct::dim3 block_dims(warp_size, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(blocks_अंक * block_dims,
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
    अन्यथा {
        ध्रुव पूर्णाङ्क work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        प्रतिज्ञा(work_group_size % (warp_size * warp_size) == 0);
        ध्रुव sycl::range<3> block_dims(1, 1, work_group_size);
        पूर्णाङ्क lsm_size =  block_dims[2] > warp_size ? work_group_size / warp_size * परिमाण(भिन्न): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भिन्न, 1> s_sum_acc_ct1(sycl::range<1>(lsm_size),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(blocks_अंक * block_dims,
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
संरचना l2_batch_ptrs {
    ध्रुव भिन्न * src[GGML_SYCL_L2_BATCH_MAX];
    भिन्न *       dst[GGML_SYCL_L2_BATCH_MAX];
};

// One stride set shared by the whole batch: the caller only groups tensors whose nb[]
// all match, so per-tensor state stays two pointers.
संरचना l2_batch_strides {
    पूर्णाङ्क     ne1, ne2;
    int64_t ss0, ss1, ss2, ss3;
    int64_t ds0, ds1, ds2, ds3;
};

ढाँचा <पूर्णाङ्क warp_size>
स्थिर शून्य l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, ध्रुव पूर्णाङ्क ncols, ध्रुव भिन्न eps,
                              ध्रुव sycl::nd_item<3> & item_ct1) {
    ध्रुव पूर्णाङ्क t   = item_ct1.get_group(0);  // tensor index
    ध्रुव पूर्णाङ्क r   = item_ct1.get_group(2);  // flattened row over ne1*ne2*ne3
    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);

    ध्रुव पूर्णाङ्क i1 = r % st.ne1;
    ध्रुव पूर्णाङ्क i2 = (r / st.ne1) % st.ne2;
    ध्रुव पूर्णाङ्क i3 = r / (st.ne1 * st.ne2);

    ध्रुव भिन्न * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    भिन्न *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    भिन्न tmp = 0.0f;
    प्रति (पूर्णाङ्क col = tid; col < ncols; col += warp_size) {
        ध्रुव भिन्न xi = x[col * st.ss0];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, (भिन्न *) nullptr, warp_size);
    ध्रुव भिन्न scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    प्रति (पूर्णाङ्क col = tid; col < ncols; col += warp_size) {
        dst[col * st.ds0] = scale * x[col * st.ss0];
    }
}

ढाँचा <पूर्णाङ्क warp_size>
स्थिर शून्य l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, ध्रुव पूर्णाङ्क n_tensors,
                                   ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows_total, ध्रुव भिन्न eps,
                                   queue_ptr stream) {
    ध्रुव dpct::dim3 blocks_अंक(nrows_total, 1, n_tensors);
    ध्रुव dpct::dim3 block_dims(warp_size, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(blocks_अंक * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                l2_norm_f32_batch<warp_size>(p, st, ncols, eps, item_ct1);
            });
    });
}

शून्य ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(dst->src[0]->data);
    भिन्न *       dst_dd  = static_cast<भिन्न *>(dst->data);

    भिन्न eps;
    स_नकल(&eps, dst->op_params, परिमाण(भिन्न));
    GGML_ASSERT(eps >= 0.0f);
    ध्रुव माप_प्रकार ts0 = ggml_type_size(src0->type);
    ध्रुव माप_प्रकार tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ध्रुव int64_t ss0 = nb00 / ts0;
    ध्रुव int64_t ss1 = nb01 / ts0;
    ध्रुव int64_t ss2 = nb02 / ts0;
    ध्रुव int64_t ss3 = nb03 / ts0;
    ध्रुव int64_t ds0 = nb0 / tdst;
    ध्रुव int64_t ds1 = nb1 / tdst;
    ध्रुव int64_t ds2 = nb2 / tdst;
    ध्रुव int64_t ds3 = nb3 / tdst;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
}

शून्य ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    पूर्णाङ्क अंक_groups = dst->op_params[0];
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(dst->src[0]->data);
    भिन्न *       dst_dd  = static_cast<भिन्न *>(dst->data);

    भिन्न eps;
    स_नकल(&eps, dst->op_params + 1, परिमाण(भिन्न));

    पूर्णाङ्क group_size = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + अंक_groups - 1) / अंक_groups);
    group_norm_f32_sycl(src0_dd, dst_dd, अंक_groups, eps, group_size, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_stream, ctx.device);
}

शून्य ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(dst->src[0]->data);
    भिन्न *       dst_dd  = static_cast<भिन्न *>(dst->data);

    भिन्न eps;
    स_नकल(&eps, dst->op_params, परिमाण(भिन्न));

    GGML_TENSOR_UNARY_OP_LOCALS
    ध्रुव माप_प्रकार ts0 = ggml_type_size(src0->type);
    ध्रुव माप_प्रकार tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ध्रुव int64_t ss0 = nb00 / ts0;
    ध्रुव int64_t ss1 = nb01 / ts0;
    ध्रुव int64_t ss2 = nb02 / ts0;
    ध्रुव int64_t ss3 = nb03 / ts0;
    ध्रुव int64_t ds0 = nb0 / tdst;
    ध्रुव int64_t ds1 = nb1 / tdst;
    ध्रुव int64_t ds2 = nb2 / tdst;
    ध्रुव int64_t ds3 = nb3 / tdst;
    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
}

// Fused rms_norm + scale (the qwen35 GDN l2-norm pair build_gdn_l2_norm emits):
// the scale factor is a host scalar in the GGML_OP_SCALE node's op_params, so
// unlike the mul variants there is no second device tensor to wire up.
शून्य ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       ggml_tensor * scale_tensor) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(src0->data);
    भिन्न *       dst_dd  = static_cast<भिन्न *>(scale_tensor->data);

    भिन्न eps;
    स_नकल(&eps, dst->op_params, परिमाण(भिन्न));
    भिन्न scale_mul;
    स_नकल(&scale_mul, scale_tensor->op_params, परिमाण(भिन्न));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    ध्रुव माप_प्रकार ts0 = ggml_type_size(src0->type);
    ध्रुव माप_प्रकार tdst = ggml_type_size(scale_tensor->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(scale_tensor->nb[0] % tdst == 0 && scale_tensor->nb[1] % tdst == 0 &&
                scale_tensor->nb[2] % tdst == 0 && scale_tensor->nb[3] % tdst == 0);
    ध्रुव int64_t ss0 = nb00 / ts0;
    ध्रुव int64_t ss1 = nb01 / ts0;
    ध्रुव int64_t ss2 = nb02 / ts0;
    ध्रुव int64_t ss3 = nb03 / ts0;
    ध्रुव int64_t ds0 = scale_tensor->nb[0] / tdst;
    ध्रुव int64_t ds1 = scale_tensor->nb[1] / tdst;
    ध्रुव int64_t ds2 = scale_tensor->nb[2] / tdst;
    ध्रुव int64_t ds3 = scale_tensor->nb[3] / tdst;
    rms_norm_scale_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_stream, ctx.device);
}

शून्य ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    ध्रुव ggml_tensor * rms_norm_src = dst->src[0];
    भिन्न eps = 0.0f;
    स_नकल(&eps, dst->op_params, परिमाण(भिन्न));

    ध्रुव भिन्न *       src0_dd = static_cast<ध्रुव भिन्न *>(rms_norm_src->data);
    ध्रुव भिन्न *       mul_dd  = nullptr;
    ध्रुव ggml_tensor * mul_src = nullptr;
    यदि (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ध्रुव भिन्न *>(mul_tensor->src[1]->data);
        mul_src = mul_tensor->src[1];
    } अन्यथा यदि (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ध्रुव भिन्न *>(mul_tensor->src[0]->data);
        mul_src = mul_tensor->src[0];
    } अन्यथा {
        GGML_ASSERT(असत्य);
    }
    भिन्न * dst_dd = static_cast<भिन्न *>(mul_tensor->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ध्रुव int64_t ne00 = rms_norm_src->ne[0];
    ध्रुव int64_t ne01 = rms_norm_src->ne[1];
    ध्रुव int64_t ne02 = rms_norm_src->ne[2];
    ध्रुव int64_t ne03 = rms_norm_src->ne[3];

    ध्रुव माप_प्रकार ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ध्रुव int64_t s00 = rms_norm_src->nb[0] / ts0;
    ध्रुव int64_t s01 = rms_norm_src->nb[1] / ts0;
    ध्रुव int64_t s02 = rms_norm_src->nb[2] / ts0;
    ध्रुव int64_t s03 = rms_norm_src->nb[3] / ts0;

    ध्रुव माप_प्रकार tdst = ggml_type_size(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    ध्रुव int64_t d00 = mul_tensor->nb[0] / tdst;
    ध्रुव int64_t d01 = mul_tensor->nb[1] / tdst;
    ध्रुव int64_t d02 = mul_tensor->nb[2] / tdst;
    ध्रुव int64_t d03 = mul_tensor->nb[3] / tdst;

    ध्रुव माप_प्रकार ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ध्रुव int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ध्रुव int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ध्रुव int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ध्रुव पूर्णाङ्क mul_nrows     = mul_src->ne[1];
    ध्रुव पूर्णाङ्क mul_nchannels = mul_src->ne[2];
    ध्रुव पूर्णाङ्क mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_stream, ctx.device);
}

शून्य ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    ध्रुव ggml_tensor * rms_norm_src = dst->src[0];
    भिन्न eps = 0.0f;
    स_नकल(&eps, dst->op_params, परिमाण(भिन्न));

    ध्रुव भिन्न *       src0_dd = static_cast<ध्रुव भिन्न *>(rms_norm_src->data);
    ध्रुव भिन्न *       mul_dd  = nullptr;
    ध्रुव ggml_tensor * mul_src = nullptr;
    यदि (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ध्रुव भिन्न *>(mul_tensor->src[1]->data);
        mul_src = mul_tensor->src[1];
    } अन्यथा यदि (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ध्रुव भिन्न *>(mul_tensor->src[0]->data);
        mul_src = mul_tensor->src[0];
    } अन्यथा {
        GGML_ASSERT(असत्य);
    }

    ध्रुव भिन्न *       add_dd  = nullptr;
    ध्रुव ggml_tensor * add_src = nullptr;
    यदि (add_tensor->src[0] == mul_tensor) {
        add_dd  = static_cast<ध्रुव भिन्न *>(add_tensor->src[1]->data);
        add_src = add_tensor->src[1];
    } अन्यथा यदि (add_tensor->src[1] == mul_tensor) {
        add_dd  = static_cast<ध्रुव भिन्न *>(add_tensor->src[0]->data);
        add_src = add_tensor->src[0];
    } अन्यथा {
        GGML_ASSERT(असत्य);
    }

    भिन्न * dst_dd = static_cast<भिन्न *>(add_tensor->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ध्रुव int64_t ne00 = rms_norm_src->ne[0];
    ध्रुव int64_t ne01 = rms_norm_src->ne[1];
    ध्रुव int64_t ne02 = rms_norm_src->ne[2];
    ध्रुव int64_t ne03 = rms_norm_src->ne[3];

    ध्रुव माप_प्रकार ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ध्रुव int64_t s00 = rms_norm_src->nb[0] / ts0;
    ध्रुव int64_t s01 = rms_norm_src->nb[1] / ts0;
    ध्रुव int64_t s02 = rms_norm_src->nb[2] / ts0;
    ध्रुव int64_t s03 = rms_norm_src->nb[3] / ts0;

    ध्रुव माप_प्रकार tdst = ggml_type_size(add_tensor->type);
    GGML_ASSERT(add_tensor->nb[0] == tdst);
    ध्रुव int64_t d00 = add_tensor->nb[0] / tdst;
    ध्रुव int64_t d01 = add_tensor->nb[1] / tdst;
    ध्रुव int64_t d02 = add_tensor->nb[2] / tdst;
    ध्रुव int64_t d03 = add_tensor->nb[3] / tdst;

    ध्रुव माप_प्रकार ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ध्रुव int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ध्रुव int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ध्रुव int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ध्रुव पूर्णाङ्क mul_nrows     = mul_src->ne[1];
    ध्रुव पूर्णाङ्क mul_nchannels = mul_src->ne[2];
    ध्रुव पूर्णाङ्क mul_nsamples  = mul_src->ne[3];

    ध्रुव माप_प्रकार ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    ध्रुव int64_t add_s01 = add_src->nb[1] / ts_add;
    ध्रुव int64_t add_s02 = add_src->nb[2] / ts_add;
    ध्रुव int64_t add_s03 = add_src->nb[3] / ts_add;
    ध्रुव पूर्णाङ्क add_nrows     = add_src->ne[1];
    ध्रुव पूर्णाङ्क add_nchannels = add_src->ne[2];
    ध्रुव पूर्णाङ्क add_nsamples  = add_src->ne[3];

    rms_norm_mul_add_f32_sycl(src0_dd, mul_dd, add_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        add_s01, add_s02, add_s03, add_nrows, add_nchannels, add_nsamples, eps, main_stream, ctx.device);
}

शून्य ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    भिन्न eps = 1e-5f;
    std::स_नकल(&eps, dst->op_params, परिमाण(भिन्न));
    यदि (!(eps > 0.0f) || !std::है_सीमित(eps)) eps = 1e-5f;

    ध्रुव भिन्न * g_base  = static_cast<ध्रुव भिन्न *>(dst->src[0]->data); // dz
    ध्रुव भिन्न * x_base  = static_cast<ध्रुव भिन्न *>(dst->src[1]->data); // x
          भिन्न * dx_base = static_cast<      भिन्न *>(dst->data);

    ध्रुव int64_t D  = dst->ne[0];
    ध्रुव int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (शून्य) n3;
    ध्रुव int64_t N  = ggml_nrows(dst);
    यदि (D == 0 || N == 0) प्रत्यागम;

    ध्रुव ggml_tensor *G = dst->src[0];
    ध्रुव ggml_tensor *X = dst->src[1];
    ध्रुव पूर्णाङ्क ts = (पूर्णाङ्क) ggml_type_size(X->type);
    GGML_ASSERT((माप_प्रकार) X->nb[0]   == (माप_प्रकार) ts);
    GGML_ASSERT((माप_प्रकार) G->nb[0]   == (माप_प्रकार) ts);
    GGML_ASSERT((माप_प्रकार) dst->nb[0] == (माप_प्रकार) ts);

    ध्रुव int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    ध्रुव int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    ध्रुव int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::queue_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    ध्रुव पूर्णाङ्क device_max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
    स्वतः roundup = [](पूर्णाङ्क v, पूर्णाङ्क m) { प्रत्यागम ((v + m - 1) / m) * m; };
    पूर्णाङ्क wg_cap = 256;
    यदि (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    पूर्णाङ्क WG = std::max(WARP_SIZE, std::min(roundup((पूर्णाङ्क)std::min<int64_t>(D, wg_cap), WARP_SIZE), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        ध्रुव पूर्णाङ्क nwarps_loc = std::max(1, WG / WARP_SIZE);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        स्वतः l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        स्वतः l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ध्रुव पूर्णाङ्क row = item_ct1.get_group(2);
                ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);

                ध्रुव int64_t i1 = row % n1;
                ध्रुव int64_t i2 = (row / n1) % n2;
                ध्रुव int64_t i3 = row / (n1 * n2);

                ध्रुव भिन्न *__restrict x_row = x_base + i3 * xs3 + i2 * xs2 + i1 * xs1;
                ध्रुव भिन्न *__restrict g_row = g_base + i3 * gs3 + i2 * gs2 + i1 * gs1;
                भिन्न *__restrict d_row       = dx_base + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                भिन्न sum_xx = 0.f, sum_xg = 0.f;
#ifndef GGML_SYCL_RMS_BACK_FAST
                भिन्न c_xx = 0.f, c_xg = 0.f;
#endif
                प्रति (int64_t col = tid; col < D; col += WG) {
                    ध्रुव भिन्न xv = x_row[col];
                    ध्रुव भिन्न gv = g_row[col];
#ifdef GGML_SYCL_RMS_BACK_FAST
                    sum_xx += xv * xv;
                    sum_xg += xv * gv;
#अन्यथा
                    भिन्न y1 = xv * xv - c_xx;
                    भिन्न t1 = sum_xx + y1;
                    c_xx = (t1 - sum_xx) - y1;
                    sum_xx = t1;

                    भिन्न y2 = xv * gv - c_xg;
                    भिन्न t2 = sum_xg + y2;
                    c_xg = (t2 - sum_xg) - y2;
                    sum_xg = t2;
#endif
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(sum_xx,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xx
#अन्यथा
                    0.f
#endif
                );
                sycl::float2 xg = sycl::float2(sum_xg,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xg
#अन्यथा
                    0.f
#endif
                );
                xx = warp_reduce_sum(xx, item_ct1);
                xg = warp_reduce_sum(xg, item_ct1);

                // cross-warp reduction using local memory (single barrier)
                ध्रुव स्वतः sub_group = item_ct1.get_sub_group();
                ध्रुव स्वतः sg_id     = sub_group.get_group_linear_id();
                ध्रुव स्वतः wi_in_sg  = sub_group.get_local_linear_id();
                ध्रुव पूर्णाङ्क nthreads   = item_ct1.get_local_range(2);
                ध्रुव पूर्णाङ्क nwarps     = nthreads / WARP_SIZE;

                sycl::float2 xx_total = xx;
                sycl::float2 xg_total = xg;
                यदि (nwarps > 1) {
                    यदि (wi_in_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);

                    यदि (sg_id == 0) {
                        ध्रुव अचिन्हित wi_u = wi_in_sg;
                        sycl::float2 xx_first = (wi_u < static_cast<अचिन्हित>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_first = (wi_u < static_cast<अचिन्हित>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_total = warp_reduce_sum(xx_first, item_ct1);
                        xg_total = warp_reduce_sum(xg_first, item_ct1);
                    } अन्यथा {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_total = xx;
                        xg_total = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                भिन्न inv_r = 0.f;
                भिन्न coeff = 0.f;
                यदि (tid == 0) {
                    ध्रुव भिन्न sum_xx_f  = xx_total.x() + xx_total.y();
                    ध्रुव भिन्न sum_xdz_f = xg_total.x() + xg_total.y();
                    ध्रुव भिन्न mean_eps  = sum_xx_f / (भिन्न) D + eps;
                    ध्रुव भिन्न sum_eps   = sum_xx_f + eps * (भिन्न) D;
                    inv_r = sycl::rsqrt(mean_eps);
                    coeff = -sum_xdz_f / sum_eps;
                }
                inv_r = sycl::group_broadcast(item_ct1.get_group(), inv_r);
                coeff = sycl::group_broadcast(item_ct1.get_group(), coeff);

                प्रति (int64_t col = tid; col < D; col += WG) {
                    d_row[col] = (g_row[col] + coeff * x_row[col]) * inv_r;
                }
            });
    });

}

शून्य ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव भिन्न * src0_d = (ध्रुव भिन्न *) src0->data;
    भिन्न * dst_d = (भिन्न *) dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    भिन्न eps;
    स_नकल(&eps, dst->op_params, परिमाण(भिन्न));
    GGML_ASSERT(eps >= 0.0f);

    ध्रुव माप_प्रकार ts0 = ggml_type_size(src0->type);
    ध्रुव माप_प्रकार tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ध्रुव int64_t ss0 = nb00 / ts0;
    ध्रुव int64_t ss1 = nb01 / ts0;
    ध्रुव int64_t ss2 = nb02 / ts0;
    ध्रुव int64_t ss3 = nb03 / ts0;
    ध्रुव int64_t ds0 = nb0 / tdst;
    ध्रुव int64_t ds1 = nb1 / tdst;
    ध्रुव int64_t ds2 = nb2 / tdst;
    ध्रुव int64_t ds3 = nb3 / tdst;

    /*support both WARP_SIZE or WARP_32_SIZE in code
      choose by hardware for better performance
    */
    l2_norm_f32_sycl<WARP_SIZE>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
शून्य ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, पूर्णाङ्क count) {
    ध्रुव ggml_tensor * s0 = nodes[0]->src[0];
    ध्रुव पूर्णाङ्क ncols       = (पूर्णाङ्क) s0->ne[0];
    ध्रुव पूर्णाङ्क nrows_total = (पूर्णाङ्क) ggml_nrows(s0);
    भिन्न eps;
    स_नकल(&eps, nodes[0]->op_params, परिमाण(भिन्न));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    प्रति (पूर्णाङ्क t = 0; t < count; ++t) {
        p.src[t] = (ध्रुव भिन्न *) nodes[t]->src[0]->data;
        p.dst[t] = (भिन्न *) nodes[t]->data;
    }

    ध्रुव ggml_tensor * d0 = nodes[0];
    ध्रुव माप_प्रकार        ts = ggml_type_size(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (पूर्णाङ्क) s0->ne[1];
    st.ne2 = (पूर्णाङ्क) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_SIZE>(p, st, count, ncols, nrows_total, eps, ctx.stream());
}
