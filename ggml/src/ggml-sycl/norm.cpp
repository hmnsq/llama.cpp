#অন্তর্ভাব "norm.hpp"
#অন্তর্ভাব "ggml-sycl/common.hpp"
#অন্তর্ভাব "ggml-sycl/presets.hpp"

স্থির শূন্য norm_f32(ধ্রুব ভিন্ন* x, ভিন্ন* dst, ধ্রুব পূর্ণাঙ্ক ncols,
    ধ্রুব int64_t src_stride_col, ধ্রুব int64_t src_stride_row, ধ্রুব int64_t src_stride_channel, ধ্রুব int64_t src_stride_sample,
    ধ্রুব int64_t dst_stride_col, ধ্রুব int64_t dst_stride_row, ধ্রুব int64_t dst_stride_channel, ধ্রুব int64_t dst_stride_sample,
    ধ্রুব ভিন্ন eps, ধ্রুব sycl::nd_item<3>& item_ct1, sycl::float2* s_sum, পূর্ণাঙ্ক block_size) {

    ধ্রুব পূর্ণাঙ্ক nthreads = item_ct1.get_local_range(2);
    ধ্রুব পূর্ণাঙ্ক sample  = item_ct1.get_group(0);
    ধ্রুব পূর্ণাঙ্ক channel = item_ct1.get_group(1);
    ধ্রুব পূর্ণাঙ্ক row     = item_ct1.get_group(2);

    ধ্রুব পূর্ণাঙ্ক tid = item_ct1.get_local_id(2);
    ধ্রুব পূর্ণাঙ্ক nwarps = nthreads / WARP_SIZE;

    ধ্রুব auto src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    ধ্রুব auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x += src_offset;
    dst += dst_offset;

    sycl::float2 mean_var = sycl::float2(0.f, 0.f);

    প্রতি (পূর্ণাঙ্ক col = tid; col < ncols; col += block_size) {
        ধ্রুব ভিন্ন xi = x[col * src_stride_col];
        mean_var.x() += xi;
        mean_var.y() += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var, item_ct1);
    যদি  (block_size > WARP_SIZE) {
        ধ্রুব auto sub_group = item_ct1.get_sub_group();
        ধ্রুব auto sg_id = sub_group.get_group_linear_id();
        ধ্রুব auto wi_in_sg = sub_group.get_local_linear_id();
        যদি (wi_in_sg == 0) {
            s_sum[sg_id] = mean_var;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        mean_var = 0.f;
        ধ্রুব size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        প্রতি (size_t i = 0; i < nreduce; i += 1)
        {
            mean_var += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        mean_var = warp_reduce_sum(mean_var, item_ct1);
    }

    ধ্রুব ভিন্ন mean = mean_var.x() / ncols;
    ধ্রুব ভিন্ন var = mean_var.y() / ncols - mean * mean;
    ধ্রুব ভিন্ন inv_std = sycl::rsqrt(var + eps);

    প্রতি (পূর্ণাঙ্ক col = tid; col < ncols; col += block_size) {
        dst[col * dst_stride_col] = (x[col * src_stride_col] - mean) * inv_std;
    }
}

স্থির শূন্য group_norm_f32(ধ্রুব ভিন্ন* x, ভিন্ন* dst, ধ্রুব পূর্ণাঙ্ক group_size, ধ্রুব পূর্ণাঙ্ক ne_elements, ধ্রুব ভিন্ন eps,
    ধ্রুব sycl::nd_item<3>& item_ct1, ভিন্ন* s_sum, পূর্ণাঙ্ক block_size) {
    পূর্ণাঙ্ক start = item_ct1.get_group(2) * group_size;
    পূর্ণাঙ্ক end = start + group_size;
    ধ্রুব পূর্ণাঙ্ক nthreads = item_ct1.get_local_range(2);
    ধ্রুব পূর্ণাঙ্ক nwarps = nthreads / WARP_SIZE;
    start += item_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_SIZE;

    যদি (end >= ne_elements) {
        end = ne_elements;
    }

    ভিন্ন tmp = 0.0f; // partial sum for thread in warp

    প্রতি (পূর্ণাঙ্ক j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    যদি (block_size > WARP_SIZE) {

        পূর্ণাঙ্ক warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        পূর্ণাঙ্ক lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        যদি (lane_id == 0) {
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
        প্রতি (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    ভিন্ন mean = tmp / group_size;
    tmp = 0.0f;

    প্রতি (পূর্ণাঙ্ক j = start; j < end; j += block_size) {
        ভিন্ন xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    যদি (block_size > WARP_SIZE) {

        পূর্ণাঙ্ক warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        পূর্ণাঙ্ক lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        যদি (lane_id == 0) {
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
        প্রতি (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    ভিন্ন variance = tmp / group_size;
    ভিন্ন scale = sycl::rsqrt(variance + eps);
    প্রতি (পূর্ণাঙ্ক j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <তার্কিক do_multiply = অসত্য, তার্কিক do_add = অসত্য>
স্থির শূন্য rms_norm_f32(ধ্রুব ভিন্ন* x, ভিন্ন* dst, ধ্রুব পূর্ণাঙ্ক ncols,
    ধ্রুব int64_t src_stride_col, ধ্রুব int64_t src_stride_row, ধ্রুব int64_t src_stride_channel, ধ্রুব int64_t src_stride_sample,
    ধ্রুব int64_t dst_stride_col, ধ্রুব int64_t dst_stride_row, ধ্রুব int64_t dst_stride_channel, ধ্রুব int64_t dst_stride_sample,
    ধ্রুব ভিন্ন eps, ধ্রুব sycl::nd_item<3>& item_ct1, ভিন্ন* s_sum, পূর্ণাঙ্ক block_size,
    ধ্রুব ভিন্ন* mul = nullptr, ধ্রুব int64_t mul_stride_row = 0, ধ্রুব int64_t mul_stride_channel = 0,
    ধ্রুব int64_t mul_stride_sample = 0, ধ্রুব পূর্ণাঙ্ক mul_nrows = 0, ধ্রুব পূর্ণাঙ্ক mul_nchannels = 0, ধ্রুব পূর্ণাঙ্ক mul_nsamples = 0,
    ধ্রুব ভিন্ন* add = nullptr, ধ্রুব int64_t add_stride_row = 0, ধ্রুব int64_t add_stride_channel = 0,
    ধ্রুব int64_t add_stride_sample = 0, ধ্রুব পূর্ণাঙ্ক add_nrows = 0, ধ্রুব পূর্ণাঙ্ক add_nchannels = 0, ধ্রুব পূর্ণাঙ্ক add_nsamples = 0,
    ধ্রুব ভিন্ন scale_mul = 1.0f) {

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    ধ্রুব পূর্ণাঙ্ক sample  = item_ct1.get_group(0);
    ধ্রুব পূর্ণাঙ্ক channel = item_ct1.get_group(1);
    ধ্রুব পূর্ণাঙ্ক row     = item_ct1.get_group(2);

    ধ্রুব পূর্ণাঙ্ক nthreads = item_ct1.get_local_range(2);

    ধ্রুব পূর্ণাঙ্ক tid = item_ct1.get_local_id(2);
    ধ্রুব পূর্ণাঙ্ক nwarps = nthreads / WARP_SIZE;

    ধ্রুব auto src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    ধ্রুব auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x   += src_offset;
    dst += dst_offset;

    যদি constexpr (do_multiply) {
        ধ্রুব পূর্ণাঙ্ক mul_row     = row     % mul_nrows;
        ধ্রুব পূর্ণাঙ্ক mul_channel = channel % mul_nchannels;
        ধ্রুব পূর্ণাঙ্ক mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    যদি constexpr (do_add) {
        ধ্রুব পূর্ণাঙ্ক add_row     = row     % add_nrows;
        ধ্রুব পূর্ণাঙ্ক add_channel = channel % add_nchannels;
        ধ্রুব পূর্ণাঙ্ক add_sample  = sample  % add_nsamples;
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    ভিন্ন tmp = 0.0f; // partial sum for thread in warp

    প্রতি (পূর্ণাঙ্ক col = tid; col < ncols; col += block_size) {
        ধ্রুব ভিন্ন xi = x[col * src_stride_col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    যদি (block_size > WARP_SIZE) {
        ধ্রুব auto sub_group = item_ct1.get_sub_group();
        ধ্রুব auto sg_id = sub_group.get_group_linear_id();
        ধ্রুব auto wi_in_sg = sub_group.get_local_linear_id();
        যদি (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        ধ্রুব size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        প্রতি (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    ধ্রুব ভিন্ন mean = tmp / ncols;
    ধ্রুব ভিন্ন scale = sycl::rsqrt(mean + eps);

    প্রতি (পূর্ণাঙ্ক col = tid; col < ncols; col += block_size) {
        যদি constexpr (do_multiply && do_add) {
            dst[col * dst_stride_col] = scale * x[col * src_stride_col] * mul[col] + add[col];
        } অন্যথা যদি constexpr (do_multiply) {
            dst[col * dst_stride_col] = scale * x[col * src_stride_col] * mul[col];
        } অন্যথা {
            // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
            // the explicit temporary keeps the float evaluation order identical to
            // running rms_norm and scale as two separate kernels
            ধ্রুব ভিন্ন v = scale * x[col * src_stride_col];
            dst[col * dst_stride_col] = v * scale_mul;
        }
    }
}

template<পূর্ণাঙ্ক warp_size>
স্থির শূন্য l2_norm_f32(ধ্রুব ভিন্ন * x, ভিন্ন * dst, ধ্রুব পূর্ণাঙ্ক ncols,
    ধ্রুব int64_t src_stride_col, ধ্রুব int64_t src_stride_row, ধ্রুব int64_t src_stride_channel,
    ধ্রুব int64_t src_stride_sample, ধ্রুব int64_t dst_stride_col, ধ্রুব int64_t dst_stride_row,
    ধ্রুব int64_t dst_stride_channel, ধ্রুব int64_t dst_stride_sample, ধ্রুব ভিন্ন eps,
    ধ্রুব sycl::nd_item<3>& item_ct1, ভিন্ন* s_sum, ধ্রুব পূর্ণাঙ্ক block_size) {

    ধ্রুব পূর্ণাঙ্ক row     = item_ct1.get_group(2);
    ধ্রুব পূর্ণাঙ্ক channel = item_ct1.get_group(1);
    ধ্রুব পূর্ণাঙ্ক sample  = item_ct1.get_group(0);
    ধ্রুব পূর্ণাঙ্ক tid     = item_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + row*src_stride_row;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + row*dst_stride_row;

    ভিন্ন tmp = 0.0f; // partial sum for thread in warp

    প্রতি (পূর্ণাঙ্ক col = tid; col < ncols; col += block_size) {
        ধ্রুব ভিন্ন xi = x[col * src_stride_col];
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, s_sum, block_size);
    ধ্রুব ভিন্ন scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    প্রতি (পূর্ণাঙ্ক col = tid; col < ncols; col += block_size) {
        dst[col * dst_stride_col] = scale * x[col * src_stride_col];
    }
}

স্থির শূন্য norm_f32_sycl(ধ্রুব ভিন্ন * x, ভিন্ন * dst, ধ্রুব পূর্ণাঙ্ক ncols, ধ্রুব পূর্ণাঙ্ক nrows, ধ্রুব পূর্ণাঙ্ক nchannels, ধ্রুব পূর্ণাঙ্ক nsamples,
    ধ্রুব int64_t src_stride_col, ধ্রুব int64_t src_stride_row, ধ্রুব int64_t src_stride_channel, ধ্রুব int64_t src_stride_sample,
    ধ্রুব int64_t dst_stride_col, ধ্রুব int64_t dst_stride_row, ধ্রুব int64_t dst_stride_channel, ধ্রুব int64_t dst_stride_sample,
        ধ্রুব ভিন্ন eps, queue_ptr stream, পূর্ণাঙ্ক device) {

    ধ্রুব sycl::range<3> global_dims(nsamples, nchannels, nrows);
    যদি (ncols < 1024) {
        ধ্রুব sycl::range<3> block_dims(1, 1, WARP_SIZE);
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
    অন্যথা {
        ধ্রুব পূর্ণাঙ্ক work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ধ্রুব sycl::range<3> block_dims(1, 1, work_group_size);
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

স্থির শূন্য group_norm_f32_sycl(ধ্রুব ভিন্ন* x, ভিন্ন* dst,
    ধ্রুব পূর্ণাঙ্ক অঙ্ক_groups, ধ্রুব ভিন্ন eps, ধ্রুব পূর্ণাঙ্ক group_size,
    ধ্রুব পূর্ণাঙ্ক ne_elements, queue_ptr stream, পূর্ণাঙ্ক device) {
    যদি (group_size < 1024) {
        ধ্রুব sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            ধ্রুব ভিন্ন eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(
                        x, dst, group_size, ne_elements, eps_ct4, item_ct1,
                        nullptr, WARP_SIZE);
                });
            });
    }
    অন্যথা {
        ধ্রুব পূর্ণাঙ্ক work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ধ্রুব sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ভিন্ন, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);

            ধ্রুব ভিন্ন eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_groups) * block_dims,
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

স্থির শূন্য rms_norm_f32_sycl(ধ্রুব ভিন্ন* x, ভিন্ন* dst, ধ্রুব পূর্ণাঙ্ক ncols, ধ্রুব পূর্ণাঙ্ক nrows, ধ্রুব পূর্ণাঙ্ক nchannels, ধ্রুব পূর্ণাঙ্ক nsamples,
    ধ্রুব int64_t src_stride_col, ধ্রুব int64_t src_stride_row, ধ্রুব int64_t src_stride_channel, ধ্রুব int64_t src_stride_sample,
    ধ্রুব int64_t dst_stride_col, ধ্রুব int64_t dst_stride_row, ধ্রুব int64_t dst_stride_channel, ধ্রুব int64_t dst_stride_sample,
    ধ্রুব ভিন্ন eps, queue_ptr stream, পূর্ণাঙ্ক device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    ধ্রুব sycl::range<3> global_dims(nsamples, nchannels, nrows);
    যদি (ncols < 1024) {
        ধ্রুব sycl::range<3> block_dims(1, 1, WARP_SIZE);
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
    অন্যথা {
        ধ্রুব পূর্ণাঙ্ক work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ধ্রুব sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ভিন্ন, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
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

স্থির শূন্য rms_norm_scale_f32_sycl(ধ্রুব ভিন্ন* x, ভিন্ন* dst, ধ্রুব পূর্ণাঙ্ক ncols, ধ্রুব পূর্ণাঙ্ক nrows,
    ধ্রুব পূর্ণাঙ্ক nchannels, ধ্রুব পূর্ণাঙ্ক nsamples,
    ধ্রুব int64_t src_stride_col, ধ্রুব int64_t src_stride_row, ধ্রুব int64_t src_stride_channel, ধ্রুব int64_t src_stride_sample,
    ধ্রুব int64_t dst_stride_col, ধ্রুব int64_t dst_stride_row, ধ্রুব int64_t dst_stride_channel, ধ্রুব int64_t dst_stride_sample,
    ধ্রুব ভিন্ন eps, ধ্রুব ভিন্ন scale_mul, queue_ptr stream, পূর্ণাঙ্ক device) {
    ধ্রুব sycl::range<3> global_dims(nsamples, nchannels, nrows);
    যদি (ncols < 1024) {
        ধ্রুব sycl::range<3> block_dims(1, 1, WARP_SIZE);
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
    অন্যথা {
        ধ্রুব পূর্ণাঙ্ক work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ধ্রুব sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ভিন্ন, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
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

স্থির শূন্য rms_norm_mul_f32_sycl(ধ্রুব ভিন্ন* x, ধ্রুব ভিন্ন* mul, ভিন্ন* dst, ধ্রুব পূর্ণাঙ্ক ncols, ধ্রুব পূর্ণাঙ্ক nrows,
        ধ্রুব পূর্ণাঙ্ক nchannels, ধ্রুব পূর্ণাঙ্ক nsamples,
        ধ্রুব int64_t src_stride_col, ধ্রুব int64_t src_stride_row, ধ্রুব int64_t src_stride_channel, ধ্রুব int64_t src_stride_sample,
        ধ্রুব int64_t dst_stride_col, ধ্রুব int64_t dst_stride_row, ধ্রুব int64_t dst_stride_channel, ধ্রুব int64_t dst_stride_sample,
        ধ্রুব int64_t mul_stride_row, ধ্রুব int64_t mul_stride_channel, ধ্রুব int64_t mul_stride_sample,
        ধ্রুব পূর্ণাঙ্ক mul_nrows, ধ্রুব পূর্ণাঙ্ক mul_nchannels, ধ্রুব পূর্ণাঙ্ক mul_nsamples,
        ধ্রুব ভিন্ন eps, queue_ptr stream, পূর্ণাঙ্ক device) {
    ধ্রুব sycl::range<3> global_dims(nsamples, nchannels, nrows);
    যদি (ncols < 1024) {
        ধ্রুব sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<সত্য>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    অন্যথা {
        ধ্রুব পূর্ণাঙ্ক work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ধ্রুব sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ভিন্ন, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<সত্য>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

স্থির শূন্য rms_norm_mul_add_f32_sycl(ধ্রুব ভিন্ন* x, ধ্রুব ভিন্ন* mul, ধ্রুব ভিন্ন* add, ভিন্ন* dst,
        ধ্রুব পূর্ণাঙ্ক ncols, ধ্রুব পূর্ণাঙ্ক nrows, ধ্রুব পূর্ণাঙ্ক nchannels, ধ্রুব পূর্ণাঙ্ক nsamples,
        ধ্রুব int64_t src_stride_col, ধ্রুব int64_t src_stride_row, ধ্রুব int64_t src_stride_channel, ধ্রুব int64_t src_stride_sample,
        ধ্রুব int64_t dst_stride_col, ধ্রুব int64_t dst_stride_row, ধ্রুব int64_t dst_stride_channel, ধ্রুব int64_t dst_stride_sample,
        ধ্রুব int64_t mul_stride_row, ধ্রুব int64_t mul_stride_channel, ধ্রুব int64_t mul_stride_sample,
        ধ্রুব পূর্ণাঙ্ক mul_nrows, ধ্রুব পূর্ণাঙ্ক mul_nchannels, ধ্রুব পূর্ণাঙ্ক mul_nsamples,
        ধ্রুব int64_t add_stride_row, ধ্রুব int64_t add_stride_channel, ধ্রুব int64_t add_stride_sample,
        ধ্রুব পূর্ণাঙ্ক add_nrows, ধ্রুব পূর্ণাঙ্ক add_nchannels, ধ্রুব পূর্ণাঙ্ক add_nsamples,
        ধ্রুব ভিন্ন eps, queue_ptr stream, পূর্ণাঙ্ক device) {
    ধ্রুব sycl::range<3> global_dims(nsamples, nchannels, nrows);
    যদি (ncols < 1024) {
        ধ্রুব sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<সত্য, সত্য>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add, add_stride_row, add_stride_channel, add_stride_sample, add_nrows, add_nchannels, add_nsamples);
                });
            });
    }
    অন্যথা {
        ধ্রুব পূর্ণাঙ্ক work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        ধ্রুব sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ভিন্ন, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<সত্য, সত্য>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add, add_stride_row, add_stride_channel, add_stride_sample, add_nrows, add_nchannels, add_nsamples);
                });
            });
    }
}

template<পূর্ণাঙ্ক warp_size>
স্থির শূন্য l2_norm_f32_sycl(ধ্রুব ভিন্ন *   x,
                             ভিন্ন *         dst,
                             ধ্রুব পূর্ণাঙ্ক       ncols,
                             ধ্রুব পূর্ণাঙ্ক       nrows,
                             ধ্রুব পূর্ণাঙ্ক       nchannels,
                             ধ্রুব পূর্ণাঙ্ক       nsamples,
                             ধ্রুব int64_t   src_stride_col,
                             ধ্রুব int64_t   src_stride_row,
                             ধ্রুব int64_t   src_stride_channel,
                             ধ্রুব int64_t   src_stride_sample,
                             ধ্রুব int64_t   dst_stride_col,
                             ধ্রুব int64_t   dst_stride_row,
                             ধ্রুব int64_t   dst_stride_channel,
                             ধ্রুব int64_t   dst_stride_sample,
                             ধ্রুব ভিন্ন     eps,
                             queue_ptr       stream,
                             পূর্ণাঙ্ক             device) {
    ধ্রুব dpct::dim3 blocks_অঙ্ক(nrows, nchannels, nsamples);

    যদি (ncols < 1024) {
        ধ্রুব dpct::dim3 block_dims(warp_size, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(blocks_অঙ্ক * block_dims,
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
    অন্যথা {
        ধ্রুব পূর্ণাঙ্ক work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_size % (warp_size * warp_size) == 0);
        ধ্রুব sycl::range<3> block_dims(1, 1, work_group_size);
        পূর্ণাঙ্ক lsm_size =  block_dims[2] > warp_size ? work_group_size / warp_size * পরিমাণ(ভিন্ন): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ভিন্ন, 1> s_sum_acc_ct1(sycl::range<1>(lsm_size),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(blocks_অঙ্ক * block_dims,
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
সংরচনা l2_batch_ptrs {
    ধ্রুব ভিন্ন * src[GGML_SYCL_L2_BATCH_MAX];
    ভিন্ন *       dst[GGML_SYCL_L2_BATCH_MAX];
};

// One stride set shared by the whole batch: the caller only groups tensors whose nb[]
// all match, so per-tensor state stays two pointers.
সংরচনা l2_batch_strides {
    পূর্ণাঙ্ক     ne1, ne2;
    int64_t ss0, ss1, ss2, ss3;
    int64_t ds0, ds1, ds2, ds3;
};

template <পূর্ণাঙ্ক warp_size>
স্থির শূন্য l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, ধ্রুব পূর্ণাঙ্ক ncols, ধ্রুব ভিন্ন eps,
                              ধ্রুব sycl::nd_item<3> & item_ct1) {
    ধ্রুব পূর্ণাঙ্ক t   = item_ct1.get_group(0);  // tensor index
    ধ্রুব পূর্ণাঙ্ক r   = item_ct1.get_group(2);  // flattened row over ne1*ne2*ne3
    ধ্রুব পূর্ণাঙ্ক tid = item_ct1.get_local_id(2);

    ধ্রুব পূর্ণাঙ্ক i1 = r % st.ne1;
    ধ্রুব পূর্ণাঙ্ক i2 = (r / st.ne1) % st.ne2;
    ধ্রুব পূর্ণাঙ্ক i3 = r / (st.ne1 * st.ne2);

    ধ্রুব ভিন্ন * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    ভিন্ন *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    ভিন্ন tmp = 0.0f;
    প্রতি (পূর্ণাঙ্ক col = tid; col < ncols; col += warp_size) {
        ধ্রুব ভিন্ন xi = x[col * st.ss0];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, (ভিন্ন *) nullptr, warp_size);
    ধ্রুব ভিন্ন scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    প্রতি (পূর্ণাঙ্ক col = tid; col < ncols; col += warp_size) {
        dst[col * st.ds0] = scale * x[col * st.ss0];
    }
}

template <পূর্ণাঙ্ক warp_size>
স্থির শূন্য l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, ধ্রুব পূর্ণাঙ্ক n_tensors,
                                   ধ্রুব পূর্ণাঙ্ক ncols, ধ্রুব পূর্ণাঙ্ক nrows_total, ধ্রুব ভিন্ন eps,
                                   queue_ptr stream) {
    ধ্রুব dpct::dim3 blocks_অঙ্ক(nrows_total, 1, n_tensors);
    ধ্রুব dpct::dim3 block_dims(warp_size, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(blocks_অঙ্ক * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                l2_norm_f32_batch<warp_size>(p, st, ncols, eps, item_ct1);
            });
    });
}

শূন্য ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ধ্রুব ভিন্ন * src0_dd = static_cast<ধ্রুব ভিন্ন *>(dst->src[0]->data);
    ভিন্ন *       dst_dd  = static_cast<ভিন্ন *>(dst->data);

    ভিন্ন eps;
    memcpy(&eps, dst->op_params, পরিমাণ(ভিন্ন));
    GGML_ASSERT(eps >= 0.0f);
    ধ্রুব size_t ts0 = ggml_type_size(src0->type);
    ধ্রুব size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ধ্রুব int64_t ss0 = nb00 / ts0;
    ধ্রুব int64_t ss1 = nb01 / ts0;
    ধ্রুব int64_t ss2 = nb02 / ts0;
    ধ্রুব int64_t ss3 = nb03 / ts0;
    ধ্রুব int64_t ds0 = nb0 / tdst;
    ধ্রুব int64_t ds1 = nb1 / tdst;
    ধ্রুব int64_t ds2 = nb2 / tdst;
    ধ্রুব int64_t ds3 = nb3 / tdst;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
}

শূন্য ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    পূর্ণাঙ্ক অঙ্ক_groups = dst->op_params[0];
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুব ভিন্ন * src0_dd = static_cast<ধ্রুব ভিন্ন *>(dst->src[0]->data);
    ভিন্ন *       dst_dd  = static_cast<ভিন্ন *>(dst->data);

    ভিন্ন eps;
    memcpy(&eps, dst->op_params + 1, পরিমাণ(ভিন্ন));

    পূর্ণাঙ্ক group_size = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + অঙ্ক_groups - 1) / অঙ্ক_groups);
    group_norm_f32_sycl(src0_dd, dst_dd, অঙ্ক_groups, eps, group_size, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_stream, ctx.device);
}

শূন্য ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ধ্রুব ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুব ভিন্ন * src0_dd = static_cast<ধ্রুব ভিন্ন *>(dst->src[0]->data);
    ভিন্ন *       dst_dd  = static_cast<ভিন্ন *>(dst->data);

    ভিন্ন eps;
    memcpy(&eps, dst->op_params, পরিমাণ(ভিন্ন));

    GGML_TENSOR_UNARY_OP_LOCALS
    ধ্রুব size_t ts0 = ggml_type_size(src0->type);
    ধ্রুব size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ধ্রুব int64_t ss0 = nb00 / ts0;
    ধ্রুব int64_t ss1 = nb01 / ts0;
    ধ্রুব int64_t ss2 = nb02 / ts0;
    ধ্রুব int64_t ss3 = nb03 / ts0;
    ধ্রুব int64_t ds0 = nb0 / tdst;
    ধ্রুব int64_t ds1 = nb1 / tdst;
    ধ্রুব int64_t ds2 = nb2 / tdst;
    ধ্রুব int64_t ds3 = nb3 / tdst;
    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
}

// Fused rms_norm + scale (the qwen35 GDN l2-norm pair build_gdn_l2_norm emits):
// the scale factor is a host scalar in the GGML_OP_SCALE node's op_params, so
// unlike the mul variants there is no second device tensor to wire up.
শূন্য ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       ggml_tensor * scale_tensor) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুব ভিন্ন * src0_dd = static_cast<ধ্রুব ভিন্ন *>(src0->data);
    ভিন্ন *       dst_dd  = static_cast<ভিন্ন *>(scale_tensor->data);

    ভিন্ন eps;
    memcpy(&eps, dst->op_params, পরিমাণ(ভিন্ন));
    ভিন্ন scale_mul;
    memcpy(&scale_mul, scale_tensor->op_params, পরিমাণ(ভিন্ন));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    ধ্রুব size_t ts0 = ggml_type_size(src0->type);
    ধ্রুব size_t tdst = ggml_type_size(scale_tensor->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(scale_tensor->nb[0] % tdst == 0 && scale_tensor->nb[1] % tdst == 0 &&
                scale_tensor->nb[2] % tdst == 0 && scale_tensor->nb[3] % tdst == 0);
    ধ্রুব int64_t ss0 = nb00 / ts0;
    ধ্রুব int64_t ss1 = nb01 / ts0;
    ধ্রুব int64_t ss2 = nb02 / ts0;
    ধ্রুব int64_t ss3 = nb03 / ts0;
    ধ্রুব int64_t ds0 = scale_tensor->nb[0] / tdst;
    ধ্রুব int64_t ds1 = scale_tensor->nb[1] / tdst;
    ধ্রুব int64_t ds2 = scale_tensor->nb[2] / tdst;
    ধ্রুব int64_t ds3 = scale_tensor->nb[3] / tdst;
    rms_norm_scale_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_stream, ctx.device);
}

শূন্য ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    ধ্রুব ggml_tensor * rms_norm_src = dst->src[0];
    ভিন্ন eps = 0.0f;
    memcpy(&eps, dst->op_params, পরিমাণ(ভিন্ন));

    ধ্রুব ভিন্ন *       src0_dd = static_cast<ধ্রুব ভিন্ন *>(rms_norm_src->data);
    ধ্রুব ভিন্ন *       mul_dd  = nullptr;
    ধ্রুব ggml_tensor * mul_src = nullptr;
    যদি (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ধ্রুব ভিন্ন *>(mul_tensor->src[1]->data);
        mul_src = mul_tensor->src[1];
    } অন্যথা যদি (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ধ্রুব ভিন্ন *>(mul_tensor->src[0]->data);
        mul_src = mul_tensor->src[0];
    } অন্যথা {
        GGML_ASSERT(অসত্য);
    }
    ভিন্ন * dst_dd = static_cast<ভিন্ন *>(mul_tensor->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ধ্রুব int64_t ne00 = rms_norm_src->ne[0];
    ধ্রুব int64_t ne01 = rms_norm_src->ne[1];
    ধ্রুব int64_t ne02 = rms_norm_src->ne[2];
    ধ্রুব int64_t ne03 = rms_norm_src->ne[3];

    ধ্রুব size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ধ্রুব int64_t s00 = rms_norm_src->nb[0] / ts0;
    ধ্রুব int64_t s01 = rms_norm_src->nb[1] / ts0;
    ধ্রুব int64_t s02 = rms_norm_src->nb[2] / ts0;
    ধ্রুব int64_t s03 = rms_norm_src->nb[3] / ts0;

    ধ্রুব size_t tdst = ggml_type_size(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    ধ্রুব int64_t d00 = mul_tensor->nb[0] / tdst;
    ধ্রুব int64_t d01 = mul_tensor->nb[1] / tdst;
    ধ্রুব int64_t d02 = mul_tensor->nb[2] / tdst;
    ধ্রুব int64_t d03 = mul_tensor->nb[3] / tdst;

    ধ্রুব size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ধ্রুব int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ধ্রুব int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ধ্রুব int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ধ্রুব পূর্ণাঙ্ক mul_nrows     = mul_src->ne[1];
    ধ্রুব পূর্ণাঙ্ক mul_nchannels = mul_src->ne[2];
    ধ্রুব পূর্ণাঙ্ক mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_stream, ctx.device);
}

শূন্য ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    ধ্রুব ggml_tensor * rms_norm_src = dst->src[0];
    ভিন্ন eps = 0.0f;
    memcpy(&eps, dst->op_params, পরিমাণ(ভিন্ন));

    ধ্রুব ভিন্ন *       src0_dd = static_cast<ধ্রুব ভিন্ন *>(rms_norm_src->data);
    ধ্রুব ভিন্ন *       mul_dd  = nullptr;
    ধ্রুব ggml_tensor * mul_src = nullptr;
    যদি (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ধ্রুব ভিন্ন *>(mul_tensor->src[1]->data);
        mul_src = mul_tensor->src[1];
    } অন্যথা যদি (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ধ্রুব ভিন্ন *>(mul_tensor->src[0]->data);
        mul_src = mul_tensor->src[0];
    } অন্যথা {
        GGML_ASSERT(অসত্য);
    }

    ধ্রুব ভিন্ন *       add_dd  = nullptr;
    ধ্রুব ggml_tensor * add_src = nullptr;
    যদি (add_tensor->src[0] == mul_tensor) {
        add_dd  = static_cast<ধ্রুব ভিন্ন *>(add_tensor->src[1]->data);
        add_src = add_tensor->src[1];
    } অন্যথা যদি (add_tensor->src[1] == mul_tensor) {
        add_dd  = static_cast<ধ্রুব ভিন্ন *>(add_tensor->src[0]->data);
        add_src = add_tensor->src[0];
    } অন্যথা {
        GGML_ASSERT(অসত্য);
    }

    ভিন্ন * dst_dd = static_cast<ভিন্ন *>(add_tensor->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ধ্রুব int64_t ne00 = rms_norm_src->ne[0];
    ধ্রুব int64_t ne01 = rms_norm_src->ne[1];
    ধ্রুব int64_t ne02 = rms_norm_src->ne[2];
    ধ্রুব int64_t ne03 = rms_norm_src->ne[3];

    ধ্রুব size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ধ্রুব int64_t s00 = rms_norm_src->nb[0] / ts0;
    ধ্রুব int64_t s01 = rms_norm_src->nb[1] / ts0;
    ধ্রুব int64_t s02 = rms_norm_src->nb[2] / ts0;
    ধ্রুব int64_t s03 = rms_norm_src->nb[3] / ts0;

    ধ্রুব size_t tdst = ggml_type_size(add_tensor->type);
    GGML_ASSERT(add_tensor->nb[0] == tdst);
    ধ্রুব int64_t d00 = add_tensor->nb[0] / tdst;
    ধ্রুব int64_t d01 = add_tensor->nb[1] / tdst;
    ধ্রুব int64_t d02 = add_tensor->nb[2] / tdst;
    ধ্রুব int64_t d03 = add_tensor->nb[3] / tdst;

    ধ্রুব size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ধ্রুব int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ধ্রুব int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ধ্রুব int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ধ্রুব পূর্ণাঙ্ক mul_nrows     = mul_src->ne[1];
    ধ্রুব পূর্ণাঙ্ক mul_nchannels = mul_src->ne[2];
    ধ্রুব পূর্ণাঙ্ক mul_nsamples  = mul_src->ne[3];

    ধ্রুব size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    ধ্রুব int64_t add_s01 = add_src->nb[1] / ts_add;
    ধ্রুব int64_t add_s02 = add_src->nb[2] / ts_add;
    ধ্রুব int64_t add_s03 = add_src->nb[3] / ts_add;
    ধ্রুব পূর্ণাঙ্ক add_nrows     = add_src->ne[1];
    ধ্রুব পূর্ণাঙ্ক add_nchannels = add_src->ne[2];
    ধ্রুব পূর্ণাঙ্ক add_nsamples  = add_src->ne[3];

    rms_norm_mul_add_f32_sycl(src0_dd, mul_dd, add_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        add_s01, add_s02, add_s03, add_nrows, add_nchannels, add_nsamples, eps, main_stream, ctx.device);
}

শূন্য ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    ভিন্ন eps = 1e-5f;
    std::memcpy(&eps, dst->op_params, পরিমাণ(ভিন্ন));
    যদি (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    ধ্রুব ভিন্ন * g_base  = static_cast<ধ্রুব ভিন্ন *>(dst->src[0]->data); // dz
    ধ্রুব ভিন্ন * x_base  = static_cast<ধ্রুব ভিন্ন *>(dst->src[1]->data); // x
          ভিন্ন * dx_base = static_cast<      ভিন্ন *>(dst->data);

    ধ্রুব int64_t D  = dst->ne[0];
    ধ্রুব int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (শূন্য) n3;
    ধ্রুব int64_t N  = ggml_nrows(dst);
    যদি (D == 0 || N == 0) প্রত্যাগম;

    ধ্রুব ggml_tensor *G = dst->src[0];
    ধ্রুব ggml_tensor *X = dst->src[1];
    ধ্রুব পূর্ণাঙ্ক ts = (পূর্ণাঙ্ক) ggml_type_size(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    ধ্রুব int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    ধ্রুব int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    ধ্রুব int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::queue_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    ধ্রুব পূর্ণাঙ্ক device_max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
    auto roundup = [](পূর্ণাঙ্ক v, পূর্ণাঙ্ক m) { প্রত্যাগম ((v + m - 1) / m) * m; };
    পূর্ণাঙ্ক wg_cap = 256;
    যদি (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    পূর্ণাঙ্ক WG = std::max(WARP_SIZE, std::min(roundup((পূর্ণাঙ্ক)std::min<int64_t>(D, wg_cap), WARP_SIZE), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        ধ্রুব পূর্ণাঙ্ক nwarps_loc = std::max(1, WG / WARP_SIZE);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        auto l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        auto l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                ধ্রুব পূর্ণাঙ্ক row = item_ct1.get_group(2);
                ধ্রুব পূর্ণাঙ্ক tid = item_ct1.get_local_id(2);

                ধ্রুব int64_t i1 = row % n1;
                ধ্রুব int64_t i2 = (row / n1) % n2;
                ধ্রুব int64_t i3 = row / (n1 * n2);

                ধ্রুব ভিন্ন *__restrict x_row = x_base + i3 * xs3 + i2 * xs2 + i1 * xs1;
                ধ্রুব ভিন্ন *__restrict g_row = g_base + i3 * gs3 + i2 * gs2 + i1 * gs1;
                ভিন্ন *__restrict d_row       = dx_base + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                ভিন্ন sum_xx = 0.f, sum_xg = 0.f;
#ifndef GGML_SYCL_RMS_BACK_FAST
                ভিন্ন c_xx = 0.f, c_xg = 0.f;
#endif
                প্রতি (int64_t col = tid; col < D; col += WG) {
                    ধ্রুব ভিন্ন xv = x_row[col];
                    ধ্রুব ভিন্ন gv = g_row[col];
#ifdef GGML_SYCL_RMS_BACK_FAST
                    sum_xx += xv * xv;
                    sum_xg += xv * gv;
#অন্যথা
                    ভিন্ন y1 = xv * xv - c_xx;
                    ভিন্ন t1 = sum_xx + y1;
                    c_xx = (t1 - sum_xx) - y1;
                    sum_xx = t1;

                    ভিন্ন y2 = xv * gv - c_xg;
                    ভিন্ন t2 = sum_xg + y2;
                    c_xg = (t2 - sum_xg) - y2;
                    sum_xg = t2;
#endif
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(sum_xx,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xx
#অন্যথা
                    0.f
#endif
                );
                sycl::float2 xg = sycl::float2(sum_xg,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xg
#অন্যথা
                    0.f
#endif
                );
                xx = warp_reduce_sum(xx, item_ct1);
                xg = warp_reduce_sum(xg, item_ct1);

                // cross-warp reduction using local memory (single barrier)
                ধ্রুব auto sub_group = item_ct1.get_sub_group();
                ধ্রুব auto sg_id     = sub_group.get_group_linear_id();
                ধ্রুব auto wi_in_sg  = sub_group.get_local_linear_id();
                ধ্রুব পূর্ণাঙ্ক nthreads   = item_ct1.get_local_range(2);
                ধ্রুব পূর্ণাঙ্ক nwarps     = nthreads / WARP_SIZE;

                sycl::float2 xx_total = xx;
                sycl::float2 xg_total = xg;
                যদি (nwarps > 1) {
                    যদি (wi_in_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);

                    যদি (sg_id == 0) {
                        ধ্রুব unsigned wi_u = wi_in_sg;
                        sycl::float2 xx_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_total = warp_reduce_sum(xx_first, item_ct1);
                        xg_total = warp_reduce_sum(xg_first, item_ct1);
                    } অন্যথা {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_total = xx;
                        xg_total = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                ভিন্ন inv_r = 0.f;
                ভিন্ন coeff = 0.f;
                যদি (tid == 0) {
                    ধ্রুব ভিন্ন sum_xx_f  = xx_total.x() + xx_total.y();
                    ধ্রুব ভিন্ন sum_xdz_f = xg_total.x() + xg_total.y();
                    ধ্রুব ভিন্ন mean_eps  = sum_xx_f / (ভিন্ন) D + eps;
                    ধ্রুব ভিন্ন sum_eps   = sum_xx_f + eps * (ভিন্ন) D;
                    inv_r = sycl::rsqrt(mean_eps);
                    coeff = -sum_xdz_f / sum_eps;
                }
                inv_r = sycl::group_broadcast(item_ct1.get_group(), inv_r);
                coeff = sycl::group_broadcast(item_ct1.get_group(), coeff);

                প্রতি (int64_t col = tid; col < D; col += WG) {
                    d_row[col] = (g_row[col] + coeff * x_row[col]) * inv_r;
                }
            });
    });

}

শূন্য ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ভিন্ন * src0_d = (ধ্রুব ভিন্ন *) src0->data;
    ভিন্ন * dst_d = (ভিন্ন *) dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    ভিন্ন eps;
    memcpy(&eps, dst->op_params, পরিমাণ(ভিন্ন));
    GGML_ASSERT(eps >= 0.0f);

    ধ্রুব size_t ts0 = ggml_type_size(src0->type);
    ধ্রুব size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ধ্রুব int64_t ss0 = nb00 / ts0;
    ধ্রুব int64_t ss1 = nb01 / ts0;
    ধ্রুব int64_t ss2 = nb02 / ts0;
    ধ্রুব int64_t ss3 = nb03 / ts0;
    ধ্রুব int64_t ds0 = nb0 / tdst;
    ধ্রুব int64_t ds1 = nb1 / tdst;
    ধ্রুব int64_t ds2 = nb2 / tdst;
    ধ্রুব int64_t ds3 = nb3 / tdst;

    /*support both WARP_SIZE or WARP_32_SIZE in code
      choose by hardware for better performance
    */
    l2_norm_f32_sycl<WARP_SIZE>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
শূন্য ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, পূর্ণাঙ্ক count) {
    ধ্রুব ggml_tensor * s0 = nodes[0]->src[0];
    ধ্রুব পূর্ণাঙ্ক ncols       = (পূর্ণাঙ্ক) s0->ne[0];
    ধ্রুব পূর্ণাঙ্ক nrows_total = (পূর্ণাঙ্ক) ggml_nrows(s0);
    ভিন্ন eps;
    memcpy(&eps, nodes[0]->op_params, পরিমাণ(ভিন্ন));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    প্রতি (পূর্ণাঙ্ক t = 0; t < count; ++t) {
        p.src[t] = (ধ্রুব ভিন্ন *) nodes[t]->src[0]->data;
        p.dst[t] = (ভিন্ন *) nodes[t]->data;
    }

    ধ্রুব ggml_tensor * d0 = nodes[0];
    ধ্রুব size_t        ts = ggml_type_size(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (পূর্ণাঙ্ক) s0->ne[1];
    st.ne2 = (পূর্ণাঙ্ক) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_SIZE>(p, st, count, ncols, nrows_total, eps, ctx.stream());
}
