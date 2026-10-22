#অন্তর্ভাব "norm.hpp"
#অন্তর্ভাব "ggml-sycl/common.hpp"
#অন্তর্ভাব "ggml-sycl/presets.hpp"

স্থির শূন্য norm_f32(ধ্রুবক দশমিক* x, দশমিক* dst, ধ্রুবক পূর্ণসংখ্যা ncols,
    ধ্রুবক int64_t src_stride_col, ধ্রুবক int64_t src_stride_row, ধ্রুবক int64_t src_stride_channel, ধ্রুবক int64_t src_stride_sample,
    ধ্রুবক int64_t dst_stride_col, ধ্রুবক int64_t dst_stride_row, ধ্রুবক int64_t dst_stride_channel, ধ্রুবক int64_t dst_stride_sample,
    ধ্রুবক দশমিক eps, ধ্রুবক sycl::nd_item<3>& item_ct1, sycl::float2* s_যোগ, পূর্ণসংখ্যা block_আকার_2) {

    ধ্রুবক পূর্ণসংখ্যা nthreads = item_ct1.get_local_range(2);
    ধ্রুবক পূর্ণসংখ্যা sample  = item_ct1.get_group(0);
    ধ্রুবক পূর্ণসংখ্যা channel = item_ct1.get_group(1);
    ধ্রুবক পূর্ণসংখ্যা row     = item_ct1.get_group(2);

    ধ্রুবক পূর্ণসংখ্যা tid = item_ct1.get_local_id(2);
    ধ্রুবক পূর্ণসংখ্যা nwarps = nthreads / WARP_আকার;

    ধ্রুবক auto src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    ধ্রুবক auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x += src_offset;
    dst += dst_offset;

    sycl::float2 mean_var = sycl::float2(0.f, 0.f);

    জন্য (পূর্ণসংখ্যা col = tid; col < ncols; col += block_আকার_2) {
        ধ্রুবক দশমিক xi = x[col * src_stride_col];
        mean_var.x() += xi;
        mean_var.y() += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_যোগ(mean_var, item_ct1);
    যদি  (block_আকার_2 > WARP_আকার) {
        ধ্রুবক auto sub_group = item_ct1.get_sub_group();
        ধ্রুবক auto sg_id = sub_group.get_group_linear_id();
        ধ্রুবক auto wi_in_sg = sub_group.get_local_linear_id();
        যদি (wi_in_sg == 0) {
            s_যোগ[sg_id] = mean_var;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        mean_var = 0.f;
        ধ্রুবক size_t nreduce = ceil_div(nwarps, WARP_আকার);
        জন্য (size_t i = 0; i < nreduce; i += 1)
        {
            mean_var += s_যোগ[wi_in_sg + i * WARP_আকার];
        }
        mean_var = warp_reduce_যোগ(mean_var, item_ct1);
    }

    ধ্রুবক দশমিক mean = mean_var.x() / ncols;
    ধ্রুবক দশমিক var = mean_var.y() / ncols - mean * mean;
    ধ্রুবক দশমিক inv_std = sycl::rsqrt(var + eps);

    জন্য (পূর্ণসংখ্যা col = tid; col < ncols; col += block_আকার_2) {
        dst[col * dst_stride_col] = (x[col * src_stride_col] - mean) * inv_std;
    }
}

স্থির শূন্য group_norm_f32(ধ্রুবক দশমিক* x, দশমিক* dst, ধ্রুবক পূর্ণসংখ্যা group_আকার, ধ্রুবক পূর্ণসংখ্যা ne_elements, ধ্রুবক দশমিক eps,
    ধ্রুবক sycl::nd_item<3>& item_ct1, দশমিক* s_যোগ, পূর্ণসংখ্যা block_আকার_2) {
    পূর্ণসংখ্যা start = item_ct1.get_group(2) * group_আকার;
    পূর্ণসংখ্যা end = start + group_আকার;
    ধ্রুবক পূর্ণসংখ্যা nthreads = item_ct1.get_local_range(2);
    ধ্রুবক পূর্ণসংখ্যা nwarps = nthreads / WARP_আকার;
    start += item_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_আকার;

    যদি (end >= ne_elements) {
        end = ne_elements;
    }

    দশমিক tmp = 0.0f; // partial sum for thread in warp

    জন্য (পূর্ণসংখ্যা j = start; j < end; j += block_আকার_2) {
        tmp += x[j];
    }

    tmp = warp_reduce_যোগ(tmp, item_ct1);
    যদি (block_আকার_2 > WARP_আকার) {

        পূর্ণসংখ্যা warp_id = item_ct1.get_local_id(2) / WARP_আকার;
        পূর্ণসংখ্যা lane_id = item_ct1.get_local_id(2) % WARP_আকার;
        যদি (lane_id == 0) {
            s_যোগ[warp_id] = tmp;
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
        জন্য (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_যোগ[lane_id + i * WARP_আকার];
        }
        tmp = warp_reduce_যোগ(tmp, item_ct1);
    }

    দশমিক mean = tmp / group_আকার;
    tmp = 0.0f;

    জন্য (পূর্ণসংখ্যা j = start; j < end; j += block_আকার_2) {
        দশমিক xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_যোগ(tmp, item_ct1);
    যদি (block_আকার_2 > WARP_আকার) {

        পূর্ণসংখ্যা warp_id = item_ct1.get_local_id(2) / WARP_আকার;
        পূর্ণসংখ্যা lane_id = item_ct1.get_local_id(2) % WARP_আকার;
        যদি (lane_id == 0) {
            s_যোগ[warp_id] = tmp;
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
        জন্য (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_যোগ[lane_id + i * WARP_আকার];
        }
        tmp = warp_reduce_যোগ(tmp, item_ct1);
    }

    দশমিক variance = tmp / group_আকার;
    দশমিক scale = sycl::rsqrt(variance + eps);
    জন্য (পূর্ণসংখ্যা j = start; j < end; j += block_আকার_2) {
        dst[j] *= scale;
    }
}

template <যৌক্তিক do_multiply = মিথ্যা, যৌক্তিক do_add = মিথ্যা>
স্থির শূন্য rms_norm_f32(ধ্রুবক দশমিক* x, দশমিক* dst, ধ্রুবক পূর্ণসংখ্যা ncols,
    ধ্রুবক int64_t src_stride_col, ধ্রুবক int64_t src_stride_row, ধ্রুবক int64_t src_stride_channel, ধ্রুবক int64_t src_stride_sample,
    ধ্রুবক int64_t dst_stride_col, ধ্রুবক int64_t dst_stride_row, ধ্রুবক int64_t dst_stride_channel, ধ্রুবক int64_t dst_stride_sample,
    ধ্রুবক দশমিক eps, ধ্রুবক sycl::nd_item<3>& item_ct1, দশমিক* s_যোগ, পূর্ণসংখ্যা block_আকার_2,
    ধ্রুবক দশমিক* mul = nullptr, ধ্রুবক int64_t mul_stride_row = 0, ধ্রুবক int64_t mul_stride_channel = 0,
    ধ্রুবক int64_t mul_stride_sample = 0, ধ্রুবক পূর্ণসংখ্যা mul_nrows = 0, ধ্রুবক পূর্ণসংখ্যা mul_nchannels = 0, ধ্রুবক পূর্ণসংখ্যা mul_nsamples = 0,
    ধ্রুবক দশমিক* add = nullptr, ধ্রুবক int64_t add_stride_row = 0, ধ্রুবক int64_t add_stride_channel = 0,
    ধ্রুবক int64_t add_stride_sample = 0, ধ্রুবক পূর্ণসংখ্যা add_nrows = 0, ধ্রুবক পূর্ণসংখ্যা add_nchannels = 0, ধ্রুবক পূর্ণসংখ্যা add_nsamples = 0,
    ধ্রুবক দশমিক scale_mul = 1.0f) {

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    ধ্রুবক পূর্ণসংখ্যা sample  = item_ct1.get_group(0);
    ধ্রুবক পূর্ণসংখ্যা channel = item_ct1.get_group(1);
    ধ্রুবক পূর্ণসংখ্যা row     = item_ct1.get_group(2);

    ধ্রুবক পূর্ণসংখ্যা nthreads = item_ct1.get_local_range(2);

    ধ্রুবক পূর্ণসংখ্যা tid = item_ct1.get_local_id(2);
    ধ্রুবক পূর্ণসংখ্যা nwarps = nthreads / WARP_আকার;

    ধ্রুবক auto src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    ধ্রুবক auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x   += src_offset;
    dst += dst_offset;

    যদি constexpr (do_multiply) {
        ধ্রুবক পূর্ণসংখ্যা mul_row     = row     % mul_nrows;
        ধ্রুবক পূর্ণসংখ্যা mul_channel = channel % mul_nchannels;
        ধ্রুবক পূর্ণসংখ্যা mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    যদি constexpr (do_add) {
        ধ্রুবক পূর্ণসংখ্যা add_row     = row     % add_nrows;
        ধ্রুবক পূর্ণসংখ্যা add_channel = channel % add_nchannels;
        ধ্রুবক পূর্ণসংখ্যা add_sample  = sample  % add_nsamples;
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    দশমিক tmp = 0.0f; // partial sum for thread in warp

    জন্য (পূর্ণসংখ্যা col = tid; col < ncols; col += block_আকার_2) {
        ধ্রুবক দশমিক xi = x[col * src_stride_col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_যোগ(tmp, item_ct1);
    যদি (block_আকার_2 > WARP_আকার) {
        ধ্রুবক auto sub_group = item_ct1.get_sub_group();
        ধ্রুবক auto sg_id = sub_group.get_group_linear_id();
        ধ্রুবক auto wi_in_sg = sub_group.get_local_linear_id();
        যদি (wi_in_sg == 0) {
            s_যোগ[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        ধ্রুবক size_t nreduce = ceil_div(nwarps, WARP_আকার);
        tmp = 0.f;
        জন্য (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_যোগ[wi_in_sg + i * WARP_আকার];
        }
        tmp = warp_reduce_যোগ(tmp, item_ct1);
    }

    ধ্রুবক দশমিক mean = tmp / ncols;
    ধ্রুবক দশমিক scale = sycl::rsqrt(mean + eps);

    জন্য (পূর্ণসংখ্যা col = tid; col < ncols; col += block_আকার_2) {
        যদি constexpr (do_multiply && do_add) {
            dst[col * dst_stride_col] = scale * x[col * src_stride_col] * mul[col] + add[col];
        } অন্যথা যদি constexpr (do_multiply) {
            dst[col * dst_stride_col] = scale * x[col * src_stride_col] * mul[col];
        } অন্যথা {
            // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
            // the explicit temporary keeps the float evaluation order identical to
            // running rms_norm and scale as two separate kernels
            ধ্রুবক দশমিক v = scale * x[col * src_stride_col];
            dst[col * dst_stride_col] = v * scale_mul;
        }
    }
}

template<পূর্ণসংখ্যা warp_আকার>
স্থির শূন্য l2_norm_f32(ধ্রুবক দশমিক * x, দশমিক * dst, ধ্রুবক পূর্ণসংখ্যা ncols,
    ধ্রুবক int64_t src_stride_col, ধ্রুবক int64_t src_stride_row, ধ্রুবক int64_t src_stride_channel,
    ধ্রুবক int64_t src_stride_sample, ধ্রুবক int64_t dst_stride_col, ধ্রুবক int64_t dst_stride_row,
    ধ্রুবক int64_t dst_stride_channel, ধ্রুবক int64_t dst_stride_sample, ধ্রুবক দশমিক eps,
    ধ্রুবক sycl::nd_item<3>& item_ct1, দশমিক* s_যোগ, ধ্রুবক পূর্ণসংখ্যা block_আকার_2) {

    ধ্রুবক পূর্ণসংখ্যা row     = item_ct1.get_group(2);
    ধ্রুবক পূর্ণসংখ্যা channel = item_ct1.get_group(1);
    ধ্রুবক পূর্ণসংখ্যা sample  = item_ct1.get_group(0);
    ধ্রুবক পূর্ণসংখ্যা tid     = item_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + row*src_stride_row;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + row*dst_stride_row;

    দশমিক tmp = 0.0f; // partial sum for thread in warp

    জন্য (পূর্ণসংখ্যা col = tid; col < ncols; col += block_আকার_2) {
        ধ্রুবক দশমিক xi = x[col * src_stride_col];
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::যোগ_4, warp_আকার>(tmp, s_যোগ, block_আকার_2);
    ধ্রুবক দশমিক scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    জন্য (পূর্ণসংখ্যা col = tid; col < ncols; col += block_আকার_2) {
        dst[col * dst_stride_col] = scale * x[col * src_stride_col];
    }
}

স্থির শূন্য norm_f32_sycl(ধ্রুবক দশমিক * x, দশমিক * dst, ধ্রুবক পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা nrows, ধ্রুবক পূর্ণসংখ্যা nchannels, ধ্রুবক পূর্ণসংখ্যা nsamples,
    ধ্রুবক int64_t src_stride_col, ধ্রুবক int64_t src_stride_row, ধ্রুবক int64_t src_stride_channel, ধ্রুবক int64_t src_stride_sample,
    ধ্রুবক int64_t dst_stride_col, ধ্রুবক int64_t dst_stride_row, ধ্রুবক int64_t dst_stride_channel, ধ্রুবক int64_t dst_stride_sample,
        ধ্রুবক দশমিক eps, queue_ptr stream, পূর্ণসংখ্যা device) {

    ধ্রুবক sycl::range<3> global_dims(nsamples, nchannels, nrows);
    যদি (ncols < 1024) {
        ধ্রুবক sycl::range<3> block_dims(1, 1, WARP_আকার);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_আকার);
                });
            });
    }
    অন্যথা {
        ধ্রুবক পূর্ণসংখ্যা work_group_আকার = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_আকার % (WARP_আকার * WARP_আকার) == 0);
        ধ্রুবক sycl::range<3> block_dims(1, 1, work_group_আকার);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_যোগ_acc_ct1(
                            sycl::range<1>(work_group_আকার / WARP_আকার), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_যোগ_acc_ct1), work_group_আকার);
                });
            });
    }
}

স্থির শূন্য group_norm_f32_sycl(ধ্রুবক দশমিক* x, দশমিক* dst,
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_groups, ধ্রুবক দশমিক eps, ধ্রুবক পূর্ণসংখ্যা group_আকার,
    ধ্রুবক পূর্ণসংখ্যা ne_elements, queue_ptr stream, পূর্ণসংখ্যা device) {
    যদি (group_আকার < 1024) {
        ধ্রুবক sycl::range<3> block_dims(1, 1, WARP_আকার);
        stream->submit([&](sycl::handler& cgh) {
            ধ্রুবক দশমিক eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    group_norm_f32(
                        x, dst, group_আকার, ne_elements, eps_ct4, item_ct1,
                        nullptr, WARP_আকার);
                });
            });
    }
    অন্যথা {
        ধ্রুবক পূর্ণসংখ্যা work_group_আকার = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_আকার % (WARP_আকার * WARP_আকার) == 0);
        ধ্রুবক sycl::range<3> block_dims(1, 1, work_group_আকার);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<দশমিক, 1> s_যোগ_acc_ct1(sycl::range<1>(work_group_আকার / WARP_আকার),
                cgh);

            ধ্রুবক দশমিক eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    group_norm_f32(x, dst, group_আকার, ne_elements,
                        eps_ct4, item_ct1,
                        get_pointer(s_যোগ_acc_ct1), work_group_আকার);
                });
            });
    }
}

স্থির শূন্য rms_norm_f32_sycl(ধ্রুবক দশমিক* x, দশমিক* dst, ধ্রুবক পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা nrows, ধ্রুবক পূর্ণসংখ্যা nchannels, ধ্রুবক পূর্ণসংখ্যা nsamples,
    ধ্রুবক int64_t src_stride_col, ধ্রুবক int64_t src_stride_row, ধ্রুবক int64_t src_stride_channel, ধ্রুবক int64_t src_stride_sample,
    ধ্রুবক int64_t dst_stride_col, ধ্রুবক int64_t dst_stride_row, ধ্রুবক int64_t dst_stride_channel, ধ্রুবক int64_t dst_stride_sample,
    ধ্রুবক দশমিক eps, queue_ptr stream, পূর্ণসংখ্যা device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    ধ্রুবক sycl::range<3> global_dims(nsamples, nchannels, nrows);
    যদি (ncols < 1024) {
        ধ্রুবক sycl::range<3> block_dims(1, 1, WARP_আকার);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_আকার);
                });
            });
    }
    অন্যথা {
        ধ্রুবক পূর্ণসংখ্যা work_group_আকার = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_আকার % (WARP_আকার * WARP_আকার) == 0);
        ধ্রুবক sycl::range<3> block_dims(1, 1, work_group_আকার);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<দশমিক, 1> s_যোগ_acc_ct1(sycl::range<1>(work_group_আকার / WARP_আকার),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_যোগ_acc_ct1), work_group_আকার);
                });
            });
    }
}

স্থির শূন্য rms_norm_scale_f32_sycl(ধ্রুবক দশমিক* x, দশমিক* dst, ধ্রুবক পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা nrows,
    ধ্রুবক পূর্ণসংখ্যা nchannels, ধ্রুবক পূর্ণসংখ্যা nsamples,
    ধ্রুবক int64_t src_stride_col, ধ্রুবক int64_t src_stride_row, ধ্রুবক int64_t src_stride_channel, ধ্রুবক int64_t src_stride_sample,
    ধ্রুবক int64_t dst_stride_col, ধ্রুবক int64_t dst_stride_row, ধ্রুবক int64_t dst_stride_channel, ধ্রুবক int64_t dst_stride_sample,
    ধ্রুবক দশমিক eps, ধ্রুবক দশমিক scale_mul, queue_ptr stream, পূর্ণসংখ্যা device) {
    ধ্রুবক sycl::range<3> global_dims(nsamples, nchannels, nrows);
    যদি (ncols < 1024) {
        ধ্রুবক sycl::range<3> block_dims(1, 1, WARP_আকার);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_আকার,
                        nullptr, 0, 0, 0, 0, 0, 0,
                        nullptr, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
    অন্যথা {
        ধ্রুবক পূর্ণসংখ্যা work_group_আকার = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_আকার % (WARP_আকার * WARP_আকার) == 0);
        ধ্রুবক sycl::range<3> block_dims(1, 1, work_group_আকার);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<দশমিক, 1> s_যোগ_acc_ct1(sycl::range<1>(work_group_আকার / WARP_আকার),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_যোগ_acc_ct1), work_group_আকার,
                        nullptr, 0, 0, 0, 0, 0, 0,
                        nullptr, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
}

স্থির শূন্য rms_norm_mul_f32_sycl(ধ্রুবক দশমিক* x, ধ্রুবক দশমিক* mul, দশমিক* dst, ধ্রুবক পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা nrows,
        ধ্রুবক পূর্ণসংখ্যা nchannels, ধ্রুবক পূর্ণসংখ্যা nsamples,
        ধ্রুবক int64_t src_stride_col, ধ্রুবক int64_t src_stride_row, ধ্রুবক int64_t src_stride_channel, ধ্রুবক int64_t src_stride_sample,
        ধ্রুবক int64_t dst_stride_col, ধ্রুবক int64_t dst_stride_row, ধ্রুবক int64_t dst_stride_channel, ধ্রুবক int64_t dst_stride_sample,
        ধ্রুবক int64_t mul_stride_row, ধ্রুবক int64_t mul_stride_channel, ধ্রুবক int64_t mul_stride_sample,
        ধ্রুবক পূর্ণসংখ্যা mul_nrows, ধ্রুবক পূর্ণসংখ্যা mul_nchannels, ধ্রুবক পূর্ণসংখ্যা mul_nsamples,
        ধ্রুবক দশমিক eps, queue_ptr stream, পূর্ণসংখ্যা device) {
    ধ্রুবক sycl::range<3> global_dims(nsamples, nchannels, nrows);
    যদি (ncols < 1024) {
        ধ্রুবক sycl::range<3> block_dims(1, 1, WARP_আকার);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    rms_norm_f32<সত্য>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_আকার,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    অন্যথা {
        ধ্রুবক পূর্ণসংখ্যা work_group_আকার = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_আকার % (WARP_আকার * WARP_আকার) == 0);
        ধ্রুবক sycl::range<3> block_dims(1, 1, work_group_আকার);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<দশমিক, 1> s_যোগ_acc_ct1(sycl::range<1>(work_group_আকার / WARP_আকার), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    rms_norm_f32<সত্য>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_যোগ_acc_ct1), work_group_আকার,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

স্থির শূন্য rms_norm_mul_add_f32_sycl(ধ্রুবক দশমিক* x, ধ্রুবক দশমিক* mul, ধ্রুবক দশমিক* add, দশমিক* dst,
        ধ্রুবক পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা nrows, ধ্রুবক পূর্ণসংখ্যা nchannels, ধ্রুবক পূর্ণসংখ্যা nsamples,
        ধ্রুবক int64_t src_stride_col, ধ্রুবক int64_t src_stride_row, ধ্রুবক int64_t src_stride_channel, ধ্রুবক int64_t src_stride_sample,
        ধ্রুবক int64_t dst_stride_col, ধ্রুবক int64_t dst_stride_row, ধ্রুবক int64_t dst_stride_channel, ধ্রুবক int64_t dst_stride_sample,
        ধ্রুবক int64_t mul_stride_row, ধ্রুবক int64_t mul_stride_channel, ধ্রুবক int64_t mul_stride_sample,
        ধ্রুবক পূর্ণসংখ্যা mul_nrows, ধ্রুবক পূর্ণসংখ্যা mul_nchannels, ধ্রুবক পূর্ণসংখ্যা mul_nsamples,
        ধ্রুবক int64_t add_stride_row, ধ্রুবক int64_t add_stride_channel, ধ্রুবক int64_t add_stride_sample,
        ধ্রুবক পূর্ণসংখ্যা add_nrows, ধ্রুবক পূর্ণসংখ্যা add_nchannels, ধ্রুবক পূর্ণসংখ্যা add_nsamples,
        ধ্রুবক দশমিক eps, queue_ptr stream, পূর্ণসংখ্যা device) {
    ধ্রুবক sycl::range<3> global_dims(nsamples, nchannels, nrows);
    যদি (ncols < 1024) {
        ধ্রুবক sycl::range<3> block_dims(1, 1, WARP_আকার);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    rms_norm_f32<সত্য, সত্য>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_আকার,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add, add_stride_row, add_stride_channel, add_stride_sample, add_nrows, add_nchannels, add_nsamples);
                });
            });
    }
    অন্যথা {
        ধ্রুবক পূর্ণসংখ্যা work_group_আকার = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_আকার % (WARP_আকার * WARP_আকার) == 0);
        ধ্রুবক sycl::range<3> block_dims(1, 1, work_group_আকার);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<দশমিক, 1> s_যোগ_acc_ct1(sycl::range<1>(work_group_আকার / WARP_আকার), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                    rms_norm_f32<সত্য, সত্য>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_যোগ_acc_ct1), work_group_আকার,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add, add_stride_row, add_stride_channel, add_stride_sample, add_nrows, add_nchannels, add_nsamples);
                });
            });
    }
}

template<পূর্ণসংখ্যা warp_আকার>
স্থির শূন্য l2_norm_f32_sycl(ধ্রুবক দশমিক *   x,
                             দশমিক *         dst,
                             ধ্রুবক পূর্ণসংখ্যা       ncols,
                             ধ্রুবক পূর্ণসংখ্যা       nrows,
                             ধ্রুবক পূর্ণসংখ্যা       nchannels,
                             ধ্রুবক পূর্ণসংখ্যা       nsamples,
                             ধ্রুবক int64_t   src_stride_col,
                             ধ্রুবক int64_t   src_stride_row,
                             ধ্রুবক int64_t   src_stride_channel,
                             ধ্রুবক int64_t   src_stride_sample,
                             ধ্রুবক int64_t   dst_stride_col,
                             ধ্রুবক int64_t   dst_stride_row,
                             ধ্রুবক int64_t   dst_stride_channel,
                             ধ্রুবক int64_t   dst_stride_sample,
                             ধ্রুবক দশমিক     eps,
                             queue_ptr       stream,
                             পূর্ণসংখ্যা             device) {
    ধ্রুবক dpct::dim3 blocks_অঙ্ক(nrows, nchannels, nsamples);

    যদি (ncols < 1024) {
        ধ্রুবক dpct::dim3 block_dims(warp_আকার, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(blocks_অঙ্ক * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(warp_আকার)]] {
                    l2_norm_f32<warp_আকার>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1,
                        nullptr, warp_আকার);
                });
            });
    }
    অন্যথা {
        ধ্রুবক পূর্ণসংখ্যা work_group_আকার = ggml_sycl_info().max_work_group_sizes[device];
        প্রতিজ্ঞা(work_group_আকার % (warp_আকার * warp_আকার) == 0);
        ধ্রুবক sycl::range<3> block_dims(1, 1, work_group_আকার);
        পূর্ণসংখ্যা lsm_আকার =  block_dims[2] > warp_আকার ? work_group_আকার / warp_আকার * আকার(দশমিক): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<দশমিক, 1> s_যোগ_acc_ct1(sycl::range<1>(lsm_আকার),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(blocks_অঙ্ক * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_আকার(warp_আকার)]] {
                    l2_norm_f32<warp_আকার>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_যোগ_acc_ct1), work_group_আকার);
                });
            });
    }
}

// Batched L2 norm: N independent same-shape F32 tensors in one launch; the tensor
// index is folded into grid dim0 and each row's reduction is identical to the
// single-tensor kernel, so the result is bit-exact.
কাঠামো l2_batch_ptrs {
    ধ্রুবক দশমিক * src[GGML_SYCL_L2_BATCH_MAX];
    দশমিক *       dst[GGML_SYCL_L2_BATCH_MAX];
};

// One stride set shared by the whole batch: the caller only groups tensors whose nb[]
// all match, so per-tensor state stays two pointers.
কাঠামো l2_batch_strides {
    পূর্ণসংখ্যা     ne1, ne2;
    int64_t ss0, ss1, ss2, ss3;
    int64_t ds0, ds1, ds2, ds3;
};

template <পূর্ণসংখ্যা warp_আকার>
স্থির শূন্য l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, ধ্রুবক পূর্ণসংখ্যা ncols, ধ্রুবক দশমিক eps,
                              ধ্রুবক sycl::nd_item<3> & item_ct1) {
    ধ্রুবক পূর্ণসংখ্যা t   = item_ct1.get_group(0);  // tensor index
    ধ্রুবক পূর্ণসংখ্যা r   = item_ct1.get_group(2);  // flattened row over ne1*ne2*ne3
    ধ্রুবক পূর্ণসংখ্যা tid = item_ct1.get_local_id(2);

    ধ্রুবক পূর্ণসংখ্যা i1 = r % st.ne1;
    ধ্রুবক পূর্ণসংখ্যা i2 = (r / st.ne1) % st.ne2;
    ধ্রুবক পূর্ণসংখ্যা i3 = r / (st.ne1 * st.ne2);

    ধ্রুবক দশমিক * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    দশমিক *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    দশমিক tmp = 0.0f;
    জন্য (পূর্ণসংখ্যা col = tid; col < ncols; col += warp_আকার) {
        ধ্রুবক দশমিক xi = x[col * st.ss0];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::যোগ_4, warp_আকার>(tmp, (দশমিক *) nullptr, warp_আকার);
    ধ্রুবক দশমিক scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    জন্য (পূর্ণসংখ্যা col = tid; col < ncols; col += warp_আকার) {
        dst[col * st.ds0] = scale * x[col * st.ss0];
    }
}

template <পূর্ণসংখ্যা warp_আকার>
স্থির শূন্য l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, ধ্রুবক পূর্ণসংখ্যা n_tensors,
                                   ধ্রুবক পূর্ণসংখ্যা ncols, ধ্রুবক পূর্ণসংখ্যা nrows_যোগ, ধ্রুবক দশমিক eps,
                                   queue_ptr stream) {
    ধ্রুবক dpct::dim3 blocks_অঙ্ক(nrows_যোগ, 1, n_tensors);
    ধ্রুবক dpct::dim3 block_dims(warp_আকার, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(blocks_অঙ্ক * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(warp_আকার)]] {
                l2_norm_f32_batch<warp_আকার>(p, st, ncols, eps, item_ct1);
            });
    });
}

শূন্য ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ধ্রুবক ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ধ্রুবক দশমিক * src0_dd = static_cast<ধ্রুবক দশমিক *>(dst->src[0]->তথ্য);
    দশমিক *       dst_dd  = static_cast<দশমিক *>(dst->তথ্য);

    দশমিক eps;
    memcpy(&eps, dst->op_params, আকার(দশমিক));
    GGML_ASSERT(eps >= 0.0f);
    ধ্রুবক size_t ts0 = ggml_type_আকার(src0->type);
    ধ্রুবক size_t tdst = ggml_type_আকার(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ধ্রুবক int64_t ss0 = nb00 / ts0;
    ধ্রুবক int64_t ss1 = nb01 / ts0;
    ধ্রুবক int64_t ss2 = nb02 / ts0;
    ধ্রুবক int64_t ss3 = nb03 / ts0;
    ধ্রুবক int64_t ds0 = nb0 / tdst;
    ধ্রুবক int64_t ds1 = nb1 / tdst;
    ধ্রুবক int64_t ds2 = nb2 / tdst;
    ধ্রুবক int64_t ds3 = nb3 / tdst;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
}

শূন্য ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    পূর্ণসংখ্যা অঙ্ক_groups = dst->op_params[0];
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুবক দশমিক * src0_dd = static_cast<ধ্রুবক দশমিক *>(dst->src[0]->তথ্য);
    দশমিক *       dst_dd  = static_cast<দশমিক *>(dst->তথ্য);

    দশমিক eps;
    memcpy(&eps, dst->op_params + 1, আকার(দশমিক));

    পূর্ণসংখ্যা group_আকার = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + অঙ্ক_groups - 1) / অঙ্ক_groups);
    group_norm_f32_sycl(src0_dd, dst_dd, অঙ্ক_groups, eps, group_আকার, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_stream, ctx.device);
}

শূন্য ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ধ্রুবক ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুবক দশমিক * src0_dd = static_cast<ধ্রুবক দশমিক *>(dst->src[0]->তথ্য);
    দশমিক *       dst_dd  = static_cast<দশমিক *>(dst->তথ্য);

    দশমিক eps;
    memcpy(&eps, dst->op_params, আকার(দশমিক));

    GGML_TENSOR_UNARY_OP_LOCALS
    ধ্রুবক size_t ts0 = ggml_type_আকার(src0->type);
    ধ্রুবক size_t tdst = ggml_type_আকার(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ধ্রুবক int64_t ss0 = nb00 / ts0;
    ধ্রুবক int64_t ss1 = nb01 / ts0;
    ধ্রুবক int64_t ss2 = nb02 / ts0;
    ধ্রুবক int64_t ss3 = nb03 / ts0;
    ধ্রুবক int64_t ds0 = nb0 / tdst;
    ধ্রুবক int64_t ds1 = nb1 / tdst;
    ধ্রুবক int64_t ds2 = nb2 / tdst;
    ধ্রুবক int64_t ds3 = nb3 / tdst;
    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
}

// Fused rms_norm + scale (the qwen35 GDN l2-norm pair build_gdn_l2_norm emits):
// the scale factor is a host scalar in the GGML_OP_SCALE node's op_params, so
// unlike the mul variants there is no second device tensor to wire up.
শূন্য ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       ggml_tensor * scale_tensor) {
    ধ্রুবক ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুবক দশমিক * src0_dd = static_cast<ধ্রুবক দশমিক *>(src0->তথ্য);
    দশমিক *       dst_dd  = static_cast<দশমিক *>(scale_tensor->তথ্য);

    দশমিক eps;
    memcpy(&eps, dst->op_params, আকার(দশমিক));
    দশমিক scale_mul;
    memcpy(&scale_mul, scale_tensor->op_params, আকার(দশমিক));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    ধ্রুবক size_t ts0 = ggml_type_আকার(src0->type);
    ধ্রুবক size_t tdst = ggml_type_আকার(scale_tensor->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(scale_tensor->nb[0] % tdst == 0 && scale_tensor->nb[1] % tdst == 0 &&
                scale_tensor->nb[2] % tdst == 0 && scale_tensor->nb[3] % tdst == 0);
    ধ্রুবক int64_t ss0 = nb00 / ts0;
    ধ্রুবক int64_t ss1 = nb01 / ts0;
    ধ্রুবক int64_t ss2 = nb02 / ts0;
    ধ্রুবক int64_t ss3 = nb03 / ts0;
    ধ্রুবক int64_t ds0 = scale_tensor->nb[0] / tdst;
    ধ্রুবক int64_t ds1 = scale_tensor->nb[1] / tdst;
    ধ্রুবক int64_t ds2 = scale_tensor->nb[2] / tdst;
    ধ্রুবক int64_t ds3 = scale_tensor->nb[3] / tdst;
    rms_norm_scale_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_stream, ctx.device);
}

শূন্য ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    ধ্রুবক ggml_tensor * rms_norm_src = dst->src[0];
    দশমিক eps = 0.0f;
    memcpy(&eps, dst->op_params, আকার(দশমিক));

    ধ্রুবক দশমিক *       src0_dd = static_cast<ধ্রুবক দশমিক *>(rms_norm_src->তথ্য);
    ধ্রুবক দশমিক *       mul_dd  = nullptr;
    ধ্রুবক ggml_tensor * mul_src = nullptr;
    যদি (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ধ্রুবক দশমিক *>(mul_tensor->src[1]->তথ্য);
        mul_src = mul_tensor->src[1];
    } অন্যথা যদি (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ধ্রুবক দশমিক *>(mul_tensor->src[0]->তথ্য);
        mul_src = mul_tensor->src[0];
    } অন্যথা {
        GGML_ASSERT(মিথ্যা);
    }
    দশমিক * dst_dd = static_cast<দশমিক *>(mul_tensor->তথ্য);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ধ্রুবক int64_t ne00 = rms_norm_src->ne[0];
    ধ্রুবক int64_t ne01 = rms_norm_src->ne[1];
    ধ্রুবক int64_t ne02 = rms_norm_src->ne[2];
    ধ্রুবক int64_t ne03 = rms_norm_src->ne[3];

    ধ্রুবক size_t ts0 = ggml_type_আকার(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ধ্রুবক int64_t s00 = rms_norm_src->nb[0] / ts0;
    ধ্রুবক int64_t s01 = rms_norm_src->nb[1] / ts0;
    ধ্রুবক int64_t s02 = rms_norm_src->nb[2] / ts0;
    ধ্রুবক int64_t s03 = rms_norm_src->nb[3] / ts0;

    ধ্রুবক size_t tdst = ggml_type_আকার(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    ধ্রুবক int64_t d00 = mul_tensor->nb[0] / tdst;
    ধ্রুবক int64_t d01 = mul_tensor->nb[1] / tdst;
    ধ্রুবক int64_t d02 = mul_tensor->nb[2] / tdst;
    ধ্রুবক int64_t d03 = mul_tensor->nb[3] / tdst;

    ধ্রুবক size_t ts_mul = ggml_type_আকার(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ধ্রুবক int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ধ্রুবক int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ধ্রুবক int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ধ্রুবক পূর্ণসংখ্যা mul_nrows     = mul_src->ne[1];
    ধ্রুবক পূর্ণসংখ্যা mul_nchannels = mul_src->ne[2];
    ধ্রুবক পূর্ণসংখ্যা mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_stream, ctx.device);
}

শূন্য ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    ধ্রুবক ggml_tensor * rms_norm_src = dst->src[0];
    দশমিক eps = 0.0f;
    memcpy(&eps, dst->op_params, আকার(দশমিক));

    ধ্রুবক দশমিক *       src0_dd = static_cast<ধ্রুবক দশমিক *>(rms_norm_src->তথ্য);
    ধ্রুবক দশমিক *       mul_dd  = nullptr;
    ধ্রুবক ggml_tensor * mul_src = nullptr;
    যদি (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<ধ্রুবক দশমিক *>(mul_tensor->src[1]->তথ্য);
        mul_src = mul_tensor->src[1];
    } অন্যথা যদি (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<ধ্রুবক দশমিক *>(mul_tensor->src[0]->তথ্য);
        mul_src = mul_tensor->src[0];
    } অন্যথা {
        GGML_ASSERT(মিথ্যা);
    }

    ধ্রুবক দশমিক *       add_dd  = nullptr;
    ধ্রুবক ggml_tensor * add_src = nullptr;
    যদি (add_tensor->src[0] == mul_tensor) {
        add_dd  = static_cast<ধ্রুবক দশমিক *>(add_tensor->src[1]->তথ্য);
        add_src = add_tensor->src[1];
    } অন্যথা যদি (add_tensor->src[1] == mul_tensor) {
        add_dd  = static_cast<ধ্রুবক দশমিক *>(add_tensor->src[0]->তথ্য);
        add_src = add_tensor->src[0];
    } অন্যথা {
        GGML_ASSERT(মিথ্যা);
    }

    দশমিক * dst_dd = static_cast<দশমিক *>(add_tensor->তথ্য);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    ধ্রুবক int64_t ne00 = rms_norm_src->ne[0];
    ধ্রুবক int64_t ne01 = rms_norm_src->ne[1];
    ধ্রুবক int64_t ne02 = rms_norm_src->ne[2];
    ধ্রুবক int64_t ne03 = rms_norm_src->ne[3];

    ধ্রুবক size_t ts0 = ggml_type_আকার(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    ধ্রুবক int64_t s00 = rms_norm_src->nb[0] / ts0;
    ধ্রুবক int64_t s01 = rms_norm_src->nb[1] / ts0;
    ধ্রুবক int64_t s02 = rms_norm_src->nb[2] / ts0;
    ধ্রুবক int64_t s03 = rms_norm_src->nb[3] / ts0;

    ধ্রুবক size_t tdst = ggml_type_আকার(add_tensor->type);
    GGML_ASSERT(add_tensor->nb[0] == tdst);
    ধ্রুবক int64_t d00 = add_tensor->nb[0] / tdst;
    ধ্রুবক int64_t d01 = add_tensor->nb[1] / tdst;
    ধ্রুবক int64_t d02 = add_tensor->nb[2] / tdst;
    ধ্রুবক int64_t d03 = add_tensor->nb[3] / tdst;

    ধ্রুবক size_t ts_mul = ggml_type_আকার(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    ধ্রুবক int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    ধ্রুবক int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    ধ্রুবক int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    ধ্রুবক পূর্ণসংখ্যা mul_nrows     = mul_src->ne[1];
    ধ্রুবক পূর্ণসংখ্যা mul_nchannels = mul_src->ne[2];
    ধ্রুবক পূর্ণসংখ্যা mul_nsamples  = mul_src->ne[3];

    ধ্রুবক size_t ts_add = ggml_type_আকার(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    ধ্রুবক int64_t add_s01 = add_src->nb[1] / ts_add;
    ধ্রুবক int64_t add_s02 = add_src->nb[2] / ts_add;
    ধ্রুবক int64_t add_s03 = add_src->nb[3] / ts_add;
    ধ্রুবক পূর্ণসংখ্যা add_nrows     = add_src->ne[1];
    ধ্রুবক পূর্ণসংখ্যা add_nchannels = add_src->ne[2];
    ধ্রুবক পূর্ণসংখ্যা add_nsamples  = add_src->ne[3];

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

    দশমিক eps = 1e-5f;
    std::memcpy(&eps, dst->op_params, আকার(দশমিক));
    যদি (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    ধ্রুবক দশমিক * g_base  = static_cast<ধ্রুবক দশমিক *>(dst->src[0]->তথ্য); // dz
    ধ্রুবক দশমিক * x_base  = static_cast<ধ্রুবক দশমিক *>(dst->src[1]->তথ্য); // x
          দশমিক * dx_base = static_cast<      দশমিক *>(dst->তথ্য);

    ধ্রুবক int64_t D  = dst->ne[0];
    ধ্রুবক int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (শূন্য) n3;
    ধ্রুবক int64_t N  = ggml_nrows(dst);
    যদি (D == 0 || N == 0) ফেরত;

    ধ্রুবক ggml_tensor *G = dst->src[0];
    ধ্রুবক ggml_tensor *X = dst->src[1];
    ধ্রুবক পূর্ণসংখ্যা ts = (পূর্ণসংখ্যা) ggml_type_আকার(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    ধ্রুবক int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    ধ্রুবক int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    ধ্রুবক int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::queue_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    ধ্রুবক পূর্ণসংখ্যা device_max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
    auto roundup = [](পূর্ণসংখ্যা v, পূর্ণসংখ্যা m) { ফেরত ((v + m - 1) / m) * m; };
    পূর্ণসংখ্যা wg_cap = 256;
    যদি (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    পূর্ণসংখ্যা WG = std::max(WARP_আকার, std::min(roundup((পূর্ণসংখ্যা)std::min<int64_t>(D, wg_cap), WARP_আকার), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        ধ্রুবক পূর্ণসংখ্যা nwarps_loc = std::max(1, WG / WARP_আকার);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        auto l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        auto l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                ধ্রুবক পূর্ণসংখ্যা row = item_ct1.get_group(2);
                ধ্রুবক পূর্ণসংখ্যা tid = item_ct1.get_local_id(2);

                ধ্রুবক int64_t i1 = row % n1;
                ধ্রুবক int64_t i2 = (row / n1) % n2;
                ধ্রুবক int64_t i3 = row / (n1 * n2);

                ধ্রুবক দশমিক *__restrict x_row = x_base + i3 * xs3 + i2 * xs2 + i1 * xs1;
                ধ্রুবক দশমিক *__restrict g_row = g_base + i3 * gs3 + i2 * gs2 + i1 * gs1;
                দশমিক *__restrict d_row       = dx_base + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                দশমিক যোগ_xx = 0.f, যোগ_xg = 0.f;
#ifndef GGML_SYCL_RMS_BACK_FAST
                দশমিক c_xx = 0.f, c_xg = 0.f;
#endif
                জন্য (int64_t col = tid; col < D; col += WG) {
                    ধ্রুবক দশমিক xv = x_row[col];
                    ধ্রুবক দশমিক gv = g_row[col];
#ifdef GGML_SYCL_RMS_BACK_FAST
                    যোগ_xx += xv * xv;
                    যোগ_xg += xv * gv;
#অন্যথা
                    দশমিক y1 = xv * xv - c_xx;
                    দশমিক t1 = যোগ_xx + y1;
                    c_xx = (t1 - যোগ_xx) - y1;
                    যোগ_xx = t1;

                    দশমিক y2 = xv * gv - c_xg;
                    দশমিক t2 = যোগ_xg + y2;
                    c_xg = (t2 - যোগ_xg) - y2;
                    যোগ_xg = t2;
#endif
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(যোগ_xx,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xx
#অন্যথা
                    0.f
#endif
                );
                sycl::float2 xg = sycl::float2(যোগ_xg,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xg
#অন্যথা
                    0.f
#endif
                );
                xx = warp_reduce_যোগ(xx, item_ct1);
                xg = warp_reduce_যোগ(xg, item_ct1);

                // cross-warp reduction using local memory (single barrier)
                ধ্রুবক auto sub_group = item_ct1.get_sub_group();
                ধ্রুবক auto sg_id     = sub_group.get_group_linear_id();
                ধ্রুবক auto wi_in_sg  = sub_group.get_local_linear_id();
                ধ্রুবক পূর্ণসংখ্যা nthreads   = item_ct1.get_local_range(2);
                ধ্রুবক পূর্ণসংখ্যা nwarps     = nthreads / WARP_আকার;

                sycl::float2 xx_যোগ = xx;
                sycl::float2 xg_যোগ = xg;
                যদি (nwarps > 1) {
                    যদি (wi_in_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);

                    যদি (sg_id == 0) {
                        ধ্রুবক unsigned wi_u = wi_in_sg;
                        sycl::float2 xx_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_যোগ = warp_reduce_যোগ(xx_first, item_ct1);
                        xg_যোগ = warp_reduce_যোগ(xg_first, item_ct1);
                    } অন্যথা {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_যোগ = xx;
                        xg_যোগ = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                দশমিক inv_r = 0.f;
                দশমিক coeff = 0.f;
                যদি (tid == 0) {
                    ধ্রুবক দশমিক যোগ_xx_f  = xx_যোগ.x() + xx_যোগ.y();
                    ধ্রুবক দশমিক যোগ_xdz_f = xg_যোগ.x() + xg_যোগ.y();
                    ধ্রুবক দশমিক mean_eps  = যোগ_xx_f / (দশমিক) D + eps;
                    ধ্রুবক দশমিক যোগ_eps   = যোগ_xx_f + eps * (দশমিক) D;
                    inv_r = sycl::rsqrt(mean_eps);
                    coeff = -যোগ_xdz_f / যোগ_eps;
                }
                inv_r = sycl::group_broadcast(item_ct1.get_group(), inv_r);
                coeff = sycl::group_broadcast(item_ct1.get_group(), coeff);

                জন্য (int64_t col = tid; col < D; col += WG) {
                    d_row[col] = (g_row[col] + coeff * x_row[col]) * inv_r;
                }
            });
    });

}

শূন্য ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    ধ্রুবক ggml_tensor * src0 = dst->src[0];
    ধ্রুবক দশমিক * src0_d = (ধ্রুবক দশমিক *) src0->তথ্য;
    দশমিক * dst_d = (দশমিক *) dst->তথ্য;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    দশমিক eps;
    memcpy(&eps, dst->op_params, আকার(দশমিক));
    GGML_ASSERT(eps >= 0.0f);

    ধ্রুবক size_t ts0 = ggml_type_আকার(src0->type);
    ধ্রুবক size_t tdst = ggml_type_আকার(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    ধ্রুবক int64_t ss0 = nb00 / ts0;
    ধ্রুবক int64_t ss1 = nb01 / ts0;
    ধ্রুবক int64_t ss2 = nb02 / ts0;
    ধ্রুবক int64_t ss3 = nb03 / ts0;
    ধ্রুবক int64_t ds0 = nb0 / tdst;
    ধ্রুবক int64_t ds1 = nb1 / tdst;
    ধ্রুবক int64_t ds2 = nb2 / tdst;
    ধ্রুবক int64_t ds3 = nb3 / tdst;

    /*support both WARP_SIZE or WARP_32_SIZE in code
      choose by hardware for better performance
    */
    l2_norm_f32_sycl<WARP_আকার>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
শূন্য ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, পূর্ণসংখ্যা count) {
    ধ্রুবক ggml_tensor * s0 = nodes[0]->src[0];
    ধ্রুবক পূর্ণসংখ্যা ncols       = (পূর্ণসংখ্যা) s0->ne[0];
    ধ্রুবক পূর্ণসংখ্যা nrows_যোগ = (পূর্ণসংখ্যা) ggml_nrows(s0);
    দশমিক eps;
    memcpy(&eps, nodes[0]->op_params, আকার(দশমিক));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    জন্য (পূর্ণসংখ্যা t = 0; t < count; ++t) {
        p.src[t] = (ধ্রুবক দশমিক *) nodes[t]->src[0]->তথ্য;
        p.dst[t] = (দশমিক *) nodes[t]->তথ্য;
    }

    ধ্রুবক ggml_tensor * d0 = nodes[0];
    ধ্রুবক size_t        ts = ggml_type_আকার(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (পূর্ণসংখ্যা) s0->ne[1];
    st.ne2 = (পূর্ণসংখ্যা) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_আকার>(p, st, count, ncols, nrows_যোগ, eps, ctx.stream());
}
