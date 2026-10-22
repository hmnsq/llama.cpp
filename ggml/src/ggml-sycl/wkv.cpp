#অন্তর্ভাৱ <sycl/sycl.hpp>
#অন্তর্ভাৱ "wkv.hpp"

constexpr পূর্ণাঙ্ক WKV_BLOCK_SIZE = 64;

// Helper function for the main kernel
template <পূর্ণাঙ্ক block_size>
স্থির শূন্য rwkv_wkv6_f32_kernel(
    ধ্রুৱ পূর্ণাঙ্ক B, ধ্রুৱ পূর্ণাঙ্ক T, ধ্রুৱ পূর্ণাঙ্ক C, ধ্রুৱ পূর্ণাঙ্ক H,
    ধ্রুৱ ভিন্ন* k, ধ্রুৱ ভিন্ন* v, ধ্রুৱ ভিন্ন* r,
    ধ্রুৱ ভিন্ন* tf, ধ্রুৱ ভিন্ন* td, ধ্রুৱ ভিন্ন* s,
    ভিন্ন* dst, ধ্রুৱ sycl::nd_item<3>& item_ct1, ভিন্ন* shared_mem) {

    ধ্রুৱ পূর্ণাঙ্ক tid = item_ct1.get_local_id(2);
    ধ্রুৱ পূর্ণাঙ্ক bid = item_ct1.get_group(2);

    ধ্রুৱ পূর্ণাঙ্ক head_size = block_size;
    ধ্রুৱ পূর্ণাঙ্ক batch_i = bid / H;
    ধ্রুৱ পূর্ণাঙ্ক head_i = bid % H;
    ধ্রুৱ পূর্ণাঙ্ক state_size = C * head_size;
    ধ্রুৱ পূর্ণাঙ্ক n_seq_tokens = T / B;

    // Set up shared memory pointers
    ভিন্ন* _k = shared_mem;
    ভিন্ন* _r = _k + head_size;
    ভিন্ন* _tf = _r + head_size;
    ভিন্ন* _td = _tf + head_size;

    // Local state array
    ভিন্ন state[block_size];

    // Load initial state
    #pragma unroll
    বাবে (পূর্ণাঙ্ক i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    // Sync threads before shared memory operations
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[head_i * head_size + tid];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    বাবে (পূর্ণাঙ্ক t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
         t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        ধ্রুৱ ভিন্ন _v = v[t];
        ভিন্ন y = 0;

        // Process in chunks of 4 for better vectorization
        sycl::float4 k4, r4, tf4, td4, s4;
        #pragma unroll
        বাবে (পূর্ণাঙ্ক j = 0; j < head_size; j += 4) {
            // Load data in vec4 chunks
            k4 = sycl::float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            r4 = sycl::float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            tf4 = sycl::float4(_tf[j], _tf[j+1], _tf[j+2], _tf[j+3]);
            td4 = sycl::float4(_td[j], _td[j+1], _td[j+2], _td[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);

            // Compute key-value product
            sycl::float4 kv4 = k4 * _v;

            // Accumulate weighted sum
            y += sycl::dot(r4, tf4 * kv4 + s4);

            // Update state
            s4 = s4 * td4 + kv4;

            // Store updated state
            state[j] = s4.x();
            state[j+1] = s4.y();
            state[j+2] = s4.z();
            state[j+3] = s4.w();
        }

        dst[t] = y;
    }

    // Save final state
    #pragma unroll
    বাবে (পূর্ণাঙ্ক i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

template <পূর্ণাঙ্ক block_size>
স্থির শূন্য rwkv_wkv7_f32_kernel(
    ধ্রুৱ পূর্ণাঙ্ক B, ধ্রুৱ পূর্ণাঙ্ক T, ধ্রুৱ পূর্ণাঙ্ক C, ধ্রুৱ পূর্ণাঙ্ক H,
    ধ্রুৱ ভিন্ন* r, ধ্রুৱ ভিন্ন* w, ধ্রুৱ ভিন্ন* k, ধ্রুৱ ভিন্ন* v,
    ধ্রুৱ ভিন্ন* a, ধ্রুৱ ভিন্ন* b, ধ্রুৱ ভিন্ন* s,
    ভিন্ন* dst, ধ্রুৱ sycl::nd_item<3>& item_ct1, ভিন্ন* shared_mem) {

    ধ্রুৱ পূর্ণাঙ্ক tid = item_ct1.get_local_id(2);
    ধ্রুৱ পূর্ণাঙ্ক bid = item_ct1.get_group(2);

    ধ্রুৱ পূর্ণাঙ্ক head_size = block_size;
    ধ্রুৱ পূর্ণাঙ্ক batch_i = bid / H;
    ধ্রুৱ পূর্ণাঙ্ক head_i = bid % H;
    ধ্রুৱ পূর্ণাঙ্ক state_size = C * head_size;
    ধ্রুৱ পূর্ণাঙ্ক n_seq_tokens = T / B;

    ভিন্ন* _r = shared_mem;
    ভিন্ন* _w = _r + head_size;
    ভিন্ন* _k = _w + head_size;
    ভিন্ন* _a = _k + head_size;
    ভিন্ন* _b = _a + head_size;

    ভিন্ন state[block_size];

    #pragma unroll
    বাবে (পূর্ণাঙ্ক i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
    }

    বাবে (পূর্ণাঙ্ক t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
         t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        ধ্রুৱ ভিন্ন _v = v[t];
        ভিন্ন y = 0, sa = 0;
        sycl::float4 a4, s4;

        #pragma unroll
        বাবে (পূর্ণাঙ্ক j = 0; j < head_size; j += 4) {
            a4 = sycl::float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa += sycl::dot(a4, s4);
        }

        sycl::float4 r4, w4, k4, b4;
        #pragma unroll
        বাবে (পূর্ণাঙ্ক j = 0; j < head_size; j += 4) {
            r4 = sycl::float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            w4 = sycl::float4(_w[j], _w[j+1], _w[j+2], _w[j+3]);
            k4 = sycl::float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            b4 = sycl::float4(_b[j], _b[j+1], _b[j+2], _b[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);

            sycl::float4 kv4 = k4 * _v;

            s4 = s4 * w4 + kv4 + sa * b4;
            y += sycl::dot(r4, s4);

            state[j] = s4.x();
            state[j+1] = s4.y();
            state[j+2] = s4.z();
            state[j+3] = s4.w();
        }

        dst[t] = y;
    }

    #pragma unroll
    বাবে (পূর্ণাঙ্ক i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
    }
}

শূন্য ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ধ্রুৱ ভিন্ন* k_d = (ধ্রুৱ ভিন্ন*)dst->src[0]->data;
    ধ্রুৱ ভিন্ন* v_d = (ধ্রুৱ ভিন্ন*)dst->src[1]->data;
    ধ্রুৱ ভিন্ন* r_d = (ধ্রুৱ ভিন্ন*)dst->src[2]->data;
    ধ্রুৱ ভিন্ন* tf_d = (ধ্রুৱ ভিন্ন*)dst->src[3]->data;
    ধ্রুৱ ভিন্ন* td_d = (ধ্রুৱ ভিন্ন*)dst->src[4]->data;
    ধ্রুৱ ভিন্ন* s_d = (ধ্রুৱ ভিন্ন*)dst->src[5]->data;
    ভিন্ন* dst_d = (ভিন্ন*)dst->data;

    ধ্রুৱ int64_t B = dst->src[5]->ne[1];
    ধ্রুৱ int64_t T = dst->src[0]->ne[2];
    ধ্রুৱ int64_t C = dst->ne[0];
    ধ্রুৱ int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_SIZE || C / H == WKV_BLOCK_SIZE * 2); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    ধ্রুৱ size_t shared_mem_size = C / H * 4 * পরিমাণ(ভিন্ন); // For k, r, tf, td
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    যদি (C / H == WKV_BLOCK_SIZE) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ভিন্ন, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_SIZE>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (ভিন্ন*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } অন্যথা {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ভিন্ন, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_SIZE * 2>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (ভিন্ন*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}

শূন্য ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ধ্রুৱ ভিন্ন* r_d = (ধ্রুৱ ভিন্ন*)dst->src[0]->data;
    ধ্রুৱ ভিন্ন* w_d = (ধ্রুৱ ভিন্ন*)dst->src[1]->data;
    ধ্রুৱ ভিন্ন* k_d = (ধ্রুৱ ভিন্ন*)dst->src[2]->data;
    ধ্রুৱ ভিন্ন* v_d = (ধ্রুৱ ভিন্ন*)dst->src[3]->data;
    ধ্রুৱ ভিন্ন* a_d = (ধ্রুৱ ভিন্ন*)dst->src[4]->data;
    ধ্রুৱ ভিন্ন* b_d = (ধ্রুৱ ভিন্ন*)dst->src[5]->data;
    ধ্রুৱ ভিন্ন* s_d = (ধ্রুৱ ভিন্ন*)dst->src[6]->data;
    ভিন্ন* dst_d = (ভিন্ন*)dst->data;

    ধ্রুৱ int64_t B = dst->src[6]->ne[1];
    ধ্রুৱ int64_t T = dst->src[0]->ne[2];
    ধ্রুৱ int64_t C = dst->ne[0];
    ধ্রুৱ int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_SIZE || C / H == WKV_BLOCK_SIZE * 2);

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    ধ্রুৱ size_t shared_mem_size = C / H * 5 * পরিমাণ(ভিন্ন); // For r, w, k, a, b
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    যদি (C / H == WKV_BLOCK_SIZE) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ভিন্ন, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_SIZE>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (ভিন্ন*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } অন্যথা {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ভিন্ন, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_SIZE * 2>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (ভিন্ন*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}
