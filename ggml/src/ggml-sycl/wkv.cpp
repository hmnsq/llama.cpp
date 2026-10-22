#ଅନ୍ତର୍ଭାଵ <sycl/sycl.hpp>
#ଅନ୍ତର୍ଭାଵ "wkv.hpp"

constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା WKV_BLOCK_SIZE = 64;

// Helper function for the main kernel
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size>
ସ୍ଥିର ଶୂନ୍ୟ rwkv_wkv6_f32_kernel(
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା B, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା T, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା C, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା H,
    ଅଚଳ ଦଶମିକ* k, ଅଚଳ ଦଶମିକ* v, ଅଚଳ ଦଶମିକ* r,
    ଅଚଳ ଦଶମିକ* tf, ଅଚଳ ଦଶମିକ* td, ଅଚଳ ଦଶମିକ* s,
    ଦଶମିକ* dst, ଅଚଳ sycl::nd_item<3>& item_ct1, ଦଶମିକ* shared_mem) {

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid = item_ct1.get_local_id(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bid = item_ct1.get_group(2);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା head_size = block_size;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା batch_i = bid / H;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା head_i = bid % H;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା state_size = C * head_size;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_seq_tokens = T / B;

    // Set up shared memory pointers
    ଦଶମିକ* _k = shared_mem;
    ଦଶମିକ* _r = _k + head_size;
    ଦଶମିକ* _tf = _r + head_size;
    ଦଶମିକ* _td = _tf + head_size;

    // Local state array
    ଦଶମିକ state[block_size];

    // Load initial state
    #pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    // Sync threads before shared memory operations
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[head_i * head_size + tid];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
         t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        ଅଚଳ ଦଶମିକ _v = v[t];
        ଦଶମିକ y = 0;

        // Process in chunks of 4 for better vectorization
        sycl::float4 k4, r4, tf4, td4, s4;
        #pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < head_size; j += 4) {
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
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size>
ସ୍ଥିର ଶୂନ୍ୟ rwkv_wkv7_f32_kernel(
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା B, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା T, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା C, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା H,
    ଅଚଳ ଦଶମିକ* r, ଅଚଳ ଦଶମିକ* w, ଅଚଳ ଦଶମିକ* k, ଅଚଳ ଦଶମିକ* v,
    ଅଚଳ ଦଶମିକ* a, ଅଚଳ ଦଶମିକ* b, ଅଚଳ ଦଶମିକ* s,
    ଦଶମିକ* dst, ଅଚଳ sycl::nd_item<3>& item_ct1, ଦଶମିକ* shared_mem) {

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid = item_ct1.get_local_id(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bid = item_ct1.get_group(2);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା head_size = block_size;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା batch_i = bid / H;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା head_i = bid % H;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା state_size = C * head_size;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_seq_tokens = T / B;

    ଦଶମିକ* _r = shared_mem;
    ଦଶମିକ* _w = _r + head_size;
    ଦଶମିକ* _k = _w + head_size;
    ଦଶମିକ* _a = _k + head_size;
    ଦଶମିକ* _b = _a + head_size;

    ଦଶମିକ state[block_size];

    #pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
         t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        ଅଚଳ ଦଶମିକ _v = v[t];
        ଦଶମିକ y = 0, sa = 0;
        sycl::float4 a4, s4;

        #pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < head_size; j += 4) {
            a4 = sycl::float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa += sycl::dot(a4, s4);
        }

        sycl::float4 r4, w4, k4, b4;
        #pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < head_size; j += 4) {
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
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
    }
}

ଶୂନ୍ୟ ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ଅଚଳ ଦଶମିକ* k_d = (ଅଚଳ ଦଶମିକ*)dst->src[0]->data;
    ଅଚଳ ଦଶମିକ* v_d = (ଅଚଳ ଦଶମିକ*)dst->src[1]->data;
    ଅଚଳ ଦଶମିକ* r_d = (ଅଚଳ ଦଶମିକ*)dst->src[2]->data;
    ଅଚଳ ଦଶମିକ* tf_d = (ଅଚଳ ଦଶମିକ*)dst->src[3]->data;
    ଅଚଳ ଦଶମିକ* td_d = (ଅଚଳ ଦଶମିକ*)dst->src[4]->data;
    ଅଚଳ ଦଶମିକ* s_d = (ଅଚଳ ଦଶମିକ*)dst->src[5]->data;
    ଦଶମିକ* dst_d = (ଦଶମିକ*)dst->data;

    ଅଚଳ int64_t B = dst->src[5]->ne[1];
    ଅଚଳ int64_t T = dst->src[0]->ne[2];
    ଅଚଳ int64_t C = dst->ne[0];
    ଅଚଳ int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_SIZE || C / H == WKV_BLOCK_SIZE * 2); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    ଅଚଳ size_t shared_mem_size = C / H * 4 * ଆକାର(ଦଶମିକ); // For k, r, tf, td
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    ଯଦି (C / H == WKV_BLOCK_SIZE) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ଦଶମିକ, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_SIZE>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (ଦଶମିକ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } ଅନ୍ୟଥା {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ଦଶମିକ, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_SIZE * 2>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (ଦଶମିକ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}

ଶୂନ୍ୟ ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ଅଚଳ ଦଶମିକ* r_d = (ଅଚଳ ଦଶମିକ*)dst->src[0]->data;
    ଅଚଳ ଦଶମିକ* w_d = (ଅଚଳ ଦଶମିକ*)dst->src[1]->data;
    ଅଚଳ ଦଶମିକ* k_d = (ଅଚଳ ଦଶମିକ*)dst->src[2]->data;
    ଅଚଳ ଦଶମିକ* v_d = (ଅଚଳ ଦଶମିକ*)dst->src[3]->data;
    ଅଚଳ ଦଶମିକ* a_d = (ଅଚଳ ଦଶମିକ*)dst->src[4]->data;
    ଅଚଳ ଦଶମିକ* b_d = (ଅଚଳ ଦଶମିକ*)dst->src[5]->data;
    ଅଚଳ ଦଶମିକ* s_d = (ଅଚଳ ଦଶମିକ*)dst->src[6]->data;
    ଦଶମିକ* dst_d = (ଦଶମିକ*)dst->data;

    ଅଚଳ int64_t B = dst->src[6]->ne[1];
    ଅଚଳ int64_t T = dst->src[0]->ne[2];
    ଅଚଳ int64_t C = dst->ne[0];
    ଅଚଳ int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_SIZE || C / H == WKV_BLOCK_SIZE * 2);

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    ଅଚଳ size_t shared_mem_size = C / H * 5 * ଆକାର(ଦଶମିକ); // For r, w, k, a, b
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    ଯଦି (C / H == WKV_BLOCK_SIZE) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ଦଶମିକ, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_SIZE>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (ଦଶମିକ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } ଅନ୍ୟଥା {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ଦଶମିକ, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_SIZE * 2>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (ଦଶମିକ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}
