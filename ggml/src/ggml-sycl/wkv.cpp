#અન્તર્ભાવ <sycl/sycl.hpp>
#અન્તર્ભાવ "wkv.hpp"

constexpr પૂર્ણાંક WKV_BLOCK_SIZE = 64;

// Helper function for the main kernel
template <પૂર્ણાંક block_size>
સ્થિર શૂન્ય rwkv_wkv6_f32_kernel(
    અચલ પૂર્ણાંક B, અચલ પૂર્ણાંક T, અચલ પૂર્ણાંક C, અચલ પૂર્ણાંક H,
    અચલ દશાંશ* k, અચલ દશાંશ* v, અચલ દશાંશ* r,
    અચલ દશાંશ* tf, અચલ દશાંશ* td, અચલ દશાંશ* s,
    દશાંશ* dst, અચલ sycl::nd_item<3>& item_ct1, દશાંશ* shared_mem) {

    અચલ પૂર્ણાંક tid = item_ct1.get_local_id(2);
    અચલ પૂર્ણાંક bid = item_ct1.get_group(2);

    અચલ પૂર્ણાંક head_size = block_size;
    અચલ પૂર્ણાંક batch_i = bid / H;
    અચલ પૂર્ણાંક head_i = bid % H;
    અચલ પૂર્ણાંક state_size = C * head_size;
    અચલ પૂર્ણાંક n_seq_tokens = T / B;

    // Set up shared memory pointers
    દશાંશ* _k = shared_mem;
    દશાંશ* _r = _k + head_size;
    દશાંશ* _tf = _r + head_size;
    દશાંશ* _td = _tf + head_size;

    // Local state array
    દશાંશ state[block_size];

    // Load initial state
    #pragma unroll
    માટે (પૂર્ણાંક i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    // Sync threads before shared memory operations
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[head_i * head_size + tid];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    માટે (પૂર્ણાંક t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
         t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        અચલ દશાંશ _v = v[t];
        દશાંશ y = 0;

        // Process in chunks of 4 for better vectorization
        sycl::float4 k4, r4, tf4, td4, s4;
        #pragma unroll
        માટે (પૂર્ણાંક j = 0; j < head_size; j += 4) {
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
    માટે (પૂર્ણાંક i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

template <પૂર્ણાંક block_size>
સ્થિર શૂન્ય rwkv_wkv7_f32_kernel(
    અચલ પૂર્ણાંક B, અચલ પૂર્ણાંક T, અચલ પૂર્ણાંક C, અચલ પૂર્ણાંક H,
    અચલ દશાંશ* r, અચલ દશાંશ* w, અચલ દશાંશ* k, અચલ દશાંશ* v,
    અચલ દશાંશ* a, અચલ દશાંશ* b, અચલ દશાંશ* s,
    દશાંશ* dst, અચલ sycl::nd_item<3>& item_ct1, દશાંશ* shared_mem) {

    અચલ પૂર્ણાંક tid = item_ct1.get_local_id(2);
    અચલ પૂર્ણાંક bid = item_ct1.get_group(2);

    અચલ પૂર્ણાંક head_size = block_size;
    અચલ પૂર્ણાંક batch_i = bid / H;
    અચલ પૂર્ણાંક head_i = bid % H;
    અચલ પૂર્ણાંક state_size = C * head_size;
    અચલ પૂર્ણાંક n_seq_tokens = T / B;

    દશાંશ* _r = shared_mem;
    દશાંશ* _w = _r + head_size;
    દશાંશ* _k = _w + head_size;
    દશાંશ* _a = _k + head_size;
    દશાંશ* _b = _a + head_size;

    દશાંશ state[block_size];

    #pragma unroll
    માટે (પૂર્ણાંક i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
    }

    માટે (પૂર્ણાંક t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
         t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        અચલ દશાંશ _v = v[t];
        દશાંશ y = 0, sa = 0;
        sycl::float4 a4, s4;

        #pragma unroll
        માટે (પૂર્ણાંક j = 0; j < head_size; j += 4) {
            a4 = sycl::float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa += sycl::dot(a4, s4);
        }

        sycl::float4 r4, w4, k4, b4;
        #pragma unroll
        માટે (પૂર્ણાંક j = 0; j < head_size; j += 4) {
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
    માટે (પૂર્ણાંક i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
    }
}

શૂન્ય ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    અચલ દશાંશ* k_d = (અચલ દશાંશ*)dst->src[0]->data;
    અચલ દશાંશ* v_d = (અચલ દશાંશ*)dst->src[1]->data;
    અચલ દશાંશ* r_d = (અચલ દશાંશ*)dst->src[2]->data;
    અચલ દશાંશ* tf_d = (અચલ દશાંશ*)dst->src[3]->data;
    અચલ દશાંશ* td_d = (અચલ દશાંશ*)dst->src[4]->data;
    અચલ દશાંશ* s_d = (અચલ દશાંશ*)dst->src[5]->data;
    દશાંશ* dst_d = (દશાંશ*)dst->data;

    અચલ int64_t B = dst->src[5]->ne[1];
    અચલ int64_t T = dst->src[0]->ne[2];
    અચલ int64_t C = dst->ne[0];
    અચલ int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_SIZE || C / H == WKV_BLOCK_SIZE * 2); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    અચલ size_t shared_mem_size = C / H * 4 * કદ(દશાંશ); // For k, r, tf, td
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    જો (C / H == WKV_BLOCK_SIZE) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<દશાંશ, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_SIZE>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (દશાંશ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } નહીંતર {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<દશાંશ, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_SIZE * 2>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (દશાંશ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}

શૂન્ય ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    અચલ દશાંશ* r_d = (અચલ દશાંશ*)dst->src[0]->data;
    અચલ દશાંશ* w_d = (અચલ દશાંશ*)dst->src[1]->data;
    અચલ દશાંશ* k_d = (અચલ દશાંશ*)dst->src[2]->data;
    અચલ દશાંશ* v_d = (અચલ દશાંશ*)dst->src[3]->data;
    અચલ દશાંશ* a_d = (અચલ દશાંશ*)dst->src[4]->data;
    અચલ દશાંશ* b_d = (અચલ દશાંશ*)dst->src[5]->data;
    અચલ દશાંશ* s_d = (અચલ દશાંશ*)dst->src[6]->data;
    દશાંશ* dst_d = (દશાંશ*)dst->data;

    અચલ int64_t B = dst->src[6]->ne[1];
    અચલ int64_t T = dst->src[0]->ne[2];
    અચલ int64_t C = dst->ne[0];
    અચલ int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_SIZE || C / H == WKV_BLOCK_SIZE * 2);

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    અચલ size_t shared_mem_size = C / H * 5 * કદ(દશાંશ); // For r, w, k, a, b
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    જો (C / H == WKV_BLOCK_SIZE) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<દશાંશ, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_SIZE>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (દશાંશ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } નહીંતર {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<દશાંશ, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_SIZE * 2>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (દશાંશ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}
