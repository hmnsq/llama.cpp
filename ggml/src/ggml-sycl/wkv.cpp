#شامل <sycl/sycl.hpp>
#شامل "wkv.hpp"

constexpr صحیح WKV_BLOCK_SIZE = 64;

// Helper function for the main kernel
template <صحیح block_size>
ایستا تهی rwkv_wkv6_f32_kernel(
    ثابت صحیح B, ثابت صحیح T, ثابت صحیح C, ثابت صحیح H,
    ثابت اعشاری* k, ثابت اعشاری* v, ثابت اعشاری* r,
    ثابت اعشاری* tf, ثابت اعشاری* td, ثابت اعشاری* s,
    اعشاری* dst, ثابت sycl::nd_item<3>& item_ct1, اعشاری* shared_mem) {

    ثابت صحیح tid = item_ct1.get_local_id(2);
    ثابت صحیح bid = item_ct1.get_group(2);

    ثابت صحیح head_size = block_size;
    ثابت صحیح batch_i = bid / H;
    ثابت صحیح head_i = bid % H;
    ثابت صحیح state_size = C * head_size;
    ثابت صحیح n_seq_tokens = T / B;

    // Set up shared memory pointers
    اعشاری* _k = shared_mem;
    اعشاری* _r = _k + head_size;
    اعشاری* _tf = _r + head_size;
    اعشاری* _td = _tf + head_size;

    // Local state array
    اعشاری state[block_size];

    // Load initial state
    #pragma unroll
    برای (صحیح i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    // Sync threads before shared memory operations
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[head_i * head_size + tid];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    برای (صحیح t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
         t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        ثابت اعشاری _v = v[t];
        اعشاری y = 0;

        // Process in chunks of 4 for better vectorization
        sycl::float4 k4, r4, tf4, td4, s4;
        #pragma unroll
        برای (صحیح j = 0; j < head_size; j += 4) {
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
    برای (صحیح i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

template <صحیح block_size>
ایستا تهی rwkv_wkv7_f32_kernel(
    ثابت صحیح B, ثابت صحیح T, ثابت صحیح C, ثابت صحیح H,
    ثابت اعشاری* r, ثابت اعشاری* w, ثابت اعشاری* k, ثابت اعشاری* v,
    ثابت اعشاری* a, ثابت اعشاری* b, ثابت اعشاری* s,
    اعشاری* dst, ثابت sycl::nd_item<3>& item_ct1, اعشاری* shared_mem) {

    ثابت صحیح tid = item_ct1.get_local_id(2);
    ثابت صحیح bid = item_ct1.get_group(2);

    ثابت صحیح head_size = block_size;
    ثابت صحیح batch_i = bid / H;
    ثابت صحیح head_i = bid % H;
    ثابت صحیح state_size = C * head_size;
    ثابت صحیح n_seq_tokens = T / B;

    اعشاری* _r = shared_mem;
    اعشاری* _w = _r + head_size;
    اعشاری* _k = _w + head_size;
    اعشاری* _a = _k + head_size;
    اعشاری* _b = _a + head_size;

    اعشاری state[block_size];

    #pragma unroll
    برای (صحیح i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
    }

    برای (صحیح t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
         t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        ثابت اعشاری _v = v[t];
        اعشاری y = 0, sa = 0;
        sycl::float4 a4, s4;

        #pragma unroll
        برای (صحیح j = 0; j < head_size; j += 4) {
            a4 = sycl::float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa += sycl::dot(a4, s4);
        }

        sycl::float4 r4, w4, k4, b4;
        #pragma unroll
        برای (صحیح j = 0; j < head_size; j += 4) {
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
    برای (صحیح i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
    }
}

تهی ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ثابت اعشاری* k_d = (ثابت اعشاری*)dst->src[0]->data;
    ثابت اعشاری* v_d = (ثابت اعشاری*)dst->src[1]->data;
    ثابت اعشاری* r_d = (ثابت اعشاری*)dst->src[2]->data;
    ثابت اعشاری* tf_d = (ثابت اعشاری*)dst->src[3]->data;
    ثابت اعشاری* td_d = (ثابت اعشاری*)dst->src[4]->data;
    ثابت اعشاری* s_d = (ثابت اعشاری*)dst->src[5]->data;
    اعشاری* dst_d = (اعشاری*)dst->data;

    ثابت int64_t B = dst->src[5]->ne[1];
    ثابت int64_t T = dst->src[0]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_SIZE || C / H == WKV_BLOCK_SIZE * 2); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    ثابت size_t shared_mem_size = C / H * 4 * اندازه(اعشاری); // For k, r, tf, td
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    اگر (C / H == WKV_BLOCK_SIZE) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_SIZE>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (اعشاری*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } وگرنه {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_SIZE * 2>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (اعشاری*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}

تهی ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ثابت اعشاری* r_d = (ثابت اعشاری*)dst->src[0]->data;
    ثابت اعشاری* w_d = (ثابت اعشاری*)dst->src[1]->data;
    ثابت اعشاری* k_d = (ثابت اعشاری*)dst->src[2]->data;
    ثابت اعشاری* v_d = (ثابت اعشاری*)dst->src[3]->data;
    ثابت اعشاری* a_d = (ثابت اعشاری*)dst->src[4]->data;
    ثابت اعشاری* b_d = (ثابت اعشاری*)dst->src[5]->data;
    ثابت اعشاری* s_d = (ثابت اعشاری*)dst->src[6]->data;
    اعشاری* dst_d = (اعشاری*)dst->data;

    ثابت int64_t B = dst->src[6]->ne[1];
    ثابت int64_t T = dst->src[0]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_SIZE || C / H == WKV_BLOCK_SIZE * 2);

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    ثابت size_t shared_mem_size = C / H * 5 * اندازه(اعشاری); // For r, w, k, a, b
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    اگر (C / H == WKV_BLOCK_SIZE) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_SIZE>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (اعشاری*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } وگرنه {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<اعشاری, 1> shared_mem_acc(shared_mem_size, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_SIZE * 2>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (اعشاری*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}
