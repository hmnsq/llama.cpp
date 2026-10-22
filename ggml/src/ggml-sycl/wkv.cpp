#অন্তর্ভাব <sycl/sycl.hpp>
#অন্তর্ভাব "wkv.hpp"

constexpr পূর্ণসংখ্যা WKV_BLOCK_আকার = 64;

// Helper function for the main kernel
template <পূর্ণসংখ্যা block_আকার_2>
স্থির শূন্য rwkv_wkv6_f32_kernel(
    ধ্রুবক পূর্ণসংখ্যা B, ধ্রুবক পূর্ণসংখ্যা T, ধ্রুবক পূর্ণসংখ্যা C, ধ্রুবক পূর্ণসংখ্যা H,
    ধ্রুবক দশমিক* k, ধ্রুবক দশমিক* v, ধ্রুবক দশমিক* r,
    ধ্রুবক দশমিক* tf, ধ্রুবক দশমিক* td, ধ্রুবক দশমিক* s,
    দশমিক* dst, ধ্রুবক sycl::nd_item<3>& item_ct1, দশমিক* shared_mem) {

    ধ্রুবক পূর্ণসংখ্যা tid = item_ct1.get_local_id(2);
    ধ্রুবক পূর্ণসংখ্যা bid = item_ct1.get_group(2);

    ধ্রুবক পূর্ণসংখ্যা শীর্ষ_আকার = block_আকার_2;
    ধ্রুবক পূর্ণসংখ্যা batch_i = bid / H;
    ধ্রুবক পূর্ণসংখ্যা শীর্ষ_i = bid % H;
    ধ্রুবক পূর্ণসংখ্যা state_আকার = C * শীর্ষ_আকার;
    ধ্রুবক পূর্ণসংখ্যা n_seq_tokens = T / B;

    // Set up shared memory pointers
    দশমিক* _k = shared_mem;
    দশমিক* _r = _k + শীর্ষ_আকার;
    দশমিক* _tf = _r + শীর্ষ_আকার;
    দশমিক* _td = _tf + শীর্ষ_আকার;

    // Local state array
    দশমিক state[block_আকার_2];

    // Load initial state
    #pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < শীর্ষ_আকার; i++) {
        state[i] = s[batch_i * state_আকার + শীর্ষ_i * শীর্ষ_আকার * শীর্ষ_আকার + i * শীর্ষ_আকার + tid];
    }

    // Sync threads before shared memory operations
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[শীর্ষ_i * শীর্ষ_আকার + tid];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    জন্য (পূর্ণসংখ্যা t = batch_i * n_seq_tokens * C + শীর্ষ_i * শীর্ষ_আকার + tid;
         t < (batch_i + 1) * n_seq_tokens * C + শীর্ষ_i * শীর্ষ_আকার + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        ধ্রুবক দশমিক _v = v[t];
        দশমিক y = 0;

        // Process in chunks of 4 for better vectorization
        sycl::float4 k4, r4, tf4, td4, s4;
        #pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < শীর্ষ_আকার; j += 4) {
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
    জন্য (পূর্ণসংখ্যা i = 0; i < শীর্ষ_আকার; i++) {
        dst[T * C + batch_i * state_আকার + শীর্ষ_i * শীর্ষ_আকার * শীর্ষ_আকার + i * শীর্ষ_আকার + tid] = state[i];
    }
}

template <পূর্ণসংখ্যা block_আকার_2>
স্থির শূন্য rwkv_wkv7_f32_kernel(
    ধ্রুবক পূর্ণসংখ্যা B, ধ্রুবক পূর্ণসংখ্যা T, ধ্রুবক পূর্ণসংখ্যা C, ধ্রুবক পূর্ণসংখ্যা H,
    ধ্রুবক দশমিক* r, ধ্রুবক দশমিক* w, ধ্রুবক দশমিক* k, ধ্রুবক দশমিক* v,
    ধ্রুবক দশমিক* a, ধ্রুবক দশমিক* b, ধ্রুবক দশমিক* s,
    দশমিক* dst, ধ্রুবক sycl::nd_item<3>& item_ct1, দশমিক* shared_mem) {

    ধ্রুবক পূর্ণসংখ্যা tid = item_ct1.get_local_id(2);
    ধ্রুবক পূর্ণসংখ্যা bid = item_ct1.get_group(2);

    ধ্রুবক পূর্ণসংখ্যা শীর্ষ_আকার = block_আকার_2;
    ধ্রুবক পূর্ণসংখ্যা batch_i = bid / H;
    ধ্রুবক পূর্ণসংখ্যা শীর্ষ_i = bid % H;
    ধ্রুবক পূর্ণসংখ্যা state_আকার = C * শীর্ষ_আকার;
    ধ্রুবক পূর্ণসংখ্যা n_seq_tokens = T / B;

    দশমিক* _r = shared_mem;
    দশমিক* _w = _r + শীর্ষ_আকার;
    দশমিক* _k = _w + শীর্ষ_আকার;
    দশমিক* _a = _k + শীর্ষ_আকার;
    দশমিক* _b = _a + শীর্ষ_আকার;

    দশমিক state[block_আকার_2];

    #pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < শীর্ষ_আকার; i++) {
        state[i] = s[batch_i * state_আকার + শীর্ষ_i * শীর্ষ_আকার * শীর্ষ_আকার + tid * শীর্ষ_আকার + i];
    }

    জন্য (পূর্ণসংখ্যা t = batch_i * n_seq_tokens * C + শীর্ষ_i * শীর্ষ_আকার + tid;
         t < (batch_i + 1) * n_seq_tokens * C + শীর্ষ_i * শীর্ষ_আকার + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        ধ্রুবক দশমিক _v = v[t];
        দশমিক y = 0, sa = 0;
        sycl::float4 a4, s4;

        #pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < শীর্ষ_আকার; j += 4) {
            a4 = sycl::float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa += sycl::dot(a4, s4);
        }

        sycl::float4 r4, w4, k4, b4;
        #pragma unroll
        জন্য (পূর্ণসংখ্যা j = 0; j < শীর্ষ_আকার; j += 4) {
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
    জন্য (পূর্ণসংখ্যা i = 0; i < শীর্ষ_আকার; i++) {
        dst[T * C + batch_i * state_আকার + শীর্ষ_i * শীর্ষ_আকার * শীর্ষ_আকার + tid * শীর্ষ_আকার + i] = state[i];
    }
}

শূন্য ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ধ্রুবক দশমিক* k_d = (ধ্রুবক দশমিক*)dst->src[0]->তথ্য;
    ধ্রুবক দশমিক* v_d = (ধ্রুবক দশমিক*)dst->src[1]->তথ্য;
    ধ্রুবক দশমিক* r_d = (ধ্রুবক দশমিক*)dst->src[2]->তথ্য;
    ধ্রুবক দশমিক* tf_d = (ধ্রুবক দশমিক*)dst->src[3]->তথ্য;
    ধ্রুবক দশমিক* td_d = (ধ্রুবক দশমিক*)dst->src[4]->তথ্য;
    ধ্রুবক দশমিক* s_d = (ধ্রুবক দশমিক*)dst->src[5]->তথ্য;
    দশমিক* dst_d = (দশমিক*)dst->তথ্য;

    ধ্রুবক int64_t B = dst->src[5]->ne[1];
    ধ্রুবক int64_t T = dst->src[0]->ne[2];
    ধ্রুবক int64_t C = dst->ne[0];
    ধ্রুবক int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_আকার || C / H == WKV_BLOCK_আকার * 2); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    ধ্রুবক size_t shared_mem_আকার = C / H * 4 * আকার(দশমিক); // For k, r, tf, td
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    যদি (C / H == WKV_BLOCK_আকার) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<দশমিক, 1> shared_mem_acc(shared_mem_আকার, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_আকার>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (দশমিক*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } অন্যথা {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<দশমিক, 1> shared_mem_acc(shared_mem_আকার, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_আকার * 2>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (দশমিক*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}

শূন্য ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ধ্রুবক দশমিক* r_d = (ধ্রুবক দশমিক*)dst->src[0]->তথ্য;
    ধ্রুবক দশমিক* w_d = (ধ্রুবক দশমিক*)dst->src[1]->তথ্য;
    ধ্রুবক দশমিক* k_d = (ধ্রুবক দশমিক*)dst->src[2]->তথ্য;
    ধ্রুবক দশমিক* v_d = (ধ্রুবক দশমিক*)dst->src[3]->তথ্য;
    ধ্রুবক দশমিক* a_d = (ধ্রুবক দশমিক*)dst->src[4]->তথ্য;
    ধ্রুবক দশমিক* b_d = (ধ্রুবক দশমিক*)dst->src[5]->তথ্য;
    ধ্রুবক দশমিক* s_d = (ধ্রুবক দশমিক*)dst->src[6]->তথ্য;
    দশমিক* dst_d = (দশমিক*)dst->তথ্য;

    ধ্রুবক int64_t B = dst->src[6]->ne[1];
    ধ্রুবক int64_t T = dst->src[0]->ne[2];
    ধ্রুবক int64_t C = dst->ne[0];
    ধ্রুবক int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_আকার || C / H == WKV_BLOCK_আকার * 2);

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    ধ্রুবক size_t shared_mem_আকার = C / H * 5 * আকার(দশমিক); // For r, w, k, a, b
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    যদি (C / H == WKV_BLOCK_আকার) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<দশমিক, 1> shared_mem_acc(shared_mem_আকার, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_আকার>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (দশমিক*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } অন্যথা {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<দশমিক, 1> shared_mem_acc(shared_mem_আকার, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_আকার * 2>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (দশমিক*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}
