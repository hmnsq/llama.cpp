#ضمن <sycl/sycl.hpp>
#ضمن "wkv.hpp"

constexpr صحيح WKV_بلاک_ناپ = 64;

// Helper function for the main kernel
سانچا <صحيح بلاک_ناپ_3>
ساكن فراغ rwkv_wkv6_f32_kernel(
    ثابت صحيح B, ثابت صحيح T, ثابت صحيح C, ثابت صحيح H,
    ثابت عائم* k, ثابت عائم* v, ثابت عائم* r,
    ثابت عائم* tf, ثابت عائم* td, ثابت عائم* s,
    عائم* dst, ثابت sycl::nd_شے<3>& شے_ct1, عائم* shared_mem) {

    ثابت صحيح tid = شے_ct1.get_local_id(2);
    ثابت صحيح bid = شے_ct1.get_گروہ(2);

    ثابت صحيح سر_ناپ = بلاک_ناپ_3;
    ثابت صحيح batch_i = bid / H;
    ثابت صحيح سر_i = bid % H;
    ثابت صحيح state_ناپ = C * سر_ناپ;
    ثابت صحيح n_seq_ٹوکن = T / B;

    // Set up shared memory pointers
    عائم* _k = shared_mem;
    عائم* _r = _k + سر_ناپ;
    عائم* _tf = _r + سر_ناپ;
    عائم* _td = _tf + سر_ناپ;

    // Local state array
    عائم state[بلاک_ناپ_3];

    // Load initial state
    #ہدایت unroll
    لكل (صحيح i = 0; i < سر_ناپ; i++) {
        state[i] = s[batch_i * state_ناپ + سر_i * سر_ناپ * سر_ناپ + i * سر_ناپ + tid];
    }

    // Sync threads before shared memory operations
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[سر_i * سر_ناپ + tid];
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    لكل (صحيح t = batch_i * n_seq_ٹوکن * C + سر_i * سر_ناپ + tid;
         t < (batch_i + 1) * n_seq_ٹوکن * C + سر_i * سر_ناپ + tid;
         t += C) {

        شے_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        شے_ct1.barrier(sycl::access::fence_space::local_space);

        ثابت عائم _v = v[t];
        عائم y = 0;

        // Process in chunks of 4 for better vectorization
        sycl::float4 k4, r4, tf4, td4, s4;
        #ہدایت unroll
        لكل (صحيح j = 0; j < سر_ناپ; j += 4) {
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
    #ہدایت unroll
    لكل (صحيح i = 0; i < سر_ناپ; i++) {
        dst[T * C + batch_i * state_ناپ + سر_i * سر_ناپ * سر_ناپ + i * سر_ناپ + tid] = state[i];
    }
}

سانچا <صحيح بلاک_ناپ_3>
ساكن فراغ rwkv_wkv7_f32_kernel(
    ثابت صحيح B, ثابت صحيح T, ثابت صحيح C, ثابت صحيح H,
    ثابت عائم* r, ثابت عائم* w, ثابت عائم* k, ثابت عائم* v,
    ثابت عائم* a, ثابت عائم* b, ثابت عائم* s,
    عائم* dst, ثابت sycl::nd_شے<3>& شے_ct1, عائم* shared_mem) {

    ثابت صحيح tid = شے_ct1.get_local_id(2);
    ثابت صحيح bid = شے_ct1.get_گروہ(2);

    ثابت صحيح سر_ناپ = بلاک_ناپ_3;
    ثابت صحيح batch_i = bid / H;
    ثابت صحيح سر_i = bid % H;
    ثابت صحيح state_ناپ = C * سر_ناپ;
    ثابت صحيح n_seq_ٹوکن = T / B;

    عائم* _r = shared_mem;
    عائم* _w = _r + سر_ناپ;
    عائم* _k = _w + سر_ناپ;
    عائم* _a = _k + سر_ناپ;
    عائم* _b = _a + سر_ناپ;

    عائم state[بلاک_ناپ_3];

    #ہدایت unroll
    لكل (صحيح i = 0; i < سر_ناپ; i++) {
        state[i] = s[batch_i * state_ناپ + سر_i * سر_ناپ * سر_ناپ + tid * سر_ناپ + i];
    }

    لكل (صحيح t = batch_i * n_seq_ٹوکن * C + سر_i * سر_ناپ + tid;
         t < (batch_i + 1) * n_seq_ٹوکن * C + سر_i * سر_ناپ + tid;
         t += C) {

        شے_ct1.barrier(sycl::access::fence_space::local_space);

        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];

        شے_ct1.barrier(sycl::access::fence_space::local_space);

        ثابت عائم _v = v[t];
        عائم y = 0, sa = 0;
        sycl::float4 a4, s4;

        #ہدایت unroll
        لكل (صحيح j = 0; j < سر_ناپ; j += 4) {
            a4 = sycl::float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa += sycl::dot(a4, s4);
        }

        sycl::float4 r4, w4, k4, b4;
        #ہدایت unroll
        لكل (صحيح j = 0; j < سر_ناپ; j += 4) {
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

    #ہدایت unroll
    لكل (صحيح i = 0; i < سر_ناپ; i++) {
        dst[T * C + batch_i * state_ناپ + سر_i * سر_ناپ * سر_ناپ + tid * سر_ناپ + i] = state[i];
    }
}

فراغ ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ثابت عائم* k_d = (ثابت عائم*)dst->src[0]->انکڑے;
    ثابت عائم* v_d = (ثابت عائم*)dst->src[1]->انکڑے;
    ثابت عائم* r_d = (ثابت عائم*)dst->src[2]->انکڑے;
    ثابت عائم* tf_d = (ثابت عائم*)dst->src[3]->انکڑے;
    ثابت عائم* td_d = (ثابت عائم*)dst->src[4]->انکڑے;
    ثابت عائم* s_d = (ثابت عائم*)dst->src[5]->انکڑے;
    عائم* dst_d = (عائم*)dst->انکڑے;

    ثابت int64_t B = dst->src[5]->ne[1];
    ثابت int64_t T = dst->src[0]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_بلاک_ناپ || C / H == WKV_بلاک_ناپ * 2); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::قطار_ptr stream = ctx.stream();

    // Calculate execution configuration
    ثابت size_t shared_mem_ناپ = C / H * 4 * حجم(عائم); // For k, r, tf, td
    sycl::range<3> بلاک_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    إذا (C / H == WKV_بلاک_ناپ) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> shared_mem_acc(shared_mem_ناپ, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_بلاک_ناپ>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        شے_ct1, (عائم*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } وإلا {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> shared_mem_acc(shared_mem_ناپ, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_بلاک_ناپ * 2>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        شے_ct1, (عائم*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}

فراغ ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ثابت عائم* r_d = (ثابت عائم*)dst->src[0]->انکڑے;
    ثابت عائم* w_d = (ثابت عائم*)dst->src[1]->انکڑے;
    ثابت عائم* k_d = (ثابت عائم*)dst->src[2]->انکڑے;
    ثابت عائم* v_d = (ثابت عائم*)dst->src[3]->انکڑے;
    ثابت عائم* a_d = (ثابت عائم*)dst->src[4]->انکڑے;
    ثابت عائم* b_d = (ثابت عائم*)dst->src[5]->انکڑے;
    ثابت عائم* s_d = (ثابت عائم*)dst->src[6]->انکڑے;
    عائم* dst_d = (عائم*)dst->انکڑے;

    ثابت int64_t B = dst->src[6]->ne[1];
    ثابت int64_t T = dst->src[0]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_بلاک_ناپ || C / H == WKV_بلاک_ناپ * 2);

    dpct::قطار_ptr stream = ctx.stream();

    // Calculate execution configuration
    ثابت size_t shared_mem_ناپ = C / H * 5 * حجم(عائم); // For r, w, k, a, b
    sycl::range<3> بلاک_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    إذا (C / H == WKV_بلاک_ناپ) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> shared_mem_acc(shared_mem_ناپ, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_بلاک_ناپ>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        شے_ct1, (عائم*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } وإلا {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<عائم, 1> shared_mem_acc(shared_mem_ناپ, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_بلاک_ناپ * 2>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        شے_ct1, (عائم*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}
