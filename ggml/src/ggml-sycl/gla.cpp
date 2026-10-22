#شامل <sycl/sycl.hpp>

#شامل "common.hpp"
#شامل "gla.hpp"

template <u_int HEAD_SIZE>
ایستا تهی gated_linear_attn_f32_kernel(ثابت dpct::queue_ptr stream, u_int B, u_int T, u_int C, u_int H, اعشاری scale,
                                         ثابت اعشاری * k, ثابت اعشاری * v, ثابت اعشاری * r, ثابت اعشاری * td,
                                         ثابت اعشاری * s, اعشاری * dst) {
    ثابت u_int head_size    = HEAD_SIZE;
    ثابت u_int state_size   = C * head_size;
    ثابت u_int n_seq_tokens = T / B;
    sycl::range<1> block_dims((C / H));
    sycl::range<1> grid_dims((B * H));
    stream->submit([&](sycl::handler & cgh) {
        /* local memory accessors*/
        auto _k  = sycl::local_accessor<اعشاری, 1>(sycl::range<1>(head_size), cgh);
        auto _r  = sycl::local_accessor<اعشاری, 1>(sycl::range<1>(head_size), cgh);
        auto _td = sycl::local_accessor<اعشاری, 1>(sycl::range<1>(head_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(grid_dims * block_dims, block_dims), [=](sycl::nd_item<1> item) {
            u_int tid = item.get_local_id(0);
            u_int bid = item.get_group(0);

            u_int batch_i = bid / H;
            u_int head_i  = bid % H;

            اعشاری state[head_size];

#pragma unroll
            برای (u_int i = 0; i < head_size; i++) {
                state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
            }

            برای (u_int t = batch_i * n_seq_tokens * C + head_i * head_size + tid;
                 t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {

                item.barrier(sycl::access::fence_space::local_space);  //sync threads
                _k[tid]  = k[t];
                _r[tid]  = r[t];
                _td[tid] = td[t];
                item.barrier(sycl::access::fence_space::local_space);  //sync threads

                ثابت اعشاری _v = v[t];
                اعشاری       y  = 0;

                برای (u_int j = 0; j < head_size; j += 4) {
                    ثابت sycl::float4 & k  = (sycl::float4 &) (_k[j]);
                    ثابت sycl::float4 & r  = (sycl::float4 &) (_r[j]);
                    ثابت sycl::float4 & td = (sycl::float4 &) (_td[j]);
                    sycl::float4 &       s  = (sycl::float4 &) (state[j]);
                    sycl::float4         kv;

                    kv.x() = k.x() * _v;
                    kv.y() = k.y() * _v;
                    kv.z() = k.z() * _v;
                    kv.w() = k.w() * _v;

                    s.x() = s.x() * td.x() + kv.x();
                    s.y() = s.y() * td.y() + kv.y();
                    s.z() = s.z() * td.z() + kv.z();
                    s.w() = s.w() * td.w() + kv.w();

                    y += r.x() * s.x();
                    y += r.y() * s.y();
                    y += r.z() * s.z();
                    y += r.w() * s.w();
                }
                dst[t] = y * scale;
            }
#pragma unroll
            برای (u_int i = 0; i < head_size; i++) {
                dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
            }
        });
    });
}

تهی ggml_sycl_op_gated_linear_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);
    ثابت اعشاری * k_d  = static_cast<ثابت اعشاری *>(dst->src[0]->data);
    ثابت اعشاری * v_d  = static_cast<ثابت اعشاری *>(dst->src[1]->data);
    ثابت اعشاری * r_d  = static_cast<ثابت اعشاری *>(dst->src[2]->data);
    ثابت اعشاری * td_d = static_cast<ثابت اعشاری *>(dst->src[3]->data);
    ثابت اعشاری * s_d  = static_cast<ثابت اعشاری *>(dst->src[4]->data);

    ثابت int64_t B = dst->src[4]->ne[1];
    ثابت int64_t T = dst->src[0]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t H = dst->src[0]->ne[1];

    dpct::queue_ptr stream = ctx.stream();
    GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == 64 || C / H == 128);

    اعشاری scale;
    memcpy(&scale, dst->op_params, اندازه(اعشاری));

    اعشاری * dst_d = (اعشاری *) dst->data;

    اگر (C / H == 64) {
        gated_linear_attn_f32_kernel<64>(stream, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    } وگرنه {
        gated_linear_attn_f32_kernel<128>(stream, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    }
}
