#شامل <sycl/sycl.hpp>

#شامل "common.hpp"
#شامل "gla.hpp"

سانچہ <u_int سر_سائز_2>
ساکن خالی gated_خطی_attn_f32_kernel(ثابت dpct::قطار_ptr stream, u_int B, u_int T, u_int C, u_int H, اعشاری scale,
                                         ثابت اعشاری * k, ثابت اعشاری * v, ثابت اعشاری * r, ثابت اعشاری * td,
                                         ثابت اعشاری * s, اعشاری * dst) {
    ثابت u_int سر_سائز    = سر_سائز_2;
    ثابت u_int state_سائز   = C * سر_سائز;
    ثابت u_int n_seq_علامات = T / B;
    sycl::range<1> بلاک_dims((C / H));
    sycl::range<1> grid_dims((B * H));
    stream->submit([&](sycl::handler & cgh) {
        /* local memory accessors*/
        خودکار _k  = sycl::local_accessor<اعشاری, 1>(sycl::range<1>(سر_سائز), cgh);
        خودکار _r  = sycl::local_accessor<اعشاری, 1>(sycl::range<1>(سر_سائز), cgh);
        خودکار _td = sycl::local_accessor<اعشاری, 1>(sycl::range<1>(سر_سائز), cgh);

        cgh.parallel_for(sycl::nd_range<1>(grid_dims * بلاک_dims, بلاک_dims), [=](sycl::nd_شے<1> شے) {
            u_int tid = شے.get_local_id(0);
            u_int bid = شے.get_گروہ(0);

            u_int batch_i = bid / H;
            u_int سر_i  = bid % H;

            اعشاری state[سر_سائز];

#ہدایت unroll
            برائے (u_int i = 0; i < سر_سائز; i++) {
                state[i] = s[batch_i * state_سائز + سر_i * سر_سائز * سر_سائز + i * سر_سائز + tid];
            }

            برائے (u_int t = batch_i * n_seq_علامات * C + سر_i * سر_سائز + tid;
                 t < (batch_i + 1) * n_seq_علامات * C + سر_i * سر_سائز + tid; t += C) {

                شے.barrier(sycl::access::fence_space::local_space);  //sync threads
                _k[tid]  = k[t];
                _r[tid]  = r[t];
                _td[tid] = td[t];
                شے.barrier(sycl::access::fence_space::local_space);  //sync threads

                ثابت اعشاری _v = v[t];
                اعشاری       y  = 0;

                برائے (u_int j = 0; j < سر_سائز; j += 4) {
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
#ہدایت unroll
            برائے (u_int i = 0; i < سر_سائز; i++) {
                dst[T * C + batch_i * state_سائز + سر_i * سر_سائز * سر_سائز + i * سر_سائز + tid] = state[i];
            }
        });
    });
}

خالی ggml_sycl_op_gated_خطی_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);
    ثابت اعشاری * k_d  = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    ثابت اعشاری * v_d  = static_cast<ثابت اعشاری *>(dst->src[1]->ڈیٹا);
    ثابت اعشاری * r_d  = static_cast<ثابت اعشاری *>(dst->src[2]->ڈیٹا);
    ثابت اعشاری * td_d = static_cast<ثابت اعشاری *>(dst->src[3]->ڈیٹا);
    ثابت اعشاری * s_d  = static_cast<ثابت اعشاری *>(dst->src[4]->ڈیٹا);

    ثابت int64_t B = dst->src[4]->ne[1];
    ثابت int64_t T = dst->src[0]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t H = dst->src[0]->ne[1];

    dpct::قطار_ptr stream = ctx.stream();
    GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == 64 || C / H == 128);

    اعشاری scale;
    حافظہ_نقل(&scale, dst->op_params, حجم(اعشاری));

    اعشاری * dst_d = (اعشاری *) dst->ڈیٹا;

    اگر (C / H == 64) {
        gated_خطی_attn_f32_kernel<64>(stream, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    } ورنہ {
        gated_خطی_attn_f32_kernel<128>(stream, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    }
}
