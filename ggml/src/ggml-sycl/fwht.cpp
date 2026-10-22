#شامل "fwht.hpp"

#شامل <cmath>
#تعریف_کرو P 1.0f
#تعریف_کرو N -1.0f

// constant Hadamard matrix via Paley I construction
ساکن constexpr اعشاری H12[12][12] = {
    { P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, P, P, P, N, N, N, P, N },
    { P, N, N, P, N, P, P, P, N, N, N, P },
    { P, P, N, N, P, N, P, P, P, N, N, N },
    { P, N, P, N, N, P, N, P, P, P, N, N },
    { P, N, N, P, N, N, P, N, P, P, P, N },
    { P, N, N, N, P, N, N, P, N, P, P, P },
    { P, P, N, N, N, P, N, N, P, N, P, P },
    { P, P, P, N, N, N, P, N, N, P, N, P },
    { P, P, P, P, N, N, N, P, N, N, P, N },
    { P, N, P, P, P, N, N, N, P, N, N, P },
    { P, P, N, P, P, P, N, N, N, P, N, N }
};

ساکن constexpr اعشاری H20[20][20] = {
    { P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N },
    { P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P },
    { P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P },
    { P, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N },
    { P, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N },
    { P, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N },
    { P, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N },
    { P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P },
    { P, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N },
    { P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P },
    { P, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N },
    { P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P },
    { P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P },
    { P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P },
    { P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P },
    { P, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N },
    { P, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N },
    { P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P },
    { P, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N }
};

#تعریف_منسوخ P
#تعریف_منسوخ N

سانچہ <صحیح N, قسم_نام T>
ساکن خالی fwht_kernel(ثابت T * __restrict__ src, اعشاری * __restrict__ dst, ثابت int64_t n_صفیں,
                        ثابت اعشاری scale, ثابت sycl::nd_شے<2> & شے) {
    ثابت sycl::sub_گروہ sg = شے.get_sub_گروہ();

    ثابت int64_t r = شے.get_global_id(0);
    اگر (r >= n_صفیں) {
        واپس;
    }

    src += r * N;
    dst += r * N;

    constexpr صحیح el_w = N / WARP_سائز;
    static_assert(el_w >= 1 && N % WARP_سائز == 0, "row must be a whole number of sub-group widths");

    اعشاری     reg[el_w];
    ثابت صحیح lane = sg.get_local_خطی_id();

#ہدایت unroll
    برائے (صحیح i = 0; i < el_w; ++i) {
        reg[i] = static_cast<اعشاری>(src[i * WARP_سائز + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#ہدایت unroll
    برائے (صحیح h = 1; h < WARP_سائز; h *= 2) {
#ہدایت unroll
        برائے (صحیح j = 0; j < el_w; ++j) {
            ثابت اعشاری val  = reg[j];
            ثابت اعشاری val2 = dpct::ترتیب_بدلو_sub_گروہ_by_xor(sg, val, h, WARP_سائز);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#ہدایت unroll
    برائے (صحیح h = WARP_سائز; h < N; h *= 2) {
        ثابت صحیح قدم = h / WARP_سائز;
#ہدایت unroll
        برائے (صحیح j = 0; j < el_w; j += 2 * قدم) {
#ہدایت unroll
            برائے (صحیح k = 0; k < قدم; ++k) {
                ثابت اعشاری x = reg[j + k];
                ثابت اعشاری y = reg[j + k + قدم];

                reg[j + k]        = x + y;
                reg[j + k + قدم] = x - y;
            }
        }
    }

#ہدایت unroll
    برائے (صحیح i = 0; i < el_w; ++i) {
        dst[i * WARP_سائز + lane] = reg[i];
    }
}

سانچہ <صحیح N, قسم_نام T>
ساکن خالی launch_fwht(ثابت T * src, اعشاری * dst, ثابت int64_t n_صفیں, ثابت اعشاری scale,
                        dpct::قطار_ptr stream) {
    constexpr صحیح صفیں_فی_بلاک = 4;

    ثابت int64_t عدد_بلاکیں = (n_صفیں + صفیں_فی_بلاک - 1) / صفیں_فی_بلاک;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ثابت sycl::range<2> global(عدد_بلاکیں * صفیں_فی_بلاک, WARP_سائز);
    ثابت sycl::range<2> local(صفیں_فی_بلاک, WARP_سائز);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_شے<2> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             fwht_kernel<N, T>(src, dst, n_صفیں, scale, شے);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
سانچہ <صحیح N, صحیح NT, قسم_نام T>
ساکن خالی fwht_kernel_wide(ثابت T * __restrict__ src,
                             اعشاری * __restrict__ dst,
                             ثابت int64_t            n_صفیں,
                             ثابت اعشاری              scale,
                             ثابت sycl::nd_شے<2> & شے,
                             اعشاری *                  smem) {
    ثابت int64_t r = شے.get_global_id(0);
    اگر (r >= n_صفیں) {
        واپس;
    }

    src += r * N;
    dst += r * N;

    constexpr صحیح el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    ثابت صحیح tid = شے.get_local_id(1);

    اعشاری reg[el_w];
#ہدایت unroll
    برائے (صحیح i = 0; i < el_w; ++i) {
        reg[i] = static_cast<اعشاری>(src[i * NT + tid]) * scale;
    }

    ثابت sycl::sub_گروہ sg   = شے.get_sub_گروہ();
    ثابت صحیح             lane = sg.get_local_خطی_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#ہدایت unroll
    برائے (صحیح h = 1; h < WARP_سائز; h *= 2) {
#ہدایت unroll
        برائے (صحیح j = 0; j < el_w; ++j) {
            ثابت اعشاری val  = reg[j];
            ثابت اعشاری val2 = dpct::ترتیب_بدلو_sub_گروہ_by_xor(sg, val, h, WARP_سائز);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    برائے (صحیح h = WARP_سائز; h < NT; h *= 2) {
#ہدایت unroll
        برائے (صحیح j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        شے.barrier(sycl::access::fence_space::local_space);
#ہدایت unroll
        برائے (صحیح j = 0; j < el_w; ++j) {
            ثابت اعشاری val  = reg[j];
            ثابت اعشاری val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        شے.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    برائے (صحیح h = NT; h < N; h *= 2) {
        ثابت صحیح قدم = h / NT;
        برائے (صحیح j = 0; j < el_w; j += 2 * قدم) {
            برائے (صحیح k = 0; k < قدم; ++k) {
                ثابت اعشاری x = reg[j + k];
                ثابت اعشاری y = reg[j + k + قدم];

                reg[j + k]        = x + y;
                reg[j + k + قدم] = x - y;
            }
        }
    }

#ہدایت unroll
    برائے (صحیح i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

سانچہ <صحیح N, صحیح NT, قسم_نام T>
ساکن خالی launch_fwht_wide(ثابت T *       src,
                             اعشاری *         dst,
                             ثابت int64_t   n_صفیں,
                             ثابت اعشاری     scale,
                             dpct::قطار_ptr stream) {
    ثابت sycl::range<2> global(n_صفیں, NT);
    ثابت sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<اعشاری, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_شے<2> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_صفیں, scale, شے, get_اشارہ_گر(smem));
                         });
    });
}

سانچہ <صحیح N, صحیح m, قسم_نام T>
ساکن خالی kronecker_kernel(ثابت T * __restrict__ src,
                             اعشاری * __restrict__ dst,
                             ثابت int64_t            n_صفیں,
                             ثابت اعشاری              scale,
                             ثابت sycl::nd_شے<2> & شے) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    ثابت sycl::sub_گروہ sg = شے.get_sub_گروہ();

    ثابت int64_t r = شے.get_global_id(0);
    اگر (r >= n_صفیں) {
        واپس;
    }

    src += r * N;
    dst += r * N;

    constexpr صحیح بلاکیں_فی_گروہ = N / m;
    constexpr صحیح el_w             = بلاکیں_فی_گروہ / WARP_سائز;
    static_assert(el_w >= 1 && بلاکیں_فی_گروہ % WARP_سائز == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    اعشاری     reg[el_w * m];
    ثابت صحیح lane = sg.get_local_خطی_id();

#ہدایت unroll
    برائے (صحیح i = 0; i < el_w; ++i) {
        ثابت صحیح b_idx = i * WARP_سائز + lane;

#ہدایت unroll
        برائے (صحیح j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<اعشاری>(src[b_idx * m + j]) * scale;
        }
    }

#ہدایت unroll
    برائے (صحیح b = 0; b < el_w; ++b) {
        اعشاری z[m] = { 0.0f };

#ہدایت unroll
        برائے (صحیح i = 0; i < m; ++i) {
#ہدایت unroll
            برائے (صحیح j = 0; j < m; ++j) {
                ثابت اعشاری h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#ہدایت unroll
        برائے (صحیح i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#ہدایت unroll
    برائے (صحیح h = 1; h < WARP_سائز; h *= 2) {
#ہدایت unroll
        برائے (صحیح j = 0; j < el_w; ++j) {
#ہدایت unroll
            برائے (صحیح k = 0; k < m; ++k) {
                ثابت اعشاری val  = reg[j * m + k];
                ثابت اعشاری val2 = dpct::ترتیب_بدلو_sub_گروہ_by_xor(sg, val, h, WARP_سائز);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#ہدایت unroll
    برائے (صحیح h = WARP_سائز; h < بلاکیں_فی_گروہ; h *= 2) {
        ثابت صحیح قدم = h / WARP_سائز;
#ہدایت unroll
        برائے (صحیح j = 0; j < el_w; j += 2 * قدم) {
#ہدایت unroll
            برائے (صحیح s = 0; s < قدم; ++s) {
#ہدایت unroll
                برائے (صحیح k = 0; k < m; ++k) {
                    ثابت اعشاری x = reg[(j + s) * m + k];
                    ثابت اعشاری y = reg[(j + s + قدم) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + قدم) * m + k] = x - y;
                }
            }
        }
    }

#ہدایت unroll
    برائے (صحیح i = 0; i < el_w; ++i) {
        ثابت صحیح b_idx = i * WARP_سائز + lane;
#ہدایت unroll
        برائے (صحیح k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

سانچہ <صحیح N, صحیح m, قسم_نام T>
ساکن خالی launch_kronecker(ثابت T *       src,
                             اعشاری *         dst,
                             ثابت int64_t   n_صفیں,
                             ثابت اعشاری     scale,
                             dpct::قطار_ptr stream) {
    constexpr صحیح صفیں_فی_بلاک = 4;

    ثابت int64_t عدد_بلاکیں = (n_صفیں + صفیں_فی_بلاک - 1) / صفیں_فی_بلاک;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ثابت sycl::range<2> global(عدد_بلاکیں * صفیں_فی_بلاک, WARP_سائز);
    ثابت sycl::range<2> local(صفیں_فی_بلاک, WARP_سائز);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_شے<2> شے) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_صفیں, scale, شے);
                         });
}

سانچہ <قسم_نام T>
ساکن منطقی ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ثابت صحیح     n    = (صحیح) src->ne[0];
    ثابت int64_t صفیں = ggml_nrows(src);

    ثابت T *       src_d  = (ثابت T *) src->ڈیٹا;
    اعشاری *         dst_d  = (اعشاری *) dst->ڈیٹا;
    dpct::قطار_ptr stream = ctx.stream();

    ثابت اعشاری scale = 1.0f / std::جذر((اعشاری) n);

    انتخاب (n) {
        حالت 64:
            launch_fwht<64>(src_d, dst_d, صفیں, scale, stream);
            واپس سچ;
        حالت 128:
            launch_fwht<128>(src_d, dst_d, صفیں, scale, stream);
            واپس سچ;
        حالت 256:
            launch_fwht<256>(src_d, dst_d, صفیں, scale, stream);
            واپس سچ;
        حالت 512:
            launch_fwht<512>(src_d, dst_d, صفیں, scale, stream);
            واپس سچ;
        حالت 384:
            launch_kronecker<384, 12>(src_d, dst_d, صفیں, scale, stream);
            واپس سچ;
        حالت 768:
            launch_kronecker<768, 12>(src_d, dst_d, صفیں, scale, stream);
            واپس سچ;
        حالت 640:
            launch_kronecker<640, 20>(src_d, dst_d, صفیں, scale, stream);
            واپس سچ;
        حالت 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, صفیں, scale, stream);
            واپس سچ;
        حالت 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, صفیں, scale, stream);
            واپس سچ;
        حالت 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, صفیں, scale, stream);
            واپس سچ;
        حالت 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, صفیں, scale, stream);
            واپس سچ;
        حالت 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, صفیں, scale, stream);
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

منطقی ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src, ggml_tensor * dst) {
    اگر (dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }
    اگر (!ggml_are_same_shape(src, dst)) {
        واپس جھوٹ;
    }
    اگر (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }

    انتخاب (src->type) {
        حالت GGML_TYPE_F32:
            واپس ggml_sycl_op_fwht_impl<اعشاری>(ctx, src, dst);
        حالت GGML_TYPE_F16:
            واپس ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        طےشدہ:
            واپس جھوٹ;
    }
}
