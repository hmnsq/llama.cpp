#ضمن "fwht.hpp"

#ضمن <cmath>
#تعریف_کرو P 1.0f
#تعریف_کرو N -1.0f

// constant Hadamard matrix via Paley I construction
ساكن constexpr عائم H12[12][12] = {
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

ساكن constexpr عائم H20[20][20] = {
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

#تعریف_رد P
#تعریف_رد N

سانچا <صحيح N, قسم_ناں T>
ساكن فراغ fwht_kernel(ثابت T * __restrict__ src, عائم * __restrict__ dst, ثابت int64_t n_صفاں,
                        ثابت عائم scale, ثابت sycl::nd_شے<2> & شے) {
    ثابت sycl::sub_گروہ sg = شے.get_sub_گروہ();

    ثابت int64_t r = شے.get_global_id(0);
    إذا (r >= n_صفاں) {
        أرجع;
    }

    src += r * N;
    dst += r * N;

    constexpr صحيح el_w = N / WARP_ناپ;
    static_assert(el_w >= 1 && N % WARP_ناپ == 0, "row must be a whole number of sub-group widths");

    عائم     reg[el_w];
    ثابت صحيح lane = sg.get_local_خطی_id();

#ہدایت unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        reg[i] = static_cast<عائم>(src[i * WARP_ناپ + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#ہدایت unroll
    لكل (صحيح h = 1; h < WARP_ناپ; h *= 2) {
#ہدایت unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
            ثابت عائم val  = reg[j];
            ثابت عائم val2 = dpct::ترتیب_بدلو_sub_گروہ_by_xor(sg, val, h, WARP_ناپ);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#ہدایت unroll
    لكل (صحيح h = WARP_ناپ; h < N; h *= 2) {
        ثابت صحيح قدم = h / WARP_ناپ;
#ہدایت unroll
        لكل (صحيح j = 0; j < el_w; j += 2 * قدم) {
#ہدایت unroll
            لكل (صحيح k = 0; k < قدم; ++k) {
                ثابت عائم x = reg[j + k];
                ثابت عائم y = reg[j + k + قدم];

                reg[j + k]        = x + y;
                reg[j + k + قدم] = x - y;
            }
        }
    }

#ہدایت unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        dst[i * WARP_ناپ + lane] = reg[i];
    }
}

سانچا <صحيح N, قسم_ناں T>
ساكن فراغ launch_fwht(ثابت T * src, عائم * dst, ثابت int64_t n_صفاں, ثابت عائم scale,
                        dpct::قطار_ptr stream) {
    constexpr صحيح صفاں_فی_بلاک = 4;

    ثابت int64_t عدد_بلاکاں = (n_صفاں + صفاں_فی_بلاک - 1) / صفاں_فی_بلاک;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ثابت sycl::range<2> global(عدد_بلاکاں * صفاں_فی_بلاک, WARP_ناپ);
    ثابت sycl::range<2> local(صفاں_فی_بلاک, WARP_ناپ);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_شے<2> شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             fwht_kernel<N, T>(src, dst, n_صفاں, scale, شے);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
سانچا <صحيح N, صحيح NT, قسم_ناں T>
ساكن فراغ fwht_kernel_wide(ثابت T * __restrict__ src,
                             عائم * __restrict__ dst,
                             ثابت int64_t            n_صفاں,
                             ثابت عائم              scale,
                             ثابت sycl::nd_شے<2> & شے,
                             عائم *                  smem) {
    ثابت int64_t r = شے.get_global_id(0);
    إذا (r >= n_صفاں) {
        أرجع;
    }

    src += r * N;
    dst += r * N;

    constexpr صحيح el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    ثابت صحيح tid = شے.get_local_id(1);

    عائم reg[el_w];
#ہدایت unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        reg[i] = static_cast<عائم>(src[i * NT + tid]) * scale;
    }

    ثابت sycl::sub_گروہ sg   = شے.get_sub_گروہ();
    ثابت صحيح             lane = sg.get_local_خطی_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#ہدایت unroll
    لكل (صحيح h = 1; h < WARP_ناپ; h *= 2) {
#ہدایت unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
            ثابت عائم val  = reg[j];
            ثابت عائم val2 = dpct::ترتیب_بدلو_sub_گروہ_by_xor(sg, val, h, WARP_ناپ);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    لكل (صحيح h = WARP_ناپ; h < NT; h *= 2) {
#ہدایت unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        شے.barrier(sycl::access::fence_space::local_space);
#ہدایت unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
            ثابت عائم val  = reg[j];
            ثابت عائم val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        شے.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    لكل (صحيح h = NT; h < N; h *= 2) {
        ثابت صحيح قدم = h / NT;
        لكل (صحيح j = 0; j < el_w; j += 2 * قدم) {
            لكل (صحيح k = 0; k < قدم; ++k) {
                ثابت عائم x = reg[j + k];
                ثابت عائم y = reg[j + k + قدم];

                reg[j + k]        = x + y;
                reg[j + k + قدم] = x - y;
            }
        }
    }

#ہدایت unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

سانچا <صحيح N, صحيح NT, قسم_ناں T>
ساكن فراغ launch_fwht_wide(ثابت T *       src,
                             عائم *         dst,
                             ثابت int64_t   n_صفاں,
                             ثابت عائم     scale,
                             dpct::قطار_ptr stream) {
    ثابت sycl::range<2> global(n_صفاں, NT);
    ثابت sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<عائم, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_شے<2> شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_صفاں, scale, شے, get_اشارہ_گر(smem));
                         });
    });
}

سانچا <صحيح N, صحيح m, قسم_ناں T>
ساكن فراغ kronecker_kernel(ثابت T * __restrict__ src,
                             عائم * __restrict__ dst,
                             ثابت int64_t            n_صفاں,
                             ثابت عائم              scale,
                             ثابت sycl::nd_شے<2> & شے) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    ثابت sycl::sub_گروہ sg = شے.get_sub_گروہ();

    ثابت int64_t r = شے.get_global_id(0);
    إذا (r >= n_صفاں) {
        أرجع;
    }

    src += r * N;
    dst += r * N;

    constexpr صحيح بلاکاں_فی_گروہ = N / m;
    constexpr صحيح el_w             = بلاکاں_فی_گروہ / WARP_ناپ;
    static_assert(el_w >= 1 && بلاکاں_فی_گروہ % WARP_ناپ == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    عائم     reg[el_w * m];
    ثابت صحيح lane = sg.get_local_خطی_id();

#ہدایت unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        ثابت صحيح b_idx = i * WARP_ناپ + lane;

#ہدایت unroll
        لكل (صحيح j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<عائم>(src[b_idx * m + j]) * scale;
        }
    }

#ہدایت unroll
    لكل (صحيح b = 0; b < el_w; ++b) {
        عائم z[m] = { 0.0f };

#ہدایت unroll
        لكل (صحيح i = 0; i < m; ++i) {
#ہدایت unroll
            لكل (صحيح j = 0; j < m; ++j) {
                ثابت عائم h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#ہدایت unroll
        لكل (صحيح i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#ہدایت unroll
    لكل (صحيح h = 1; h < WARP_ناپ; h *= 2) {
#ہدایت unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
#ہدایت unroll
            لكل (صحيح k = 0; k < m; ++k) {
                ثابت عائم val  = reg[j * m + k];
                ثابت عائم val2 = dpct::ترتیب_بدلو_sub_گروہ_by_xor(sg, val, h, WARP_ناپ);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#ہدایت unroll
    لكل (صحيح h = WARP_ناپ; h < بلاکاں_فی_گروہ; h *= 2) {
        ثابت صحيح قدم = h / WARP_ناپ;
#ہدایت unroll
        لكل (صحيح j = 0; j < el_w; j += 2 * قدم) {
#ہدایت unroll
            لكل (صحيح s = 0; s < قدم; ++s) {
#ہدایت unroll
                لكل (صحيح k = 0; k < m; ++k) {
                    ثابت عائم x = reg[(j + s) * m + k];
                    ثابت عائم y = reg[(j + s + قدم) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + قدم) * m + k] = x - y;
                }
            }
        }
    }

#ہدایت unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        ثابت صحيح b_idx = i * WARP_ناپ + lane;
#ہدایت unroll
        لكل (صحيح k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

سانچا <صحيح N, صحيح m, قسم_ناں T>
ساكن فراغ launch_kronecker(ثابت T *       src,
                             عائم *         dst,
                             ثابت int64_t   n_صفاں,
                             ثابت عائم     scale,
                             dpct::قطار_ptr stream) {
    constexpr صحيح صفاں_فی_بلاک = 4;

    ثابت int64_t عدد_بلاکاں = (n_صفاں + صفاں_فی_بلاک - 1) / صفاں_فی_بلاک;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ثابت sycl::range<2> global(عدد_بلاکاں * صفاں_فی_بلاک, WARP_ناپ);
    ثابت sycl::range<2> local(صفاں_فی_بلاک, WARP_ناپ);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_شے<2> شے) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_صفاں, scale, شے);
                         });
}

سانچا <قسم_ناں T>
ساكن منطقي ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ثابت صحيح     n    = (صحيح) src->ne[0];
    ثابت int64_t صفاں = ggml_nrows(src);

    ثابت T *       src_d  = (ثابت T *) src->انکڑے;
    عائم *         dst_d  = (عائم *) dst->انکڑے;
    dpct::قطار_ptr stream = ctx.stream();

    ثابت عائم scale = 1.0f / std::جذر((عائم) n);

    بدل (n) {
        حالة 64:
            launch_fwht<64>(src_d, dst_d, صفاں, scale, stream);
            أرجع سچ;
        حالة 128:
            launch_fwht<128>(src_d, dst_d, صفاں, scale, stream);
            أرجع سچ;
        حالة 256:
            launch_fwht<256>(src_d, dst_d, صفاں, scale, stream);
            أرجع سچ;
        حالة 512:
            launch_fwht<512>(src_d, dst_d, صفاں, scale, stream);
            أرجع سچ;
        حالة 384:
            launch_kronecker<384, 12>(src_d, dst_d, صفاں, scale, stream);
            أرجع سچ;
        حالة 768:
            launch_kronecker<768, 12>(src_d, dst_d, صفاں, scale, stream);
            أرجع سچ;
        حالة 640:
            launch_kronecker<640, 20>(src_d, dst_d, صفاں, scale, stream);
            أرجع سچ;
        حالة 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, صفاں, scale, stream);
            أرجع سچ;
        حالة 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, صفاں, scale, stream);
            أرجع سچ;
        حالة 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, صفاں, scale, stream);
            أرجع سچ;
        حالة 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, صفاں, scale, stream);
            أرجع سچ;
        حالة 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, صفاں, scale, stream);
            أرجع سچ;
        افتراضي:
            أرجع خطأ;
    }
}

منطقي ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src, ggml_tensor * dst) {
    إذا (dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }
    إذا (!ggml_are_same_shape(src, dst)) {
        أرجع خطأ;
    }
    إذا (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }

    بدل (src->type) {
        حالة GGML_TYPE_F32:
            أرجع ggml_sycl_op_fwht_impl<عائم>(ctx, src, dst);
        حالة GGML_TYPE_F16:
            أرجع ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        افتراضي:
            أرجع خطأ;
    }
}
