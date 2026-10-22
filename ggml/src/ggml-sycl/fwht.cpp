#ضمن "fwht.hpp"

#ضمن <cmath>
#تعریف_کن P 1.0f
#تعریف_کن N -1.0f

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

#لغو_تعریف P
#لغو_تعریف N

قالب <صحيح N, نام_نوع T>
ساكن فراغ fwht_kernel(ثابت T * __restrict__ src, عائم * __restrict__ dst, ثابت int64_t n_ردیفها,
                        ثابت عائم scale, ثابت sycl::nd_قلم<2> & قلم) {
    ثابت sycl::sub_گروه sg = قلم.get_sub_گروه();

    ثابت int64_t r = قلم.get_global_id(0);
    إذا (r >= n_ردیفها) {
        أرجع;
    }

    src += r * N;
    dst += r * N;

    constexpr صحيح el_w = N / WARP_اندازه;
    static_assert(el_w >= 1 && N % WARP_اندازه == 0, "row must be a whole number of sub-group widths");

    عائم     reg[el_w];
    ثابت صحيح lane = sg.get_local_خطی_id();

#رهنمود unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        reg[i] = static_cast<عائم>(src[i * WARP_اندازه + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#رهنمود unroll
    لكل (صحيح h = 1; h < WARP_اندازه; h *= 2) {
#رهنمود unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
            ثابت عائم val  = reg[j];
            ثابت عائم val2 = dpct::جایگشت_sub_گروه_by_xor(sg, val, h, WARP_اندازه);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#رهنمود unroll
    لكل (صحيح h = WARP_اندازه; h < N; h *= 2) {
        ثابت صحيح گام = h / WARP_اندازه;
#رهنمود unroll
        لكل (صحيح j = 0; j < el_w; j += 2 * گام) {
#رهنمود unroll
            لكل (صحيح k = 0; k < گام; ++k) {
                ثابت عائم x = reg[j + k];
                ثابت عائم y = reg[j + k + گام];

                reg[j + k]        = x + y;
                reg[j + k + گام] = x - y;
            }
        }
    }

#رهنمود unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        dst[i * WARP_اندازه + lane] = reg[i];
    }
}

قالب <صحيح N, نام_نوع T>
ساكن فراغ launch_fwht(ثابت T * src, عائم * dst, ثابت int64_t n_ردیفها, ثابت عائم scale,
                        dpct::صف_ptr stream) {
    constexpr صحيح ردیفها_در_هر_بلوک = 4;

    ثابت int64_t رقم_بلوکها = (n_ردیفها + ردیفها_در_هر_بلوک - 1) / ردیفها_در_هر_بلوک;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ثابت sycl::range<2> global(رقم_بلوکها * ردیفها_در_هر_بلوک, WARP_اندازه);
    ثابت sycl::range<2> local(ردیفها_در_هر_بلوک, WARP_اندازه);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_قلم<2> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                             fwht_kernel<N, T>(src, dst, n_ردیفها, scale, قلم);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
قالب <صحيح N, صحيح NT, نام_نوع T>
ساكن فراغ fwht_kernel_wide(ثابت T * __restrict__ src,
                             عائم * __restrict__ dst,
                             ثابت int64_t            n_ردیفها,
                             ثابت عائم              scale,
                             ثابت sycl::nd_قلم<2> & قلم,
                             عائم *                  smem) {
    ثابت int64_t r = قلم.get_global_id(0);
    إذا (r >= n_ردیفها) {
        أرجع;
    }

    src += r * N;
    dst += r * N;

    constexpr صحيح el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    ثابت صحيح tid = قلم.get_local_id(1);

    عائم reg[el_w];
#رهنمود unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        reg[i] = static_cast<عائم>(src[i * NT + tid]) * scale;
    }

    ثابت sycl::sub_گروه sg   = قلم.get_sub_گروه();
    ثابت صحيح             lane = sg.get_local_خطی_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#رهنمود unroll
    لكل (صحيح h = 1; h < WARP_اندازه; h *= 2) {
#رهنمود unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
            ثابت عائم val  = reg[j];
            ثابت عائم val2 = dpct::جایگشت_sub_گروه_by_xor(sg, val, h, WARP_اندازه);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    لكل (صحيح h = WARP_اندازه; h < NT; h *= 2) {
#رهنمود unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        قلم.barrier(sycl::access::fence_space::local_space);
#رهنمود unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
            ثابت عائم val  = reg[j];
            ثابت عائم val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        قلم.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    لكل (صحيح h = NT; h < N; h *= 2) {
        ثابت صحيح گام = h / NT;
        لكل (صحيح j = 0; j < el_w; j += 2 * گام) {
            لكل (صحيح k = 0; k < گام; ++k) {
                ثابت عائم x = reg[j + k];
                ثابت عائم y = reg[j + k + گام];

                reg[j + k]        = x + y;
                reg[j + k + گام] = x - y;
            }
        }
    }

#رهنمود unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

قالب <صحيح N, صحيح NT, نام_نوع T>
ساكن فراغ launch_fwht_wide(ثابت T *       src,
                             عائم *         dst,
                             ثابت int64_t   n_ردیفها,
                             ثابت عائم     scale,
                             dpct::صف_ptr stream) {
    ثابت sycl::range<2> global(n_ردیفها, NT);
    ثابت sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<عائم, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_قلم<2> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_ردیفها, scale, قلم, get_اشاره_گر(smem));
                         });
    });
}

قالب <صحيح N, صحيح m, نام_نوع T>
ساكن فراغ kronecker_kernel(ثابت T * __restrict__ src,
                             عائم * __restrict__ dst,
                             ثابت int64_t            n_ردیفها,
                             ثابت عائم              scale,
                             ثابت sycl::nd_قلم<2> & قلم) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    ثابت sycl::sub_گروه sg = قلم.get_sub_گروه();

    ثابت int64_t r = قلم.get_global_id(0);
    إذا (r >= n_ردیفها) {
        أرجع;
    }

    src += r * N;
    dst += r * N;

    constexpr صحيح بلوکها_در_هر_گروه = N / m;
    constexpr صحيح el_w             = بلوکها_در_هر_گروه / WARP_اندازه;
    static_assert(el_w >= 1 && بلوکها_در_هر_گروه % WARP_اندازه == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    عائم     reg[el_w * m];
    ثابت صحيح lane = sg.get_local_خطی_id();

#رهنمود unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        ثابت صحيح b_idx = i * WARP_اندازه + lane;

#رهنمود unroll
        لكل (صحيح j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<عائم>(src[b_idx * m + j]) * scale;
        }
    }

#رهنمود unroll
    لكل (صحيح b = 0; b < el_w; ++b) {
        عائم z[m] = { 0.0f };

#رهنمود unroll
        لكل (صحيح i = 0; i < m; ++i) {
#رهنمود unroll
            لكل (صحيح j = 0; j < m; ++j) {
                ثابت عائم h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#رهنمود unroll
        لكل (صحيح i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#رهنمود unroll
    لكل (صحيح h = 1; h < WARP_اندازه; h *= 2) {
#رهنمود unroll
        لكل (صحيح j = 0; j < el_w; ++j) {
#رهنمود unroll
            لكل (صحيح k = 0; k < m; ++k) {
                ثابت عائم val  = reg[j * m + k];
                ثابت عائم val2 = dpct::جایگشت_sub_گروه_by_xor(sg, val, h, WARP_اندازه);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#رهنمود unroll
    لكل (صحيح h = WARP_اندازه; h < بلوکها_در_هر_گروه; h *= 2) {
        ثابت صحيح گام = h / WARP_اندازه;
#رهنمود unroll
        لكل (صحيح j = 0; j < el_w; j += 2 * گام) {
#رهنمود unroll
            لكل (صحيح s = 0; s < گام; ++s) {
#رهنمود unroll
                لكل (صحيح k = 0; k < m; ++k) {
                    ثابت عائم x = reg[(j + s) * m + k];
                    ثابت عائم y = reg[(j + s + گام) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + گام) * m + k] = x - y;
                }
            }
        }
    }

#رهنمود unroll
    لكل (صحيح i = 0; i < el_w; ++i) {
        ثابت صحيح b_idx = i * WARP_اندازه + lane;
#رهنمود unroll
        لكل (صحيح k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

قالب <صحيح N, صحيح m, نام_نوع T>
ساكن فراغ launch_kronecker(ثابت T *       src,
                             عائم *         dst,
                             ثابت int64_t   n_ردیفها,
                             ثابت عائم     scale,
                             dpct::صف_ptr stream) {
    constexpr صحيح ردیفها_در_هر_بلوک = 4;

    ثابت int64_t رقم_بلوکها = (n_ردیفها + ردیفها_در_هر_بلوک - 1) / ردیفها_در_هر_بلوک;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    ثابت sycl::range<2> global(رقم_بلوکها * ردیفها_در_هر_بلوک, WARP_اندازه);
    ثابت sycl::range<2> local(ردیفها_در_هر_بلوک, WARP_اندازه);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_قلم<2> قلم) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_ردیفها, scale, قلم);
                         });
}

قالب <نام_نوع T>
ساكن منطقي ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src, ggml_tensor * dst) {
    ثابت صحيح     n    = (صحيح) src->ne[0];
    ثابت int64_t ردیفها = ggml_nrows(src);

    ثابت T *       src_d  = (ثابت T *) src->داده;
    عائم *         dst_d  = (عائم *) dst->داده;
    dpct::صف_ptr stream = ctx.stream();

    ثابت عائم scale = 1.0f / std::جذر((عائم) n);

    بدل (n) {
        حالة 64:
            launch_fwht<64>(src_d, dst_d, ردیفها, scale, stream);
            أرجع درست;
        حالة 128:
            launch_fwht<128>(src_d, dst_d, ردیفها, scale, stream);
            أرجع درست;
        حالة 256:
            launch_fwht<256>(src_d, dst_d, ردیفها, scale, stream);
            أرجع درست;
        حالة 512:
            launch_fwht<512>(src_d, dst_d, ردیفها, scale, stream);
            أرجع درست;
        حالة 384:
            launch_kronecker<384, 12>(src_d, dst_d, ردیفها, scale, stream);
            أرجع درست;
        حالة 768:
            launch_kronecker<768, 12>(src_d, dst_d, ردیفها, scale, stream);
            أرجع درست;
        حالة 640:
            launch_kronecker<640, 20>(src_d, dst_d, ردیفها, scale, stream);
            أرجع درست;
        حالة 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, ردیفها, scale, stream);
            أرجع درست;
        حالة 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, ردیفها, scale, stream);
            أرجع درست;
        حالة 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, ردیفها, scale, stream);
            أرجع درست;
        حالة 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, ردیفها, scale, stream);
            أرجع درست;
        حالة 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, ردیفها, scale, stream);
            أرجع درست;
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
