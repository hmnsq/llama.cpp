#అన్తర్భావ "fwht.hpp"

#అన్తర్భావ <cmath>
#నిర్వచించు P 1.0f
#నిర్వచించు N -1.0f

// constant Hadamard matrix via Paley I construction
స్థిర constexpr దశాంశం H12[12][12] = {
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

స్థిర constexpr దశాంశం H20[20][20] = {
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

#నిర్వచనం_రద్దు P
#నిర్వచనం_రద్దు N

మూస <పూర్ణాంకం N, రకం_పేరు T>
స్థిర శూన్య fwht_kernel(స్థిరరాశి T * __restrict__ src, దశాంశం * __restrict__ dst, స్థిరరాశి int64_t n_అడ్డువరుసలు,
                        స్థిరరాశి దశాంశం scale, స్థిరరాశి sycl::nd_అంశం<2> & అంశం) {
    స్థిరరాశి sycl::sub_సమూహం sg = అంశం.get_sub_సమూహం();

    స్థిరరాశి int64_t r = అంశం.get_global_id(0);
    ఒకవేళ (r >= n_అడ్డువరుసలు) {
        తిరిగి;
    }

    src += r * N;
    dst += r * N;

    constexpr పూర్ణాంకం el_w = N / WARP_కొలత;
    static_assert(el_w >= 1 && N % WARP_కొలత == 0, "row must be a whole number of sub-group widths");

    దశాంశం     reg[el_w];
    స్థిరరాశి పూర్ణాంకం lane = sg.get_local_రేఖీయ_id();

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < el_w; ++i) {
        reg[i] = static_cast<దశాంశం>(src[i * WARP_కొలత + lane]) * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#సూచన unroll
    కొరకు (పూర్ణాంకం h = 1; h < WARP_కొలత; h *= 2) {
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < el_w; ++j) {
            స్థిరరాశి దశాంశం val  = reg[j];
            స్థిరరాశి దశాంశం val2 = dpct::క్రమమార్పు_sub_సమూహం_by_xor(sg, val, h, WARP_కొలత);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#సూచన unroll
    కొరకు (పూర్ణాంకం h = WARP_కొలత; h < N; h *= 2) {
        స్థిరరాశి పూర్ణాంకం అడుగు = h / WARP_కొలత;
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < el_w; j += 2 * అడుగు) {
#సూచన unroll
            కొరకు (పూర్ణాంకం k = 0; k < అడుగు; ++k) {
                స్థిరరాశి దశాంశం x = reg[j + k];
                స్థిరరాశి దశాంశం y = reg[j + k + అడుగు];

                reg[j + k]        = x + y;
                reg[j + k + అడుగు] = x - y;
            }
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < el_w; ++i) {
        dst[i * WARP_కొలత + lane] = reg[i];
    }
}

మూస <పూర్ణాంకం N, రకం_పేరు T>
స్థిర శూన్య launch_fwht(స్థిరరాశి T * src, దశాంశం * dst, స్థిరరాశి int64_t n_అడ్డువరుసలు, స్థిరరాశి దశాంశం scale,
                        dpct::వరుస_ptr stream) {
    constexpr పూర్ణాంకం అడ్డువరుసలు_చొప్పున_ఖండం = 4;

    స్థిరరాశి int64_t అంకె_ఖండాలు = (n_అడ్డువరుసలు + అడ్డువరుసలు_చొప్పున_ఖండం - 1) / అడ్డువరుసలు_చొప్పున_ఖండం;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    స్థిరరాశి sycl::range<2> global(అంకె_ఖండాలు * అడ్డువరుసలు_చొప్పున_ఖండం, WARP_కొలత);
    స్థిరరాశి sycl::range<2> local(అడ్డువరుసలు_చొప్పున_ఖండం, WARP_కొలత);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_అంశం<2> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             fwht_kernel<N, T>(src, dst, n_అడ్డువరుసలు, scale, అంశం);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
మూస <పూర్ణాంకం N, పూర్ణాంకం NT, రకం_పేరు T>
స్థిర శూన్య fwht_kernel_wide(స్థిరరాశి T * __restrict__ src,
                             దశాంశం * __restrict__ dst,
                             స్థిరరాశి int64_t            n_అడ్డువరుసలు,
                             స్థిరరాశి దశాంశం              scale,
                             స్థిరరాశి sycl::nd_అంశం<2> & అంశం,
                             దశాంశం *                  smem) {
    స్థిరరాశి int64_t r = అంశం.get_global_id(0);
    ఒకవేళ (r >= n_అడ్డువరుసలు) {
        తిరిగి;
    }

    src += r * N;
    dst += r * N;

    constexpr పూర్ణాంకం el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    స్థిరరాశి పూర్ణాంకం tid = అంశం.get_local_id(1);

    దశాంశం reg[el_w];
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < el_w; ++i) {
        reg[i] = static_cast<దశాంశం>(src[i * NT + tid]) * scale;
    }

    స్థిరరాశి sycl::sub_సమూహం sg   = అంశం.get_sub_సమూహం();
    స్థిరరాశి పూర్ణాంకం             lane = sg.get_local_రేఖీయ_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#సూచన unroll
    కొరకు (పూర్ణాంకం h = 1; h < WARP_కొలత; h *= 2) {
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < el_w; ++j) {
            స్థిరరాశి దశాంశం val  = reg[j];
            స్థిరరాశి దశాంశం val2 = dpct::క్రమమార్పు_sub_సమూహం_by_xor(sg, val, h, WARP_కొలత);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    కొరకు (పూర్ణాంకం h = WARP_కొలత; h < NT; h *= 2) {
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        అంశం.barrier(sycl::access::fence_space::local_space);
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < el_w; ++j) {
            స్థిరరాశి దశాంశం val  = reg[j];
            స్థిరరాశి దశాంశం val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        అంశం.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    కొరకు (పూర్ణాంకం h = NT; h < N; h *= 2) {
        స్థిరరాశి పూర్ణాంకం అడుగు = h / NT;
        కొరకు (పూర్ణాంకం j = 0; j < el_w; j += 2 * అడుగు) {
            కొరకు (పూర్ణాంకం k = 0; k < అడుగు; ++k) {
                స్థిరరాశి దశాంశం x = reg[j + k];
                స్థిరరాశి దశాంశం y = reg[j + k + అడుగు];

                reg[j + k]        = x + y;
                reg[j + k + అడుగు] = x - y;
            }
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

మూస <పూర్ణాంకం N, పూర్ణాంకం NT, రకం_పేరు T>
స్థిర శూన్య launch_fwht_wide(స్థిరరాశి T *       src,
                             దశాంశం *         dst,
                             స్థిరరాశి int64_t   n_అడ్డువరుసలు,
                             స్థిరరాశి దశాంశం     scale,
                             dpct::వరుస_ptr stream) {
    స్థిరరాశి sycl::range<2> global(n_అడ్డువరుసలు, NT);
    స్థిరరాశి sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<దశాంశం, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_అంశం<2> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             fwht_kernel_wide<N, NT, T>(src, dst, n_అడ్డువరుసలు, scale, అంశం, get_సూచిక(smem));
                         });
    });
}

మూస <పూర్ణాంకం N, పూర్ణాంకం m, రకం_పేరు T>
స్థిర శూన్య kronecker_kernel(స్థిరరాశి T * __restrict__ src,
                             దశాంశం * __restrict__ dst,
                             స్థిరరాశి int64_t            n_అడ్డువరుసలు,
                             స్థిరరాశి దశాంశం              scale,
                             స్థిరరాశి sycl::nd_అంశం<2> & అంశం) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    స్థిరరాశి sycl::sub_సమూహం sg = అంశం.get_sub_సమూహం();

    స్థిరరాశి int64_t r = అంశం.get_global_id(0);
    ఒకవేళ (r >= n_అడ్డువరుసలు) {
        తిరిగి;
    }

    src += r * N;
    dst += r * N;

    constexpr పూర్ణాంకం ఖండాలు_చొప్పున_సమూహం = N / m;
    constexpr పూర్ణాంకం el_w             = ఖండాలు_చొప్పున_సమూహం / WARP_కొలత;
    static_assert(el_w >= 1 && ఖండాలు_చొప్పున_సమూహం % WARP_కొలత == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    దశాంశం     reg[el_w * m];
    స్థిరరాశి పూర్ణాంకం lane = sg.get_local_రేఖీయ_id();

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < el_w; ++i) {
        స్థిరరాశి పూర్ణాంకం b_idx = i * WARP_కొలత + lane;

#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < m; ++j) {
            reg[i * m + j] = static_cast<దశాంశం>(src[b_idx * m + j]) * scale;
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం b = 0; b < el_w; ++b) {
        దశాంశం z[m] = { 0.0f };

#సూచన unroll
        కొరకు (పూర్ణాంకం i = 0; i < m; ++i) {
#సూచన unroll
            కొరకు (పూర్ణాంకం j = 0; j < m; ++j) {
                స్థిరరాశి దశాంశం h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#సూచన unroll
        కొరకు (పూర్ణాంకం i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం h = 1; h < WARP_కొలత; h *= 2) {
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < el_w; ++j) {
#సూచన unroll
            కొరకు (పూర్ణాంకం k = 0; k < m; ++k) {
                స్థిరరాశి దశాంశం val  = reg[j * m + k];
                స్థిరరాశి దశాంశం val2 = dpct::క్రమమార్పు_sub_సమూహం_by_xor(sg, val, h, WARP_కొలత);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం h = WARP_కొలత; h < ఖండాలు_చొప్పున_సమూహం; h *= 2) {
        స్థిరరాశి పూర్ణాంకం అడుగు = h / WARP_కొలత;
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < el_w; j += 2 * అడుగు) {
#సూచన unroll
            కొరకు (పూర్ణాంకం s = 0; s < అడుగు; ++s) {
#సూచన unroll
                కొరకు (పూర్ణాంకం k = 0; k < m; ++k) {
                    స్థిరరాశి దశాంశం x = reg[(j + s) * m + k];
                    స్థిరరాశి దశాంశం y = reg[(j + s + అడుగు) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + అడుగు) * m + k] = x - y;
                }
            }
        }
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < el_w; ++i) {
        స్థిరరాశి పూర్ణాంకం b_idx = i * WARP_కొలత + lane;
#సూచన unroll
        కొరకు (పూర్ణాంకం k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

మూస <పూర్ణాంకం N, పూర్ణాంకం m, రకం_పేరు T>
స్థిర శూన్య launch_kronecker(స్థిరరాశి T *       src,
                             దశాంశం *         dst,
                             స్థిరరాశి int64_t   n_అడ్డువరుసలు,
                             స్థిరరాశి దశాంశం     scale,
                             dpct::వరుస_ptr stream) {
    constexpr పూర్ణాంకం అడ్డువరుసలు_చొప్పున_ఖండం = 4;

    స్థిరరాశి int64_t అంకె_ఖండాలు = (n_అడ్డువరుసలు + అడ్డువరుసలు_చొప్పున_ఖండం - 1) / అడ్డువరుసలు_చొప్పున_ఖండం;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    స్థిరరాశి sycl::range<2> global(అంకె_ఖండాలు * అడ్డువరుసలు_చొప్పున_ఖండం, WARP_కొలత);
    స్థిరరాశి sycl::range<2> local(అడ్డువరుసలు_చొప్పున_ఖండం, WARP_కొలత);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_అంశం<2> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             kronecker_kernel<N, m, T>(src, dst, n_అడ్డువరుసలు, scale, అంశం);
                         });
}

మూస <రకం_పేరు T>
స్థిర తార్కిక ggml_sycl_op_fwht_impl(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst) {
    స్థిరరాశి పూర్ణాంకం     n    = (పూర్ణాంకం) src->ne[0];
    స్థిరరాశి int64_t అడ్డువరుసలు = ggml_nrows(src);

    స్థిరరాశి T *       src_d  = (స్థిరరాశి T *) src->దత్తాంశం;
    దశాంశం *         dst_d  = (దశాంశం *) dst->దత్తాంశం;
    dpct::వరుస_ptr stream = ctx.stream();

    స్థిరరాశి దశాంశం scale = 1.0f / std::వర్గమూలం((దశాంశం) n);

    ఎంపిక (n) {
        స్థితి 64:
            launch_fwht<64>(src_d, dst_d, అడ్డువరుసలు, scale, stream);
            తిరిగి నిజం;
        స్థితి 128:
            launch_fwht<128>(src_d, dst_d, అడ్డువరుసలు, scale, stream);
            తిరిగి నిజం;
        స్థితి 256:
            launch_fwht<256>(src_d, dst_d, అడ్డువరుసలు, scale, stream);
            తిరిగి నిజం;
        స్థితి 512:
            launch_fwht<512>(src_d, dst_d, అడ్డువరుసలు, scale, stream);
            తిరిగి నిజం;
        స్థితి 384:
            launch_kronecker<384, 12>(src_d, dst_d, అడ్డువరుసలు, scale, stream);
            తిరిగి నిజం;
        స్థితి 768:
            launch_kronecker<768, 12>(src_d, dst_d, అడ్డువరుసలు, scale, stream);
            తిరిగి నిజం;
        స్థితి 640:
            launch_kronecker<640, 20>(src_d, dst_d, అడ్డువరుసలు, scale, stream);
            తిరిగి నిజం;
        స్థితి 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, అడ్డువరుసలు, scale, stream);
            తిరిగి నిజం;
        స్థితి 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, అడ్డువరుసలు, scale, stream);
            తిరిగి నిజం;
        స్థితి 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, అడ్డువరుసలు, scale, stream);
            తిరిగి నిజం;
        స్థితి 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, అడ్డువరుసలు, scale, stream);
            తిరిగి నిజం;
        స్థితి 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, అడ్డువరుసలు, scale, stream);
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

తార్కిక ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst) {
    ఒకవేళ (dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_are_same_shape(src, dst)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }

    ఎంపిక (src->type) {
        స్థితి GGML_TYPE_F32:
            తిరిగి ggml_sycl_op_fwht_impl<దశాంశం>(ctx, src, dst);
        స్థితి GGML_TYPE_F16:
            తిరిగి ggml_sycl_op_fwht_impl<sycl::half>(ctx, src, dst);
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}
