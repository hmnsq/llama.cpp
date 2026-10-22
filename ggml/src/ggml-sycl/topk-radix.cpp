#ਅਨ੍ਤਰ੍ਭਾਵ "topk-radix.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>

// Large-k top-k by radix select on an order-preserving unsigned key.
//
// The k-th largest key of a row is found by four most-significant-first passes over its
// 8-bit digits: histogram the digit over the candidate set, walk the buckets from the
// top, and recurse into the bucket where the running count reaches what is still
// needed. Everything strictly above that bucket is in the top-k. A final pass emits
// every column whose key beats the pivot, then exactly as many pivot-equal columns as
// are still missing, so duplicate keys yield exactly k distinct indices.
//
// SLM holds only the histogram, so unlike the scan-merge kernels the cost does not grow
// with k. One work-group owns a row and runs every pass, so a top-k is one launch and
// needs no pool scratch. The row is re-read once per pass rather than compacted, which
// keeps the candidate set implicit: (key & mask) == prefix.
//
// The output is the set of winning indices in no particular order, which is what the
// reference op provides (it swaps its first two outputs to say so) and what
// test-backend-ops compares.

ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_ਬਿੱਟ       = 8;
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_BUCKETS    = 1 << SYCL_ਸਿਖਰ_K_RADIX_ਬਿੱਟ;
// Private histogram copies, interleaved per bucket so neighbouring lanes hit
// neighbouring banks. Lanes of one instruction spread over the copies, which is what
// bounds the atomic serialisation on tie-heavy rows.
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_HIST_COPIES = 8;
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_HIST_ਮਾਪ   = SYCL_ਸਿਖਰ_K_RADIX_BUCKETS * SYCL_ਸਿਖਰ_K_RADIX_HIST_COPIES;
// Past the histogram: pivot digit, pivot bucket count, remaining need, then the two
// emit counters.
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_SLM_ਸ਼ਬਦ   = SYCL_ਸਿਖਰ_K_RADIX_HIST_ਮਾਪ + 5;

// Larger float <=> larger key. The reference comparator is a plain float '>', under which
// -0.0 and +0.0 tie, so -0.0 is folded onto +0.0 first. NaN has no defined order in the
// reference (its comparator is not a strict weak order on NaN); here a positive NaN keys
// above +inf and a negative NaN below -inf, which at least makes the result deterministic.
ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t ਸਿਖਰ_k_radix_ਕੁੰਜੀ(ਦਸ਼ਮਲਵ f) {
    uint32_t u = sycl::ਦੋਆਧਾਰੀ_ਅੰਕ_cast<uint32_t>(f);
    ਜੇ (u == 0x80000000u) {
        u = 0u;
    }
    ਵਾਪਸ (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

ਸਥਿਰ ਖਾਲੀ ਸਿਖਰ_k_radix_select_f32(
    ਅਚਲ ਦਸ਼ਮਲਵ *   src,
    int32_t *       dst_idx,
    ਅਚਲ ਪੂਰਨ       ncols,
    ਅਚਲ ਪੂਰਨ       k,
    uint32_t *      slm,
    ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1
) {
    ਵਰਤੋਂ local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::ਸਮ੍ਰਿਤੀ_scope::work_ਸਮੂਹ,
                                          sycl::access::ਪਤਾ_space::local_space>;

    ਅਚਲ ਪੂਰਨ tid        = ਵਸਤੂ_ct1.get_local_id(0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = ਵਸਤੂ_ct1.get_local_range(0);

    uint32_t * hist     = slm;
    uint32_t * s_digit  = slm + SYCL_ਸਿਖਰ_K_RADIX_HIST_ਮਾਪ;
    uint32_t * s_ਬਾਲਟੀ = slm + SYCL_ਸਿਖਰ_K_RADIX_HIST_ਮਾਪ + 1;
    uint32_t * s_need   = slm + SYCL_ਸਿਖਰ_K_RADIX_HIST_ਮਾਪ + 2;
    uint32_t * s_cnt_gt = slm + SYCL_ਸਿਖਰ_K_RADIX_HIST_ਮਾਪ + 3;
    uint32_t * s_cnt_eq = slm + SYCL_ਸਿਖਰ_K_RADIX_HIST_ਮਾਪ + 4;

    ਜੇ (tid == 0) {
        *s_cnt_gt = 0;
        *s_cnt_eq = 0;
    }

    ਅਚਲ ਪੂਰਨ copy = tid & (SYCL_ਸਿਖਰ_K_RADIX_HIST_COPIES - 1);

    uint32_t prefix = 0;   // digits fixed so far, in place
    uint32_t ਨਕਾਬ   = 0;   // which bits of prefix are fixed
    uint32_t need   = (uint32_t) k;

    ਲਈ (ਪੂਰਨ ਖਿਸਕਾਓ = 32 - SYCL_ਸਿਖਰ_K_RADIX_ਬਿੱਟ; ਖਿਸਕਾਓ >= 0; ਖਿਸਕਾਓ -= SYCL_ਸਿਖਰ_K_RADIX_ਬਿੱਟ) {
        ਲਈ (ਪੂਰਨ i = tid; i < SYCL_ਸਿਖਰ_K_RADIX_HIST_ਮਾਪ; i += ਖੰਡ_ਮਾਪ_3) {
            hist[i] = 0;
        }
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

        ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += ਖੰਡ_ਮਾਪ_3) {
            ਅਚਲ uint32_t ਕੁੰਜੀ = ਸਿਖਰ_k_radix_ਕੁੰਜੀ(src[ਖਾਨਾ]);
            ਜੇ ((ਕੁੰਜੀ & ਨਕਾਬ) == prefix) {
                ਅਚਲ uint32_t bucket = (ਕੁੰਜੀ >> ਖਿਸਕਾਓ) & (SYCL_ਸਿਖਰ_K_RADIX_BUCKETS - 1);
                local_atomic(hist[bucket * SYCL_ਸਿਖਰ_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
            }
        }
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

        // Lane t takes bucket 255 - t, so an inclusive scan over lanes counts from the top
        // bucket downward. The pivot is the unique bucket whose cumulative count first
        // reaches need; the previous cumulative count is what the higher buckets contribute.
        uint32_t cnt = 0;
        ਜੇ (tid < SYCL_ਸਿਖਰ_K_RADIX_BUCKETS) {
            ਅਚਲ uint32_t * h = hist + (SYCL_ਸਿਖਰ_K_RADIX_BUCKETS - 1 - tid) * SYCL_ਸਿਖਰ_K_RADIX_HIST_COPIES;
            ਲਈ (ਪੂਰਨ c = 0; c < SYCL_ਸਿਖਰ_K_RADIX_HIST_COPIES; c++) {
                cnt += h[c];
            }
        }
        ਅਚਲ uint32_t incl = sycl::inclusive_ਜਾਂਚ_over_ਸਮੂਹ(ਵਸਤੂ_ct1.get_ਸਮੂਹ(), cnt, sycl::plus<uint32_t>());

        ਜੇ (tid < SYCL_ਸਿਖਰ_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
            *s_digit = (uint32_t) (SYCL_ਸਿਖਰ_K_RADIX_BUCKETS - 1 - tid);
            *s_ਬਾਲਟੀ = cnt;
            *s_need   = need - (incl - cnt);
        }
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

        ਅਚਲ uint32_t digit      = *s_digit;
        ਅਚਲ uint32_t ਬਾਲਟੀ_cnt = *s_ਬਾਲਟੀ;
        need   = *s_need;
        prefix |= digit << ਖਿਸਕਾਓ;
        ਨਕਾਬ   |= (uint32_t) (SYCL_ਸਿਖਰ_K_RADIX_BUCKETS - 1) << ਖਿਸਕਾਓ;

        // Every candidate in the pivot bucket is wanted: the remaining digits cannot
        // change the answer, and the masked emit below is exact as it stands.
        ਜੇ (ਬਾਲਟੀ_cnt == need) {
            ਰੋਕੋ;
        }
        // The next pass rewrites hist and s_*; the reads above must land first.
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);
    }

    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    // Exactly k - need columns have (key & mask) > prefix; the first need of the pivot-equal
    // columns fill the tail. Both counters live in SLM since the whole row is this group.
    ਅਚਲ uint32_t ਅਧਾਰ_eq = (uint32_t) k - need;

    ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += ਖੰਡ_ਮਾਪ_3) {
        ਅਚਲ uint32_t kp = ਸਿਖਰ_k_radix_ਕੁੰਜੀ(src[ਖਾਨਾ]) & ਨਕਾਬ;
        ਜੇ (kp > prefix) {
            ਅਚਲ uint32_t pos = local_atomic(*s_cnt_gt).fetch_add(1u);
            dst_idx[pos] = ਖਾਨਾ;
        } ਨਹੀਂਤਾਂ ਜੇ (kp == prefix) {
            ਅਚਲ uint32_t pos = local_atomic(*s_cnt_eq).fetch_add(1u);
            ਜੇ (pos < need) {
                dst_idx[ਅਧਾਰ_eq + pos] = ਖਾਨਾ;
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ਸਿਖਰ_k_radix_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ਅਚਲ ਦਸ਼ਮਲਵ * src,
    int32_t * dst_indices,
    ਅਚਲ int64_t ncols,
    ਅਚਲ int64_t nrows,
    ਅਚਲ ਪੂਰਨ k,
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ
) {
    GGML_ASSERT(ncols <= INT32_MAX);

    // One group per row; every pass is a strided sweep of the row, so lanes in flight is the
    // only lever, and the device's own limit is the answer -- there is nothing here that
    // wants a smaller group. Must still cover the 256 buckets for the scan step.
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[ctx.device];
    GGML_ASSERT(ਖੰਡ_ਮਾਪ_3 >= SYCL_ਸਿਖਰ_K_RADIX_BUCKETS);

    ਅਚਲ sycl::range<1> ਖੰਡ_dims(ਖੰਡ_ਮਾਪ_3);
    ਅਚਲ sycl::range<1> grid_dims(nrows);

    main_ਧਾਰਾ->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_ਸਿਖਰ_K_RADIX_SLM_ਸ਼ਬਦ), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
                ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);

                ਸਿਖਰ_k_radix_select_f32(
                    src + (int64_t) ਪੰਕਤੀ * ncols, dst_indices + (int64_t) ਪੰਕਤੀ * k,
                    (ਪੂਰਨ) ncols, k,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    ਵਸਤੂ_ct1);
            });
    });
}

// One work-group owns a whole row above, which leaves the device idle whenever a graph
// has fewer rows than it has cores -- the common case at batch size 1, where the
// sparse-attention indexer and the backend sampler both top-k a single row. The kernels
// below spread one row over several groups instead.
//
// A digit pass now needs the whole row's histogram before any group can pick the pivot,
// so the per-pass state moves to global memory and the passes become separate launches:
// a work-group barrier no longer spans the row. Each group still accumulates into SLM
// and contributes 256 global atomics at the end, so global traffic is per-group, not
// per-element. The last group to finish a pass (the one whose fetch_add returns G - 1)
// does the scan for the row and clears the histogram for the next pass, which keeps the
// launch count at one per digit rather than two.
//
// Running all four digits unconditionally costs nothing in correctness: once a bucket
// holds exactly the elements still needed, later digits only extend the prefix, and the
// count of columns above that longer prefix grows by exactly as much as `need` shrinks.
// The emit below therefore stays exact whatever pass the answer settled on.

ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਪੂਰਾ   = SYCL_ਸਿਖਰ_K_RADIX_BUCKETS + 0;
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਅਗੇਤਰ = SYCL_ਸਿਖਰ_K_RADIX_BUCKETS + 1;
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਨਕਾਬ   = SYCL_ਸਿਖਰ_K_RADIX_BUCKETS + 2;
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_NEED   = SYCL_ਸਿਖਰ_K_RADIX_BUCKETS + 3;
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_CNT_GT = SYCL_ਸਿਖਰ_K_RADIX_BUCKETS + 4;
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_CNT_EQ = SYCL_ਸਿਖਰ_K_RADIX_BUCKETS + 5;
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਸ਼ਬਦ  = SYCL_ਸਿਖਰ_K_RADIX_BUCKETS + 6;

// How wide the split goes is a property of the device, not of the model: enough groups to
// cover the cores, and no more. Past that the extra groups add histogram traffic without
// adding bandwidth (measured on this device: 20 and 40 groups tie, 60 and 160 lose).
//
// nsm is max_compute_units / 16, i.e. it counts an Xe core as 16 EUs. That is a core's
// width on Xe-HPG, but an Xe2 core is 8 XVEs wide, so on Battlemage the field reads half
// the cores actually present (10 for a 20-core B60). The measured curve is flat from one
// group per core to two and only falls off at three, so a factor of two covers the device
// on Xe2 and lands in the flat region on Xe-HPG. It is the one number here that a correct
// core count would remove; it was tuned on Xe2 and has not been measured on Xe-HPG.
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_GROUPS_ਪ੍ਰਤੀ_NSM = 2;
// Splitting trades one kernel for five. Below the width at which the single-group kernel
// runs longer than those four extra launches, it wins on its own; measured break-even on
// this device sits just under 64K columns.
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_MIN_ਵੰਡੋ_ਖਾਨੇ_ਗਿਣਤੀ = 65536;
// A partition thinner than this cannot keep a group's sweep busy.
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_ਸਿਖਰ_K_RADIX_MIN_PART_ਖਾਨੇ_ਗਿਣਤੀ  = 4096;

ਸਥਿਰ ਪੂਰਨ ਸਿਖਰ_k_radix_ਵੰਡੋ_groups(ਅਚਲ ਪੂਰਨ device, ਅਚਲ int64_t ncols, ਅਚਲ int64_t nrows) {
    ਅਚਲ int64_t ਨਿਸ਼ਾਨਾ = (int64_t) SYCL_ਸਿਖਰ_K_RADIX_GROUPS_ਪ੍ਰਤੀ_NSM * ggml_sycl_info().devices[device].nsm;

    // One group per row already, so a graph with rows enough to cover the device gains
    // nothing from splitting and would only pay the extra launches.
    ਜੇ (ncols < SYCL_ਸਿਖਰ_K_RADIX_MIN_ਵੰਡੋ_ਖਾਨੇ_ਗਿਣਤੀ || nrows >= ਨਿਸ਼ਾਨਾ) {
        ਵਾਪਸ 1;
    }

    ਅਚਲ int64_t by_ਪੰਕਤੀਆਂ = ਨਿਸ਼ਾਨਾ / nrows;   // floor: never overshoot a row that is nearly covered
    ਅਚਲ int64_t by_ਖਾਨੇ_ਗਿਣਤੀ = ncols / SYCL_ਸਿਖਰ_K_RADIX_MIN_PART_ਖਾਨੇ_ਗਿਣਤੀ;

    ਵਾਪਸ (ਪੂਰਨ) std::max<int64_t>(1, std::min(by_ਪੰਕਤੀਆਂ, by_ਖਾਨੇ_ਗਿਣਤੀ));
}

ਵਰਤੋਂ ਸਿਖਰ_k_radix_gatomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                             sycl::ਸਮ੍ਰਿਤੀ_scope::device,
                                             sycl::access::ਪਤਾ_space::global_space>;

ਸਥਿਰ ਖਾਲੀ ਸਿਖਰ_k_radix_ਵੰਡੋ_pass_f32(
    ਅਚਲ ਦਸ਼ਮਲਵ *   src,
    uint32_t *      state,
    ਅਚਲ ਪੂਰਨ       ncols,
    ਅਚਲ ਪੂਰਨ       k,
    ਅਚਲ ਪੂਰਨ       ਖਿਸਕਾਓ,
    ਅਚਲ ਤਾਰਕਿਕ      ਪਹਿਲਾ,
    ਅਚਲ ਪੂਰਨ       part,
    ਅਚਲ ਪੂਰਨ       nparts,
    uint32_t *      slm,
    ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1
) {
    ਵਰਤੋਂ local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::ਸਮ੍ਰਿਤੀ_scope::work_ਸਮੂਹ,
                                          sycl::access::ਪਤਾ_space::local_space>;

    ਅਚਲ ਪੂਰਨ tid        = ਵਸਤੂ_ct1.get_local_id(0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = ਵਸਤੂ_ct1.get_local_range(0);

    uint32_t * hist   = slm;
    uint32_t * s_ਆਖਰੀ = slm + SYCL_ਸਿਖਰ_K_RADIX_HIST_ਮਾਪ;
    uint32_t * s_ਪੰਕਤੀ  = slm + SYCL_ਸਿਖਰ_K_RADIX_HIST_ਮਾਪ + 1;   // prefix, mask, need

    // The previous launch is the barrier that publishes these, so a plain load is enough.
    // One lane reads them and the group takes them from SLM: a device-scope atomic load
    // is uncached here, and having every work-item issue three of them off the same
    // address costs more than the whole sweep below.
    ਜੇ (tid == 0) {
        s_ਪੰਕਤੀ[0] = ਪਹਿਲਾ ? 0u : state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਅਗੇਤਰ];
        s_ਪੰਕਤੀ[1] = ਪਹਿਲਾ ? 0u : state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਨਕਾਬ];
        s_ਪੰਕਤੀ[2] = ਪਹਿਲਾ ? (uint32_t) k : state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_NEED];
    }

    ਲਈ (ਪੂਰਨ i = tid; i < SYCL_ਸਿਖਰ_K_RADIX_HIST_ਮਾਪ; i += ਖੰਡ_ਮਾਪ_3) {
        hist[i] = 0;
    }
    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    ਅਚਲ uint32_t prefix = s_ਪੰਕਤੀ[0];
    ਅਚਲ uint32_t ਨਕਾਬ   = s_ਪੰਕਤੀ[1];
    ਅਚਲ uint32_t need   = s_ਪੰਕਤੀ[2];

    ਅਚਲ ਪੂਰਨ copy  = tid & (SYCL_ਸਿਖਰ_K_RADIX_HIST_COPIES - 1);
    ਅਚਲ ਪੂਰਨ chunk = (ncols + nparts - 1) / nparts;
    ਅਚਲ ਪੂਰਨ col0  = part * chunk;
    ਅਚਲ ਪੂਰਨ col1  = std::min(ncols, col0 + chunk);

    ਲਈ (ਪੂਰਨ ਖਾਨਾ = col0 + tid; ਖਾਨਾ < col1; ਖਾਨਾ += ਖੰਡ_ਮਾਪ_3) {
        ਅਚਲ uint32_t ਕੁੰਜੀ = ਸਿਖਰ_k_radix_ਕੁੰਜੀ(src[ਖਾਨਾ]);
        ਜੇ ((ਕੁੰਜੀ & ਨਕਾਬ) == prefix) {
            ਅਚਲ uint32_t bucket = (ਕੁੰਜੀ >> ਖਿਸਕਾਓ) & (SYCL_ਸਿਖਰ_K_RADIX_BUCKETS - 1);
            local_atomic(hist[bucket * SYCL_ਸਿਖਰ_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
        }
    }
    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    // One global atomic per bucket per group, not per element.
    ਲਈ (ਪੂਰਨ b = tid; b < SYCL_ਸਿਖਰ_K_RADIX_BUCKETS; b += ਖੰਡ_ਮਾਪ_3) {
        uint32_t ਜੋੜ = 0;
        ਲਈ (ਪੂਰਨ c = 0; c < SYCL_ਸਿਖਰ_K_RADIX_HIST_COPIES; c++) {
            ਜੋੜ += hist[b * SYCL_ਸਿਖਰ_K_RADIX_HIST_COPIES + c];
        }
        ਜੇ (ਜੋੜ) {
            ਸਿਖਰ_k_radix_gatomic(state[b]).fetch_add(ਜੋੜ);
        }
    }

    // Publish this group's bins, then claim the scan if this group is the row's last.
    // The group-wide barrier flushes the atomics above; only the claiming lane needs the
    // release, so the device-scope fence is paid once per group rather than per work-item.
    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::global_and_local);
    ਜੇ (tid == 0) {
        sycl::atomic_fence(sycl::memory_order::release, sycl::ਸਮ੍ਰਿਤੀ_scope::device);
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::ਸਮ੍ਰਿਤੀ_scope::device,
                         sycl::access::ਪਤਾ_space::global_space> ਪੂਰਾ(state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਪੂਰਾ]);
        *s_ਆਖਰੀ = (ਪੂਰਾ.fetch_add(1u) == (uint32_t) (nparts - 1)) ? 1u : 0u;
    }
    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    ਜੇ (*s_ਆਖਰੀ == 0u) {
        ਵਾਪਸ;
    }
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::ਸਮ੍ਰਿਤੀ_scope::device);

    // Lane t takes bucket 255 - t, so an inclusive scan counts down from the top bucket.
    uint32_t cnt = 0;
    ਜੇ (tid < SYCL_ਸਿਖਰ_K_RADIX_BUCKETS) {
        cnt = ਸਿਖਰ_k_radix_gatomic(state[SYCL_ਸਿਖਰ_K_RADIX_BUCKETS - 1 - tid]).load();
    }
    ਅਚਲ uint32_t incl = sycl::inclusive_ਜਾਂਚ_over_ਸਮੂਹ(ਵਸਤੂ_ct1.get_ਸਮੂਹ(), cnt, sycl::plus<uint32_t>());

    ਜੇ (tid < SYCL_ਸਿਖਰ_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
        ਅਚਲ uint32_t digit = (uint32_t) (SYCL_ਸਿਖਰ_K_RADIX_BUCKETS - 1 - tid);
        ਸਿਖਰ_k_radix_gatomic(state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਅਗੇਤਰ]).store(prefix | (digit << ਖਿਸਕਾਓ));
        ਸਿਖਰ_k_radix_gatomic(state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਨਕਾਬ]).store(
            ਨਕਾਬ | ((uint32_t) (SYCL_ਸਿਖਰ_K_RADIX_BUCKETS - 1) << ਖਿਸਕਾਓ));
        ਸਿਖਰ_k_radix_gatomic(state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_NEED]).store(need - (incl - cnt));
    }
    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    // Clear for the next pass; the next launch is the barrier that orders this.
    ਲਈ (ਪੂਰਨ b = tid; b < SYCL_ਸਿਖਰ_K_RADIX_BUCKETS; b += ਖੰਡ_ਮਾਪ_3) {
        ਸਿਖਰ_k_radix_gatomic(state[b]).store(0u);
    }
    ਜੇ (tid == 0) {
        ਸਿਖਰ_k_radix_gatomic(state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਪੂਰਾ]).store(0u);
    }
}

ਸਥਿਰ ਖਾਲੀ ਸਿਖਰ_k_radix_ਵੰਡੋ_emit_f32(
    ਅਚਲ ਦਸ਼ਮਲਵ *   src,
    int32_t *       dst_idx,
    uint32_t *      state,
    ਅਚਲ ਪੂਰਨ       ncols,
    ਅਚਲ ਪੂਰਨ       k,
    ਅਚਲ ਪੂਰਨ       part,
    ਅਚਲ ਪੂਰਨ       nparts,
    uint32_t *      slm,
    ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1
) {
    ਵਰਤੋਂ local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::ਸਮ੍ਰਿਤੀ_scope::work_ਸਮੂਹ,
                                          sycl::access::ਪਤਾ_space::local_space>;

    ਅਚਲ ਪੂਰਨ tid        = ਵਸਤੂ_ct1.get_local_id(0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = ਵਸਤੂ_ct1.get_local_range(0);

    uint32_t * s_gt      = slm;
    uint32_t * s_eq      = slm + 1;
    uint32_t * s_ਅਧਾਰ_gt = slm + 2;
    uint32_t * s_ਅਧਾਰ_eq = slm + 3;

    uint32_t * s_ਪੰਕਤੀ = slm + 4;   // prefix, mask, need

    ਜੇ (tid == 0) {
        *s_gt = 0;
        *s_eq = 0;
        s_ਪੰਕਤੀ[0] = state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਅਗੇਤਰ];
        s_ਪੰਕਤੀ[1] = state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਨਕਾਬ];
        s_ਪੰਕਤੀ[2] = state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_NEED];
    }
    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    ਅਚਲ uint32_t prefix = s_ਪੰਕਤੀ[0];
    ਅਚਲ uint32_t ਨਕਾਬ   = s_ਪੰਕਤੀ[1];
    ਅਚਲ uint32_t need   = s_ਪੰਕਤੀ[2];

    // Exactly k - need columns beat the pivot; the first need pivot-equal ones fill the tail.
    ਅਚਲ uint32_t ਅਧਾਰ_eq = (uint32_t) k - need;

    ਅਚਲ ਪੂਰਨ chunk = (ncols + nparts - 1) / nparts;
    ਅਚਲ ਪੂਰਨ col0  = part * chunk;
    ਅਚਲ ਪੂਰਨ col1  = std::min(ncols, col0 + chunk);

    // Counting first and reserving one range per group keeps the row's two counters out of
    // the inner loop: a per-element global atomic on a single address serialises the whole
    // emit, and at k in the thousands that alone outweighs every read the kernel does.
    ਲਈ (ਪੂਰਨ ਖਾਨਾ = col0 + tid; ਖਾਨਾ < col1; ਖਾਨਾ += ਖੰਡ_ਮਾਪ_3) {
        ਅਚਲ uint32_t kp = ਸਿਖਰ_k_radix_ਕੁੰਜੀ(src[ਖਾਨਾ]) & ਨਕਾਬ;
        ਜੇ (kp > prefix) {
            local_atomic(*s_gt).fetch_add(1u);
        } ਨਹੀਂਤਾਂ ਜੇ (kp == prefix) {
            local_atomic(*s_eq).fetch_add(1u);
        }
    }
    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    ਜੇ (tid == 0) {
        ਅਚਲ uint32_t n_gt = *s_gt;
        ਅਚਲ uint32_t n_eq = *s_eq;
        *s_ਅਧਾਰ_gt = n_gt ? ਸਿਖਰ_k_radix_gatomic(state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_CNT_GT]).fetch_add(n_gt) : 0u;
        *s_ਅਧਾਰ_eq = n_eq ? ਸਿਖਰ_k_radix_gatomic(state[SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_CNT_EQ]).fetch_add(n_eq) : 0u;
        *s_gt = 0;
        *s_eq = 0;
    }
    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    ਅਚਲ uint32_t ਅਧਾਰ_gt_g = *s_ਅਧਾਰ_gt;
    ਅਚਲ uint32_t ਅਧਾਰ_eq_g = *s_ਅਧਾਰ_eq;

    ਲਈ (ਪੂਰਨ ਖਾਨਾ = col0 + tid; ਖਾਨਾ < col1; ਖਾਨਾ += ਖੰਡ_ਮਾਪ_3) {
        ਅਚਲ uint32_t kp = ਸਿਖਰ_k_radix_ਕੁੰਜੀ(src[ਖਾਨਾ]) & ਨਕਾਬ;
        ਜੇ (kp > prefix) {
            dst_idx[ਅਧਾਰ_gt_g + local_atomic(*s_gt).fetch_add(1u)] = ਖਾਨਾ;
        } ਨਹੀਂਤਾਂ ਜੇ (kp == prefix) {
            ਅਚਲ uint32_t pos = ਅਧਾਰ_eq_g + local_atomic(*s_eq).fetch_add(1u);
            ਜੇ (pos < need) {
                dst_idx[ਅਧਾਰ_eq + pos] = ਖਾਨਾ;
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ਸਿਖਰ_k_radix_ਵੰਡੋ_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ਅਚਲ ਦਸ਼ਮਲਵ * src,
    int32_t * dst_indices,
    ਅਚਲ int64_t ncols,
    ਅਚਲ int64_t nrows,
    ਅਚਲ ਪੂਰਨ k,
    ਅਚਲ ਪੂਰਨ nparts,
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ
) {
    GGML_ASSERT(ncols <= INT32_MAX);
    GGML_ASSERT(nparts > 1);

    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[ctx.device];
    GGML_ASSERT(ਖੰਡ_ਮਾਪ_3 >= SYCL_ਸਿਖਰ_K_RADIX_BUCKETS);

    ਅਚਲ size_t state_ਸ਼ਬਦ = (size_t) nrows * SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਸ਼ਬਦ;
    ggml_sycl_pool_alloc<uint32_t> state_alloc(ctx.pool(), state_ਸ਼ਬਦ);
    uint32_t * state = state_alloc.get();

    // Zero histogram, done counter and both emit counters. prefix/mask/need are seeded by
    // the first pass, which ignores the stored values.
    // The queue is in-order, so the passes below are already ordered after this fill.
    SYCL_CHECK(CHECK_TRY_ERROR(main_ਧਾਰਾ->memset(state, 0, state_ਸ਼ਬਦ * ਆਕਾਰ(uint32_t))));

    ਅਚਲ sycl::range<1> ਖੰਡ_dims(ਖੰਡ_ਮਾਪ_3);
    ਅਚਲ sycl::range<1> grid_dims(nrows * nparts);

    ਤਾਰਕਿਕ ਪਹਿਲਾ = ਸੱਚ;
    ਲਈ (ਪੂਰਨ ਖਿਸਕਾਓ = 32 - SYCL_ਸਿਖਰ_K_RADIX_ਬਿੱਟ; ਖਿਸਕਾਓ >= 0; ਖਿਸਕਾਓ -= SYCL_ਸਿਖਰ_K_RADIX_ਬਿੱਟ) {
        ਅਚਲ ਤਾਰਕਿਕ is_ਪਹਿਲਾ = ਪਹਿਲਾ;
        ਪਹਿਲਾ = ਝੂਠ;
        main_ਧਾਰਾ->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_ਸਿਖਰ_K_RADIX_HIST_ਮਾਪ + 4), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
                    ਅਚਲ ਪੂਰਨ g    = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);
                    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ  = g / nparts;
                    ਅਚਲ ਪੂਰਨ part = g % nparts;

                    ਸਿਖਰ_k_radix_ਵੰਡੋ_pass_f32(
                        src + (int64_t) ਪੰਕਤੀ * ncols,
                        state + (int64_t) ਪੰਕਤੀ * SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਸ਼ਬਦ,
                        (ਪੂਰਨ) ncols, k, ਖਿਸਕਾਓ, is_ਪਹਿਲਾ, part, nparts,
                        slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                        ਵਸਤੂ_ct1);
                });
        });
    }

    main_ਧਾਰਾ->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(8), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
                ਅਚਲ ਪੂਰਨ g    = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);
                ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ  = g / nparts;
                ਅਚਲ ਪੂਰਨ part = g % nparts;

                ਸਿਖਰ_k_radix_ਵੰਡੋ_emit_f32(
                    src + (int64_t) ਪੰਕਤੀ * ncols,
                    dst_indices + (int64_t) ਪੰਕਤੀ * k,
                    state + (int64_t) ਪੰਕਤੀ * SYCL_ਸਿਖਰ_K_RADIX_ਪੰਕਤੀ_ਸ਼ਬਦ,
                    (ਪੂਰਨ) ncols, k, part, nparts,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    ਵਸਤੂ_ct1);
            });
    });
}

ਖਾਲੀ ggml_sycl_ਸਿਖਰ_k_radix(
    ggml_backend_sycl_context & ctx,
    ਅਚਲ ਦਸ਼ਮਲਵ *   src,
    int32_t *       dst_indices,
    ਅਚਲ int64_t   ncols,
    ਅਚਲ int64_t   nrows,
    ਅਚਲ ਪੂਰਨ       k,
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ
) {
    ਅਚਲ ਪੂਰਨ nparts = ਸਿਖਰ_k_radix_ਵੰਡੋ_groups(ctx.device, ncols, nrows);
    ਜੇ (nparts > 1) {
        ਸਿਖਰ_k_radix_ਵੰਡੋ_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, nparts, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ {
        ਸਿਖਰ_k_radix_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, main_ਧਾਰਾ);
    }
}
