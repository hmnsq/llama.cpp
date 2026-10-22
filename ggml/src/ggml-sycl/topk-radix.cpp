#ضمن "topk-radix.hpp"

#ضمن "common.hpp"

#ضمن <algorithm>

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

ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_بٹ       = 8;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_BUCKETS    = 1 << SYCL_چوٹی_K_RADIX_بٹ;
// Private histogram copies, interleaved per bucket so neighbouring lanes hit
// neighbouring banks. Lanes of one instruction spread over the copies, which is what
// bounds the atomic serialisation on tie-heavy rows.
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_HIST_COPIES = 8;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_HIST_ناپ   = SYCL_چوٹی_K_RADIX_BUCKETS * SYCL_چوٹی_K_RADIX_HIST_COPIES;
// Past the histogram: pivot digit, pivot bucket count, remaining need, then the two
// emit counters.
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_SLM_الفاظ   = SYCL_چوٹی_K_RADIX_HIST_ناپ + 5;

// Larger float <=> larger key. The reference comparator is a plain float '>', under which
// -0.0 and +0.0 tie, so -0.0 is folded onto +0.0 first. NaN has no defined order in the
// reference (its comparator is not a strict weak order on NaN); here a positive NaN keys
// above +inf and a negative NaN below -inf, which at least makes the result deterministic.
ساكن مضمن uint32_t چوٹی_k_radix_کلید(عائم f) {
    uint32_t u = sycl::ثنائی_ہندسہ_cast<uint32_t>(f);
    إذا (u == 0x80000000u) {
        u = 0u;
    }
    أرجع (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

ساكن فراغ چوٹی_k_radix_select_f32(
    ثابت عائم *   src,
    int32_t *       dst_idx,
    ثابت صحيح       ncols,
    ثابت صحيح       k,
    uint32_t *      slm,
    ثابت sycl::nd_شے<1> & شے_ct1
) {
    ورتوں local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::حافظہ_scope::work_گروہ,
                                          sycl::access::پتہ_space::local_space>;

    ثابت صحيح tid        = شے_ct1.get_local_id(0);
    ثابت صحيح بلاک_ناپ_3 = شے_ct1.get_local_range(0);

    uint32_t * hist     = slm;
    uint32_t * s_digit  = slm + SYCL_چوٹی_K_RADIX_HIST_ناپ;
    uint32_t * s_بالٹی = slm + SYCL_چوٹی_K_RADIX_HIST_ناپ + 1;
    uint32_t * s_need   = slm + SYCL_چوٹی_K_RADIX_HIST_ناپ + 2;
    uint32_t * s_cnt_gt = slm + SYCL_چوٹی_K_RADIX_HIST_ناپ + 3;
    uint32_t * s_cnt_eq = slm + SYCL_چوٹی_K_RADIX_HIST_ناپ + 4;

    إذا (tid == 0) {
        *s_cnt_gt = 0;
        *s_cnt_eq = 0;
    }

    ثابت صحيح copy = tid & (SYCL_چوٹی_K_RADIX_HIST_COPIES - 1);

    uint32_t prefix = 0;   // digits fixed so far, in place
    uint32_t نقاب   = 0;   // which bits of prefix are fixed
    uint32_t need   = (uint32_t) k;

    لكل (صحيح کھسکاؤ = 32 - SYCL_چوٹی_K_RADIX_بٹ; کھسکاؤ >= 0; کھسکاؤ -= SYCL_چوٹی_K_RADIX_بٹ) {
        لكل (صحيح i = tid; i < SYCL_چوٹی_K_RADIX_HIST_ناپ; i += بلاک_ناپ_3) {
            hist[i] = 0;
        }
        شے_ct1.barrier(sycl::access::fence_space::local_space);

        لكل (صحيح خانہ = tid; خانہ < ncols; خانہ += بلاک_ناپ_3) {
            ثابت uint32_t کلید = چوٹی_k_radix_کلید(src[خانہ]);
            إذا ((کلید & نقاب) == prefix) {
                ثابت uint32_t bucket = (کلید >> کھسکاؤ) & (SYCL_چوٹی_K_RADIX_BUCKETS - 1);
                local_atomic(hist[bucket * SYCL_چوٹی_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
            }
        }
        شے_ct1.barrier(sycl::access::fence_space::local_space);

        // Lane t takes bucket 255 - t, so an inclusive scan over lanes counts from the top
        // bucket downward. The pivot is the unique bucket whose cumulative count first
        // reaches need; the previous cumulative count is what the higher buckets contribute.
        uint32_t cnt = 0;
        إذا (tid < SYCL_چوٹی_K_RADIX_BUCKETS) {
            ثابت uint32_t * h = hist + (SYCL_چوٹی_K_RADIX_BUCKETS - 1 - tid) * SYCL_چوٹی_K_RADIX_HIST_COPIES;
            لكل (صحيح c = 0; c < SYCL_چوٹی_K_RADIX_HIST_COPIES; c++) {
                cnt += h[c];
            }
        }
        ثابت uint32_t incl = sycl::inclusive_جانچ_over_گروہ(شے_ct1.get_گروہ(), cnt, sycl::plus<uint32_t>());

        إذا (tid < SYCL_چوٹی_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
            *s_digit = (uint32_t) (SYCL_چوٹی_K_RADIX_BUCKETS - 1 - tid);
            *s_بالٹی = cnt;
            *s_need   = need - (incl - cnt);
        }
        شے_ct1.barrier(sycl::access::fence_space::local_space);

        ثابت uint32_t digit      = *s_digit;
        ثابت uint32_t بالٹی_cnt = *s_بالٹی;
        need   = *s_need;
        prefix |= digit << کھسکاؤ;
        نقاب   |= (uint32_t) (SYCL_چوٹی_K_RADIX_BUCKETS - 1) << کھسکاؤ;

        // Every candidate in the pivot bucket is wanted: the remaining digits cannot
        // change the answer, and the masked emit below is exact as it stands.
        إذا (بالٹی_cnt == need) {
            اكسر;
        }
        // The next pass rewrites hist and s_*; the reads above must land first.
        شے_ct1.barrier(sycl::access::fence_space::local_space);
    }

    شے_ct1.barrier(sycl::access::fence_space::local_space);

    // Exactly k - need columns have (key & mask) > prefix; the first need of the pivot-equal
    // columns fill the tail. Both counters live in SLM since the whole row is this group.
    ثابت uint32_t ادھار_eq = (uint32_t) k - need;

    لكل (صحيح خانہ = tid; خانہ < ncols; خانہ += بلاک_ناپ_3) {
        ثابت uint32_t kp = چوٹی_k_radix_کلید(src[خانہ]) & نقاب;
        إذا (kp > prefix) {
            ثابت uint32_t pos = local_atomic(*s_cnt_gt).fetch_add(1u);
            dst_idx[pos] = خانہ;
        } وإلا إذا (kp == prefix) {
            ثابت uint32_t pos = local_atomic(*s_cnt_eq).fetch_add(1u);
            إذا (pos < need) {
                dst_idx[ادھار_eq + pos] = خانہ;
            }
        }
    }
}

ساكن فراغ چوٹی_k_radix_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ثابت عائم * src,
    int32_t * dst_indices,
    ثابت int64_t ncols,
    ثابت int64_t nrows,
    ثابت صحيح k,
    dpct::قطار_ptr main_دھارا
) {
    GGML_ASSERT(ncols <= INT32_MAX);

    // One group per row; every pass is a strided sweep of the row, so lanes in flight is the
    // only lever, and the device's own limit is the answer -- there is nothing here that
    // wants a smaller group. Must still cover the 256 buckets for the scan step.
    ثابت صحيح بلاک_ناپ_3 = ggml_sycl_info().max_work_گروہ_sizes[ctx.device];
    GGML_ASSERT(بلاک_ناپ_3 >= SYCL_چوٹی_K_RADIX_BUCKETS);

    ثابت sycl::range<1> بلاک_dims(بلاک_ناپ_3);
    ثابت sycl::range<1> grid_dims(nrows);

    main_دھارا->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_چوٹی_K_RADIX_SLM_الفاظ), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<1> شے_ct1) {
                ثابت صحيح صف = شے_ct1.get_گروہ(0);

                چوٹی_k_radix_select_f32(
                    src + (int64_t) صف * ncols, dst_indices + (int64_t) صف * k,
                    (صحيح) ncols, k,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    شے_ct1);
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

ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_صف_پورا   = SYCL_چوٹی_K_RADIX_BUCKETS + 0;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_صف_اگیتر = SYCL_چوٹی_K_RADIX_BUCKETS + 1;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_صف_نقاب   = SYCL_چوٹی_K_RADIX_BUCKETS + 2;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_صف_NEED   = SYCL_چوٹی_K_RADIX_BUCKETS + 3;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_صف_CNT_GT = SYCL_چوٹی_K_RADIX_BUCKETS + 4;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_صف_CNT_EQ = SYCL_چوٹی_K_RADIX_BUCKETS + 5;
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_صف_الفاظ  = SYCL_چوٹی_K_RADIX_BUCKETS + 6;

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
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_GROUPS_فی_NSM = 2;
// Splitting trades one kernel for five. Below the width at which the single-group kernel
// runs longer than those four extra launches, it wins on its own; measured break-even on
// this device sits just under 64K columns.
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_MIN_ونڈو_خانے_گنتی = 65536;
// A partition thinner than this cannot keep a group's sweep busy.
ساكن constexpr صحيح SYCL_چوٹی_K_RADIX_MIN_PART_خانے_گنتی  = 4096;

ساكن صحيح چوٹی_k_radix_ونڈو_groups(ثابت صحيح device, ثابت int64_t ncols, ثابت int64_t nrows) {
    ثابت int64_t نشانہ = (int64_t) SYCL_چوٹی_K_RADIX_GROUPS_فی_NSM * ggml_sycl_info().devices[device].nsm;

    // One group per row already, so a graph with rows enough to cover the device gains
    // nothing from splitting and would only pay the extra launches.
    إذا (ncols < SYCL_چوٹی_K_RADIX_MIN_ونڈو_خانے_گنتی || nrows >= نشانہ) {
        أرجع 1;
    }

    ثابت int64_t by_صفاں = نشانہ / nrows;   // floor: never overshoot a row that is nearly covered
    ثابت int64_t by_خانے_گنتی = ncols / SYCL_چوٹی_K_RADIX_MIN_PART_خانے_گنتی;

    أرجع (صحيح) std::max<int64_t>(1, std::min(by_صفاں, by_خانے_گنتی));
}

ورتوں چوٹی_k_radix_gatomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                             sycl::حافظہ_scope::device,
                                             sycl::access::پتہ_space::global_space>;

ساكن فراغ چوٹی_k_radix_ونڈو_pass_f32(
    ثابت عائم *   src,
    uint32_t *      state,
    ثابت صحيح       ncols,
    ثابت صحيح       k,
    ثابت صحيح       کھسکاؤ,
    ثابت منطقي      پہلا,
    ثابت صحيح       part,
    ثابت صحيح       nparts,
    uint32_t *      slm,
    ثابت sycl::nd_شے<1> & شے_ct1
) {
    ورتوں local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::حافظہ_scope::work_گروہ,
                                          sycl::access::پتہ_space::local_space>;

    ثابت صحيح tid        = شے_ct1.get_local_id(0);
    ثابت صحيح بلاک_ناپ_3 = شے_ct1.get_local_range(0);

    uint32_t * hist   = slm;
    uint32_t * s_آخری = slm + SYCL_چوٹی_K_RADIX_HIST_ناپ;
    uint32_t * s_صف  = slm + SYCL_چوٹی_K_RADIX_HIST_ناپ + 1;   // prefix, mask, need

    // The previous launch is the barrier that publishes these, so a plain load is enough.
    // One lane reads them and the group takes them from SLM: a device-scope atomic load
    // is uncached here, and having every work-item issue three of them off the same
    // address costs more than the whole sweep below.
    إذا (tid == 0) {
        s_صف[0] = پہلا ? 0u : state[SYCL_چوٹی_K_RADIX_صف_اگیتر];
        s_صف[1] = پہلا ? 0u : state[SYCL_چوٹی_K_RADIX_صف_نقاب];
        s_صف[2] = پہلا ? (uint32_t) k : state[SYCL_چوٹی_K_RADIX_صف_NEED];
    }

    لكل (صحيح i = tid; i < SYCL_چوٹی_K_RADIX_HIST_ناپ; i += بلاک_ناپ_3) {
        hist[i] = 0;
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    ثابت uint32_t prefix = s_صف[0];
    ثابت uint32_t نقاب   = s_صف[1];
    ثابت uint32_t need   = s_صف[2];

    ثابت صحيح copy  = tid & (SYCL_چوٹی_K_RADIX_HIST_COPIES - 1);
    ثابت صحيح chunk = (ncols + nparts - 1) / nparts;
    ثابت صحيح col0  = part * chunk;
    ثابت صحيح col1  = std::min(ncols, col0 + chunk);

    لكل (صحيح خانہ = col0 + tid; خانہ < col1; خانہ += بلاک_ناپ_3) {
        ثابت uint32_t کلید = چوٹی_k_radix_کلید(src[خانہ]);
        إذا ((کلید & نقاب) == prefix) {
            ثابت uint32_t bucket = (کلید >> کھسکاؤ) & (SYCL_چوٹی_K_RADIX_BUCKETS - 1);
            local_atomic(hist[bucket * SYCL_چوٹی_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
        }
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    // One global atomic per bucket per group, not per element.
    لكل (صحيح b = tid; b < SYCL_چوٹی_K_RADIX_BUCKETS; b += بلاک_ناپ_3) {
        uint32_t جوڑ = 0;
        لكل (صحيح c = 0; c < SYCL_چوٹی_K_RADIX_HIST_COPIES; c++) {
            جوڑ += hist[b * SYCL_چوٹی_K_RADIX_HIST_COPIES + c];
        }
        إذا (جوڑ) {
            چوٹی_k_radix_gatomic(state[b]).fetch_add(جوڑ);
        }
    }

    // Publish this group's bins, then claim the scan if this group is the row's last.
    // The group-wide barrier flushes the atomics above; only the claiming lane needs the
    // release, so the device-scope fence is paid once per group rather than per work-item.
    شے_ct1.barrier(sycl::access::fence_space::global_and_local);
    إذا (tid == 0) {
        sycl::atomic_fence(sycl::memory_order::release, sycl::حافظہ_scope::device);
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::حافظہ_scope::device,
                         sycl::access::پتہ_space::global_space> پورا(state[SYCL_چوٹی_K_RADIX_صف_پورا]);
        *s_آخری = (پورا.fetch_add(1u) == (uint32_t) (nparts - 1)) ? 1u : 0u;
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    إذا (*s_آخری == 0u) {
        أرجع;
    }
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::حافظہ_scope::device);

    // Lane t takes bucket 255 - t, so an inclusive scan counts down from the top bucket.
    uint32_t cnt = 0;
    إذا (tid < SYCL_چوٹی_K_RADIX_BUCKETS) {
        cnt = چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_BUCKETS - 1 - tid]).load();
    }
    ثابت uint32_t incl = sycl::inclusive_جانچ_over_گروہ(شے_ct1.get_گروہ(), cnt, sycl::plus<uint32_t>());

    إذا (tid < SYCL_چوٹی_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
        ثابت uint32_t digit = (uint32_t) (SYCL_چوٹی_K_RADIX_BUCKETS - 1 - tid);
        چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_صف_اگیتر]).store(prefix | (digit << کھسکاؤ));
        چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_صف_نقاب]).store(
            نقاب | ((uint32_t) (SYCL_چوٹی_K_RADIX_BUCKETS - 1) << کھسکاؤ));
        چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_صف_NEED]).store(need - (incl - cnt));
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    // Clear for the next pass; the next launch is the barrier that orders this.
    لكل (صحيح b = tid; b < SYCL_چوٹی_K_RADIX_BUCKETS; b += بلاک_ناپ_3) {
        چوٹی_k_radix_gatomic(state[b]).store(0u);
    }
    إذا (tid == 0) {
        چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_صف_پورا]).store(0u);
    }
}

ساكن فراغ چوٹی_k_radix_ونڈو_emit_f32(
    ثابت عائم *   src,
    int32_t *       dst_idx,
    uint32_t *      state,
    ثابت صحيح       ncols,
    ثابت صحيح       k,
    ثابت صحيح       part,
    ثابت صحيح       nparts,
    uint32_t *      slm,
    ثابت sycl::nd_شے<1> & شے_ct1
) {
    ورتوں local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::حافظہ_scope::work_گروہ,
                                          sycl::access::پتہ_space::local_space>;

    ثابت صحيح tid        = شے_ct1.get_local_id(0);
    ثابت صحيح بلاک_ناپ_3 = شے_ct1.get_local_range(0);

    uint32_t * s_gt      = slm;
    uint32_t * s_eq      = slm + 1;
    uint32_t * s_ادھار_gt = slm + 2;
    uint32_t * s_ادھار_eq = slm + 3;

    uint32_t * s_صف = slm + 4;   // prefix, mask, need

    إذا (tid == 0) {
        *s_gt = 0;
        *s_eq = 0;
        s_صف[0] = state[SYCL_چوٹی_K_RADIX_صف_اگیتر];
        s_صف[1] = state[SYCL_چوٹی_K_RADIX_صف_نقاب];
        s_صف[2] = state[SYCL_چوٹی_K_RADIX_صف_NEED];
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    ثابت uint32_t prefix = s_صف[0];
    ثابت uint32_t نقاب   = s_صف[1];
    ثابت uint32_t need   = s_صف[2];

    // Exactly k - need columns beat the pivot; the first need pivot-equal ones fill the tail.
    ثابت uint32_t ادھار_eq = (uint32_t) k - need;

    ثابت صحيح chunk = (ncols + nparts - 1) / nparts;
    ثابت صحيح col0  = part * chunk;
    ثابت صحيح col1  = std::min(ncols, col0 + chunk);

    // Counting first and reserving one range per group keeps the row's two counters out of
    // the inner loop: a per-element global atomic on a single address serialises the whole
    // emit, and at k in the thousands that alone outweighs every read the kernel does.
    لكل (صحيح خانہ = col0 + tid; خانہ < col1; خانہ += بلاک_ناپ_3) {
        ثابت uint32_t kp = چوٹی_k_radix_کلید(src[خانہ]) & نقاب;
        إذا (kp > prefix) {
            local_atomic(*s_gt).fetch_add(1u);
        } وإلا إذا (kp == prefix) {
            local_atomic(*s_eq).fetch_add(1u);
        }
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    إذا (tid == 0) {
        ثابت uint32_t n_gt = *s_gt;
        ثابت uint32_t n_eq = *s_eq;
        *s_ادھار_gt = n_gt ? چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_صف_CNT_GT]).fetch_add(n_gt) : 0u;
        *s_ادھار_eq = n_eq ? چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_صف_CNT_EQ]).fetch_add(n_eq) : 0u;
        *s_gt = 0;
        *s_eq = 0;
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    ثابت uint32_t ادھار_gt_g = *s_ادھار_gt;
    ثابت uint32_t ادھار_eq_g = *s_ادھار_eq;

    لكل (صحيح خانہ = col0 + tid; خانہ < col1; خانہ += بلاک_ناپ_3) {
        ثابت uint32_t kp = چوٹی_k_radix_کلید(src[خانہ]) & نقاب;
        إذا (kp > prefix) {
            dst_idx[ادھار_gt_g + local_atomic(*s_gt).fetch_add(1u)] = خانہ;
        } وإلا إذا (kp == prefix) {
            ثابت uint32_t pos = ادھار_eq_g + local_atomic(*s_eq).fetch_add(1u);
            إذا (pos < need) {
                dst_idx[ادھار_eq + pos] = خانہ;
            }
        }
    }
}

ساكن فراغ چوٹی_k_radix_ونڈو_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ثابت عائم * src,
    int32_t * dst_indices,
    ثابت int64_t ncols,
    ثابت int64_t nrows,
    ثابت صحيح k,
    ثابت صحيح nparts,
    dpct::قطار_ptr main_دھارا
) {
    GGML_ASSERT(ncols <= INT32_MAX);
    GGML_ASSERT(nparts > 1);

    ثابت صحيح بلاک_ناپ_3 = ggml_sycl_info().max_work_گروہ_sizes[ctx.device];
    GGML_ASSERT(بلاک_ناپ_3 >= SYCL_چوٹی_K_RADIX_BUCKETS);

    ثابت size_t state_الفاظ = (size_t) nrows * SYCL_چوٹی_K_RADIX_صف_الفاظ;
    ggml_sycl_pool_alloc<uint32_t> state_alloc(ctx.pool(), state_الفاظ);
    uint32_t * state = state_alloc.get();

    // Zero histogram, done counter and both emit counters. prefix/mask/need are seeded by
    // the first pass, which ignores the stored values.
    // The queue is in-order, so the passes below are already ordered after this fill.
    SYCL_CHECK(CHECK_TRY_ERROR(main_دھارا->memset(state, 0, state_الفاظ * حجم(uint32_t))));

    ثابت sycl::range<1> بلاک_dims(بلاک_ناپ_3);
    ثابت sycl::range<1> grid_dims(nrows * nparts);

    منطقي پہلا = سچ;
    لكل (صحيح کھسکاؤ = 32 - SYCL_چوٹی_K_RADIX_بٹ; کھسکاؤ >= 0; کھسکاؤ -= SYCL_چوٹی_K_RADIX_بٹ) {
        ثابت منطقي is_پہلا = پہلا;
        پہلا = خطأ;
        main_دھارا->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_چوٹی_K_RADIX_HIST_ناپ + 4), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(grid_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<1> شے_ct1) {
                    ثابت صحيح g    = شے_ct1.get_گروہ(0);
                    ثابت صحيح صف  = g / nparts;
                    ثابت صحيح part = g % nparts;

                    چوٹی_k_radix_ونڈو_pass_f32(
                        src + (int64_t) صف * ncols,
                        state + (int64_t) صف * SYCL_چوٹی_K_RADIX_صف_الفاظ,
                        (صحيح) ncols, k, کھسکاؤ, is_پہلا, part, nparts,
                        slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                        شے_ct1);
                });
        });
    }

    main_دھارا->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(8), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<1> شے_ct1) {
                ثابت صحيح g    = شے_ct1.get_گروہ(0);
                ثابت صحيح صف  = g / nparts;
                ثابت صحيح part = g % nparts;

                چوٹی_k_radix_ونڈو_emit_f32(
                    src + (int64_t) صف * ncols,
                    dst_indices + (int64_t) صف * k,
                    state + (int64_t) صف * SYCL_چوٹی_K_RADIX_صف_الفاظ,
                    (صحيح) ncols, k, part, nparts,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    شے_ct1);
            });
    });
}

فراغ ggml_sycl_چوٹی_k_radix(
    ggml_backend_sycl_context & ctx,
    ثابت عائم *   src,
    int32_t *       dst_indices,
    ثابت int64_t   ncols,
    ثابت int64_t   nrows,
    ثابت صحيح       k,
    dpct::قطار_ptr main_دھارا
) {
    ثابت صحيح nparts = چوٹی_k_radix_ونڈو_groups(ctx.device, ncols, nrows);
    إذا (nparts > 1) {
        چوٹی_k_radix_ونڈو_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, nparts, main_دھارا);
    } وإلا {
        چوٹی_k_radix_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, main_دھارا);
    }
}
