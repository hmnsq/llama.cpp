#شامل "topk-radix.hpp"

#شامل "common.hpp"

#شامل <algorithm>

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

ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_بٹ       = 8;
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_BUCKETS    = 1 << SYCL_چوٹی_K_RADIX_بٹ;
// Private histogram copies, interleaved per bucket so neighbouring lanes hit
// neighbouring banks. Lanes of one instruction spread over the copies, which is what
// bounds the atomic serialisation on tie-heavy rows.
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_HIST_COPIES = 8;
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_HIST_سائز   = SYCL_چوٹی_K_RADIX_BUCKETS * SYCL_چوٹی_K_RADIX_HIST_COPIES;
// Past the histogram: pivot digit, pivot bucket count, remaining need, then the two
// emit counters.
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_SLM_الفاظ   = SYCL_چوٹی_K_RADIX_HIST_سائز + 5;

// Larger float <=> larger key. The reference comparator is a plain float '>', under which
// -0.0 and +0.0 tie, so -0.0 is folded onto +0.0 first. NaN has no defined order in the
// reference (its comparator is not a strict weak order on NaN); here a positive NaN keys
// above +inf and a negative NaN below -inf, which at least makes the result deterministic.
ساکن اندرونی uint32_t چوٹی_k_radix_کلید(اعشاری f) {
    uint32_t u = sycl::ثنائی_ہندسہ_cast<uint32_t>(f);
    اگر (u == 0x80000000u) {
        u = 0u;
    }
    واپس (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

ساکن خالی چوٹی_k_radix_select_f32(
    ثابت اعشاری *   src,
    int32_t *       dst_idx,
    ثابت صحیح       ncols,
    ثابت صحیح       k,
    uint32_t *      slm,
    ثابت sycl::nd_شے<1> & شے_ct1
) {
    استعمال local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::حافظہ_scope::work_گروہ,
                                          sycl::access::پتہ_space::local_space>;

    ثابت صحیح tid        = شے_ct1.get_local_id(0);
    ثابت صحیح بلاک_سائز_3 = شے_ct1.get_local_range(0);

    uint32_t * hist     = slm;
    uint32_t * s_digit  = slm + SYCL_چوٹی_K_RADIX_HIST_سائز;
    uint32_t * s_بالٹی = slm + SYCL_چوٹی_K_RADIX_HIST_سائز + 1;
    uint32_t * s_need   = slm + SYCL_چوٹی_K_RADIX_HIST_سائز + 2;
    uint32_t * s_cnt_gt = slm + SYCL_چوٹی_K_RADIX_HIST_سائز + 3;
    uint32_t * s_cnt_eq = slm + SYCL_چوٹی_K_RADIX_HIST_سائز + 4;

    اگر (tid == 0) {
        *s_cnt_gt = 0;
        *s_cnt_eq = 0;
    }

    ثابت صحیح copy = tid & (SYCL_چوٹی_K_RADIX_HIST_COPIES - 1);

    uint32_t prefix = 0;   // digits fixed so far, in place
    uint32_t نقاب   = 0;   // which bits of prefix are fixed
    uint32_t need   = (uint32_t) k;

    برائے (صحیح سرکاؤ = 32 - SYCL_چوٹی_K_RADIX_بٹ; سرکاؤ >= 0; سرکاؤ -= SYCL_چوٹی_K_RADIX_بٹ) {
        برائے (صحیح i = tid; i < SYCL_چوٹی_K_RADIX_HIST_سائز; i += بلاک_سائز_3) {
            hist[i] = 0;
        }
        شے_ct1.barrier(sycl::access::fence_space::local_space);

        برائے (صحیح ستون = tid; ستون < ncols; ستون += بلاک_سائز_3) {
            ثابت uint32_t کلید = چوٹی_k_radix_کلید(src[ستون]);
            اگر ((کلید & نقاب) == prefix) {
                ثابت uint32_t bucket = (کلید >> سرکاؤ) & (SYCL_چوٹی_K_RADIX_BUCKETS - 1);
                local_atomic(hist[bucket * SYCL_چوٹی_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
            }
        }
        شے_ct1.barrier(sycl::access::fence_space::local_space);

        // Lane t takes bucket 255 - t, so an inclusive scan over lanes counts from the top
        // bucket downward. The pivot is the unique bucket whose cumulative count first
        // reaches need; the previous cumulative count is what the higher buckets contribute.
        uint32_t cnt = 0;
        اگر (tid < SYCL_چوٹی_K_RADIX_BUCKETS) {
            ثابت uint32_t * h = hist + (SYCL_چوٹی_K_RADIX_BUCKETS - 1 - tid) * SYCL_چوٹی_K_RADIX_HIST_COPIES;
            برائے (صحیح c = 0; c < SYCL_چوٹی_K_RADIX_HIST_COPIES; c++) {
                cnt += h[c];
            }
        }
        ثابت uint32_t incl = sycl::inclusive_جائزہ_over_گروہ(شے_ct1.get_گروہ(), cnt, sycl::plus<uint32_t>());

        اگر (tid < SYCL_چوٹی_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
            *s_digit = (uint32_t) (SYCL_چوٹی_K_RADIX_BUCKETS - 1 - tid);
            *s_بالٹی = cnt;
            *s_need   = need - (incl - cnt);
        }
        شے_ct1.barrier(sycl::access::fence_space::local_space);

        ثابت uint32_t digit      = *s_digit;
        ثابت uint32_t بالٹی_cnt = *s_بالٹی;
        need   = *s_need;
        prefix |= digit << سرکاؤ;
        نقاب   |= (uint32_t) (SYCL_چوٹی_K_RADIX_BUCKETS - 1) << سرکاؤ;

        // Every candidate in the pivot bucket is wanted: the remaining digits cannot
        // change the answer, and the masked emit below is exact as it stands.
        اگر (بالٹی_cnt == need) {
            توڑو;
        }
        // The next pass rewrites hist and s_*; the reads above must land first.
        شے_ct1.barrier(sycl::access::fence_space::local_space);
    }

    شے_ct1.barrier(sycl::access::fence_space::local_space);

    // Exactly k - need columns have (key & mask) > prefix; the first need of the pivot-equal
    // columns fill the tail. Both counters live in SLM since the whole row is this group.
    ثابت uint32_t بنیاد_eq = (uint32_t) k - need;

    برائے (صحیح ستون = tid; ستون < ncols; ستون += بلاک_سائز_3) {
        ثابت uint32_t kp = چوٹی_k_radix_کلید(src[ستون]) & نقاب;
        اگر (kp > prefix) {
            ثابت uint32_t pos = local_atomic(*s_cnt_gt).fetch_add(1u);
            dst_idx[pos] = ستون;
        } ورنہ اگر (kp == prefix) {
            ثابت uint32_t pos = local_atomic(*s_cnt_eq).fetch_add(1u);
            اگر (pos < need) {
                dst_idx[بنیاد_eq + pos] = ستون;
            }
        }
    }
}

ساکن خالی چوٹی_k_radix_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ثابت اعشاری * src,
    int32_t * dst_indices,
    ثابت int64_t ncols,
    ثابت int64_t nrows,
    ثابت صحیح k,
    dpct::قطار_ptr main_دھارا
) {
    GGML_ASSERT(ncols <= INT32_MAX);

    // One group per row; every pass is a strided sweep of the row, so lanes in flight is the
    // only lever, and the device's own limit is the answer -- there is nothing here that
    // wants a smaller group. Must still cover the 256 buckets for the scan step.
    ثابت صحیح بلاک_سائز_3 = ggml_sycl_info().max_work_گروہ_sizes[ctx.device];
    GGML_ASSERT(بلاک_سائز_3 >= SYCL_چوٹی_K_RADIX_BUCKETS);

    ثابت sycl::range<1> بلاک_dims(بلاک_سائز_3);
    ثابت sycl::range<1> grid_dims(nrows);

    main_دھارا->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_چوٹی_K_RADIX_SLM_الفاظ), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<1> شے_ct1) {
                ثابت صحیح صف = شے_ct1.get_گروہ(0);

                چوٹی_k_radix_select_f32(
                    src + (int64_t) صف * ncols, dst_indices + (int64_t) صف * k,
                    (صحیح) ncols, k,
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

ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_صف_مکمل   = SYCL_چوٹی_K_RADIX_BUCKETS + 0;
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_صف_سابقہ = SYCL_چوٹی_K_RADIX_BUCKETS + 1;
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_صف_نقاب   = SYCL_چوٹی_K_RADIX_BUCKETS + 2;
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_صف_NEED   = SYCL_چوٹی_K_RADIX_BUCKETS + 3;
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_صف_CNT_GT = SYCL_چوٹی_K_RADIX_BUCKETS + 4;
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_صف_CNT_EQ = SYCL_چوٹی_K_RADIX_BUCKETS + 5;
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_صف_الفاظ  = SYCL_چوٹی_K_RADIX_BUCKETS + 6;

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
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_GROUPS_فی_NSM = 2;
// Splitting trades one kernel for five. Below the width at which the single-group kernel
// runs longer than those four extra launches, it wins on its own; measured break-even on
// this device sits just under 64K columns.
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_MIN_بانٹو_ستون_تعداد = 65536;
// A partition thinner than this cannot keep a group's sweep busy.
ساکن constexpr صحیح SYCL_چوٹی_K_RADIX_MIN_PART_ستون_تعداد  = 4096;

ساکن صحیح چوٹی_k_radix_بانٹو_groups(ثابت صحیح device, ثابت int64_t ncols, ثابت int64_t nrows) {
    ثابت int64_t ہدف = (int64_t) SYCL_چوٹی_K_RADIX_GROUPS_فی_NSM * ggml_sycl_info().devices[device].nsm;

    // One group per row already, so a graph with rows enough to cover the device gains
    // nothing from splitting and would only pay the extra launches.
    اگر (ncols < SYCL_چوٹی_K_RADIX_MIN_بانٹو_ستون_تعداد || nrows >= ہدف) {
        واپس 1;
    }

    ثابت int64_t by_صفیں = ہدف / nrows;   // floor: never overshoot a row that is nearly covered
    ثابت int64_t by_ستون_تعداد = ncols / SYCL_چوٹی_K_RADIX_MIN_PART_ستون_تعداد;

    واپس (صحیح) std::max<int64_t>(1, std::min(by_صفیں, by_ستون_تعداد));
}

استعمال چوٹی_k_radix_gatomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                             sycl::حافظہ_scope::device,
                                             sycl::access::پتہ_space::global_space>;

ساکن خالی چوٹی_k_radix_بانٹو_pass_f32(
    ثابت اعشاری *   src,
    uint32_t *      state,
    ثابت صحیح       ncols,
    ثابت صحیح       k,
    ثابت صحیح       سرکاؤ,
    ثابت منطقی      پہلا,
    ثابت صحیح       part,
    ثابت صحیح       nparts,
    uint32_t *      slm,
    ثابت sycl::nd_شے<1> & شے_ct1
) {
    استعمال local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::حافظہ_scope::work_گروہ,
                                          sycl::access::پتہ_space::local_space>;

    ثابت صحیح tid        = شے_ct1.get_local_id(0);
    ثابت صحیح بلاک_سائز_3 = شے_ct1.get_local_range(0);

    uint32_t * hist   = slm;
    uint32_t * s_آخری = slm + SYCL_چوٹی_K_RADIX_HIST_سائز;
    uint32_t * s_صف  = slm + SYCL_چوٹی_K_RADIX_HIST_سائز + 1;   // prefix, mask, need

    // The previous launch is the barrier that publishes these, so a plain load is enough.
    // One lane reads them and the group takes them from SLM: a device-scope atomic load
    // is uncached here, and having every work-item issue three of them off the same
    // address costs more than the whole sweep below.
    اگر (tid == 0) {
        s_صف[0] = پہلا ? 0u : state[SYCL_چوٹی_K_RADIX_صف_سابقہ];
        s_صف[1] = پہلا ? 0u : state[SYCL_چوٹی_K_RADIX_صف_نقاب];
        s_صف[2] = پہلا ? (uint32_t) k : state[SYCL_چوٹی_K_RADIX_صف_NEED];
    }

    برائے (صحیح i = tid; i < SYCL_چوٹی_K_RADIX_HIST_سائز; i += بلاک_سائز_3) {
        hist[i] = 0;
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    ثابت uint32_t prefix = s_صف[0];
    ثابت uint32_t نقاب   = s_صف[1];
    ثابت uint32_t need   = s_صف[2];

    ثابت صحیح copy  = tid & (SYCL_چوٹی_K_RADIX_HIST_COPIES - 1);
    ثابت صحیح chunk = (ncols + nparts - 1) / nparts;
    ثابت صحیح col0  = part * chunk;
    ثابت صحیح col1  = std::min(ncols, col0 + chunk);

    برائے (صحیح ستون = col0 + tid; ستون < col1; ستون += بلاک_سائز_3) {
        ثابت uint32_t کلید = چوٹی_k_radix_کلید(src[ستون]);
        اگر ((کلید & نقاب) == prefix) {
            ثابت uint32_t bucket = (کلید >> سرکاؤ) & (SYCL_چوٹی_K_RADIX_BUCKETS - 1);
            local_atomic(hist[bucket * SYCL_چوٹی_K_RADIX_HIST_COPIES + copy]).fetch_add(1u);
        }
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    // One global atomic per bucket per group, not per element.
    برائے (صحیح b = tid; b < SYCL_چوٹی_K_RADIX_BUCKETS; b += بلاک_سائز_3) {
        uint32_t مجموع = 0;
        برائے (صحیح c = 0; c < SYCL_چوٹی_K_RADIX_HIST_COPIES; c++) {
            مجموع += hist[b * SYCL_چوٹی_K_RADIX_HIST_COPIES + c];
        }
        اگر (مجموع) {
            چوٹی_k_radix_gatomic(state[b]).fetch_add(مجموع);
        }
    }

    // Publish this group's bins, then claim the scan if this group is the row's last.
    // The group-wide barrier flushes the atomics above; only the claiming lane needs the
    // release, so the device-scope fence is paid once per group rather than per work-item.
    شے_ct1.barrier(sycl::access::fence_space::global_and_local);
    اگر (tid == 0) {
        sycl::atomic_fence(sycl::memory_order::release, sycl::حافظہ_scope::device);
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::حافظہ_scope::device,
                         sycl::access::پتہ_space::global_space> مکمل(state[SYCL_چوٹی_K_RADIX_صف_مکمل]);
        *s_آخری = (مکمل.fetch_add(1u) == (uint32_t) (nparts - 1)) ? 1u : 0u;
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    اگر (*s_آخری == 0u) {
        واپس;
    }
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::حافظہ_scope::device);

    // Lane t takes bucket 255 - t, so an inclusive scan counts down from the top bucket.
    uint32_t cnt = 0;
    اگر (tid < SYCL_چوٹی_K_RADIX_BUCKETS) {
        cnt = چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_BUCKETS - 1 - tid]).load();
    }
    ثابت uint32_t incl = sycl::inclusive_جائزہ_over_گروہ(شے_ct1.get_گروہ(), cnt, sycl::plus<uint32_t>());

    اگر (tid < SYCL_چوٹی_K_RADIX_BUCKETS && incl >= need && incl - cnt < need) {
        ثابت uint32_t digit = (uint32_t) (SYCL_چوٹی_K_RADIX_BUCKETS - 1 - tid);
        چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_صف_سابقہ]).store(prefix | (digit << سرکاؤ));
        چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_صف_نقاب]).store(
            نقاب | ((uint32_t) (SYCL_چوٹی_K_RADIX_BUCKETS - 1) << سرکاؤ));
        چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_صف_NEED]).store(need - (incl - cnt));
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    // Clear for the next pass; the next launch is the barrier that orders this.
    برائے (صحیح b = tid; b < SYCL_چوٹی_K_RADIX_BUCKETS; b += بلاک_سائز_3) {
        چوٹی_k_radix_gatomic(state[b]).store(0u);
    }
    اگر (tid == 0) {
        چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_صف_مکمل]).store(0u);
    }
}

ساکن خالی چوٹی_k_radix_بانٹو_emit_f32(
    ثابت اعشاری *   src,
    int32_t *       dst_idx,
    uint32_t *      state,
    ثابت صحیح       ncols,
    ثابت صحیح       k,
    ثابت صحیح       part,
    ثابت صحیح       nparts,
    uint32_t *      slm,
    ثابت sycl::nd_شے<1> & شے_ct1
) {
    استعمال local_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                          sycl::حافظہ_scope::work_گروہ,
                                          sycl::access::پتہ_space::local_space>;

    ثابت صحیح tid        = شے_ct1.get_local_id(0);
    ثابت صحیح بلاک_سائز_3 = شے_ct1.get_local_range(0);

    uint32_t * s_gt      = slm;
    uint32_t * s_eq      = slm + 1;
    uint32_t * s_بنیاد_gt = slm + 2;
    uint32_t * s_بنیاد_eq = slm + 3;

    uint32_t * s_صف = slm + 4;   // prefix, mask, need

    اگر (tid == 0) {
        *s_gt = 0;
        *s_eq = 0;
        s_صف[0] = state[SYCL_چوٹی_K_RADIX_صف_سابقہ];
        s_صف[1] = state[SYCL_چوٹی_K_RADIX_صف_نقاب];
        s_صف[2] = state[SYCL_چوٹی_K_RADIX_صف_NEED];
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    ثابت uint32_t prefix = s_صف[0];
    ثابت uint32_t نقاب   = s_صف[1];
    ثابت uint32_t need   = s_صف[2];

    // Exactly k - need columns beat the pivot; the first need pivot-equal ones fill the tail.
    ثابت uint32_t بنیاد_eq = (uint32_t) k - need;

    ثابت صحیح chunk = (ncols + nparts - 1) / nparts;
    ثابت صحیح col0  = part * chunk;
    ثابت صحیح col1  = std::min(ncols, col0 + chunk);

    // Counting first and reserving one range per group keeps the row's two counters out of
    // the inner loop: a per-element global atomic on a single address serialises the whole
    // emit, and at k in the thousands that alone outweighs every read the kernel does.
    برائے (صحیح ستون = col0 + tid; ستون < col1; ستون += بلاک_سائز_3) {
        ثابت uint32_t kp = چوٹی_k_radix_کلید(src[ستون]) & نقاب;
        اگر (kp > prefix) {
            local_atomic(*s_gt).fetch_add(1u);
        } ورنہ اگر (kp == prefix) {
            local_atomic(*s_eq).fetch_add(1u);
        }
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    اگر (tid == 0) {
        ثابت uint32_t n_gt = *s_gt;
        ثابت uint32_t n_eq = *s_eq;
        *s_بنیاد_gt = n_gt ? چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_صف_CNT_GT]).fetch_add(n_gt) : 0u;
        *s_بنیاد_eq = n_eq ? چوٹی_k_radix_gatomic(state[SYCL_چوٹی_K_RADIX_صف_CNT_EQ]).fetch_add(n_eq) : 0u;
        *s_gt = 0;
        *s_eq = 0;
    }
    شے_ct1.barrier(sycl::access::fence_space::local_space);

    ثابت uint32_t بنیاد_gt_g = *s_بنیاد_gt;
    ثابت uint32_t بنیاد_eq_g = *s_بنیاد_eq;

    برائے (صحیح ستون = col0 + tid; ستون < col1; ستون += بلاک_سائز_3) {
        ثابت uint32_t kp = چوٹی_k_radix_کلید(src[ستون]) & نقاب;
        اگر (kp > prefix) {
            dst_idx[بنیاد_gt_g + local_atomic(*s_gt).fetch_add(1u)] = ستون;
        } ورنہ اگر (kp == prefix) {
            ثابت uint32_t pos = بنیاد_eq_g + local_atomic(*s_eq).fetch_add(1u);
            اگر (pos < need) {
                dst_idx[بنیاد_eq + pos] = ستون;
            }
        }
    }
}

ساکن خالی چوٹی_k_radix_بانٹو_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ثابت اعشاری * src,
    int32_t * dst_indices,
    ثابت int64_t ncols,
    ثابت int64_t nrows,
    ثابت صحیح k,
    ثابت صحیح nparts,
    dpct::قطار_ptr main_دھارا
) {
    GGML_ASSERT(ncols <= INT32_MAX);
    GGML_ASSERT(nparts > 1);

    ثابت صحیح بلاک_سائز_3 = ggml_sycl_info().max_work_گروہ_sizes[ctx.device];
    GGML_ASSERT(بلاک_سائز_3 >= SYCL_چوٹی_K_RADIX_BUCKETS);

    ثابت size_t state_الفاظ = (size_t) nrows * SYCL_چوٹی_K_RADIX_صف_الفاظ;
    ggml_sycl_pool_alloc<uint32_t> state_alloc(ctx.pool(), state_الفاظ);
    uint32_t * state = state_alloc.get();

    // Zero histogram, done counter and both emit counters. prefix/mask/need are seeded by
    // the first pass, which ignores the stored values.
    // The queue is in-order, so the passes below are already ordered after this fill.
    SYCL_CHECK(CHECK_TRY_ERROR(main_دھارا->memset(state, 0, state_الفاظ * حجم(uint32_t))));

    ثابت sycl::range<1> بلاک_dims(بلاک_سائز_3);
    ثابت sycl::range<1> grid_dims(nrows * nparts);

    منطقی پہلا = سچ;
    برائے (صحیح سرکاؤ = 32 - SYCL_چوٹی_K_RADIX_بٹ; سرکاؤ >= 0; سرکاؤ -= SYCL_چوٹی_K_RADIX_بٹ) {
        ثابت منطقی is_پہلا = پہلا;
        پہلا = جھوٹ;
        main_دھارا->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(SYCL_چوٹی_K_RADIX_HIST_سائز + 4), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(grid_dims * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<1> شے_ct1) {
                    ثابت صحیح g    = شے_ct1.get_گروہ(0);
                    ثابت صحیح صف  = g / nparts;
                    ثابت صحیح part = g % nparts;

                    چوٹی_k_radix_بانٹو_pass_f32(
                        src + (int64_t) صف * ncols,
                        state + (int64_t) صف * SYCL_چوٹی_K_RADIX_صف_الفاظ,
                        (صحیح) ncols, k, سرکاؤ, is_پہلا, part, nparts,
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
                ثابت صحیح g    = شے_ct1.get_گروہ(0);
                ثابت صحیح صف  = g / nparts;
                ثابت صحیح part = g % nparts;

                چوٹی_k_radix_بانٹو_emit_f32(
                    src + (int64_t) صف * ncols,
                    dst_indices + (int64_t) صف * k,
                    state + (int64_t) صف * SYCL_چوٹی_K_RADIX_صف_الفاظ,
                    (صحیح) ncols, k, part, nparts,
                    slm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    شے_ct1);
            });
    });
}

خالی ggml_sycl_چوٹی_k_radix(
    ggml_backend_sycl_context & ctx,
    ثابت اعشاری *   src,
    int32_t *       dst_indices,
    ثابت int64_t   ncols,
    ثابت int64_t   nrows,
    ثابت صحیح       k,
    dpct::قطار_ptr main_دھارا
) {
    ثابت صحیح nparts = چوٹی_k_radix_بانٹو_groups(ctx.device, ncols, nrows);
    اگر (nparts > 1) {
        چوٹی_k_radix_بانٹو_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, nparts, main_دھارا);
    } ورنہ {
        چوٹی_k_radix_f32_sycl(ctx, src, dst_indices, ncols, nrows, k, main_دھارا);
    }
}
