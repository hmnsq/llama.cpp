#define _USE_MATH_DEFINES

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <inttypes.h>
#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <string.h>
#অন্তর্ভাব <time.h>

#অন্তর্ভাব <atomic>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <chrono>
#অন্তর্ভাব <mutex>
#অন্তর্ভাব <thread>
#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <stdexcept>
#অন্তর্ভাব <string>
#অন্তর্ভাব <sstream>
#অন্তর্ভাব <iomanip>
#অন্তর্ভাব <unordered_set>
#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <regex>
#অন্তর্ভাব <queue>
#অন্তর্ভাব <deque>
#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <initializer_list>

#ifndef M_LOG2E
#    define M_LOG2E 1.44269504088896340736
#endif

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#       define NOMINMAX
#    endif
#    include <windows.h>
#    include <sal.h>
#অন্যথা
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"

#অন্তর্ভাব <AEEStdErr.h>
#অন্তর্ভাব <dspqueue.h>
#অন্তর্ভাব <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-alloc.h"
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "ggml-hexagon.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-quants.h"
#অন্তর্ভাব "htp-opnode.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp/matmul-ops.h"
#অন্তর্ভাব "htp/binary-ops.h"
#অন্তর্ভাব "htp/flash-attn-ops.h"
#অন্তর্ভাব "htp/unary-ops.h"
#অন্তর্ভাব "htp/get-rows-ops.h"
#অন্তর্ভাব "htp/set-rows-ops.h"
#অন্তর্ভাব "htp/softmax-ops.h"
#অন্তর্ভাব "htp/pool-ops.h"
#অন্তর্ভাব "htp/rope-ops.h"
#অন্তর্ভাব "htp/ssm-conv.h"
#অন্তর্ভাব "htp/gated-delta-net-ops.h"
#অন্তর্ভাব "htp/argsort-ops.h"
#অন্তর্ভাব "htp/concat-ops.h"
#অন্তর্ভাব "htp/cpy-ops.h"
#অন্তর্ভাব "htp/im2col-ops.h"
#অন্তর্ভাব "htp_iface.h"
#অন্তর্ভাব "htp-drv.h"
#অন্তর্ভাব "htp-repack.h"

using intvec  = std::vector<পূর্ণাঙ্ক>;
using uintvec = std::vector<unsigned পূর্ণাঙ্ক>;
using u32vec  = std::vector<uint32_t>;

#define GGML_HEXAGON_MAX_SESSIONS          16

#define GGML_HEXAGON_FENCE_SLOT_SIZE       128

সংরচনা ggml_hexagon_device_config {
    পূর্ণাঙ্ক         physical_idx = 0;
    পূর্ণাঙ্ক         virtual_idx  = 0;
    পূর্ণাঙ্ক         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

স্থির ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

স্থির পূর্ণাঙ্ক    opt_arch    = 0; // autodetect
স্থির size_t opt_ndev    = 1;
স্থির size_t opt_nhvx    = 0; // use all
স্থির পূর্ণাঙ্ক    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
স্থির size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
স্থির পূর্ণাঙ্ক    opt_etm     = 0;
স্থির পূর্ণাঙ্ক    opt_verbose = 0;
স্থির পূর্ণাঙ্ক    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
স্থির তার্কিক   opt_hostbuf = অসত্য;
স্থির তার্কিক   opt_dma64   = অসত্য;
স্থির পূর্ণাঙ্ক    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start

স্থির ধ্রুব std::pair<ধ্রুব বর্ণ *, পূর্ণাঙ্ক> htp_power_mode_names[] = {
    { "low_power_saver",            HTP_POWER_MODE_নিচের_POWER_SAVER },
    { "power_saver",                HTP_POWER_MODE_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_MODE_উপরের_POWER_SAVER },
    { "low_balanced",               HTP_POWER_MODE_নিচের_BALANCED },
    { "balanced",                   HTP_POWER_MODE_BALANCED },
    { "high_performance",           HTP_POWER_MODE_উপরের_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_MODE_SUSTAINED_উপরের_PERFORMANCE },
    { "burst",                      HTP_POWER_MODE_BURST },
    { "default",                    HTP_POWER_MODE_BURST }, // flip this to change the default mode
};

স্থির পূর্ণাঙ্ক htp_power_mode_from_str(ধ্রুব বর্ণ * s) {
    প্রতি (auto & e : htp_power_mode_names) {
#যদি defined(_MSC_VER)
        যদি (_stricmp(s, e.first) == 0) {
#অন্যথা
        যদি (strcasecmp(s, e.first) == 0) {
#endif
            প্রত্যাগম e.second;
        }
    }
    প্রত্যাগম -1;
}

স্থির ধ্রুব বর্ণ * htp_power_mode_to_str(পূর্ণাঙ্ক mode) {
    প্রতি (auto & e : htp_power_mode_names) {
        যদি (e.second == mode) {
            প্রত্যাগম e.first;
        }
    }
    প্রত্যাগম "unknown";
}

স্থির size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
স্থির size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
স্থির size_t opt_mbuf_total  = 0;                        // total buffer space limit (0 = unconstrained)

স্থির পূর্ণাঙ্ক    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
স্থির পূর্ণাঙ্ক    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
স্থির পূর্ণাঙ্ক    opt_fa_head_split = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
স্থির পূর্ণাঙ্ক    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
স্থির পূর্ণাঙ্ক    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
স্থির পূর্ণাঙ্ক    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
স্থির u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

স্থির পূর্ণাঙ্ক opt_opbatch  = 1280; // max number of ops in a batch
স্থির পূর্ণাঙ্ক opt_opqueue  = 8;   // max number of pending batches
স্থির পূর্ণাঙ্ক opt_optrace  = 0;    // trace buffer size per thread (0 means default)
স্থির পূর্ণাঙ্ক opt_oppoll   = 0;    // polling for batch completions
স্থির পূর্ণাঙ্ক opt_opfusion  = 1;    // enable/disable op fusion
স্থির পূর্ণাঙ্ক opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

গণনা ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ADD = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ADD   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

স্থির অন্তঃস্থ তার্কিক ggml_hexagon_is_fusion_enabled(পূর্ণাঙ্ক flag) {
    যদি (opt_opfusion <= 0) প্রত্যাগম অসত্য;
    যদি (opt_opfusion == 1) প্রত্যাগম সত্য; // 1 enables all
    প্রত্যাগম (opt_opfusion & flag) != 0;
}

স্থির std::regex* opt_opfilter = রিক্ত; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    যদি (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

স্থির ধ্রুব বর্ণ * status_to_str(uint32_t status) {
    বিকল্প (status) {
        পক্ষ HTP_STATUS_OK:
            প্রত্যাগম "OK";
        পক্ষ HTP_STATUS_NO_SUPPORT:
            প্রত্যাগম "NO-SUPPORT";
        পক্ষ HTP_STATUS_INVAL_PARAMS:
            প্রত্যাগম "INVAL-PARAMS";
        পক্ষ HTP_STATUS_VTCM_TOO_SMALL:
            প্রত্যাগম "VTCM-TOO-SMALL";
        পক্ষ HTP_STATUS_INTERNAL_ERR:
            প্রত্যাগম "INTERNAL-ERROR";
        স্বভাব:
            প্রত্যাগম "UNKNOWN";
    }
}

// ** debug helpers

স্থির শূন্য ggml_hexagon_dump_op_exec(ধ্রুব std::string &sess_name, ধ্রুব htp_opnode & গ্রন্থি, ধ্রুব uint32_t req_flags) {
    যদি (!opt_verbose) প্রত্যাগম;

    htp_opformat fmt(গ্রন্থি);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                গ্রন্থি.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

স্থির শূন্য ggml_hexagon_dump_op_supp(ধ্রুব std::string &sess_name, ধ্রুব সংরচনা ggml_tensor * op, তার্কিক supp) {
    যদি (!opt_verbose) প্রত্যাগম;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

স্থির ধ্রুব বর্ণ * htp_event_name(uint16_t id) {
    বিকল্প (id) {
        পক্ষ HTP_TRACE_EVT_DMA:            প্রত্যাগম "DMA";
        পক্ষ HTP_TRACE_EVT_HVX_COMP:       প্রত্যাগম "HVX_COMP";
        পক্ষ HTP_TRACE_EVT_HVX_A_QUANT:    প্রত্যাগম "HVX_A_QUANT";
        পক্ষ HTP_TRACE_EVT_HVX_A_PREP:     প্রত্যাগম "HVX_A_PREP";
        পক্ষ HTP_TRACE_EVT_HVX_W_DEQUANT:  প্রত্যাগম "HVX_W_DEQUANT";
        পক্ষ HTP_TRACE_EVT_HVX_W_PREP:     প্রত্যাগম "HVX_W_PREP";
        পক্ষ HTP_TRACE_EVT_HVX_O_PROC:     প্রত্যাগম "HVX_O_PROC";
        পক্ষ HTP_TRACE_EVT_HVX_FA_QK:      প্রত্যাগম "HVX_QK_FA";
        পক্ষ HTP_TRACE_EVT_HVX_FA_SFM:     প্রত্যাগম "HVX_SFM_FA";
        পক্ষ HTP_TRACE_EVT_HVX_FA_Q_PREP:  প্রত্যাগম "HVX_Q_PREP";
        পক্ষ HTP_TRACE_EVT_HVX_FA_K_PREP:  প্রত্যাগম "HVX_K_PREP";
        পক্ষ HTP_TRACE_EVT_HVX_FA_V_PREP:  প্রত্যাগম "HVX_V_PREP";
        পক্ষ HTP_TRACE_EVT_HVX_GDN_PREP:   প্রত্যাগম "HVX_GDN_PREP";
        পক্ষ HTP_TRACE_EVT_HVX_GDN_SOLVE:  প্রত্যাগম "HVX_GDN_SOLVE";
        পক্ষ HTP_TRACE_EVT_HVX_GDN_V_PREP: প্রত্যাগম "HVX_GDN_V_PREP";
        পক্ষ HTP_TRACE_EVT_HVX_GDN_D_PREP: প্রত্যাগম "HVX_GDN_D_PREP";
        পক্ষ HTP_TRACE_EVT_HVX_GDN_OUT:    প্রত্যাগম "HVX_GDN_OUT";
        পক্ষ HTP_TRACE_EVT_HVX_GDN_STATE:  প্রত্যাগম "HVX_GDN_STATE";
        পক্ষ HTP_TRACE_EVT_HVX_GDN_REM:    প্রত্যাগম "HVX_GDN_REM";
        পক্ষ HTP_TRACE_EVT_HMX_COMP:       প্রত্যাগম "HMX_COMP";
        পক্ষ HTP_TRACE_EVT_L2FLUSH:        প্রত্যাগম "L2FLUSH";
        পক্ষ HTP_TRACE_EVT_INIT:           প্রত্যাগম "INIT";
        পক্ষ HTP_TRACE_EVT_BUFF:           প্রত্যাগম "BUFF";
        পক্ষ HTP_TRACE_EVT_FENCE:          প্রত্যাগম "FENCE";
        স্বভাব:                           প্রত্যাগম "UNKNOWN";
    }
}

স্থির শূন্য ggml_hexagon_dump_op_prof(ধ্রুব std::string &sess_name, ধ্রুব htp_opnode & গ্রন্থি, ধ্রুব htp_prof_desc & pd) {
    যদি (!opt_profile) প্রত্যাগম;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    ধ্রুব uint32_t * pmu = pd.pmu;

    বর্ণ pmu_str[256] = "";
    যদি (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, পরিমাণ(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(গ্রন্থি);
    ভিন্ন mhz = op_usec > 0 ? (ভিন্ন) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            গ্রন্থি.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

স্থির শূন্য ggml_hexagon_dump_batch_prof(ধ্রুব std::string & sess_name, ধ্রুব htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    ভিন্ন batch_mhz = rsp.usecs > 0 ? (ভিন্ন) batch_cycles / rsp.usecs : 0.0f;

    বর্ণ evt_str[256] = "----";
    যদি (opt_profile == 3) {
        snprintf(evt_str, পরিমাণ(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (unsigned long long) batch_cycles, (unsigned long long) rsp.cycles_start, batch_mhz);
}

স্থির শূন্য ggml_hexagon_dump_trace_events(ধ্রুব std::string & sess_name, ধ্রুব htp_opbatch_rsp & rsp,
                                           ধ্রুব htp_trace_desc * trace_events, uint32_t n_traces) {
    যদি (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        প্রতি (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        প্রতি (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            প্রতি (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                ধ্রুব auto & e = trace_events[t * n_traces + idx];
                তার্কিক is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

গণনা ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_WEIGHT    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

সংরচনা ggml_hexagon_session;

স্থির শূন্য ggml_hexagon_precompute_matmul_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_mm_kernel_params * kparams
);

স্থির শূন্য ggml_hexagon_precompute_fused_matmul_add_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * src2,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_mm_kernel_params * kparams
);

স্থির শূন্য ggml_hexagon_precompute_unary_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    uint32_t op,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_unary_kernel_params * kparams
);

স্থির তার্কিক ggml_hexagon_precompute_দ্বিমিক_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    uint32_t op,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_দ্বিমিক_kernel_params * kparams,
    ধ্রুব সংরচনা ggml_tensor * src2 = nullptr
);

স্থির শূন্য ggml_hexagon_precompute_get_rows_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_get_rows_kernel_params * kparams
);

স্থির শূন্য ggml_hexagon_precompute_set_rows_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_set_rows_kernel_params * kparams
);

স্থির শূন্য ggml_hexagon_precompute_softmax_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_softmax_kernel_params * kparams
);

স্থির শূন্য ggml_hexagon_precompute_rope_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_rope_kernel_params * kparams
);

স্থির শূন্য ggml_hexagon_precompute_ssm_conv_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_ssm_conv_kernel_params * kparams
);

স্থির শূন্য ggml_hexagon_precompute_gated_delta_net_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_gdn_kernel_params * kparams
);

স্থির শূন্য ggml_hexagon_precompute_sort_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    তার্কিক is_শীর্ষ_k,
    সংরচনা htp_sort_kernel_params * kparams
);

স্থির শূন্য ggml_hexagon_precompute_pool_2d_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_pool_2d_kernel_params * kparams,
    তার্কিক is_pool_1d
);

স্থির তার্কিক ggml_hexagon_precompute_concat_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_concat_kernel_params * kparams
);

স্থির তার্কিক ggml_hexagon_precompute_cpy_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_copy_kernel_params * kparams
);

স্থির তার্কিক ggml_hexagon_precompute_im2col_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_im2col_kernel_params * kparams
);

স্থির তার্কিক ggml_hexagon_precompute_im2col_3d_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_im2col_kernel_params * kparams
);

স্থির শূন্য ggml_hexagon_precompute_fused_mmnx_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * act,
    int32_t n_weights,
    সংরচনা htp_mm_kernel_params * kparams
);

স্থির শূন্য ggml_hexagon_precompute_fused_mmidnx_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * act,
    ধ্রুব সংরচনা ggml_tensor * dst,
    int32_t n_weights,
    সংরচনা htp_mm_kernel_params * kparams
);

স্থির তার্কিক ggml_hexagon_precompute_allreduce_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    তার্কিক has_add,
    তার্কিক is_row_bcast,
    তার্কিক is_shard_ok,
    সংরচনা htp_allreduce_kernel_params * kparams
);

স্থির তার্কিক ggml_hexagon_rows_stride(ধ্রুব int64_t * ne, ধ্রুব size_t * nb, size_t * stride);
স্থির তার্কিক ggml_hexagon_matmul_can_collapse(ধ্রুব সংরচনা ggml_tensor * src0, ধ্রুব সংরচনা ggml_tensor * src1, ধ্রুব সংরচনা ggml_tensor * dst);
স্থির ggml_tensor ggml_hexagon_tensor_collapse_rows(ধ্রুব সংরচনা ggml_tensor * t);

স্থির তার্কিক mm_is_hmx_eligible(ধ্রুব ggml_tensor * t);
স্থির htp_op_code op_remap_to_htp(ধ্রুব ggml_tensor * t);
স্থির তার্কিক is_supported_mul_mat_nx_kernel(ধ্রুব ggml_tensor * src0, ধ্রুব সংরচনা htp_mm_kernel_params * kparams);
স্থির তার্কিক is_supported_mul_mat_id_nx_kernel(ধ্রুব ggml_tensor * src0, ধ্রুব সংরচনা htp_mm_kernel_params * kparams);
স্থির তার্কিক is_mergeable_mul_mat(ধ্রুব ggml_tensor * t);
স্থির তার্কিক is_mergeable_mul_mat_pair(ধ্রুব ggml_tensor * n1, ধ্রুব ggml_tensor * n2);
স্থির তার্কিক is_mergeable_mul_mat_id(ধ্রুব ggml_tensor * t);
স্থির তার্কিক is_mergeable_mul_mat_id_pair(ধ্রুব ggml_tensor * n1, ধ্রুব ggml_tensor * n2);

// ** backend sessions

সংরচনা ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_size { 0 };
    uint32_t             flags { 0 };
};

স্থির অন্তঃস্থ তার্কিক ggml_hexagon_tensor_is_fuseable(ধ্রুব সংরচনা ggml_tensor * t) {
    যদি (!t->extra) প্রত্যাগম অসত্য;
    auto extra = (ধ্রুব সংরচনা ggml_hexagon_tensor_extra *) t->extra;
    প্রত্যাগম (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

স্থির অন্তঃস্থ তার্কিক ggml_hexagon_can_rt_repack(ধ্রুব সংরচনা ggml_tensor * t) {
    যদি (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
        প্রত্যাগম অসত্য;
    }
    যদি (t->buffer && ggml_backend_buft_is_host(t->buffer->buft)) {
        প্রত্যাগম অসত্য;
    }
    প্রত্যাগম opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

স্থির অন্তঃস্থ তার্কিক ggml_hexagon_tensor_is_tiled(ধ্রুব সংরচনা ggml_tensor * t) {
    ধ্রুব auto * extra = (ধ্রুব ggml_hexagon_tensor_extra *) t->extra;
    যদি (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        প্রত্যাগম সত্য;
    }
    প্রত্যাগম ggml_hexagon_can_rt_repack(t);
}

স্থির অন্তঃস্থ তার্কিক ggml_hexagon_needs_repack(ধ্রুব সংরচনা ggml_tensor * t) {
    ধ্রুব auto * extra = (ধ্রুব ggml_hexagon_tensor_extra *) t->extra;
    যদি (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        প্রত্যাগম অসত্য;
    }
    প্রত্যাগম ggml_hexagon_can_rt_repack(t);
}

স্থির অন্তঃস্থ তার্কিক ggml_hexagon_tensors_overlap(ধ্রুব সংরচনা ggml_tensor * a, ধ্রুব সংরচনা ggml_tensor * b) {
    ধ্রুব uintptr_t a0 = (uintptr_t) a->data;
    ধ্রুব uintptr_t b0 = (uintptr_t) b->data;
    ধ্রুব uintptr_t a1 = a0 + ggml_nbytes(a);
    ধ্রুব uintptr_t b1 = b0 + ggml_nbytes(b);

    প্রত্যাগম a0 < b1 && b0 < a1;
}

স্থির অন্তঃস্থ তার্কিক ggml_hexagon_can_row_partition(ধ্রুব সংরচনা ggml_tensor * t) {
    যদি (t->ne[1] > 1 && (t->nb[1] & 127) != 0) প্রত্যাগম অসত্য;
    যদি (t->ne[2] > 1 && (t->nb[2] & 127) != 0) প্রত্যাগম অসত্য;
    যদি (t->ne[3] > 1 && (t->nb[3] & 127) != 0) প্রত্যাগম অসত্য;
    প্রত্যাগম সত্য;
}

সংরচনা htp_opnode;

সংরচনা ggml_hexagon_opbatch;
সংরচনা ggml_hexagon_opqueue;
সংরচনা ggml_hexagon_shared_buffer;
সংরচনা ggml_hexagon_fence_buffer;
সংরচনা ggml_hexagon_session;
সংরচনা ggml_backend_hexagon_device_context;

সংরচনা ggml_hexagon_mdev_group {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

সংরচনা ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    চঞ্চল uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

সংরচনা ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    চঞ্চল uint32_t *    fence_slot   = nullptr;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

সংরচনা ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    পূর্ণাঙ্ক              phys_idx;
    পূর্ণাঙ্ক              virt_idx;
    তার্কিক             valid_session;
    তার্কিক             valid_handle;
    তার্কিক             valid_queue;
    তার্কিক             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<পূর্ণাঙ্ক, std::unique_ptr<ggml_hexagon_shared_buffer>> cloned_buffers;
    std::unordered_set<ggml_hexagon_session *>                           virt_peers;
    std::unordered_set<ggml_hexagon_session *>                           phys_peers;

    uint32_t n_threads   = 0;
    uint32_t n_hvx       = 0;
    uint32_t n_hmx       = 0;
    uint64_t vtcm_size   = 0;
    size_t   max_vmem    = 0;
    uint32_t fence_seq   = 0;

    std::atomic<uint64_t> batch_req_seq{0};
    std::atomic<uint64_t> batch_rsp_seq{0};
    std::atomic<uint32_t> last_error{HTP_STATUS_OK};

    uint64_t                cached_uid = 0;
    std::vector<htp_opnode> cached_nodes;

    ggml_hexagon_mdev_group                mdev;
    ggml_backend_dev_t                     dev       = nullptr;
    ggml_backend_hexagon_device_context *  dev_ctx   = nullptr;
    ggml_hexagon_fence_buffer *            fence_buf = nullptr;

    ggml_hexagon_session(ধ্রুব ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_গুনতি = 0) noexcept(অসত্য);
    ~ggml_hexagon_session() noexcept(সত্য);

    ধ্রুব বর্ণ* c_name() ধ্রুব { প্রত্যাগম name.c_str(); }

    শূন্য allocate(ধ্রুব ggml_hexagon_device_config & config) noexcept(অসত্য);
    শূন্য release() noexcept(সত্য);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    শূন্য      free_fence(শূন্য * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, চঞ্চল uint32_t *> cpy_fence_slots;

    শূন্য enqueue_mdev_group();
    শূন্য enqueue_op(ধ্রুব htp_opnode & গ্রন্থি);
    শূন্য enqueue_cpy(ধ্রুব ggml_tensor * src, ggml_tensor * dst, ধ্রুব ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    শূন্য enqueue_fence(ধ্রুব ggml_tensor * sync_tensor, uint32_t fence_seq = 0, তার্কিক wait = সত্য);
    শূন্য enqueue_allreduce(ধ্রুব ggml_tensor * dst, ধ্রুব std::vector<ধ্রুব ggml_tensor *> & src_tensors,
                           ধ্রুব std::vector<ধ্রুব ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);

    শূন্য start_batch();
    শূন্য flush_sync(তার্কিক all = সত্য);
    শূন্য flush_async();
    শূন্য flush_batch(size_t min_ops = 1);
    শূন্য flush_peers();
    শূন্য flush_pending(তার্কিক all = সত্য);

    ggml_hexagon_shared_buffer * mmap_tensor(ধ্রুব ggml_tensor * t);
    তার্কিক clone_buffer(ধ্রুব ggml_hexagon_shared_buffer*);
    শূন্য release_buffer(ধ্রুব ggml_hexagon_shared_buffer*);
    শূন্য unclone_buffer(ধ্রুব ggml_hexagon_shared_buffer*);

    শূন্য add_peer(ggml_hexagon_session * peer) {
        যদি (এতত্->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } অন্যথা {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

সংরচনা ggml_backend_hexagon_device_context {
    পূর্ণাঙ্ক                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_buffer_type buffer_type       = {};
    ggml_backend_buffer_type host_buffer_type  = {};
    ggml_backend_buffer_type fence_buffer_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(পূর্ণাঙ্ক dev_id, ধ্রুব ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    ধ্রুব বর্ণ * c_name() ধ্রুব { প্রত্যাগম config.name.c_str(); }

    ggml_hexagon_session * session() {
        যদি (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        প্রত্যাগম sess.get();
    }
};

সংরচনা ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(ধ্রুব std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        এতত্->dev_ctx = dev_ctx;
        এতত্->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

সংরচনা ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    পূর্ণাঙ্ক       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        যদি (!base) {
            ক্ষিপ std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        যদি (fd < 0) {
            rpcmem_free(base);
            ক্ষিপ std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        এতত্->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        যদি (base) {
            rpcmem_free(base);
        }
    }
};

সংরচনা ggml_hexagon_shared_buffer {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    তার্কিক     mapped;
    তার্কিক     pinned;
    তার্কিক     extended;

    ধ্রুব বর্ণ * c_name() ধ্রুব { প্রত্যাগম sess->c_name(); }
    uint8_t *    base()   ধ্রুব { প্রত্যাগম mem ? mem->base : nullptr; }
    size_t       size()   ধ্রুব { প্রত্যাগম mem ? mem->size : 0;  }
    পূর্ণাঙ্ক          fd()     ধ্রুব { প্রত্যাগম mem ? mem->fd   : -1; }

    শূন্য mmap(তার্কিক extended = অসত্য) {
        যদি (!এতত্->mem)   প্রত্যাগম;
        যদি (এতত্->mapped) প্রত্যাগম;

        GGML_ASSERT(!এতত্->pinned || !extended);

        এতত্->extended = extended;
        fastrpc_map_flags flags = এতত্->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        পূর্ণাঙ্ক err = fastrpc_mmap(sess->domain_id, fd(), (শূন্য *) base(), 0, size(), flags);
        যদি (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (unsigned) err);
            ক্ষিপ std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (শূন্য *) base(), size(), fd(), pinned, extended);

        এতত্->mapped = সত্য;
    }

    শূন্য unmap() {
        যদি (!এতত্->mapped) প্রত্যাগম;

        যদি (!এতত্->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        যদি (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (শূন্য *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (শূন্য *) base(), size(), fd());

        এতত্->mapped = অসত্য;
    }

    শূন্য alloc(size_t size) {
        যদি (এতত্->mem) প্রত্যাগম;

        এতত্->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (শূন্য *) base(), এতত্->size(), fd(), (পূর্ণাঙ্ক) pinned);
        যদি (এতত্->pinned) {
            mmap();
        }
    }

    শূন্য মুক্ত() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (শূন্য *) base(), size(), fd());
        এতত্->mem  = nullptr;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, তার্কিক pinned = অসত্য) {
        এতত্->sess     = sess;
        এতত্->mapped   = অসত্য;
        এতত্->pinned   = pinned;
        এতত্->extended = অসত্য;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_offset = (size + 4095) & ~4095;
        size_t total_size   = guard_offset + 4096;
        যদি (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            total_size = (total_size + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(total_size);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, ধ্রুব ggml_hexagon_shared_buffer & other) {
        এতত্->sess   = sess;
        এতত্->mem    = other.mem;
        এতত্->mapped = অসত্য;
        এতত্->pinned = other.pinned;
        এতত্->extended = other.extended;
    }

    ~ggml_hexagon_shared_buffer() {
        মুক্ত();
        প্রতি (auto * extra : tensor_extra) {
            delete extra;
        }
    }
};

সংরচনা ggml_hexagon_fence_buffer : সার্বজনিক ggml_hexagon_shared_buffer {
    uint32_t              slot_গুনতি = 0;
    uint32_t              slot_head  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_buffer   backend_buffer{};

    ggml_hexagon_fence_buffer(ggml_hexagon_session * sess, ggml_backend_buffer_type_t buft, size_t size)
        : ggml_hexagon_shared_buffer(sess, size, অসত্য /* pinned */),
          slot_গুনতি(size / GGML_HEXAGON_FENCE_SLOT_SIZE),
          slot_head(0) {
        backend_buffer.buft    = buft;
        backend_buffer.context = static_cast<ggml_hexagon_shared_buffer *>(এতত্);
        backend_buffer.size    = size;
        mmap(অসত্য);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        যদি (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        } অন্যথা যদি (slot_head + n_slots <= slot_গুনতি) {
            uint32_t slot = slot_head;
            slot_head += n_slots;
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        }
        যদি (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_SLOT_SIZE);
        }
        প্রত্যাগম ptr;
    }

    শূন্য free_slot(শূন্য * ptr, uint32_t n_slots = 1) {
        যদি (!ptr) প্রত্যাগম;
        uint32_t slot = ((uint8_t *) ptr - base()) / GGML_HEXAGON_FENCE_SLOT_SIZE;
        প্রতি (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

অন্তঃস্থ uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    প্রত্যাগম ptr;
}

অন্তঃস্থ শূন্য ggml_hexagon_session::free_fence(শূন্য * ptr, uint32_t n_slots) {
    যদি (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

স্থির শূন্য ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    sbuf->sess->unclone_buffer(sbuf);
    delete sbuf;
}

স্থির শূন্য * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    প্রত্যাগম sbuf->base();
}

স্থির গণনা ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (শূন্য *) sbuf->base(), tensor->data, ggml_nbytes(tensor));

    auto extra = নূতন ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    প্রত্যাগম GGML_STATUS_SUCCESS;
}

স্থির শূন্য ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   ধ্রুব শূন্য *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    যদি (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        যদি (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, data, offset, size, (পূর্ণাঙ্ক) buffer->usage, extra->flags);

    যদি ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy((বর্ণ *) tensor->data + offset, data, size);
        প্রত্যাগম;
    }

    যদি (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        htp_repack_tensor_tiled(tensor, data, size);
        প্রত্যাগম;
    }

    যদি (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    memcpy(extra->shadow_buf.data() + offset, data, size);
    extra->shadow_size += size;

    যদি (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

স্থির শূন্য ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   ধ্রুব ggml_tensor *   tensor,
                                                   শূন্য *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, data, offset, size, (পূর্ণাঙ্ক) buffer->usage, extra->flags);

    যদি ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy(data, (ধ্রুব বর্ণ *) tensor->data + offset, size);
        প্রত্যাগম;
    }

    htp_unpack_tensor_tiled(data, tensor, offset, size);
}

স্থির তার্কিক ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   ধ্রুব সংরচনা ggml_tensor * src,
                                                   সংরচনা ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    প্রত্যাগম অসত্য;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

স্থির শূন্য ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                          ggml_tensor *         tensor,
                                                          ধ্রুব শূন্য *          data,
                                                          size_t                offset,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    যদি (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        যদি (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (পূর্ণাঙ্ক) buffer->usage, extra->flags);

    যদি ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        প্রতি (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (ধ্রুব uint8_t *) data + i * stride_data, size);
        }
        প্রত্যাগম;
    }

    যদি (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    প্রতি (size_t i = 0; i < n_copies; i++) {
        memcpy(extra->shadow_buf.data() + offset + i * stride_tensor, (ধ্রুব uint8_t *) data + i * stride_data, size);
    }
    extra->shadow_size += n_copies * size;

    যদি (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

স্থির শূন্য ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                      ধ্রুব ggml_tensor *   tensor,
                                                      শূন্য *                data,
                                                      size_t                offset,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (পূর্ণাঙ্ক) buffer->usage);

    যদি ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        প্রতি (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *)data + i * stride_data, (ধ্রুব uint8_t *)tensor->data + offset + i * stride_tensor, size);
        }
        প্রত্যাগম;
    }

    size_t temp_size      = n_copies > 0 ? (n_copies - 1) * stride_tensor + size : 0;
    size_t slice_size     = tensor->ne[1] * ggml_row_size(tensor->type, tensor->ne[0]);
    size_t slice_offset   = offset % slice_size;
    size_t row_size_bytes = ggml_row_size(tensor->type, tensor->ne[0]);

    GGML_ASSERT((slice_offset % row_size_bytes) == 0 && "offset must be aligned to row boundary");
    GGML_ASSERT((temp_size % row_size_bytes)    == 0 && "temp_size must be a multiple of row size");
    GGML_ASSERT((offset + temp_size) <= ggml_nbytes(tensor));

    std::vector<uint8_t> temp_buf(temp_size);
    htp_unpack_tensor_tiled(temp_buf.data(), tensor, offset, temp_size);

    প্রতি (size_t i = 0; i < n_copies; i++) {
        memcpy((uint8_t *) data + i * stride_data, temp_buf.data() + i * stride_tensor, size);
    }
}

স্থির শূন্য ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (শূন্য *) sbuf->base(), sbuf->size());
    memset(sbuf->base(), value, sbuf->size());
}

স্থির ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ রিক্ত,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ রিক্ত,
};

// ** backend buffer type

স্থির শূন্য ggml_backend_hexagon_host_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                        ggml_tensor *         tensor,
                                                        ধ্রুব শূন্য *          data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy((বর্ণ *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

স্থির শূন্য ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                        ধ্রুব ggml_tensor *   tensor,
                                                        শূন্য *                data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy(data, (ধ্রুব বর্ণ *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

স্থির ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ রিক্ত,
    /* .set_tensor      = */ ggml_backend_hexagon_host_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_buffer_get_tensor,
    /* .set_tensor_2d   = */ রিক্ত,
    /* .get_tensor_2d   = */ রিক্ত,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ রিক্ত,
};

// ** backend buffer type

স্থির ধ্রুব বর্ণ * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    প্রত্যাগম static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

স্থির ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    যদি (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        প্রত্যাগম nullptr;
    }
    যত্ন {
        ggml_hexagon_shared_buffer * sbuf = নূতন ggml_hexagon_shared_buffer(sess, size, অসত্য);
        প্রত্যাগম ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } গৃহাণ (ধ্রুব std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        প্রত্যাগম nullptr;
    }
}

স্থির ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    যদি (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        প্রত্যাগম nullptr;
    }
    যত্ন {
        ggml_hexagon_shared_buffer * sbuf = নূতন ggml_hexagon_shared_buffer(sess, size, অসত্য);
        প্রত্যাগম ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_host_buffer_interface, sbuf, size);
    } গৃহাণ (ধ্রুব std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        প্রত্যাগম nullptr;
    }
}

স্থির size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম 128;  // HVX alignment
    GGML_UNUSED(buft);
}

স্থির size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ধ্রুব সংরচনা ggml_tensor * t) {
    যদি (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_size = htp_tiled_row_size(t->type, ne0) * ne1 * ne2 * ne3;
        প্রত্যাগম std::max(tiled_size, ggml_nbytes(t));
    }
    প্রত্যাগম ggml_nbytes(t);
}

স্থির size_t parse_size(ধ্রুব বর্ণ * str, size_t default_unit = 1024 * 1024) {
    যদি (!str || str[0] == '\0') {
        প্রত্যাগম 0;
    }
    বর্ণ * end = রিক্ত;
    দ্বিগুণ val = strtod(str, &end);
    যদি (val < 0) {
        প্রত্যাগম 0;
    }
    যদি (end && *end) {
        যাবত্ (*end == ' ') end++;
        যদি (*end == 'k' || *end == 'K') {
            প্রত্যাগম (size_t) (val * 1024);
        } অন্যথা যদি (*end == 'm' || *end == 'M') {
            প্রত্যাগম (size_t) (val * 1024 * 1024);
        } অন্যথা যদি (*end == 'g' || *end == 'G') {
            প্রত্যাগম (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    প্রত্যাগম (size_t) (val * default_unit);
}

স্থির size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম opt_mbuf_dyn;
    GGML_UNUSED(buft);
}

স্থির তার্কিক ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম অসত্য;
    GGML_UNUSED(buft);
}

স্থির তার্কিক ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম সত্য;
    GGML_UNUSED(buft);
}

সংরচনা ggml_backend_hexagon_alloc_buffer_n_plan_item {
    size_t size;
    পূর্ণাঙ্ক    first;
    পূর্ণাঙ্ক    last;
};

using ggml_backend_hexagon_alloc_buffer_n_plan_t = std::vector<ggml_backend_hexagon_alloc_buffer_n_plan_item>;

স্থির ধ্রুব বর্ণ * ggml_hexagon_kv_layer_suffix(ধ্রুব সংরচনা ggml_tensor * t) {
    যদি (strncmp(t->name, "cache_", 6) != 0) {
        প্রত্যাগম রিক্ত;
    }
    ধ্রুব বর্ণ * p = strstr(t->name, "_l");
    যদি (!p || !isdigit((unsigned বর্ণ)p[2])) {
        প্রত্যাগম রিক্ত;
    }
    প্রত্যাগম p;
}

সংরচনা ggml_backend_hexagon_alloc_unit {
    size_t size;
    পূর্ণাঙ্ক    first;
    পূর্ণাঙ্ক    last;
};

স্থির ggml_backend_hexagon_alloc_buffer_n_plan_t ggml_backend_hexagon_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, সংরচনা ggml_tensor ** tensors, পূর্ণাঙ্ক n_tensors) {
    ggml_backend_hexagon_alloc_buffer_n_plan_t plan;

    ধ্রুব size_t alignment = ggml_backend_buft_get_alignment(buft);
    ধ্রুব size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    পূর্ণাঙ্ক i = 0;
    যাবত্ (i < n_tensors) {
        সংরচনা ggml_tensor * t = tensors[i];
        size_t unit_size = 0;
        পূর্ণাঙ্ক unit_first = i;
        পূর্ণাঙ্ক unit_last = i + 1;

        যদি (t->data == রিক্ত && t->view_src == রিক্ত) {
            unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        ধ্রুব বর্ণ * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        যাবত্ (unit_last < n_tensors) {
            সংরচনা ggml_tensor * next = tensors[unit_last];

            যদি (next->view_src != রিক্ত) {
                unit_last++;
                অনুবর্ত;
            }

            যদি (layer_suffix != রিক্ত) {
                ধ্রুব বর্ণ * next_suffix = ggml_hexagon_kv_layer_suffix(next);
                যদি (next_suffix != রিক্ত && strcmp(layer_suffix, next_suffix) == 0) {
                    যদি (next->data == রিক্ত) {
                        unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, next), alignment);
                    }
                    unit_last++;
                    অনুবর্ত;
                }
            }

            ভঙ্গ;
        }

        units.push_back({ unit_size, unit_first, unit_last });
        i = unit_last;
    }

    size_t cur_buf_size  = 0;
    পূর্ণাঙ্ক    cur_buf_first = 0;

    প্রতি (ধ্রুব auto & unit : units) {
        যদি (unit.size == 0) {
            অনুবর্ত;
        }

        যদি (cur_buf_size > 0 && (cur_buf_size + unit.size) > max_size) {
            plan.push_back({ cur_buf_size, cur_buf_first, unit.first });
            cur_buf_size  = 0;
            cur_buf_first = unit.first;
        }

        cur_buf_size += unit.size;
    }

    যদি (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, cur_buf_first, n_tensors });
    }

    প্রত্যাগম plan;
}

স্থির ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, সংরচনা ggml_tensor ** tensors, পূর্ণাঙ্ক n_tensors) {
    ধ্রুব ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    প্রতি (ধ্রুব auto & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        যদি (buffer == রিক্ত) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            প্রতি (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            প্রত্যাগম রিক্ত;
        }

        সংরচনা ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        সংরচনা ggml_tensor * t_failed = রিক্ত;
        প্রতি (পূর্ণাঙ্ক j = item.first; j < item.last; j++) {
            সংরচনা ggml_tensor * t = tensors[j];
            যদি (t->data == রিক্ত) {
                যদি (t->view_src == রিক্ত) {
                    যদি (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ভঙ্গ;
                    }
                } অন্যথা যদি (t->buffer == রিক্ত) {
                    যদি (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ভঙ্গ;
                    }
                }
            } অন্যথা {
                যদি (t->view_src != রিক্ত && t->buffer == রিক্ত) {
                    যদি (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ভঙ্গ;
                    }
                }
            }
        }
        যদি (t_failed != রিক্ত) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            প্রতি (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            প্রত্যাগম রিক্ত;
        }

        buffers.push_back(buffer);
    }

    যদি (buffers.empty()) {
        প্রত্যাগম রিক্ত;
    }

    যদি (buffers.size() == 1) {
        প্রত্যাগম buffers[0];
    }

    প্রত্যাগম ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

স্থির size_t ggml_backend_hexagon_buffer_type_get_alloc_size_n(
        ggml_backend_buffer_type_t buft, সংরচনা ggml_tensor ** tensors, পূর্ণাঙ্ক n_tensors) {
    ধ্রুব ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    প্রতি (ধ্রুব auto & item : plan) {
        total += item.size;
    }
    প্রত্যাগম total;
}

স্থির ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_buffer_type_is_host,
};

স্থির ggml_backend_buffer_type_i ggml_backend_hexagon_host_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_host_buffer_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(পূর্ণাঙ্ক dev_id, ধ্রুব ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    buffer_type.device  = dev;
    buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    buffer_type.context = নূতন ggml_backend_hexagon_buffer_type_context(config.name, এতত্);

    host_buffer_type.device  = dev;
    host_buffer_type.iface   = ggml_backend_hexagon_host_buffer_type_interface;
    host_buffer_type.context = নূতন ggml_backend_hexagon_buffer_type_context(config.name + "-HOST", এতত্);

    fence_buffer_type.device  = dev;
    fence_buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    fence_buffer_type.context = নূতন ggml_backend_hexagon_buffer_type_context(config.name + "-FENCE", এতত্);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(host_buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(fence_buffer_type.context);
}

স্থির তার্কিক ggml_backend_buffer_is_hexagon(ধ্রুব সংরচনা ggml_backend_buffer * b) {
    প্রত্যাগম b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

সংরচনা ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<পূর্ণাঙ্ক, পূর্ণাঙ্ক>                b_map; // buffer fd   to index
    std::unordered_map<ধ্রুব ggml_tensor*, পূর্ণাঙ্ক> t_map; // tensor ptr  to index
    std::unordered_multimap<শূন্য*, পূর্ণাঙ্ক>         d_map; // tensor data to index

    unsigned পূর্ণাঙ্ক n_bufs;     // num buffers in the batch
    unsigned পূর্ণাঙ্ক n_tens;     // num tensors ...
    unsigned পূর্ণাঙ্ক n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    unsigned পূর্ণাঙ্ক n_bufs_max;
    unsigned পূর্ণাঙ্ক n_tens_max;
    unsigned পূর্ণাঙ্ক n_ops_max;
    size_t       b_vmem_max;

    শূন্য reset() {
        n_bufs = 0;
        n_tens = 0;
        n_ops  = 0;
        b_vmem = 0;

        b_map.clear();
        t_map.clear();
        d_map.clear();
        ops.resize(n_ops_max);
    }

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_size, size_t max_vmem) {
        এতত্->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_size;
        n_tens_max = std::min<size_t>(n_ops_max + n_ops_max * HTP_OP_MAX_INPUTS, HTP_OP_MAX_TENSORS);

        b_vmem_max = max_vmem;

        ops.resize(n_ops_max);

        h_bufs.resize(n_bufs_max);
        h_tens.resize(n_tens_max);
        h_ops.resize(n_ops_max);

        b_map.reserve(n_bufs_max);
        t_map.reserve(n_tens_max);
        d_map.reserve(n_tens_max);

        GGML_LOG_INFO("ggml-hex: %s op batching: n-bufs %u n-tensors %u n-ops %u vmem %zu\n",
                sess->c_name(), n_bufs_max, n_tens_max, n_ops_max, b_vmem_max);

        reset();
    }

    তার্কিক empty() ধ্রুব { প্রত্যাগম n_ops == 0; }

    // add buffer and return its index
    পূর্ণাঙ্ক add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        auto it = b_map.find(sbuf->fd());
        যদি (it != b_map.end()) { প্রত্যাগম it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        পূর্ণাঙ্ক bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        যদি (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (শূন্য*) sbuf->base(), (size_t) b.size, b_vmem);

        প্রত্যাগম bi;
    }

    তার্কিক same_shape(ধ্রুব htp_tensor * h, ধ্রুব ggml_tensor * t) ধ্রুব {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;

        ধ্রুব তার্কিক is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ধ্রুব int64_t ne0_padded = hex_round_up(t->ne[0], 32);
        ধ্রুব int64_t ne1_padded = hex_round_up(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_row_size(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? পরিমাণ(ggml_fp16_t) : t->nb[0];
        গণনা ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        প্রত্যাগম (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    পূর্ণাঙ্ক add_tensor(ধ্রুব ggml_tensor * t) {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;
        auto sbuf  = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        প্রতি (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            যদি (same_shape(h, t)) { প্রত্যাগম it->second; }
        }

        // Lookup by tensor ptr
        auto it = t_map.find(t);
        যদি (it != t_map.end()) { প্রত্যাগম it->second; }

        // Add new tensor to the batch
        পূর্ণাঙ্ক ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->data, ti});

        uint64_t t_offset = (uint8_t *) t->data - sbuf->base();
        size_t   t_size   = ggml_nbytes(t);

        htp_tensor &h = h_tens[ti];
        h.bi    = add_buffer(sbuf);
        h.ti    = ti;
        h.data  = t_offset;
        h.type  = t->type;
        h.ne[0] = t->ne[0];
        h.ne[1] = t->ne[1];
        h.ne[2] = t->ne[2];
        h.ne[3] = t->ne[3];

        ধ্রুব তার্কিক is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        যদি (is_tiled) {
            যদি (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = পরিমাণ(ggml_fp16_t);
            } অন্যথা {
                h.nb[0] = t->nb[0];
            }
            ধ্রুব int64_t ne0_padded = hex_round_up(t->ne[0], 32);
            ধ্রুব int64_t ne1_padded = hex_round_up(t->ne[1], 32);
            h.nb[1] = htp_tiled_row_size(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_size  = h.size;
        } অন্যথা {
            h.size  = t_size;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        যদি ((extra->flags & GGML_HEXAGON_TENSOR_WEIGHT) != 0) {
            h.flags |= HTP_TENSOR_WEIGHT;
        }
        যদি ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        যদি ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (শূন্য*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        প্রত্যাগম ti;
    }

    তার্কিক fit_op(ধ্রুব htp_opnode & গ্রন্থি) ধ্রুব {
        যদি (n_ops >= n_ops_max) প্রত্যাগম অসত্য;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        পূর্ণাঙ্ক seen_bufs[HTP_OP_MAX_BUFS];
        পূর্ণাঙ্ক n_seen_bufs = 0;

        auto fit_tensor = [&](ধ্রুব ggml_tensor *t) {
            যদি (!t) প্রত্যাগম;
            যদি (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                পূর্ণাঙ্ক fd = sbuf->fd();
                যদি (!b_map.count(fd)) {
                    প্রতি (পূর্ণাঙ্ক i = 0; i < n_seen_bufs; i++) {
                        যদি (seen_bufs[i] == fd) প্রত্যাগম;
                    }
                    যদি (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    যদি (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        প্রতি (ধ্রুব auto * src : গ্রন্থি.get_inputs()) {
            fit_tensor(src);
        }
        প্রতি (ধ্রুব auto * output : গ্রন্থি.get_outputs()) {
            fit_tensor(output);
        }

        যদি ((extra_bufs + n_bufs) > n_bufs_max) প্রত্যাগম অসত্য;
        যদি ((extra_tens + n_tens) > n_tens_max) প্রত্যাগম অসত্য;
        যদি ((extra_vmem + b_vmem) > b_vmem_max) প্রত্যাগম অসত্য;

        প্রত্যাগম সত্য;
    }

    // assumes that fit_op() was called first and returned true
    শূন্য add_op(ধ্রুব htp_opnode & গ্রন্থি) {
        // Add new op

        unsigned পূর্ণাঙ্ক n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = গ্রন্থি;

        htp_op_desc &o = h_ops[n];
        memcpy(o.params,        গ্রন্থি.গ্রন্থি->op_params, পরিমাণ(গ্রন্থি.গ্রন্থি->op_params));
        memcpy(o.kernel_params, গ্রন্থি.kernel_params,   পরিমাণ(o.kernel_params));
        o.opcode = গ্রন্থি.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        auto inputs = গ্রন্থি.get_inputs();
        প্রতি (unsigned পূর্ণাঙ্ক i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        auto outputs = গ্রন্থি.get_outputs();
        প্রতি (unsigned পূর্ণাঙ্ক i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    শূন্য sort_buffers() {
        যদি (n_bufs <= 1) প্রত্যাগম;

        std::vector<পূর্ণাঙ্ক> order(n_bufs);
        প্রতি (unsigned পূর্ণাঙ্ক i = 0; i < n_bufs; i++) { order[i] = (পূর্ণাঙ্ক) i; }

        std::stable_sort(order.begin(), order.end(), [&](পূর্ণাঙ্ক a, পূর্ণাঙ্ক b) {
            প্রত্যাগম h_bufs[a].size > h_bufs[b].size;
        });

        তার্কিক already_sorted = সত্য;
        প্রতি (unsigned পূর্ণাঙ্ক i = 0; i < n_bufs; i++) {
            যদি (order[i] != (পূর্ণাঙ্ক) i) {
                already_sorted = অসত্য;
                ভঙ্গ;
            }
        }
        যদি (already_sorted) প্রত্যাগম;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        প্রতি (unsigned পূর্ণাঙ্ক new_bi = 0; new_bi < n_bufs; new_bi++) {
            পূর্ণাঙ্ক old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        প্রতি (unsigned পূর্ণাঙ্ক i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        প্রতি (unsigned পূর্ণাঙ্ক i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    শূন্য update_mdev_group(uint32_t mdev_idx) {
        যদি (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    তার্কিক try_fuse_common(std::initializer_list<ধ্রুব ggml_tensor *> tensors) ধ্রুব {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        পূর্ণাঙ্ক seen_bufs[HTP_OP_MAX_BUFS];
        পূর্ণাঙ্ক n_seen_bufs = 0;

        প্রতি (ধ্রুব auto * t : tensors) {
            যদি (!t || t_map.count(t)) {
                অনুবর্ত;
            }
            extra_tens++;
            auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            পূর্ণাঙ্ক fd = sbuf->fd();
            যদি (!b_map.count(fd)) {
                তার্কিক found = অসত্য;
                প্রতি (পূর্ণাঙ্ক i = 0; i < n_seen_bufs; i++) {
                    যদি (seen_bufs[i] == fd) {
                        found = সত্য;
                        ভঙ্গ;
                    }
                }
                যদি (!found) {
                    যদি (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    যদি (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        যদি ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            প্রত্যাগম অসত্য;
        }

        প্রত্যাগম সত্য;
    }

    তার্কিক try_fuse_common(ধ্রুব ggml_tensor * t1, ধ্রুব ggml_tensor * t2) ধ্রুব {
        প্রত্যাগম try_fuse_common({t1, t2});
    }

    তার্কিক try_fuse_allreduce_add(ধ্রুব htp_opnode & গ্রন্থি) {
        যদি (n_ops == 0 || opt_ar_select != 2) প্রত্যাগম অসত্য;
        যদি (গ্রন্থি.opcode != HTP_OP_ADD) প্রত্যাগম অসত্য;

        htp_opnode & last_গ্রন্থি = ops[n_ops - 1];
        যদি (last_গ্রন্থি.opcode != HTP_OP_ALLREDUCE) প্রত্যাগম অসত্য;

        auto * ar_kparams = (সংরচনা htp_allreduce_kernel_params *) last_গ্রন্থি.kernel_params;
        ধ্রুব uint32_t rank    = (uint32_t) ar_kparams->rank;
        ধ্রুব uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        ধ্রুব ggml_tensor * ar_local = last_গ্রন্থি.inputs[rank];
        ধ্রুব ggml_tensor * add_src0 = গ্রন্থি.src0();
        ধ্রুব ggml_tensor * add_src1 = গ্রন্থি.src1();
        ধ্রুব ggml_tensor * add_dst  = গ্রন্থি.dst();

        যদি (!ggml_hexagon_tensor_is_fuseable(ar_local)) প্রত্যাগম অসত্য;

        ধ্রুব ggml_tensor * res_tensor;
        যদি (add_src0 == ar_local || add_src0->data == ar_local->data) {
            res_tensor = add_src1;
        } অন্যথা যদি (add_src1 == ar_local || add_src1->data == ar_local->data) {
            res_tensor = add_src0;
        } অন্যথা {
            প্রত্যাগম অসত্য;
        }

        যদি (ar_local->type != res_tensor->type) প্রত্যাগম অসত্য;

        ধ্রুব তার্কিক is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        ধ্রুব তার্কিক is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        যদি (!is_same_shape && !is_row_bcast) প্রত্যাগম অসত্য;

        যদি (is_same_shape) {
            যদি (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                প্রত্যাগম অসত্য;
            }
            যদি (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                প্রত্যাগম অসত্য;
            }
        }
        যদি (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) {
            প্রত্যাগম অসত্য;
        }

        // scatter is only valid for in-place add within max outputs
        ধ্রুব তার্কিক is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

        সংরচনা htp_allreduce_kernel_params new_kparams;
        যদি (!ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, সত্য, is_row_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            প্রত্যাগম অসত্য;
        }

        ধ্রুব তার্কিক scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

        প্রতি (uint32_t r = 0; r < n_ranks; r++) {
            ধ্রুব ggml_tensor * ar_src = last_গ্রন্থি.inputs[r];
            যদি (!ggml_hexagon_tensors_overlap(add_dst, ar_src)) {
                অনুবর্ত;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            যদি (scatter_ok && r == rank) {
                অনুবর্ত;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            প্রত্যাগম অসত্য;
        }

        যদি (!try_fuse_common(res_tensor, add_dst)) {
            প্রত্যাগম অসত্য;
        }

        last_গ্রন্থি.opcode = HTP_OP_ALLREDUCE_ADD;
        last_গ্রন্থি.name   = "ALLREDUCE+ADD";
        last_গ্রন্থি.inputs.push_back(res_tensor);
        last_গ্রন্থি.outputs.clear();
        last_গ্রন্থি.outputs.push_back(add_dst);
        last_গ্রন্থি.fused.push_back(গ্রন্থি.গ্রন্থি);
        memcpy(last_গ্রন্থি.kernel_params, &new_kparams, পরিমাণ(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ADD;
        memcpy(o.kernel_params, &new_kparams, পরিমাণ(new_kparams));

        o.src[2 * n_ranks] = add_tensor(res_tensor);
        যদি (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            প্রতি (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            প্রতি (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } অন্যথা {
            o.dst[0] = add_tensor(add_dst);
            প্রতি (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (পূর্ণাঙ্ক) new_kparams.mode, (পূর্ণাঙ্ক) new_kparams.n_dsts);
        প্রত্যাগম সত্য;
    }

    তার্কিক try_fuse_rms_norm_mul(ধ্রুব htp_opnode & গ্রন্থি) {
        যদি (n_ops == 0) প্রত্যাগম অসত্য;
        যদি (গ্রন্থি.opcode != HTP_OP_MUL) প্রত্যাগম অসত্য;

        htp_opnode & last_গ্রন্থি = ops[n_ops - 1];
        যদি (last_গ্রন্থি.opcode != HTP_OP_RMS_NORM) প্রত্যাগম অসত্য;

        ধ্রুব ggml_tensor * mul_src0 = গ্রন্থি.src0();
        ধ্রুব ggml_tensor * mul_src1 = গ্রন্থি.src1();
        ধ্রুব ggml_tensor * rms_out  = last_গ্রন্থি.dst();

        যদি (!ggml_hexagon_tensor_is_fuseable(rms_out)) প্রত্যাগম অসত্য;

        ধ্রুব ggml_tensor * weight;
        যদি (mul_src0 == rms_out || mul_src0->data == rms_out->data) {
            weight = mul_src1;
        } অন্যথা যদি (mul_src1 == rms_out || mul_src1->data == rms_out->data) {
            weight = mul_src0;
        } অন্যথা {
            প্রত্যাগম অসত্য;
        }

        ধ্রুব ggml_tensor * src0 = last_গ্রন্থি.src0();

        যদি (src0->ne[0] != weight->ne[0] || src0->ne[0] != গ্রন্থি.dst()->ne[0]) {
            প্রত্যাগম অসত্য;
        }

        ধ্রুব তার্কিক is_row_bcast = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
        ধ্রুব তার্কিক is_same_shape = (src0->ne[0] == weight->ne[0] && src0->ne[1] == weight->ne[1] &&
                                    src0->ne[2] == weight->ne[2] && src0->ne[3] == weight->ne[3]);
        যদি (!is_row_bcast && !is_same_shape) প্রত্যাগম অসত্য;

        যদি (!ggml_are_same_shape(src0, গ্রন্থি.dst())) {
            প্রত্যাগম অসত্য;
        }
        যদি (ggml_is_contiguous(src0) != ggml_is_contiguous(গ্রন্থি.dst())) {
            প্রত্যাগম অসত্য;
        }

        সংরচনা htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, weight, গ্রন্থি.dst(), &new_kparams
        );

        যদি ((size_t) new_kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_size, sess->vtcm_size);
            প্রত্যাগম অসত্য;
        }

        যদি (!try_fuse_common(weight, গ্রন্থি.dst())) {
            প্রত্যাগম অসত্য;
        }

        last_গ্রন্থি.opcode = HTP_OP_RMS_NORM_MUL;
        last_গ্রন্থি.name   = "RMS_NORM+MUL";
        last_গ্রন্থি.inputs.clear();
        last_গ্রন্থি.inputs.push_back(src0);
        last_গ্রন্থি.inputs.push_back(weight);
        last_গ্রন্থি.outputs.clear();
        last_গ্রন্থি.outputs.push_back(গ্রন্থি.dst());
        last_গ্রন্থি.fused.push_back(গ্রন্থি.গ্রন্থি);
        memcpy(last_গ্রন্থি.kernel_params, &new_kparams, পরিমাণ(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        memcpy(o.kernel_params, &new_kparams, পরিমাণ(new_kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(weight);
        প্রতি (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(গ্রন্থি.dst());
        প্রতি (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        প্রত্যাগম সত্য;
    }

    তার্কিক try_fuse_mul_mat_add(ধ্রুব htp_opnode & গ্রন্থি) {
        যদি (n_ops == 0) প্রত্যাগম অসত্য;
        যদি (গ্রন্থি.opcode != HTP_OP_ADD) প্রত্যাগম অসত্য;

        htp_opnode & last_গ্রন্থি = ops[n_ops - 1];
        যদি (last_গ্রন্থি.opcode != HTP_OP_MUL_MAT) প্রত্যাগম অসত্য;

        ধ্রুব ggml_tensor * add_src0 = গ্রন্থি.src0();
        ধ্রুব ggml_tensor * add_src1 = গ্রন্থি.src1();
        ধ্রুব ggml_tensor * mm_out   = last_গ্রন্থি.dst();

        যদি (!ggml_hexagon_tensor_is_fuseable(mm_out)) প্রত্যাগম অসত্য;

        ধ্রুব ggml_tensor * src2;
        যদি (add_src0 == mm_out || add_src0->data == mm_out->data) {
            src2 = add_src1;
        } অন্যথা যদি (add_src1 == mm_out || add_src1->data == mm_out->data) {
            src2 = add_src0;
        } অন্যথা {
            প্রত্যাগম অসত্য;
        }

        ধ্রুব ggml_tensor * src0 = last_গ্রন্থি.src0();
        ধ্রুব ggml_tensor * src1 = last_গ্রন্থি.src1();

        যদি (src2->type != GGML_TYPE_F32) প্রত্যাগম অসত্য;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        যদি (!ggml_are_same_shape(mm_out, গ্রন্থি.dst())) প্রত্যাগম অসত্য;

        ধ্রুব সংরচনা htp_mm_kernel_params * orig_kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) last_গ্রন্থি.kernel_params;
        সংরচনা htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_add_params(sess, src0, src1, src2, গ্রন্থি.dst(), &kparams);
        যদি (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            প্রত্যাগম অসত্য;
        }

        যদি (orig_kparams->collapse != kparams.collapse) {
            প্রত্যাগম অসত্য;
        }

        ধ্রুব পূর্ণাঙ্ক src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        ধ্রুব তার্কিক can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        যদি (!can_fuse) প্রত্যাগম অসত্য;

        যদি ((size_t) kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
            প্রত্যাগম অসত্য;
        }

        যদি (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            যদি (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_threads < orig_kparams->n_act_threads) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_threads, kparams.n_act_threads);
                প্রত্যাগম অসত্য;
            }
        }

        যদি (!try_fuse_common(src2, গ্রন্থি.dst())) {
            প্রত্যাগম অসত্য;
        }

        last_গ্রন্থি.opcode = HTP_OP_MUL_MAT_ADD;
        last_গ্রন্থি.name   = "MUL_MAT+ADD";
        last_গ্রন্থি.inputs.clear();
        last_গ্রন্থি.inputs.push_back(src0);
        last_গ্রন্থি.inputs.push_back(src1);
        last_গ্রন্থি.inputs.push_back(src2);
        last_গ্রন্থি.outputs.clear();
        last_গ্রন্থি.outputs.push_back(গ্রন্থি.dst());
        last_গ্রন্থি.fused.push_back(গ্রন্থি.গ্রন্থি);
        memcpy(last_গ্রন্থি.kernel_params, &kparams, পরিমাণ(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ADD;
        memcpy(o.kernel_params, &kparams, পরিমাণ(kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(src1);
        o.src[2] = add_tensor(src2);
        প্রতি (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(গ্রন্থি.dst());
        প্রতি (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        প্রত্যাগম সত্য;
    }

    তার্কিক try_fuse_mul_mat_nx(ধ্রুব htp_opnode & গ্রন্থি) {
        যদি (n_ops == 0 || গ্রন্থি.opcode != HTP_OP_MUL_MAT) প্রত্যাগম অসত্য;
        যদি (!is_mergeable_mul_mat(গ্রন্থি.গ্রন্থি)) প্রত্যাগম অসত্য;

        ধ্রুব ggml_tensor * w_in = গ্রন্থি.src0();
        ধ্রুব ggml_tensor * x_in = গ্রন্থি.src1();
        ধ্রুব ggml_tensor * d_in = গ্রন্থি.dst();

        htp_opnode & last_গ্রন্থি = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        যদি (last_গ্রন্থি.opcode == HTP_OP_MUL_MAT_NX) {
            ধ্রুব uint32_t curr_n = (uint32_t) last_গ্রন্থি.outputs.size();
            যদি (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                প্রত্যাগম অসত্য;
            }

            ধ্রুব ggml_tensor * w0 = last_গ্রন্থি.inputs[0];
            ধ্রুব ggml_tensor * x  = last_গ্রন্থি.inputs[curr_n];

            যদি (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                প্রত্যাগম অসত্য;
            }
            যদি (!last_গ্রন্থি.fused.empty() && (mm_is_hmx_eligible(last_গ্রন্থি.fused[0]) != mm_is_hmx_eligible(গ্রন্থি.গ্রন্থি))) {
                প্রত্যাগম অসত্য;
            }

            ধ্রুব সংরচনা htp_mm_kernel_params * orig_kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) last_গ্রন্থি.kernel_params;
            ধ্রুব তার্কিক collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_in, x, d_in);
            যদি (orig_kparams->collapse && !collapse) {
                প্রত্যাগম অসত্য;
            }

            সংরচনা htp_mm_kernel_params kparams;
            যদি (collapse) {
                ধ্রুব ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } অন্যথা {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            যদি (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                প্রত্যাগম অসত্য;
            }
            যদি ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                প্রত্যাগম অসত্য;
            }

            যদি (!try_fuse_common(w_in, d_in)) {
                প্রত্যাগম অসত্য;
            }

            last_গ্রন্থি.inputs[curr_n] = w_in;
            last_গ্রন্থি.inputs.push_back(x);
            last_গ্রন্থি.outputs.push_back(d_in);
            last_গ্রন্থি.fused.push_back(গ্রন্থি.গ্রন্থি);
            memcpy(last_গ্রন্থি.kernel_params, &kparams, পরিমাণ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, পরিমাণ(kparams));

            প্রতি (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = add_tensor(last_গ্রন্থি.inputs[s]);
            }
            প্রতি (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            প্রতি (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_গ্রন্থি.outputs[d]);
            }
            প্রতি (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            প্রত্যাগম সত্য;
        }

        // Case 2: last_node is single MUL_MAT
        যদি (last_গ্রন্থি.opcode == HTP_OP_MUL_MAT) {
            যদি (!is_mergeable_mul_mat_pair(last_গ্রন্থি.গ্রন্থি, গ্রন্থি.গ্রন্থি)) {
                প্রত্যাগম অসত্য;
            }

            ধ্রুব ggml_tensor * w0 = last_গ্রন্থি.src0();
            ধ্রুব ggml_tensor * x  = last_গ্রন্থি.src1();
            ধ্রুব ggml_tensor * w1 = গ্রন্থি.src0();
            ধ্রুব ggml_tensor * dst_0 = last_গ্রন্থি.dst();
            ধ্রুব ggml_tensor * dst_1 = গ্রন্থি.dst();

            ধ্রুব সংরচনা htp_mm_kernel_params * orig_kparams = (ধ্রুব সংরচনা htp_mm_kernel_params *) last_গ্রন্থি.kernel_params;
            ধ্রুব তার্কিক collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            যদি (orig_kparams->collapse && !collapse) {
                প্রত্যাগম অসত্য;
            }

            সংরচনা htp_mm_kernel_params kparams;
            যদি (collapse) {
                ধ্রুব ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } অন্যথা {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            যদি (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                প্রত্যাগম অসত্য;
            }
            যদি ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                প্রত্যাগম অসত্য;
            }

            যদি (!try_fuse_common(w1, গ্রন্থি.dst())) {
                প্রত্যাগম অসত্য;
            }

            last_গ্রন্থি.opcode = HTP_OP_MUL_MAT_NX;
            last_গ্রন্থি.name   = "MUL_MAT_NX";
            last_গ্রন্থি.inputs.clear();
            last_গ্রন্থি.inputs.push_back(w0);
            last_গ্রন্থি.inputs.push_back(w1);
            last_গ্রন্থি.inputs.push_back(x);
            last_গ্রন্থি.outputs.clear();
            last_গ্রন্থি.outputs.push_back(dst_0);
            last_গ্রন্থি.outputs.push_back(dst_1);
            last_গ্রন্থি.fused.push_back(গ্রন্থি.গ্রন্থি);
            memcpy(last_গ্রন্থি.kernel_params, &kparams, পরিমাণ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            memcpy(o.kernel_params, &kparams, পরিমাণ(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            প্রতি (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            প্রতি (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            প্রত্যাগম সত্য;
        }

        প্রত্যাগম অসত্য;
    }

    তার্কিক try_fuse_mul_mat_id_nx(ধ্রুব htp_opnode & গ্রন্থি) {
        যদি (n_ops == 0 || গ্রন্থি.opcode != HTP_OP_MUL_MAT_ID) প্রত্যাগম অসত্য;
        যদি (!is_mergeable_mul_mat_id(গ্রন্থি.গ্রন্থি)) প্রত্যাগম অসত্য;

        ধ্রুব ggml_tensor * w_in   = গ্রন্থি.src0();
        ধ্রুব ggml_tensor * x_in   = গ্রন্থি.src1();
        ধ্রুব ggml_tensor * ids_in = গ্রন্থি.গ্রন্থি->src[2];
        ধ্রুব ggml_tensor * d_in   = গ্রন্থি.dst();

        htp_opnode & last_গ্রন্থি = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        যদি (last_গ্রন্থি.opcode == HTP_OP_MUL_MAT_ID_NX) {
            ধ্রুব uint32_t curr_n = (uint32_t) last_গ্রন্থি.outputs.size();
            যদি (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                প্রত্যাগম অসত্য;
            }

            ধ্রুব ggml_tensor * w0  = last_গ্রন্থি.inputs[0];
            ধ্রুব ggml_tensor * x   = last_গ্রন্থি.inputs[curr_n];
            ধ্রুব ggml_tensor * ids = last_গ্রন্থি.inputs[curr_n + 1];

            যদি (x_in != x || ids_in != ids || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                প্রত্যাগম অসত্য;
            }
            যদি (!last_গ্রন্থি.fused.empty() && (mm_is_hmx_eligible(last_গ্রন্থি.fused[0]) != mm_is_hmx_eligible(গ্রন্থি.গ্রন্থি))) {
                প্রত্যাগম অসত্য;
            }

            সংরচনা htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
            যদি (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                প্রত্যাগম অসত্য;
            }
            যদি ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                প্রত্যাগম অসত্য;
            }

            যদি (!try_fuse_common(w_in, d_in)) {
                প্রত্যাগম অসত্য;
            }

            last_গ্রন্থি.inputs[curr_n] = w_in;
            last_গ্রন্থি.inputs[curr_n + 1] = x;
            last_গ্রন্থি.inputs.push_back(ids);
            last_গ্রন্থি.outputs.push_back(d_in);
            last_গ্রন্থি.fused.push_back(গ্রন্থি.গ্রন্থি);
            memcpy(last_গ্রন্থি.kernel_params, &kparams, পরিমাণ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, পরিমাণ(kparams));

            প্রতি (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = add_tensor(last_গ্রন্থি.inputs[s]);
            }
            প্রতি (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            প্রতি (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_গ্রন্থি.outputs[d]);
            }
            প্রতি (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            প্রত্যাগম সত্য;
        }

        // Case 2: last_node is single MUL_MAT_ID
        যদি (last_গ্রন্থি.opcode == HTP_OP_MUL_MAT_ID) {
            যদি (!is_mergeable_mul_mat_id_pair(last_গ্রন্থি.গ্রন্থি, গ্রন্থি.গ্রন্থি)) {
                প্রত্যাগম অসত্য;
            }

            ধ্রুব ggml_tensor * w0  = last_গ্রন্থি.src0();
            ধ্রুব ggml_tensor * x   = last_গ্রন্থি.src1();
            ধ্রুব ggml_tensor * ids = last_গ্রন্থি.গ্রন্থি->src[2];
            ধ্রুব ggml_tensor * w1  = গ্রন্থি.src0();

            সংরচনা htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, গ্রন্থি.dst(), 2, &kparams);
            যদি (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                প্রত্যাগম অসত্য;
            }
            যদি ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                প্রত্যাগম অসত্য;
            }

            যদি (!try_fuse_common(w1, গ্রন্থি.dst())) {
                প্রত্যাগম অসত্য;
            }

            ধ্রুব ggml_tensor * dst_0 = last_গ্রন্থি.dst();
            ধ্রুব ggml_tensor * dst_1 = গ্রন্থি.dst();

            last_গ্রন্থি.opcode = HTP_OP_MUL_MAT_ID_NX;
            last_গ্রন্থি.name   = "MUL_MAT_ID_NX";
            last_গ্রন্থি.inputs.clear();
            last_গ্রন্থি.inputs.push_back(w0);
            last_গ্রন্থি.inputs.push_back(w1);
            last_গ্রন্থি.inputs.push_back(x);
            last_গ্রন্থি.inputs.push_back(ids);
            last_গ্রন্থি.outputs.clear();
            last_গ্রন্থি.outputs.push_back(dst_0);
            last_গ্রন্থি.outputs.push_back(dst_1);
            last_গ্রন্থি.fused.push_back(গ্রন্থি.গ্রন্থি);
            memcpy(last_গ্রন্থি.kernel_params, &kparams, পরিমাণ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            memcpy(o.kernel_params, &kparams, পরিমাণ(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            o.src[3] = add_tensor(ids);
            প্রতি (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            প্রতি (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            প্রত্যাগম সত্য;
        }

        প্রত্যাগম অসত্য;
    }

    তার্কিক try_fuse_gdn_cpy(ধ্রুব htp_opnode & গ্রন্থি) {
        যদি (n_ops == 0 || গ্রন্থি.opcode != HTP_OP_CPY) প্রত্যাগম অসত্য;

        htp_opnode & last_গ্রন্থি = ops[n_ops - 1];
        যদি (last_গ্রন্থি.opcode != HTP_OP_GATED_DELTA_NET) প্রত্যাগম অসত্য;
        যদি (last_গ্রন্থি.outputs.size() != 1) প্রত্যাগম অসত্য;

        ধ্রুব ggml_tensor * gdn_out  = last_গ্রন্থি.dst();
        ধ্রুব ggml_tensor * cpy_গ্রন্থি = গ্রন্থি.গ্রন্থি;
        ধ্রুব ggml_tensor * cpy_src  = গ্রন্থি.src0();
        ধ্রুব ggml_tensor * cpy_dst  = গ্রন্থি.dst();

        যদি (!cpy_src || !cpy_dst || !cpy_dst->data) প্রত্যাগম অসত্য;
        যদি (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) প্রত্যাগম অসত্য;
        যদি ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_গ্রন্থি->flags & GGML_TENSOR_FLAG_OUTPUT)) প্রত্যাগম অসত্য;

        ধ্রুব ggml_tensor * v = last_গ্রন্থি.গ্রন্থি->src[2];
        যদি (!v) প্রত্যাগম অসত্য;

        ধ্রুব int64_t S_v      = v->ne[0];
        ধ্রুব int64_t H        = v->ne[1];
        ধ্রুব int64_t n_tokens = v->ne[2];
        ধ্রুব int64_t n_seqs   = v->ne[3];
        ধ্রুব int64_t K        = ggml_get_op_params_i32(last_গ্রন্থি.গ্রন্থি, 0);
        ধ্রুব size_t  tail_off = (size_t) S_v * H * n_tokens * n_seqs * পরিমাণ(ভিন্ন);

        ধ্রুব int64_t D         = S_v * S_v * H;
        ধ্রুব int64_t n_written = std::min<int64_t>(n_tokens, K);

        যদি (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_out && cpy_src->view_src->data != gdn_out->data) ||
            cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
            প্রত্যাগম অসত্য;
        }

        যদি (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != পরিমাণ(ভিন্ন)) {
            প্রত্যাগম অসত্য;
        }
        যদি (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * পরিমাণ(ভিন্ন)) {
            প্রত্যাগম অসত্য;
        }
        যদি (n_written > 1) {
            যদি (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * পরিমাণ(ভিন্ন)) {
                প্রত্যাগম অসত্য;
            }
        }

        যদি (!try_fuse_common({cpy_dst})) {
            প্রত্যাগম অসত্য;
        }

        last_গ্রন্থি.name += "+CPY";
        last_গ্রন্থি.outputs.push_back(cpy_dst);
        last_গ্রন্থি.fused.push_back(গ্রন্থি.গ্রন্থি);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = add_tensor(cpy_dst);
        প্রতি (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        প্রত্যাগম সত্য;
    }

    তার্কিক try_fuse(ধ্রুব htp_opnode & গ্রন্থি) {
        যদি (!opt_opfusion) প্রত্যাগম অসত্য;
        যদি (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ADD) && try_fuse_allreduce_add(গ্রন্থি)) প্রত্যাগম সত্য;
        যদি (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(গ্রন্থি))  প্রত্যাগম সত্য;
        যদি (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ADD)   && try_fuse_mul_mat_add(গ্রন্থি))   প্রত্যাগম সত্য;
        যদি (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(গ্রন্থি))    প্রত্যাগম সত্য;
        যদি (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(গ্রন্থি)) প্রত্যাগম সত্য;
        যদি (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(গ্রন্থি))       প্রত্যাগম সত্য;
        প্রত্যাগম অসত্য;
    }
};

সংরচনা ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

সংরচনা ggml_hexagon_opqueue {
    // Shared buffer for storing batches
    ggml_hexagon_shared_buffer *shm_buf;
    size_t                      shm_blk_size;
    size_t                      depth;

    using opvec = std::vector<htp_opnode>;

    std::vector<opvec>          op_cache;       // per batch op cache
    std::vector<uint64_t>       start_usec;     // per batch start time

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, size_t batch_size, size_t depth) : depth(depth) {
        size_t n_bufs    = HTP_OP_MAX_BUFS;
        size_t n_ops     = batch_size;
        size_t n_tensors = n_ops * HTP_OP_MAX_OUTPUTS + n_ops * HTP_OP_MAX_INPUTS;

        size_t tr_size = 0;
        যদি (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * পরিমাণ(htp_trace_desc);
        }

        shm_blk_size = পরিমাণ(htp_buf_desc)  * n_bufs    +
                       পরিমাণ(htp_tensor)    * n_tensors +
                       পরিমাণ(htp_op_desc)   * n_ops     +
                       পরিমাণ(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = নূতন ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, সত্য /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        যদি (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size(), shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    size_t shm_size() ধ্রুব { প্রত্যাগম shm_buf ? shm_buf->size() : 0; }

    // push new batch
    তার্কিক ঠেলো(htp_opbatch_req& req, dspqueue_buffer& dbuf, ধ্রুব ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(পরিমাণ(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(পরিমাণ(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(পরিমাণ(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(পরিমাণ(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(পরিমাণ(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(পরিমাণ(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        যদি (seq - shm_buf->sess->batch_rsp_seq > depth) { প্রত্যাগম অসত্য; }

        ধ্রুব uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[slot]   = op_batch->ops;
        start_usec[slot] = ggml_time_us();

        ধ্রুব size_t b_size = পরিমাণ(htp_buf_desc)  * req.n_bufs;
        ধ্রুব size_t t_size = পরিমাণ(htp_tensor)    * req.n_tensors;
        ধ্রুব size_t o_size = পরিমাণ(htp_op_desc)   * req.n_ops;
        ধ্রুব size_t p_size = পরিমাণ(htp_prof_desc) * req.n_ops;

        size_t tr_size = 0;
        যদি (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * পরিমাণ(htp_trace_desc);
        } অন্যথা {
            req.n_traces = 0;
        }

        dbuf.ptr      = shm_buf->base() + ((size_t) slot * shm_blk_size);
        dbuf.fd       = shm_buf->fd();
        dbuf.flags    = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
        dbuf.offset   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->base();
        dbuf.size     = b_size + t_size + o_size + p_size + tr_size;

        GGML_ASSERT(dbuf.size <= shm_blk_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_size;
        uint8_t * t_ptr = m_ptr; m_ptr += t_size;
        uint8_t * o_ptr = m_ptr;

        memcpy(b_ptr, (শূন্য *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (শূন্য *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (শূন্য *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        যদি (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            প্রতি (unsigned পূর্ণাঙ্ক i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (শূন্য *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            প্রতি (unsigned পূর্ণাঙ্ক i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (unsigned long long) t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        প্রত্যাগম সত্য;
    }

    শূন্য বের_করো(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        ধ্রুব uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        ধ্রুব size_t b_size = পরিমাণ(htp_buf_desc)  * rsp.n_bufs;
        ধ্রুব size_t t_size = পরিমাণ(htp_tensor)    * rsp.n_tensors;
        ধ্রুব size_t o_size = পরিমাণ(htp_op_desc)   * rsp.n_ops;
        ধ্রুব size_t p_size = পরিমাণ(htp_prof_desc) * rsp.n_ops;

        size_t tr_size = 0;
        uint32_t n_traces = 0;
        যদি (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * পরিমাণ(htp_trace_desc);
        }

        ধ্রুব size_t m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        যদি (rsp.n_ops > 0) {
            auto & ops = op_cache[slot];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            ধ্রুব htp_prof_desc * pd = (ধ্রুব htp_prof_desc *) p_ptr;
            ধ্রুব htp_trace_desc * trace_events = nullptr;
            যদি (opt_profile == 3) {
                trace_events = (ধ্রুব htp_trace_desc *) (p_ptr + p_size);
            }

            যদি (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            প্রতি (uint32_t i = 0; i < rsp.n_ops; i++) {
                যদি (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            যদি (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

শূন্য ggml_hexagon_session::flush_peers() {
    auto vpeers = std::move(virt_peers);
    virt_peers.clear();
    প্রতি (auto * peer : vpeers) {
        peer->flush_sync();
    }

    auto ppeers = std::move(phys_peers);
    phys_peers.clear();
    প্রতি (auto * peer : ppeers) {
        peer->flush_async();
    }

    প্রতি (auto & sub : এতত্->mdev.sessions) {
        sub->flush_peers();
    }
}

শূন্য ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

শূন্য ggml_hexagon_session::flush_pending(তার্কিক all) {
    প্রতি (auto & sub : এতত্->mdev.sessions) {
        sub->flush_pending(all);
        যদি (sub->last_error > HTP_STATUS_OK) {
            এতত্->last_error = sub->last_error.load();
        }
    }

    যাবত্ (এতত্->batch_rsp_seq < এতত্->batch_req_seq) {
        সংরচনা htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        সংরচনা dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        ধ্রুব uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        পূর্ণাঙ্ক err = dspqueue_read(এতত্->queue, &flags, 1, &n_dbufs, &dbuf, পরিমাণ(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        যদি (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            অনুবর্ত;
        }

        যদি (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        যদি (rsp_size != পরিমাণ(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", এতত্->c_name(), rsp_size, n_dbufs);
        }

        যদি (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", এতত্->c_name(), status_to_str(rsp.status));
            এতত্->last_error = rsp.status;
            প্রতি (auto & sub : এতত্->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->বের_করো(rsp, dbuf);

        GGML_ASSERT(rsp.seq == এতত্->batch_rsp_seq + 1);
        এতত্->batch_rsp_seq = rsp.seq;

        যদি (!all) ভঙ্গ;
    }
}

শূন্য ggml_hexagon_session::flush_sync(তার্কিক all) {
    flush_async();
    flush_pending(all);
}

শূন্য ggml_hexagon_session::start_batch() {
    যদি (এতত্->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

শূন্য ggml_hexagon_session::flush_batch(size_t min_ops) {
    যদি (op_batch->n_ops < min_ops) { প্রত্যাগম; }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    ধ্রুব uint64_t seq = ++এতত্->batch_req_seq;

    op_batch->update_mdev_group(এতত্->mdev.idx);

    যদি (!op_queue->ঠেলো(req, dbuf, op_batch, seq)) {
        flush_pending(অসত্য);
        op_queue->ঠেলো(req, dbuf, op_batch, seq);
    }

    প্রতি (auto & sub : এতত্->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_buffer sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        যদি (!sub->op_queue->ঠেলো(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(অসত্য);
            sub->op_queue->ঠেলো(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        পূর্ণাঙ্ক err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, পরিমাণ(sub_req), (ধ্রুব uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        যদি (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (unsigned) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", এতত্->c_name(), dbuf.ptr, dbuf.size);

    পূর্ণাঙ্ক err = dspqueue_write(এতত্->queue, 0, 1, &dbuf, পরিমাণ(req), (ধ্রুব uint8_t*) &req, DSPQUEUE_TIMEOUT);
    যদি (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", এতত্->c_name(), (unsigned) err);
    }

    op_batch->reset();
}

শূন্য ggml_hexagon_session::enqueue_op(ধ্রুব htp_opnode & গ্রন্থি) {
    auto clone_tensor_buffer = [এতত্](ধ্রুব ggml_tensor * t) {
        auto sbuf = এতত্->mmap_tensor(t);
        যদি (!sbuf) প্রত্যাগম;
        যদি (sbuf->sess != এতত্) {
            এতত্->clone_buffer(sbuf);
        }
        প্রতি (auto & sub : এতত্->mdev.sessions) {
            sub->clone_buffer(sbuf);
        }
    };

    প্রতি (auto t : গ্রন্থি.get_inputs()) {
        clone_tensor_buffer(t);
    }
    প্রতি (auto t : গ্রন্থি.get_outputs()) {
        clone_tensor_buffer(t);
    }

    যদি (opt_opfusion && op_batch->try_fuse(গ্রন্থি)) {
        প্রত্যাগম;
    }

    যদি (!op_batch->fit_op(গ্রন্থি)) {
        flush_async();
    }

    যদি (op_batch->empty()) {
        start_batch();
    }

    যদি (!op_batch->fit_op(গ্রন্থি)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->add_op(গ্রন্থি);
}

শূন্য ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_গ্রন্থি(HTP_OP_MDEV_GROUP);

    uint8_t * fence_slot = এতত্->mdev_fence_slot;

    স্থির ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.buffer       = &এতত্->fence_buf->backend_buffer;
    dummy_t.extra        = &fence_extra;
    dummy_t.data         = (শূন্য *) fence_slot;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.ne[1]        = (int64_t) এতত্->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) এতত্->mdev.idx;

    ggml_tensor * গ্রন্থি = group_গ্রন্থি.add_dummy(dummy_t);
    গ্রন্থি->src[0] = গ্রন্থি;
    group_গ্রন্থি.init(গ্রন্থি);
    group_গ্রন্থি.outputs.clear();
    group_গ্রন্থি.name = "MDEV_GROUP";

    প্রতি (auto & sub : এতত্->mdev.sessions) {
        sub->clone_buffer(এতত্->fence_buf);
    }

    op_batch->add_op(group_গ্রন্থি);
}

শূন্য ggml_hexagon_session::enqueue_cpy(ধ্রুব ggml_tensor * src, ggml_tensor * dst, ধ্রুব ggml_tensor * sync_tensor, uint32_t fence_seq) {
    ধ্রুব তার্কিক with_fence = sync_tensor != nullptr;
    htp_opnode cpy_গ্রন্থি(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* গ্রন্থি = cpy_গ্রন্থি.add_dummy(*dst);
    গ্রন্থি->op     = GGML_OP_CPY;
    গ্রন্থি->src[0] = const_cast<ggml_tensor *>(src);
    গ্রন্থি->src[1] = with_fence ? cpy_গ্রন্থি.add_dummy(*sync_tensor) : nullptr;
    যদি (with_fence) {
        গ্রন্থি->op_params[0] = (int32_t) fence_seq;
    }

    cpy_গ্রন্থি.init(গ্রন্থি);
    যদি (with_fence) {
        cpy_গ্রন্থি.name = "CPY+FENCE";
    }
    ধ্রুব তার্কিক ok = ggml_hexagon_precompute_cpy_params(এতত্, গ্রন্থি, (সংরচনা htp_copy_kernel_params *) cpy_গ্রন্থি.kernel_params);
    ধ্রুব auto * kparams = (ধ্রুব সংরচনা htp_copy_kernel_params *) cpy_গ্রন্থি.kernel_params;
    যদি (ok && !with_fence && kparams->total_elems == 0) {
        প্রত্যাগম;
    }
    এতত্->enqueue_op(cpy_গ্রন্থি);
}

শূন্য ggml_hexagon_session::enqueue_fence(ধ্রুব ggml_tensor * sync_tensor, uint32_t fence_seq, তার্কিক wait) {
    htp_opnode sync_গ্রন্থি(HTP_OP_FENCE);

    ggml_tensor* গ্রন্থি = sync_গ্রন্থি.add_dummy(*sync_tensor);
    গ্রন্থি->op           = GGML_OP_NONE;
    গ্রন্থি->src[0]       = গ্রন্থি;
    গ্রন্থি->op_params[0] = (int32_t) fence_seq;
    গ্রন্থি->op_params[1] = wait ? 0 : 1;

    sync_গ্রন্থি.init(গ্রন্থি);
    sync_গ্রন্থি.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    এতত্->enqueue_op(sync_গ্রন্থি);
}

স্থির তার্কিক ggml_hexagon_precompute_allreduce_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    তার্কিক has_add,
    তার্কিক is_row_bcast,
    তার্কিক is_shard_ok,
    সংরচনা htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    ধ্রুব uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    ধ্রুব uint32_t elem_size = (dst->type == GGML_TYPE_F16) ? পরিমাণ(ggml_fp16_t) : পরিমাণ(ভিন্ন);
    ধ্রুব তার্কিক is_contiguous = ggml_is_contiguous(dst);

    ধ্রুব uint32_t ne0 = (uint32_t) dst->ne[0];
    ধ্রুব uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    ধ্রুব তার্কিক use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    যদি (has_add) {
        // sharded reduce-scatter for contiguous in-place add
        যদি (opt_ar_scatter && is_shard_ok && use_1d && !is_row_bcast) {
            ধ্রুব uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            ধ্রুব uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            ধ্রুব uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_start = (int32_t) rank_elem_start;
            kparams->rank_nelem      = (int32_t) (rank_elem_end - rank_elem_start);
        } অন্যথা যদি (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } অন্যথা {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } অন্যথা {
        kparams->n_dsts = (int32_t) n_ranks;
        যদি (use_1d) {
            ধ্রুব uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            ধ্রুব uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            ধ্রুব uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            ধ্রুব uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } অন্যথা {
            ধ্রুব uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            ধ্রুব uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            ধ্রুব uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            ধ্রুব uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    যদি (use_1d) {
        ধ্রুব uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        ধ্রুব uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        ধ্রুব size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_গুনতি(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        যদি (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        যাবত্ ((size_t) kparams->vtcm_size > sess->vtcm_size && block_elems > 128) {
            ধ্রুব size_t max_bytes_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((size_t) (max_bytes_per_buf / elem_size), 128);
            যদি (block_elems < 128) ভঙ্গ;
            kparams->block_elems          = block_elems;
            kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
        }

        যদি (sess->vtcm_size < (size_t) kparams->vtcm_size || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            প্রত্যাগম অসত্য;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        প্রত্যাগম সত্য;
    } অন্যথা {
        ধ্রুব uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        ধ্রুব uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        ধ্রুব size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_গুনতি(n_ranks, n_threads, has_add, is_row_bcast);

        ধ্রুব uint32_t row_bytes = ne0 * elem_size;
        ধ্রুব uint32_t row_size_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_size_aligned = row_size_aligned;

        ধ্রুব uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        যাবত্ ((size_t) kparams->vtcm_size > sess->vtcm_size && block_rows > 1) {
            ধ্রুব size_t max_rows_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2 * row_size_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
            যদি (max_rows_per_buf == 0) ভঙ্গ;
        }

        যদি (sess->vtcm_size < (size_t) kparams->vtcm_size || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            প্রত্যাগম অসত্য;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        প্রত্যাগম সত্য;
    }
}

শূন্য ggml_hexagon_session::enqueue_allreduce(
    ধ্রুব ggml_tensor * dst,
    ধ্রুব std::vector<ধ্রুব ggml_tensor *> & src_tensors,
    ধ্রুব std::vector<ধ্রুব ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_গ্রন্থি(HTP_OP_ALLREDUCE);

    ggml_tensor* গ্রন্থি = ar_গ্রন্থি.add_dummy(*dst);
    গ্রন্থি->op           = GGML_OP_NONE;
    গ্রন্থি->op_params[0] = (int32_t) fence_seq_entry;
    গ্রন্থি->op_params[1] = (int32_t) fence_seq_exit;

    ar_গ্রন্থি.init(গ্রন্থি);

    ar_গ্রন্থি.inputs.clear();
    প্রতি (size_t i = 0; i < src_tensors.size(); i++) {
        ar_গ্রন্থি.inputs.push_back(src_tensors[i]);
    }
    প্রতি (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_গ্রন্থি.inputs.push_back(ar_গ্রন্থি.add_dummy(*sync_tensors[i]));
    }

    ar_গ্রন্থি.outputs.clear();
    প্রতি (size_t i = 0; i < src_tensors.size(); i++) {
        ar_গ্রন্থি.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        এতত্, dst, rank, n_ranks, অসত্য, অসত্য, /*is_shard_ok=*/ অসত্য,
        (সংরচনা htp_allreduce_kernel_params *) ar_গ্রন্থি.kernel_params);

    ar_গ্রন্থি.name = "ALLREDUCE";
    এতত্->enqueue_op(ar_গ্রন্থি);
}

ggml_hexagon_shared_buffer * ggml_hexagon_session::mmap_tensor(ধ্রুব ggml_tensor * t) {
    যদি (!t) প্রত্যাগম nullptr;

    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
    যদি (!sbuf->mapped) {
        ধ্রুব তার্কিক is_weight = ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        ধ্রুব তার্কিক extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    প্রত্যাগম sbuf;
}

তার্কিক ggml_hexagon_session::clone_buffer(ধ্রুব ggml_hexagon_shared_buffer * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    যদি (sbuf->sess == এতত্) প্রত্যাগম সত্য;

    auto mem = sbuf->mem;
    পূর্ণাঙ্ক   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    যদি (এতত্->cloned_buffers.find(fd) != এতত্->cloned_buffers.end()) প্রত্যাগম সত্য;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", এতত্->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    auto clone = std::make_unique<ggml_hexagon_shared_buffer>(এতত্, *sbuf);
    যত্ন {
        clone->mmap(sbuf->extended);
    } গৃহাণ (ধ্রুব std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", এতত্->c_name(), exc.what());
        প্রত্যাগম অসত্য;
    }

    এতত্->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(এতত্);
    প্রত্যাগম সত্য;
}

শূন্য ggml_hexagon_session::release_buffer(ধ্রুব ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    পূর্ণাঙ্ক   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    auto it = এতত্->cloned_buffers.find(fd);
    যদি (it != এতত্->cloned_buffers.end()) {
        auto clone = std::move(it->second);
        এতত্->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(এতত্);
}

শূন্য ggml_hexagon_session::unclone_buffer(ধ্রুব ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    প্রতি (auto * sess : sessions) {
        sess->release_buffer(sbuf);
    }
}

স্থির size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    ধ্রুব size_t MiB = 1024 * 1024;
    ধ্রুব size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t step = 256u * MiB;

    যত্ন {
        sbufs.push_back(নূতন ggml_hexagon_shared_buffer(sess, GiB, সত্য)); vmem += GiB;
        sbufs.push_back(নূতন ggml_hexagon_shared_buffer(sess, GiB, সত্য)); vmem += GiB;
        sbufs.push_back(নূতন ggml_hexagon_shared_buffer(sess, GiB, সত্য)); vmem += GiB;

        যাবত্ (1) {
            sbufs.push_back(নূতন ggml_hexagon_shared_buffer(sess, step, সত্য));
            vmem += step;
        }
    } গৃহাণ (...) { }

    প্রতি (auto b : sbufs) { delete b; }

    প্রত্যাগম vmem - step; // backoff to account for overhead from internal mappings
}

শূন্য ggml_hexagon_session::allocate(ধ্রুব ggml_hexagon_device_config & config) noexcept(অসত্য) {
    পূর্ণাঙ্ক phys_idx = config.physical_idx;
    পূর্ণাঙ্ক virt_idx = config.virtual_idx;

    এতত্->valid_session = অসত্য;
    এতত্->valid_handle  = অসত্য;
    এতত্->valid_queue   = অসত্য;
    এতত্->valid_iface   = অসত্য;

    এতত্->name          = config.name;
    এতত্->phys_idx      = phys_idx;
    এতত্->virt_idx      = virt_idx;
    এতত্->domain_id     = config.domain_id;
    এতত্->session_id    = 0;
    এতত্->batch_req_seq = 0;
    এতত্->batch_rsp_seq = 0;
    এতত্->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", এতত্->name.c_str(), এতত্->domain_id, phys_idx, virt_idx);

    যদি (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        ক্ষিপ std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    ধ্রুব std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    যদি (virt_idx > 0) {
        সংরচনা remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<বর্ণ *>(dom_name.c_str());
        n.session_name     = const_cast<বর্ণ *>(এতত্->name.c_str());
        n.session_name_len = এতত্->name.size();
        n.session_id       = virt_idx;

        পূর্ণাঙ্ক err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (শূন্য *) &n, পরিমাণ(n));
        যদি (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           এতত্->c_name(), phys_idx, virt_idx, err);
            ক্ষিপ std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        এতত্->session_id    = n.session_id;
        এতত্->domain_id     = n.effective_domain_id;
        এতত্->valid_session = সত্য;
    } অন্যথা {
        সংরচনা remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<বর্ণ *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        পূর্ণাঙ্ক err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (শূন্য *) &eff, পরিমাণ(eff));
        যদি (err == AEE_SUCCESS) {
            এতত্->domain_id = eff.effective_domain_id;
        } অন্যথা {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           এতত্->name.c_str(), err, এতত্->domain_id);
        }
    }

    // Enable unsigned modules
    {
        সংরচনা remote_rpc_control_unsigned_module u;
        u.domain = এতত্->domain_id;
        u.enable = 1;
        পূর্ণাঙ্ক err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (শূন্য *) &u, পরিমাণ(u));
        যদি (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", এতত্->c_name(), err);
            ক্ষিপ std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    বর্ণ session_uri[256];
    {
        বর্ণ htp_uri[256];
        snprintf(htp_uri, পরিমাণ(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        সংরচনা remote_rpc_get_uri u = {};
        u.session_id      = এতত্->session_id;
        u.domain_name     = const_cast<বর্ণ *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<বর্ণ *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = পরিমাণ(session_uri);

        পূর্ণাঙ্ক err = remote_session_control(FASTRPC_GET_URI, (শূন্য *) &u, পরিমাণ(u));
        যদি (err != AEE_SUCCESS) {
            snprintf(session_uri, পরিমাণ(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), এতত্->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          এতত্->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    পূর্ণাঙ্ক err = htp_iface_open(session_uri, &এতত্->handle);
    যদি (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", এতত্->c_name(), session_uri, err);
        ক্ষিপ std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    এতত্->valid_handle = সত্য;

    // Query HW info and resolve session options
    {
        unsigned পূর্ণাঙ্ক hw_n_threads = 0;
        unsigned পূর্ণাঙ্ক hw_n_hvx     = 0;
        unsigned পূর্ণাঙ্ক hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        পূর্ণাঙ্ক hw_err = htp_iface_hwinfo(এতত্->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        যদি (hw_err == 0) {
            ধ্রুব uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            এতত্->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_threads) : max_n_threads;
            এতত্->n_hvx     = এতত্->n_threads;
            এতত্->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            এতত্->vtcm_size = (uint64_t)hw_vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          এতত্->c_name(), এতত্->n_threads, এতত্->n_hvx, এতত্->n_hmx,
                          (unsigned long long)(এতত্->vtcm_size / (1024 * 1024)));
        } অন্যথা {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", এতত্->c_name(), hw_err);
            ধ্রুব uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            এতত্->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_threads;
            এতত্->n_hvx     = এতত্->n_threads;
            এতত্->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            এতত্->vtcm_size = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        সংরচনা remote_rpc_control_latency l;
        l.enable = 1;

        পূর্ণাঙ্ক err = remote_handle64_control(এতত্->handle, DSPRPC_CONTROL_LATENCY, (শূন্য *) &l, পরিমাণ(l));
        যদি (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", এতত্->c_name(),
                  এতত্->session_id, এতত্->domain_id, session_uri, (unsigned long) এতত্->handle);

    ধ্রুব size_t req_q_size = (পরিমাণ(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    ধ্রুব size_t rsp_q_size = (পরিমাণ(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(এতত্->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (শূন্য *) এতত্,  // Callback context
                          &queue);
    যদি (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", এতত্->name.c_str(), (unsigned) err);
        ক্ষিপ std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    এতত্->valid_queue = সত্য;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &এতত্->queue_id);
    যদি (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        ক্ষিপ std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    যদি (opt_etm) {
        err = htp_iface_etm(এতত্->handle, 1);
        যদি (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    এতত্->op_queue = নূতন ggml_hexagon_opqueue(এতত্, opt_opbatch, opt_opqueue);

    এতত্->fence_buf = নূতন ggml_hexagon_fence_buffer(এতত্, &dev_ctx->fence_buffer_type, 64 * 1024);
    যদি (এতত্->mdev.count > 1) {
        এতত্->mdev_fence_slot = এতত্->alloc_fence(এতত্->mdev.count);
    }

    যদি (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(এতত্);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", এতত্->c_name(), opt_vmem);
    }
    ধ্রুব size_t shm_size = এতত্->op_queue->shm_size();
    এতত্->max_vmem = (opt_vmem > shm_size) ? (opt_vmem - shm_size) : opt_vmem;

    এতত্->op_batch = নূতন ggml_hexagon_opbatch(এতত্, opt_opbatch, এতত্->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(এতত্->handle, এতত্->session_id, এতত্->queue_id, এতত্->n_threads, opt_nhmx, এতত্->max_vmem, (uint32_t) opt_power_mode);
    যদি (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", এতত্->c_name(), (unsigned) err);
        ক্ষিপ std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    এতত্->valid_iface = সত্য;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", এতত্->c_name(), htp_power_mode_to_str(opt_power_mode));

    যদি (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(এতত্->handle, opt_profile, &pmu_conf);
        যদি (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }
}

শূন্য ggml_hexagon_session::release() noexcept(সত্য) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", এতত্->name.c_str());

    এতত্->mdev.sessions.clear();

    পূর্ণাঙ্ক err;

    যদি (এতত্->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(এতত্->handle);
        যদি (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    delete এতত্->op_batch;
    delete এতত্->op_queue;
    প্রতি (auto & it : এতত্->cpy_fence_slots) {
        free_fence((শূন্য *) it.second, 1);
    }
    এতত্->cpy_fence_slots.clear();

    যদি (এতত্->fence_buf) {
        unclone_buffer(এতত্->fence_buf);
        delete এতত্->fence_buf;
        এতত্->fence_buf = nullptr;
    }
    যাবত্ (!এতত্->cloned_buffers.empty()) {
        release_buffer(এতত্->cloned_buffers.begin()->second.get());
    }

    যদি (opt_etm) {
        err = htp_iface_etm(এতত্->handle, 0);
        যদি (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    যদি (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(এতত্->handle, 0, &pmu_conf);
        যদি (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    যদি (এতত্->valid_queue) {
        err = dspqueue_close(queue);
        যদি (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (unsigned) err);
        }
    }

    যদি (এতত্->valid_handle) {
        htp_iface_close(এতত্->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(ধ্রুব ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_গুনতি) noexcept(অসত্য) {
    এতত্->dev        = dev;
    এতত্->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    এতত্->mdev.idx   = mdev_idx;
    এতত্->mdev.count = mdev_গুনতি > 0 ? mdev_গুনতি : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)এতত্) & 0xFFFF;

    যত্ন {
        allocate(config);
        যদি (এতত্->mdev.idx == 0 && !config.mdev_group.empty()) {
            প্রতি (size_t i = 0; i < config.mdev_group.size(); i++) {
                এতত্->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], এতত্->dev, (uint32_t) (i + 1), এতত্->mdev.count));
            }
        }
    } গৃহাণ (ধ্রুব std::exception & exc) {
        release();
        ক্ষিপ;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(সত্য) {
    release();
}

// ** backend interface

স্থির তার্কিক ggml_hexagon_flash_attn_is_hmx_eligible(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * q,
    ধ্রুব সংরচনা ggml_tensor * k,
    ধ্রুব সংরচনা ggml_tensor * v,
    ধ্রুব সংরচনা ggml_tensor * sinks
) {
    যদি (sess->n_hmx == 0) {
        প্রত্যাগম অসত্য;
    }

    যদি (opt_fa_select < 2) {
        প্রত্যাগম অসত্য;
    }

    যদি ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t DK = q->ne[0];
    ধ্রুব uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    যদি (DK % 8 != 0 || DV % 8 != 0) {
        প্রত্যাগম অসত্য;
    }

    GGML_UNUSED(sinks);

    // Explicit force mode
    যদি (opt_fa_select > 2) {
        প্রত্যাগম সত্য;
    }

    ধ্রুব uint32_t M = q->ne[1];

    // Prefill or batched decode
    যদি (M > 1) {
        প্রত্যাগম সত্য;
    }

    ধ্রুব uint32_t n_head = q->ne[2];
    ধ্রুব uint32_t n_kv_heads = k->ne[2];
    ধ্রুব uint32_t G = n_kv_heads > 0 ? n_head / n_kv_heads : 1;
    ধ্রুব uint32_t S = k->ne[1];
    ধ্রুব uint32_t total_threads = sess->n_threads * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    যদি (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < total_threads))) {
        প্রত্যাগম সত্য;
    }

    // Tile alignment for 32-row HMX tiles
    ধ্রুব তার্কিক is_tile_aligned = (G > 0 && (32 % G == 0));
    যদি (!is_tile_aligned) {
        যদি (sess->n_threads >= 6) {
            প্রত্যাগম অসত্য;
        }
        প্রত্যাগম S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    যদি (DK == 256) {
        s_cross = 4096;
    } অন্যথা যদি (DK <= 64) {
        s_cross = (sess->n_threads >= 8) ? 2048 : ((sess->n_threads >= 6) ? 768 : 512);
    } অন্যথা {
        s_cross = (sess->n_threads >= 8) ? 1024 : 512;
    }

    প্রত্যাগম S >= s_cross;
}

স্থির তার্কিক ggml_hexagon_precompute_flash_attn_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_fa_kernel_params * kparams
) {
    যদি (opt_fa_select < 1) {
        প্রত্যাগম অসত্য;
    }

    memset(kparams, 0, পরিমাণ(*kparams));

    ধ্রুব সংরচনা ggml_tensor * q    = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * k    = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * v    = op->src[2];
    ধ্রুব সংরচনা ggml_tensor * mask = op->src[3];

    ধ্রুব uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ধ্রুব uint32_t neq1 = q->ne[1];  // n_tokens
    ধ্রুব uint32_t neq2 = q->ne[2];  // n_heads

    ধ্রুব uint32_t nek1 = k->ne[1];  // kv_len

    ধ্রুব uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ধ্রুব uint32_t DK = neq0;
    ধ্রুব uint32_t DV = nev0;

    ধ্রুব uint32_t n_kv_heads = k->ne[2];
    ধ্রুব uint32_t G          = neq2 / n_kv_heads;

    ভিন্ন scale         = 1.0f;
    ভিন্ন max_bias      = 0.0f;
    ভিন্ন logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], পরিমাণ(ভিন্ন));
    memcpy(&max_bias,      &op->op_params[1], পরিমাণ(ভিন্ন));
    memcpy(&logit_softcap, &op->op_params[2], পরিমাণ(ভিন্ন));

    যদি (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->head_split = (opt_fa_head_split != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    ধ্রুব uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    ধ্রুব সংরচনা ggml_tensor * sinks = op->src[4];
    যদি (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        ধ্রুব uint32_t DK_pad = hex_round_up(DK, 64);
        ধ্রুব uint32_t DV_pad = hex_round_up(DV, 64);
        size_t Br = 0, Bc = 0;
        পূর্ণাঙ্ক ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
        যদি (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            যদি (logit_softcap == 0.0f) {
                kparams->scale = scale * (ভিন্ন) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } অন্যথা {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (ভিন্ন) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);

            ধ্রুব size_t row_vec_bytes = hex_align_up(Bc * পরিমাণ(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            ধ্রুব size_t m_সারি_bytes = hex_align_up(Bc * পরিমাণ(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_সারি_bytes / পরিমাণ(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_মানসমূহ(G);
            যদি (mask) {
                kparams->src3_div2 = init_fastdiv_মানসমূহ(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_মানসমূহ(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            প্রত্যাগম সত্য;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    ধ্রুব size_t size_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    ধ্রুব size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    ধ্রুব size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), mask != nullptr, sinks != nullptr, n_head, sess->n_threads);

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_মানসমূহ(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_মানসমূহ(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_মানসমূহ(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_মানসমূহ(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_মানসমূহ(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_মানসমূহ(q->ne[3]/v->ne[3]);
    যদি (mask) {
        kparams->src3_div2 = init_fastdiv_মানসমূহ(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_মানসমূহ(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    প্রত্যাগম সত্য;
}

স্থির তার্কিক ggml_hexagon_supported_flash_attn_ext(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * src2 = op->src[2];
    ধ্রুব সংরচনা ggml_tensor * src3 = op->src[3];
    ধ্রুব সংরচনা ggml_tensor * src4 = op->src[4];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    যদি ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        প্রত্যাগম অসত্য;
    }

    যদি (src3 && src3->type != GGML_TYPE_F16) {  // mask
        প্রত্যাগম অসত্য;
    }

    যদি (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        প্রত্যাগম অসত্য;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    যদি (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        প্রত্যাগম অসত্য;
    }

    সংরচনা htp_fa_kernel_params kparams;
    যদি (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        প্রত্যাগম অসত্য;
    }

    যদি ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;
}

স্থির তার্কিক ggml_hexagon_supported_gated_delta_net(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    যদি (opt_gdn_select < 1) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব সংরচনা ggml_tensor * q     = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * k     = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * v     = op->src[2];
    ধ্রুব সংরচনা ggml_tensor * g     = op->src[3];
    ধ্রুব সংরচনা ggml_tensor * beta  = op->src[4];
    ধ্রুব সংরচনা ggml_tensor * state = op->src[5];
    ধ্রুব সংরচনা ggml_tensor * dst   = op;

    যদি (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    যদি (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব int64_t S_v      = v->ne[0];
    ধ্রুব int64_t H        = v->ne[1];
    ধ্রুব int64_t n_tokens = v->ne[2];
    ধ্রুব int64_t n_seqs   = v->ne[3];
    ধ্রুব int64_t K        = ggml_get_op_params_i32(op, 0);

    যদি (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        প্রত্যাগম অসত্য;
    }
    যদি (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        প্রত্যাগম অসত্য;
    }
    যদি ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        প্রত্যাগম অসত্য;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    যদি (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        প্রত্যাগম অসত্য;
    }
    যদি (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t total_rows = (uint32_t) (H * n_seqs);
    ধ্রুব uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    ধ্রুব তার্কিক can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    যদি (can_use_hmx) {
        সংরচনা htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        যদি (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, সত্য, &n_heads_batch)) {
            প্রত্যাগম অসত্য;
        }
    } অন্যথা {
        সংরচনা htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_threads);
        যদি (layout.total_bytes > sess->vtcm_size) {
            প্রত্যাগম অসত্য;
        }
    }

    প্রত্যাগম সত্য;
}

স্থির তার্কিক ggml_hexagon_matmul_is_hmx_eligible(
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    তার্কিক is_matmul_id,
    তার্কিক is_batched
) {
    GGML_UNUSED(dst);

    যদি (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব পূর্ণাঙ্ক ne00  = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক ne01  = src0->ne[1];
    ধ্রুব পূর্ণাঙ্ক ne11  = src1->ne[1];
    ধ্রুব পূর্ণাঙ্ক ne12  = src1->ne[2];
    ধ্রুব পূর্ণাঙ্ক wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    যদি (ne01 % HTP_TENSOR_TILE_N_ROWS != 0 && is_matmul_id) {
        প্রত্যাগম অসত্য;
    }

    // HMX supports F16, F32, and repack quantized types.
    যদি (!htp_is_hmx_weight_type((ggml_type) wtype)) {
        প্রত্যাগম অসত্য;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    যদি (ne00 % HTP_TENSOR_TILE_N_COLS != 0) {
        প্রত্যাগম অসত্য;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    ধ্রুব তার্কিক is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    যদি (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        প্রত্যাগম অসত্য;
    }
    যদি (!is_matmul_id && is_batched && is_tiled) {
        প্রত্যাগম অসত্য;
    }

    // HMX assumes contiguous row-major layout.
    যদি (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        প্রত্যাগম অসত্য;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    ধ্রুব পূর্ণাঙ্ক m = is_matmul_id ? ne12 : ne11;
    যদি (m <= HTP_MM_HMX_MIN_NROWS) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(dst);
}

স্থির তার্কিক ggml_hexagon_precompute_hmx_mm_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * act,
    ধ্রুব সংরচনা ggml_tensor * dst,
    পূর্ণাঙ্ক wtype,
    পূর্ণাঙ্ক ne00_padded,
    পূর্ণাঙ্ক ne01_padded,
    পূর্ণাঙ্ক ne02,
    পূর্ণাঙ্ক ne11,
    পূর্ণাঙ্ক ne12,
    পূর্ণাঙ্ক ne11_padded,
    তার্কিক is_matmul_id,
    তার্কিক is_batched,
    size_t src2_size,
    size_t vtcm_budget,
    সংরচনা htp_mm_kernel_params * kparams
) {
    ধ্রুব তার্কিক is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ধ্রুব পূর্ণাঙ্ক aligned_tile_size = is_tiled ? htp_tensor_get_tile_padded_size(wtype) : 0;
    ধ্রুব পূর্ণাঙ্ক n_threads = (পূর্ণাঙ্ক)sess->n_threads;
    ধ্রুব পূর্ণাঙ্ক ne10 = act->ne[0];

    পূর্ণাঙ্ক m_for_solver = ne11;
    পূর্ণাঙ্ক m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    যদি (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        ধ্রুব তার্কিক dst_row_split = dst ? ggml_hexagon_can_row_partition(dst) : সত্য;
        ধ্রুব তার্কিক act_row_split = ggml_hexagon_can_row_partition(act);
        যদি (dst_row_split && act_row_split) {
            m_for_solver = (ne11 + (পূর্ণাঙ্ক) sess->mdev.count - 1) / (পূর্ণাঙ্ক) sess->mdev.count;
            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
        }
    }
    ধ্রুব তার্কিক pipeline = is_matmul_id ? অসত্য : htp_mm_hmx_pipeline(m_for_solver);

    ধ্রুব তার্কিক is_batched_val = is_matmul_id ? অসত্য : is_batched;
    ধ্রুব পূর্ণাঙ্ক group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    তার্কিক use_grouped = অসত্য;
    পূর্ণাঙ্ক act_threads_selected = 0;

    যদি (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && group_size > 1) {
        // Try grouped path first
        যদি (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = সত্য;
        }
    }

    যদি (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        পূর্ণাঙ্ক m_id_rows = 0;
        যদি (dst && is_matmul_id) {
            ধ্রুব পূর্ণাঙ্ক n_experts = ne02 > 0 ? ne02 : 1;
            ধ্রুব size_t total_expert_rows = (size_t) dst->ne[1] * dst->ne[2];
            পূর্ণাঙ্ক m_per_expert = (পূর্ণাঙ্ক) ((total_expert_rows + n_experts - 1) / n_experts);
            যদি (sess->mdev.count > 1 && ggml_hexagon_can_row_partition(dst)) {
                m_per_expert = (m_per_expert + (পূর্ণাঙ্ক) sess->mdev.count - 1) / (পূর্ণাঙ্ক) sess->mdev.count;
            }
            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
        }
        ধ্রুব uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
        যদি (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            প্রত্যাগম অসত্য;
        }
    }

    kparams->n_hmx = 1;
    kparams->pipeline = pipeline ? 1 : 0;
    kparams->m_chunk = m_chunk;
    kparams->n_chunk = n_chunk;
    kparams->n_threads = n_threads;
    kparams->n_act_threads = act_threads_selected;
    kparams->tile_size = is_tiled ? htp_tensor_get_tile_size(wtype) : 0;
    kparams->aligned_tile_size = aligned_tile_size;
    kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
    kparams->vtcm_size = vtcm_size;
    kparams->vtcm_src0_size = 0;
    kparams->div_n_act_threads = init_fastdiv_মানসমূহ(act_threads_selected);
    kparams->div_ne00_padded   = init_fastdiv_মানসমূহ(ne00_padded);
    kparams->vtcm_act_size     = 0;
    kparams->vtcm_bias_size    = (int32_t) src2_size;
    kparams->vtcm_dst_size     = 0;

    যদি (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } অন্যথা {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    প্রত্যাগম সত্য;
}

স্থির শূন্য ggml_hexagon_precompute_hvx_mm_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    পূর্ণাঙ্ক wtype,
    তার্কিক is_tiled,
    পূর্ণাঙ্ক ne02,
    পূর্ণাঙ্ক ne03,
    পূর্ণাঙ্ক ne10,
    পূর্ণাঙ্ক ne11,
    পূর্ণাঙ্ক ne12,
    পূর্ণাঙ্ক ne13,
    তার্কিক is_matmul_id,
    ধ্রুব size_t src2_row_size,
    size_t vtcm_budget,
    সংরচনা htp_mm_kernel_params * kparams
) {
    যদি (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        প্রত্যাগম;
    }

    kparams->n_hmx = 0;
    kparams->n_threads = sess->n_threads;

    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);

    ধ্রুব পূর্ণাঙ্ক src1_nrows = ne11 * ne12 * ne13;

    যদি (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_size = htp_tensor_get_tile_size(wtype);
        kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);

        ধ্রুব তার্কিক k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        যদি (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (পূর্ণাঙ্ক) sess->n_threads)
                                  ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);

            সংরচনা htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            প্রতি (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], kparams->act_row_size, 0, d, সত্য, অসত্য
                );
                যদি (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    ভঙ্গ;
                }
            }
            যদি (L.total_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                প্রত্যাগম;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = 0;
            kparams->vtcm_dst_size  = L.dst_bytes;
            প্রত্যাগম;
        } অন্যথা {
            তার্কিক try_tiled = (k_align && opt_mm_select >= 1);
            যদি (try_tiled) {
                kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                যদি (src1_nrows < (পূর্ণাঙ্ক) sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } অন্যথা {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                সংরচনা htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                প্রতি (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, অসত্য, অসত্য
                    );
                    যদি (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        ভঙ্গ;
                    }
                }

                uint32_t m_chunk = 0;
                যদি (htp_mm_hvx_solve_vtcm_params(
                        kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, best_n_prefetch, vtcm_budget,
                        &L, &m_chunk)) {
                    kparams->n_prefetch     = best_n_prefetch;
                    kparams->m_chunk        = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
                    kparams->vtcm_size      = L.total_bytes;
                    kparams->vtcm_src0_size = L.src0_bytes;
                    kparams->vtcm_act_size  = L.act_bytes;
                    kparams->vtcm_bias_size = L.bias_bytes;
                    kparams->vtcm_dst_size  = L.dst_bytes;
                    প্রত্যাগম;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            প্রত্যাগম;
        }
    } অন্যথা যদি (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        প্রত্যাগম;
    } অন্যথা যদি (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        সংরচনা htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        যদি (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_row_size = hex_round_up(ne10 * 2, 128);
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = L.bias_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
            প্রত্যাগম;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        প্রত্যাগম;
    } অন্যথা যদি (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        সংরচনা htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        যদি (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_row_size = hex_round_up(ne10 * 4, 128);
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = L.bias_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
            প্রত্যাগম;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        প্রত্যাগম;
    } অন্যথা যদি (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        সংরচনা htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        যদি (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_BF16_F32_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_BF16_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_row_size = hex_round_up(ne10 * 4, 128);
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = L.bias_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
            প্রত্যাগম;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        প্রত্যাগম;
    } অন্যথা {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        প্রত্যাগম;
    }
}

স্থির শূন্য ggml_hexagon_precompute_matmul_params_impl(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    ধ্রুব size_t src2_row_size,
    ধ্রুব size_t src2_size,
    সংরচনা htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));

    ধ্রুব পূর্ণাঙ্ক ne00 = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক ne01 = src0->ne[1];
    ধ্রুব পূর্ণাঙ্ক ne02 = src0->ne[2];
    ধ্রুব পূর্ণাঙ্ক ne03 = src0->ne[3];

    ধ্রুব পূর্ণাঙ্ক ne10 = src1->ne[0];
    ধ্রুব পূর্ণাঙ্ক ne11 = src1->ne[1];
    ধ্রুব পূর্ণাঙ্ক ne12 = src1->ne[2];
    ধ্রুব পূর্ণাঙ্ক ne13 = src1->ne[3];

    ধ্রুব তার্কিক is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ধ্রুব পূর্ণাঙ্ক wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ধ্রুব পূর্ণাঙ্ক ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    ধ্রুব পূর্ণাঙ্ক ne11_padded = hex_round_up(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    ধ্রুব পূর্ণাঙ্ক ne01_tiled  = hex_round_up(ne01, 32);

    ধ্রুব তার্কিক is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    ধ্রুব তার্কিক is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ধ্রুব size_t vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    তার্কিক hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    যদি (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        যদি (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
            গচ্ছ finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_size, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_মানসমূহ(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_মানসমূহ(ne11);
    kparams->div_r2       = init_fastdiv_মানসমূহ(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_মানসমূহ(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_মানসমূহ(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
স্থির তার্কিক ggml_hexagon_rows_stride(ধ্রুব int64_t * ne, ধ্রুব size_t * nb, size_t * stride) {
    size_t s = 0, next = 0;
    প্রতি (পূর্ণাঙ্ক i = 1; i < GGML_MAX_DIMS; i++) {
        যদি (ne[i] == 1) অনুবর্ত;
        যদি (s == 0) { s = nb[i]; next = s * ne[i]; অনুবর্ত; }
        যদি (nb[i] != next) প্রত্যাগম অসত্য;
        next *= ne[i];
    }
    *stride = s ? s : nb[1];
    প্রত্যাগম সত্য;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
স্থির তার্কিক ggml_hexagon_matmul_can_collapse(ধ্রুব সংরচনা ggml_tensor * src0, ধ্রুব সংরচনা ggml_tensor * src1, ধ্রুব সংরচনা ggml_tensor * dst) {
    size_t s1, sd;
    প্রত্যাগম (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_size(src1->type) && ggml_hexagon_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_size(dst->type) && ggml_hexagon_rows_stride(dst->ne, dst->nb, &sd);
}

স্থির তার্কিক ggml_hexagon_matmul_add_can_collapse(
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * src2,
    ধ্রুব সংরচনা ggml_tensor * dst
) {
    যদি (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        প্রত্যাগম অসত্য;
    }
    যদি (!src2) {
        প্রত্যাগম সত্য;
    }
    ধ্রুব int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    যদি (src2_nrows == 1) {
        প্রত্যাগম সত্য;
    }
    size_t s2;
    প্রত্যাগম src2->nb[0] == ggml_type_size(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_rows_stride(src2->ne, src2->nb, &s2);
}

স্থির ggml_tensor ggml_hexagon_tensor_collapse_rows(ধ্রুব সংরচনা ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    প্রত্যাগম c;
}

স্থির শূন্য ggml_hexagon_precompute_matmul_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_mm_kernel_params * kparams
) {
    যদি (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ধ্রুব ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        ধ্রুব ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        প্রত্যাগম;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

স্থির শূন্য ggml_hexagon_precompute_fused_matmul_add_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * src2,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_mm_kernel_params * kparams
) {
    যদি (ggml_hexagon_matmul_add_can_collapse(src0, src1, src2, dst)) {
        ধ্রুব ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        ধ্রুব ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ধ্রুব ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        ধ্রুব সংরচনা ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        ধ্রুব size_t src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
        kparams->collapse = 1;
        প্রত্যাগম;
    }
    ধ্রুব size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

স্থির তার্কিক ggml_hexagon_precompute_দ্বিমিক_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    uint32_t op,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_দ্বিমিক_kernel_params * kparams,
    ধ্রুব সংরচনা ggml_tensor * src2
) {
    memset(kparams, 0, পরিমাণ(*kparams));

    ধ্রুব size_t elem_size = ggml_type_size(src0->type);
    ধ্রুব size_t src0_row_size = src0->ne[0] * elem_size;
    ধ্রুব size_t src1_row_size = src1->ne[0] * elem_size;
    ধ্রুব size_t dst_row_size  = dst->ne[0]  * elem_size;

    ধ্রুব size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    ধ্রুব size_t src1_row_size_aligned = hex_round_up(src1_row_size, 128);
    ধ্রুব size_t dst_row_size_aligned  = hex_round_up(dst_row_size,  128);

    যদি (!htp_দ্বিমিক_strides_safe(src0, src1, dst, elem_size)) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব তার্কিক is_add_id = op == HTP_OP_ADD_ID;
    ধ্রুব তার্কিক is_scalar = !is_add_id && src1->ne[0] == 1;
    ধ্রুব তার্কিক is_row_bcast = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    ধ্রুব তার্কিক is_same_shape = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    ধ্রুব তার্কিক is_complex   = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    ধ্রুব তার্কিক is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    ধ্রুব তার্কিক is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    যদি (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        ধ্রুব uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        ধ্রুব uint32_t n_threads = sess->n_threads;
        ধ্রুব uint32_t max_chunk_elems = 32768 / elem_size;
        ধ্রুব uint32_t min_chunk_elems = 256;
        ধ্রুব uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
        ধ্রুব uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        ধ্রুব uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

        kparams->kernel_type           = HTP_দ্বিমিক_KERNEL_CHUNKED;
        kparams->n_threads             = n_threads;
        kparams->rows_per_buffer       = 1;
        kparams->src0_row_size_aligned = src0_row_size_aligned;
        kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
        kparams->dst_row_size_aligned  = dst_row_size_aligned;
        kparams->src1_size             = 0;
        kparams->chunk_size            = chunk_size;
        kparams->chunk_bytes           = chunk_bytes;
        kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

        সংরচনা htp_দ্বিমিক_vtcm_layout L;
        htp_দ্বিমিক_vtcm_layout_build(&L, kparams, sess->vtcm_size);
        যদি (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
            প্রত্যাগম অসত্য;
        }

        kparams->vtcm_size = L.total_bytes;
        প্রত্যাগম সত্য;
    }

    গণনা htp_দ্বিমিক_kernel_type kernel_type;
    size_t src1_size = 0;
    size_t src2_size = 0;

    যদি (is_add_id) {
        GGML_ASSERT(src2 != nullptr);
        kernel_type = HTP_দ্বিমিক_KERNEL_ADD_ID;
        src1_size = hex_round_up(src1->ne[1] * src1_row_size_aligned, 128);
        src2_size = hex_round_up(ggml_nelements(src2) * পরিমাণ(int32_t), 128);
        যদি (src1_size + src2_size > sess->vtcm_size) {
            প্রত্যাগম অসত্য;
        }
    } অন্যথা যদি (is_row_bcast) {
        kernel_type = HTP_দ্বিমিক_KERNEL_ROW_BCAST;
        src1_size = src1_row_size_aligned;
    } অন্যথা যদি (is_scalar) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size, 128);
        যদি (src1_size > sess->vtcm_size) {
            প্রত্যাগম অসত্য;
        }
        kernel_type = HTP_দ্বিমিক_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } অন্যথা যদি (is_same_shape) {
        kernel_type = HTP_দ্বিমিক_KERNEL_SAME_SHAPE;
    } অন্যথা যদি (is_complex) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        যদি (src1_size > sess->vtcm_size) {
            প্রত্যাগম অসত্য;
        }
        kernel_type = HTP_দ্বিমিক_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } অন্যথা {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        যদি (src1_size > sess->vtcm_size) {
            প্রত্যাগম অসত্য;
        }
        kernel_type = HTP_দ্বিমিক_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_threads = sess->n_threads;
    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;
    kparams->src1_size             = src1_size;
    kparams->src2_size             = src2_size;

    ধ্রুব তার্কিক src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    ধ্রুব তার্কিক src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    ধ্রুব তার্কিক dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    ধ্রুব তার্কিক src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    ধ্রুব তার্কিক src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    ধ্রুব তার্কিক dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    তার্কিক split_at_ne01 = অসত্য;
    যদি (src0->ne[2] > 1) {
        যদি (kernel_type == HTP_দ্বিমিক_KERNEL_SAME_SHAPE) {
            split_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } অন্যথা যদি (kernel_type == HTP_দ্বিমিক_KERNEL_ROW_BCAST) {
            split_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } অন্যথা {
            split_at_ne01 = সত্য;
        }
    }
    kparams->split_at_ne01 = split_at_ne01 ? 1 : 0;

    তার্কিক split_at_ne02 = অসত্য;
    যদি (src0->ne[3] > 1) {
        যদি (kernel_type == HTP_দ্বিমিক_KERNEL_SAME_SHAPE) {
            split_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } অন্যথা যদি (kernel_type == HTP_দ্বিমিক_KERNEL_ROW_BCAST) {
            split_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } অন্যথা {
            split_at_ne02 = সত্য;
        }
    }
    kparams->split_at_ne02 = split_at_ne02 ? 1 : 0;

    সংরচনা htp_দ্বিমিক_vtcm_layout L;
    htp_দ্বিমিক_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    যদি (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {
        যদি (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            ধ্রুব uint32_t total_elems = (uint32_t) ggml_nelements(src0);
            ধ্রুব uint32_t n_threads = sess->n_threads;
            ধ্রুব uint32_t max_chunk_elems = 32768 / elem_size;
            ধ্রুব uint32_t min_chunk_elems = 256;
            ধ্রুব uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
            ধ্রুব uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
            ধ্রুব uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

            kparams->kernel_type           = HTP_দ্বিমিক_KERNEL_CHUNKED;
            kparams->n_threads             = n_threads;
            kparams->rows_per_buffer       = 1;
            kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
            kparams->src1_size             = 0;
            kparams->chunk_size            = chunk_size;
            kparams->chunk_bytes           = chunk_bytes;
            kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

            htp_দ্বিমিক_vtcm_layout_build(&L, kparams, sess->vtcm_size);
            যদি (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
                প্রত্যাগম অসত্য;
            }

            kparams->vtcm_size = L.total_bytes;
            প্রত্যাগম সত্য;
        }
        প্রত্যাগম অসত্য;
    }

    kparams->rows_per_buffer = L.rows_per_buffer;
    kparams->vtcm_size       = L.total_bytes;

    প্রত্যাগম সত্য;
}

স্থির শূন্য ggml_hexagon_precompute_unary_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    uint32_t op,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));

    ধ্রুব uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুব uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    ধ্রুব size_t elem_size = ggml_type_size(src0->type);

    ধ্রুব size_t src0_data_row_size = src0->ne[0] * elem_size;
    ধ্রুব size_t dst_data_row_size  = dst->ne[0]  * ggml_type_size(dst->type);

    ধ্রুব size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    ধ্রুব size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = 0;
    তার্কিক broadcast_weight = অসত্য;

    যদি (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_data_row_size = src1->ne[0] * ggml_type_size(src1->type);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->broadcast_weight      = broadcast_weight;

    সংরচনা htp_unary_vtcm_layout L;
    uint32_t col_tile = 0;
    uint32_t vtcm_row_per_thread = 0;

    htp_unary_vtcm_layout_build(&L, op, src0->ne[0], dst->ne[0],
                                op == HTP_OP_RMS_NORM_MUL ? src1->ne[0] : 0,
                                broadcast_weight, n_threads, sess->vtcm_size, elem_size,
                                &col_tile, &vtcm_row_per_thread);

    kparams->col_tile = col_tile;
    kparams->vtcm_row_per_thread = vtcm_row_per_thread;
    kparams->vtcm_size = L.total_bytes;

    kparams->vtcm_src0_size_per_thread = L.src0_bytes;
    kparams->vtcm_src1_size_per_thread = L.src1_bytes;
    kparams->vtcm_dst_size_per_thread  = L.dst_bytes;

    kparams->vtcm_src0_size = L.src0_bytes * n_threads;
    kparams->vtcm_src1_size = L.src1_bytes * n_threads;
    kparams->vtcm_dst_size  = L.dst_bytes * n_threads;

    kparams->block = col_tile ? 0 : ((L.src0_bytes / 2) / src0_row_size_aligned);

    ধ্রুব uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_মানসমূহ(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_মানসমূহ(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_মানসমূহ(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_মানসমূহ(tiles_per_row);
}

স্থির শূন্য ggml_hexagon_precompute_get_rows_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_get_rows_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));

    ধ্রুব uint32_t ne00 = src0->ne[0];
    ধ্রুব uint32_t ne02 = src0->ne[2];
    ধ্রুব uint32_t ne03 = src0->ne[3];

    ধ্রুব uint32_t ne10 = src1->ne[0];
    ধ্রুব uint32_t ne11 = src1->ne[1];
    ধ্রুব uint32_t ne12 = src1->ne[2];
    ধ্রুব uint32_t nr = ne10 * ne11 * ne12;

    ধ্রুব তার্কিক tiled = ggml_hexagon_tensor_is_tiled(src0);
    ধ্রুব পূর্ণাঙ্ক wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    যদি (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } অন্যথা যদি (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } অন্যথা {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    ধ্রুব uint32_t total_tasks = nr;

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, total_tasks);

    সংরচনা htp_get_rows_vtcm_layout vtcm_layout = {};
    যাবত্ (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, kparams->n_threads);
        যদি (vtcm_layout.total_bytes <= sess->vtcm_size) {
            ভঙ্গ;
        }
        --kparams->n_threads;
    }

    যদি (kparams->n_threads == 0 && total_tasks > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, 1);
    }

    kparams->vtcm_size = (total_tasks == 0) ? 0 : vtcm_layout.total_bytes;
    kparams->tasks_per_thread = kparams->n_threads > 0 ? (total_tasks + kparams->n_threads - 1) / kparams->n_threads : 0;
    kparams->total_tasks = total_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_মানসমূহ(ne10) : fastdiv_মানসমূহ{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_মানসমূহ(ne10 * ne11) : fastdiv_মানসমূহ{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_মানসমূহ(ne02) : fastdiv_মানসমূহ{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_মানসমূহ(ne03) : fastdiv_মানসমূহ{0, 0};
}

স্থির শূন্য ggml_hexagon_precompute_set_rows_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0, //মানসমূহ
    ধ্রুব সংরচনা ggml_tensor * src1, // indices
    ধ্রুব সংরচনা ggml_tensor * dst,  // destination
    সংরচনা htp_set_rows_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));

    ধ্রুব uint32_t nr = src0->ne[1];

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, nr);
    kparams->tasks_per_thread = (nr + kparams->n_threads - 1) / kparams->n_threads;
    kparams->total_tasks = nr;

    kparams->div_ne11 = init_fastdiv_মানসমূহ(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_মানসমূহ(src1->ne[2]);
    kparams->div_tasks_per_thread = init_fastdiv_মানসমূহ(kparams->tasks_per_thread);
    kparams->div_ne02 = init_fastdiv_মানসমূহ(src0->ne[2]);

    সংরচনা htp_set_rows_vtcm_layout vtcm_layout;
    htp_set_rows_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_threads);
    kparams->vtcm_size = vtcm_layout.total_bytes;
}

স্থির শূন্য ggml_hexagon_precompute_softmax_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));

    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];

    ধ্রুব uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুব uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);

    ভিন্ন scale = 1.0f;
    ভিন্ন max_bias = 0.0f;
    memcpy(&scale,    &op->op_params[0], পরিমাণ(ভিন্ন));
    memcpy(&max_bias, &op->op_params[1], পরিমাণ(ভিন্ন));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    ধ্রুব uint32_t n_head = src0->ne[2];
    ধ্রুব uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    kparams->n_head      = n_head;
    kparams->n_head_log2 = n_head_log2;

    যদি (max_bias > 0.0f && n_head_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_head_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    } অন্যথা {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    ধ্রুব uint32_t ne00 = src0->ne[0];
    ধ্রুব uint32_t ne10 = src1 ? src1->ne[0] : 1;

    সংরচনা htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, ne00, ne10, kparams->use_src1 != 0, kparams->use_f16 != 0, n_threads);

    kparams->n_threads                 = n_threads;
    kparams->src0_nrows                = src0_nrows;
    kparams->src0_nrows_per_thread     = (src0_nrows + n_threads - 1) / n_threads;
    kparams->vtcm_size                 = (uint32_t) layout.total_bytes;
    kparams->vtcm_src0_size_per_thread = (uint32_t) layout.src0_bytes_per_thread;
    kparams->vtcm_src1_size_per_thread = (uint32_t) layout.src1_bytes_per_thread;
    kparams->vtcm_dst_size_per_thread  = (uint32_t) layout.dst_bytes_per_thread;
    kparams->src0_row_size_aligned     = (uint32_t) layout.src0_spad_half_size;
    kparams->src1_row_size_aligned     = (uint32_t) layout.src1_spad_half_size;
    kparams->dst_row_size_aligned      = (uint32_t) layout.dst_spad_half_size;
    kparams->src0_spad_half_size       = (uint32_t) layout.src0_spad_half_size;
    kparams->src1_spad_half_size       = (uint32_t) layout.src1_spad_half_size;
    kparams->dst_spad_half_size        = (uint32_t) layout.dst_spad_half_size;
    যদি (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } অন্যথা যদি (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F16;
    } অন্যথা {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    যদি (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_মানসমূহ(src0->ne[1]);
    যদি (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_মানসমূহ(src0->ne[2]);
    ধ্রুব uint32_t ne12 = src1 ? src1->ne[2] : 1;
    ধ্রুব uint32_t ne13 = src1 ? src1->ne[3] : 1;
    যদি (ne12 > 0) kparams->div_ne12 = init_fastdiv_মানসমূহ(ne12);
    যদি (ne13 > 0) kparams->div_ne13 = init_fastdiv_মানসমূহ(ne13);
}

স্থির শূন্য ggml_hexagon_precompute_rope_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));

    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src2 = op->src[2];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    ধ্রুব uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুব uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ধ্রুব uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    সংরচনা htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);

    kparams->n_threads              = n_threads;
    kparams->src0_nrows             = src0_nrows;
    kparams->src0_nrows_per_thread  = (src0_nrows + n_threads - 1) / n_threads;
    kparams->vtcm_size              = (uint32_t) layout.total_bytes;
    kparams->spad_per_thread        = (uint32_t) layout.bytes_per_thread;
    kparams->theta_cache_offset     = (uint32_t) layout.theta_cache_size_aligned;
    kparams->src0_row_size_aligned  = (uint32_t) layout.src0_row_size_aligned;
    kparams->freq_factors_offset    = (uint32_t) (layout.bytes_per_thread * n_threads);
    kparams->freq_factors_size      = (uint32_t) layout.freq_factors_size_aligned;

    যদি (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_মানসমূহ(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_মানসমূহ(dst->ne[1]);
    }
}

স্থির শূন্য ggml_hexagon_precompute_ssm_conv_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * src1,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));

    ধ্রুব uint32_t d_conv  = (uint32_t) src1->ne[0];
    ধ্রুব uint32_t d_inner = (uint32_t) src0->ne[1];
    ধ্রুব uint32_t n_t     = (uint32_t) dst->ne[1];
    ধ্রুব uint32_t n_s     = (uint32_t) dst->ne[2];
    ধ্রুব uint32_t ncs     = (uint32_t) src0->ne[0];

    ধ্রুব uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    ধ্রুব uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
    ধ্রুব uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);

    ধ্রুব uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * পরিমাণ(ভিন্ন), 128) + 128;
    ধ্রুব uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * পরিমাণ(ভিন্ন), 128);
    ধ্রুব uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

    uint32_t vtcm_src0_per_thread = 0;
    uint32_t vtcm_dst_per_thread  = 0;

    যদি (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;

        ধ্রুব uint32_t src0_tile_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * পরিমাণ(ভিন্ন), 128);
        ধ্রুব uint32_t src0_T_bytes        = hex_round_up(d_conv * d_inner_per_thread * পরিমাণ(ভিন্ন), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw_bytes + src0_T_bytes;

        ধ্রুব uint32_t dst_tile_bytes = hex_round_up(d_inner_per_thread * পরিমাণ(ভিন্ন), 128);
        vtcm_dst_per_thread = 2 * dst_tile_bytes;
    } অন্যথা {
        ধ্রুব size_t vtcm_budget = (sess->vtcm_size > 0 ? sess->vtcm_size / n_threads : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        ধ্রুব uint32_t src0_block_T    = hex_round_up(ncs * 32 * পরিমাণ(ভিন্ন), 128);
        ধ্রুব size_t   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
        ধ্রুব size_t   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);

        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
        target_max_tile = (std::max)(target_max_tile, 32u);
        target_max_tile = (std::min)(target_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * পরিমাণ(ভিন্ন)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        যদি (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        যদি (d_inner_tile > target_max_tile) {
            d_inner_tile = target_max_tile;
        }
        যদি (d_inner_tile > d_inner_per_thread) {
            d_inner_tile = d_inner_per_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        ধ্রুব uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * পরিমাণ(ভিন্ন), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw + src0_block_T;

        vtcm_dst_per_thread = 2 * hex_round_up(d_inner_tile * n_t * পরিমাণ(ভিন্ন), 128);
    }

    kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
    kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
    kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;

    kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
    kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
    kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
    kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
}

স্থির শূন্য ggml_hexagon_precompute_gated_delta_net_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));

    ধ্রুব সংরচনা ggml_tensor * q     = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * k     = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * v     = op->src[2];
    ধ্রুব সংরচনা ggml_tensor * g     = op->src[3];
    ধ্রুব সংরচনা ggml_tensor * state = op->src[5];

    ধ্রুব uint32_t S_v      = (uint32_t) v->ne[0];
    ধ্রুব uint32_t H        = (uint32_t) v->ne[1];
    ধ্রুব uint32_t n_tokens = (uint32_t) v->ne[2];
    ধ্রুব uint32_t n_seqs   = (uint32_t) v->ne[3];
    ধ্রুব uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    ধ্রুব uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    ধ্রুব uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    ধ্রুব uint32_t total_rows = H * n_seqs;
    ধ্রুব uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    ধ্রুব তার্কিক can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    সংরচনা htp_gdn_hmx_vtcm_layout hmx_layout;
    সংরচনা htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    যদি (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, সত্য, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_size      = HTP_GDN_CHUNK_SIZE;
        kparams->n_chunks        = (n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_size       = (uint32_t) hmx_layout.total_bytes;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_bytes;
        kparams->vtcm_per_thread = (uint32_t) (hmx_layout.total_bytes / (n_threads > 0 ? n_threads : 1));
    } অন্যথা {
        htp_gdn_vtcm_layout_build(&hvx_layout, S_v, n_threads);
        kparams->kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
        kparams->pipeline        = 0;
        kparams->n_heads_batch   = 1;
        kparams->state_aligned   = (uint32_t) hvx_layout.state_aligned;
        kparams->vtcm_per_thread = (uint32_t) hvx_layout.bytes_per_thread;
        kparams->vtcm_size       = (uint32_t) hvx_layout.total_bytes;
    }

    kparams->n_threads           = n_threads;
    kparams->S_v                 = S_v;
    kparams->H                   = H;
    kparams->n_tokens            = n_tokens;
    kparams->n_seqs              = n_seqs;
    kparams->K                   = K;
    kparams->total_rows          = total_rows;
    kparams->rows_per_thread     = (total_rows + kparams->n_threads - 1) / kparams->n_threads;
    kparams->kda                 = (g->ne[0] == S_v) ? 1 : 0;
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / পরিমাণ(ভিন্ন));
    kparams->state_size_per_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((ভিন্ন) S_v);

    যদি (H > 0)                  kparams->div_H         = init_fastdiv_মানসমূহ(H);
    যদি (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_মানসমূহ((uint32_t) q->ne[1]);
    যদি (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_মানসমূহ((uint32_t) k->ne[1]);
    যদি (rq3 > 0)                kparams->div_rq3       = init_fastdiv_মানসমূহ(rq3);
    যদি (rk3 > 0)                kparams->div_rk3       = init_fastdiv_মানসমূহ(rk3);
    যদি (kparams->n_threads > 0) kparams->div_n_threads = init_fastdiv_মানসমূহ(kparams->n_threads);
}

স্থির শূন্য ggml_hexagon_precompute_sort_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    তার্কিক is_শীর্ষ_k,
    সংরচনা htp_sort_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));

    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    ধ্রুব uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুব uint32_t ne00       = src0->ne[0];
    ধ্রুব uint32_t k          = dst->ne[0];

    int32_t order = GGML_SORT_ORDER_DESC;
    যদি (!is_শীর্ষ_k) {
        order = ((ধ্রুব int32_t *) op->op_params)[0];
    }

    ধ্রুব uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ধ্রুব size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    সংরচনা htp_sort_vtcm_layout layout;
    তার্কিক ok = htp_sort_solve_layout(&layout, ne00, total_rows, k, n_threads_max, vtcm_budget, is_শীর্ষ_k);
    GGML_ASSERT(ok);

    kparams->n_threads         = (int32_t) layout.n_threads;
    kparams->total_rows        = (int32_t) total_rows;
    kparams->row_start         = 0;
    kparams->row_end           = (int32_t) total_rows;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->order             = order;
    kparams->is_শীর্ষ_k          = is_শীর্ষ_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_size         = (int32_t) layout.total_bytes;
    kparams->phase1_slot_size  = (int32_t) layout.phase1_slot_size;
    kparams->merge_মানসমূহ_off  = (int32_t) layout.merge_মানসমূহ_off;
    kparams->merge_indices_off = (int32_t) layout.merge_indices_off;
    kparams->merge_elems       = (int32_t) layout.merge_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

স্থির শূন্য ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    ধ্রুব uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    ধ্রুব int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    যদি (numer_hi < 0) {
        *ox_hi = 0;
    } অন্যথা {
        ধ্রুব uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    যদি (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

স্থির তার্কিক ggml_hexagon_pool_shape_fits(
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * dst
) {
    প্রত্যাগম ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

স্থির শূন্য ggml_hexagon_precompute_pool_2d_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0,
    ধ্রুব সংরচনা ggml_tensor * dst,
    সংরচনা htp_pool_2d_kernel_params * kparams,
    তার্কিক is_pool_1d
) {
    memset(kparams, 0, পরিমাণ(*kparams));

    ধ্রুব uint32_t src_x = (uint32_t) src0->ne[0];
    ধ্রুব uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    ধ্রুব uint32_t dst_x = (uint32_t) dst->ne[0];
    ধ্রুব uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    ধ্রুব uint32_t src_plane_bytes = src_x * src_y * পরিমাণ(ভিন্ন);
    ধ্রুব uint32_t dst_plane_bytes = dst_x * dst_y * পরিমাণ(ভিন্ন);
    ধ্রুব uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    ধ্রুব uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    সংরচনা htp_pool_vtcm_layout layout;
    ধ্রুব তার্কিক ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, sess->vtcm_size);
    GGML_ASSERT(ok);

    kparams->src_x = src_x;
    kparams->src_y = src_y;
    kparams->dst_x = dst_x;
    kparams->dst_y = dst_y;
    kparams->kernel_x = (uint32_t) ggml_get_op_params_i32(dst, 1);
    kparams->kernel_y = is_pool_1d ? 1 : (uint32_t) ggml_get_op_params_i32(dst, 2);
    kparams->stride_x = (uint32_t) ggml_get_op_params_i32(dst, is_pool_1d ? 2 : 3);
    kparams->stride_y = is_pool_1d ? 1 : (uint32_t) ggml_get_op_params_i32(dst, 4);
    kparams->pad_x = ggml_get_op_params_i32(dst, is_pool_1d ? 3 : 5);
    kparams->pad_y = is_pool_1d ? 0 : ggml_get_op_params_i32(dst, 6);
    kparams->src_plane_bytes = src_plane_bytes;
    kparams->dst_plane_bytes = dst_plane_bytes;
    kparams->src_plane_bytes_aligned = (uint32_t) layout.src_spad_half_size;
    kparams->dst_plane_bytes_aligned = (uint32_t) layout.dst_spad_half_size;
    kparams->n_threads = n_threads;
    kparams->planes = planes;
    kparams->pool_op = (uint32_t) ggml_get_op_params_i32(dst, 0);
    // Fast HVX path requires exact tiling and a supported kernel width.
    kparams->fast_path = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->stride_x == kparams->kernel_x && kparams->stride_y == kparams->kernel_y &&
                           (kparams->kernel_x == 1 || kparams->kernel_x == 2)) ? 1 : 0;
    kparams->global_path = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                            kparams->stride_x == kparams->kernel_x &&
                            kparams->stride_y == kparams->kernel_y &&
                            kparams->kernel_x == kparams->src_x &&
                            kparams->kernel_y == kparams->src_y &&
                            kparams->dst_x == 1 && kparams->dst_y == 1) ? 1 : 0;
    kparams->block_path = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->kernel_y == 1 && kparams->stride_y == 1 &&
                           kparams->stride_x == kparams->kernel_x &&
                           kparams->kernel_x >= 32) ? 1 : 0;
    kparams->avg_divide_গুনতি = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    ধ্রুব তার্কিক narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_path = (kparams->fast_path && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (ভিন্ন) (kparams->kernel_x * kparams->kernel_y);
}

স্থির তার্কিক ggml_hexagon_precompute_concat_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (!src0 || !src1 || !dst) {
        প্রত্যাগম অসত্য;
    }

    পূর্ণাঙ্ক dim = ((ধ্রুব int32_t *) op->op_params)[0];
    যদি (dim < 0 || dim >= GGML_MAX_DIMS) {
        প্রত্যাগম অসত্য;
    }
    kparams->dim = dim;

    যদি (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        প্রত্যাগম অসত্য;
    }
    যদি (src0->type != dst->type || src1->type != dst->type) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t type_size = ggml_type_size(dst->type);

    প্রতি (পূর্ণাঙ্ক d = 0; d < GGML_MAX_DIMS; d++) {
        ধ্রুব int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        যদি (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            প্রত্যাগম অসত্য;
        }
    }

    ধ্রুব uint32_t max_gather_ne0 = 128 / type_size;
    ধ্রুব uint32_t total_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    ধ্রুব তার্কিক b_dense = (src1->nb[0] == type_size);
    ধ্রুব তার্কিক gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_size &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_size
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_size);

    যদি (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_size && dst->nb[1] == (size_t) dst->ne[0] * type_size &&
        src0->nb[0] == type_size &&
        (src1->nb[0] == type_size || src1->nb[1] == type_size) &&
        gather_strides_ok &&
        htp_gather_rows_tensor_outer_contiguous(dst) &&
        htp_gather_rows_tensor_outer_contiguous(src0) &&
        htp_gather_rows_tensor_outer_contiguous(src1)) {

        যদি (htp_gather_rows_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_size, total_rows,
                                         sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ROWS;
            kparams->n_threads   = (uint8_t) sess->n_threads;
            kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
            প্রত্যাগম সত্য;
        }
    }

    ধ্রুব তার্কিক dma_strides_ok = (src0->nb[0] == type_size && src1->nb[0] == type_size && dst->nb[0] == type_size);

    যদি (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_threads   = 1;
        প্রত্যাগম সত্য;
    }

    ধ্রুব তার্কিক is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    ধ্রুব তার্কিক is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    ধ্রুব তার্কিক transposed_rows_ok = (src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size);

    যদি (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        ধ্রুব uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 8;
        সংরচনা htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_size, n_threads);

        যদি (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
            প্রত্যাগম অসত্য;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_threads                         = n_threads;
        kparams->vtcm_size                         = layout.total_bytes;
        kparams->u.transposed.spad0_size_per_thread = layout.src0_spad_size_per_thread;
        kparams->u.transposed.spad1_size_per_thread = layout.src1_spad_size_per_thread;
        প্রত্যাগম সত্য;
    }

    প্রত্যাগম অসত্য;
}

স্থির তার্কিক ggml_hexagon_precompute_cpy_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (!src0 || !dst) {
        প্রত্যাগম অসত্য;
    }

    যদি (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        প্রত্যাগম অসত্য;
    }
    যদি (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব int64_t nelem_src = ggml_nelements(src0);
    ধ্রুব int64_t nelem_dst = ggml_nelements(dst);
    যদি (nelem_src != nelem_dst || nelem_src < 0) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t src_type_size = ggml_type_size(src0->type);
    ধ্রুব uint32_t dst_type_size = ggml_type_size(dst->type);

    kparams->src0_type_size = (uint8_t) src_type_size;
    kparams->dst_type_size  = (uint8_t) dst_type_size;
    kparams->total_elems    = (uint32_t) nelem_src;

    যদি (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        প্রত্যাগম সত্য;
    }

    যদি (nelem_src == 1) {
        যদি (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            প্রত্যাগম সত্য;
        }
        যদি ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            প্রত্যাগম সত্য;
        }
        প্রত্যাগম অসত্য;
    }

    ধ্রুব তার্কিক sametype = (src0->type == dst->type);

    তার্কিক same_extents = সত্য;
    প্রতি (পূর্ণাঙ্ক d = 0; d < GGML_MAX_DIMS; d++) {
        যদি (src0->ne[d] != dst->ne[d]) {
            same_extents = অসত্য;
            ভঙ্গ;
        }
    }

    ধ্রুব তার্কিক transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_size) || (dst->nb[0] != dst_type_size) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_size) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_size);
    ধ্রুব তার্কিক sameshape  = same_extents && !transposed;

    ধ্রুব তার্কিক src_is_contiguous = ggml_is_contiguous(src0);
    ধ্রুব তার্কিক dst_is_contiguous = ggml_is_contiguous(dst);

    যদি (sametype) {
        যদি (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            প্রত্যাগম সত্য;
        }

        ধ্রুব uint32_t max_gather_ne0 = 128 / src_type_size;
        ধ্রুব uint32_t total_rows = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        যদি ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_size && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_size &&
            total_rows > 1 &&
            htp_gather_rows_tensor_outer_contiguous(src0)) {

            যদি (htp_gather_rows_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, অসত্য,
                                             src_type_size, total_rows,
                                             sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ROWS;
                kparams->n_threads   = (uint8_t) sess->n_threads;
                kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
                প্রত্যাগম সত্য;
            }
        }

        যদি (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            প্রত্যাগম সত্য;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_threads   = sess->n_threads > 0 ? (uint8_t) sess->n_threads : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_মানসমূহ((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_মানসমূহ((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_মানসমূহ((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_মানসমূহ((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_মানসমূহ((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_মানসমূহ((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        প্রত্যাগম সত্য;
    }

    যদি (!sameshape) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব তার্কিক valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    যদি (!valid_conversion) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 4;
    সংরচনা htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_size, dst_type_size, n_threads);

    যদি (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
        প্রত্যাগম অসত্য;
    }

    kparams->kernel_type             = HTP_COPY_KERNEL_SAMESHAPE_CONVERT;
    kparams->total_rows              = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    kparams->n_threads               = (uint8_t) n_threads;
    kparams->vtcm_size               = layout.total_bytes;
    kparams->u.convert.src0_buf_size = layout.src0_buf_size;
    kparams->u.convert.dst_buf_size  = layout.dst_buf_size;
    kparams->u.convert.spad0_size_per_thread = layout.spad0_size_per_thread;
    kparams->u.convert.spad1_size_per_thread = layout.spad1_size_per_thread;
    kparams->u.convert.src0_row_stride = layout.src0_row_stride;
    kparams->u.convert.dst_row_stride  = layout.dst_row_stride;
    kparams->u.convert.blk_rows        = layout.blk_rows;
    kparams->u.convert.div_ne01      = init_fastdiv_মানসমূহ((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_মানসমূহ((uint32_t) (src0->ne[2] * src0->ne[1]));

    প্রত্যাগম সত্য;
}

স্থির শূন্য ggml_hexagon_precompute_fused_mmnx_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0, // W0
    ধ্রুব সংরচনা ggml_tensor * act,  // x
    int32_t n_weights,
    সংরচনা htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));
    kparams->n_threads = sess->n_threads;

    ধ্রুব পূর্ণাঙ্ক ne00 = src0->ne[0];
    ধ্রুব পূর্ণাঙ্ক ne01 = src0->ne[1];
    ধ্রুব পূর্ণাঙ্ক ne02 = src0->ne[2];
    ধ্রুব পূর্ণাঙ্ক ne03 = src0->ne[3];

    ধ্রুব পূর্ণাঙ্ক ne10 = act->ne[0];
    ধ্রুব পূর্ণাঙ্ক ne11 = act->ne[1];
    ধ্রুব পূর্ণাঙ্ক ne12 = act->ne[2];
    ধ্রুব পূর্ণাঙ্ক ne13 = act->ne[3];

    ধ্রুব তার্কিক is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ধ্রুব পূর্ণাঙ্ক wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ধ্রুব পূর্ণাঙ্ক ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    ধ্রুব পূর্ণাঙ্ক ne11_padded = hex_round_up(ne11, 32);
    ধ্রুব পূর্ণাঙ্ক ne01_tiled  = hex_round_up(ne01, 32);

    ধ্রুব size_t vtcm_budget = sess->vtcm_size;
    ধ্রুব তার্কিক is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    তার্কিক hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    যদি (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, nullptr, অসত্য, is_batched)) {
        যদি (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, অসত্য, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            গচ্ছ finalize;
        }
    }

    যদি (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        প্রত্যাগম;
    }

    {
        ধ্রুব পূর্ণাঙ্ক act_nrows = ne11 * ne12 * ne13;
        ধ্রুব size_t act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
        ধ্রুব size_t src0_row_size = src0->nb[1];

        ধ্রুব uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        প্রতি (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            সংরচনা htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
                0, src0_row_size, act_row_size, 0, d, অসত্য, সত্য
            );
            যদি (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                ভঙ্গ;
            }
        }

        সংরচনা htp_mm_hvx_vtcm_layout L;
        তার্কিক try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
            0, src0_row_size, act_row_size, 0, best_n_prefetch, অসত্য, সত্য
        );

        যদি (try_tiled && L.total_bytes <= sess->vtcm_size) {
            kparams->kernel_type       = HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->tile_size         = htp_tensor_get_tile_size(wtype);
            kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);
            kparams->act_row_size      = act_row_size;
            kparams->vtcm_src0_size    = L.src0_bytes;
            kparams->vtcm_act_size     = L.act_bytes;
            kparams->vtcm_bias_size    = 0;
            kparams->vtcm_dst_size     = L.dst_bytes;
            kparams->vtcm_size         = L.total_bytes;
            kparams->n_prefetch        = best_n_prefetch;
            kparams->n_weights         = n_weights;
        } অন্যথা {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            প্রত্যাগম;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_মানসমূহ(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_মানসমূহ(ne11);
    kparams->div_r2       = init_fastdiv_মানসমূহ(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_মানসমূহ(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_মানসমূহ(ne12);
}

স্থির শূন্য ggml_hexagon_precompute_fused_mmidnx_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * src0, // W0
    ধ্রুব সংরচনা ggml_tensor * act,  // x
    ধ্রুব সংরচনা ggml_tensor * dst,  // dst0
    int32_t n_weights,
    সংরচনা htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

স্থির তার্কিক ggml_hexagon_op_mul_mat_use_fwht(ধ্রুব সংরচনা ggml_tensor * dst) {
    যদি (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব সংরচনা ggml_tensor * src0 = dst->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = dst->src[1];

    ধ্রুব int64_t n = src1->ne[0];
    যদি (n < 32 || (n & (n - 1)) != 0) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_rows(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

স্থির তার্কিক ggml_hexagon_supported_mul_mat(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * dst) {
    ধ্রুব সংরচনা ggml_tensor * src0 = dst->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = dst->src[1];

    যদি (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        যদি (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            প্রত্যাগম অসত্য;
        }
        সংরচনা htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, nullptr, dst, &ukparams);
        প্রত্যাগম ukparams.vtcm_row_per_thread > 0 && (size_t) ukparams.vtcm_size <= sess->vtcm_size;
    }

    যদি (dst->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    যদি (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        প্রত্যাগম অসত্য;
    }

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_Q1_0:
        পক্ষ GGML_TYPE_Q2_0:
        পক্ষ GGML_TYPE_Q4_0:
        পক্ষ GGML_TYPE_Q4_1:
        পক্ষ GGML_TYPE_Q8_0:
        পক্ষ GGML_TYPE_IQ4_NL:
        পক্ষ GGML_TYPE_MXFP4:
        পক্ষ GGML_TYPE_Q4_K:
        পক্ষ GGML_TYPE_Q5_K:
        পক্ষ GGML_TYPE_Q6_K:
        পক্ষ GGML_TYPE_Q3_K:
        পক্ষ GGML_TYPE_Q2_K: {
            যদি (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                প্রত্যাগম অসত্য;
            }

            ধ্রুব int64_t qk = ggml_blck_size(src0->type);
            যদি (src0->ne[0] % qk) {
                প্রত্যাগম অসত্য;
            }

            যদি (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                প্রত্যাগম অসত্য;
            }
            যদি (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                প্রত্যাগম অসত্য;
            }
            ভঙ্গ;
        }

        পক্ষ GGML_TYPE_F16:
            যদি (src0->nb[1] < src0->nb[0]) {
                প্রত্যাগম অসত্য;
            }
            যদি (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                প্রত্যাগম অসত্য;
            }
            যদি (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                প্রত্যাগম অসত্য;
            }
            ভঙ্গ;

        পক্ষ GGML_TYPE_F32:
            যদি (src1->type != GGML_TYPE_F32) {
                প্রত্যাগম অসত্য;
            }
            যদি (src0->nb[1] < src0->nb[0]) {
                প্রত্যাগম অসত্য;
            }
            যদি (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                প্রত্যাগম অসত্য;
            }
            যদি (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                প্রত্যাগম অসত্য;
            }
            ভঙ্গ;

        পক্ষ GGML_TYPE_BF16:
            যদি (src0->nb[1] < src0->nb[0]) {
                প্রত্যাগম অসত্য;
            }
            যদি (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                প্রত্যাগম অসত্য;
            }
            যদি (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                প্রত্যাগম অসত্য;
            }
            ভঙ্গ;

        স্বভাব:
            প্রত্যাগম অসত্য;
    }

    সংরচনা htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    যদি (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;
}

স্থির তার্কিক ggml_hexagon_supported_mul_mat_id(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * src2 = op->src[2];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        প্রত্যাগম অসত্য;
    }

    যদি (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        প্রত্যাগম অসত্য;
    }

    বিকল্প (src0->type) {
        পক্ষ GGML_TYPE_Q1_0:
        পক্ষ GGML_TYPE_Q2_0:
        পক্ষ GGML_TYPE_Q4_0:
        পক্ষ GGML_TYPE_Q4_1:
        পক্ষ GGML_TYPE_Q8_0:
        পক্ষ GGML_TYPE_IQ4_NL:
        পক্ষ GGML_TYPE_MXFP4:
        পক্ষ GGML_TYPE_Q4_K:
        পক্ষ GGML_TYPE_Q5_K:
        পক্ষ GGML_TYPE_Q6_K:
        পক্ষ GGML_TYPE_Q3_K:
        পক্ষ GGML_TYPE_Q2_K:
            যদি (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                প্রত্যাগম অসত্য;
            }

            যদি (src0->ne[0] % ggml_blck_size(src0->type) != 0) {
                প্রত্যাগম অসত্য;
            }
            ভঙ্গ;

        পক্ষ GGML_TYPE_F16:
        পক্ষ GGML_TYPE_F32:
        পক্ষ GGML_TYPE_BF16:
            যদি (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                প্রত্যাগম অসত্য;
            }
            যদি (src0->ne[0] % 32 != 0) {
                প্রত্যাগম অসত্য;
            }
            ভঙ্গ;

        স্বভাব:
            প্রত্যাগম অসত্য;
    }

    সংরচনা htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    যদি (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;
}

স্থির তার্কিক ggml_hexagon_supported_দ্বিমিক(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src0->type == GGML_TYPE_F32) {
        যদি (src1->type != GGML_TYPE_F32) {
            প্রত্যাগম অসত্য;
        }
        যদি (dst->type != GGML_TYPE_F32) {
            প্রত্যাগম অসত্য;
        }
    }
    অন্যথা যদি (src0->type == GGML_TYPE_F16) {
        যদি (src1->type != GGML_TYPE_F16) {
            প্রত্যাগম অসত্য;
        }
        যদি (dst->type != GGML_TYPE_F16) {
            প্রত্যাগম অসত্য;
        }
    }
    অন্যথা {
        প্রত্যাগম অসত্য;
    }

    যদি (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        প্রত্যাগম অসত্য;
    }
    যদি (!ggml_are_same_shape(src0, dst)) {
        প্রত্যাগম অসত্য;
    }
    যদি (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        প্রত্যাগম অসত্য;
    }

    সংরচনা htp_দ্বিমিক_kernel_params kparams;
    প্রত্যাগম ggml_hexagon_precompute_দ্বিমিক_params(sess, op_remap_to_htp(op), src0, src1, dst, &kparams);
}

স্থির তার্কিক ggml_hexagon_supported_add_id(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * src2 = op->src[2];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (!src2) {
        প্রত্যাগম অসত্য;
    }
    যদি (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        প্রত্যাগম অসত্য;
    }
    যদি (!ggml_are_same_shape(src0, dst)) {
        প্রত্যাগম অসত্য;
    }
    যদি (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        প্রত্যাগম অসত্য;
    }
    যদি (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        প্রত্যাগম অসত্য;
    }
    যদি (src0->nb[0] != পরিমাণ(ভিন্ন) || src1->nb[0] != পরিমাণ(ভিন্ন) || dst->nb[0] != পরিমাণ(ভিন্ন) || src2->nb[0] != পরিমাণ(int32_t)) {
        প্রত্যাগম অসত্য;
    }

    // REVISIT: add support for non-contiguous tensors
    যদি (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        প্রত্যাগম অসত্য;
    }
    যদি (src2->nb[1] < src2->ne[0] * পরিমাণ(int32_t)) {
        প্রত্যাগম অসত্য;
    }

    সংরচনা htp_দ্বিমিক_kernel_params kparams;
    প্রত্যাগম ggml_hexagon_precompute_দ্বিমিক_params(sess, HTP_OP_ADD_ID, src0, src1, dst, &kparams, src2);
}

স্থির তার্কিক ggml_hexagon_supported_unary(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        প্রত্যাগম অসত্য;
    }
    যদি (dst->type != src0->type) {
        প্রত্যাগম অসত্য;
    }
    যদি (!ggml_is_contiguous_rows(src0)) {
        প্রত্যাগম অসত্য;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    যদি (src0->type == GGML_TYPE_F16) {
        বিকল্প (op->op) {
            পক্ষ GGML_OP_NORM:
            পক্ষ GGML_OP_RMS_NORM:
            পক্ষ GGML_OP_L2_NORM:
            পক্ষ GGML_OP_SCALE:
            পক্ষ GGML_OP_CLAMP:
            পক্ষ GGML_OP_SQR:
            পক্ষ GGML_OP_SQRT:
            পক্ষ GGML_OP_LOG:
                ভঙ্গ;
            পক্ষ GGML_OP_UNARY:
                যদি (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_STEP) {
                    প্রত্যাগম অসত্য;
                }
                ভঙ্গ;
            স্বভাব:
                প্রত্যাগম অসত্য;
        }
    }

    যদি (!ggml_are_same_shape(src0, dst)) {
        প্রত্যাগম অসত্য;
    }

    // dst must be contiguous; src0 may be non-contiguous
    যদি (!ggml_is_contiguous(dst)) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_sum(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src0->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }
    যদি (dst->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    যদি (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_sum_rows(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src0->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }
    যদি (dst->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    // TODO: add support for non-contigiuos tensors
    যদি (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_argmax(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src0->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }
    যদি (dst->type != GGML_TYPE_I32) {
        প্রত্যাগম অসত্য;
    }

    যদি (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_pool_2d(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব int32_t * params = op->op_params;

    যদি (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        প্রত্যাগম অসত্য;
    }
    যদি (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }
    যদি (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        প্রত্যাগম অসত্য;
    }
    যদি (!ggml_hexagon_pool_shape_fits(src0, op)) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব int32_t kernel_x = params[1];
    ধ্রুব int32_t kernel_y = params[2];
    ধ্রুব int32_t stride_x = params[3];
    ধ্রুব int32_t stride_y = params[4];
    ধ্রুব int32_t pad_x    = params[5];
    ধ্রুব int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    যদি (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    ধ্রুব uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    সংরচনা htp_pool_vtcm_layout layout;
    প্রত্যাগম htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_threads, sess->vtcm_size);
}

স্থির তার্কিক ggml_hexagon_supported_pool_1d(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব int32_t * params = op->op_params;

    যদি (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        প্রত্যাগম অসত্য;
    }
    যদি (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }
    যদি (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        প্রত্যাগম অসত্য;
    }
    যদি (!ggml_hexagon_pool_shape_fits(src0, op)) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব int32_t kernel = params[1];
    ধ্রুব int32_t stride = params[2];
    ধ্রুব int32_t pad    = params[3];
    যদি (kernel <= 0 || stride <= 0 || pad < 0) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ধ্রুব uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    সংরচনা htp_pool_vtcm_layout layout;
    প্রত্যাগম htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_threads, sess->vtcm_size);
}

স্থির তার্কিক ggml_hexagon_supported_activations(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        প্রত্যাগম অসত্য;
    }
    যদি (dst->type != src0->type) {
        প্রত্যাগম অসত্য;
    }

    যদি (!ggml_is_contiguous_1(src0)) {
        প্রত্যাগম অসত্য;
    }
    যদি (!ggml_is_contiguous(dst)) {
        প্রত্যাগম অসত্য;
    }

    যদি (src1) {
        যদি (src1->type != src0->type) {
            প্রত্যাগম অসত্য;
        }
        যদি (!ggml_are_same_shape(src0, src1)) {
            প্রত্যাগম অসত্য;
        }
        যদি (!ggml_is_contiguous_1(src1)) {
            প্রত্যাগম অসত্য;
        }
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_softmax(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * src2 = op->src[2];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src2) {
        প্রত্যাগম অসত্য;  // FIXME: add support for sinks
    }

    যদি (src0->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }
    যদি (dst->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    যদি (src0->ne[2] > 512) {
        প্রত্যাগম অসত্য;
    }

    যদি (src1) {
        যদি (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            প্রত্যাগম অসত্য;
        }
        যদি (src0->ne[0] != src1->ne[0]) {
            প্রত্যাগম অসত্য;
        }
        যদি (src1->ne[1] < src0->ne[1]) {
            প্রত্যাগম অসত্য;
        }
        যদি (src0->ne[2] % src1->ne[2] != 0) {
            প্রত্যাগম অসত্য;
        }
        যদি (src0->ne[3] % src1->ne[3] != 0) {
            প্রত্যাগম অসত্য;
        }
    }

    যদি (src1) {
        যদি (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            প্রত্যাগম অসত্য;
        }
    } অন্যথা {
        যদি (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            প্রত্যাগম অসত্য;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    ধ্রুব int64_t ne0 = src0->ne[0];
    যদি (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        প্রত্যাগম অসত্য;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    যদি (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুব uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    সংরচনা htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != nullptr, src1 && src1->type == GGML_TYPE_F16, n_threads);
    যদি (layout.total_bytes > sess->vtcm_size) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_set_rows(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0]; //মানসমূহ
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1]; // indices
    ধ্রুব সংরচনা ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    যদি (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        প্রত্যাগম অসত্য;
    }

    যদি (src0->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    যদি (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        প্রত্যাগম অসত্য;
    }

    যদি (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_get_rows(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0]; //মানসমূহ
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1]; // indices
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        প্রত্যাগম অসত্য;
    }

    যদি (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(dst)) {
        প্রত্যাগম অসত্য;
    }

    যদি (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        প্রত্যাগম অসত্য;
    }

    যদি (src0->type == dst->type) {
        যদি (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            প্রত্যাগম অসত্য;
        }
    } অন্যথা যদি (src0->type == GGML_TYPE_I32) {
        প্রত্যাগম অসত্য;
    } অন্যথা যদি (dst->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    যদি (ggml_is_empty(op)) {
        প্রত্যাগম সত্য;
    }

    সংরচনা htp_get_rows_kernel_params kparams;
    ggml_hexagon_precompute_get_rows_params(sess, src0, src1, dst, &kparams);
    যদি (kparams.n_threads == 0 || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_argsort(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0]; //মানসমূহ
    ধ্রুব সংরচনা ggml_tensor * dst  = op;         // indices

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুব uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ধ্রুব size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    সংরচনা htp_sort_vtcm_layout layout;
    যদি (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, অসত্য)) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;
}

স্থির তার্কিক ggml_hexagon_supported_শীর্ষ_k(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0]; //মানসমূহ
    ধ্রুব সংরচনা ggml_tensor * dst  = op;         // indices

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুব uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ধ্রুব size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    সংরচনা htp_sort_vtcm_layout layout;
    যদি (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, সত্য)) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;
}

স্থির তার্কিক ggml_hexagon_supported_rope(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * src2 = op->src[2];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (!ggml_are_same_shape(src0, dst)) {
        প্রত্যাগম অসত্য;
    }

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        প্রত্যাগম অসত্য;
    }

    যদি (src0->ne[0] <= 0) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    যদি (src0_nrows == 0) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব int32_t * op_params = &op->op_params[0];
    ধ্রুব পূর্ণাঙ্ক n_dims = op_params[1];
    ধ্রুব পূর্ণাঙ্ক mode   = op_params[2];
    ধ্রুব পূর্ণাঙ্ক n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    যদি (n_dims < 0 || n_dims % 2 != 0) {
        প্রত্যাগম অসত্য;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    যদি (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        প্রত্যাগম অসত্য;
    }

    ভিন্ন freq_base;
    memcpy(&freq_base, op_params + 5, পরিমাণ(ভিন্ন));
    যদি (freq_base < 0.0f) {
        প্রত্যাগম অসত্য;
    }

    যদি (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব তার্কিক is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    যদি (mode == GGML_ROPE_TYPE_VISION) {
        যদি (n_dims != (পূর্ণাঙ্ক) (src0->ne[0] / 2) || n_offs != 0) {
            প্রত্যাগম অসত্য;
        }
    }

    যদি (is_mrope) {
        ধ্রুব int32_t * sections = op_params + 11;
        যদি (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            প্রত্যাগম অসত্য;
        }
    }

    ধ্রুব int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    যদি (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        প্রত্যাগম অসত্য;
    }

    যদি (src2) {
        যদি (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            প্রত্যাগম অসত্য;
        }
        যদি (src2->ne[0] < (n_dims / 2)) {
            প্রত্যাগম অসত্য;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    যদি (src0->nb[0] != পরিমাণ(ভিন্ন) || dst->nb[0] != পরিমাণ(ভিন্ন)) {
        প্রত্যাগম অসত্য;
    }
    যদি (src0->nb[1] < src0->ne[0] * পরিমাণ(ভিন্ন) || dst->nb[1] < dst->ne[0] * পরিমাণ(ভিন্ন)) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ধ্রুব uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    সংরচনা htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    যদি (layout.total_bytes > sess->vtcm_size) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;
}

স্থির তার্কিক ggml_hexagon_supported_ssm_conv(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    // Only support FP32 for now
    যদি (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    // Check IO tensor shapes and dims
    যদি (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        প্রত্যাগম অসত্য; // src0 should be effectively 3D
    }

    ধ্রুব পূর্ণাঙ্ক d_conv  = src1->ne[0];
    ধ্রুব পূর্ণাঙ্ক d_inner = src0->ne[1];
    ধ্রুব পূর্ণাঙ্ক n_t = dst->ne[1];
    ধ্রুব পূর্ণাঙ্ক n_s = dst->ne[2];

    যদি (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        প্রত্যাগম অসত্য;
    }
    যদি (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        প্রত্যাগম অসত্য;
    }
    যদি (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        প্রত্যাগম অসত্য;
    }
    যদি (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        প্রত্যাগম অসত্য;
    }
    যদি (src0->nb[0] != পরিমাণ(ভিন্ন) || src1->nb[0] != পরিমাণ(ভিন্ন) || dst->nb[0] != পরিমাণ(ভিন্ন)) {
        প্রত্যাগম অসত্য;
    }
    যদি (src0->nb[1] != src0->ne[0] * পরিমাণ(ভিন্ন) || src1->nb[1] != src1->ne[0] * পরিমাণ(ভিন্ন)) {
        প্রত্যাগম অসত্য;
    }

    সংরচনা htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    যদি ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;
}

স্থির তার্কিক im2col_3d_calc_output_size(uint32_t প্রবেশ, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    যদি (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব int64_t numerator = (int64_t) প্রবেশ + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    যদি (numerator < 0) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint64_t size = (uint64_t) (numerator / stride) + 1;
    যদি (size == 0 || size > UINT32_MAX) {
        প্রত্যাগম অসত্য;
    }

    *output = (uint32_t) size;
    প্রত্যাগম সত্য;
}

স্থির তার্কিক ggml_hexagon_precompute_im2col_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (!src0 || !src1 || !dst) {
        প্রত্যাগম অসত্য;
    }

    যদি (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        প্রত্যাগম অসত্য;
    }

    যদি (src1->nb[0] != পরিমাণ(ভিন্ন) || !ggml_is_contiguous(dst)) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব int32_t * op_params = (ধ্রুব int32_t *) op->op_params;
    ধ্রুব int32_t s0 = op_params[0];
    ধ্রুব int32_t s1 = op_params[1];
    ধ্রুব int32_t p0 = op_params[2];
    ধ্রুব int32_t p1 = op_params[3];
    ধ্রুব int32_t d0 = op_params[4];
    ধ্রুব int32_t d1 = op_params[5];
    ধ্রুব তার্কিক is_2D = (op_params[6] == 1);

    যদি (s0 <= 0 || d0 <= 0 || p0 < 0) {
        প্রত্যাগম অসত্য;
    }
    যদি (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    ধ্রুব uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    ধ্রুব uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    ধ্রুব uint32_t IW = (uint32_t) src1->ne[0];
    ধ্রুব uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    ধ্রুব uint32_t KW = (uint32_t) src0->ne[0];
    ধ্রুব uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    ধ্রুব uint32_t OW = (uint32_t) dst->ne[1];

    যদি (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    ধ্রুব uint64_t total_rows_64   = (uint64_t) N * OH;
    যদি (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX) {
        প্রত্যাগম অসত্য;
    }
    ধ্রুব uint32_t patch_stride = (uint32_t) patch_stride_64;
    ধ্রুব uint32_t total_rows   = (uint32_t) total_rows_64;

    যদি (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        প্রত্যাগম অসত্য;
    }
    যদি (is_2D && dst->ne[2] != OH) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? পরিমাণ(ggml_fp16_t) : পরিমাণ(ভিন্ন);

    তার্কিক is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    যদি (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    ধ্রুব uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    ধ্রুব uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    ধ্রুব uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    যদি (is_patchembed) {
        ধ্রুব uint64_t src_row_bytes_raw = (uint64_t) IC * KH * IW * পরিমাণ(ভিন্ন);
        ধ্রুব uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        যদি (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            ধ্রুব uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            ধ্রুব uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            সংরচনা htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            যদি (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
                kparams->kernel_type          = HTP_IM2COL_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = is_2D ? 1 : 0;
                kparams->is_3d                = 0;
                kparams->n_threads            = (uint8_t) n_threads;
                kparams->vtcm_size            = layout.total_bytes;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_bytes_per_thread = layout.src_bytes_per_thread;
                kparams->dst_bytes_per_thread = layout.dst_bytes_per_thread;
                kparams->src_row_bytes        = src_row_bytes;
                kparams->dst_row_bytes        = dst_row_bytes;
                kparams->owb                  = OW;
                kparams->wb                   = IW;
                kparams->nbpr                 = 1;
                kparams->N                    = N;
                kparams->IC                   = IC;
                kparams->ID                   = 1;
                kparams->IH                   = IH;
                kparams->IW                   = IW;
                kparams->KD                   = 1;
                kparams->KH                   = KH;
                kparams->KW                   = KW;
                kparams->OD                   = 1;
                kparams->OH                   = OH;
                kparams->OW                   = OW;
                kparams->patch_stride         = patch_stride;
                kparams->total_rows           = total_rows;
                প্রত্যাগম সত্য;
            }
        }
    }

    প্রতি (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ধ্রুব uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        যদি (Wb > UINT32_MAX) {
            যদি (owb == 1) ভঙ্গ;
            অনুবর্ত;
        }

        ধ্রুব uint64_t src_row_bytes_raw = (uint64_t) IC * KH * Wb * পরিমাণ(ভিন্ন);
        ধ্রুব uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        যদি (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            যদি (owb == 1) ভঙ্গ;
            অনুবর্ত;
        }

        ধ্রুব uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        ধ্রুব uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        সংরচনা htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        যদি (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
            kparams->kernel_type          = HTP_IM2COL_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = is_2D ? 1 : 0;
            kparams->is_3d                = 0;
            kparams->n_threads            = (uint8_t) n_threads;
            kparams->vtcm_size            = layout.total_bytes;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_bytes_per_thread = layout.src_bytes_per_thread;
            kparams->dst_bytes_per_thread = layout.dst_bytes_per_thread;
            kparams->src_row_bytes        = src_row_bytes;
            kparams->dst_row_bytes        = dst_row_bytes;
            kparams->owb                  = owb;
            kparams->wb                   = (uint32_t) Wb;
            kparams->nbpr                 = (OW + owb - 1) / owb;
            kparams->N                    = N;
            kparams->IC                   = IC;
            kparams->ID                   = 1;
            kparams->IH                   = IH;
            kparams->IW                   = IW;
            kparams->KD                   = 1;
            kparams->KH                   = KH;
            kparams->KW                   = KW;
            kparams->OD                   = 1;
            kparams->OH                   = OH;
            kparams->OW                   = OW;
            kparams->patch_stride         = patch_stride;
            kparams->total_rows           = total_rows;
            প্রত্যাগম সত্য;
        }
        যদি (owb == 1) ভঙ্গ;
    }

    প্রত্যাগম অসত্য;
}

স্থির তার্কিক ggml_hexagon_precompute_im2col_3d_params(
    ধ্রুব সংরচনা ggml_hexagon_session * sess,
    ধ্রুব সংরচনা ggml_tensor * op,
    সংরচনা htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, পরিমাণ(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (!src0 || !src1 || !dst) {
        প্রত্যাগম অসত্য;
    }

    যদি (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        প্রত্যাগম অসত্য;
    }

    যদি (src1->nb[0] != পরিমাণ(ভিন্ন) || !ggml_is_contiguous(dst)) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব int32_t * params = (ধ্রুব int32_t *) op->op_params;
    ধ্রুব int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    ধ্রুব int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    ধ্রুব int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    ধ্রুব int32_t ic = params[9];

    যদি (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        প্রত্যাগম অসত্য;
    }

    যদি (src1->ne[3] % ic != 0) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t IC = (uint32_t) ic;
    ধ্রুব uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    যদি (N == 0) {
        প্রত্যাগম অসত্য;
    }

    uint32_t OW, OH, OD;
    যদি (!im2col_3d_calc_output_size(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_size(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_size(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t KD = (uint32_t) src0->ne[2];
    ধ্রুব uint32_t KH = (uint32_t) src0->ne[1];
    ধ্রুব uint32_t KW = (uint32_t) src0->ne[0];
    ধ্রুব uint32_t ID = (uint32_t) src1->ne[2];
    ধ্রুব uint32_t IH = (uint32_t) src1->ne[1];
    ধ্রুব uint32_t IW = (uint32_t) src1->ne[0];

    ধ্রুব uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    ধ্রুব uint64_t total_rows_64   = (uint64_t) N * OD * OH;
    ধ্রুব uint64_t dst_ne3_64      = (uint64_t) OD * N;

    যদি (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        প্রত্যাগম অসত্য;
    }
    ধ্রুব uint32_t patch_stride = (uint32_t) patch_stride_64;
    ধ্রুব uint32_t total_rows   = (uint32_t) total_rows_64;
    ধ্রুব uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    যদি (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? পরিমাণ(ggml_fp16_t) : পরিমাণ(ভিন্ন);

    ধ্রুব তার্কিক is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    ধ্রুব uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    ধ্রুব uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    ধ্রুব uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    যদি (is_patchembed) {
        ধ্রুব uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * IW * পরিমাণ(ভিন্ন);
        ধ্রুব uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        যদি (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            ধ্রুব uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            ধ্রুব uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            সংরচনা htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            যদি (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
                kparams->kernel_type          = HTP_IM2COL_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = 0;
                kparams->is_3d                = 1;
                kparams->n_threads            = (uint8_t) n_threads;
                kparams->vtcm_size            = layout.total_bytes;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_bytes_per_thread = layout.src_bytes_per_thread;
                kparams->dst_bytes_per_thread = layout.dst_bytes_per_thread;
                kparams->src_row_bytes        = src_row_bytes;
                kparams->dst_row_bytes        = dst_row_bytes;
                kparams->owb                  = OW;
                kparams->wb                   = IW;
                kparams->nbpr                 = 1;
                kparams->N                    = N;
                kparams->IC                   = IC;
                kparams->ID                   = ID;
                kparams->IH                   = IH;
                kparams->IW                   = IW;
                kparams->KD                   = KD;
                kparams->KH                   = KH;
                kparams->KW                   = KW;
                kparams->OD                   = OD;
                kparams->OH                   = OH;
                kparams->OW                   = OW;
                kparams->patch_stride         = patch_stride;
                kparams->total_rows           = total_rows;
                প্রত্যাগম সত্য;
            }
        }
    }

    প্রতি (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ধ্রুব uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        যদি (Wb > UINT32_MAX) {
            যদি (owb == 1) ভঙ্গ;
            অনুবর্ত;
        }

        ধ্রুব uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * Wb * পরিমাণ(ভিন্ন);
        ধ্রুব uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        যদি (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            যদি (owb == 1) ভঙ্গ;
            অনুবর্ত;
        }

        ধ্রুব uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        ধ্রুব uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        সংরচনা htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        যদি (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
            kparams->kernel_type          = HTP_IM2COL_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = 0;
            kparams->is_3d                = 1;
            kparams->n_threads            = (uint8_t) n_threads;
            kparams->vtcm_size            = layout.total_bytes;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_bytes_per_thread = layout.src_bytes_per_thread;
            kparams->dst_bytes_per_thread = layout.dst_bytes_per_thread;
            kparams->src_row_bytes        = src_row_bytes;
            kparams->dst_row_bytes        = dst_row_bytes;
            kparams->owb                  = owb;
            kparams->wb                   = (uint32_t) Wb;
            kparams->nbpr                 = (OW + owb - 1) / owb;
            kparams->N                    = N;
            kparams->IC                   = IC;
            kparams->ID                   = ID;
            kparams->IH                   = IH;
            kparams->IW                   = IW;
            kparams->KD                   = KD;
            kparams->KH                   = KH;
            kparams->KW                   = KW;
            kparams->OD                   = OD;
            kparams->OH                   = OH;
            kparams->OW                   = OW;
            kparams->patch_stride         = patch_stride;
            kparams->total_rows           = total_rows;
            প্রত্যাগম সত্য;
        }
        যদি (owb == 1) ভঙ্গ;
    }

    প্রত্যাগম অসত্য;
}

স্থির তার্কিক ggml_hexagon_supported_im2col(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    সংরচনা htp_im2col_kernel_params kparams;
    প্রত্যাগম ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

স্থির তার্কিক ggml_hexagon_supported_im2col_3d(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    সংরচনা htp_im2col_kernel_params kparams;
    প্রত্যাগম ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

স্থির তার্কিক ggml_hexagon_supported_pad(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    ধ্রুব int32_t lp0 = ((ধ্রুব int32_t *) op->op_params)[0];
    ধ্রুব int32_t rp0 = ((ধ্রুব int32_t *) op->op_params)[1];
    ধ্রুব int32_t circular = ((ধ্রুব int32_t *) op->op_params)[8];

    যদি (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_cumsum(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    যদি (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_diag(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    // diag only supports F32 currently
    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    // Input must have ne[1] == 1 (vector input)
    যদি (src0->ne[1] != 1) {
        প্রত্যাগম অসত্য;
    }

    // Output must be square in first two dimensions
    যদি (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_solve_tri(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0]; // A
    ধ্রুব সংরচনা ggml_tensor * src1 = op->src[1]; // B
    ধ্রুব সংরচনা ggml_tensor * dst  = op;         // X

    যদি (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    যদি (src0->ne[0] != src0->ne[1]) {
        প্রত্যাগম অসত্য;
    }

    যদি (src0->ne[1] != src1->ne[1]) {
        প্রত্যাগম অসত্য;
    }

    যদি (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        প্রত্যাগম অসত্য;
    }

    যদি (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_tri(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {

    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src0->type != GGML_TYPE_F32) { প্রত্যাগম অসত্য; }
    যদি (dst->type  != GGML_TYPE_F32) { প্রত্যাগম অসত্য; }
    যদি (!ggml_are_same_shape(src0, dst)) { প্রত্যাগম অসত্য; }
    যদি (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { প্রত্যাগম অসত্য; }

    প্রত্যাগম সত্য;

    GGML_UNUSED(sess);
}

স্থির ধ্রুব বর্ণ * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    প্রত্যাগম sess->c_name();
}

স্থির শূন্য ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

স্থির htp_op_code op_remap_to_htp(ধ্রুব ggml_tensor * t) {
    বিকল্প (t->op) {
        পক্ষ GGML_OP_FLASH_ATTN_EXT:  প্রত্যাগম HTP_OP_FLASH_ATTN_EXT;
        পক্ষ GGML_OP_MUL_MAT:
            যদি (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                প্রত্যাগম HTP_OP_HADAMARD;
            }
            প্রত্যাগম HTP_OP_MUL_MAT;
        পক্ষ GGML_OP_MUL_MAT_ID:      প্রত্যাগম HTP_OP_MUL_MAT_ID;
        পক্ষ GGML_OP_MUL:             প্রত্যাগম HTP_OP_MUL;
        পক্ষ GGML_OP_ADD:             প্রত্যাগম HTP_OP_ADD;
        পক্ষ GGML_OP_ADD_ID:          প্রত্যাগম HTP_OP_ADD_ID;
        পক্ষ GGML_OP_SUB:             প্রত্যাগম HTP_OP_SUB;
        পক্ষ GGML_OP_DIV:             প্রত্যাগম HTP_OP_DIV;
        পক্ষ GGML_OP_DUP:
        পক্ষ GGML_OP_CPY:             প্রত্যাগম HTP_OP_CPY;
        পক্ষ GGML_OP_CONT:            প্রত্যাগম HTP_OP_CPY;
        পক্ষ GGML_OP_GET_ROWS:        প্রত্যাগম HTP_OP_GET_ROWS;
        পক্ষ GGML_OP_SET_ROWS:        প্রত্যাগম HTP_OP_SET_ROWS;
        পক্ষ GGML_OP_SUM:             প্রত্যাগম HTP_OP_SUM;
        পক্ষ GGML_OP_SUM_ROWS:        প্রত্যাগম HTP_OP_SUM_ROWS;
        পক্ষ GGML_OP_POOL_2D:         প্রত্যাগম HTP_OP_POOL_2D;
        পক্ষ GGML_OP_POOL_1D:         প্রত্যাগম HTP_OP_POOL_1D;
        পক্ষ GGML_OP_ARGSORT:         প্রত্যাগম HTP_OP_ARGSORT;
        পক্ষ GGML_OP_শীর্ষ_K:           প্রত্যাগম HTP_OP_শীর্ষ_K;
        পক্ষ GGML_OP_ARGMAX:          প্রত্যাগম HTP_OP_ARGMAX;
        পক্ষ GGML_OP_NORM:            প্রত্যাগম HTP_OP_NORM;
        পক্ষ GGML_OP_L2_NORM:         প্রত্যাগম HTP_OP_L2_NORM;
        পক্ষ GGML_OP_RMS_NORM:        প্রত্যাগম HTP_OP_RMS_NORM;
        পক্ষ GGML_OP_CONCAT:          প্রত্যাগম HTP_OP_CONCAT;
        পক্ষ GGML_OP_SCALE:           প্রত্যাগম HTP_OP_SCALE;
        পক্ষ GGML_OP_CLAMP:           প্রত্যাগম HTP_OP_CLAMP;
        পক্ষ GGML_OP_LEAKY_RELU:      প্রত্যাগম HTP_OP_LEAKY_RELU;
        পক্ষ GGML_OP_SQR:             প্রত্যাগম HTP_OP_SQR;
        পক্ষ GGML_OP_SQRT:            প্রত্যাগম HTP_OP_SQRT;
        পক্ষ GGML_OP_LOG:             প্রত্যাগম HTP_OP_UNARY_LOG;
        পক্ষ GGML_OP_SOFT_MAX:        প্রত্যাগম HTP_OP_SOFTMAX;
        পক্ষ GGML_OP_SSM_CONV:        প্রত্যাগম HTP_OP_SSM_CONV;
        পক্ষ GGML_OP_GATED_DELTA_NET: প্রত্যাগম HTP_OP_GATED_DELTA_NET;
        পক্ষ GGML_OP_ROPE:            প্রত্যাগম HTP_OP_ROPE;
        পক্ষ GGML_OP_REPEAT:          প্রত্যাগম HTP_OP_REPEAT;
        পক্ষ GGML_OP_CUMSUM:          প্রত্যাগম HTP_OP_CUMSUM;
        পক্ষ GGML_OP_FILL:            প্রত্যাগম HTP_OP_FILL;
        পক্ষ GGML_OP_DIAG:            প্রত্যাগম HTP_OP_DIAG;
        পক্ষ GGML_OP_SOLVE_TRI:       প্রত্যাগম HTP_OP_SOLVE_TRI;
        পক্ষ GGML_OP_TRI:             প্রত্যাগম HTP_OP_TRI;
        পক্ষ GGML_OP_PAD:             প্রত্যাগম HTP_OP_PAD;
        পক্ষ GGML_OP_IM2COL:          প্রত্যাগম HTP_OP_IM2COL;
        পক্ষ GGML_OP_IM2COL_3D:       প্রত্যাগম HTP_OP_IM2COL_3D;
        পক্ষ GGML_OP_ROLL:            প্রত্যাগম HTP_OP_ROLL;

        পক্ষ GGML_OP_UNARY:
            বিকল্প (ggml_get_unary_op(t)) {
                পক্ষ GGML_UNARY_OP_SILU:       প্রত্যাগম HTP_OP_UNARY_SILU;
                পক্ষ GGML_UNARY_OP_GELU:       প্রত্যাগম HTP_OP_UNARY_GELU;
                পক্ষ GGML_UNARY_OP_GELU_QUICK: প্রত্যাগম HTP_OP_UNARY_GELU_QUICK;
                পক্ষ GGML_UNARY_OP_GELU_ERF:   প্রত্যাগম HTP_OP_UNARY_GELU_ERF;
                পক্ষ GGML_UNARY_OP_SIGMOID:    প্রত্যাগম HTP_OP_UNARY_SIGMOID;
                পক্ষ GGML_UNARY_OP_NEG:        প্রত্যাগম HTP_OP_UNARY_NEG;
                পক্ষ GGML_UNARY_OP_EXP:        প্রত্যাগম HTP_OP_UNARY_EXP;
                পক্ষ GGML_UNARY_OP_SOFTPLUS:   প্রত্যাগম HTP_OP_UNARY_SOFTPLUS;
                পক্ষ GGML_UNARY_OP_TANH:       প্রত্যাগম HTP_OP_UNARY_TANH;
                পক্ষ GGML_UNARY_OP_ABS:        প্রত্যাগম HTP_OP_UNARY_ABS;
                পক্ষ GGML_UNARY_OP_RELU:       প্রত্যাগম HTP_OP_UNARY_RELU;
                পক্ষ GGML_UNARY_OP_STEP:       প্রত্যাগম HTP_OP_UNARY_STEP;
            স্বভাব:
                ভঙ্গ;
            }
            ভঙ্গ;

        পক্ষ GGML_OP_GLU:
            বিকল্প (ggml_get_glu_op(t)) {
                পক্ষ GGML_GLU_OP_SWIGLU:       প্রত্যাগম HTP_OP_GLU_SWIGLU;
                পক্ষ GGML_GLU_OP_SWIGLU_OAI:   প্রত্যাগম HTP_OP_GLU_SWIGLU_OAI;
                পক্ষ GGML_GLU_OP_SWIGLU_CLAMP: প্রত্যাগম HTP_OP_GLU_SWIGLU_CLAMP;
                পক্ষ GGML_GLU_OP_GEGLU:        প্রত্যাগম HTP_OP_GLU_GEGLU;
                পক্ষ GGML_GLU_OP_GEGLU_QUICK:  প্রত্যাগম HTP_OP_GLU_GEGLU_QUICK;
                পক্ষ GGML_GLU_OP_GEGLU_ERF:    প্রত্যাগম HTP_OP_GLU_GEGLU_ERF;
                স্বভাব: ভঙ্গ;
            }
            ভঙ্গ;

        স্বভাব:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    প্রত্যাগম HTP_OP_INVALID;
}

স্থির অন্তঃস্থ তার্কিক op_is_compute(ggml_tensor *গ্রন্থি)
{
    প্রত্যাগম !ggml_op_is_empty(গ্রন্থি->op) && !ggml_is_empty(গ্রন্থি) && (গ্রন্থি->flags & GGML_TENSOR_FLAG_COMPUTE);
}

স্থির তার্কিক mm_is_hmx_eligible(ধ্রুব ggml_tensor * t) {
    যদি (opt_nhmx == 0) { প্রত্যাগম অসত্য; }

    ধ্রুব ggml_tensor * src0 = t->src[0];
    ধ্রুব ggml_tensor * src1 = t->src[1];

    যদি (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        ধ্রুব ggml_tensor src1_c = ggml_hexagon_tensor_collapse_rows(src1);
        প্রত্যাগম ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, অসত্য, অসত্য);
    }

    ধ্রুব তার্কিক is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    ধ্রুব তার্কিক is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    প্রত্যাগম ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

স্থির তার্কিক is_supported_mul_mat_nx_kernel(ধ্রুব ggml_tensor * src0, ধ্রুব সংরচনা htp_mm_kernel_params * kparams) {
    যদি (kparams->n_hmx) {
        প্রত্যাগম kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    যদি (!ggml_hexagon_tensor_is_tiled(src0)) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
}

স্থির তার্কিক is_supported_mul_mat_id_nx_kernel(ধ্রুব ggml_tensor * src0, ধ্রুব সংরচনা htp_mm_kernel_params * kparams) {
    যদি (kparams->n_hmx) {
        প্রত্যাগম kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    যদি (!ggml_hexagon_tensor_is_tiled(src0)) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK;
}

স্থির তার্কিক is_mergeable_mul_mat(ধ্রুব ggml_tensor * t) {
    যদি (t->op != GGML_OP_MUL_MAT) প্রত্যাগম অসত্য;
    যদি (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) প্রত্যাগম অসত্য;

    ধ্রুব ggml_tensor * src0 = t->src[0];
    ধ্রুব ggml_tensor * src1 = t->src[1];
    যদি (src0->ne[2] != 1 || src0->ne[3] != 1) প্রত্যাগম অসত্য;

    যদি (mm_is_hmx_eligible(t)) {
        প্রত্যাগম htp_is_hmx_weight_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    যদি (src1->type != GGML_TYPE_F32) প্রত্যাগম অসত্য;

    প্রত্যাগম ggml_hexagon_tensor_is_tiled(src0);
}

স্থির তার্কিক is_mergeable_mul_mat_pair(ধ্রুব ggml_tensor * n1, ধ্রুব ggml_tensor * n2) {
    যদি (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        প্রত্যাগম অসত্য;
    }
    যদি (n1->src[1] != n2->src[1]) {
        প্রত্যাগম অসত্য;
    }
    যদি (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        প্রত্যাগম অসত্য;
    }
    যদি (n1->src[0]->type != n2->src[0]->type) {
        প্রত্যাগম অসত্য;
    }
    যদি (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        প্রত্যাগম অসত্য;
    }
    যদি (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        প্রত্যাগম অসত্য;
    }
    প্রত্যাগম সত্য;
}

স্থির তার্কিক is_mergeable_mul_mat_id(ধ্রুব ggml_tensor * t) {
    যদি (t->op != GGML_OP_MUL_MAT_ID) প্রত্যাগম অসত্য;

    ধ্রুব ggml_tensor * src0 = t->src[0];
    যদি (mm_is_hmx_eligible(t)) {
        প্রত্যাগম htp_is_hmx_weight_type(src0->type);
    }

    প্রত্যাগম ggml_hexagon_tensor_is_tiled(src0);
}

স্থির তার্কিক is_mergeable_mul_mat_id_pair(ধ্রুব ggml_tensor * n1, ধ্রুব ggml_tensor * n2) {
    যদি (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        প্রত্যাগম অসত্য;
    }
    যদি (n1->src[1] != n2->src[1]) {
        প্রত্যাগম অসত্য;
    }
    যদি (n1->src[2] != n2->src[2]) {
        প্রত্যাগম অসত্য;
    }
    যদি (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        প্রত্যাগম অসত্য;
    }
    যদি (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        প্রত্যাগম অসত্য;
    }
    যদি (n1->src[0]->type != n2->src[0]->type) {
        প্রত্যাগম অসত্য;
    }
    যদি (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        প্রত্যাগম অসত্য;
    }
    যদি (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        প্রত্যাগম অসত্য;
    }
    প্রত্যাগম সত্য;
}

স্থির ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    যদি (sess->last_error > HTP_STATUS_OK) {
        প্রত্যাগম GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    ধ্রুব std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    তার্কিক cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
    যদি (cache_hit) {
        nodes_ptr = &sess->cached_nodes;
    } অন্যথা {
        // Tag fusable tensors in graph
        প্রতি (পূর্ণাঙ্ক i = 0; i < graph->n_nodes; i++) {
            auto * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            যদি (!extra) অনুবর্ত;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            যদি (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } অন্যথা যদি ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                যদি ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_গ্রন্থি_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        প্রতি (পূর্ণাঙ্ক i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            যদি (!op_is_compute(n)) {
                অনুবর্ত;
            }

            htp_opnode গ্রন্থি(HTP_OP_INVALID, n);
            গ্রন্থি.opcode = op_remap_to_htp(n);
            যদি (গ্রন্থি.opcode == HTP_OP_MUL_MAT || গ্রন্থি.opcode == HTP_OP_MUL_MAT_ID || গ্রন্থি.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = গ্রন্থি.গ্রন্থি->src[0];
                যদি (ggml_hexagon_needs_repack(src0)) {
                    যদি (opt_verbose) {
                        বর্ণ dims_buf[64];
                        বর্ণ strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, পরিমাণ(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, পরিমাণ(strides_buf), src0);
                        GGML_LOG_DEBUG("ggml-hex: %s rt-repack %s|%s|%s|%s|size %zu\n",
                                       sess->c_name(), src0->name, dims_buf, ggml_type_name(src0->type), strides_buf, ggml_nbytes(src0));
                    }
                    std::vector<uint8_t> raw(ggml_nbytes(src0));
                    memcpy(raw.data(), src0->data, raw.size());
                    htp_repack_tensor_tiled(src0, raw.data(), raw.size());
                    auto * extra = (ggml_hexagon_tensor_extra *) src0->extra;
                    extra->flags |= GGML_HEXAGON_TENSOR_TILED;
                }
            }

            যদি (গ্রন্থি.opcode == HTP_OP_MUL_MAT || গ্রন্থি.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    গ্রন্থি.গ্রন্থি->src[0], গ্রন্থি.গ্রন্থি->src[1], গ্রন্থি.গ্রন্থি,
                    (সংরচনা htp_mm_kernel_params *)গ্রন্থি.kernel_params
                );
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_MUL || গ্রন্থি.opcode == HTP_OP_ADD || গ্রন্থি.opcode == HTP_OP_ADD_ID || গ্রন্থি.opcode == HTP_OP_SUB || গ্রন্থি.opcode == HTP_OP_DIV) {
                ধ্রুব ggml_tensor * src1 = গ্রন্থি.গ্রন্থি->src[1];
                ধ্রুব ggml_tensor * src2 = (গ্রন্থি.opcode == HTP_OP_ADD_ID) ? গ্রন্থি.গ্রন্থি->src[2] : nullptr;
                GGML_ASSERT(ggml_hexagon_precompute_দ্বিমিক_params(sess,
                    গ্রন্থি.opcode, গ্রন্থি.গ্রন্থি->src[0], src1, গ্রন্থি.গ্রন্থি,
                    (সংরচনা htp_দ্বিমিক_kernel_params *) গ্রন্থি.kernel_params, src2));
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    গ্রন্থি.গ্রন্থি,
                    (সংরচনা htp_fa_kernel_params *)গ্রন্থি.kernel_params
                );
            } অন্যথা যদি (htp_op_is_unary(গ্রন্থি.opcode)) {
                auto inputs = গ্রন্থি.get_inputs();
                ধ্রুব সংরচনা ggml_tensor * src0 = inputs[0];
                ধ্রুব সংরচনা ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    গ্রন্থি.opcode, src0, src1, গ্রন্থি.dst(),
                    (সংরচনা htp_unary_kernel_params *)গ্রন্থি.kernel_params
                );
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_GET_ROWS) {
                ggml_hexagon_precompute_get_rows_params(sess,
                    গ্রন্থি.গ্রন্থি->src[0], গ্রন্থি.গ্রন্থি->src[1], গ্রন্থি.dst(),
                    (সংরচনা htp_get_rows_kernel_params *)গ্রন্থি.kernel_params
                );
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_SET_ROWS) {
                ggml_hexagon_precompute_set_rows_params(sess,
                    গ্রন্থি.গ্রন্থি->src[0], গ্রন্থি.গ্রন্থি->src[1], গ্রন্থি.dst(),
                    (সংরচনা htp_set_rows_kernel_params *)গ্রন্থি.kernel_params
                );
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    গ্রন্থি.গ্রন্থি,
                    (সংরচনা htp_rope_kernel_params *)গ্রন্থি.kernel_params
                );
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    গ্রন্থি.গ্রন্থি->src[0], গ্রন্থি.গ্রন্থি->src[1], গ্রন্থি.dst(),
                    (সংরচনা htp_ssm_conv_kernel_params *)গ্রন্থি.kernel_params
                );
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    গ্রন্থি.গ্রন্থি,
                    (সংরচনা htp_softmax_kernel_params *)গ্রন্থি.kernel_params
                );
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    গ্রন্থি.গ্রন্থি,
                    (সংরচনা htp_gdn_kernel_params *)গ্রন্থি.kernel_params
                );
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_ARGSORT || গ্রন্থি.opcode == HTP_OP_শীর্ষ_K) {
                ggml_hexagon_precompute_sort_params(sess,
                    গ্রন্থি.গ্রন্থি,
                    গ্রন্থি.opcode == HTP_OP_শীর্ষ_K,
                    (সংরচনা htp_sort_kernel_params *) গ্রন্থি.kernel_params
                );
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_POOL_2D || গ্রন্থি.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, গ্রন্থি.গ্রন্থি->src[0], গ্রন্থি.dst(),
                    (সংরচনা htp_pool_2d_kernel_params *)গ্রন্থি.kernel_params,
                    গ্রন্থি.opcode == HTP_OP_POOL_1D);
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    গ্রন্থি.গ্রন্থি,
                    (সংরচনা htp_concat_kernel_params *) গ্রন্থি.kernel_params
                );
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_CPY || গ্রন্থি.opcode == HTP_OP_CPY_FENCE) {
                ধ্রুব তার্কিক ok = ggml_hexagon_precompute_cpy_params(sess,
                    গ্রন্থি.গ্রন্থি,
                    (সংরচনা htp_copy_kernel_params *) গ্রন্থি.kernel_params
                );
                ধ্রুব auto * kparams = (ধ্রুব সংরচনা htp_copy_kernel_params *) গ্রন্থি.kernel_params;
                যদি (ok && গ্রন্থি.opcode == HTP_OP_CPY && kparams->total_elems == 0) {
                    অনুবর্ত;
                }
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_IM2COL) {
                ggml_hexagon_precompute_im2col_params(sess,
                    গ্রন্থি.গ্রন্থি,
                    (সংরচনা htp_im2col_kernel_params *) গ্রন্থি.kernel_params
                );
            } অন্যথা যদি (গ্রন্থি.opcode == HTP_OP_IM2COL_3D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    গ্রন্থি.গ্রন্থি,
                    (সংরচনা htp_im2col_kernel_params *) গ্রন্থি.kernel_params
                );
            }
            computed_nodes.push_back(std::move(গ্রন্থি));
        }

        যদি (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } অন্যথা {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    প্রতি (ধ্রুব auto & গ্রন্থি : *nodes_ptr) {
        sess->enqueue_op(গ্রন্থি);
    }

    যদি (sess->last_error > HTP_STATUS_OK) {
        প্রত্যাগম GGML_STATUS_FAILED;
    }

    প্রত্যাগম GGML_STATUS_SUCCESS;
}

স্থির শূন্য ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    যদি (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
}

গণনা ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

সংরচনা ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

সংরচনা ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    শূন্য reset() {
        ranges.clear();
    }

    শূন্য add(ধ্রুব ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    তার্কিক check(ধ্রুব ggml_hexagon_mem_range & mr) ধ্রুব {
        প্রতি (ধ্রুব auto & cmp : ranges) {
            যদি (mr.pb != cmp.pb) {
                অনুবর্ত;
            }
            যদি (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                অনুবর্ত;
            }
            যদি (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                প্রত্যাগম অসত্য;
            }
        }
        প্রত্যাগম সত্য;
    }
};

স্থির ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(ধ্রুব ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    ধ্রুব ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    যদি (tensor->buffer) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } অন্যথা {
        mr = {
            /*.pb =*/ (uint64_t) base,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    প্রত্যাগম mr;
}

স্থির শূন্য ggml_hexagon_mem_ranges_add_গ্রন্থি(ggml_hexagon_mem_ranges & mrs, ধ্রুব htp_opnode & গ্রন্থি) {
    যদি (গ্রন্থি.is_empty()) প্রত্যাগম;

    প্রতি (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
        যদি (গ্রন্থি.গ্রন্থি->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(গ্রন্থি.গ্রন্থি->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    প্রতি (ধ্রুব auto * fused : গ্রন্থি.fused) {
        প্রতি (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
            যদি (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(গ্রন্থি.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

স্থির তার্কিক ggml_hexagon_mem_ranges_check_গ্রন্থি(ধ্রুব ggml_hexagon_mem_ranges & mrs, ধ্রুব htp_opnode & গ্রন্থি) {
    যদি (গ্রন্থি.is_empty()) প্রত্যাগম সত্য;

    প্রতি (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
        যদি (গ্রন্থি.গ্রন্থি->src[i]) {
            যদি (!mrs.check(ggml_hexagon_mem_range_from_tensor(গ্রন্থি.গ্রন্থি->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                প্রত্যাগম অসত্য;
            }
        }
    }
    প্রতি (ধ্রুব auto * fused : গ্রন্থি.fused) {
        প্রতি (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
            যদি (fused->src[i]) {
                যদি (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    প্রত্যাগম অসত্য;
                }
            }
        }
    }
    প্রত্যাগম mrs.check(ggml_hexagon_mem_range_from_tensor(গ্রন্থি.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

স্থির std::vector<পূর্ণাঙ্ক> ggml_hexagon_graph_optimize_reorder(ধ্রুব std::vector<htp_opnode> & nodes) {
    ধ্রুব পূর্ণাঙ্ক n = nodes.size();

    std::vector<পূর্ণাঙ্ক> res;
    res.reserve(n);

    std::vector<তার্কিক> used(n, অসত্য);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    প্রতি (পূর্ণাঙ্ক i0 = 0; i0 < n; i0++) {
        যদি (used[i0]) {
            অনুবর্ত;
        }

        ধ্রুব auto & node0 = nodes[i0];

        যদি (!node0.stackable()) {
            res.push_back(i0);
            used[i0] = সত্য;
            অনুবর্ত;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr পূর্ণাঙ্ক N_FORWARD = 16;

        std::vector<পূর্ণাঙ্ক> স্তূপ;
        স্তূপ.push_back(i0);

        mrs.reset();

        প্রতি (পূর্ণাঙ্ক i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            যদি (used[i1]) {
                অনুবর্ত;
            }

            ধ্রুব auto & node1 = nodes[i1];

            যদি (node1.stackable() && node1.same_প্রবেশ(node0) && ggml_hexagon_mem_ranges_check_গ্রন্থি(mrs, node1)) {
                স্তূপ.push_back(i1);
            } অন্যথা {
                ggml_hexagon_mem_ranges_add_গ্রন্থি(mrs, node1);
            }
        }

        প্রতি (পূর্ণাঙ্ক idx : স্তূপ) {
            res.push_back(idx);
            used[idx] = সত্য;
        }
    }

    প্রত্যাগম res;
}

স্থির শূন্য ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    ধ্রুব পূর্ণাঙ্ক n = gf->n_nodes;

    constexpr পূর্ণাঙ্ক MAX_FUSE = 16;

    গণনা ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    প্রতি (পূর্ণাঙ্ক i = 0; i < n; i++) {
        htp_opnode গ্রন্থি(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        যদি (গ্রন্থি.op() == GGML_OP_ADD ||
            গ্রন্থি.op() == GGML_OP_NORM ||
            গ্রন্থি.op() == GGML_OP_RMS_NORM) {
            ops[0] = গ্রন্থি.op();

            পূর্ণাঙ্ক f = i + 1;
            যাবত্ (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                যদি (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    ভঙ্গ;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            প্রতি (; f > 1; f--) {
                যদি (ggml_can_fuse(gf, i, ops, f)) {
                    ভঙ্গ;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            প্রতি (পূর্ণাঙ্ক k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                গ্রন্থি.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(গ্রন্থি));
    }

    ধ্রুব auto order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        পূর্ণাঙ্ক j = 0;
        প্রতি (ধ্রুব auto i : order) {
            ধ্রুব auto & গ্রন্থি = nodes[i];

            gf->nodes[j++] = গ্রন্থি.গ্রন্থি;

            প্রতি (auto * fused : গ্রন্থি.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

স্থির uint64_t ggml_hexagon_session_চাবি(ধ্রুব ggml_hexagon_session * sess) {
    প্রত্যাগম ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

স্থির তার্কিক ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, ধ্রুব ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    auto sbuf_dst = sess_dst->mmap_tensor(dst);

    যদি (!sess_src->clone_buffer(sbuf_dst)) { প্রত্যাগম অসত্য; }

    ধ্রুব uint64_t src_চাবি = ggml_hexagon_session_চাবি(sess_src);
    auto & fence_slot = sess_dst->cpy_fence_slots[src_চাবি];
    যদি (!fence_slot) {
        fence_slot = (চঞ্চল uint32_t *) sess_dst->alloc_fence(1);
    }

    যদি (!sess_src->clone_buffer(sess_dst->fence_buf)) { প্রত্যাগম অসত্য; }

    যদি (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    স্থির ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.buffer = &sess_dst->fence_buf->backend_buffer;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.data   = (শূন্য *) fence_slot;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = পরিমাণ(int32_t);
    fence_tensor.nb[1]  = পরিমাণ(int32_t);
    fence_tensor.nb[2]  = পরিমাণ(int32_t);
    fence_tensor.nb[3]  = পরিমাণ(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ সত্য);

    sess_dst->add_peer(sess_src);

    প্রত্যাগম সত্য;
}

স্থির তার্কিক ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, ধ্রুব ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    auto sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    যদি (!sess_dst->clone_buffer(sbuf_src)) { প্রত্যাগম অসত্য; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    প্রত্যাগম সত্য;
}

স্থির তার্কিক ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ধ্রুব ggml_tensor * src, ggml_tensor * dst) {
    যদি (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        প্রত্যাগম অসত্য;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    যদি (!dst->extra) {
        ggml_backend_buffer_init_tensor(dst->buffer, dst);
    }

    auto * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    ধ্রুব auto * src_extra = static_cast<ধ্রুব ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    যদি (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        প্রত্যাগম সত্য;
    }

    যদি (sess_src->phys_idx != sess_dst->phys_idx)
        প্রত্যাগম ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    প্রত্যাগম ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

স্থির ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = নূতন ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_slot = (চঞ্চল uint32_t *) sess->alloc_fence(1);

    স্থির ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.buffer = &sess->fence_buf->backend_buffer;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.data   = (শূন্য *) hex_event->fence_slot;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = পরিমাণ(int32_t);
    hex_event->fence_tensor.nb[1]  = পরিমাণ(int32_t);
    hex_event->fence_tensor.nb[2]  = পরিমাণ(int32_t);
    hex_event->fence_tensor.nb[3]  = পরিমাণ(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (শূন্য *)hex_event, (শূন্য *)hex_event->fence_slot);

    প্রত্যাগম নূতন ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

স্থির শূন্য ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    যদি (hex_event->seq == 0) {
        প্রত্যাগম;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (শূন্য *)hex_event, hex_event->seq, (শূন্য *)hex_event->fence_slot);

    auto * fence = reinterpret_cast<ধ্রুব চঞ্চল std::atomic<uint32_t> *>(hex_event->fence_slot);

    যদি ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    যাবত্ (সত্য) {
        যদি ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            যদি (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            ভঙ্গ;
        }
        std::this_thread::yield();
    }
}

স্থির শূন্য ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (শূন্য *)hex_event);
    hex_event->fence_sess->free_fence((শূন্য *) hex_event->fence_slot, 1);
    delete hex_event;
    delete event;
}

স্থির শূন্য ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

স্থির শূন্য ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    যদি (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ অসত্য);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (শূন্য *)hex_event, hex_event->seq, (শূন্য *)hex_event->fence_slot);
}

স্থির শূন্য ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    যদি (hex_event->seq == 0) {
        প্রত্যাগম;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (শূন্য *)hex_event, hex_event->seq, (শূন্য *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    যদি (sess->phys_idx == hex_event->sess->phys_idx) {
        যদি (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        প্রত্যাগম;
    }

    sess->clone_buffer(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ সত্য);
}

স্থির শূন্য ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, সংরচনা ggml_tensor * tensor, ধ্রুব শূন্য * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (পূর্ণাঙ্ক) tensor->buffer->usage : -1);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

স্থির শূন্য ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, ধ্রুব সংরচনা ggml_tensor * tensor, শূন্য * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (পূর্ণাঙ্ক) tensor->buffer->usage : -1);
    sess->flush_sync();
    যদি (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

স্থির শূন্য ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     সংরচনা ggml_tensor * tensor,
                                                     ধ্রুব শূন্য * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (পূর্ণাঙ্ক) tensor->buffer->usage : -1);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

স্থির শূন্য ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     ধ্রুব সংরচনা ggml_tensor * tensor,
                                                     শূন্য * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (পূর্ণাঙ্ক) tensor->buffer->usage : -1);
    sess->flush_sync();
    যদি (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

স্থির সংরচনা ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ রিক্ত,
    /* .graph_plan_free         = */ রিক্ত,
    /* .graph_plan_update       = */ রিক্ত,
    /* .graph_plan_compute      = */ রিক্ত,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_record,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

স্থির ggml_guid_t ggml_backend_hexagon_guid() {
    স্থির ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    প্রত্যাগম &guid;
}

তার্কিক ggml_backend_is_hexagon(ggml_backend_t backend) {
    প্রত্যাগম backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

স্থির ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, ধ্রুব বর্ণ * params) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    প্রত্যাগম নূতন ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

স্থির ধ্রুব বর্ণ * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    প্রত্যাগম dev_ctx->c_name();

    GGML_UNUSED(dev);
}

স্থির ধ্রুব বর্ণ * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    প্রত্যাগম "Hexagon";
    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * total) {
    *মুক্ত  = opt_mbuf_total;
    *total = opt_mbuf_total;

    GGML_UNUSED(dev);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    প্রত্যাগম GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

স্থির শূন্য ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, সংরচনা ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ সত্য,
        /* .host_buffer           = */ অসত্য,
        /* .buffer_from_host_ptr  = */ অসত্য,
        /* .events                = */ সত্য,
        /* .mmap_support          = */ অসত্য,
    };
}

স্থির ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    প্রত্যাগম &dev_ctx->buffer_type;
}

স্থির ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    যদি (!opt_hostbuf) {
        প্রত্যাগম রিক্ত;
    }
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    প্রত্যাগম &dev_ctx->host_buffer_type;
}

স্থির তার্কিক ggml_hexagon_supported_cpy(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    সংরচনা htp_copy_kernel_params kparams;
    প্রত্যাগম ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

স্থির তার্কিক ggml_hexagon_supported_cont(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    সংরচনা htp_copy_kernel_params kparams;
    প্রত্যাগম ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

স্থির তার্কিক ggml_hexagon_supported_repeat(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    GGML_UNUSED(sess);
    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    // Support f32 and f16
    যদি (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) প্রত্যাগম অসত্য;

    // src and dst must be the same type
    যদি (src0->type != dst->type) প্রত্যাগম অসত্য;

    // dst dims must be multiples of src dims
    যদি (dst->ne[0] % src0->ne[0] != 0) প্রত্যাগম অসত্য;
    যদি (dst->ne[1] % src0->ne[1] != 0) প্রত্যাগম অসত্য;
    যদি (dst->ne[2] % src0->ne[2] != 0) প্রত্যাগম অসত্য;
    যদি (dst->ne[3] % src0->ne[3] != 0) প্রত্যাগম অসত্য;

    // require contiguous tensors (no transposition)
    যদি (ggml_is_transposed(src0) || ggml_is_transposed(dst)) প্রত্যাগম অসত্য;

    প্রত্যাগম সত্য;
}

স্থির তার্কিক ggml_hexagon_supported_concat(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    সংরচনা htp_concat_kernel_params kparams;
    প্রত্যাগম ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

স্থির তার্কিক ggml_hexagon_supported_fill(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    ধ্রুব সংরচনা ggml_tensor * dst = op;

    যদি (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;
    GGML_UNUSED(sess);
}

স্থির তার্কিক ggml_hexagon_supported_roll(ধ্রুব সংরচনা ggml_hexagon_session * sess, ধ্রুব সংরচনা ggml_tensor * op) {
    GGML_UNUSED(sess);

    ধ্রুব সংরচনা ggml_tensor * src0 = op->src[0];
    ধ্রুব সংরচনা ggml_tensor * dst  = op;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    যদি (!ggml_are_same_shape(src0, dst)) {
        প্রত্যাগম অসত্য;
    }

    যদি (src0->nb[0] != ggml_type_size(src0->type) || dst->nb[0] != ggml_type_size(dst->type)) {
        প্রত্যাগম অসত্য;
    }

    যদি (!ggml_is_contiguous(dst)) {
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
স্থির তার্কিক ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, ধ্রুব সংরচনা ggml_tensor * op) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    // reject ops that match the filter
    যদি (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        প্রত্যাগম অসত্য;
    }

    তার্কিক supp = অসত্য;
    বিকল্প (op->op) {
        পক্ষ GGML_OP_NONE:
        পক্ষ GGML_OP_RESHAPE:
        পক্ষ GGML_OP_VIEW:
        পক্ষ GGML_OP_PERMUTE:
        পক্ষ GGML_OP_TRANSPOSE:
            supp = সত্য;
            ভঙ্গ;

        পক্ষ GGML_OP_MUL:
        পক্ষ GGML_OP_ADD:
        পক্ষ GGML_OP_SUB:
        পক্ষ GGML_OP_DIV:
            supp = ggml_hexagon_supported_দ্বিমিক(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_NORM:
        পক্ষ GGML_OP_L2_NORM:
        পক্ষ GGML_OP_RMS_NORM:
        পক্ষ GGML_OP_SCALE:
        পক্ষ GGML_OP_CLAMP:
        পক্ষ GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_SQR:
        পক্ষ GGML_OP_SQRT:
        পক্ষ GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_SUM:
            supp = ggml_hexagon_supported_sum(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_UNARY:
            বিকল্প (ggml_get_unary_op(op)) {
                পক্ষ GGML_UNARY_OP_NEG:
                পক্ষ GGML_UNARY_OP_EXP:
                পক্ষ GGML_UNARY_OP_SIGMOID:
                পক্ষ GGML_UNARY_OP_SOFTPLUS:
                পক্ষ GGML_UNARY_OP_TANH:
                পক্ষ GGML_UNARY_OP_ABS:
                পক্ষ GGML_UNARY_OP_SILU:
                পক্ষ GGML_UNARY_OP_GELU:
                পক্ষ GGML_UNARY_OP_GELU_QUICK:
                পক্ষ GGML_UNARY_OP_GELU_ERF:
                পক্ষ GGML_UNARY_OP_RELU:
                পক্ষ GGML_UNARY_OP_STEP:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    ভঙ্গ;
                স্বভাব:
                    supp = অসত্য;
                    ভঙ্গ;
            }
            ভঙ্গ;

        পক্ষ GGML_OP_GLU:
            বিকল্প (ggml_get_glu_op(op)) {
                পক্ষ GGML_GLU_OP_SWIGLU:
                পক্ষ GGML_GLU_OP_SWIGLU_OAI:
                পক্ষ GGML_GLU_OP_SWIGLU_CLAMP:
                পক্ষ GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    ভঙ্গ;
                পক্ষ GGML_GLU_OP_GEGLU_QUICK:
                পক্ষ GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    ভঙ্গ;
                স্বভাব:
                    supp = অসত্য;
                    ভঙ্গ;
            }
            ভঙ্গ;

        পক্ষ GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_DUP:
        পক্ষ GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_শীর্ষ_K:
            supp = ggml_hexagon_supported_শীর্ষ_k(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_IM2COL_3D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            ভঙ্গ;

        পক্ষ GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            ভঙ্গ;

        স্বভাব:
            ভঙ্গ;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    প্রত্যাগম supp;
}

স্থির তার্কিক ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    তার্কিক supp = (buft == &dev_ctx->host_buffer_type) || (buft == &dev_ctx->buffer_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    প্রত্যাগম supp;
}

স্থির ধ্রুব সংরচনা ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ রিক্ত,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ রিক্ত,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    প্রতি (size_t i = 0; i < opt_ndev; i++) {
        ধ্রুব auto & cfg = opt_device_configs[i];
        যদি (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } অন্যথা {
            std::string peers_str;
            প্রতি (ধ্রুব auto & p : cfg.mdev_group) {
                যদি (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = নূতন ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    প্রতি (size_t i = 0; i < opt_ndev; i++) {
        auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        delete dev_ctx;
    }
}

স্থির ধ্রুব বর্ণ * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    প্রত্যাগম "HTP";
    GGML_UNUSED(reg);
}

স্থির size_t ggml_backend_hexagon_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    প্রত্যাগম opt_ndev;
    GGML_UNUSED(reg);
}

স্থির ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    যদি (index >= opt_ndev || !hreg->devices[index].context) {
        প্রত্যাগম nullptr;
    }

    প্রত্যাগম &hreg->devices[index];
}

// ** communication context for tensor-split allreduce

স্থির শূন্য * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    যদি (n_backends < 2 || n_backends > 4) {
        প্রত্যাগম nullptr;
    }

    প্রতি (size_t i = 0; i < n_backends; ++i) {
        যদি (!ggml_backend_is_hexagon(backends[i])) {
            প্রত্যাগম nullptr;
        }
    }

    প্রতি (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        প্রতি (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            যদি (sess_i->phys_idx == sess_j->phys_idx) {
                প্রত্যাগম nullptr;
            }
        }
    }

    auto * ctx = নূতন ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    স্থির ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    প্রতি (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (চঞ্চল uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].buffer = &sess_i->fence_buf->backend_buffer;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].data   = (শূন্য *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = পরিমাণ(int32_t);
        ctx->fence_tensors[i].nb[1]  = পরিমাণ(int32_t);
        ctx->fence_tensors[i].nb[2]  = পরিমাণ(int32_t);
        ctx->fence_tensors[i].nb[3]  = পরিমাণ(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    প্রত্যাগম ctx;
}

স্থির শূন্য ggml_backend_hexagon_comm_free(শূন্য * comm_ctx_v) {
    যদি (!comm_ctx_v) প্রত্যাগম;
    auto * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    প্রতি (size_t i = 0; i < ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((শূন্য *) ctx->fence_slots[i], 1);
    }
    delete ctx;
}

স্থির তার্কিক ggml_backend_hexagon_comm_allreduce_tensor(শূন্য * comm_ctx_v, সংরচনা ggml_tensor ** tensors) {
    যদি (opt_ar_select == 0 || !comm_ctx_v) প্রত্যাগম অসত্য;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ধ্রুব size_t n_backends = comm_ctx->n_backends;

    যদি (n_backends < 2 || n_backends > 4) প্রত্যাগম অসত্য;

    প্রতি (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        প্রতি (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            যদি (sess_i->phys_idx == sess_j->phys_idx) {
                প্রত্যাগম অসত্য;
            }
        }
    }

    প্রতি (size_t i = 0; i < n_backends; i++) {
        যদি (!tensors[i] || !tensors[i]->buffer || !ggml_backend_buffer_is_hexagon(tensors[i]->buffer)) {
            প্রত্যাগম অসত্য;
        }
        যদি (tensors[i]->type != tensors[0]->type) {
            প্রত্যাগম অসত্য;
        }
        যদি (!ggml_is_contiguous(tensors[i])) {
            প্রত্যাগম অসত্য;
        }
        যদি (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            প্রত্যাগম অসত্য;
        }
    }

    যদি (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        প্রত্যাগম অসত্য;
    }

    প্রতি (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        সংরচনা htp_allreduce_kernel_params kparams;
        যদি (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, অসত্য, অসত্য, /*is_shard_ok=*/ অসত্য, &kparams)) {
            প্রত্যাগম অসত্য;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    প্রতি (size_t i = 1; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        যদি ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    যদি (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_entry = max_seq;
    যদি (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    প্রতি (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<ধ্রুব ggml_tensor *> data_tensors(n_backends);
    std::vector<ধ্রুব ggml_tensor *> sync_tensors(n_backends);
    প্রতি (size_t i = 0; i < n_backends; i++) {
        data_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    প্রতি (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], data_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        প্রতি (size_t j = 0; j < n_backends; j++) {
            যদি (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    প্রত্যাগম সত্য;
}

স্থির ggml_backend_buffer_type_t ggml_backend_hexagon_split_buffer_type(পূর্ণাঙ্ক main_device, ধ্রুব ভিন্ন * tensor_split) {
    GGML_UNUSED(tensor_split);
    auto reg = ggml_backend_hexagon_reg();
    auto dev = ggml_backend_reg_dev_get(reg, main_device);
    যদি (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    যদি (!dev) প্রত্যাগম nullptr;
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    প্রত্যাগম &dev_ctx->buffer_type;
}

স্থির শূন্য * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, ধ্রুব বর্ণ * name) {
    GGML_UNUSED(reg);
    যদি (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        প্রত্যাগম (শূন্য *) ggml_backend_hexagon_split_buffer_type;
    }
    যদি (strcmp(name, "ggml_backend_comm_init") == 0) {
        প্রত্যাগম (শূন্য *) ggml_backend_hexagon_comm_init;
    }
    যদি (strcmp(name, "ggml_backend_comm_free") == 0) {
        প্রত্যাগম (শূন্য *) ggml_backend_hexagon_comm_free;
    }
    যদি (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        প্রত্যাগম (শূন্য *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    প্রত্যাগম রিক্ত;
}

template<typename T> std::vector<T> str_to_vec(ধ্রুব বর্ণ* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    যাবত্ (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    প্রত্যাগম v;
}

template<typename T, পূর্ণাঙ্ক BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    প্রতি (auto i : v) { ss << i << ','; }
    auto str = ss.str(); str.pop_back(); // drop last comma
    প্রত্যাগম str;
}

স্থির শূন্য ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, তার্কিক discovery_supported, ধ্রুব std::unordered_map<পূর্ণাঙ্ক, fastrpc_domain> & cdsp_map) {
    যদি (discovery_supported) {
        auto it = cdsp_map.find(cfg.physical_idx);
        যদি (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } অন্যথা {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } অন্যথা {
        বিকল্প (cfg.physical_idx) {
            পক্ষ 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                ভঙ্গ;
            পক্ষ 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                ভঙ্গ;
            স্বভাব:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                ভঙ্গ;
        }
    }
    প্রতি (auto & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
স্থির শূন্য ggml_hexagon_discover_devices() {
    std::unordered_map<পূর্ণাঙ্ক, fastrpc_domain> cdsp_map;
    তার্কিক discovery_supported = অসত্য;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    পূর্ণাঙ্ক err = remote_system_request(&domain_info);
    যদি (err == AEE_SUCCESS && domain_info.sys.অঙ্ক_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.অঙ্ক_domains);
        domain_info.sys.domains     = domains.data();
        domain_info.sys.max_domains = (পূর্ণাঙ্ক) domains.size();

        err = remote_system_request(&domain_info);
        যদি (err == AEE_SUCCESS) {
            discovery_supported = সত্য;
            ধ্রুব পূর্ণাঙ্ক n_domains = std::min(domain_info.sys.অঙ্ক_domains, (পূর্ণাঙ্ক) domains.size());
            প্রতি (পূর্ণাঙ্ক i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (পূর্ণাঙ্ক) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                যদি (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (পূর্ণাঙ্ক) domains[i].type);
                    অনুবর্ত;
                }
                যদি (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    অনুবর্ত;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } অন্যথা {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (unsigned) err);
        }
    } অন্যথা যদি (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (unsigned) err);
    }

    // Populate domain IDs and names for all configured devices
    প্রতি (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

স্থির শূন্য ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_Q4_0 == (unsigned পূর্ণাঙ্ক) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_Q4_1 == (unsigned পূর্ণাঙ্ক) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_Q8_0 == (unsigned পূর্ণাঙ্ক) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_MXFP4 == (unsigned পূর্ণাঙ্ক) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_IQ4_NL == (unsigned পূর্ণাঙ্ক) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_Q4_K == (unsigned পূর্ণাঙ্ক) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_Q5_K == (unsigned পূর্ণাঙ্ক) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_Q6_K == (unsigned পূর্ণাঙ্ক) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_Q3_K == (unsigned পূর্ণাঙ্ক) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_Q2_K == (unsigned পূর্ণাঙ্ক) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_Q1_0 == (unsigned পূর্ণাঙ্ক) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_Q2_0 == (unsigned পূর্ণাঙ্ক) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned পূর্ণাঙ্ক) HTP_TYPE_BF16 == (unsigned পূর্ণাঙ্ক) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    ধ্রুব বর্ণ * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    ধ্রুব বর্ণ * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    ধ্রুব বর্ণ * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    ধ্রুব বর্ণ * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    ধ্রুব বর্ণ * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    ধ্রুব বর্ণ * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    ধ্রুব বর্ণ * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    ধ্রুব বর্ণ * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    ধ্রুব বর্ণ * str_etm      = getenv("GGML_HEXAGON_ETM");
    ধ্রুব বর্ণ * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    ধ্রুব বর্ণ * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    ধ্রুব বর্ণ * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    ধ্রুব বর্ণ * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    ধ্রুব বর্ণ * str_fa_head_split = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    ধ্রুব বর্ণ * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    ধ্রুব বর্ণ * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    ধ্রুব বর্ণ * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    ধ্রুব বর্ণ * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    ধ্রুব বর্ণ * str_arch     = getenv("GGML_HEXAGON_ARCH");
    ধ্রুব বর্ণ * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    ধ্রুব বর্ণ * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    ধ্রুব বর্ণ * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    ধ্রুব বর্ণ * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    ধ্রুব বর্ণ * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    ধ্রুব বর্ণ * str_power_mode = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    যদি (!str_arch) {
        পূর্ণাঙ্ক err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        যদি (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } অন্যথা {
            যদি (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } অন্যথা যদি (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } অন্যথা {
        যদি (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, রিক্ত, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? নূতন std::regex(str_opfilter, RE_ICASE) : রিক্ত;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, রিক্ত, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, রিক্ত, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, রিক্ত, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  রিক্ত, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    যদি (str_power_mode) {
        পূর্ণাঙ্ক m = htp_power_mode_from_str(str_power_mode);
        যদি (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_mode);
        } অন্যথা {
            opt_power_mode = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, রিক্ত, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_head_split = str_fa_head_split ? atoi(str_fa_head_split)       : opt_fa_head_split;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    যদি (str_mbuf) {
        ধ্রুব বর্ণ * p = str_mbuf;
        প্রতি (পূর্ণাঙ্ক idx = 0; idx < 3 && p && *p; idx++) {
            যাবত্ (*p == ' ') p++;
            ধ্রুব বর্ণ * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : strlen(p);
            যাবত্ (len > 0 && p[len - 1] == ' ') len--;
            যদি (len > 0) {
                std::string token(p, len);
                যদি (idx == 0) opt_mbuf_dyn    = parse_size(token.c_str());
                যদি (idx == 1) opt_mbuf_static = parse_size(token.c_str());
                যদি (idx == 2) opt_mbuf_total  = parse_size(token.c_str());
            }
            যদি (!comma) ভঙ্গ;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_size(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    ধ্রুব বর্ণ * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    যদি (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    যদি (str_devices && str_devices[0] != '\0') {
        তার্কিক is_single_সংখ্যা = সত্য;
        প্রতি (পূর্ণাঙ্ক i = 0; str_devices[i] != '\0'; i++) {
            যদি (!isdigit((unsigned বর্ণ)str_devices[i])) {
                is_single_সংখ্যা = অসত্য;
                ভঙ্গ;
            }
        }
        যদি (is_single_সংখ্যা) {
            পূর্ণাঙ্ক n = atoi(str_devices);
            যদি (n < 1) n = 1;
            যদি (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            প্রতি (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (পূর্ণাঙ্ক)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.clear();
            }
        } অন্যথা {
            std::string s_devices(str_devices);
            std::vector<std::string> বস্তুসমূহ;
            std::string curr_item;
            পূর্ণাঙ্ক bracket_depth = 0;
            প্রতি (বর্ণ ch : s_devices) {
                যদি (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } অন্যথা যদি (ch == ']') {
                    যদি (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } অন্যথা যদি (ch == ',' && bracket_depth == 0) {
                    size_t s = curr_item.find_first_not_of(" \t\r\n");
                    size_t e = curr_item.find_last_not_of(" \t\r\n");
                    যদি (s != std::string::npos) {
                        বস্তুসমূহ.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.clear();
                } অন্যথা {
                    curr_item += ch;
                }
            }
            size_t s = curr_item.find_first_not_of(" \t\r\n");
            size_t e = curr_item.find_last_not_of(" \t\r\n");
            যদি (s != std::string::npos) {
                বস্তুসমূহ.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            প্রতি (ধ্রুব auto & item : বস্তুসমূহ) {
                size_t b_open  = item.find('[');
                size_t b_close = item.rfind(']');

                যদি (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    পূর্ণাঙ্ক virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    যদি (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        যত্ন {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } গৃহাণ (...) {
                            virt = 0;
                        }
                    } অন্যথা {
                        size_t dev_colon = dev_name.find(':');
                        যদি (dev_colon != std::string::npos) {
                            যত্ন {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } গৃহাণ (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<পূর্ণাঙ্ক> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    যাবত্ (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        যদি (ps == std::string::npos) অনুবর্ত;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        যদি (dash_pos != std::string::npos) {
                            যত্ন {
                                পূর্ণাঙ্ক p_start = std::stoi(p_part.substr(0, dash_pos));
                                পূর্ণাঙ্ক p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                প্রতি (পূর্ণাঙ্ক p = p_start; p <= p_end; p++) {
                                    যদি (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } গৃহাণ (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } অন্যথা {
                            যত্ন {
                                পূর্ণাঙ্ক p = std::stoi(p_part);
                                যদি (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } গৃহাণ (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    যদি (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    যদি (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        auto & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.clear();

                        প্রতি (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_group.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } অন্যথা {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } অন্যথা যদি (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    size_t colon_pos = rest.find(':');
                    পূর্ণাঙ্ক phys = 0;
                    পূর্ণাঙ্ক virt = 0;
                    যত্ন {
                        যদি (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } অন্যথা {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } গৃহাণ (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        অনুবর্ত;
                    }

                    যদি (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_group.clear();
                        opt_ndev++;
                    } অন্যথা {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } অন্যথা {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", item.c_str());
                }
            }
        }
    } অন্যথা {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_group.clear();
    }

#যদি defined(__ANDROID__)
    যদি (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    যদি (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            auto v  = str_to_vec<uint32_t>(str_profile);
            বিকল্প (v.size()) {
                পক্ষ 1:  opt_profile = v[0]; প্রত্যাগম opt_pmu_evt; // mode with default pmu events
                পক্ষ 8:  opt_profile = 2;    প্রত্যাগম v;           // mode with custom  pmu events
                স্বভাব: opt_profile = 0;    প্রত্যাগম {};          // garbage input
            }}();
        যদি (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = নূতন ggml_hexagon_registry(reg);
}

স্থির ধ্রুব সংরচনা ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_গুনতি,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(শূন্য) {
    স্থির তার্কিক initialized = অসত্য;

    স্থির ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ রিক্ত };

    {
        স্থির std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        যদি (!initialized) {
            auto nErr = htpdrv_init();
            যদি (nErr != AEE_SUCCESS) {
                প্রত্যাগম রিক্ত;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = সত্য;
    }

    প্রত্যাগম &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
