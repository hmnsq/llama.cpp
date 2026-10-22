#define _USE_MATH_DEFINES

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <memory>
#include <chrono>
#include <mutex>
#include <thread>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <regex>
#include <queue>
#include <deque>
#include <algorithm>
#include <cmath>
#include <initializer_list>

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
#否则
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"

#include <AEEStdErr.h>
#include <dspqueue.h>
#include <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-common.h"
#include "ggml-hexagon.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "htp-opnode.h"
#include "htp-ops.h"
#include "htp/matmul-ops.h"
#include "htp/binary-ops.h"
#include "htp/flash-attn-ops.h"
#include "htp/unary-ops.h"
#include "htp/get-rows-ops.h"
#include "htp/set-rows-ops.h"
#include "htp/softmax-ops.h"
#include "htp/pool-ops.h"
#include "htp/rope-ops.h"
#include "htp/ssm-conv.h"
#include "htp/gated-delta-net-ops.h"
#include "htp/argsort-ops.h"
#include "htp/concat-ops.h"
#include "htp/cpy-ops.h"
#include "htp/im2col-ops.h"
#include "htp_iface.h"
#include "htp-drv.h"
#include "htp-repack.h"

using intvec  = std::vector<整数>;
using uintvec = std::vector<unsigned 整数>;
using u32vec  = std::vector<uint32_t>;

#define GGML_HEXAGON_MAX_SESSIONS          16

#define GGML_HEXAGON_FENCE_SLOT_SIZE       128

struct ggml_hexagon_device_config {
    整数         physical_idx = 0;
    整数         virtual_idx  = 0;
    整数         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

static ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

static 整数    opt_arch    = 0; // autodetect
static size_t opt_ndev    = 1;
static size_t opt_nhvx    = 0; // use all
static 整数    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
static size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
static 整数    opt_etm     = 0;
static 整数    opt_verbose = 0;
static 整数    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
static bool   opt_hostbuf = false;
static bool   opt_dma64   = false;
static 整数    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start

static const std::pair<const char *, 整数> htp_power_mode_names[] = {
    { "low_power_saver",            HTP_POWER_MODE_LOW_POWER_SAVER },
    { "power_saver",                HTP_POWER_MODE_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_MODE_HIGH_POWER_SAVER },
    { "low_balanced",               HTP_POWER_MODE_LOW_BALANCED },
    { "balanced",                   HTP_POWER_MODE_BALANCED },
    { "high_performance",           HTP_POWER_MODE_HIGH_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_MODE_SUSTAINED_HIGH_PERFORMANCE },
    { "burst",                      HTP_POWER_MODE_BURST },
    { "default",                    HTP_POWER_MODE_BURST }, // flip this to change the default mode
};

static 整数 htp_power_mode_from_str(const char * s) {
    为 (auto & e : htp_power_mode_names) {
#如果 defined(_MSC_VER)
        如果 (_stricmp(s, e.first) == 0) {
#否则
        如果 (strcasecmp(s, e.first) == 0) {
#endif
            返回 e.second;
        }
    }
    返回 -1;
}

static const char * htp_power_mode_to_str(整数 mode) {
    为 (auto & e : htp_power_mode_names) {
        如果 (e.second == mode) {
            返回 e.first;
        }
    }
    返回 "unknown";
}

static size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
static size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
static size_t opt_mbuf_total  = 0;                        // total buffer space limit (0 = unconstrained)

static 整数    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
static 整数    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
static 整数    opt_fa_head_split = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
static 整数    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
static 整数    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
static 整数    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
static u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

static 整数 opt_opbatch  = 1280; // max number of ops in a batch
static 整数 opt_opqueue  = 8;   // max number of pending batches
static 整数 opt_optrace  = 0;    // trace buffer size per thread (0 means default)
static 整数 opt_oppoll   = 0;    // polling for batch completions
static 整数 opt_opfusion  = 1;    // enable/disable op fusion
static 整数 opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

enum ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ADD = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ADD   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

static inline bool ggml_hexagon_is_fusion_enabled(整数 flag) {
    如果 (opt_opfusion <= 0) 返回 false;
    如果 (opt_opfusion == 1) 返回 true; // 1 enables all
    返回 (opt_opfusion & flag) != 0;
}

static std::regex* opt_opfilter = NULL; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    如果 (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

static const char * status_to_str(uint32_t status) {
    switch (status) {
        case HTP_STATUS_OK:
            返回 "OK";
        case HTP_STATUS_NO_SUPPORT:
            返回 "NO-SUPPORT";
        case HTP_STATUS_INVAL_PARAMS:
            返回 "INVAL-PARAMS";
        case HTP_STATUS_VTCM_TOO_SMALL:
            返回 "VTCM-TOO-SMALL";
        case HTP_STATUS_INTERNAL_ERR:
            返回 "INTERNAL-ERROR";
        default:
            返回 "UNKNOWN";
    }
}

// ** debug helpers

static void ggml_hexagon_dump_op_exec(const std::string &sess_name, const htp_opnode & node, const uint32_t req_flags) {
    如果 (!opt_verbose) 返回;

    htp_opformat fmt(node);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                node.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

static void ggml_hexagon_dump_op_supp(const std::string &sess_name, const struct ggml_tensor * op, bool supp) {
    如果 (!opt_verbose) 返回;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

static const char * htp_event_name(uint16_t id) {
    switch (id) {
        case HTP_TRACE_EVT_DMA:            返回 "DMA";
        case HTP_TRACE_EVT_HVX_COMP:       返回 "HVX_COMP";
        case HTP_TRACE_EVT_HVX_A_QUANT:    返回 "HVX_A_QUANT";
        case HTP_TRACE_EVT_HVX_A_PREP:     返回 "HVX_A_PREP";
        case HTP_TRACE_EVT_HVX_W_DEQUANT:  返回 "HVX_W_DEQUANT";
        case HTP_TRACE_EVT_HVX_W_PREP:     返回 "HVX_W_PREP";
        case HTP_TRACE_EVT_HVX_O_PROC:     返回 "HVX_O_PROC";
        case HTP_TRACE_EVT_HVX_FA_QK:      返回 "HVX_QK_FA";
        case HTP_TRACE_EVT_HVX_FA_SFM:     返回 "HVX_SFM_FA";
        case HTP_TRACE_EVT_HVX_FA_Q_PREP:  返回 "HVX_Q_PREP";
        case HTP_TRACE_EVT_HVX_FA_K_PREP:  返回 "HVX_K_PREP";
        case HTP_TRACE_EVT_HVX_FA_V_PREP:  返回 "HVX_V_PREP";
        case HTP_TRACE_EVT_HVX_GDN_PREP:   返回 "HVX_GDN_PREP";
        case HTP_TRACE_EVT_HVX_GDN_SOLVE:  返回 "HVX_GDN_SOLVE";
        case HTP_TRACE_EVT_HVX_GDN_V_PREP: 返回 "HVX_GDN_V_PREP";
        case HTP_TRACE_EVT_HVX_GDN_D_PREP: 返回 "HVX_GDN_D_PREP";
        case HTP_TRACE_EVT_HVX_GDN_OUT:    返回 "HVX_GDN_OUT";
        case HTP_TRACE_EVT_HVX_GDN_STATE:  返回 "HVX_GDN_STATE";
        case HTP_TRACE_EVT_HVX_GDN_REM:    返回 "HVX_GDN_REM";
        case HTP_TRACE_EVT_HMX_COMP:       返回 "HMX_COMP";
        case HTP_TRACE_EVT_L2FLUSH:        返回 "L2FLUSH";
        case HTP_TRACE_EVT_INIT:           返回 "INIT";
        case HTP_TRACE_EVT_BUFF:           返回 "BUFF";
        case HTP_TRACE_EVT_FENCE:          返回 "FENCE";
        default:                           返回 "UNKNOWN";
    }
}

static void ggml_hexagon_dump_op_prof(const std::string &sess_name, const htp_opnode & node, const htp_prof_desc & pd) {
    如果 (!opt_profile) 返回;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    const uint32_t * pmu = pd.pmu;

    char pmu_str[256] = "";
    如果 (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, sizeof(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(node);
    float mhz = op_usec > 0 ? (float) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            node.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

static void ggml_hexagon_dump_batch_prof(const std::string & sess_name, const htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    float batch_mhz = rsp.usecs > 0 ? (float) batch_cycles / rsp.usecs : 0.0f;

    char evt_str[256] = "----";
    如果 (opt_profile == 3) {
        snprintf(evt_str, sizeof(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (unsigned long long) batch_cycles, (unsigned long long) rsp.cycles_start, batch_mhz);
}

static void ggml_hexagon_dump_trace_events(const std::string & sess_name, const htp_opbatch_rsp & rsp,
                                           const htp_trace_desc * trace_events, uint32_t n_traces) {
    如果 (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        为 (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        为 (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            为 (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                const auto & e = trace_events[t * n_traces + idx];
                bool is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

enum ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_WEIGHT    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

struct ggml_hexagon_session;

static void ggml_hexagon_precompute_matmul_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
);

static void ggml_hexagon_precompute_fused_matmul_add_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
);

static void ggml_hexagon_precompute_unary_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_unary_kernel_params * kparams
);

static bool ggml_hexagon_precompute_binary_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_binary_kernel_params * kparams,
    const struct ggml_tensor * src2 = nullptr
);

static void ggml_hexagon_precompute_get_rows_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_get_rows_kernel_params * kparams
);

static void ggml_hexagon_precompute_set_rows_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_set_rows_kernel_params * kparams
);

static void ggml_hexagon_precompute_softmax_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_softmax_kernel_params * kparams
);

static void ggml_hexagon_precompute_rope_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_rope_kernel_params * kparams
);

static void ggml_hexagon_precompute_ssm_conv_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_ssm_conv_kernel_params * kparams
);

static void ggml_hexagon_precompute_gated_delta_net_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_gdn_kernel_params * kparams
);

static void ggml_hexagon_precompute_sort_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    bool is_top_k,
    struct htp_sort_kernel_params * kparams
);

static void ggml_hexagon_precompute_pool_2d_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * dst,
    struct htp_pool_2d_kernel_params * kparams,
    bool is_pool_1d
);

static bool ggml_hexagon_precompute_concat_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_concat_kernel_params * kparams
);

static bool ggml_hexagon_precompute_cpy_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_copy_kernel_params * kparams
);

static bool ggml_hexagon_precompute_im2col_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_im2col_kernel_params * kparams
);

static bool ggml_hexagon_precompute_im2col_3d_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_im2col_kernel_params * kparams
);

static void ggml_hexagon_precompute_fused_mmnx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * act,
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
);

static void ggml_hexagon_precompute_fused_mmidnx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * act,
    const struct ggml_tensor * dst,
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
);

static bool ggml_hexagon_precompute_allreduce_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    bool has_add,
    bool is_row_bcast,
    bool is_shard_ok,
    struct htp_allreduce_kernel_params * kparams
);

static bool ggml_hexagon_rows_stride(const int64_t * ne, const size_t * nb, size_t * stride);
static bool ggml_hexagon_matmul_can_collapse(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
static ggml_tensor ggml_hexagon_tensor_collapse_rows(const struct ggml_tensor * t);

static bool mm_is_hmx_eligible(const ggml_tensor * t);
static htp_op_code op_remap_to_htp(const ggml_tensor * t);
static bool is_supported_mul_mat_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams);
static bool is_supported_mul_mat_id_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams);
static bool is_mergeable_mul_mat(const ggml_tensor * t);
static bool is_mergeable_mul_mat_pair(const ggml_tensor * n1, const ggml_tensor * n2);
static bool is_mergeable_mul_mat_id(const ggml_tensor * t);
static bool is_mergeable_mul_mat_id_pair(const ggml_tensor * n1, const ggml_tensor * n2);

// ** backend sessions

struct ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_size { 0 };
    uint32_t             flags { 0 };
};

static inline bool ggml_hexagon_tensor_is_fuseable(const struct ggml_tensor * t) {
    如果 (!t->extra) 返回 false;
    auto extra = (const struct ggml_hexagon_tensor_extra *) t->extra;
    返回 (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

static inline bool ggml_hexagon_can_rt_repack(const struct ggml_tensor * t) {
    如果 (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
        返回 false;
    }
    如果 (t->buffer && ggml_backend_buft_is_host(t->buffer->buft)) {
        返回 false;
    }
    返回 opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

static inline bool ggml_hexagon_tensor_is_tiled(const struct ggml_tensor * t) {
    const auto * extra = (const ggml_hexagon_tensor_extra *) t->extra;
    如果 (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        返回 true;
    }
    返回 ggml_hexagon_can_rt_repack(t);
}

static inline bool ggml_hexagon_needs_repack(const struct ggml_tensor * t) {
    const auto * extra = (const ggml_hexagon_tensor_extra *) t->extra;
    如果 (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        返回 false;
    }
    返回 ggml_hexagon_can_rt_repack(t);
}

static inline bool ggml_hexagon_tensors_overlap(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    const uintptr_t a0 = (uintptr_t) a->data;
    const uintptr_t b0 = (uintptr_t) b->data;
    const uintptr_t a1 = a0 + ggml_nbytes(a);
    const uintptr_t b1 = b0 + ggml_nbytes(b);

    返回 a0 < b1 && b0 < a1;
}

static inline bool ggml_hexagon_can_row_partition(const struct ggml_tensor * t) {
    如果 (t->ne[1] > 1 && (t->nb[1] & 127) != 0) 返回 false;
    如果 (t->ne[2] > 1 && (t->nb[2] & 127) != 0) 返回 false;
    如果 (t->ne[3] > 1 && (t->nb[3] & 127) != 0) 返回 false;
    返回 true;
}

struct htp_opnode;

struct ggml_hexagon_opbatch;
struct ggml_hexagon_opqueue;
struct ggml_hexagon_shared_buffer;
struct ggml_hexagon_fence_buffer;
struct ggml_hexagon_session;
struct ggml_backend_hexagon_device_context;

struct ggml_hexagon_mdev_group {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

struct ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    volatile uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

struct ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    volatile uint32_t *    fence_slot   = nullptr;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

struct ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    整数              phys_idx;
    整数              virt_idx;
    bool             valid_session;
    bool             valid_handle;
    bool             valid_queue;
    bool             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<整数, std::unique_ptr<ggml_hexagon_shared_buffer>> cloned_buffers;
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

    ggml_hexagon_session(const ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_count = 0) noexcept(false);
    ~ggml_hexagon_session() noexcept(true);

    const char* c_name() const { 返回 name.c_str(); }

    void allocate(const ggml_hexagon_device_config & config) noexcept(false);
    void release() noexcept(true);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    void      free_fence(void * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, volatile uint32_t *> cpy_fence_slots;

    void enqueue_mdev_group();
    void enqueue_op(const htp_opnode & node);
    void enqueue_cpy(const ggml_tensor * src, ggml_tensor * dst, const ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    void enqueue_fence(const ggml_tensor * sync_tensor, uint32_t fence_seq = 0, bool wait = true);
    void enqueue_allreduce(const ggml_tensor * dst, const std::vector<const ggml_tensor *> & src_tensors,
                           const std::vector<const ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);

    void start_batch();
    void flush_sync(bool all = true);
    void flush_async();
    void flush_batch(size_t min_ops = 1);
    void flush_peers();
    void flush_pending(bool all = true);

    ggml_hexagon_shared_buffer * mmap_tensor(const ggml_tensor * t);
    bool clone_buffer(const ggml_hexagon_shared_buffer*);
    void release_buffer(const ggml_hexagon_shared_buffer*);
    void unclone_buffer(const ggml_hexagon_shared_buffer*);

    void add_peer(ggml_hexagon_session * peer) {
        如果 (this->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } 否则 {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

struct ggml_backend_hexagon_device_context {
    整数                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_buffer_type buffer_type       = {};
    ggml_backend_buffer_type host_buffer_type  = {};
    ggml_backend_buffer_type fence_buffer_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(整数 dev_id, const ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    const char * c_name() const { 返回 config.name.c_str(); }

    ggml_hexagon_session * session() {
        如果 (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        返回 sess.get();
    }
};

struct ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(const std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        this->dev_ctx = dev_ctx;
        this->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

struct ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    整数       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        如果 (!base) {
            throw std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        如果 (fd < 0) {
            rpcmem_free(base);
            throw std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        this->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        如果 (base) {
            rpcmem_free(base);
        }
    }
};

struct ggml_hexagon_shared_buffer {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    bool     mapped;
    bool     pinned;
    bool     extended;

    const char * c_name() const { 返回 sess->c_name(); }
    uint8_t *    base()   const { 返回 mem ? mem->base : nullptr; }
    size_t       size()   const { 返回 mem ? mem->size : 0;  }
    整数          fd()     const { 返回 mem ? mem->fd   : -1; }

    void mmap(bool extended = false) {
        如果 (!this->mem)   返回;
        如果 (this->mapped) 返回;

        GGML_ASSERT(!this->pinned || !extended);

        this->extended = extended;
        fastrpc_map_flags flags = this->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        整数 err = fastrpc_mmap(sess->domain_id, fd(), (void *) base(), 0, size(), flags);
        如果 (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (unsigned) err);
            throw std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (void *) base(), size(), fd(), pinned, extended);

        this->mapped = true;
    }

    void unmap() {
        如果 (!this->mapped) 返回;

        如果 (!this->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        如果 (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (void *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (void *) base(), size(), fd());

        this->mapped = false;
    }

    void alloc(size_t size) {
        如果 (this->mem) 返回;

        this->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (void *) base(), this->size(), fd(), (整数) pinned);
        如果 (this->pinned) {
            mmap();
        }
    }

    void free() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (void *) base(), size(), fd());
        this->mem  = nullptr;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, bool pinned = false) {
        this->sess     = sess;
        this->mapped   = false;
        this->pinned   = pinned;
        this->extended = false;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_offset = (size + 4095) & ~4095;
        size_t total_size   = guard_offset + 4096;
        如果 (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            total_size = (total_size + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(total_size);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, const ggml_hexagon_shared_buffer & other) {
        this->sess   = sess;
        this->mem    = other.mem;
        this->mapped = false;
        this->pinned = other.pinned;
        this->extended = other.extended;
    }

    ~ggml_hexagon_shared_buffer() {
        free();
        为 (auto * extra : tensor_extra) {
            delete extra;
        }
    }
};

struct ggml_hexagon_fence_buffer : public ggml_hexagon_shared_buffer {
    uint32_t              slot_count = 0;
    uint32_t              slot_head  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_buffer   backend_buffer{};

    ggml_hexagon_fence_buffer(ggml_hexagon_session * sess, ggml_backend_buffer_type_t buft, size_t size)
        : ggml_hexagon_shared_buffer(sess, size, false /* pinned */),
          slot_count(size / GGML_HEXAGON_FENCE_SLOT_SIZE),
          slot_head(0) {
        backend_buffer.buft    = buft;
        backend_buffer.context = static_cast<ggml_hexagon_shared_buffer *>(this);
        backend_buffer.size    = size;
        mmap(false);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        如果 (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        } 否则 如果 (slot_head + n_slots <= slot_count) {
            uint32_t slot = slot_head;
            slot_head += n_slots;
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        }
        如果 (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_SLOT_SIZE);
        }
        返回 ptr;
    }

    void free_slot(void * ptr, uint32_t n_slots = 1) {
        如果 (!ptr) 返回;
        uint32_t slot = ((uint8_t *) ptr - base()) / GGML_HEXAGON_FENCE_SLOT_SIZE;
        为 (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

inline uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    返回 ptr;
}

inline void ggml_hexagon_session::free_fence(void * ptr, uint32_t n_slots) {
    如果 (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

static void ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    sbuf->sess->unclone_buffer(sbuf);
    delete sbuf;
}

static void * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    返回 sbuf->base();
}

static enum ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (void *) sbuf->base(), tensor->data, ggml_nbytes(tensor));

    auto extra = new ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    返回 GGML_STATUS_SUCCESS;
}

static void ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   const void *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    如果 (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        如果 (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, data, offset, size, (整数) buffer->usage, extra->flags);

    如果 ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy((char *) tensor->data + offset, data, size);
        返回;
    }

    如果 (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        htp_repack_tensor_tiled(tensor, data, size);
        返回;
    }

    如果 (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    memcpy(extra->shadow_buf.data() + offset, data, size);
    extra->shadow_size += size;

    如果 (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

static void ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   const ggml_tensor *   tensor,
                                                   void *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, data, offset, size, (整数) buffer->usage, extra->flags);

    如果 ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy(data, (const char *) tensor->data + offset, size);
        返回;
    }

    htp_unpack_tensor_tiled(data, tensor, offset, size);
}

static bool ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   const struct ggml_tensor * src,
                                                   struct ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    返回 false;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

static void ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                          ggml_tensor *         tensor,
                                                          const void *          data,
                                                          size_t                offset,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    如果 (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        如果 (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (整数) buffer->usage, extra->flags);

    如果 ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        为 (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (const uint8_t *) data + i * stride_data, size);
        }
        返回;
    }

    如果 (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    为 (size_t i = 0; i < n_copies; i++) {
        memcpy(extra->shadow_buf.data() + offset + i * stride_tensor, (const uint8_t *) data + i * stride_data, size);
    }
    extra->shadow_size += n_copies * size;

    如果 (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

static void ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                      const ggml_tensor *   tensor,
                                                      void *                data,
                                                      size_t                offset,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (整数) buffer->usage);

    如果 ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        为 (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *)data + i * stride_data, (const uint8_t *)tensor->data + offset + i * stride_tensor, size);
        }
        返回;
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

    为 (size_t i = 0; i < n_copies; i++) {
        memcpy((uint8_t *) data + i * stride_data, temp_buf.data() + i * stride_tensor, size);
    }
}

static void ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (void *) sbuf->base(), sbuf->size());
    memset(sbuf->base(), value, sbuf->size());
}

static ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ NULL,
};

// ** backend buffer type

static void ggml_backend_hexagon_host_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                        ggml_tensor *         tensor,
                                                        const void *          data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                        const ggml_tensor *   tensor,
                                                        void *                data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_hexagon_host_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ NULL,
};

// ** backend buffer type

static const char * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    返回 static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    如果 (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        返回 nullptr;
    }
    try {
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size, false);
        返回 ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        返回 nullptr;
    }
}

static ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    如果 (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        返回 nullptr;
    }
    try {
        ggml_hexagon_shared_buffer * sbuf = new ggml_hexagon_shared_buffer(sess, size, false);
        返回 ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_host_buffer_interface, sbuf, size);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        返回 nullptr;
    }
}

static size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    返回 128;  // HVX alignment
    GGML_UNUSED(buft);
}

static size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * t) {
    如果 (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_size = htp_tiled_row_size(t->type, ne0) * ne1 * ne2 * ne3;
        返回 std::max(tiled_size, ggml_nbytes(t));
    }
    返回 ggml_nbytes(t);
}

static size_t parse_size(const char * str, size_t default_unit = 1024 * 1024) {
    如果 (!str || str[0] == '\0') {
        返回 0;
    }
    char * end = NULL;
    double val = strtod(str, &end);
    如果 (val < 0) {
        返回 0;
    }
    如果 (end && *end) {
        当 (*end == ' ') end++;
        如果 (*end == 'k' || *end == 'K') {
            返回 (size_t) (val * 1024);
        } 否则 如果 (*end == 'm' || *end == 'M') {
            返回 (size_t) (val * 1024 * 1024);
        } 否则 如果 (*end == 'g' || *end == 'G') {
            返回 (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    返回 (size_t) (val * default_unit);
}

static size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    返回 opt_mbuf_dyn;
    GGML_UNUSED(buft);
}

static bool ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    返回 false;
    GGML_UNUSED(buft);
}

static bool ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    返回 true;
    GGML_UNUSED(buft);
}

struct ggml_backend_hexagon_alloc_buffer_n_plan_item {
    size_t size;
    整数    first;
    整数    last;
};

using ggml_backend_hexagon_alloc_buffer_n_plan_t = std::vector<ggml_backend_hexagon_alloc_buffer_n_plan_item>;

static const char * ggml_hexagon_kv_layer_suffix(const struct ggml_tensor * t) {
    如果 (strncmp(t->name, "cache_", 6) != 0) {
        返回 NULL;
    }
    const char * p = strstr(t->name, "_l");
    如果 (!p || !isdigit((unsigned char)p[2])) {
        返回 NULL;
    }
    返回 p;
}

struct ggml_backend_hexagon_alloc_unit {
    size_t size;
    整数    first;
    整数    last;
};

static ggml_backend_hexagon_alloc_buffer_n_plan_t ggml_backend_hexagon_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, 整数 n_tensors) {
    ggml_backend_hexagon_alloc_buffer_n_plan_t plan;

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    整数 i = 0;
    当 (i < n_tensors) {
        struct ggml_tensor * t = tensors[i];
        size_t unit_size = 0;
        整数 unit_first = i;
        整数 unit_last = i + 1;

        如果 (t->data == NULL && t->view_src == NULL) {
            unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        const char * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        当 (unit_last < n_tensors) {
            struct ggml_tensor * next = tensors[unit_last];

            如果 (next->view_src != NULL) {
                unit_last++;
                continue;
            }

            如果 (layer_suffix != NULL) {
                const char * next_suffix = ggml_hexagon_kv_layer_suffix(next);
                如果 (next_suffix != NULL && strcmp(layer_suffix, next_suffix) == 0) {
                    如果 (next->data == NULL) {
                        unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, next), alignment);
                    }
                    unit_last++;
                    continue;
                }
            }

            break;
        }

        units.push_back({ unit_size, unit_first, unit_last });
        i = unit_last;
    }

    size_t cur_buf_size  = 0;
    整数    cur_buf_first = 0;

    为 (const auto & unit : units) {
        如果 (unit.size == 0) {
            continue;
        }

        如果 (cur_buf_size > 0 && (cur_buf_size + unit.size) > max_size) {
            plan.push_back({ cur_buf_size, cur_buf_first, unit.first });
            cur_buf_size  = 0;
            cur_buf_first = unit.first;
        }

        cur_buf_size += unit.size;
    }

    如果 (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, cur_buf_first, n_tensors });
    }

    返回 plan;
}

static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, 整数 n_tensors) {
    const ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    为 (const auto & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        如果 (buffer == NULL) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            为 (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            返回 NULL;
        }

        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        struct ggml_tensor * t_failed = NULL;
        为 (整数 j = item.first; j < item.last; j++) {
            struct ggml_tensor * t = tensors[j];
            如果 (t->data == NULL) {
                如果 (t->view_src == NULL) {
                    如果 (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } 否则 如果 (t->buffer == NULL) {
                    如果 (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } 否则 {
                如果 (t->view_src != NULL && t->buffer == NULL) {
                    如果 (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        如果 (t_failed != NULL) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            为 (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            返回 NULL;
        }

        buffers.push_back(buffer);
    }

    如果 (buffers.empty()) {
        返回 NULL;
    }

    如果 (buffers.size() == 1) {
        返回 buffers[0];
    }

    返回 ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

static size_t ggml_backend_hexagon_buffer_type_get_alloc_size_n(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, 整数 n_tensors) {
    const ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    为 (const auto & item : plan) {
        total += item.size;
    }
    返回 total;
}

static ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_buffer_type_is_host,
};

static ggml_backend_buffer_type_i ggml_backend_hexagon_host_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_host_buffer_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(整数 dev_id, const ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    buffer_type.device  = dev;
    buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    buffer_type.context = new ggml_backend_hexagon_buffer_type_context(config.name, this);

    host_buffer_type.device  = dev;
    host_buffer_type.iface   = ggml_backend_hexagon_host_buffer_type_interface;
    host_buffer_type.context = new ggml_backend_hexagon_buffer_type_context(config.name + "-HOST", this);

    fence_buffer_type.device  = dev;
    fence_buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    fence_buffer_type.context = new ggml_backend_hexagon_buffer_type_context(config.name + "-FENCE", this);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(host_buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(fence_buffer_type.context);
}

static bool ggml_backend_buffer_is_hexagon(const struct ggml_backend_buffer * b) {
    返回 b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

struct ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<整数, 整数>                b_map; // buffer fd   to index
    std::unordered_map<const ggml_tensor*, 整数> t_map; // tensor ptr  to index
    std::unordered_multimap<void*, 整数>         d_map; // tensor data to index

    unsigned 整数 n_bufs;     // num buffers in the batch
    unsigned 整数 n_tens;     // num tensors ...
    unsigned 整数 n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    unsigned 整数 n_bufs_max;
    unsigned 整数 n_tens_max;
    unsigned 整数 n_ops_max;
    size_t       b_vmem_max;

    void reset() {
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
        this->sess = sess;

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

    bool empty() const { 返回 n_ops == 0; }

    // add buffer and return its index
    整数 add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        auto it = b_map.find(sbuf->fd());
        如果 (it != b_map.end()) { 返回 it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        整数 bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        如果 (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (void*) sbuf->base(), (size_t) b.size, b_vmem);

        返回 bi;
    }

    bool same_shape(const htp_tensor * h, const ggml_tensor * t) const {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;

        const bool is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        const int64_t ne0_padded = hex_round_up(t->ne[0], 32);
        const int64_t ne1_padded = hex_round_up(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_row_size(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? sizeof(ggml_fp16_t) : t->nb[0];
        enum ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        返回 (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    整数 add_tensor(const ggml_tensor * t) {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;
        auto sbuf  = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        为 (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            如果 (same_shape(h, t)) { 返回 it->second; }
        }

        // Lookup by tensor ptr
        auto it = t_map.find(t);
        如果 (it != t_map.end()) { 返回 it->second; }

        // Add new tensor to the batch
        整数 ti = n_tens++;
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

        const bool is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        如果 (is_tiled) {
            如果 (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = sizeof(ggml_fp16_t);
            } 否则 {
                h.nb[0] = t->nb[0];
            }
            const int64_t ne0_padded = hex_round_up(t->ne[0], 32);
            const int64_t ne1_padded = hex_round_up(t->ne[1], 32);
            h.nb[1] = htp_tiled_row_size(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_size  = h.size;
        } 否则 {
            h.size  = t_size;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        如果 ((extra->flags & GGML_HEXAGON_TENSOR_WEIGHT) != 0) {
            h.flags |= HTP_TENSOR_WEIGHT;
        }
        如果 ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        如果 ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (void*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        返回 ti;
    }

    bool fit_op(const htp_opnode & node) const {
        如果 (n_ops >= n_ops_max) 返回 false;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        整数 seen_bufs[HTP_OP_MAX_BUFS];
        整数 n_seen_bufs = 0;

        auto fit_tensor = [&](const ggml_tensor *t) {
            如果 (!t) 返回;
            如果 (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                整数 fd = sbuf->fd();
                如果 (!b_map.count(fd)) {
                    为 (整数 i = 0; i < n_seen_bufs; i++) {
                        如果 (seen_bufs[i] == fd) 返回;
                    }
                    如果 (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    如果 (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        为 (const auto * src : node.get_inputs()) {
            fit_tensor(src);
        }
        为 (const auto * output : node.get_outputs()) {
            fit_tensor(output);
        }

        如果 ((extra_bufs + n_bufs) > n_bufs_max) 返回 false;
        如果 ((extra_tens + n_tens) > n_tens_max) 返回 false;
        如果 ((extra_vmem + b_vmem) > b_vmem_max) 返回 false;

        返回 true;
    }

    // assumes that fit_op() was called first and returned true
    void add_op(const htp_opnode & node) {
        // Add new op

        unsigned 整数 n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = node;

        htp_op_desc &o = h_ops[n];
        memcpy(o.params,        node.node->op_params, sizeof(node.node->op_params));
        memcpy(o.kernel_params, node.kernel_params,   sizeof(o.kernel_params));
        o.opcode = node.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        auto inputs = node.get_inputs();
        为 (unsigned 整数 i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        auto outputs = node.get_outputs();
        为 (unsigned 整数 i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    void sort_buffers() {
        如果 (n_bufs <= 1) 返回;

        std::vector<整数> order(n_bufs);
        为 (unsigned 整数 i = 0; i < n_bufs; i++) { order[i] = (整数) i; }

        std::stable_sort(order.begin(), order.end(), [&](整数 a, 整数 b) {
            返回 h_bufs[a].size > h_bufs[b].size;
        });

        bool already_sorted = true;
        为 (unsigned 整数 i = 0; i < n_bufs; i++) {
            如果 (order[i] != (整数) i) {
                already_sorted = false;
                break;
            }
        }
        如果 (already_sorted) 返回;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        为 (unsigned 整数 new_bi = 0; new_bi < n_bufs; new_bi++) {
            整数 old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        为 (unsigned 整数 i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        为 (unsigned 整数 i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    void update_mdev_group(uint32_t mdev_idx) {
        如果 (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    bool try_fuse_common(std::initializer_list<const ggml_tensor *> tensors) const {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        整数 seen_bufs[HTP_OP_MAX_BUFS];
        整数 n_seen_bufs = 0;

        为 (const auto * t : tensors) {
            如果 (!t || t_map.count(t)) {
                continue;
            }
            extra_tens++;
            auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            整数 fd = sbuf->fd();
            如果 (!b_map.count(fd)) {
                bool found = false;
                为 (整数 i = 0; i < n_seen_bufs; i++) {
                    如果 (seen_bufs[i] == fd) {
                        found = true;
                        break;
                    }
                }
                如果 (!found) {
                    如果 (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    如果 (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        如果 ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            返回 false;
        }

        返回 true;
    }

    bool try_fuse_common(const ggml_tensor * t1, const ggml_tensor * t2) const {
        返回 try_fuse_common({t1, t2});
    }

    bool try_fuse_allreduce_add(const htp_opnode & node) {
        如果 (n_ops == 0 || opt_ar_select != 2) 返回 false;
        如果 (node.opcode != HTP_OP_ADD) 返回 false;

        htp_opnode & last_node = ops[n_ops - 1];
        如果 (last_node.opcode != HTP_OP_ALLREDUCE) 返回 false;

        auto * ar_kparams = (struct htp_allreduce_kernel_params *) last_node.kernel_params;
        const uint32_t rank    = (uint32_t) ar_kparams->rank;
        const uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        const ggml_tensor * ar_local = last_node.inputs[rank];
        const ggml_tensor * add_src0 = node.src0();
        const ggml_tensor * add_src1 = node.src1();
        const ggml_tensor * add_dst  = node.dst();

        如果 (!ggml_hexagon_tensor_is_fuseable(ar_local)) 返回 false;

        const ggml_tensor * res_tensor;
        如果 (add_src0 == ar_local || add_src0->data == ar_local->data) {
            res_tensor = add_src1;
        } 否则 如果 (add_src1 == ar_local || add_src1->data == ar_local->data) {
            res_tensor = add_src0;
        } 否则 {
            返回 false;
        }

        如果 (ar_local->type != res_tensor->type) 返回 false;

        const bool is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        const bool is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        如果 (!is_same_shape && !is_row_bcast) 返回 false;

        如果 (is_same_shape) {
            如果 (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                返回 false;
            }
            如果 (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                返回 false;
            }
        }
        如果 (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) {
            返回 false;
        }

        // scatter is only valid for in-place add within max outputs
        const bool is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

        struct htp_allreduce_kernel_params new_kparams;
        如果 (!ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, true, is_row_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            返回 false;
        }

        const bool scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

        为 (uint32_t r = 0; r < n_ranks; r++) {
            const ggml_tensor * ar_src = last_node.inputs[r];
            如果 (!ggml_hexagon_tensors_overlap(add_dst, ar_src)) {
                continue;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            如果 (scatter_ok && r == rank) {
                continue;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            返回 false;
        }

        如果 (!try_fuse_common(res_tensor, add_dst)) {
            返回 false;
        }

        last_node.opcode = HTP_OP_ALLREDUCE_ADD;
        last_node.name   = "ALLREDUCE+ADD";
        last_node.inputs.push_back(res_tensor);
        last_node.outputs.clear();
        last_node.outputs.push_back(add_dst);
        last_node.fused.push_back(node.node);
        memcpy(last_node.kernel_params, &new_kparams, sizeof(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ADD;
        memcpy(o.kernel_params, &new_kparams, sizeof(new_kparams));

        o.src[2 * n_ranks] = add_tensor(res_tensor);
        如果 (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            为 (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            为 (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } 否则 {
            o.dst[0] = add_tensor(add_dst);
            为 (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (整数) new_kparams.mode, (整数) new_kparams.n_dsts);
        返回 true;
    }

    bool try_fuse_rms_norm_mul(const htp_opnode & node) {
        如果 (n_ops == 0) 返回 false;
        如果 (node.opcode != HTP_OP_MUL) 返回 false;

        htp_opnode & last_node = ops[n_ops - 1];
        如果 (last_node.opcode != HTP_OP_RMS_NORM) 返回 false;

        const ggml_tensor * mul_src0 = node.src0();
        const ggml_tensor * mul_src1 = node.src1();
        const ggml_tensor * rms_out  = last_node.dst();

        如果 (!ggml_hexagon_tensor_is_fuseable(rms_out)) 返回 false;

        const ggml_tensor * weight;
        如果 (mul_src0 == rms_out || mul_src0->data == rms_out->data) {
            weight = mul_src1;
        } 否则 如果 (mul_src1 == rms_out || mul_src1->data == rms_out->data) {
            weight = mul_src0;
        } 否则 {
            返回 false;
        }

        const ggml_tensor * src0 = last_node.src0();

        如果 (src0->ne[0] != weight->ne[0] || src0->ne[0] != node.dst()->ne[0]) {
            返回 false;
        }

        const bool is_row_bcast = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
        const bool is_same_shape = (src0->ne[0] == weight->ne[0] && src0->ne[1] == weight->ne[1] &&
                                    src0->ne[2] == weight->ne[2] && src0->ne[3] == weight->ne[3]);
        如果 (!is_row_bcast && !is_same_shape) 返回 false;

        如果 (!ggml_are_same_shape(src0, node.dst())) {
            返回 false;
        }
        如果 (ggml_is_contiguous(src0) != ggml_is_contiguous(node.dst())) {
            返回 false;
        }

        struct htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, weight, node.dst(), &new_kparams
        );

        如果 ((size_t) new_kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_size, sess->vtcm_size);
            返回 false;
        }

        如果 (!try_fuse_common(weight, node.dst())) {
            返回 false;
        }

        last_node.opcode = HTP_OP_RMS_NORM_MUL;
        last_node.name   = "RMS_NORM+MUL";
        last_node.inputs.clear();
        last_node.inputs.push_back(src0);
        last_node.inputs.push_back(weight);
        last_node.outputs.clear();
        last_node.outputs.push_back(node.dst());
        last_node.fused.push_back(node.node);
        memcpy(last_node.kernel_params, &new_kparams, sizeof(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        memcpy(o.kernel_params, &new_kparams, sizeof(new_kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(weight);
        为 (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(node.dst());
        为 (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        返回 true;
    }

    bool try_fuse_mul_mat_add(const htp_opnode & node) {
        如果 (n_ops == 0) 返回 false;
        如果 (node.opcode != HTP_OP_ADD) 返回 false;

        htp_opnode & last_node = ops[n_ops - 1];
        如果 (last_node.opcode != HTP_OP_MUL_MAT) 返回 false;

        const ggml_tensor * add_src0 = node.src0();
        const ggml_tensor * add_src1 = node.src1();
        const ggml_tensor * mm_out   = last_node.dst();

        如果 (!ggml_hexagon_tensor_is_fuseable(mm_out)) 返回 false;

        const ggml_tensor * src2;
        如果 (add_src0 == mm_out || add_src0->data == mm_out->data) {
            src2 = add_src1;
        } 否则 如果 (add_src1 == mm_out || add_src1->data == mm_out->data) {
            src2 = add_src0;
        } 否则 {
            返回 false;
        }

        const ggml_tensor * src0 = last_node.src0();
        const ggml_tensor * src1 = last_node.src1();

        如果 (src2->type != GGML_TYPE_F32) 返回 false;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        如果 (!ggml_are_same_shape(mm_out, node.dst())) 返回 false;

        const struct htp_mm_kernel_params * orig_kparams = (const struct htp_mm_kernel_params *) last_node.kernel_params;
        struct htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_add_params(sess, src0, src1, src2, node.dst(), &kparams);
        如果 (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            返回 false;
        }

        如果 (orig_kparams->collapse != kparams.collapse) {
            返回 false;
        }

        const 整数 src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        const bool can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        如果 (!can_fuse) 返回 false;

        如果 ((size_t) kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
            返回 false;
        }

        如果 (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            如果 (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_threads < orig_kparams->n_act_threads) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_threads, kparams.n_act_threads);
                返回 false;
            }
        }

        如果 (!try_fuse_common(src2, node.dst())) {
            返回 false;
        }

        last_node.opcode = HTP_OP_MUL_MAT_ADD;
        last_node.name   = "MUL_MAT+ADD";
        last_node.inputs.clear();
        last_node.inputs.push_back(src0);
        last_node.inputs.push_back(src1);
        last_node.inputs.push_back(src2);
        last_node.outputs.clear();
        last_node.outputs.push_back(node.dst());
        last_node.fused.push_back(node.node);
        memcpy(last_node.kernel_params, &kparams, sizeof(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ADD;
        memcpy(o.kernel_params, &kparams, sizeof(kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(src1);
        o.src[2] = add_tensor(src2);
        为 (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(node.dst());
        为 (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        返回 true;
    }

    bool try_fuse_mul_mat_nx(const htp_opnode & node) {
        如果 (n_ops == 0 || node.opcode != HTP_OP_MUL_MAT) 返回 false;
        如果 (!is_mergeable_mul_mat(node.node)) 返回 false;

        const ggml_tensor * w_in = node.src0();
        const ggml_tensor * x_in = node.src1();
        const ggml_tensor * d_in = node.dst();

        htp_opnode & last_node = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        如果 (last_node.opcode == HTP_OP_MUL_MAT_NX) {
            const uint32_t curr_n = (uint32_t) last_node.outputs.size();
            如果 (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                返回 false;
            }

            const ggml_tensor * w0 = last_node.inputs[0];
            const ggml_tensor * x  = last_node.inputs[curr_n];

            如果 (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                返回 false;
            }
            如果 (!last_node.fused.empty() && (mm_is_hmx_eligible(last_node.fused[0]) != mm_is_hmx_eligible(node.node))) {
                返回 false;
            }

            const struct htp_mm_kernel_params * orig_kparams = (const struct htp_mm_kernel_params *) last_node.kernel_params;
            const bool collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_in, x, d_in);
            如果 (orig_kparams->collapse && !collapse) {
                返回 false;
            }

            struct htp_mm_kernel_params kparams;
            如果 (collapse) {
                const ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } 否则 {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            如果 (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                返回 false;
            }
            如果 ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                返回 false;
            }

            如果 (!try_fuse_common(w_in, d_in)) {
                返回 false;
            }

            last_node.inputs[curr_n] = w_in;
            last_node.inputs.push_back(x);
            last_node.outputs.push_back(d_in);
            last_node.fused.push_back(node.node);
            memcpy(last_node.kernel_params, &kparams, sizeof(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, sizeof(kparams));

            为 (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = add_tensor(last_node.inputs[s]);
            }
            为 (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            为 (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_node.outputs[d]);
            }
            为 (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            返回 true;
        }

        // Case 2: last_node is single MUL_MAT
        如果 (last_node.opcode == HTP_OP_MUL_MAT) {
            如果 (!is_mergeable_mul_mat_pair(last_node.node, node.node)) {
                返回 false;
            }

            const ggml_tensor * w0 = last_node.src0();
            const ggml_tensor * x  = last_node.src1();
            const ggml_tensor * w1 = node.src0();
            const ggml_tensor * dst_0 = last_node.dst();
            const ggml_tensor * dst_1 = node.dst();

            const struct htp_mm_kernel_params * orig_kparams = (const struct htp_mm_kernel_params *) last_node.kernel_params;
            const bool collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            如果 (orig_kparams->collapse && !collapse) {
                返回 false;
            }

            struct htp_mm_kernel_params kparams;
            如果 (collapse) {
                const ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } 否则 {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            如果 (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                返回 false;
            }
            如果 ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                返回 false;
            }

            如果 (!try_fuse_common(w1, node.dst())) {
                返回 false;
            }

            last_node.opcode = HTP_OP_MUL_MAT_NX;
            last_node.name   = "MUL_MAT_NX";
            last_node.inputs.clear();
            last_node.inputs.push_back(w0);
            last_node.inputs.push_back(w1);
            last_node.inputs.push_back(x);
            last_node.outputs.clear();
            last_node.outputs.push_back(dst_0);
            last_node.outputs.push_back(dst_1);
            last_node.fused.push_back(node.node);
            memcpy(last_node.kernel_params, &kparams, sizeof(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            memcpy(o.kernel_params, &kparams, sizeof(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            为 (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            为 (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            返回 true;
        }

        返回 false;
    }

    bool try_fuse_mul_mat_id_nx(const htp_opnode & node) {
        如果 (n_ops == 0 || node.opcode != HTP_OP_MUL_MAT_ID) 返回 false;
        如果 (!is_mergeable_mul_mat_id(node.node)) 返回 false;

        const ggml_tensor * w_in   = node.src0();
        const ggml_tensor * x_in   = node.src1();
        const ggml_tensor * ids_in = node.node->src[2];
        const ggml_tensor * d_in   = node.dst();

        htp_opnode & last_node = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        如果 (last_node.opcode == HTP_OP_MUL_MAT_ID_NX) {
            const uint32_t curr_n = (uint32_t) last_node.outputs.size();
            如果 (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                返回 false;
            }

            const ggml_tensor * w0  = last_node.inputs[0];
            const ggml_tensor * x   = last_node.inputs[curr_n];
            const ggml_tensor * ids = last_node.inputs[curr_n + 1];

            如果 (x_in != x || ids_in != ids || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                返回 false;
            }
            如果 (!last_node.fused.empty() && (mm_is_hmx_eligible(last_node.fused[0]) != mm_is_hmx_eligible(node.node))) {
                返回 false;
            }

            struct htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
            如果 (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                返回 false;
            }
            如果 ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                返回 false;
            }

            如果 (!try_fuse_common(w_in, d_in)) {
                返回 false;
            }

            last_node.inputs[curr_n] = w_in;
            last_node.inputs[curr_n + 1] = x;
            last_node.inputs.push_back(ids);
            last_node.outputs.push_back(d_in);
            last_node.fused.push_back(node.node);
            memcpy(last_node.kernel_params, &kparams, sizeof(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, sizeof(kparams));

            为 (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = add_tensor(last_node.inputs[s]);
            }
            为 (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            为 (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_node.outputs[d]);
            }
            为 (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            返回 true;
        }

        // Case 2: last_node is single MUL_MAT_ID
        如果 (last_node.opcode == HTP_OP_MUL_MAT_ID) {
            如果 (!is_mergeable_mul_mat_id_pair(last_node.node, node.node)) {
                返回 false;
            }

            const ggml_tensor * w0  = last_node.src0();
            const ggml_tensor * x   = last_node.src1();
            const ggml_tensor * ids = last_node.node->src[2];
            const ggml_tensor * w1  = node.src0();

            struct htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, node.dst(), 2, &kparams);
            如果 (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                返回 false;
            }
            如果 ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                返回 false;
            }

            如果 (!try_fuse_common(w1, node.dst())) {
                返回 false;
            }

            const ggml_tensor * dst_0 = last_node.dst();
            const ggml_tensor * dst_1 = node.dst();

            last_node.opcode = HTP_OP_MUL_MAT_ID_NX;
            last_node.name   = "MUL_MAT_ID_NX";
            last_node.inputs.clear();
            last_node.inputs.push_back(w0);
            last_node.inputs.push_back(w1);
            last_node.inputs.push_back(x);
            last_node.inputs.push_back(ids);
            last_node.outputs.clear();
            last_node.outputs.push_back(dst_0);
            last_node.outputs.push_back(dst_1);
            last_node.fused.push_back(node.node);
            memcpy(last_node.kernel_params, &kparams, sizeof(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            memcpy(o.kernel_params, &kparams, sizeof(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            o.src[3] = add_tensor(ids);
            为 (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            为 (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            返回 true;
        }

        返回 false;
    }

    bool try_fuse_gdn_cpy(const htp_opnode & node) {
        如果 (n_ops == 0 || node.opcode != HTP_OP_CPY) 返回 false;

        htp_opnode & last_node = ops[n_ops - 1];
        如果 (last_node.opcode != HTP_OP_GATED_DELTA_NET) 返回 false;
        如果 (last_node.outputs.size() != 1) 返回 false;

        const ggml_tensor * gdn_out  = last_node.dst();
        const ggml_tensor * cpy_node = node.node;
        const ggml_tensor * cpy_src  = node.src0();
        const ggml_tensor * cpy_dst  = node.dst();

        如果 (!cpy_src || !cpy_dst || !cpy_dst->data) 返回 false;
        如果 (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) 返回 false;
        如果 ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_node->flags & GGML_TENSOR_FLAG_OUTPUT)) 返回 false;

        const ggml_tensor * v = last_node.node->src[2];
        如果 (!v) 返回 false;

        const int64_t S_v      = v->ne[0];
        const int64_t H        = v->ne[1];
        const int64_t n_tokens = v->ne[2];
        const int64_t n_seqs   = v->ne[3];
        const int64_t K        = ggml_get_op_params_i32(last_node.node, 0);
        const size_t  tail_off = (size_t) S_v * H * n_tokens * n_seqs * sizeof(float);

        const int64_t D         = S_v * S_v * H;
        const int64_t n_written = std::min<int64_t>(n_tokens, K);

        如果 (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_out && cpy_src->view_src->data != gdn_out->data) ||
            cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
            返回 false;
        }

        如果 (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != sizeof(float)) {
            返回 false;
        }
        如果 (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * sizeof(float)) {
            返回 false;
        }
        如果 (n_written > 1) {
            如果 (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * sizeof(float)) {
                返回 false;
            }
        }

        如果 (!try_fuse_common({cpy_dst})) {
            返回 false;
        }

        last_node.name += "+CPY";
        last_node.outputs.push_back(cpy_dst);
        last_node.fused.push_back(node.node);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = add_tensor(cpy_dst);
        为 (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        返回 true;
    }

    bool try_fuse(const htp_opnode & node) {
        如果 (!opt_opfusion) 返回 false;
        如果 (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ADD) && try_fuse_allreduce_add(node)) 返回 true;
        如果 (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(node))  返回 true;
        如果 (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ADD)   && try_fuse_mul_mat_add(node))   返回 true;
        如果 (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(node))    返回 true;
        如果 (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(node)) 返回 true;
        如果 (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(node))       返回 true;
        返回 false;
    }
};

struct ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

struct ggml_hexagon_opqueue {
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
        如果 (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * sizeof(htp_trace_desc);
        }

        shm_blk_size = sizeof(htp_buf_desc)  * n_bufs    +
                       sizeof(htp_tensor)    * n_tensors +
                       sizeof(htp_op_desc)   * n_ops     +
                       sizeof(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = new ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, true /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        如果 (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size(), shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    size_t shm_size() const { 返回 shm_buf ? shm_buf->size() : 0; }

    // push new batch
    bool push(htp_opbatch_req& req, dspqueue_buffer& dbuf, const ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(sizeof(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(sizeof(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(sizeof(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(sizeof(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(sizeof(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(sizeof(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        如果 (seq - shm_buf->sess->batch_rsp_seq > depth) { 返回 false; }

        const uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[slot]   = op_batch->ops;
        start_usec[slot] = ggml_time_us();

        const size_t b_size = sizeof(htp_buf_desc)  * req.n_bufs;
        const size_t t_size = sizeof(htp_tensor)    * req.n_tensors;
        const size_t o_size = sizeof(htp_op_desc)   * req.n_ops;
        const size_t p_size = sizeof(htp_prof_desc) * req.n_ops;

        size_t tr_size = 0;
        如果 (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * sizeof(htp_trace_desc);
        } 否则 {
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

        memcpy(b_ptr, (void *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (void *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (void *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        如果 (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            为 (unsigned 整数 i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (void *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            为 (unsigned 整数 i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (unsigned long long) t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        返回 true;
    }

    void pop(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        const uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        const size_t b_size = sizeof(htp_buf_desc)  * rsp.n_bufs;
        const size_t t_size = sizeof(htp_tensor)    * rsp.n_tensors;
        const size_t o_size = sizeof(htp_op_desc)   * rsp.n_ops;
        const size_t p_size = sizeof(htp_prof_desc) * rsp.n_ops;

        size_t tr_size = 0;
        uint32_t n_traces = 0;
        如果 (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * sizeof(htp_trace_desc);
        }

        const size_t m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        如果 (rsp.n_ops > 0) {
            auto & ops = op_cache[slot];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            const htp_prof_desc * pd = (const htp_prof_desc *) p_ptr;
            const htp_trace_desc * trace_events = nullptr;
            如果 (opt_profile == 3) {
                trace_events = (const htp_trace_desc *) (p_ptr + p_size);
            }

            如果 (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            为 (uint32_t i = 0; i < rsp.n_ops; i++) {
                如果 (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            如果 (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

void ggml_hexagon_session::flush_peers() {
    auto vpeers = std::move(virt_peers);
    virt_peers.clear();
    为 (auto * peer : vpeers) {
        peer->flush_sync();
    }

    auto ppeers = std::move(phys_peers);
    phys_peers.clear();
    为 (auto * peer : ppeers) {
        peer->flush_async();
    }

    为 (auto & sub : this->mdev.sessions) {
        sub->flush_peers();
    }
}

void ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

void ggml_hexagon_session::flush_pending(bool all) {
    为 (auto & sub : this->mdev.sessions) {
        sub->flush_pending(all);
        如果 (sub->last_error > HTP_STATUS_OK) {
            this->last_error = sub->last_error.load();
        }
    }

    当 (this->batch_rsp_seq < this->batch_req_seq) {
        struct htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        struct dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        const uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        整数 err = dspqueue_read(this->queue, &flags, 1, &n_dbufs, &dbuf, sizeof(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        如果 (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            continue;
        }

        如果 (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        如果 (rsp_size != sizeof(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", this->c_name(), rsp_size, n_dbufs);
        }

        如果 (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", this->c_name(), status_to_str(rsp.status));
            this->last_error = rsp.status;
            为 (auto & sub : this->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->pop(rsp, dbuf);

        GGML_ASSERT(rsp.seq == this->batch_rsp_seq + 1);
        this->batch_rsp_seq = rsp.seq;

        如果 (!all) break;
    }
}

void ggml_hexagon_session::flush_sync(bool all) {
    flush_async();
    flush_pending(all);
}

void ggml_hexagon_session::start_batch() {
    如果 (this->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

void ggml_hexagon_session::flush_batch(size_t min_ops) {
    如果 (op_batch->n_ops < min_ops) { 返回; }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    const uint64_t seq = ++this->batch_req_seq;

    op_batch->update_mdev_group(this->mdev.idx);

    如果 (!op_queue->push(req, dbuf, op_batch, seq)) {
        flush_pending(false);
        op_queue->push(req, dbuf, op_batch, seq);
    }

    为 (auto & sub : this->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_buffer sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        如果 (!sub->op_queue->push(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(false);
            sub->op_queue->push(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        整数 err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, sizeof(sub_req), (const uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        如果 (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (unsigned) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", this->c_name(), dbuf.ptr, dbuf.size);

    整数 err = dspqueue_write(this->queue, 0, 1, &dbuf, sizeof(req), (const uint8_t*) &req, DSPQUEUE_TIMEOUT);
    如果 (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", this->c_name(), (unsigned) err);
    }

    op_batch->reset();
}

void ggml_hexagon_session::enqueue_op(const htp_opnode & node) {
    auto clone_tensor_buffer = [this](const ggml_tensor * t) {
        auto sbuf = this->mmap_tensor(t);
        如果 (!sbuf) 返回;
        如果 (sbuf->sess != this) {
            this->clone_buffer(sbuf);
        }
        为 (auto & sub : this->mdev.sessions) {
            sub->clone_buffer(sbuf);
        }
    };

    为 (auto t : node.get_inputs()) {
        clone_tensor_buffer(t);
    }
    为 (auto t : node.get_outputs()) {
        clone_tensor_buffer(t);
    }

    如果 (opt_opfusion && op_batch->try_fuse(node)) {
        返回;
    }

    如果 (!op_batch->fit_op(node)) {
        flush_async();
    }

    如果 (op_batch->empty()) {
        start_batch();
    }

    如果 (!op_batch->fit_op(node)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->add_op(node);
}

void ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_node(HTP_OP_MDEV_GROUP);

    uint8_t * fence_slot = this->mdev_fence_slot;

    static ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.buffer       = &this->fence_buf->backend_buffer;
    dummy_t.extra        = &fence_extra;
    dummy_t.data         = (void *) fence_slot;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.ne[1]        = (int64_t) this->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) this->mdev.idx;

    ggml_tensor * node = group_node.add_dummy(dummy_t);
    node->src[0] = node;
    group_node.init(node);
    group_node.outputs.clear();
    group_node.name = "MDEV_GROUP";

    为 (auto & sub : this->mdev.sessions) {
        sub->clone_buffer(this->fence_buf);
    }

    op_batch->add_op(group_node);
}

void ggml_hexagon_session::enqueue_cpy(const ggml_tensor * src, ggml_tensor * dst, const ggml_tensor * sync_tensor, uint32_t fence_seq) {
    const bool with_fence = sync_tensor != nullptr;
    htp_opnode cpy_node(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* node = cpy_node.add_dummy(*dst);
    node->op     = GGML_OP_CPY;
    node->src[0] = const_cast<ggml_tensor *>(src);
    node->src[1] = with_fence ? cpy_node.add_dummy(*sync_tensor) : nullptr;
    如果 (with_fence) {
        node->op_params[0] = (int32_t) fence_seq;
    }

    cpy_node.init(node);
    如果 (with_fence) {
        cpy_node.name = "CPY+FENCE";
    }
    const bool ok = ggml_hexagon_precompute_cpy_params(this, node, (struct htp_copy_kernel_params *) cpy_node.kernel_params);
    const auto * kparams = (const struct htp_copy_kernel_params *) cpy_node.kernel_params;
    如果 (ok && !with_fence && kparams->total_elems == 0) {
        返回;
    }
    this->enqueue_op(cpy_node);
}

void ggml_hexagon_session::enqueue_fence(const ggml_tensor * sync_tensor, uint32_t fence_seq, bool wait) {
    htp_opnode sync_node(HTP_OP_FENCE);

    ggml_tensor* node = sync_node.add_dummy(*sync_tensor);
    node->op           = GGML_OP_NONE;
    node->src[0]       = node;
    node->op_params[0] = (int32_t) fence_seq;
    node->op_params[1] = wait ? 0 : 1;

    sync_node.init(node);
    sync_node.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    this->enqueue_op(sync_node);
}

static bool ggml_hexagon_precompute_allreduce_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    bool has_add,
    bool is_row_bcast,
    bool is_shard_ok,
    struct htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    const uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    const uint32_t elem_size = (dst->type == GGML_TYPE_F16) ? sizeof(ggml_fp16_t) : sizeof(float);
    const bool is_contiguous = ggml_is_contiguous(dst);

    const uint32_t ne0 = (uint32_t) dst->ne[0];
    const uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    const bool use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    如果 (has_add) {
        // sharded reduce-scatter for contiguous in-place add
        如果 (opt_ar_scatter && is_shard_ok && use_1d && !is_row_bcast) {
            const uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            const uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            const uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_start = (int32_t) rank_elem_start;
            kparams->rank_nelem      = (int32_t) (rank_elem_end - rank_elem_start);
        } 否则 如果 (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } 否则 {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } 否则 {
        kparams->n_dsts = (int32_t) n_ranks;
        如果 (use_1d) {
            const uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            const uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            const uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            const uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } 否则 {
            const uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            const uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            const uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            const uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    如果 (use_1d) {
        const uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        const size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_count(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        如果 (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        当 ((size_t) kparams->vtcm_size > sess->vtcm_size && block_elems > 128) {
            const size_t max_bytes_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((size_t) (max_bytes_per_buf / elem_size), 128);
            如果 (block_elems < 128) break;
            kparams->block_elems          = block_elems;
            kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
        }

        如果 (sess->vtcm_size < (size_t) kparams->vtcm_size || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            返回 false;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        返回 true;
    } 否则 {
        const uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        const size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_count(n_ranks, n_threads, has_add, is_row_bcast);

        const uint32_t row_bytes = ne0 * elem_size;
        const uint32_t row_size_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_size_aligned = row_size_aligned;

        const uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        当 ((size_t) kparams->vtcm_size > sess->vtcm_size && block_rows > 1) {
            const size_t max_rows_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2 * row_size_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
            如果 (max_rows_per_buf == 0) break;
        }

        如果 (sess->vtcm_size < (size_t) kparams->vtcm_size || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            返回 false;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        返回 true;
    }
}

void ggml_hexagon_session::enqueue_allreduce(
    const ggml_tensor * dst,
    const std::vector<const ggml_tensor *> & src_tensors,
    const std::vector<const ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_node(HTP_OP_ALLREDUCE);

    ggml_tensor* node = ar_node.add_dummy(*dst);
    node->op           = GGML_OP_NONE;
    node->op_params[0] = (int32_t) fence_seq_entry;
    node->op_params[1] = (int32_t) fence_seq_exit;

    ar_node.init(node);

    ar_node.inputs.clear();
    为 (size_t i = 0; i < src_tensors.size(); i++) {
        ar_node.inputs.push_back(src_tensors[i]);
    }
    为 (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_node.inputs.push_back(ar_node.add_dummy(*sync_tensors[i]));
    }

    ar_node.outputs.clear();
    为 (size_t i = 0; i < src_tensors.size(); i++) {
        ar_node.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        this, dst, rank, n_ranks, false, false, /*is_shard_ok=*/ false,
        (struct htp_allreduce_kernel_params *) ar_node.kernel_params);

    ar_node.name = "ALLREDUCE";
    this->enqueue_op(ar_node);
}

ggml_hexagon_shared_buffer * ggml_hexagon_session::mmap_tensor(const ggml_tensor * t) {
    如果 (!t) 返回 nullptr;

    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
    如果 (!sbuf->mapped) {
        const bool is_weight = ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        const bool extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    返回 sbuf;
}

bool ggml_hexagon_session::clone_buffer(const ggml_hexagon_shared_buffer * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    如果 (sbuf->sess == this) 返回 true;

    auto mem = sbuf->mem;
    整数   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    如果 (this->cloned_buffers.find(fd) != this->cloned_buffers.end()) 返回 true;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", this->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    auto clone = std::make_unique<ggml_hexagon_shared_buffer>(this, *sbuf);
    try {
        clone->mmap(sbuf->extended);
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", this->c_name(), exc.what());
        返回 false;
    }

    this->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(this);
    返回 true;
}

void ggml_hexagon_session::release_buffer(const ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    整数   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    auto it = this->cloned_buffers.find(fd);
    如果 (it != this->cloned_buffers.end()) {
        auto clone = std::move(it->second);
        this->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(this);
}

void ggml_hexagon_session::unclone_buffer(const ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    为 (auto * sess : sessions) {
        sess->release_buffer(sbuf);
    }
}

static size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    const size_t MiB = 1024 * 1024;
    const size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t step = 256u * MiB;

    try {
        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;
        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;
        sbufs.push_back(new ggml_hexagon_shared_buffer(sess, GiB, true)); vmem += GiB;

        当 (1) {
            sbufs.push_back(new ggml_hexagon_shared_buffer(sess, step, true));
            vmem += step;
        }
    } catch (...) { }

    为 (auto b : sbufs) { delete b; }

    返回 vmem - step; // backoff to account for overhead from internal mappings
}

void ggml_hexagon_session::allocate(const ggml_hexagon_device_config & config) noexcept(false) {
    整数 phys_idx = config.physical_idx;
    整数 virt_idx = config.virtual_idx;

    this->valid_session = false;
    this->valid_handle  = false;
    this->valid_queue   = false;
    this->valid_iface   = false;

    this->name          = config.name;
    this->phys_idx      = phys_idx;
    this->virt_idx      = virt_idx;
    this->domain_id     = config.domain_id;
    this->session_id    = 0;
    this->batch_req_seq = 0;
    this->batch_rsp_seq = 0;
    this->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", this->name.c_str(), this->domain_id, phys_idx, virt_idx);

    如果 (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        throw std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    const std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    如果 (virt_idx > 0) {
        struct remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<char *>(dom_name.c_str());
        n.session_name     = const_cast<char *>(this->name.c_str());
        n.session_name_len = this->name.size();
        n.session_id       = virt_idx;

        整数 err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (void *) &n, sizeof(n));
        如果 (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           this->c_name(), phys_idx, virt_idx, err);
            throw std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        this->session_id    = n.session_id;
        this->domain_id     = n.effective_domain_id;
        this->valid_session = true;
    } 否则 {
        struct remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<char *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        整数 err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (void *) &eff, sizeof(eff));
        如果 (err == AEE_SUCCESS) {
            this->domain_id = eff.effective_domain_id;
        } 否则 {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           this->name.c_str(), err, this->domain_id);
        }
    }

    // Enable unsigned modules
    {
        struct remote_rpc_control_unsigned_module u;
        u.domain = this->domain_id;
        u.enable = 1;
        整数 err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *) &u, sizeof(u));
        如果 (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", this->c_name(), err);
            throw std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    char session_uri[256];
    {
        char htp_uri[256];
        snprintf(htp_uri, sizeof(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        struct remote_rpc_get_uri u = {};
        u.session_id      = this->session_id;
        u.domain_name     = const_cast<char *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<char *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = sizeof(session_uri);

        整数 err = remote_session_control(FASTRPC_GET_URI, (void *) &u, sizeof(u));
        如果 (err != AEE_SUCCESS) {
            snprintf(session_uri, sizeof(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), this->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          this->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    整数 err = htp_iface_open(session_uri, &this->handle);
    如果 (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", this->c_name(), session_uri, err);
        throw std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    this->valid_handle = true;

    // Query HW info and resolve session options
    {
        unsigned 整数 hw_n_threads = 0;
        unsigned 整数 hw_n_hvx     = 0;
        unsigned 整数 hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        整数 hw_err = htp_iface_hwinfo(this->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        如果 (hw_err == 0) {
            const uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            this->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_threads) : max_n_threads;
            this->n_hvx     = this->n_threads;
            this->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            this->vtcm_size = (uint64_t)hw_vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          this->c_name(), this->n_threads, this->n_hvx, this->n_hmx,
                          (unsigned long long)(this->vtcm_size / (1024 * 1024)));
        } 否则 {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", this->c_name(), hw_err);
            const uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            this->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_threads;
            this->n_hvx     = this->n_threads;
            this->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            this->vtcm_size = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        struct remote_rpc_control_latency l;
        l.enable = 1;

        整数 err = remote_handle64_control(this->handle, DSPRPC_CONTROL_LATENCY, (void *) &l, sizeof(l));
        如果 (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", this->c_name(),
                  this->session_id, this->domain_id, session_uri, (unsigned long) this->handle);

    const size_t req_q_size = (sizeof(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    const size_t rsp_q_size = (sizeof(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(this->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (void *) this,  // Callback context
                          &queue);
    如果 (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", this->name.c_str(), (unsigned) err);
        throw std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    this->valid_queue = true;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &this->queue_id);
    如果 (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        throw std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    如果 (opt_etm) {
        err = htp_iface_etm(this->handle, 1);
        如果 (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    this->op_queue = new ggml_hexagon_opqueue(this, opt_opbatch, opt_opqueue);

    this->fence_buf = new ggml_hexagon_fence_buffer(this, &dev_ctx->fence_buffer_type, 64 * 1024);
    如果 (this->mdev.count > 1) {
        this->mdev_fence_slot = this->alloc_fence(this->mdev.count);
    }

    如果 (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(this);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", this->c_name(), opt_vmem);
    }
    const size_t shm_size = this->op_queue->shm_size();
    this->max_vmem = (opt_vmem > shm_size) ? (opt_vmem - shm_size) : opt_vmem;

    this->op_batch = new ggml_hexagon_opbatch(this, opt_opbatch, this->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(this->handle, this->session_id, this->queue_id, this->n_threads, opt_nhmx, this->max_vmem, (uint32_t) opt_power_mode);
    如果 (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", this->c_name(), (unsigned) err);
        throw std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    this->valid_iface = true;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", this->c_name(), htp_power_mode_to_str(opt_power_mode));

    如果 (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(this->handle, opt_profile, &pmu_conf);
        如果 (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }
}

void ggml_hexagon_session::release() noexcept(true) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", this->name.c_str());

    this->mdev.sessions.clear();

    整数 err;

    如果 (this->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(this->handle);
        如果 (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    delete this->op_batch;
    delete this->op_queue;
    为 (auto & it : this->cpy_fence_slots) {
        free_fence((void *) it.second, 1);
    }
    this->cpy_fence_slots.clear();

    如果 (this->fence_buf) {
        unclone_buffer(this->fence_buf);
        delete this->fence_buf;
        this->fence_buf = nullptr;
    }
    当 (!this->cloned_buffers.empty()) {
        release_buffer(this->cloned_buffers.begin()->second.get());
    }

    如果 (opt_etm) {
        err = htp_iface_etm(this->handle, 0);
        如果 (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    如果 (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(this->handle, 0, &pmu_conf);
        如果 (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    如果 (this->valid_queue) {
        err = dspqueue_close(queue);
        如果 (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (unsigned) err);
        }
    }

    如果 (this->valid_handle) {
        htp_iface_close(this->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(const ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_count) noexcept(false) {
    this->dev        = dev;
    this->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    this->mdev.idx   = mdev_idx;
    this->mdev.count = mdev_count > 0 ? mdev_count : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)this) & 0xFFFF;

    try {
        allocate(config);
        如果 (this->mdev.idx == 0 && !config.mdev_group.empty()) {
            为 (size_t i = 0; i < config.mdev_group.size(); i++) {
                this->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], this->dev, (uint32_t) (i + 1), this->mdev.count));
            }
        }
    } catch (const std::exception & exc) {
        release();
        throw;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(true) {
    release();
}

// ** backend interface

static bool ggml_hexagon_flash_attn_is_hmx_eligible(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * q,
    const struct ggml_tensor * k,
    const struct ggml_tensor * v,
    const struct ggml_tensor * sinks
) {
    如果 (sess->n_hmx == 0) {
        返回 false;
    }

    如果 (opt_fa_select < 2) {
        返回 false;
    }

    如果 ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        返回 false;
    }

    const uint32_t DK = q->ne[0];
    const uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    如果 (DK % 8 != 0 || DV % 8 != 0) {
        返回 false;
    }

    GGML_UNUSED(sinks);

    // Explicit force mode
    如果 (opt_fa_select > 2) {
        返回 true;
    }

    const uint32_t M = q->ne[1];

    // Prefill or batched decode
    如果 (M > 1) {
        返回 true;
    }

    const uint32_t n_head = q->ne[2];
    const uint32_t n_kv_heads = k->ne[2];
    const uint32_t G = n_kv_heads > 0 ? n_head / n_kv_heads : 1;
    const uint32_t S = k->ne[1];
    const uint32_t total_threads = sess->n_threads * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    如果 (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < total_threads))) {
        返回 true;
    }

    // Tile alignment for 32-row HMX tiles
    const bool is_tile_aligned = (G > 0 && (32 % G == 0));
    如果 (!is_tile_aligned) {
        如果 (sess->n_threads >= 6) {
            返回 false;
        }
        返回 S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    如果 (DK == 256) {
        s_cross = 4096;
    } 否则 如果 (DK <= 64) {
        s_cross = (sess->n_threads >= 8) ? 2048 : ((sess->n_threads >= 6) ? 768 : 512);
    } 否则 {
        s_cross = (sess->n_threads >= 8) ? 1024 : 512;
    }

    返回 S >= s_cross;
}

static bool ggml_hexagon_precompute_flash_attn_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_fa_kernel_params * kparams
) {
    如果 (opt_fa_select < 1) {
        返回 false;
    }

    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * q    = op->src[0];
    const struct ggml_tensor * k    = op->src[1];
    const struct ggml_tensor * v    = op->src[2];
    const struct ggml_tensor * mask = op->src[3];

    const uint32_t neq0 = q->ne[0];  // head_dim (DK)
    const uint32_t neq1 = q->ne[1];  // n_tokens
    const uint32_t neq2 = q->ne[2];  // n_heads

    const uint32_t nek1 = k->ne[1];  // kv_len

    const uint32_t nev0 = v->ne[0];  // head_dim (DV)

    const uint32_t DK = neq0;
    const uint32_t DV = nev0;

    const uint32_t n_kv_heads = k->ne[2];
    const uint32_t G          = neq2 / n_kv_heads;

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], sizeof(float));
    memcpy(&max_bias,      &op->op_params[1], sizeof(float));
    memcpy(&logit_softcap, &op->op_params[2], sizeof(float));

    如果 (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->head_split = (opt_fa_head_split != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    const uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    const struct ggml_tensor * sinks = op->src[4];
    如果 (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        const uint32_t DK_pad = hex_round_up(DK, 64);
        const uint32_t DV_pad = hex_round_up(DV, 64);
        size_t Br = 0, Bc = 0;
        整数 ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
        如果 (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            如果 (logit_softcap == 0.0f) {
                kparams->scale = scale * (float) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } 否则 {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (float) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);

            const size_t row_vec_bytes = hex_align_up(Bc * sizeof(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            const size_t m_line_bytes = hex_align_up(Bc * sizeof(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_line_bytes / sizeof(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_values(G);
            如果 (mask) {
                kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            返回 true;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    const size_t size_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    const size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    const size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), mask != nullptr, sinks != nullptr, n_head, sess->n_threads);

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_values(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_values(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_values(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_values(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_values(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_values(q->ne[3]/v->ne[3]);
    如果 (mask) {
        kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    返回 true;
}

static bool ggml_hexagon_supported_flash_attn_ext(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * src3 = op->src[3];
    const struct ggml_tensor * src4 = op->src[4];
    const struct ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    如果 ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        返回 false;
    }

    如果 (src3 && src3->type != GGML_TYPE_F16) {  // mask
        返回 false;
    }

    如果 (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        返回 false;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    如果 (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        返回 false;
    }

    struct htp_fa_kernel_params kparams;
    如果 (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        返回 false;
    }

    如果 ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        返回 false;
    }

    返回 true;
}

static bool ggml_hexagon_supported_gated_delta_net(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    如果 (opt_gdn_select < 1) {
        返回 false;
    }

    const struct ggml_tensor * q     = op->src[0];
    const struct ggml_tensor * k     = op->src[1];
    const struct ggml_tensor * v     = op->src[2];
    const struct ggml_tensor * g     = op->src[3];
    const struct ggml_tensor * beta  = op->src[4];
    const struct ggml_tensor * state = op->src[5];
    const struct ggml_tensor * dst   = op;

    如果 (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        返回 false;
    }

    如果 (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        返回 false;
    }

    const int64_t S_v      = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(op, 0);

    如果 (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        返回 false;
    }
    如果 (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        返回 false;
    }
    如果 ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        返回 false;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    如果 (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        返回 false;
    }
    如果 (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        返回 false;
    }

    const uint32_t total_rows = (uint32_t) (H * n_seqs);
    const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    const bool can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    如果 (can_use_hmx) {
        struct htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        如果 (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, true, &n_heads_batch)) {
            返回 false;
        }
    } 否则 {
        struct htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_threads);
        如果 (layout.total_bytes > sess->vtcm_size) {
            返回 false;
        }
    }

    返回 true;
}

static bool ggml_hexagon_matmul_is_hmx_eligible(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    bool is_matmul_id,
    bool is_batched
) {
    GGML_UNUSED(dst);

    如果 (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        返回 false;
    }

    const 整数 ne00  = src0->ne[0];
    const 整数 ne01  = src0->ne[1];
    const 整数 ne11  = src1->ne[1];
    const 整数 ne12  = src1->ne[2];
    const 整数 wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    如果 (ne01 % HTP_TENSOR_TILE_N_ROWS != 0 && is_matmul_id) {
        返回 false;
    }

    // HMX supports F16, F32, and repack quantized types.
    如果 (!htp_is_hmx_weight_type((ggml_type) wtype)) {
        返回 false;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    如果 (ne00 % HTP_TENSOR_TILE_N_COLS != 0) {
        返回 false;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    const bool is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    如果 (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        返回 false;
    }
    如果 (!is_matmul_id && is_batched && is_tiled) {
        返回 false;
    }

    // HMX assumes contiguous row-major layout.
    如果 (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        返回 false;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    const 整数 m = is_matmul_id ? ne12 : ne11;
    如果 (m <= HTP_MM_HMX_MIN_NROWS) {
        返回 false;
    }

    返回 true;

    GGML_UNUSED(dst);
}

static bool ggml_hexagon_precompute_hmx_mm_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * act,
    const struct ggml_tensor * dst,
    整数 wtype,
    整数 ne00_padded,
    整数 ne01_padded,
    整数 ne02,
    整数 ne11,
    整数 ne12,
    整数 ne11_padded,
    bool is_matmul_id,
    bool is_batched,
    size_t src2_size,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    const bool is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    const 整数 aligned_tile_size = is_tiled ? htp_tensor_get_tile_padded_size(wtype) : 0;
    const 整数 n_threads = (整数)sess->n_threads;
    const 整数 ne10 = act->ne[0];

    整数 m_for_solver = ne11;
    整数 m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    如果 (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        const bool dst_row_split = dst ? ggml_hexagon_can_row_partition(dst) : true;
        const bool act_row_split = ggml_hexagon_can_row_partition(act);
        如果 (dst_row_split && act_row_split) {
            m_for_solver = (ne11 + (整数) sess->mdev.count - 1) / (整数) sess->mdev.count;
            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
        }
    }
    const bool pipeline = is_matmul_id ? false : htp_mm_hmx_pipeline(m_for_solver);

    const bool is_batched_val = is_matmul_id ? false : is_batched;
    const 整数 group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    bool use_grouped = false;
    整数 act_threads_selected = 0;

    如果 (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && group_size > 1) {
        // Try grouped path first
        如果 (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = true;
        }
    }

    如果 (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        整数 m_id_rows = 0;
        如果 (dst && is_matmul_id) {
            const 整数 n_experts = ne02 > 0 ? ne02 : 1;
            const size_t total_expert_rows = (size_t) dst->ne[1] * dst->ne[2];
            整数 m_per_expert = (整数) ((total_expert_rows + n_experts - 1) / n_experts);
            如果 (sess->mdev.count > 1 && ggml_hexagon_can_row_partition(dst)) {
                m_per_expert = (m_per_expert + (整数) sess->mdev.count - 1) / (整数) sess->mdev.count;
            }
            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
        }
        const uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
        如果 (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            返回 false;
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
    kparams->div_n_act_threads = init_fastdiv_values(act_threads_selected);
    kparams->div_ne00_padded   = init_fastdiv_values(ne00_padded);
    kparams->vtcm_act_size     = 0;
    kparams->vtcm_bias_size    = (int32_t) src2_size;
    kparams->vtcm_dst_size     = 0;

    如果 (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } 否则 {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    返回 true;
}

static void ggml_hexagon_precompute_hvx_mm_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    整数 wtype,
    bool is_tiled,
    整数 ne02,
    整数 ne03,
    整数 ne10,
    整数 ne11,
    整数 ne12,
    整数 ne13,
    bool is_matmul_id,
    const size_t src2_row_size,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    如果 (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        返回;
    }

    kparams->n_hmx = 0;
    kparams->n_threads = sess->n_threads;

    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);

    const 整数 src1_nrows = ne11 * ne12 * ne13;

    如果 (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_size = htp_tensor_get_tile_size(wtype);
        kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);

        const bool k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        如果 (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (整数) sess->n_threads)
                                  ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);

            struct htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            为 (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], kparams->act_row_size, 0, d, true, false
                );
                如果 (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    break;
                }
            }
            如果 (L.total_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                返回;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = 0;
            kparams->vtcm_dst_size  = L.dst_bytes;
            返回;
        } 否则 {
            bool try_tiled = (k_align && opt_mm_select >= 1);
            如果 (try_tiled) {
                kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                如果 (src1_nrows < (整数) sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } 否则 {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                struct htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                为 (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, false, false
                    );
                    如果 (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        break;
                    }
                }

                uint32_t m_chunk = 0;
                如果 (htp_mm_hvx_solve_vtcm_params(
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
                    返回;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            返回;
        }
    } 否则 如果 (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        返回;
    } 否则 如果 (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        struct htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        如果 (htp_mm_hvx_solve_vtcm_params(
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
            返回;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        返回;
    } 否则 如果 (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        struct htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        如果 (htp_mm_hvx_solve_vtcm_params(
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
            返回;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        返回;
    } 否则 如果 (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        struct htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        如果 (htp_mm_hvx_solve_vtcm_params(
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
            返回;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        返回;
    } 否则 {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        返回;
    }
}

static void ggml_hexagon_precompute_matmul_params_impl(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    const size_t src2_row_size,
    const size_t src2_size,
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const 整数 ne00 = src0->ne[0];
    const 整数 ne01 = src0->ne[1];
    const 整数 ne02 = src0->ne[2];
    const 整数 ne03 = src0->ne[3];

    const 整数 ne10 = src1->ne[0];
    const 整数 ne11 = src1->ne[1];
    const 整数 ne12 = src1->ne[2];
    const 整数 ne13 = src1->ne[3];

    const bool is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    const 整数 wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    const 整数 ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    const 整数 ne11_padded = hex_round_up(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    const 整数 ne01_tiled  = hex_round_up(ne01, 32);

    const bool is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    const bool is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    const size_t vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    bool hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    如果 (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        如果 (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
            goto finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_size, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_values(ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_values(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
static bool ggml_hexagon_rows_stride(const int64_t * ne, const size_t * nb, size_t * stride) {
    size_t s = 0, next = 0;
    为 (整数 i = 1; i < GGML_MAX_DIMS; i++) {
        如果 (ne[i] == 1) continue;
        如果 (s == 0) { s = nb[i]; next = s * ne[i]; continue; }
        如果 (nb[i] != next) 返回 false;
        next *= ne[i];
    }
    *stride = s ? s : nb[1];
    返回 true;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
static bool ggml_hexagon_matmul_can_collapse(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    size_t s1, sd;
    返回 (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_size(src1->type) && ggml_hexagon_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_size(dst->type) && ggml_hexagon_rows_stride(dst->ne, dst->nb, &sd);
}

static bool ggml_hexagon_matmul_add_can_collapse(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst
) {
    如果 (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        返回 false;
    }
    如果 (!src2) {
        返回 true;
    }
    const int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    如果 (src2_nrows == 1) {
        返回 true;
    }
    size_t s2;
    返回 src2->nb[0] == ggml_type_size(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_rows_stride(src2->ne, src2->nb, &s2);
}

static ggml_tensor ggml_hexagon_tensor_collapse_rows(const struct ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    返回 c;
}

static void ggml_hexagon_precompute_matmul_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
) {
    如果 (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        const ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        const ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        返回;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

static void ggml_hexagon_precompute_fused_matmul_add_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * src2,
    const struct ggml_tensor * dst,
    struct htp_mm_kernel_params * kparams
) {
    如果 (ggml_hexagon_matmul_add_can_collapse(src0, src1, src2, dst)) {
        const ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        const ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        const ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        const struct ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        const size_t src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
        kparams->collapse = 1;
        返回;
    }
    const size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

static bool ggml_hexagon_precompute_binary_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_binary_kernel_params * kparams,
    const struct ggml_tensor * src2
) {
    memset(kparams, 0, sizeof(*kparams));

    const size_t elem_size = ggml_type_size(src0->type);
    const size_t src0_row_size = src0->ne[0] * elem_size;
    const size_t src1_row_size = src1->ne[0] * elem_size;
    const size_t dst_row_size  = dst->ne[0]  * elem_size;

    const size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    const size_t src1_row_size_aligned = hex_round_up(src1_row_size, 128);
    const size_t dst_row_size_aligned  = hex_round_up(dst_row_size,  128);

    如果 (!htp_binary_strides_safe(src0, src1, dst, elem_size)) {
        返回 false;
    }

    const bool is_add_id = op == HTP_OP_ADD_ID;
    const bool is_scalar = !is_add_id && src1->ne[0] == 1;
    const bool is_row_bcast = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    const bool is_same_shape = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    const bool is_complex   = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    const bool is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    const bool is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    如果 (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        const uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        const uint32_t n_threads = sess->n_threads;
        const uint32_t max_chunk_elems = 32768 / elem_size;
        const uint32_t min_chunk_elems = 256;
        const uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
        const uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        const uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

        kparams->kernel_type           = HTP_BINARY_KERNEL_CHUNKED;
        kparams->n_threads             = n_threads;
        kparams->rows_per_buffer       = 1;
        kparams->src0_row_size_aligned = src0_row_size_aligned;
        kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
        kparams->dst_row_size_aligned  = dst_row_size_aligned;
        kparams->src1_size             = 0;
        kparams->chunk_size            = chunk_size;
        kparams->chunk_bytes           = chunk_bytes;
        kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

        struct htp_binary_vtcm_layout L;
        htp_binary_vtcm_layout_build(&L, kparams, sess->vtcm_size);
        如果 (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
            返回 false;
        }

        kparams->vtcm_size = L.total_bytes;
        返回 true;
    }

    enum htp_binary_kernel_type kernel_type;
    size_t src1_size = 0;
    size_t src2_size = 0;

    如果 (is_add_id) {
        GGML_ASSERT(src2 != nullptr);
        kernel_type = HTP_BINARY_KERNEL_ADD_ID;
        src1_size = hex_round_up(src1->ne[1] * src1_row_size_aligned, 128);
        src2_size = hex_round_up(ggml_nelements(src2) * sizeof(int32_t), 128);
        如果 (src1_size + src2_size > sess->vtcm_size) {
            返回 false;
        }
    } 否则 如果 (is_row_bcast) {
        kernel_type = HTP_BINARY_KERNEL_ROW_BCAST;
        src1_size = src1_row_size_aligned;
    } 否则 如果 (is_scalar) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size, 128);
        如果 (src1_size > sess->vtcm_size) {
            返回 false;
        }
        kernel_type = HTP_BINARY_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } 否则 如果 (is_same_shape) {
        kernel_type = HTP_BINARY_KERNEL_SAME_SHAPE;
    } 否则 如果 (is_complex) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        如果 (src1_size > sess->vtcm_size) {
            返回 false;
        }
        kernel_type = HTP_BINARY_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } 否则 {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        如果 (src1_size > sess->vtcm_size) {
            返回 false;
        }
        kernel_type = HTP_BINARY_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_threads = sess->n_threads;
    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;
    kparams->src1_size             = src1_size;
    kparams->src2_size             = src2_size;

    const bool src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    const bool src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    const bool dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    const bool src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    const bool src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    const bool dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    bool split_at_ne01 = false;
    如果 (src0->ne[2] > 1) {
        如果 (kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE) {
            split_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } 否则 如果 (kernel_type == HTP_BINARY_KERNEL_ROW_BCAST) {
            split_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } 否则 {
            split_at_ne01 = true;
        }
    }
    kparams->split_at_ne01 = split_at_ne01 ? 1 : 0;

    bool split_at_ne02 = false;
    如果 (src0->ne[3] > 1) {
        如果 (kernel_type == HTP_BINARY_KERNEL_SAME_SHAPE) {
            split_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } 否则 如果 (kernel_type == HTP_BINARY_KERNEL_ROW_BCAST) {
            split_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } 否则 {
            split_at_ne02 = true;
        }
    }
    kparams->split_at_ne02 = split_at_ne02 ? 1 : 0;

    struct htp_binary_vtcm_layout L;
    htp_binary_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    如果 (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {
        如果 (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            const uint32_t total_elems = (uint32_t) ggml_nelements(src0);
            const uint32_t n_threads = sess->n_threads;
            const uint32_t max_chunk_elems = 32768 / elem_size;
            const uint32_t min_chunk_elems = 256;
            const uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
            const uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
            const uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

            kparams->kernel_type           = HTP_BINARY_KERNEL_CHUNKED;
            kparams->n_threads             = n_threads;
            kparams->rows_per_buffer       = 1;
            kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
            kparams->src1_size             = 0;
            kparams->chunk_size            = chunk_size;
            kparams->chunk_bytes           = chunk_bytes;
            kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

            htp_binary_vtcm_layout_build(&L, kparams, sess->vtcm_size);
            如果 (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
                返回 false;
            }

            kparams->vtcm_size = L.total_bytes;
            返回 true;
        }
        返回 false;
    }

    kparams->rows_per_buffer = L.rows_per_buffer;
    kparams->vtcm_size       = L.total_bytes;

    返回 true;
}

static void ggml_hexagon_precompute_unary_params(
    const struct ggml_hexagon_session * sess,
    uint32_t op,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    const size_t elem_size = ggml_type_size(src0->type);

    const size_t src0_data_row_size = src0->ne[0] * elem_size;
    const size_t dst_data_row_size  = dst->ne[0]  * ggml_type_size(dst->type);

    const size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    const size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = 0;
    bool broadcast_weight = false;

    如果 (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_data_row_size = src1->ne[0] * ggml_type_size(src1->type);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->broadcast_weight      = broadcast_weight;

    struct htp_unary_vtcm_layout L;
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

    const uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_values(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_values(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_values(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_values(tiles_per_row);
}

static void ggml_hexagon_precompute_get_rows_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_get_rows_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];

    const uint32_t ne10 = src1->ne[0];
    const uint32_t ne11 = src1->ne[1];
    const uint32_t ne12 = src1->ne[2];
    const uint32_t nr = ne10 * ne11 * ne12;

    const bool tiled = ggml_hexagon_tensor_is_tiled(src0);
    const 整数 wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    如果 (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } 否则 如果 (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } 否则 {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    const uint32_t total_tasks = nr;

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, total_tasks);

    struct htp_get_rows_vtcm_layout vtcm_layout = {};
    当 (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, kparams->n_threads);
        如果 (vtcm_layout.total_bytes <= sess->vtcm_size) {
            break;
        }
        --kparams->n_threads;
    }

    如果 (kparams->n_threads == 0 && total_tasks > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, 1);
    }

    kparams->vtcm_size = (total_tasks == 0) ? 0 : vtcm_layout.total_bytes;
    kparams->tasks_per_thread = kparams->n_threads > 0 ? (total_tasks + kparams->n_threads - 1) / kparams->n_threads : 0;
    kparams->total_tasks = total_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_values(ne10) : fastdiv_values{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_values(ne10 * ne11) : fastdiv_values{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_values(ne02) : fastdiv_values{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_values(ne03) : fastdiv_values{0, 0};
}

static void ggml_hexagon_precompute_set_rows_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0, // values
    const struct ggml_tensor * src1, // indices
    const struct ggml_tensor * dst,  // destination
    struct htp_set_rows_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t nr = src0->ne[1];

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, nr);
    kparams->tasks_per_thread = (nr + kparams->n_threads - 1) / kparams->n_threads;
    kparams->total_tasks = nr;

    kparams->div_ne11 = init_fastdiv_values(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_values(src1->ne[2]);
    kparams->div_tasks_per_thread = init_fastdiv_values(kparams->tasks_per_thread);
    kparams->div_ne02 = init_fastdiv_values(src0->ne[2]);

    struct htp_set_rows_vtcm_layout vtcm_layout;
    htp_set_rows_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_threads);
    kparams->vtcm_size = vtcm_layout.total_bytes;
}

static void ggml_hexagon_precompute_softmax_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);

    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    &op->op_params[0], sizeof(float));
    memcpy(&max_bias, &op->op_params[1], sizeof(float));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    const uint32_t n_head = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    kparams->n_head      = n_head;
    kparams->n_head_log2 = n_head_log2;

    如果 (max_bias > 0.0f && n_head_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_head_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    } 否则 {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    const uint32_t ne00 = src0->ne[0];
    const uint32_t ne10 = src1 ? src1->ne[0] : 1;

    struct htp_softmax_vtcm_layout layout;
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
    如果 (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } 否则 如果 (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F16;
    } 否则 {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    如果 (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_values(src0->ne[1]);
    如果 (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_values(src0->ne[2]);
    const uint32_t ne12 = src1 ? src1->ne[2] : 1;
    const uint32_t ne13 = src1 ? src1->ne[3] : 1;
    如果 (ne12 > 0) kparams->div_ne12 = init_fastdiv_values(ne12);
    如果 (ne13 > 0) kparams->div_ne13 = init_fastdiv_values(ne13);
}

static void ggml_hexagon_precompute_rope_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    const uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    struct htp_rope_vtcm_layout layout;
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

    如果 (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_values(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_values(dst->ne[1]);
    }
}

static void ggml_hexagon_precompute_ssm_conv_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    struct htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t d_conv  = (uint32_t) src1->ne[0];
    const uint32_t d_inner = (uint32_t) src0->ne[1];
    const uint32_t n_t     = (uint32_t) dst->ne[1];
    const uint32_t n_s     = (uint32_t) dst->ne[2];
    const uint32_t ncs     = (uint32_t) src0->ne[0];

    const uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    const uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
    const uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);

    const uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * sizeof(float), 128) + 128;
    const uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 128);
    const uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

    uint32_t vtcm_src0_per_thread = 0;
    uint32_t vtcm_dst_per_thread  = 0;

    如果 (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;

        const uint32_t src0_tile_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * sizeof(float), 128);
        const uint32_t src0_T_bytes        = hex_round_up(d_conv * d_inner_per_thread * sizeof(float), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw_bytes + src0_T_bytes;

        const uint32_t dst_tile_bytes = hex_round_up(d_inner_per_thread * sizeof(float), 128);
        vtcm_dst_per_thread = 2 * dst_tile_bytes;
    } 否则 {
        const size_t vtcm_budget = (sess->vtcm_size > 0 ? sess->vtcm_size / n_threads : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        const uint32_t src0_block_T    = hex_round_up(ncs * 32 * sizeof(float), 128);
        const size_t   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
        const size_t   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);

        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
        target_max_tile = (std::max)(target_max_tile, 32u);
        target_max_tile = (std::min)(target_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * sizeof(float)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        如果 (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        如果 (d_inner_tile > target_max_tile) {
            d_inner_tile = target_max_tile;
        }
        如果 (d_inner_tile > d_inner_per_thread) {
            d_inner_tile = d_inner_per_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        const uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * sizeof(float), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw + src0_block_T;

        vtcm_dst_per_thread = 2 * hex_round_up(d_inner_tile * n_t * sizeof(float), 128);
    }

    kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
    kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
    kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;

    kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
    kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
    kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
    kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
}

static void ggml_hexagon_precompute_gated_delta_net_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * q     = op->src[0];
    const struct ggml_tensor * k     = op->src[1];
    const struct ggml_tensor * v     = op->src[2];
    const struct ggml_tensor * g     = op->src[3];
    const struct ggml_tensor * state = op->src[5];

    const uint32_t S_v      = (uint32_t) v->ne[0];
    const uint32_t H        = (uint32_t) v->ne[1];
    const uint32_t n_tokens = (uint32_t) v->ne[2];
    const uint32_t n_seqs   = (uint32_t) v->ne[3];
    const uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    const uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    const uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    const uint32_t total_rows = H * n_seqs;
    const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    const bool can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    struct htp_gdn_hmx_vtcm_layout hmx_layout;
    struct htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    如果 (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, true, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_size      = HTP_GDN_CHUNK_SIZE;
        kparams->n_chunks        = (n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_size       = (uint32_t) hmx_layout.total_bytes;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_bytes;
        kparams->vtcm_per_thread = (uint32_t) (hmx_layout.total_bytes / (n_threads > 0 ? n_threads : 1));
    } 否则 {
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
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / sizeof(float));
    kparams->state_size_per_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((float) S_v);

    如果 (H > 0)                  kparams->div_H         = init_fastdiv_values(H);
    如果 (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_values((uint32_t) q->ne[1]);
    如果 (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_values((uint32_t) k->ne[1]);
    如果 (rq3 > 0)                kparams->div_rq3       = init_fastdiv_values(rq3);
    如果 (rk3 > 0)                kparams->div_rk3       = init_fastdiv_values(rk3);
    如果 (kparams->n_threads > 0) kparams->div_n_threads = init_fastdiv_values(kparams->n_threads);
}

static void ggml_hexagon_precompute_sort_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    bool is_top_k,
    struct htp_sort_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t ne00       = src0->ne[0];
    const uint32_t k          = dst->ne[0];

    int32_t order = GGML_SORT_ORDER_DESC;
    如果 (!is_top_k) {
        order = ((const int32_t *) op->op_params)[0];
    }

    const uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    const size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    struct htp_sort_vtcm_layout layout;
    bool ok = htp_sort_solve_layout(&layout, ne00, total_rows, k, n_threads_max, vtcm_budget, is_top_k);
    GGML_ASSERT(ok);

    kparams->n_threads         = (int32_t) layout.n_threads;
    kparams->total_rows        = (int32_t) total_rows;
    kparams->row_start         = 0;
    kparams->row_end           = (int32_t) total_rows;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->order             = order;
    kparams->is_top_k          = is_top_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_size         = (int32_t) layout.total_bytes;
    kparams->phase1_slot_size  = (int32_t) layout.phase1_slot_size;
    kparams->merge_values_off  = (int32_t) layout.merge_values_off;
    kparams->merge_indices_off = (int32_t) layout.merge_indices_off;
    kparams->merge_elems       = (int32_t) layout.merge_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

static void ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    const uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    const int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    如果 (numer_hi < 0) {
        *ox_hi = 0;
    } 否则 {
        const uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    如果 (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

static bool ggml_hexagon_pool_shape_fits(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * dst
) {
    返回 ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

static void ggml_hexagon_precompute_pool_2d_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * dst,
    struct htp_pool_2d_kernel_params * kparams,
    bool is_pool_1d
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t src_x = (uint32_t) src0->ne[0];
    const uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    const uint32_t dst_x = (uint32_t) dst->ne[0];
    const uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    const uint32_t src_plane_bytes = src_x * src_y * sizeof(float);
    const uint32_t dst_plane_bytes = dst_x * dst_y * sizeof(float);
    const uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    const uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    struct htp_pool_vtcm_layout layout;
    const bool ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, sess->vtcm_size);
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
    kparams->avg_divide_count = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    const bool narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_path = (kparams->fast_path && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (float) (kparams->kernel_x * kparams->kernel_y);
}

static bool ggml_hexagon_precompute_concat_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    如果 (!src0 || !src1 || !dst) {
        返回 false;
    }

    整数 dim = ((const int32_t *) op->op_params)[0];
    如果 (dim < 0 || dim >= GGML_MAX_DIMS) {
        返回 false;
    }
    kparams->dim = dim;

    如果 (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        返回 false;
    }
    如果 (src0->type != dst->type || src1->type != dst->type) {
        返回 false;
    }

    const uint32_t type_size = ggml_type_size(dst->type);

    为 (整数 d = 0; d < GGML_MAX_DIMS; d++) {
        const int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        如果 (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            返回 false;
        }
    }

    const uint32_t max_gather_ne0 = 128 / type_size;
    const uint32_t total_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    const bool b_dense = (src1->nb[0] == type_size);
    const bool gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_size &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_size
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_size);

    如果 (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_size && dst->nb[1] == (size_t) dst->ne[0] * type_size &&
        src0->nb[0] == type_size &&
        (src1->nb[0] == type_size || src1->nb[1] == type_size) &&
        gather_strides_ok &&
        htp_gather_rows_tensor_outer_contiguous(dst) &&
        htp_gather_rows_tensor_outer_contiguous(src0) &&
        htp_gather_rows_tensor_outer_contiguous(src1)) {

        如果 (htp_gather_rows_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_size, total_rows,
                                         sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ROWS;
            kparams->n_threads   = (uint8_t) sess->n_threads;
            kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
            返回 true;
        }
    }

    const bool dma_strides_ok = (src0->nb[0] == type_size && src1->nb[0] == type_size && dst->nb[0] == type_size);

    如果 (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_threads   = 1;
        返回 true;
    }

    const bool is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    const bool is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    const bool transposed_rows_ok = (src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size);

    如果 (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        const uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 8;
        struct htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_size, n_threads);

        如果 (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
            返回 false;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_threads                         = n_threads;
        kparams->vtcm_size                         = layout.total_bytes;
        kparams->u.transposed.spad0_size_per_thread = layout.src0_spad_size_per_thread;
        kparams->u.transposed.spad1_size_per_thread = layout.src1_spad_size_per_thread;
        返回 true;
    }

    返回 false;
}

static bool ggml_hexagon_precompute_cpy_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    如果 (!src0 || !dst) {
        返回 false;
    }

    如果 (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        返回 false;
    }
    如果 (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        返回 false;
    }

    const int64_t nelem_src = ggml_nelements(src0);
    const int64_t nelem_dst = ggml_nelements(dst);
    如果 (nelem_src != nelem_dst || nelem_src < 0) {
        返回 false;
    }

    const uint32_t src_type_size = ggml_type_size(src0->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    kparams->src0_type_size = (uint8_t) src_type_size;
    kparams->dst_type_size  = (uint8_t) dst_type_size;
    kparams->total_elems    = (uint32_t) nelem_src;

    如果 (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        返回 true;
    }

    如果 (nelem_src == 1) {
        如果 (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            返回 true;
        }
        如果 ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            返回 true;
        }
        返回 false;
    }

    const bool sametype = (src0->type == dst->type);

    bool same_extents = true;
    为 (整数 d = 0; d < GGML_MAX_DIMS; d++) {
        如果 (src0->ne[d] != dst->ne[d]) {
            same_extents = false;
            break;
        }
    }

    const bool transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_size) || (dst->nb[0] != dst_type_size) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_size) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_size);
    const bool sameshape  = same_extents && !transposed;

    const bool src_is_contiguous = ggml_is_contiguous(src0);
    const bool dst_is_contiguous = ggml_is_contiguous(dst);

    如果 (sametype) {
        如果 (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            返回 true;
        }

        const uint32_t max_gather_ne0 = 128 / src_type_size;
        const uint32_t total_rows = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        如果 ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_size && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_size &&
            total_rows > 1 &&
            htp_gather_rows_tensor_outer_contiguous(src0)) {

            如果 (htp_gather_rows_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, false,
                                             src_type_size, total_rows,
                                             sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ROWS;
                kparams->n_threads   = (uint8_t) sess->n_threads;
                kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
                返回 true;
            }
        }

        如果 (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            返回 true;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_threads   = sess->n_threads > 0 ? (uint8_t) sess->n_threads : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_values((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_values((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_values((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_values((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_values((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_values((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        返回 true;
    }

    如果 (!sameshape) {
        返回 false;
    }

    const bool valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    如果 (!valid_conversion) {
        返回 false;
    }

    const uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 4;
    struct htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_size, dst_type_size, n_threads);

    如果 (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
        返回 false;
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
    kparams->u.convert.div_ne01      = init_fastdiv_values((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_values((uint32_t) (src0->ne[2] * src0->ne[1]));

    返回 true;
}

static void ggml_hexagon_precompute_fused_mmnx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0, // W0
    const struct ggml_tensor * act,  // x
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->n_threads = sess->n_threads;

    const 整数 ne00 = src0->ne[0];
    const 整数 ne01 = src0->ne[1];
    const 整数 ne02 = src0->ne[2];
    const 整数 ne03 = src0->ne[3];

    const 整数 ne10 = act->ne[0];
    const 整数 ne11 = act->ne[1];
    const 整数 ne12 = act->ne[2];
    const 整数 ne13 = act->ne[3];

    const bool is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    const 整数 wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    const 整数 ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    const 整数 ne11_padded = hex_round_up(ne11, 32);
    const 整数 ne01_tiled  = hex_round_up(ne01, 32);

    const size_t vtcm_budget = sess->vtcm_size;
    const bool is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    bool hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    如果 (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, nullptr, false, is_batched)) {
        如果 (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, false, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            goto finalize;
        }
    }

    如果 (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        返回;
    }

    {
        const 整数 act_nrows = ne11 * ne12 * ne13;
        const size_t act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
        const size_t src0_row_size = src0->nb[1];

        const uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        为 (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            struct htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
                0, src0_row_size, act_row_size, 0, d, false, true
            );
            如果 (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                break;
            }
        }

        struct htp_mm_hvx_vtcm_layout L;
        bool try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
            0, src0_row_size, act_row_size, 0, best_n_prefetch, false, true
        );

        如果 (try_tiled && L.total_bytes <= sess->vtcm_size) {
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
        } 否则 {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            返回;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_values(ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_values(ne12);
}

static void ggml_hexagon_precompute_fused_mmidnx_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * src0, // W0
    const struct ggml_tensor * act,  // x
    const struct ggml_tensor * dst,  // dst0
    int32_t n_weights,
    struct htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

static bool ggml_hexagon_op_mul_mat_use_fwht(const struct ggml_tensor * dst) {
    如果 (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        返回 false;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int64_t n = src1->ne[0];
    如果 (n < 32 || (n & (n - 1)) != 0) {
        返回 false;
    }

    返回 src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_rows(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

static bool ggml_hexagon_supported_mul_mat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    如果 (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        如果 (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            返回 false;
        }
        struct htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, nullptr, dst, &ukparams);
        返回 ukparams.vtcm_row_per_thread > 0 && (size_t) ukparams.vtcm_size <= sess->vtcm_size;
    }

    如果 (dst->type != GGML_TYPE_F32) {
        返回 false;
    }

    如果 (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        返回 false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q2_K: {
            如果 (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                返回 false;
            }

            const int64_t qk = ggml_blck_size(src0->type);
            如果 (src0->ne[0] % qk) {
                返回 false;
            }

            如果 (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                返回 false;
            }
            如果 (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                返回 false;
            }
            break;
        }

        case GGML_TYPE_F16:
            如果 (src0->nb[1] < src0->nb[0]) {
                返回 false;
            }
            如果 (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                返回 false;
            }
            如果 (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                返回 false;
            }
            break;

        case GGML_TYPE_F32:
            如果 (src1->type != GGML_TYPE_F32) {
                返回 false;
            }
            如果 (src0->nb[1] < src0->nb[0]) {
                返回 false;
            }
            如果 (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                返回 false;
            }
            如果 (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                返回 false;
            }
            break;

        case GGML_TYPE_BF16:
            如果 (src0->nb[1] < src0->nb[0]) {
                返回 false;
            }
            如果 (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                返回 false;
            }
            如果 (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                返回 false;
            }
            break;

        default:
            返回 false;
    }

    struct htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    如果 (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        返回 false;
    }

    返回 true;
}

static bool ggml_hexagon_supported_mul_mat_id(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    如果 (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        返回 false;
    }

    如果 (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        返回 false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q2_K:
            如果 (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                返回 false;
            }

            如果 (src0->ne[0] % ggml_blck_size(src0->type) != 0) {
                返回 false;
            }
            break;

        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
            如果 (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                返回 false;
            }
            如果 (src0->ne[0] % 32 != 0) {
                返回 false;
            }
            break;

        default:
            返回 false;
    }

    struct htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    如果 (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        返回 false;
    }

    返回 true;
}

static bool ggml_hexagon_supported_binary(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    如果 (src0->type == GGML_TYPE_F32) {
        如果 (src1->type != GGML_TYPE_F32) {
            返回 false;
        }
        如果 (dst->type != GGML_TYPE_F32) {
            返回 false;
        }
    }
    否则 如果 (src0->type == GGML_TYPE_F16) {
        如果 (src1->type != GGML_TYPE_F16) {
            返回 false;
        }
        如果 (dst->type != GGML_TYPE_F16) {
            返回 false;
        }
    }
    否则 {
        返回 false;
    }

    如果 (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        返回 false;
    }
    如果 (!ggml_are_same_shape(src0, dst)) {
        返回 false;
    }
    如果 (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        返回 false;
    }

    struct htp_binary_kernel_params kparams;
    返回 ggml_hexagon_precompute_binary_params(sess, op_remap_to_htp(op), src0, src1, dst, &kparams);
}

static bool ggml_hexagon_supported_add_id(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    如果 (!src2) {
        返回 false;
    }
    如果 (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        返回 false;
    }
    如果 (!ggml_are_same_shape(src0, dst)) {
        返回 false;
    }
    如果 (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        返回 false;
    }
    如果 (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        返回 false;
    }
    如果 (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || src2->nb[0] != sizeof(int32_t)) {
        返回 false;
    }

    // REVISIT: add support for non-contiguous tensors
    如果 (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        返回 false;
    }
    如果 (src2->nb[1] < src2->ne[0] * sizeof(int32_t)) {
        返回 false;
    }

    struct htp_binary_kernel_params kparams;
    返回 ggml_hexagon_precompute_binary_params(sess, HTP_OP_ADD_ID, src0, src1, dst, &kparams, src2);
}

static bool ggml_hexagon_supported_unary(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    如果 (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        返回 false;
    }
    如果 (dst->type != src0->type) {
        返回 false;
    }
    如果 (!ggml_is_contiguous_rows(src0)) {
        返回 false;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    如果 (src0->type == GGML_TYPE_F16) {
        switch (op->op) {
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_L2_NORM:
            case GGML_OP_SCALE:
            case GGML_OP_CLAMP:
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
                break;
            case GGML_OP_UNARY:
                如果 (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_STEP) {
                    返回 false;
                }
                break;
            default:
                返回 false;
        }
    }

    如果 (!ggml_are_same_shape(src0, dst)) {
        返回 false;
    }

    // dst must be contiguous; src0 may be non-contiguous
    如果 (!ggml_is_contiguous(dst)) {
        返回 false;
    }

    返回 true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_sum(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    如果 (src0->type != GGML_TYPE_F32) {
        返回 false;
    }
    如果 (dst->type != GGML_TYPE_F32) {
        返回 false;
    }

    如果 (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        返回 false;
    }

    返回 true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_sum_rows(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    如果 (src0->type != GGML_TYPE_F32) {
        返回 false;
    }
    如果 (dst->type != GGML_TYPE_F32) {
        返回 false;
    }

    // TODO: add support for non-contigiuos tensors
    如果 (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        返回 false;
    }

    返回 true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_argmax(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    如果 (src0->type != GGML_TYPE_F32) {
        返回 false;
    }
    如果 (dst->type != GGML_TYPE_I32) {
        返回 false;
    }

    如果 (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        返回 false;
    }

    返回 true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_pool_2d(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const int32_t * params = op->op_params;

    如果 (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        返回 false;
    }
    如果 (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        返回 false;
    }
    如果 (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        返回 false;
    }
    如果 (!ggml_hexagon_pool_shape_fits(src0, op)) {
        返回 false;
    }

    const int32_t kernel_x = params[1];
    const int32_t kernel_y = params[2];
    const int32_t stride_x = params[3];
    const int32_t stride_y = params[4];
    const int32_t pad_x    = params[5];
    const int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    如果 (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        返回 false;
    }

    const uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    const uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    struct htp_pool_vtcm_layout layout;
    返回 htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_threads, sess->vtcm_size);
}

static bool ggml_hexagon_supported_pool_1d(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const int32_t * params = op->op_params;

    如果 (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        返回 false;
    }
    如果 (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        返回 false;
    }
    如果 (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        返回 false;
    }
    如果 (!ggml_hexagon_pool_shape_fits(src0, op)) {
        返回 false;
    }

    const int32_t kernel = params[1];
    const int32_t stride = params[2];
    const int32_t pad    = params[3];
    如果 (kernel <= 0 || stride <= 0 || pad < 0) {
        返回 false;
    }

    const uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    const uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    struct htp_pool_vtcm_layout layout;
    返回 htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_threads, sess->vtcm_size);
}

static bool ggml_hexagon_supported_activations(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    如果 (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        返回 false;
    }
    如果 (dst->type != src0->type) {
        返回 false;
    }

    如果 (!ggml_is_contiguous_1(src0)) {
        返回 false;
    }
    如果 (!ggml_is_contiguous(dst)) {
        返回 false;
    }

    如果 (src1) {
        如果 (src1->type != src0->type) {
            返回 false;
        }
        如果 (!ggml_are_same_shape(src0, src1)) {
            返回 false;
        }
        如果 (!ggml_is_contiguous_1(src1)) {
            返回 false;
        }
    }

    返回 true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_softmax(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    如果 (src2) {
        返回 false;  // FIXME: add support for sinks
    }

    如果 (src0->type != GGML_TYPE_F32) {
        返回 false;
    }
    如果 (dst->type != GGML_TYPE_F32) {
        返回 false;
    }

    如果 (src0->ne[2] > 512) {
        返回 false;
    }

    如果 (src1) {
        如果 (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            返回 false;
        }
        如果 (src0->ne[0] != src1->ne[0]) {
            返回 false;
        }
        如果 (src1->ne[1] < src0->ne[1]) {
            返回 false;
        }
        如果 (src0->ne[2] % src1->ne[2] != 0) {
            返回 false;
        }
        如果 (src0->ne[3] % src1->ne[3] != 0) {
            返回 false;
        }
    }

    如果 (src1) {
        如果 (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            返回 false;
        }
    } 否则 {
        如果 (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            返回 false;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    const int64_t ne0 = src0->ne[0];
    如果 (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        返回 false;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    如果 (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        返回 false;
    }

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    struct htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != nullptr, src1 && src1->type == GGML_TYPE_F16, n_threads);
    如果 (layout.total_bytes > sess->vtcm_size) {
        返回 false;
    }

    返回 true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_set_rows(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * src1 = op->src[1]; // indices
    const struct ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    如果 (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        返回 false;
    }

    如果 (src0->type != GGML_TYPE_F32) {
        返回 false;
    }

    如果 (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        返回 false;
    }

    如果 (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        返回 false;
    }

    返回 true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_get_rows(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * src1 = op->src[1]; // indices
    const struct ggml_tensor * dst  = op;

    如果 (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        返回 false;
    }

    如果 (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(dst)) {
        返回 false;
    }

    如果 (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        返回 false;
    }

    如果 (src0->type == dst->type) {
        如果 (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            返回 false;
        }
    } 否则 如果 (src0->type == GGML_TYPE_I32) {
        返回 false;
    } 否则 如果 (dst->type != GGML_TYPE_F32) {
        返回 false;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    如果 (ggml_is_empty(op)) {
        返回 true;
    }

    struct htp_get_rows_kernel_params kparams;
    ggml_hexagon_precompute_get_rows_params(sess, src0, src1, dst, &kparams);
    如果 (kparams.n_threads == 0 || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        返回 false;
    }

    返回 true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_argsort(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * dst  = op;         // indices

    如果 (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        返回 false;
    }

    const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    const size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    struct htp_sort_vtcm_layout layout;
    如果 (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, false)) {
        返回 false;
    }

    返回 true;
}

static bool ggml_hexagon_supported_top_k(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // values
    const struct ggml_tensor * dst  = op;         // indices

    如果 (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        返回 false;
    }

    const uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    const size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    struct htp_sort_vtcm_layout layout;
    如果 (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, true)) {
        返回 false;
    }

    返回 true;
}

static bool ggml_hexagon_supported_rope(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];
    const struct ggml_tensor * dst  = op;

    如果 (!ggml_are_same_shape(src0, dst)) {
        返回 false;
    }

    如果 (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        返回 false;
    }

    如果 (src0->ne[0] <= 0) {
        返回 false;
    }

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    如果 (src0_nrows == 0) {
        返回 false;
    }

    const int32_t * op_params = &op->op_params[0];
    const 整数 n_dims = op_params[1];
    const 整数 mode   = op_params[2];
    const 整数 n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    如果 (n_dims < 0 || n_dims % 2 != 0) {
        返回 false;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    如果 (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        返回 false;
    }

    float freq_base;
    memcpy(&freq_base, op_params + 5, sizeof(float));
    如果 (freq_base < 0.0f) {
        返回 false;
    }

    如果 (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        返回 false;
    }

    const bool is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    如果 (mode == GGML_ROPE_TYPE_VISION) {
        如果 (n_dims != (整数) (src0->ne[0] / 2) || n_offs != 0) {
            返回 false;
        }
    }

    如果 (is_mrope) {
        const int32_t * sections = op_params + 11;
        如果 (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            返回 false;
        }
    }

    const int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    如果 (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        返回 false;
    }

    如果 (src2) {
        如果 (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            返回 false;
        }
        如果 (src2->ne[0] < (n_dims / 2)) {
            返回 false;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    如果 (src0->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        返回 false;
    }
    如果 (src0->nb[1] < src0->ne[0] * sizeof(float) || dst->nb[1] < dst->ne[0] * sizeof(float)) {
        返回 false;
    }

    const uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    const uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    struct htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    如果 (layout.total_bytes > sess->vtcm_size) {
        返回 false;
    }

    返回 true;
}

static bool ggml_hexagon_supported_ssm_conv(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    // Only support FP32 for now
    如果 (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        返回 false;
    }

    // Check IO tensor shapes and dims
    如果 (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        返回 false; // src0 should be effectively 3D
    }

    const 整数 d_conv  = src1->ne[0];
    const 整数 d_inner = src0->ne[1];
    const 整数 n_t = dst->ne[1];
    const 整数 n_s = dst->ne[2];

    如果 (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        返回 false;
    }
    如果 (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        返回 false;
    }
    如果 (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        返回 false;
    }
    如果 (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        返回 false;
    }
    如果 (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        返回 false;
    }
    如果 (src0->nb[1] != src0->ne[0] * sizeof(float) || src1->nb[1] != src1->ne[0] * sizeof(float)) {
        返回 false;
    }

    struct htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    如果 ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        返回 false;
    }

    返回 true;
}

static bool im2col_3d_calc_output_size(uint32_t input, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    如果 (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        返回 false;
    }

    const int64_t numerator = (int64_t) input + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    如果 (numerator < 0) {
        返回 false;
    }

    const uint64_t size = (uint64_t) (numerator / stride) + 1;
    如果 (size == 0 || size > UINT32_MAX) {
        返回 false;
    }

    *output = (uint32_t) size;
    返回 true;
}

static bool ggml_hexagon_precompute_im2col_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    如果 (!src0 || !src1 || !dst) {
        返回 false;
    }

    如果 (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        返回 false;
    }

    如果 (src1->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        返回 false;
    }

    const int32_t * op_params = (const int32_t *) op->op_params;
    const int32_t s0 = op_params[0];
    const int32_t s1 = op_params[1];
    const int32_t p0 = op_params[2];
    const int32_t p1 = op_params[3];
    const int32_t d0 = op_params[4];
    const int32_t d1 = op_params[5];
    const bool is_2D = (op_params[6] == 1);

    如果 (s0 <= 0 || d0 <= 0 || p0 < 0) {
        返回 false;
    }
    如果 (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        返回 false;
    }

    const uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    const uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    const uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    const uint32_t IW = (uint32_t) src1->ne[0];
    const uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    const uint32_t KW = (uint32_t) src0->ne[0];
    const uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    const uint32_t OW = (uint32_t) dst->ne[1];

    如果 (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        返回 false;
    }

    const uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    const uint64_t total_rows_64   = (uint64_t) N * OH;
    如果 (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX) {
        返回 false;
    }
    const uint32_t patch_stride = (uint32_t) patch_stride_64;
    const uint32_t total_rows   = (uint32_t) total_rows_64;

    如果 (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        返回 false;
    }
    如果 (is_2D && dst->ne[2] != OH) {
        返回 false;
    }

    const uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? sizeof(ggml_fp16_t) : sizeof(float);

    bool is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    如果 (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    const uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    const uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    const uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    如果 (is_patchembed) {
        const uint64_t src_row_bytes_raw = (uint64_t) IC * KH * IW * sizeof(float);
        const uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        如果 (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            const uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            const uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            struct htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            如果 (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                返回 true;
            }
        }
    }

    为 (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        const uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        如果 (Wb > UINT32_MAX) {
            如果 (owb == 1) break;
            continue;
        }

        const uint64_t src_row_bytes_raw = (uint64_t) IC * KH * Wb * sizeof(float);
        const uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        如果 (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            如果 (owb == 1) break;
            continue;
        }

        const uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        const uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        struct htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        如果 (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            返回 true;
        }
        如果 (owb == 1) break;
    }

    返回 false;
}

static bool ggml_hexagon_precompute_im2col_3d_params(
    const struct ggml_hexagon_session * sess,
    const struct ggml_tensor * op,
    struct htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;

    如果 (!src0 || !src1 || !dst) {
        返回 false;
    }

    如果 (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        返回 false;
    }

    如果 (src1->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        返回 false;
    }

    const int32_t * params = (const int32_t *) op->op_params;
    const int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    const int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    const int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    const int32_t ic = params[9];

    如果 (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        返回 false;
    }

    如果 (src1->ne[3] % ic != 0) {
        返回 false;
    }

    const uint32_t IC = (uint32_t) ic;
    const uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    如果 (N == 0) {
        返回 false;
    }

    uint32_t OW, OH, OD;
    如果 (!im2col_3d_calc_output_size(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_size(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_size(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        返回 false;
    }

    const uint32_t KD = (uint32_t) src0->ne[2];
    const uint32_t KH = (uint32_t) src0->ne[1];
    const uint32_t KW = (uint32_t) src0->ne[0];
    const uint32_t ID = (uint32_t) src1->ne[2];
    const uint32_t IH = (uint32_t) src1->ne[1];
    const uint32_t IW = (uint32_t) src1->ne[0];

    const uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    const uint64_t total_rows_64   = (uint64_t) N * OD * OH;
    const uint64_t dst_ne3_64      = (uint64_t) OD * N;

    如果 (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        返回 false;
    }
    const uint32_t patch_stride = (uint32_t) patch_stride_64;
    const uint32_t total_rows   = (uint32_t) total_rows_64;
    const uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    如果 (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        返回 false;
    }

    const uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? sizeof(ggml_fp16_t) : sizeof(float);

    const bool is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    const uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    const uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    const uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    如果 (is_patchembed) {
        const uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * IW * sizeof(float);
        const uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        如果 (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            const uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            const uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            struct htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            如果 (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                返回 true;
            }
        }
    }

    为 (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        const uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        如果 (Wb > UINT32_MAX) {
            如果 (owb == 1) break;
            continue;
        }

        const uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * Wb * sizeof(float);
        const uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        如果 (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            如果 (owb == 1) break;
            continue;
        }

        const uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        const uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        struct htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        如果 (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            返回 true;
        }
        如果 (owb == 1) break;
    }

    返回 false;
}

static bool ggml_hexagon_supported_im2col(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    struct htp_im2col_kernel_params kparams;
    返回 ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

static bool ggml_hexagon_supported_im2col_3d(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    struct htp_im2col_kernel_params kparams;
    返回 ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

static bool ggml_hexagon_supported_pad(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    如果 (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        返回 false;
    }

    const int32_t lp0 = ((const int32_t *) op->op_params)[0];
    const int32_t rp0 = ((const int32_t *) op->op_params)[1];
    const int32_t circular = ((const int32_t *) op->op_params)[8];

    如果 (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        返回 false;
    }

    返回 true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_cumsum(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    如果 (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        返回 false;
    }

    如果 (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        返回 false;
    }

    返回 true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_diag(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    // diag only supports F32 currently
    如果 (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        返回 false;
    }

    // Input must have ne[1] == 1 (vector input)
    如果 (src0->ne[1] != 1) {
        返回 false;
    }

    // Output must be square in first two dimensions
    如果 (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        返回 false;
    }

    返回 true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_solve_tri(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0]; // A
    const struct ggml_tensor * src1 = op->src[1]; // B
    const struct ggml_tensor * dst  = op;         // X

    如果 (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        返回 false;
    }

    如果 (src0->ne[0] != src0->ne[1]) {
        返回 false;
    }

    如果 (src0->ne[1] != src1->ne[1]) {
        返回 false;
    }

    如果 (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        返回 false;
    }

    如果 (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        返回 false;
    }

    返回 true;

    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_tri(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    如果 (src0->type != GGML_TYPE_F32) { 返回 false; }
    如果 (dst->type  != GGML_TYPE_F32) { 返回 false; }
    如果 (!ggml_are_same_shape(src0, dst)) { 返回 false; }
    如果 (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { 返回 false; }

    返回 true;

    GGML_UNUSED(sess);
}

static const char * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    返回 sess->c_name();
}

static void ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

static htp_op_code op_remap_to_htp(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_FLASH_ATTN_EXT:  返回 HTP_OP_FLASH_ATTN_EXT;
        case GGML_OP_MUL_MAT:
            如果 (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                返回 HTP_OP_HADAMARD;
            }
            返回 HTP_OP_MUL_MAT;
        case GGML_OP_MUL_MAT_ID:      返回 HTP_OP_MUL_MAT_ID;
        case GGML_OP_MUL:             返回 HTP_OP_MUL;
        case GGML_OP_ADD:             返回 HTP_OP_ADD;
        case GGML_OP_ADD_ID:          返回 HTP_OP_ADD_ID;
        case GGML_OP_SUB:             返回 HTP_OP_SUB;
        case GGML_OP_DIV:             返回 HTP_OP_DIV;
        case GGML_OP_DUP:
        case GGML_OP_CPY:             返回 HTP_OP_CPY;
        case GGML_OP_CONT:            返回 HTP_OP_CPY;
        case GGML_OP_GET_ROWS:        返回 HTP_OP_GET_ROWS;
        case GGML_OP_SET_ROWS:        返回 HTP_OP_SET_ROWS;
        case GGML_OP_SUM:             返回 HTP_OP_SUM;
        case GGML_OP_SUM_ROWS:        返回 HTP_OP_SUM_ROWS;
        case GGML_OP_POOL_2D:         返回 HTP_OP_POOL_2D;
        case GGML_OP_POOL_1D:         返回 HTP_OP_POOL_1D;
        case GGML_OP_ARGSORT:         返回 HTP_OP_ARGSORT;
        case GGML_OP_TOP_K:           返回 HTP_OP_TOP_K;
        case GGML_OP_ARGMAX:          返回 HTP_OP_ARGMAX;
        case GGML_OP_NORM:            返回 HTP_OP_NORM;
        case GGML_OP_L2_NORM:         返回 HTP_OP_L2_NORM;
        case GGML_OP_RMS_NORM:        返回 HTP_OP_RMS_NORM;
        case GGML_OP_CONCAT:          返回 HTP_OP_CONCAT;
        case GGML_OP_SCALE:           返回 HTP_OP_SCALE;
        case GGML_OP_CLAMP:           返回 HTP_OP_CLAMP;
        case GGML_OP_LEAKY_RELU:      返回 HTP_OP_LEAKY_RELU;
        case GGML_OP_SQR:             返回 HTP_OP_SQR;
        case GGML_OP_SQRT:            返回 HTP_OP_SQRT;
        case GGML_OP_LOG:             返回 HTP_OP_UNARY_LOG;
        case GGML_OP_SOFT_MAX:        返回 HTP_OP_SOFTMAX;
        case GGML_OP_SSM_CONV:        返回 HTP_OP_SSM_CONV;
        case GGML_OP_GATED_DELTA_NET: 返回 HTP_OP_GATED_DELTA_NET;
        case GGML_OP_ROPE:            返回 HTP_OP_ROPE;
        case GGML_OP_REPEAT:          返回 HTP_OP_REPEAT;
        case GGML_OP_CUMSUM:          返回 HTP_OP_CUMSUM;
        case GGML_OP_FILL:            返回 HTP_OP_FILL;
        case GGML_OP_DIAG:            返回 HTP_OP_DIAG;
        case GGML_OP_SOLVE_TRI:       返回 HTP_OP_SOLVE_TRI;
        case GGML_OP_TRI:             返回 HTP_OP_TRI;
        case GGML_OP_PAD:             返回 HTP_OP_PAD;
        case GGML_OP_IM2COL:          返回 HTP_OP_IM2COL;
        case GGML_OP_IM2COL_3D:       返回 HTP_OP_IM2COL_3D;
        case GGML_OP_ROLL:            返回 HTP_OP_ROLL;

        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_SILU:       返回 HTP_OP_UNARY_SILU;
                case GGML_UNARY_OP_GELU:       返回 HTP_OP_UNARY_GELU;
                case GGML_UNARY_OP_GELU_QUICK: 返回 HTP_OP_UNARY_GELU_QUICK;
                case GGML_UNARY_OP_GELU_ERF:   返回 HTP_OP_UNARY_GELU_ERF;
                case GGML_UNARY_OP_SIGMOID:    返回 HTP_OP_UNARY_SIGMOID;
                case GGML_UNARY_OP_NEG:        返回 HTP_OP_UNARY_NEG;
                case GGML_UNARY_OP_EXP:        返回 HTP_OP_UNARY_EXP;
                case GGML_UNARY_OP_SOFTPLUS:   返回 HTP_OP_UNARY_SOFTPLUS;
                case GGML_UNARY_OP_TANH:       返回 HTP_OP_UNARY_TANH;
                case GGML_UNARY_OP_ABS:        返回 HTP_OP_UNARY_ABS;
                case GGML_UNARY_OP_RELU:       返回 HTP_OP_UNARY_RELU;
                case GGML_UNARY_OP_STEP:       返回 HTP_OP_UNARY_STEP;
            default:
                break;
            }
            break;

        case GGML_OP_GLU:
            switch (ggml_get_glu_op(t)) {
                case GGML_GLU_OP_SWIGLU:       返回 HTP_OP_GLU_SWIGLU;
                case GGML_GLU_OP_SWIGLU_OAI:   返回 HTP_OP_GLU_SWIGLU_OAI;
                case GGML_GLU_OP_SWIGLU_CLAMP: 返回 HTP_OP_GLU_SWIGLU_CLAMP;
                case GGML_GLU_OP_GEGLU:        返回 HTP_OP_GLU_GEGLU;
                case GGML_GLU_OP_GEGLU_QUICK:  返回 HTP_OP_GLU_GEGLU_QUICK;
                case GGML_GLU_OP_GEGLU_ERF:    返回 HTP_OP_GLU_GEGLU_ERF;
                default: break;
            }
            break;

        default:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    返回 HTP_OP_INVALID;
}

static inline bool op_is_compute(ggml_tensor *node)
{
    返回 !ggml_op_is_empty(node->op) && !ggml_is_empty(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE);
}

static bool mm_is_hmx_eligible(const ggml_tensor * t) {
    如果 (opt_nhmx == 0) { 返回 false; }

    const ggml_tensor * src0 = t->src[0];
    const ggml_tensor * src1 = t->src[1];

    如果 (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        const ggml_tensor src1_c = ggml_hexagon_tensor_collapse_rows(src1);
        返回 ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, false, false);
    }

    const bool is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    const bool is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    返回 ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

static bool is_supported_mul_mat_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams) {
    如果 (kparams->n_hmx) {
        返回 kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    如果 (!ggml_hexagon_tensor_is_tiled(src0)) {
        返回 false;
    }

    返回 kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
}

static bool is_supported_mul_mat_id_nx_kernel(const ggml_tensor * src0, const struct htp_mm_kernel_params * kparams) {
    如果 (kparams->n_hmx) {
        返回 kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    如果 (!ggml_hexagon_tensor_is_tiled(src0)) {
        返回 false;
    }

    返回 kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK;
}

static bool is_mergeable_mul_mat(const ggml_tensor * t) {
    如果 (t->op != GGML_OP_MUL_MAT) 返回 false;
    如果 (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) 返回 false;

    const ggml_tensor * src0 = t->src[0];
    const ggml_tensor * src1 = t->src[1];
    如果 (src0->ne[2] != 1 || src0->ne[3] != 1) 返回 false;

    如果 (mm_is_hmx_eligible(t)) {
        返回 htp_is_hmx_weight_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    如果 (src1->type != GGML_TYPE_F32) 返回 false;

    返回 ggml_hexagon_tensor_is_tiled(src0);
}

static bool is_mergeable_mul_mat_pair(const ggml_tensor * n1, const ggml_tensor * n2) {
    如果 (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        返回 false;
    }
    如果 (n1->src[1] != n2->src[1]) {
        返回 false;
    }
    如果 (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        返回 false;
    }
    如果 (n1->src[0]->type != n2->src[0]->type) {
        返回 false;
    }
    如果 (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        返回 false;
    }
    如果 (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        返回 false;
    }
    返回 true;
}

static bool is_mergeable_mul_mat_id(const ggml_tensor * t) {
    如果 (t->op != GGML_OP_MUL_MAT_ID) 返回 false;

    const ggml_tensor * src0 = t->src[0];
    如果 (mm_is_hmx_eligible(t)) {
        返回 htp_is_hmx_weight_type(src0->type);
    }

    返回 ggml_hexagon_tensor_is_tiled(src0);
}

static bool is_mergeable_mul_mat_id_pair(const ggml_tensor * n1, const ggml_tensor * n2) {
    如果 (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        返回 false;
    }
    如果 (n1->src[1] != n2->src[1]) {
        返回 false;
    }
    如果 (n1->src[2] != n2->src[2]) {
        返回 false;
    }
    如果 (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        返回 false;
    }
    如果 (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        返回 false;
    }
    如果 (n1->src[0]->type != n2->src[0]->type) {
        返回 false;
    }
    如果 (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        返回 false;
    }
    如果 (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        返回 false;
    }
    返回 true;
}

static ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    如果 (sess->last_error > HTP_STATUS_OK) {
        返回 GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    const std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    bool cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
    如果 (cache_hit) {
        nodes_ptr = &sess->cached_nodes;
    } 否则 {
        // Tag fusable tensors in graph
        为 (整数 i = 0; i < graph->n_nodes; i++) {
            auto * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            如果 (!extra) continue;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            如果 (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } 否则 如果 ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                如果 ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_node_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        为 (整数 i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            如果 (!op_is_compute(n)) {
                continue;
            }

            htp_opnode node(HTP_OP_INVALID, n);
            node.opcode = op_remap_to_htp(n);
            如果 (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID || node.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = node.node->src[0];
                如果 (ggml_hexagon_needs_repack(src0)) {
                    如果 (opt_verbose) {
                        char dims_buf[64];
                        char strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, sizeof(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, sizeof(strides_buf), src0);
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

            如果 (node.opcode == HTP_OP_MUL_MAT || node.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    node.node->src[0], node.node->src[1], node.node,
                    (struct htp_mm_kernel_params *)node.kernel_params
                );
            } 否则 如果 (node.opcode == HTP_OP_MUL || node.opcode == HTP_OP_ADD || node.opcode == HTP_OP_ADD_ID || node.opcode == HTP_OP_SUB || node.opcode == HTP_OP_DIV) {
                const ggml_tensor * src1 = node.node->src[1];
                const ggml_tensor * src2 = (node.opcode == HTP_OP_ADD_ID) ? node.node->src[2] : nullptr;
                GGML_ASSERT(ggml_hexagon_precompute_binary_params(sess,
                    node.opcode, node.node->src[0], src1, node.node,
                    (struct htp_binary_kernel_params *) node.kernel_params, src2));
            } 否则 如果 (node.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    node.node,
                    (struct htp_fa_kernel_params *)node.kernel_params
                );
            } 否则 如果 (htp_op_is_unary(node.opcode)) {
                auto inputs = node.get_inputs();
                const struct ggml_tensor * src0 = inputs[0];
                const struct ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    node.opcode, src0, src1, node.dst(),
                    (struct htp_unary_kernel_params *)node.kernel_params
                );
            } 否则 如果 (node.opcode == HTP_OP_GET_ROWS) {
                ggml_hexagon_precompute_get_rows_params(sess,
                    node.node->src[0], node.node->src[1], node.dst(),
                    (struct htp_get_rows_kernel_params *)node.kernel_params
                );
            } 否则 如果 (node.opcode == HTP_OP_SET_ROWS) {
                ggml_hexagon_precompute_set_rows_params(sess,
                    node.node->src[0], node.node->src[1], node.dst(),
                    (struct htp_set_rows_kernel_params *)node.kernel_params
                );
            } 否则 如果 (node.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    node.node,
                    (struct htp_rope_kernel_params *)node.kernel_params
                );
            } 否则 如果 (node.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    node.node->src[0], node.node->src[1], node.dst(),
                    (struct htp_ssm_conv_kernel_params *)node.kernel_params
                );
            } 否则 如果 (node.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    node.node,
                    (struct htp_softmax_kernel_params *)node.kernel_params
                );
            } 否则 如果 (node.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    node.node,
                    (struct htp_gdn_kernel_params *)node.kernel_params
                );
            } 否则 如果 (node.opcode == HTP_OP_ARGSORT || node.opcode == HTP_OP_TOP_K) {
                ggml_hexagon_precompute_sort_params(sess,
                    node.node,
                    node.opcode == HTP_OP_TOP_K,
                    (struct htp_sort_kernel_params *) node.kernel_params
                );
            } 否则 如果 (node.opcode == HTP_OP_POOL_2D || node.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, node.node->src[0], node.dst(),
                    (struct htp_pool_2d_kernel_params *)node.kernel_params,
                    node.opcode == HTP_OP_POOL_1D);
            } 否则 如果 (node.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    node.node,
                    (struct htp_concat_kernel_params *) node.kernel_params
                );
            } 否则 如果 (node.opcode == HTP_OP_CPY || node.opcode == HTP_OP_CPY_FENCE) {
                const bool ok = ggml_hexagon_precompute_cpy_params(sess,
                    node.node,
                    (struct htp_copy_kernel_params *) node.kernel_params
                );
                const auto * kparams = (const struct htp_copy_kernel_params *) node.kernel_params;
                如果 (ok && node.opcode == HTP_OP_CPY && kparams->total_elems == 0) {
                    continue;
                }
            } 否则 如果 (node.opcode == HTP_OP_IM2COL) {
                ggml_hexagon_precompute_im2col_params(sess,
                    node.node,
                    (struct htp_im2col_kernel_params *) node.kernel_params
                );
            } 否则 如果 (node.opcode == HTP_OP_IM2COL_3D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    node.node,
                    (struct htp_im2col_kernel_params *) node.kernel_params
                );
            }
            computed_nodes.push_back(std::move(node));
        }

        如果 (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } 否则 {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    为 (const auto & node : *nodes_ptr) {
        sess->enqueue_op(node);
    }

    如果 (sess->last_error > HTP_STATUS_OK) {
        返回 GGML_STATUS_FAILED;
    }

    返回 GGML_STATUS_SUCCESS;
}

static void ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    如果 (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
}

enum ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

struct ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

struct ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    void reset() {
        ranges.clear();
    }

    void add(const ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    bool check(const ggml_hexagon_mem_range & mr) const {
        为 (const auto & cmp : ranges) {
            如果 (mr.pb != cmp.pb) {
                continue;
            }
            如果 (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                continue;
            }
            如果 (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                返回 false;
            }
        }
        返回 true;
    }
};

static ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(const ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    const ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    如果 (tensor->buffer) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } 否则 {
        mr = {
            /*.pb =*/ (uint64_t) base,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    返回 mr;
}

static void ggml_hexagon_mem_ranges_add_node(ggml_hexagon_mem_ranges & mrs, const htp_opnode & node) {
    如果 (node.is_empty()) 返回;

    为 (整数 i = 0; i < GGML_MAX_SRC; i++) {
        如果 (node.node->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(node.node->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    为 (const auto * fused : node.fused) {
        为 (整数 i = 0; i < GGML_MAX_SRC; i++) {
            如果 (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(node.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

static bool ggml_hexagon_mem_ranges_check_node(const ggml_hexagon_mem_ranges & mrs, const htp_opnode & node) {
    如果 (node.is_empty()) 返回 true;

    为 (整数 i = 0; i < GGML_MAX_SRC; i++) {
        如果 (node.node->src[i]) {
            如果 (!mrs.check(ggml_hexagon_mem_range_from_tensor(node.node->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                返回 false;
            }
        }
    }
    为 (const auto * fused : node.fused) {
        为 (整数 i = 0; i < GGML_MAX_SRC; i++) {
            如果 (fused->src[i]) {
                如果 (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    返回 false;
                }
            }
        }
    }
    返回 mrs.check(ggml_hexagon_mem_range_from_tensor(node.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

static std::vector<整数> ggml_hexagon_graph_optimize_reorder(const std::vector<htp_opnode> & nodes) {
    const 整数 n = nodes.size();

    std::vector<整数> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    为 (整数 i0 = 0; i0 < n; i0++) {
        如果 (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        如果 (!node0.stackable()) {
            res.push_back(i0);
            used[i0] = true;
            continue;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr 整数 N_FORWARD = 16;

        std::vector<整数> stack;
        stack.push_back(i0);

        mrs.reset();

        为 (整数 i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            如果 (used[i1]) {
                continue;
            }

            const auto & node1 = nodes[i1];

            如果 (node1.stackable() && node1.same_input(node0) && ggml_hexagon_mem_ranges_check_node(mrs, node1)) {
                stack.push_back(i1);
            } 否则 {
                ggml_hexagon_mem_ranges_add_node(mrs, node1);
            }
        }

        为 (整数 idx : stack) {
            res.push_back(idx);
            used[idx] = true;
        }
    }

    返回 res;
}

static void ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    const 整数 n = gf->n_nodes;

    constexpr 整数 MAX_FUSE = 16;

    enum ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    为 (整数 i = 0; i < n; i++) {
        htp_opnode node(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        如果 (node.op() == GGML_OP_ADD ||
            node.op() == GGML_OP_NORM ||
            node.op() == GGML_OP_RMS_NORM) {
            ops[0] = node.op();

            整数 f = i + 1;
            当 (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                如果 (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    break;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            为 (; f > 1; f--) {
                如果 (ggml_can_fuse(gf, i, ops, f)) {
                    break;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            为 (整数 k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                node.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(node));
    }

    const auto order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        整数 j = 0;
        为 (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            为 (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

static uint64_t ggml_hexagon_session_key(const ggml_hexagon_session * sess) {
    返回 ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

static bool ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    auto sbuf_dst = sess_dst->mmap_tensor(dst);

    如果 (!sess_src->clone_buffer(sbuf_dst)) { 返回 false; }

    const uint64_t src_key = ggml_hexagon_session_key(sess_src);
    auto & fence_slot = sess_dst->cpy_fence_slots[src_key];
    如果 (!fence_slot) {
        fence_slot = (volatile uint32_t *) sess_dst->alloc_fence(1);
    }

    如果 (!sess_src->clone_buffer(sess_dst->fence_buf)) { 返回 false; }

    如果 (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    static ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.buffer = &sess_dst->fence_buf->backend_buffer;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.data   = (void *) fence_slot;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = sizeof(int32_t);
    fence_tensor.nb[1]  = sizeof(int32_t);
    fence_tensor.nb[2]  = sizeof(int32_t);
    fence_tensor.nb[3]  = sizeof(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ true);

    sess_dst->add_peer(sess_src);

    返回 true;
}

static bool ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    auto sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    如果 (!sess_dst->clone_buffer(sbuf_src)) { 返回 false; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    返回 true;
}

static bool ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    如果 (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        返回 false;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    如果 (!dst->extra) {
        ggml_backend_buffer_init_tensor(dst->buffer, dst);
    }

    auto * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    const auto * src_extra = static_cast<const ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    如果 (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        返回 true;
    }

    如果 (sess_src->phys_idx != sess_dst->phys_idx)
        返回 ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    返回 ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

static ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = new ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_slot = (volatile uint32_t *) sess->alloc_fence(1);

    static ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.buffer = &sess->fence_buf->backend_buffer;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.data   = (void *) hex_event->fence_slot;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = sizeof(int32_t);
    hex_event->fence_tensor.nb[1]  = sizeof(int32_t);
    hex_event->fence_tensor.nb[2]  = sizeof(int32_t);
    hex_event->fence_tensor.nb[3]  = sizeof(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (void *)hex_event, (void *)hex_event->fence_slot);

    返回 new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

static void ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    如果 (hex_event->seq == 0) {
        返回;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (void *)hex_event, hex_event->seq, (void *)hex_event->fence_slot);

    auto * fence = reinterpret_cast<const volatile std::atomic<uint32_t> *>(hex_event->fence_slot);

    如果 ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    当 (true) {
        如果 ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            如果 (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            break;
        }
        std::this_thread::yield();
    }
}

static void ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (void *)hex_event);
    hex_event->fence_sess->free_fence((void *) hex_event->fence_slot, 1);
    delete hex_event;
    delete event;
}

static void ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

static void ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    如果 (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ false);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (void *)hex_event, hex_event->seq, (void *)hex_event->fence_slot);
}

static void ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    如果 (hex_event->seq == 0) {
        返回;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (void *)hex_event, hex_event->seq, (void *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    如果 (sess->phys_idx == hex_event->sess->phys_idx) {
        如果 (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        返回;
    }

    sess->clone_buffer(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ true);
}

static void ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (整数) tensor->buffer->usage : -1);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

static void ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (整数) tensor->buffer->usage : -1);
    sess->flush_sync();
    如果 (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

static void ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     struct ggml_tensor * tensor,
                                                     const void * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (整数) tensor->buffer->usage : -1);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

static void ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     const struct ggml_tensor * tensor,
                                                     void * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (整数) tensor->buffer->usage : -1);
    sess->flush_sync();
    如果 (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

static struct ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_record,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

static ggml_guid_t ggml_backend_hexagon_guid() {
    static ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    返回 &guid;
}

bool ggml_backend_is_hexagon(ggml_backend_t backend) {
    返回 backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

static ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, const char * params) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    返回 new ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

static const char * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    返回 dev_ctx->c_name();

    GGML_UNUSED(dev);
}

static const char * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    返回 "Hexagon";
    GGML_UNUSED(dev);
}

static void ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = opt_mbuf_total;
    *total = opt_mbuf_total;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    返回 GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    返回 &dev_ctx->buffer_type;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    如果 (!opt_hostbuf) {
        返回 NULL;
    }
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    返回 &dev_ctx->host_buffer_type;
}

static bool ggml_hexagon_supported_cpy(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    struct htp_copy_kernel_params kparams;
    返回 ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

static bool ggml_hexagon_supported_cont(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    struct htp_copy_kernel_params kparams;
    返回 ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

static bool ggml_hexagon_supported_repeat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    // Support f32 and f16
    如果 (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) 返回 false;

    // src and dst must be the same type
    如果 (src0->type != dst->type) 返回 false;

    // dst dims must be multiples of src dims
    如果 (dst->ne[0] % src0->ne[0] != 0) 返回 false;
    如果 (dst->ne[1] % src0->ne[1] != 0) 返回 false;
    如果 (dst->ne[2] % src0->ne[2] != 0) 返回 false;
    如果 (dst->ne[3] % src0->ne[3] != 0) 返回 false;

    // require contiguous tensors (no transposition)
    如果 (ggml_is_transposed(src0) || ggml_is_transposed(dst)) 返回 false;

    返回 true;
}

static bool ggml_hexagon_supported_concat(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    struct htp_concat_kernel_params kparams;
    返回 ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

static bool ggml_hexagon_supported_fill(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    const struct ggml_tensor * dst = op;

    如果 (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        返回 false;
    }

    返回 true;
    GGML_UNUSED(sess);
}

static bool ggml_hexagon_supported_roll(const struct ggml_hexagon_session * sess, const struct ggml_tensor * op) {
    GGML_UNUSED(sess);

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * dst  = op;

    如果 (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        返回 false;
    }

    如果 (!ggml_are_same_shape(src0, dst)) {
        返回 false;
    }

    如果 (src0->nb[0] != ggml_type_size(src0->type) || dst->nb[0] != ggml_type_size(dst->type)) {
        返回 false;
    }

    如果 (!ggml_is_contiguous(dst)) {
        返回 false;
    }

    返回 true;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
static bool ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    // reject ops that match the filter
    如果 (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        返回 false;
    }

    bool supp = false;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            supp = true;
            break;

        case GGML_OP_MUL:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
            supp = ggml_hexagon_supported_binary(sess, op);
            break;

        case GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            break;

        case GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            break;

        case GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            break;

        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            break;

        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            break;

        case GGML_OP_SUM:
            supp = ggml_hexagon_supported_sum(sess, op);
            break;

        case GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            break;

        case GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            break;

        case GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            break;

        case GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            break;

        case GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            break;

        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_STEP:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    break;
                default:
                    supp = false;
                    break;
            }
            break;

        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                case GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    break;
                default:
                    supp = false;
                    break;
            }
            break;

        case GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            break;

        case GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            break;

        case GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            break;

        case GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            break;

        case GGML_OP_DUP:
        case GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            break;

        case GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            break;

        case GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            break;

        case GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            break;

        case GGML_OP_TOP_K:
            supp = ggml_hexagon_supported_top_k(sess, op);
            break;

        case GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            break;

        case GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            break;

        case GGML_OP_IM2COL_3D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            break;

        case GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            break;

        case GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            break;

        case GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            break;

        case GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            break;

        case GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            break;

        case GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            break;

        case GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            break;

        case GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            break;

        case GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            break;

        default:
            break;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    返回 supp;
}

static bool ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    bool supp = (buft == &dev_ctx->host_buffer_type) || (buft == &dev_ctx->buffer_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    返回 supp;
}

static const struct ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ NULL,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    为 (size_t i = 0; i < opt_ndev; i++) {
        const auto & cfg = opt_device_configs[i];
        如果 (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } 否则 {
            std::string peers_str;
            为 (const auto & p : cfg.mdev_group) {
                如果 (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = new ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    为 (size_t i = 0; i < opt_ndev; i++) {
        auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        delete dev_ctx;
    }
}

static const char * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    返回 "HTP";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_hexagon_reg_get_device_count(ggml_backend_reg_t reg) {
    返回 opt_ndev;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    如果 (index >= opt_ndev || !hreg->devices[index].context) {
        返回 nullptr;
    }

    返回 &hreg->devices[index];
}

// ** communication context for tensor-split allreduce

static void * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    如果 (n_backends < 2 || n_backends > 4) {
        返回 nullptr;
    }

    为 (size_t i = 0; i < n_backends; ++i) {
        如果 (!ggml_backend_is_hexagon(backends[i])) {
            返回 nullptr;
        }
    }

    为 (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        为 (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            如果 (sess_i->phys_idx == sess_j->phys_idx) {
                返回 nullptr;
            }
        }
    }

    auto * ctx = new ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    static ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    为 (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (volatile uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].buffer = &sess_i->fence_buf->backend_buffer;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].data   = (void *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = sizeof(int32_t);
        ctx->fence_tensors[i].nb[1]  = sizeof(int32_t);
        ctx->fence_tensors[i].nb[2]  = sizeof(int32_t);
        ctx->fence_tensors[i].nb[3]  = sizeof(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    返回 ctx;
}

static void ggml_backend_hexagon_comm_free(void * comm_ctx_v) {
    如果 (!comm_ctx_v) 返回;
    auto * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    为 (size_t i = 0; i < ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((void *) ctx->fence_slots[i], 1);
    }
    delete ctx;
}

static bool ggml_backend_hexagon_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    如果 (opt_ar_select == 0 || !comm_ctx_v) 返回 false;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    const size_t n_backends = comm_ctx->n_backends;

    如果 (n_backends < 2 || n_backends > 4) 返回 false;

    为 (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        为 (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            如果 (sess_i->phys_idx == sess_j->phys_idx) {
                返回 false;
            }
        }
    }

    为 (size_t i = 0; i < n_backends; i++) {
        如果 (!tensors[i] || !tensors[i]->buffer || !ggml_backend_buffer_is_hexagon(tensors[i]->buffer)) {
            返回 false;
        }
        如果 (tensors[i]->type != tensors[0]->type) {
            返回 false;
        }
        如果 (!ggml_is_contiguous(tensors[i])) {
            返回 false;
        }
        如果 (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            返回 false;
        }
    }

    如果 (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        返回 false;
    }

    为 (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        struct htp_allreduce_kernel_params kparams;
        如果 (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, false, false, /*is_shard_ok=*/ false, &kparams)) {
            返回 false;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    为 (size_t i = 1; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        如果 ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    如果 (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_entry = max_seq;
    如果 (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    为 (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<const ggml_tensor *> data_tensors(n_backends);
    std::vector<const ggml_tensor *> sync_tensors(n_backends);
    为 (size_t i = 0; i < n_backends; i++) {
        data_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    为 (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], data_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        为 (size_t j = 0; j < n_backends; j++) {
            如果 (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    返回 true;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_split_buffer_type(整数 main_device, const float * tensor_split) {
    GGML_UNUSED(tensor_split);
    auto reg = ggml_backend_hexagon_reg();
    auto dev = ggml_backend_reg_dev_get(reg, main_device);
    如果 (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    如果 (!dev) 返回 nullptr;
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    返回 &dev_ctx->buffer_type;
}

static void * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    如果 (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        返回 (void *) ggml_backend_hexagon_split_buffer_type;
    }
    如果 (strcmp(name, "ggml_backend_comm_init") == 0) {
        返回 (void *) ggml_backend_hexagon_comm_init;
    }
    如果 (strcmp(name, "ggml_backend_comm_free") == 0) {
        返回 (void *) ggml_backend_hexagon_comm_free;
    }
    如果 (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        返回 (void *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    返回 NULL;
}

template<typename T> std::vector<T> str_to_vec(const char* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    当 (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    返回 v;
}

template<typename T, 整数 BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    为 (auto i : v) { ss << i << ','; }
    auto str = ss.str(); str.pop_back(); // drop last comma
    返回 str;
}

static void ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, bool discovery_supported, const std::unordered_map<整数, fastrpc_domain> & cdsp_map) {
    如果 (discovery_supported) {
        auto it = cdsp_map.find(cfg.physical_idx);
        如果 (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } 否则 {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } 否则 {
        switch (cfg.physical_idx) {
            case 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                break;
            case 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                break;
            default:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                break;
        }
    }
    为 (auto & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
static void ggml_hexagon_discover_devices() {
    std::unordered_map<整数, fastrpc_domain> cdsp_map;
    bool discovery_supported = false;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    整数 err = remote_system_request(&domain_info);
    如果 (err == AEE_SUCCESS && domain_info.sys.num_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.num_domains);
        domain_info.sys.domains     = domains.data();
        domain_info.sys.max_domains = (整数) domains.size();

        err = remote_system_request(&domain_info);
        如果 (err == AEE_SUCCESS) {
            discovery_supported = true;
            const 整数 n_domains = std::min(domain_info.sys.num_domains, (整数) domains.size());
            为 (整数 i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (整数) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                如果 (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (整数) domains[i].type);
                    continue;
                }
                如果 (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    continue;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } 否则 {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (unsigned) err);
        }
    } 否则 如果 (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (unsigned) err);
    }

    // Populate domain IDs and names for all configured devices
    为 (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

static void ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned 整数) HTP_TYPE_Q4_0 == (unsigned 整数) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned 整数) HTP_TYPE_Q4_1 == (unsigned 整数) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned 整数) HTP_TYPE_Q8_0 == (unsigned 整数) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned 整数) HTP_TYPE_MXFP4 == (unsigned 整数) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned 整数) HTP_TYPE_IQ4_NL == (unsigned 整数) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned 整数) HTP_TYPE_Q4_K == (unsigned 整数) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned 整数) HTP_TYPE_Q5_K == (unsigned 整数) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned 整数) HTP_TYPE_Q6_K == (unsigned 整数) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned 整数) HTP_TYPE_Q3_K == (unsigned 整数) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned 整数) HTP_TYPE_Q2_K == (unsigned 整数) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned 整数) HTP_TYPE_Q1_0 == (unsigned 整数) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned 整数) HTP_TYPE_Q2_0 == (unsigned 整数) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned 整数) HTP_TYPE_BF16 == (unsigned 整数) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    const char * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    const char * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    const char * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    const char * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    const char * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    const char * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    const char * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    const char * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    const char * str_etm      = getenv("GGML_HEXAGON_ETM");
    const char * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    const char * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    const char * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    const char * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    const char * str_fa_head_split = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    const char * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    const char * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    const char * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    const char * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    const char * str_arch     = getenv("GGML_HEXAGON_ARCH");
    const char * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    const char * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    const char * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    const char * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    const char * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    const char * str_power_mode = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    如果 (!str_arch) {
        整数 err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        如果 (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } 否则 {
            如果 (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } 否则 如果 (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } 否则 {
        如果 (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, NULL, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? new std::regex(str_opfilter, RE_ICASE) : NULL;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, NULL, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, NULL, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, NULL, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  NULL, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    如果 (str_power_mode) {
        整数 m = htp_power_mode_from_str(str_power_mode);
        如果 (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_mode);
        } 否则 {
            opt_power_mode = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, NULL, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_head_split = str_fa_head_split ? atoi(str_fa_head_split)       : opt_fa_head_split;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    如果 (str_mbuf) {
        const char * p = str_mbuf;
        为 (整数 idx = 0; idx < 3 && p && *p; idx++) {
            当 (*p == ' ') p++;
            const char * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : strlen(p);
            当 (len > 0 && p[len - 1] == ' ') len--;
            如果 (len > 0) {
                std::string token(p, len);
                如果 (idx == 0) opt_mbuf_dyn    = parse_size(token.c_str());
                如果 (idx == 1) opt_mbuf_static = parse_size(token.c_str());
                如果 (idx == 2) opt_mbuf_total  = parse_size(token.c_str());
            }
            如果 (!comma) break;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_size(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    const char * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    如果 (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    如果 (str_devices && str_devices[0] != '\0') {
        bool is_single_number = true;
        为 (整数 i = 0; str_devices[i] != '\0'; i++) {
            如果 (!isdigit((unsigned char)str_devices[i])) {
                is_single_number = false;
                break;
            }
        }
        如果 (is_single_number) {
            整数 n = atoi(str_devices);
            如果 (n < 1) n = 1;
            如果 (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            为 (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (整数)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.clear();
            }
        } 否则 {
            std::string s_devices(str_devices);
            std::vector<std::string> items;
            std::string curr_item;
            整数 bracket_depth = 0;
            为 (char ch : s_devices) {
                如果 (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } 否则 如果 (ch == ']') {
                    如果 (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } 否则 如果 (ch == ',' && bracket_depth == 0) {
                    size_t s = curr_item.find_first_not_of(" \t\r\n");
                    size_t e = curr_item.find_last_not_of(" \t\r\n");
                    如果 (s != std::string::npos) {
                        items.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.clear();
                } 否则 {
                    curr_item += ch;
                }
            }
            size_t s = curr_item.find_first_not_of(" \t\r\n");
            size_t e = curr_item.find_last_not_of(" \t\r\n");
            如果 (s != std::string::npos) {
                items.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            为 (const auto & item : items) {
                size_t b_open  = item.find('[');
                size_t b_close = item.rfind(']');

                如果 (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    整数 virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    如果 (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        try {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } catch (...) {
                            virt = 0;
                        }
                    } 否则 {
                        size_t dev_colon = dev_name.find(':');
                        如果 (dev_colon != std::string::npos) {
                            try {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } catch (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<整数> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    当 (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        如果 (ps == std::string::npos) continue;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        如果 (dash_pos != std::string::npos) {
                            try {
                                整数 p_start = std::stoi(p_part.substr(0, dash_pos));
                                整数 p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                为 (整数 p = p_start; p <= p_end; p++) {
                                    如果 (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } catch (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } 否则 {
                            try {
                                整数 p = std::stoi(p_part);
                                如果 (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } catch (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    如果 (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    如果 (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        auto & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.clear();

                        为 (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_group.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } 否则 {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } 否则 如果 (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    size_t colon_pos = rest.find(':');
                    整数 phys = 0;
                    整数 virt = 0;
                    try {
                        如果 (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } 否则 {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } catch (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        continue;
                    }

                    如果 (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_group.clear();
                        opt_ndev++;
                    } 否则 {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } 否则 {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", item.c_str());
                }
            }
        }
    } 否则 {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_group.clear();
    }

#如果 defined(__ANDROID__)
    如果 (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    如果 (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            auto v  = str_to_vec<uint32_t>(str_profile);
            switch (v.size()) {
                case 1:  opt_profile = v[0]; 返回 opt_pmu_evt; // mode with default pmu events
                case 8:  opt_profile = 2;    返回 v;           // mode with custom  pmu events
                default: opt_profile = 0;    返回 {};          // garbage input
            }}();
        如果 (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = new ggml_hexagon_registry(reg);
}

static const struct ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(void) {
    static bool initialized = false;

    static ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ NULL };

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        如果 (!initialized) {
            auto nErr = htpdrv_init();
            如果 (nErr != AEE_SUCCESS) {
                返回 NULL;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = true;
    }

    返回 &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
