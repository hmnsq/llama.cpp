#define _USE_MATH_DEFINES

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <inttypes.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <string.h>
#અન્તર્ભાવ <time.h>

#અન્તર્ભાવ <atomic>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <chrono>
#અન્તર્ભાવ <mutex>
#અન્તર્ભાવ <thread>
#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <stdexcept>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <sstream>
#અન્તર્ભાવ <iomanip>
#અન્તર્ભાવ <unordered_set>
#અન્તર્ભાવ <unordered_map>
#અન્તર્ભાવ <regex>
#અન્તર્ભાવ <queue>
#અન્તર્ભાવ <deque>
#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <initializer_list>

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
#નહીંતર
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"

#અન્તર્ભાવ <AEEStdErr.h>
#અન્તર્ભાવ <dspqueue.h>
#અન્તર્ભાવ <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-alloc.h"
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "ggml-hexagon.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-quants.h"
#અન્તર્ભાવ "htp-opnode.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "htp/matmul-ops.h"
#અન્તર્ભાવ "htp/binary-ops.h"
#અન્તર્ભાવ "htp/flash-attn-ops.h"
#અન્તર્ભાવ "htp/unary-ops.h"
#અન્તર્ભાવ "htp/get-rows-ops.h"
#અન્તર્ભાવ "htp/set-rows-ops.h"
#અન્તર્ભાવ "htp/softmax-ops.h"
#અન્તર્ભાવ "htp/pool-ops.h"
#અન્તર્ભાવ "htp/rope-ops.h"
#અન્તર્ભાવ "htp/ssm-conv.h"
#અન્તર્ભાવ "htp/gated-delta-net-ops.h"
#અન્તર્ભાવ "htp/argsort-ops.h"
#અન્તર્ભાવ "htp/concat-ops.h"
#અન્તર્ભાવ "htp/cpy-ops.h"
#અન્તર્ભાવ "htp/im2col-ops.h"
#અન્તર્ભાવ "htp_iface.h"
#અન્તર્ભાવ "htp-drv.h"
#અન્તર્ભાવ "htp-repack.h"

using intvec  = std::vector<પૂર્ણાંક>;
using uintvec = std::vector<unsigned પૂર્ણાંક>;
using u32vec  = std::vector<uint32_t>;

#define GGML_HEXAGON_MAX_SESSIONS          16

#define GGML_HEXAGON_FENCE_SLOT_SIZE       128

સંરચના ggml_hexagon_device_config {
    પૂર્ણાંક         physical_idx = 0;
    પૂર્ણાંક         virtual_idx  = 0;
    પૂર્ણાંક         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

સ્થિર ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

સ્થિર પૂર્ણાંક    opt_arch    = 0; // autodetect
સ્થિર size_t opt_ndev    = 1;
સ્થિર size_t opt_nhvx    = 0; // use all
સ્થિર પૂર્ણાંક    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
સ્થિર size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
સ્થિર પૂર્ણાંક    opt_etm     = 0;
સ્થિર પૂર્ણાંક    opt_verbose = 0;
સ્થિર પૂર્ણાંક    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
સ્થિર તાર્કિક   opt_hostbuf = ખોટું;
સ્થિર તાર્કિક   opt_dma64   = ખોટું;
સ્થિર પૂર્ણાંક    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start

સ્થિર અચલ std::pair<અચલ અક્ષર *, પૂર્ણાંક> htp_power_mode_names[] = {
    { "low_power_saver",            HTP_POWER_MODE_નીચલું_POWER_SAVER },
    { "power_saver",                HTP_POWER_MODE_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_MODE_ઉપલું_POWER_SAVER },
    { "low_balanced",               HTP_POWER_MODE_નીચલું_BALANCED },
    { "balanced",                   HTP_POWER_MODE_BALANCED },
    { "high_performance",           HTP_POWER_MODE_ઉપલું_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_MODE_SUSTAINED_ઉપલું_PERFORMANCE },
    { "burst",                      HTP_POWER_MODE_BURST },
    { "default",                    HTP_POWER_MODE_BURST }, // flip this to change the default mode
};

સ્થિર પૂર્ણાંક htp_power_mode_from_str(અચલ અક્ષર * s) {
    માટે (auto & e : htp_power_mode_names) {
#જો defined(_MSC_VER)
        જો (_stricmp(s, e.first) == 0) {
#નહીંતર
        જો (strcasecmp(s, e.first) == 0) {
#endif
            પાછા e.second;
        }
    }
    પાછા -1;
}

સ્થિર અચલ અક્ષર * htp_power_mode_to_str(પૂર્ણાંક mode) {
    માટે (auto & e : htp_power_mode_names) {
        જો (e.second == mode) {
            પાછા e.first;
        }
    }
    પાછા "unknown";
}

સ્થિર size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
સ્થિર size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
સ્થિર size_t opt_mbuf_total  = 0;                        // total buffer space limit (0 = unconstrained)

સ્થિર પૂર્ણાંક    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
સ્થિર પૂર્ણાંક    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
સ્થિર પૂર્ણાંક    opt_fa_head_split = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
સ્થિર પૂર્ણાંક    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
સ્થિર પૂર્ણાંક    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
સ્થિર પૂર્ણાંક    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
સ્થિર u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

સ્થિર પૂર્ણાંક opt_opbatch  = 1280; // max number of ops in a batch
સ્થિર પૂર્ણાંક opt_opqueue  = 8;   // max number of pending batches
સ્થિર પૂર્ણાંક opt_optrace  = 0;    // trace buffer size per thread (0 means default)
સ્થિર પૂર્ણાંક opt_oppoll   = 0;    // polling for batch completions
સ્થિર પૂર્ણાંક opt_opfusion  = 1;    // enable/disable op fusion
સ્થિર પૂર્ણાંક opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

ગણના ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ADD = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ADD   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

સ્થિર અંતઃરેખ તાર્કિક ggml_hexagon_is_fusion_enabled(પૂર્ણાંક flag) {
    જો (opt_opfusion <= 0) પાછા ખોટું;
    જો (opt_opfusion == 1) પાછા સાચું; // 1 enables all
    પાછા (opt_opfusion & flag) != 0;
}

સ્થિર std::regex* opt_opfilter = ખાલી; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    જો (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

સ્થિર અચલ અક્ષર * status_to_str(uint32_t status) {
    પસંદગી (status) {
        સ્થિતિ HTP_STATUS_OK:
            પાછા "OK";
        સ્થિતિ HTP_STATUS_NO_SUPPORT:
            પાછા "NO-SUPPORT";
        સ્થિતિ HTP_STATUS_INVAL_PARAMS:
            પાછા "INVAL-PARAMS";
        સ્થિતિ HTP_STATUS_VTCM_TOO_SMALL:
            પાછા "VTCM-TOO-SMALL";
        સ્થિતિ HTP_STATUS_INTERNAL_ERR:
            પાછા "INTERNAL-ERROR";
        મૂળભૂત:
            પાછા "UNKNOWN";
    }
}

// ** debug helpers

સ્થિર શૂન્ય ggml_hexagon_dump_op_exec(અચલ std::string &sess_name, અચલ htp_opnode & ગાંઠ, અચલ uint32_t req_flags) {
    જો (!opt_verbose) પાછા;

    htp_opformat fmt(ગાંઠ);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                ગાંઠ.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

સ્થિર શૂન્ય ggml_hexagon_dump_op_supp(અચલ std::string &sess_name, અચલ સંરચના ggml_tensor * op, તાર્કિક supp) {
    જો (!opt_verbose) પાછા;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

સ્થિર અચલ અક્ષર * htp_event_name(uint16_t id) {
    પસંદગી (id) {
        સ્થિતિ HTP_TRACE_EVT_DMA:            પાછા "DMA";
        સ્થિતિ HTP_TRACE_EVT_HVX_COMP:       પાછા "HVX_COMP";
        સ્થિતિ HTP_TRACE_EVT_HVX_A_QUANT:    પાછા "HVX_A_QUANT";
        સ્થિતિ HTP_TRACE_EVT_HVX_A_PREP:     પાછા "HVX_A_PREP";
        સ્થિતિ HTP_TRACE_EVT_HVX_W_DEQUANT:  પાછા "HVX_W_DEQUANT";
        સ્થિતિ HTP_TRACE_EVT_HVX_W_PREP:     પાછા "HVX_W_PREP";
        સ્થિતિ HTP_TRACE_EVT_HVX_O_PROC:     પાછા "HVX_O_PROC";
        સ્થિતિ HTP_TRACE_EVT_HVX_FA_QK:      પાછા "HVX_QK_FA";
        સ્થિતિ HTP_TRACE_EVT_HVX_FA_SFM:     પાછા "HVX_SFM_FA";
        સ્થિતિ HTP_TRACE_EVT_HVX_FA_Q_PREP:  પાછા "HVX_Q_PREP";
        સ્થિતિ HTP_TRACE_EVT_HVX_FA_K_PREP:  પાછા "HVX_K_PREP";
        સ્થિતિ HTP_TRACE_EVT_HVX_FA_V_PREP:  પાછા "HVX_V_PREP";
        સ્થિતિ HTP_TRACE_EVT_HVX_GDN_PREP:   પાછા "HVX_GDN_PREP";
        સ્થિતિ HTP_TRACE_EVT_HVX_GDN_SOLVE:  પાછા "HVX_GDN_SOLVE";
        સ્થિતિ HTP_TRACE_EVT_HVX_GDN_V_PREP: પાછા "HVX_GDN_V_PREP";
        સ્થિતિ HTP_TRACE_EVT_HVX_GDN_D_PREP: પાછા "HVX_GDN_D_PREP";
        સ્થિતિ HTP_TRACE_EVT_HVX_GDN_OUT:    પાછા "HVX_GDN_OUT";
        સ્થિતિ HTP_TRACE_EVT_HVX_GDN_STATE:  પાછા "HVX_GDN_STATE";
        સ્થિતિ HTP_TRACE_EVT_HVX_GDN_REM:    પાછા "HVX_GDN_REM";
        સ્થિતિ HTP_TRACE_EVT_HMX_COMP:       પાછા "HMX_COMP";
        સ્થિતિ HTP_TRACE_EVT_L2FLUSH:        પાછા "L2FLUSH";
        સ્થિતિ HTP_TRACE_EVT_INIT:           પાછા "INIT";
        સ્થિતિ HTP_TRACE_EVT_BUFF:           પાછા "BUFF";
        સ્થિતિ HTP_TRACE_EVT_FENCE:          પાછા "FENCE";
        મૂળભૂત:                           પાછા "UNKNOWN";
    }
}

સ્થિર શૂન્ય ggml_hexagon_dump_op_prof(અચલ std::string &sess_name, અચલ htp_opnode & ગાંઠ, અચલ htp_prof_desc & pd) {
    જો (!opt_profile) પાછા;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    અચલ uint32_t * pmu = pd.pmu;

    અક્ષર pmu_str[256] = "";
    જો (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, કદ(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(ગાંઠ);
    દશાંશ mhz = op_usec > 0 ? (દશાંશ) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            ગાંઠ.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

સ્થિર શૂન્ય ggml_hexagon_dump_batch_prof(અચલ std::string & sess_name, અચલ htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    દશાંશ batch_mhz = rsp.usecs > 0 ? (દશાંશ) batch_cycles / rsp.usecs : 0.0f;

    અક્ષર evt_str[256] = "----";
    જો (opt_profile == 3) {
        snprintf(evt_str, કદ(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (unsigned long long) batch_cycles, (unsigned long long) rsp.cycles_start, batch_mhz);
}

સ્થિર શૂન્ય ggml_hexagon_dump_trace_events(અચલ std::string & sess_name, અચલ htp_opbatch_rsp & rsp,
                                           અચલ htp_trace_desc * trace_events, uint32_t n_traces) {
    જો (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        માટે (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        માટે (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            માટે (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                અચલ auto & e = trace_events[t * n_traces + idx];
                તાર્કિક is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

ગણના ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_WEIGHT    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

સંરચના ggml_hexagon_session;

સ્થિર શૂન્ય ggml_hexagon_precompute_matmul_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_mm_kernel_params * kparams
);

સ્થિર શૂન્ય ggml_hexagon_precompute_fused_matmul_add_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * src2,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_mm_kernel_params * kparams
);

સ્થિર શૂન્ય ggml_hexagon_precompute_unary_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    uint32_t op,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_unary_kernel_params * kparams
);

સ્થિર તાર્કિક ggml_hexagon_precompute_દ્વિઅંકી_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    uint32_t op,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_દ્વિઅંકી_kernel_params * kparams,
    અચલ સંરચના ggml_tensor * src2 = nullptr
);

સ્થિર શૂન્ય ggml_hexagon_precompute_get_rows_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_get_rows_kernel_params * kparams
);

સ્થિર શૂન્ય ggml_hexagon_precompute_set_rows_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_set_rows_kernel_params * kparams
);

સ્થિર શૂન્ય ggml_hexagon_precompute_softmax_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_softmax_kernel_params * kparams
);

સ્થિર શૂન્ય ggml_hexagon_precompute_rope_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_rope_kernel_params * kparams
);

સ્થિર શૂન્ય ggml_hexagon_precompute_ssm_conv_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_ssm_conv_kernel_params * kparams
);

સ્થિર શૂન્ય ggml_hexagon_precompute_gated_delta_net_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_gdn_kernel_params * kparams
);

સ્થિર શૂન્ય ggml_hexagon_precompute_sort_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    તાર્કિક is_ટોચ_k,
    સંરચના htp_sort_kernel_params * kparams
);

સ્થિર શૂન્ય ggml_hexagon_precompute_pool_2d_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_pool_2d_kernel_params * kparams,
    તાર્કિક is_pool_1d
);

સ્થિર તાર્કિક ggml_hexagon_precompute_concat_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_concat_kernel_params * kparams
);

સ્થિર તાર્કિક ggml_hexagon_precompute_cpy_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_copy_kernel_params * kparams
);

સ્થિર તાર્કિક ggml_hexagon_precompute_im2col_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_im2col_kernel_params * kparams
);

સ્થિર તાર્કિક ggml_hexagon_precompute_im2col_3d_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_im2col_kernel_params * kparams
);

સ્થિર શૂન્ય ggml_hexagon_precompute_fused_mmnx_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * act,
    int32_t n_weights,
    સંરચના htp_mm_kernel_params * kparams
);

સ્થિર શૂન્ય ggml_hexagon_precompute_fused_mmidnx_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * act,
    અચલ સંરચના ggml_tensor * dst,
    int32_t n_weights,
    સંરચના htp_mm_kernel_params * kparams
);

સ્થિર તાર્કિક ggml_hexagon_precompute_allreduce_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    તાર્કિક has_add,
    તાર્કિક is_row_bcast,
    તાર્કિક is_shard_ok,
    સંરચના htp_allreduce_kernel_params * kparams
);

સ્થિર તાર્કિક ggml_hexagon_rows_stride(અચલ int64_t * ne, અચલ size_t * nb, size_t * stride);
સ્થિર તાર્કિક ggml_hexagon_matmul_can_collapse(અચલ સંરચના ggml_tensor * src0, અચલ સંરચના ggml_tensor * src1, અચલ સંરચના ggml_tensor * dst);
સ્થિર ggml_tensor ggml_hexagon_tensor_collapse_rows(અચલ સંરચના ggml_tensor * t);

સ્થિર તાર્કિક mm_is_hmx_eligible(અચલ ggml_tensor * t);
સ્થિર htp_op_code op_remap_to_htp(અચલ ggml_tensor * t);
સ્થિર તાર્કિક is_supported_mul_mat_nx_kernel(અચલ ggml_tensor * src0, અચલ સંરચના htp_mm_kernel_params * kparams);
સ્થિર તાર્કિક is_supported_mul_mat_id_nx_kernel(અચલ ggml_tensor * src0, અચલ સંરચના htp_mm_kernel_params * kparams);
સ્થિર તાર્કિક is_mergeable_mul_mat(અચલ ggml_tensor * t);
સ્થિર તાર્કિક is_mergeable_mul_mat_pair(અચલ ggml_tensor * n1, અચલ ggml_tensor * n2);
સ્થિર તાર્કિક is_mergeable_mul_mat_id(અચલ ggml_tensor * t);
સ્થિર તાર્કિક is_mergeable_mul_mat_id_pair(અચલ ggml_tensor * n1, અચલ ggml_tensor * n2);

// ** backend sessions

સંરચના ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_size { 0 };
    uint32_t             flags { 0 };
};

સ્થિર અંતઃરેખ તાર્કિક ggml_hexagon_tensor_is_fuseable(અચલ સંરચના ggml_tensor * t) {
    જો (!t->extra) પાછા ખોટું;
    auto extra = (અચલ સંરચના ggml_hexagon_tensor_extra *) t->extra;
    પાછા (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

સ્થિર અંતઃરેખ તાર્કિક ggml_hexagon_can_rt_repack(અચલ સંરચના ggml_tensor * t) {
    જો (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
        પાછા ખોટું;
    }
    જો (t->buffer && ggml_backend_buft_is_host(t->buffer->buft)) {
        પાછા ખોટું;
    }
    પાછા opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

સ્થિર અંતઃરેખ તાર્કિક ggml_hexagon_tensor_is_tiled(અચલ સંરચના ggml_tensor * t) {
    અચલ auto * extra = (અચલ ggml_hexagon_tensor_extra *) t->extra;
    જો (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        પાછા સાચું;
    }
    પાછા ggml_hexagon_can_rt_repack(t);
}

સ્થિર અંતઃરેખ તાર્કિક ggml_hexagon_needs_repack(અચલ સંરચના ggml_tensor * t) {
    અચલ auto * extra = (અચલ ggml_hexagon_tensor_extra *) t->extra;
    જો (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        પાછા ખોટું;
    }
    પાછા ggml_hexagon_can_rt_repack(t);
}

સ્થિર અંતઃરેખ તાર્કિક ggml_hexagon_tensors_overlap(અચલ સંરચના ggml_tensor * a, અચલ સંરચના ggml_tensor * b) {
    અચલ uintptr_t a0 = (uintptr_t) a->data;
    અચલ uintptr_t b0 = (uintptr_t) b->data;
    અચલ uintptr_t a1 = a0 + ggml_nbytes(a);
    અચલ uintptr_t b1 = b0 + ggml_nbytes(b);

    પાછા a0 < b1 && b0 < a1;
}

સ્થિર અંતઃરેખ તાર્કિક ggml_hexagon_can_row_partition(અચલ સંરચના ggml_tensor * t) {
    જો (t->ne[1] > 1 && (t->nb[1] & 127) != 0) પાછા ખોટું;
    જો (t->ne[2] > 1 && (t->nb[2] & 127) != 0) પાછા ખોટું;
    જો (t->ne[3] > 1 && (t->nb[3] & 127) != 0) પાછા ખોટું;
    પાછા સાચું;
}

સંરચના htp_opnode;

સંરચના ggml_hexagon_opbatch;
સંરચના ggml_hexagon_opqueue;
સંરચના ggml_hexagon_shared_buffer;
સંરચના ggml_hexagon_fence_buffer;
સંરચના ggml_hexagon_session;
સંરચના ggml_backend_hexagon_device_context;

સંરચના ggml_hexagon_mdev_group {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

સંરચના ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    અસ્થિર uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

સંરચના ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    અસ્થિર uint32_t *    fence_slot   = nullptr;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

સંરચના ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    પૂર્ણાંક              phys_idx;
    પૂર્ણાંક              virt_idx;
    તાર્કિક             valid_session;
    તાર્કિક             valid_handle;
    તાર્કિક             valid_queue;
    તાર્કિક             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<પૂર્ણાંક, std::unique_ptr<ggml_hexagon_shared_buffer>> cloned_buffers;
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

    ggml_hexagon_session(અચલ ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_ગણતરી = 0) noexcept(ખોટું);
    ~ggml_hexagon_session() noexcept(સાચું);

    અચલ અક્ષર* c_name() અચલ { પાછા name.c_str(); }

    શૂન્ય allocate(અચલ ggml_hexagon_device_config & config) noexcept(ખોટું);
    શૂન્ય release() noexcept(સાચું);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    શૂન્ય      free_fence(શૂન્ય * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, અસ્થિર uint32_t *> cpy_fence_slots;

    શૂન્ય enqueue_mdev_group();
    શૂન્ય enqueue_op(અચલ htp_opnode & ગાંઠ);
    શૂન્ય enqueue_cpy(અચલ ggml_tensor * src, ggml_tensor * dst, અચલ ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    શૂન્ય enqueue_fence(અચલ ggml_tensor * sync_tensor, uint32_t fence_seq = 0, તાર્કિક wait = સાચું);
    શૂન્ય enqueue_allreduce(અચલ ggml_tensor * dst, અચલ std::vector<અચલ ggml_tensor *> & src_tensors,
                           અચલ std::vector<અચલ ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);

    શૂન્ય start_batch();
    શૂન્ય flush_sync(તાર્કિક all = સાચું);
    શૂન્ય flush_async();
    શૂન્ય flush_batch(size_t min_ops = 1);
    શૂન્ય flush_peers();
    શૂન્ય flush_pending(તાર્કિક all = સાચું);

    ggml_hexagon_shared_buffer * mmap_tensor(અચલ ggml_tensor * t);
    તાર્કિક clone_buffer(અચલ ggml_hexagon_shared_buffer*);
    શૂન્ય release_buffer(અચલ ggml_hexagon_shared_buffer*);
    શૂન્ય unclone_buffer(અચલ ggml_hexagon_shared_buffer*);

    શૂન્ય add_peer(ggml_hexagon_session * peer) {
        જો (આ->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } નહીંતર {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

સંરચના ggml_backend_hexagon_device_context {
    પૂર્ણાંક                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_buffer_type buffer_type       = {};
    ggml_backend_buffer_type host_buffer_type  = {};
    ggml_backend_buffer_type fence_buffer_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(પૂર્ણાંક dev_id, અચલ ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    અચલ અક્ષર * c_name() અચલ { પાછા config.name.c_str(); }

    ggml_hexagon_session * session() {
        જો (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        પાછા sess.get();
    }
};

સંરચના ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(અચલ std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        આ->dev_ctx = dev_ctx;
        આ->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

સંરચના ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    પૂર્ણાંક       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        જો (!base) {
            ફેંકો std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        જો (fd < 0) {
            rpcmem_free(base);
            ફેંકો std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        આ->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        જો (base) {
            rpcmem_free(base);
        }
    }
};

સંરચના ggml_hexagon_shared_buffer {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    તાર્કિક     mapped;
    તાર્કિક     pinned;
    તાર્કિક     extended;

    અચલ અક્ષર * c_name() અચલ { પાછા sess->c_name(); }
    uint8_t *    base()   અચલ { પાછા mem ? mem->base : nullptr; }
    size_t       size()   અચલ { પાછા mem ? mem->size : 0;  }
    પૂર્ણાંક          fd()     અચલ { પાછા mem ? mem->fd   : -1; }

    શૂન્ય mmap(તાર્કિક extended = ખોટું) {
        જો (!આ->mem)   પાછા;
        જો (આ->mapped) પાછા;

        GGML_ASSERT(!આ->pinned || !extended);

        આ->extended = extended;
        fastrpc_map_flags flags = આ->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        પૂર્ણાંક err = fastrpc_mmap(sess->domain_id, fd(), (શૂન્ય *) base(), 0, size(), flags);
        જો (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (unsigned) err);
            ફેંકો std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (શૂન્ય *) base(), size(), fd(), pinned, extended);

        આ->mapped = સાચું;
    }

    શૂન્ય unmap() {
        જો (!આ->mapped) પાછા;

        જો (!આ->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        જો (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (શૂન્ય *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (શૂન્ય *) base(), size(), fd());

        આ->mapped = ખોટું;
    }

    શૂન્ય alloc(size_t size) {
        જો (આ->mem) પાછા;

        આ->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (શૂન્ય *) base(), આ->size(), fd(), (પૂર્ણાંક) pinned);
        જો (આ->pinned) {
            mmap();
        }
    }

    શૂન્ય મુક્ત() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (શૂન્ય *) base(), size(), fd());
        આ->mem  = nullptr;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, તાર્કિક pinned = ખોટું) {
        આ->sess     = sess;
        આ->mapped   = ખોટું;
        આ->pinned   = pinned;
        આ->extended = ખોટું;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_offset = (size + 4095) & ~4095;
        size_t total_size   = guard_offset + 4096;
        જો (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            total_size = (total_size + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(total_size);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, અચલ ggml_hexagon_shared_buffer & other) {
        આ->sess   = sess;
        આ->mem    = other.mem;
        આ->mapped = ખોટું;
        આ->pinned = other.pinned;
        આ->extended = other.extended;
    }

    ~ggml_hexagon_shared_buffer() {
        મુક્ત();
        માટે (auto * extra : tensor_extra) {
            delete extra;
        }
    }
};

સંરચના ggml_hexagon_fence_buffer : જાહેર ggml_hexagon_shared_buffer {
    uint32_t              slot_ગણતરી = 0;
    uint32_t              slot_head  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_buffer   backend_buffer{};

    ggml_hexagon_fence_buffer(ggml_hexagon_session * sess, ggml_backend_buffer_type_t buft, size_t size)
        : ggml_hexagon_shared_buffer(sess, size, ખોટું /* pinned */),
          slot_ગણતરી(size / GGML_HEXAGON_FENCE_SLOT_SIZE),
          slot_head(0) {
        backend_buffer.buft    = buft;
        backend_buffer.context = static_cast<ggml_hexagon_shared_buffer *>(આ);
        backend_buffer.size    = size;
        mmap(ખોટું);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        જો (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        } નહીંતર જો (slot_head + n_slots <= slot_ગણતરી) {
            uint32_t slot = slot_head;
            slot_head += n_slots;
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        }
        જો (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_SLOT_SIZE);
        }
        પાછા ptr;
    }

    શૂન્ય free_slot(શૂન્ય * ptr, uint32_t n_slots = 1) {
        જો (!ptr) પાછા;
        uint32_t slot = ((uint8_t *) ptr - base()) / GGML_HEXAGON_FENCE_SLOT_SIZE;
        માટે (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

અંતઃરેખ uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    પાછા ptr;
}

અંતઃરેખ શૂન્ય ggml_hexagon_session::free_fence(શૂન્ય * ptr, uint32_t n_slots) {
    જો (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

સ્થિર શૂન્ય ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    sbuf->sess->unclone_buffer(sbuf);
    delete sbuf;
}

સ્થિર શૂન્ય * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    પાછા sbuf->base();
}

સ્થિર ગણના ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (શૂન્ય *) sbuf->base(), tensor->data, ggml_nbytes(tensor));

    auto extra = નવું ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    પાછા GGML_STATUS_SUCCESS;
}

સ્થિર શૂન્ય ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   અચલ શૂન્ય *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    જો (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        જો (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, data, offset, size, (પૂર્ણાંક) buffer->usage, extra->flags);

    જો ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy((અક્ષર *) tensor->data + offset, data, size);
        પાછા;
    }

    જો (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        htp_repack_tensor_tiled(tensor, data, size);
        પાછા;
    }

    જો (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    memcpy(extra->shadow_buf.data() + offset, data, size);
    extra->shadow_size += size;

    જો (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

સ્થિર શૂન્ય ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   અચલ ggml_tensor *   tensor,
                                                   શૂન્ય *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, data, offset, size, (પૂર્ણાંક) buffer->usage, extra->flags);

    જો ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy(data, (અચલ અક્ષર *) tensor->data + offset, size);
        પાછા;
    }

    htp_unpack_tensor_tiled(data, tensor, offset, size);
}

સ્થિર તાર્કિક ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   અચલ સંરચના ggml_tensor * src,
                                                   સંરચના ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    પાછા ખોટું;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

સ્થિર શૂન્ય ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                          ggml_tensor *         tensor,
                                                          અચલ શૂન્ય *          data,
                                                          size_t                offset,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    જો (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        જો (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (પૂર્ણાંક) buffer->usage, extra->flags);

    જો ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        માટે (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (અચલ uint8_t *) data + i * stride_data, size);
        }
        પાછા;
    }

    જો (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    માટે (size_t i = 0; i < n_copies; i++) {
        memcpy(extra->shadow_buf.data() + offset + i * stride_tensor, (અચલ uint8_t *) data + i * stride_data, size);
    }
    extra->shadow_size += n_copies * size;

    જો (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

સ્થિર શૂન્ય ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                      અચલ ggml_tensor *   tensor,
                                                      શૂન્ય *                data,
                                                      size_t                offset,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (પૂર્ણાંક) buffer->usage);

    જો ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        માટે (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *)data + i * stride_data, (અચલ uint8_t *)tensor->data + offset + i * stride_tensor, size);
        }
        પાછા;
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

    માટે (size_t i = 0; i < n_copies; i++) {
        memcpy((uint8_t *) data + i * stride_data, temp_buf.data() + i * stride_tensor, size);
    }
}

સ્થિર શૂન્ય ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (શૂન્ય *) sbuf->base(), sbuf->size());
    memset(sbuf->base(), value, sbuf->size());
}

સ્થિર ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ ખાલી,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ ખાલી,
};

// ** backend buffer type

સ્થિર શૂન્ય ggml_backend_hexagon_host_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                        ggml_tensor *         tensor,
                                                        અચલ શૂન્ય *          data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy((અક્ષર *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

સ્થિર શૂન્ય ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                        અચલ ggml_tensor *   tensor,
                                                        શૂન્ય *                data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy(data, (અચલ અક્ષર *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

સ્થિર ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ ખાલી,
    /* .set_tensor      = */ ggml_backend_hexagon_host_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_buffer_get_tensor,
    /* .set_tensor_2d   = */ ખાલી,
    /* .get_tensor_2d   = */ ખાલી,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ ખાલી,
};

// ** backend buffer type

સ્થિર અચલ અક્ષર * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    પાછા static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

સ્થિર ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    જો (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        પાછા nullptr;
    }
    પ્રયાસ {
        ggml_hexagon_shared_buffer * sbuf = નવું ggml_hexagon_shared_buffer(sess, size, ખોટું);
        પાછા ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } પકડો (અચલ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        પાછા nullptr;
    }
}

સ્થિર ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    જો (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        પાછા nullptr;
    }
    પ્રયાસ {
        ggml_hexagon_shared_buffer * sbuf = નવું ggml_hexagon_shared_buffer(sess, size, ખોટું);
        પાછા ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_host_buffer_interface, sbuf, size);
    } પકડો (અચલ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        પાછા nullptr;
    }
}

સ્થિર size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    પાછા 128;  // HVX alignment
    GGML_UNUSED(buft);
}

સ્થિર size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, અચલ સંરચના ggml_tensor * t) {
    જો (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_size = htp_tiled_row_size(t->type, ne0) * ne1 * ne2 * ne3;
        પાછા std::max(tiled_size, ggml_nbytes(t));
    }
    પાછા ggml_nbytes(t);
}

સ્થિર size_t parse_size(અચલ અક્ષર * str, size_t default_unit = 1024 * 1024) {
    જો (!str || str[0] == '\0') {
        પાછા 0;
    }
    અક્ષર * end = ખાલી;
    દ્વિગુણ val = strtod(str, &end);
    જો (val < 0) {
        પાછા 0;
    }
    જો (end && *end) {
        જ્યારે (*end == ' ') end++;
        જો (*end == 'k' || *end == 'K') {
            પાછા (size_t) (val * 1024);
        } નહીંતર જો (*end == 'm' || *end == 'M') {
            પાછા (size_t) (val * 1024 * 1024);
        } નહીંતર જો (*end == 'g' || *end == 'G') {
            પાછા (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    પાછા (size_t) (val * default_unit);
}

સ્થિર size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    પાછા opt_mbuf_dyn;
    GGML_UNUSED(buft);
}

સ્થિર તાર્કિક ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    પાછા ખોટું;
    GGML_UNUSED(buft);
}

સ્થિર તાર્કિક ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    પાછા સાચું;
    GGML_UNUSED(buft);
}

સંરચના ggml_backend_hexagon_alloc_buffer_n_plan_item {
    size_t size;
    પૂર્ણાંક    first;
    પૂર્ણાંક    last;
};

using ggml_backend_hexagon_alloc_buffer_n_plan_t = std::vector<ggml_backend_hexagon_alloc_buffer_n_plan_item>;

સ્થિર અચલ અક્ષર * ggml_hexagon_kv_layer_suffix(અચલ સંરચના ggml_tensor * t) {
    જો (strncmp(t->name, "cache_", 6) != 0) {
        પાછા ખાલી;
    }
    અચલ અક્ષર * p = strstr(t->name, "_l");
    જો (!p || !isdigit((unsigned અક્ષર)p[2])) {
        પાછા ખાલી;
    }
    પાછા p;
}

સંરચના ggml_backend_hexagon_alloc_unit {
    size_t size;
    પૂર્ણાંક    first;
    પૂર્ણાંક    last;
};

સ્થિર ggml_backend_hexagon_alloc_buffer_n_plan_t ggml_backend_hexagon_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, સંરચના ggml_tensor ** tensors, પૂર્ણાંક n_tensors) {
    ggml_backend_hexagon_alloc_buffer_n_plan_t plan;

    અચલ size_t alignment = ggml_backend_buft_get_alignment(buft);
    અચલ size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    પૂર્ણાંક i = 0;
    જ્યારે (i < n_tensors) {
        સંરચના ggml_tensor * t = tensors[i];
        size_t unit_size = 0;
        પૂર્ણાંક unit_first = i;
        પૂર્ણાંક unit_last = i + 1;

        જો (t->data == ખાલી && t->view_src == ખાલી) {
            unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        અચલ અક્ષર * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        જ્યારે (unit_last < n_tensors) {
            સંરચના ggml_tensor * next = tensors[unit_last];

            જો (next->view_src != ખાલી) {
                unit_last++;
                ચાલુ;
            }

            જો (layer_suffix != ખાલી) {
                અચલ અક્ષર * next_suffix = ggml_hexagon_kv_layer_suffix(next);
                જો (next_suffix != ખાલી && strcmp(layer_suffix, next_suffix) == 0) {
                    જો (next->data == ખાલી) {
                        unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, next), alignment);
                    }
                    unit_last++;
                    ચાલુ;
                }
            }

            તોડો;
        }

        units.push_back({ unit_size, unit_first, unit_last });
        i = unit_last;
    }

    size_t cur_buf_size  = 0;
    પૂર્ણાંક    cur_buf_first = 0;

    માટે (અચલ auto & unit : units) {
        જો (unit.size == 0) {
            ચાલુ;
        }

        જો (cur_buf_size > 0 && (cur_buf_size + unit.size) > max_size) {
            plan.push_back({ cur_buf_size, cur_buf_first, unit.first });
            cur_buf_size  = 0;
            cur_buf_first = unit.first;
        }

        cur_buf_size += unit.size;
    }

    જો (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, cur_buf_first, n_tensors });
    }

    પાછા plan;
}

સ્થિર ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, સંરચના ggml_tensor ** tensors, પૂર્ણાંક n_tensors) {
    અચલ ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    માટે (અચલ auto & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        જો (buffer == ખાલી) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            માટે (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            પાછા ખાલી;
        }

        સંરચના ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        સંરચના ggml_tensor * t_failed = ખાલી;
        માટે (પૂર્ણાંક j = item.first; j < item.last; j++) {
            સંરચના ggml_tensor * t = tensors[j];
            જો (t->data == ખાલી) {
                જો (t->view_src == ખાલી) {
                    જો (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        તોડો;
                    }
                } નહીંતર જો (t->buffer == ખાલી) {
                    જો (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        તોડો;
                    }
                }
            } નહીંતર {
                જો (t->view_src != ખાલી && t->buffer == ખાલી) {
                    જો (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        તોડો;
                    }
                }
            }
        }
        જો (t_failed != ખાલી) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            માટે (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            પાછા ખાલી;
        }

        buffers.push_back(buffer);
    }

    જો (buffers.empty()) {
        પાછા ખાલી;
    }

    જો (buffers.size() == 1) {
        પાછા buffers[0];
    }

    પાછા ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

સ્થિર size_t ggml_backend_hexagon_buffer_type_get_alloc_size_n(
        ggml_backend_buffer_type_t buft, સંરચના ggml_tensor ** tensors, પૂર્ણાંક n_tensors) {
    અચલ ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    માટે (અચલ auto & item : plan) {
        total += item.size;
    }
    પાછા total;
}

સ્થિર ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_buffer_type_is_host,
};

સ્થિર ggml_backend_buffer_type_i ggml_backend_hexagon_host_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_host_buffer_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(પૂર્ણાંક dev_id, અચલ ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    buffer_type.device  = dev;
    buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    buffer_type.context = નવું ggml_backend_hexagon_buffer_type_context(config.name, આ);

    host_buffer_type.device  = dev;
    host_buffer_type.iface   = ggml_backend_hexagon_host_buffer_type_interface;
    host_buffer_type.context = નવું ggml_backend_hexagon_buffer_type_context(config.name + "-HOST", આ);

    fence_buffer_type.device  = dev;
    fence_buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    fence_buffer_type.context = નવું ggml_backend_hexagon_buffer_type_context(config.name + "-FENCE", આ);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(host_buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(fence_buffer_type.context);
}

સ્થિર તાર્કિક ggml_backend_buffer_is_hexagon(અચલ સંરચના ggml_backend_buffer * b) {
    પાછા b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

સંરચના ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<પૂર્ણાંક, પૂર્ણાંક>                b_map; // buffer fd   to index
    std::unordered_map<અચલ ggml_tensor*, પૂર્ણાંક> t_map; // tensor ptr  to index
    std::unordered_multimap<શૂન્ય*, પૂર્ણાંક>         d_map; // tensor data to index

    unsigned પૂર્ણાંક n_bufs;     // num buffers in the batch
    unsigned પૂર્ણાંક n_tens;     // num tensors ...
    unsigned પૂર્ણાંક n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    unsigned પૂર્ણાંક n_bufs_max;
    unsigned પૂર્ણાંક n_tens_max;
    unsigned પૂર્ણાંક n_ops_max;
    size_t       b_vmem_max;

    શૂન્ય reset() {
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
        આ->sess = sess;

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

    તાર્કિક empty() અચલ { પાછા n_ops == 0; }

    // add buffer and return its index
    પૂર્ણાંક add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        auto it = b_map.find(sbuf->fd());
        જો (it != b_map.end()) { પાછા it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        પૂર્ણાંક bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        જો (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (શૂન્ય*) sbuf->base(), (size_t) b.size, b_vmem);

        પાછા bi;
    }

    તાર્કિક same_shape(અચલ htp_tensor * h, અચલ ggml_tensor * t) અચલ {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;

        અચલ તાર્કિક is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        અચલ int64_t ne0_padded = hex_round_up(t->ne[0], 32);
        અચલ int64_t ne1_padded = hex_round_up(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_row_size(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? કદ(ggml_fp16_t) : t->nb[0];
        ગણના ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        પાછા (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    પૂર્ણાંક add_tensor(અચલ ggml_tensor * t) {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;
        auto sbuf  = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        માટે (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            જો (same_shape(h, t)) { પાછા it->second; }
        }

        // Lookup by tensor ptr
        auto it = t_map.find(t);
        જો (it != t_map.end()) { પાછા it->second; }

        // Add new tensor to the batch
        પૂર્ણાંક ti = n_tens++;
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

        અચલ તાર્કિક is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        જો (is_tiled) {
            જો (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = કદ(ggml_fp16_t);
            } નહીંતર {
                h.nb[0] = t->nb[0];
            }
            અચલ int64_t ne0_padded = hex_round_up(t->ne[0], 32);
            અચલ int64_t ne1_padded = hex_round_up(t->ne[1], 32);
            h.nb[1] = htp_tiled_row_size(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_size  = h.size;
        } નહીંતર {
            h.size  = t_size;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        જો ((extra->flags & GGML_HEXAGON_TENSOR_WEIGHT) != 0) {
            h.flags |= HTP_TENSOR_WEIGHT;
        }
        જો ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        જો ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (શૂન્ય*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        પાછા ti;
    }

    તાર્કિક fit_op(અચલ htp_opnode & ગાંઠ) અચલ {
        જો (n_ops >= n_ops_max) પાછા ખોટું;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        પૂર્ણાંક seen_bufs[HTP_OP_MAX_BUFS];
        પૂર્ણાંક n_seen_bufs = 0;

        auto fit_tensor = [&](અચલ ggml_tensor *t) {
            જો (!t) પાછા;
            જો (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                પૂર્ણાંક fd = sbuf->fd();
                જો (!b_map.count(fd)) {
                    માટે (પૂર્ણાંક i = 0; i < n_seen_bufs; i++) {
                        જો (seen_bufs[i] == fd) પાછા;
                    }
                    જો (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    જો (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        માટે (અચલ auto * src : ગાંઠ.get_inputs()) {
            fit_tensor(src);
        }
        માટે (અચલ auto * output : ગાંઠ.get_outputs()) {
            fit_tensor(output);
        }

        જો ((extra_bufs + n_bufs) > n_bufs_max) પાછા ખોટું;
        જો ((extra_tens + n_tens) > n_tens_max) પાછા ખોટું;
        જો ((extra_vmem + b_vmem) > b_vmem_max) પાછા ખોટું;

        પાછા સાચું;
    }

    // assumes that fit_op() was called first and returned true
    શૂન્ય add_op(અચલ htp_opnode & ગાંઠ) {
        // Add new op

        unsigned પૂર્ણાંક n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = ગાંઠ;

        htp_op_desc &o = h_ops[n];
        memcpy(o.params,        ગાંઠ.ગાંઠ->op_params, કદ(ગાંઠ.ગાંઠ->op_params));
        memcpy(o.kernel_params, ગાંઠ.kernel_params,   કદ(o.kernel_params));
        o.opcode = ગાંઠ.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        auto inputs = ગાંઠ.get_inputs();
        માટે (unsigned પૂર્ણાંક i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        auto outputs = ગાંઠ.get_outputs();
        માટે (unsigned પૂર્ણાંક i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    શૂન્ય sort_buffers() {
        જો (n_bufs <= 1) પાછા;

        std::vector<પૂર્ણાંક> order(n_bufs);
        માટે (unsigned પૂર્ણાંક i = 0; i < n_bufs; i++) { order[i] = (પૂર્ણાંક) i; }

        std::stable_sort(order.begin(), order.end(), [&](પૂર્ણાંક a, પૂર્ણાંક b) {
            પાછા h_bufs[a].size > h_bufs[b].size;
        });

        તાર્કિક already_sorted = સાચું;
        માટે (unsigned પૂર્ણાંક i = 0; i < n_bufs; i++) {
            જો (order[i] != (પૂર્ણાંક) i) {
                already_sorted = ખોટું;
                તોડો;
            }
        }
        જો (already_sorted) પાછા;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        માટે (unsigned પૂર્ણાંક new_bi = 0; new_bi < n_bufs; new_bi++) {
            પૂર્ણાંક old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        માટે (unsigned પૂર્ણાંક i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        માટે (unsigned પૂર્ણાંક i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    શૂન્ય update_mdev_group(uint32_t mdev_idx) {
        જો (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    તાર્કિક try_fuse_common(std::initializer_list<અચલ ggml_tensor *> tensors) અચલ {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        પૂર્ણાંક seen_bufs[HTP_OP_MAX_BUFS];
        પૂર્ણાંક n_seen_bufs = 0;

        માટે (અચલ auto * t : tensors) {
            જો (!t || t_map.count(t)) {
                ચાલુ;
            }
            extra_tens++;
            auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            પૂર્ણાંક fd = sbuf->fd();
            જો (!b_map.count(fd)) {
                તાર્કિક found = ખોટું;
                માટે (પૂર્ણાંક i = 0; i < n_seen_bufs; i++) {
                    જો (seen_bufs[i] == fd) {
                        found = સાચું;
                        તોડો;
                    }
                }
                જો (!found) {
                    જો (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    જો (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        જો ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            પાછા ખોટું;
        }

        પાછા સાચું;
    }

    તાર્કિક try_fuse_common(અચલ ggml_tensor * t1, અચલ ggml_tensor * t2) અચલ {
        પાછા try_fuse_common({t1, t2});
    }

    તાર્કિક try_fuse_allreduce_add(અચલ htp_opnode & ગાંઠ) {
        જો (n_ops == 0 || opt_ar_select != 2) પાછા ખોટું;
        જો (ગાંઠ.opcode != HTP_OP_ADD) પાછા ખોટું;

        htp_opnode & last_ગાંઠ = ops[n_ops - 1];
        જો (last_ગાંઠ.opcode != HTP_OP_ALLREDUCE) પાછા ખોટું;

        auto * ar_kparams = (સંરચના htp_allreduce_kernel_params *) last_ગાંઠ.kernel_params;
        અચલ uint32_t rank    = (uint32_t) ar_kparams->rank;
        અચલ uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        અચલ ggml_tensor * ar_local = last_ગાંઠ.inputs[rank];
        અચલ ggml_tensor * add_src0 = ગાંઠ.src0();
        અચલ ggml_tensor * add_src1 = ગાંઠ.src1();
        અચલ ggml_tensor * add_dst  = ગાંઠ.dst();

        જો (!ggml_hexagon_tensor_is_fuseable(ar_local)) પાછા ખોટું;

        અચલ ggml_tensor * res_tensor;
        જો (add_src0 == ar_local || add_src0->data == ar_local->data) {
            res_tensor = add_src1;
        } નહીંતર જો (add_src1 == ar_local || add_src1->data == ar_local->data) {
            res_tensor = add_src0;
        } નહીંતર {
            પાછા ખોટું;
        }

        જો (ar_local->type != res_tensor->type) પાછા ખોટું;

        અચલ તાર્કિક is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        અચલ તાર્કિક is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        જો (!is_same_shape && !is_row_bcast) પાછા ખોટું;

        જો (is_same_shape) {
            જો (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                પાછા ખોટું;
            }
            જો (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                પાછા ખોટું;
            }
        }
        જો (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) {
            પાછા ખોટું;
        }

        // scatter is only valid for in-place add within max outputs
        અચલ તાર્કિક is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

        સંરચના htp_allreduce_kernel_params new_kparams;
        જો (!ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, સાચું, is_row_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            પાછા ખોટું;
        }

        અચલ તાર્કિક scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

        માટે (uint32_t r = 0; r < n_ranks; r++) {
            અચલ ggml_tensor * ar_src = last_ગાંઠ.inputs[r];
            જો (!ggml_hexagon_tensors_overlap(add_dst, ar_src)) {
                ચાલુ;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            જો (scatter_ok && r == rank) {
                ચાલુ;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            પાછા ખોટું;
        }

        જો (!try_fuse_common(res_tensor, add_dst)) {
            પાછા ખોટું;
        }

        last_ગાંઠ.opcode = HTP_OP_ALLREDUCE_ADD;
        last_ગાંઠ.name   = "ALLREDUCE+ADD";
        last_ગાંઠ.inputs.push_back(res_tensor);
        last_ગાંઠ.outputs.clear();
        last_ગાંઠ.outputs.push_back(add_dst);
        last_ગાંઠ.fused.push_back(ગાંઠ.ગાંઠ);
        memcpy(last_ગાંઠ.kernel_params, &new_kparams, કદ(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ADD;
        memcpy(o.kernel_params, &new_kparams, કદ(new_kparams));

        o.src[2 * n_ranks] = add_tensor(res_tensor);
        જો (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            માટે (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            માટે (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } નહીંતર {
            o.dst[0] = add_tensor(add_dst);
            માટે (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (પૂર્ણાંક) new_kparams.mode, (પૂર્ણાંક) new_kparams.n_dsts);
        પાછા સાચું;
    }

    તાર્કિક try_fuse_rms_norm_mul(અચલ htp_opnode & ગાંઠ) {
        જો (n_ops == 0) પાછા ખોટું;
        જો (ગાંઠ.opcode != HTP_OP_MUL) પાછા ખોટું;

        htp_opnode & last_ગાંઠ = ops[n_ops - 1];
        જો (last_ગાંઠ.opcode != HTP_OP_RMS_NORM) પાછા ખોટું;

        અચલ ggml_tensor * mul_src0 = ગાંઠ.src0();
        અચલ ggml_tensor * mul_src1 = ગાંઠ.src1();
        અચલ ggml_tensor * rms_out  = last_ગાંઠ.dst();

        જો (!ggml_hexagon_tensor_is_fuseable(rms_out)) પાછા ખોટું;

        અચલ ggml_tensor * weight;
        જો (mul_src0 == rms_out || mul_src0->data == rms_out->data) {
            weight = mul_src1;
        } નહીંતર જો (mul_src1 == rms_out || mul_src1->data == rms_out->data) {
            weight = mul_src0;
        } નહીંતર {
            પાછા ખોટું;
        }

        અચલ ggml_tensor * src0 = last_ગાંઠ.src0();

        જો (src0->ne[0] != weight->ne[0] || src0->ne[0] != ગાંઠ.dst()->ne[0]) {
            પાછા ખોટું;
        }

        અચલ તાર્કિક is_row_bcast = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
        અચલ તાર્કિક is_same_shape = (src0->ne[0] == weight->ne[0] && src0->ne[1] == weight->ne[1] &&
                                    src0->ne[2] == weight->ne[2] && src0->ne[3] == weight->ne[3]);
        જો (!is_row_bcast && !is_same_shape) પાછા ખોટું;

        જો (!ggml_are_same_shape(src0, ગાંઠ.dst())) {
            પાછા ખોટું;
        }
        જો (ggml_is_contiguous(src0) != ggml_is_contiguous(ગાંઠ.dst())) {
            પાછા ખોટું;
        }

        સંરચના htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, weight, ગાંઠ.dst(), &new_kparams
        );

        જો ((size_t) new_kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_size, sess->vtcm_size);
            પાછા ખોટું;
        }

        જો (!try_fuse_common(weight, ગાંઠ.dst())) {
            પાછા ખોટું;
        }

        last_ગાંઠ.opcode = HTP_OP_RMS_NORM_MUL;
        last_ગાંઠ.name   = "RMS_NORM+MUL";
        last_ગાંઠ.inputs.clear();
        last_ગાંઠ.inputs.push_back(src0);
        last_ગાંઠ.inputs.push_back(weight);
        last_ગાંઠ.outputs.clear();
        last_ગાંઠ.outputs.push_back(ગાંઠ.dst());
        last_ગાંઠ.fused.push_back(ગાંઠ.ગાંઠ);
        memcpy(last_ગાંઠ.kernel_params, &new_kparams, કદ(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        memcpy(o.kernel_params, &new_kparams, કદ(new_kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(weight);
        માટે (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(ગાંઠ.dst());
        માટે (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        પાછા સાચું;
    }

    તાર્કિક try_fuse_mul_mat_add(અચલ htp_opnode & ગાંઠ) {
        જો (n_ops == 0) પાછા ખોટું;
        જો (ગાંઠ.opcode != HTP_OP_ADD) પાછા ખોટું;

        htp_opnode & last_ગાંઠ = ops[n_ops - 1];
        જો (last_ગાંઠ.opcode != HTP_OP_MUL_MAT) પાછા ખોટું;

        અચલ ggml_tensor * add_src0 = ગાંઠ.src0();
        અચલ ggml_tensor * add_src1 = ગાંઠ.src1();
        અચલ ggml_tensor * mm_out   = last_ગાંઠ.dst();

        જો (!ggml_hexagon_tensor_is_fuseable(mm_out)) પાછા ખોટું;

        અચલ ggml_tensor * src2;
        જો (add_src0 == mm_out || add_src0->data == mm_out->data) {
            src2 = add_src1;
        } નહીંતર જો (add_src1 == mm_out || add_src1->data == mm_out->data) {
            src2 = add_src0;
        } નહીંતર {
            પાછા ખોટું;
        }

        અચલ ggml_tensor * src0 = last_ગાંઠ.src0();
        અચલ ggml_tensor * src1 = last_ગાંઠ.src1();

        જો (src2->type != GGML_TYPE_F32) પાછા ખોટું;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        જો (!ggml_are_same_shape(mm_out, ગાંઠ.dst())) પાછા ખોટું;

        અચલ સંરચના htp_mm_kernel_params * orig_kparams = (અચલ સંરચના htp_mm_kernel_params *) last_ગાંઠ.kernel_params;
        સંરચના htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_add_params(sess, src0, src1, src2, ગાંઠ.dst(), &kparams);
        જો (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            પાછા ખોટું;
        }

        જો (orig_kparams->collapse != kparams.collapse) {
            પાછા ખોટું;
        }

        અચલ પૂર્ણાંક src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        અચલ તાર્કિક can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        જો (!can_fuse) પાછા ખોટું;

        જો ((size_t) kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
            પાછા ખોટું;
        }

        જો (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            જો (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_threads < orig_kparams->n_act_threads) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_threads, kparams.n_act_threads);
                પાછા ખોટું;
            }
        }

        જો (!try_fuse_common(src2, ગાંઠ.dst())) {
            પાછા ખોટું;
        }

        last_ગાંઠ.opcode = HTP_OP_MUL_MAT_ADD;
        last_ગાંઠ.name   = "MUL_MAT+ADD";
        last_ગાંઠ.inputs.clear();
        last_ગાંઠ.inputs.push_back(src0);
        last_ગાંઠ.inputs.push_back(src1);
        last_ગાંઠ.inputs.push_back(src2);
        last_ગાંઠ.outputs.clear();
        last_ગાંઠ.outputs.push_back(ગાંઠ.dst());
        last_ગાંઠ.fused.push_back(ગાંઠ.ગાંઠ);
        memcpy(last_ગાંઠ.kernel_params, &kparams, કદ(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ADD;
        memcpy(o.kernel_params, &kparams, કદ(kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(src1);
        o.src[2] = add_tensor(src2);
        માટે (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(ગાંઠ.dst());
        માટે (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        પાછા સાચું;
    }

    તાર્કિક try_fuse_mul_mat_nx(અચલ htp_opnode & ગાંઠ) {
        જો (n_ops == 0 || ગાંઠ.opcode != HTP_OP_MUL_MAT) પાછા ખોટું;
        જો (!is_mergeable_mul_mat(ગાંઠ.ગાંઠ)) પાછા ખોટું;

        અચલ ggml_tensor * w_in = ગાંઠ.src0();
        અચલ ggml_tensor * x_in = ગાંઠ.src1();
        અચલ ggml_tensor * d_in = ગાંઠ.dst();

        htp_opnode & last_ગાંઠ = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        જો (last_ગાંઠ.opcode == HTP_OP_MUL_MAT_NX) {
            અચલ uint32_t curr_n = (uint32_t) last_ગાંઠ.outputs.size();
            જો (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                પાછા ખોટું;
            }

            અચલ ggml_tensor * w0 = last_ગાંઠ.inputs[0];
            અચલ ggml_tensor * x  = last_ગાંઠ.inputs[curr_n];

            જો (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                પાછા ખોટું;
            }
            જો (!last_ગાંઠ.fused.empty() && (mm_is_hmx_eligible(last_ગાંઠ.fused[0]) != mm_is_hmx_eligible(ગાંઠ.ગાંઠ))) {
                પાછા ખોટું;
            }

            અચલ સંરચના htp_mm_kernel_params * orig_kparams = (અચલ સંરચના htp_mm_kernel_params *) last_ગાંઠ.kernel_params;
            અચલ તાર્કિક collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_in, x, d_in);
            જો (orig_kparams->collapse && !collapse) {
                પાછા ખોટું;
            }

            સંરચના htp_mm_kernel_params kparams;
            જો (collapse) {
                અચલ ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } નહીંતર {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            જો (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                પાછા ખોટું;
            }
            જો ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                પાછા ખોટું;
            }

            જો (!try_fuse_common(w_in, d_in)) {
                પાછા ખોટું;
            }

            last_ગાંઠ.inputs[curr_n] = w_in;
            last_ગાંઠ.inputs.push_back(x);
            last_ગાંઠ.outputs.push_back(d_in);
            last_ગાંઠ.fused.push_back(ગાંઠ.ગાંઠ);
            memcpy(last_ગાંઠ.kernel_params, &kparams, કદ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, કદ(kparams));

            માટે (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = add_tensor(last_ગાંઠ.inputs[s]);
            }
            માટે (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            માટે (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_ગાંઠ.outputs[d]);
            }
            માટે (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            પાછા સાચું;
        }

        // Case 2: last_node is single MUL_MAT
        જો (last_ગાંઠ.opcode == HTP_OP_MUL_MAT) {
            જો (!is_mergeable_mul_mat_pair(last_ગાંઠ.ગાંઠ, ગાંઠ.ગાંઠ)) {
                પાછા ખોટું;
            }

            અચલ ggml_tensor * w0 = last_ગાંઠ.src0();
            અચલ ggml_tensor * x  = last_ગાંઠ.src1();
            અચલ ggml_tensor * w1 = ગાંઠ.src0();
            અચલ ggml_tensor * dst_0 = last_ગાંઠ.dst();
            અચલ ggml_tensor * dst_1 = ગાંઠ.dst();

            અચલ સંરચના htp_mm_kernel_params * orig_kparams = (અચલ સંરચના htp_mm_kernel_params *) last_ગાંઠ.kernel_params;
            અચલ તાર્કિક collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            જો (orig_kparams->collapse && !collapse) {
                પાછા ખોટું;
            }

            સંરચના htp_mm_kernel_params kparams;
            જો (collapse) {
                અચલ ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } નહીંતર {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            જો (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                પાછા ખોટું;
            }
            જો ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                પાછા ખોટું;
            }

            જો (!try_fuse_common(w1, ગાંઠ.dst())) {
                પાછા ખોટું;
            }

            last_ગાંઠ.opcode = HTP_OP_MUL_MAT_NX;
            last_ગાંઠ.name   = "MUL_MAT_NX";
            last_ગાંઠ.inputs.clear();
            last_ગાંઠ.inputs.push_back(w0);
            last_ગાંઠ.inputs.push_back(w1);
            last_ગાંઠ.inputs.push_back(x);
            last_ગાંઠ.outputs.clear();
            last_ગાંઠ.outputs.push_back(dst_0);
            last_ગાંઠ.outputs.push_back(dst_1);
            last_ગાંઠ.fused.push_back(ગાંઠ.ગાંઠ);
            memcpy(last_ગાંઠ.kernel_params, &kparams, કદ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            memcpy(o.kernel_params, &kparams, કદ(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            માટે (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            માટે (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            પાછા સાચું;
        }

        પાછા ખોટું;
    }

    તાર્કિક try_fuse_mul_mat_id_nx(અચલ htp_opnode & ગાંઠ) {
        જો (n_ops == 0 || ગાંઠ.opcode != HTP_OP_MUL_MAT_ID) પાછા ખોટું;
        જો (!is_mergeable_mul_mat_id(ગાંઠ.ગાંઠ)) પાછા ખોટું;

        અચલ ggml_tensor * w_in   = ગાંઠ.src0();
        અચલ ggml_tensor * x_in   = ગાંઠ.src1();
        અચલ ggml_tensor * ids_in = ગાંઠ.ગાંઠ->src[2];
        અચલ ggml_tensor * d_in   = ગાંઠ.dst();

        htp_opnode & last_ગાંઠ = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        જો (last_ગાંઠ.opcode == HTP_OP_MUL_MAT_ID_NX) {
            અચલ uint32_t curr_n = (uint32_t) last_ગાંઠ.outputs.size();
            જો (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                પાછા ખોટું;
            }

            અચલ ggml_tensor * w0  = last_ગાંઠ.inputs[0];
            અચલ ggml_tensor * x   = last_ગાંઠ.inputs[curr_n];
            અચલ ggml_tensor * ids = last_ગાંઠ.inputs[curr_n + 1];

            જો (x_in != x || ids_in != ids || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                પાછા ખોટું;
            }
            જો (!last_ગાંઠ.fused.empty() && (mm_is_hmx_eligible(last_ગાંઠ.fused[0]) != mm_is_hmx_eligible(ગાંઠ.ગાંઠ))) {
                પાછા ખોટું;
            }

            સંરચના htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
            જો (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                પાછા ખોટું;
            }
            જો ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                પાછા ખોટું;
            }

            જો (!try_fuse_common(w_in, d_in)) {
                પાછા ખોટું;
            }

            last_ગાંઠ.inputs[curr_n] = w_in;
            last_ગાંઠ.inputs[curr_n + 1] = x;
            last_ગાંઠ.inputs.push_back(ids);
            last_ગાંઠ.outputs.push_back(d_in);
            last_ગાંઠ.fused.push_back(ગાંઠ.ગાંઠ);
            memcpy(last_ગાંઠ.kernel_params, &kparams, કદ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, કદ(kparams));

            માટે (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = add_tensor(last_ગાંઠ.inputs[s]);
            }
            માટે (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            માટે (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_ગાંઠ.outputs[d]);
            }
            માટે (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            પાછા સાચું;
        }

        // Case 2: last_node is single MUL_MAT_ID
        જો (last_ગાંઠ.opcode == HTP_OP_MUL_MAT_ID) {
            જો (!is_mergeable_mul_mat_id_pair(last_ગાંઠ.ગાંઠ, ગાંઠ.ગાંઠ)) {
                પાછા ખોટું;
            }

            અચલ ggml_tensor * w0  = last_ગાંઠ.src0();
            અચલ ggml_tensor * x   = last_ગાંઠ.src1();
            અચલ ggml_tensor * ids = last_ગાંઠ.ગાંઠ->src[2];
            અચલ ggml_tensor * w1  = ગાંઠ.src0();

            સંરચના htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, ગાંઠ.dst(), 2, &kparams);
            જો (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                પાછા ખોટું;
            }
            જો ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                પાછા ખોટું;
            }

            જો (!try_fuse_common(w1, ગાંઠ.dst())) {
                પાછા ખોટું;
            }

            અચલ ggml_tensor * dst_0 = last_ગાંઠ.dst();
            અચલ ggml_tensor * dst_1 = ગાંઠ.dst();

            last_ગાંઠ.opcode = HTP_OP_MUL_MAT_ID_NX;
            last_ગાંઠ.name   = "MUL_MAT_ID_NX";
            last_ગાંઠ.inputs.clear();
            last_ગાંઠ.inputs.push_back(w0);
            last_ગાંઠ.inputs.push_back(w1);
            last_ગાંઠ.inputs.push_back(x);
            last_ગાંઠ.inputs.push_back(ids);
            last_ગાંઠ.outputs.clear();
            last_ગાંઠ.outputs.push_back(dst_0);
            last_ગાંઠ.outputs.push_back(dst_1);
            last_ગાંઠ.fused.push_back(ગાંઠ.ગાંઠ);
            memcpy(last_ગાંઠ.kernel_params, &kparams, કદ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            memcpy(o.kernel_params, &kparams, કદ(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            o.src[3] = add_tensor(ids);
            માટે (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            માટે (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            પાછા સાચું;
        }

        પાછા ખોટું;
    }

    તાર્કિક try_fuse_gdn_cpy(અચલ htp_opnode & ગાંઠ) {
        જો (n_ops == 0 || ગાંઠ.opcode != HTP_OP_CPY) પાછા ખોટું;

        htp_opnode & last_ગાંઠ = ops[n_ops - 1];
        જો (last_ગાંઠ.opcode != HTP_OP_GATED_DELTA_NET) પાછા ખોટું;
        જો (last_ગાંઠ.outputs.size() != 1) પાછા ખોટું;

        અચલ ggml_tensor * gdn_out  = last_ગાંઠ.dst();
        અચલ ggml_tensor * cpy_ગાંઠ = ગાંઠ.ગાંઠ;
        અચલ ggml_tensor * cpy_src  = ગાંઠ.src0();
        અચલ ggml_tensor * cpy_dst  = ગાંઠ.dst();

        જો (!cpy_src || !cpy_dst || !cpy_dst->data) પાછા ખોટું;
        જો (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) પાછા ખોટું;
        જો ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_ગાંઠ->flags & GGML_TENSOR_FLAG_OUTPUT)) પાછા ખોટું;

        અચલ ggml_tensor * v = last_ગાંઠ.ગાંઠ->src[2];
        જો (!v) પાછા ખોટું;

        અચલ int64_t S_v      = v->ne[0];
        અચલ int64_t H        = v->ne[1];
        અચલ int64_t n_tokens = v->ne[2];
        અચલ int64_t n_seqs   = v->ne[3];
        અચલ int64_t K        = ggml_get_op_params_i32(last_ગાંઠ.ગાંઠ, 0);
        અચલ size_t  tail_off = (size_t) S_v * H * n_tokens * n_seqs * કદ(દશાંશ);

        અચલ int64_t D         = S_v * S_v * H;
        અચલ int64_t n_written = std::min<int64_t>(n_tokens, K);

        જો (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_out && cpy_src->view_src->data != gdn_out->data) ||
            cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
            પાછા ખોટું;
        }

        જો (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != કદ(દશાંશ)) {
            પાછા ખોટું;
        }
        જો (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * કદ(દશાંશ)) {
            પાછા ખોટું;
        }
        જો (n_written > 1) {
            જો (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * કદ(દશાંશ)) {
                પાછા ખોટું;
            }
        }

        જો (!try_fuse_common({cpy_dst})) {
            પાછા ખોટું;
        }

        last_ગાંઠ.name += "+CPY";
        last_ગાંઠ.outputs.push_back(cpy_dst);
        last_ગાંઠ.fused.push_back(ગાંઠ.ગાંઠ);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = add_tensor(cpy_dst);
        માટે (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        પાછા સાચું;
    }

    તાર્કિક try_fuse(અચલ htp_opnode & ગાંઠ) {
        જો (!opt_opfusion) પાછા ખોટું;
        જો (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ADD) && try_fuse_allreduce_add(ગાંઠ)) પાછા સાચું;
        જો (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(ગાંઠ))  પાછા સાચું;
        જો (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ADD)   && try_fuse_mul_mat_add(ગાંઠ))   પાછા સાચું;
        જો (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(ગાંઠ))    પાછા સાચું;
        જો (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(ગાંઠ)) પાછા સાચું;
        જો (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(ગાંઠ))       પાછા સાચું;
        પાછા ખોટું;
    }
};

સંરચના ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

સંરચના ggml_hexagon_opqueue {
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
        જો (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * કદ(htp_trace_desc);
        }

        shm_blk_size = કદ(htp_buf_desc)  * n_bufs    +
                       કદ(htp_tensor)    * n_tensors +
                       કદ(htp_op_desc)   * n_ops     +
                       કદ(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = નવું ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, સાચું /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        જો (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size(), shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    size_t shm_size() અચલ { પાછા shm_buf ? shm_buf->size() : 0; }

    // push new batch
    તાર્કિક ધકેલો(htp_opbatch_req& req, dspqueue_buffer& dbuf, અચલ ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(કદ(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(કદ(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(કદ(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(કદ(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(કદ(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(કદ(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        જો (seq - shm_buf->sess->batch_rsp_seq > depth) { પાછા ખોટું; }

        અચલ uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[slot]   = op_batch->ops;
        start_usec[slot] = ggml_time_us();

        અચલ size_t b_size = કદ(htp_buf_desc)  * req.n_bufs;
        અચલ size_t t_size = કદ(htp_tensor)    * req.n_tensors;
        અચલ size_t o_size = કદ(htp_op_desc)   * req.n_ops;
        અચલ size_t p_size = કદ(htp_prof_desc) * req.n_ops;

        size_t tr_size = 0;
        જો (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * કદ(htp_trace_desc);
        } નહીંતર {
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

        memcpy(b_ptr, (શૂન્ય *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (શૂન્ય *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (શૂન્ય *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        જો (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            માટે (unsigned પૂર્ણાંક i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (શૂન્ય *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            માટે (unsigned પૂર્ણાંક i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (unsigned long long) t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        પાછા સાચું;
    }

    શૂન્ય કાઢો(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        અચલ uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        અચલ size_t b_size = કદ(htp_buf_desc)  * rsp.n_bufs;
        અચલ size_t t_size = કદ(htp_tensor)    * rsp.n_tensors;
        અચલ size_t o_size = કદ(htp_op_desc)   * rsp.n_ops;
        અચલ size_t p_size = કદ(htp_prof_desc) * rsp.n_ops;

        size_t tr_size = 0;
        uint32_t n_traces = 0;
        જો (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * કદ(htp_trace_desc);
        }

        અચલ size_t m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        જો (rsp.n_ops > 0) {
            auto & ops = op_cache[slot];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            અચલ htp_prof_desc * pd = (અચલ htp_prof_desc *) p_ptr;
            અચલ htp_trace_desc * trace_events = nullptr;
            જો (opt_profile == 3) {
                trace_events = (અચલ htp_trace_desc *) (p_ptr + p_size);
            }

            જો (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            માટે (uint32_t i = 0; i < rsp.n_ops; i++) {
                જો (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            જો (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

શૂન્ય ggml_hexagon_session::flush_peers() {
    auto vpeers = std::move(virt_peers);
    virt_peers.clear();
    માટે (auto * peer : vpeers) {
        peer->flush_sync();
    }

    auto ppeers = std::move(phys_peers);
    phys_peers.clear();
    માટે (auto * peer : ppeers) {
        peer->flush_async();
    }

    માટે (auto & sub : આ->mdev.sessions) {
        sub->flush_peers();
    }
}

શૂન્ય ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

શૂન્ય ggml_hexagon_session::flush_pending(તાર્કિક all) {
    માટે (auto & sub : આ->mdev.sessions) {
        sub->flush_pending(all);
        જો (sub->last_error > HTP_STATUS_OK) {
            આ->last_error = sub->last_error.load();
        }
    }

    જ્યારે (આ->batch_rsp_seq < આ->batch_req_seq) {
        સંરચના htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        સંરચના dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        અચલ uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        પૂર્ણાંક err = dspqueue_read(આ->queue, &flags, 1, &n_dbufs, &dbuf, કદ(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        જો (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            ચાલુ;
        }

        જો (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        જો (rsp_size != કદ(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", આ->c_name(), rsp_size, n_dbufs);
        }

        જો (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", આ->c_name(), status_to_str(rsp.status));
            આ->last_error = rsp.status;
            માટે (auto & sub : આ->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->કાઢો(rsp, dbuf);

        GGML_ASSERT(rsp.seq == આ->batch_rsp_seq + 1);
        આ->batch_rsp_seq = rsp.seq;

        જો (!all) તોડો;
    }
}

શૂન્ય ggml_hexagon_session::flush_sync(તાર્કિક all) {
    flush_async();
    flush_pending(all);
}

શૂન્ય ggml_hexagon_session::start_batch() {
    જો (આ->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

શૂન્ય ggml_hexagon_session::flush_batch(size_t min_ops) {
    જો (op_batch->n_ops < min_ops) { પાછા; }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    અચલ uint64_t seq = ++આ->batch_req_seq;

    op_batch->update_mdev_group(આ->mdev.idx);

    જો (!op_queue->ધકેલો(req, dbuf, op_batch, seq)) {
        flush_pending(ખોટું);
        op_queue->ધકેલો(req, dbuf, op_batch, seq);
    }

    માટે (auto & sub : આ->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_buffer sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        જો (!sub->op_queue->ધકેલો(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(ખોટું);
            sub->op_queue->ધકેલો(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        પૂર્ણાંક err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, કદ(sub_req), (અચલ uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        જો (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (unsigned) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", આ->c_name(), dbuf.ptr, dbuf.size);

    પૂર્ણાંક err = dspqueue_write(આ->queue, 0, 1, &dbuf, કદ(req), (અચલ uint8_t*) &req, DSPQUEUE_TIMEOUT);
    જો (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", આ->c_name(), (unsigned) err);
    }

    op_batch->reset();
}

શૂન્ય ggml_hexagon_session::enqueue_op(અચલ htp_opnode & ગાંઠ) {
    auto clone_tensor_buffer = [આ](અચલ ggml_tensor * t) {
        auto sbuf = આ->mmap_tensor(t);
        જો (!sbuf) પાછા;
        જો (sbuf->sess != આ) {
            આ->clone_buffer(sbuf);
        }
        માટે (auto & sub : આ->mdev.sessions) {
            sub->clone_buffer(sbuf);
        }
    };

    માટે (auto t : ગાંઠ.get_inputs()) {
        clone_tensor_buffer(t);
    }
    માટે (auto t : ગાંઠ.get_outputs()) {
        clone_tensor_buffer(t);
    }

    જો (opt_opfusion && op_batch->try_fuse(ગાંઠ)) {
        પાછા;
    }

    જો (!op_batch->fit_op(ગાંઠ)) {
        flush_async();
    }

    જો (op_batch->empty()) {
        start_batch();
    }

    જો (!op_batch->fit_op(ગાંઠ)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->add_op(ગાંઠ);
}

શૂન્ય ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_ગાંઠ(HTP_OP_MDEV_GROUP);

    uint8_t * fence_slot = આ->mdev_fence_slot;

    સ્થિર ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.buffer       = &આ->fence_buf->backend_buffer;
    dummy_t.extra        = &fence_extra;
    dummy_t.data         = (શૂન્ય *) fence_slot;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.ne[1]        = (int64_t) આ->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) આ->mdev.idx;

    ggml_tensor * ગાંઠ = group_ગાંઠ.add_dummy(dummy_t);
    ગાંઠ->src[0] = ગાંઠ;
    group_ગાંઠ.init(ગાંઠ);
    group_ગાંઠ.outputs.clear();
    group_ગાંઠ.name = "MDEV_GROUP";

    માટે (auto & sub : આ->mdev.sessions) {
        sub->clone_buffer(આ->fence_buf);
    }

    op_batch->add_op(group_ગાંઠ);
}

શૂન્ય ggml_hexagon_session::enqueue_cpy(અચલ ggml_tensor * src, ggml_tensor * dst, અચલ ggml_tensor * sync_tensor, uint32_t fence_seq) {
    અચલ તાર્કિક with_fence = sync_tensor != nullptr;
    htp_opnode cpy_ગાંઠ(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* ગાંઠ = cpy_ગાંઠ.add_dummy(*dst);
    ગાંઠ->op     = GGML_OP_CPY;
    ગાંઠ->src[0] = const_cast<ggml_tensor *>(src);
    ગાંઠ->src[1] = with_fence ? cpy_ગાંઠ.add_dummy(*sync_tensor) : nullptr;
    જો (with_fence) {
        ગાંઠ->op_params[0] = (int32_t) fence_seq;
    }

    cpy_ગાંઠ.init(ગાંઠ);
    જો (with_fence) {
        cpy_ગાંઠ.name = "CPY+FENCE";
    }
    અચલ તાર્કિક ok = ggml_hexagon_precompute_cpy_params(આ, ગાંઠ, (સંરચના htp_copy_kernel_params *) cpy_ગાંઠ.kernel_params);
    અચલ auto * kparams = (અચલ સંરચના htp_copy_kernel_params *) cpy_ગાંઠ.kernel_params;
    જો (ok && !with_fence && kparams->total_elems == 0) {
        પાછા;
    }
    આ->enqueue_op(cpy_ગાંઠ);
}

શૂન્ય ggml_hexagon_session::enqueue_fence(અચલ ggml_tensor * sync_tensor, uint32_t fence_seq, તાર્કિક wait) {
    htp_opnode sync_ગાંઠ(HTP_OP_FENCE);

    ggml_tensor* ગાંઠ = sync_ગાંઠ.add_dummy(*sync_tensor);
    ગાંઠ->op           = GGML_OP_NONE;
    ગાંઠ->src[0]       = ગાંઠ;
    ગાંઠ->op_params[0] = (int32_t) fence_seq;
    ગાંઠ->op_params[1] = wait ? 0 : 1;

    sync_ગાંઠ.init(ગાંઠ);
    sync_ગાંઠ.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    આ->enqueue_op(sync_ગાંઠ);
}

સ્થિર તાર્કિક ggml_hexagon_precompute_allreduce_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    તાર્કિક has_add,
    તાર્કિક is_row_bcast,
    તાર્કિક is_shard_ok,
    સંરચના htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    અચલ uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    અચલ uint32_t elem_size = (dst->type == GGML_TYPE_F16) ? કદ(ggml_fp16_t) : કદ(દશાંશ);
    અચલ તાર્કિક is_contiguous = ggml_is_contiguous(dst);

    અચલ uint32_t ne0 = (uint32_t) dst->ne[0];
    અચલ uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    અચલ તાર્કિક use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    જો (has_add) {
        // sharded reduce-scatter for contiguous in-place add
        જો (opt_ar_scatter && is_shard_ok && use_1d && !is_row_bcast) {
            અચલ uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            અચલ uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            અચલ uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_start = (int32_t) rank_elem_start;
            kparams->rank_nelem      = (int32_t) (rank_elem_end - rank_elem_start);
        } નહીંતર જો (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } નહીંતર {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } નહીંતર {
        kparams->n_dsts = (int32_t) n_ranks;
        જો (use_1d) {
            અચલ uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            અચલ uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            અચલ uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            અચલ uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } નહીંતર {
            અચલ uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            અચલ uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            અચલ uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            અચલ uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    જો (use_1d) {
        અચલ uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        અચલ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        અચલ size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_ગણતરી(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        જો (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        જ્યારે ((size_t) kparams->vtcm_size > sess->vtcm_size && block_elems > 128) {
            અચલ size_t max_bytes_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((size_t) (max_bytes_per_buf / elem_size), 128);
            જો (block_elems < 128) તોડો;
            kparams->block_elems          = block_elems;
            kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
        }

        જો (sess->vtcm_size < (size_t) kparams->vtcm_size || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            પાછા ખોટું;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        પાછા સાચું;
    } નહીંતર {
        અચલ uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        અચલ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        અચલ size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_ગણતરી(n_ranks, n_threads, has_add, is_row_bcast);

        અચલ uint32_t row_bytes = ne0 * elem_size;
        અચલ uint32_t row_size_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_size_aligned = row_size_aligned;

        અચલ uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        જ્યારે ((size_t) kparams->vtcm_size > sess->vtcm_size && block_rows > 1) {
            અચલ size_t max_rows_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2 * row_size_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
            જો (max_rows_per_buf == 0) તોડો;
        }

        જો (sess->vtcm_size < (size_t) kparams->vtcm_size || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            પાછા ખોટું;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        પાછા સાચું;
    }
}

શૂન્ય ggml_hexagon_session::enqueue_allreduce(
    અચલ ggml_tensor * dst,
    અચલ std::vector<અચલ ggml_tensor *> & src_tensors,
    અચલ std::vector<અચલ ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_ગાંઠ(HTP_OP_ALLREDUCE);

    ggml_tensor* ગાંઠ = ar_ગાંઠ.add_dummy(*dst);
    ગાંઠ->op           = GGML_OP_NONE;
    ગાંઠ->op_params[0] = (int32_t) fence_seq_entry;
    ગાંઠ->op_params[1] = (int32_t) fence_seq_exit;

    ar_ગાંઠ.init(ગાંઠ);

    ar_ગાંઠ.inputs.clear();
    માટે (size_t i = 0; i < src_tensors.size(); i++) {
        ar_ગાંઠ.inputs.push_back(src_tensors[i]);
    }
    માટે (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_ગાંઠ.inputs.push_back(ar_ગાંઠ.add_dummy(*sync_tensors[i]));
    }

    ar_ગાંઠ.outputs.clear();
    માટે (size_t i = 0; i < src_tensors.size(); i++) {
        ar_ગાંઠ.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        આ, dst, rank, n_ranks, ખોટું, ખોટું, /*is_shard_ok=*/ ખોટું,
        (સંરચના htp_allreduce_kernel_params *) ar_ગાંઠ.kernel_params);

    ar_ગાંઠ.name = "ALLREDUCE";
    આ->enqueue_op(ar_ગાંઠ);
}

ggml_hexagon_shared_buffer * ggml_hexagon_session::mmap_tensor(અચલ ggml_tensor * t) {
    જો (!t) પાછા nullptr;

    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
    જો (!sbuf->mapped) {
        અચલ તાર્કિક is_weight = ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        અચલ તાર્કિક extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    પાછા sbuf;
}

તાર્કિક ggml_hexagon_session::clone_buffer(અચલ ggml_hexagon_shared_buffer * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    જો (sbuf->sess == આ) પાછા સાચું;

    auto mem = sbuf->mem;
    પૂર્ણાંક   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    જો (આ->cloned_buffers.find(fd) != આ->cloned_buffers.end()) પાછા સાચું;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", આ->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    auto clone = std::make_unique<ggml_hexagon_shared_buffer>(આ, *sbuf);
    પ્રયાસ {
        clone->mmap(sbuf->extended);
    } પકડો (અચલ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", આ->c_name(), exc.what());
        પાછા ખોટું;
    }

    આ->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(આ);
    પાછા સાચું;
}

શૂન્ય ggml_hexagon_session::release_buffer(અચલ ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    પૂર્ણાંક   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    auto it = આ->cloned_buffers.find(fd);
    જો (it != આ->cloned_buffers.end()) {
        auto clone = std::move(it->second);
        આ->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(આ);
}

શૂન્ય ggml_hexagon_session::unclone_buffer(અચલ ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    માટે (auto * sess : sessions) {
        sess->release_buffer(sbuf);
    }
}

સ્થિર size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    અચલ size_t MiB = 1024 * 1024;
    અચલ size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t step = 256u * MiB;

    પ્રયાસ {
        sbufs.push_back(નવું ggml_hexagon_shared_buffer(sess, GiB, સાચું)); vmem += GiB;
        sbufs.push_back(નવું ggml_hexagon_shared_buffer(sess, GiB, સાચું)); vmem += GiB;
        sbufs.push_back(નવું ggml_hexagon_shared_buffer(sess, GiB, સાચું)); vmem += GiB;

        જ્યારે (1) {
            sbufs.push_back(નવું ggml_hexagon_shared_buffer(sess, step, સાચું));
            vmem += step;
        }
    } પકડો (...) { }

    માટે (auto b : sbufs) { delete b; }

    પાછા vmem - step; // backoff to account for overhead from internal mappings
}

શૂન્ય ggml_hexagon_session::allocate(અચલ ggml_hexagon_device_config & config) noexcept(ખોટું) {
    પૂર્ણાંક phys_idx = config.physical_idx;
    પૂર્ણાંક virt_idx = config.virtual_idx;

    આ->valid_session = ખોટું;
    આ->valid_handle  = ખોટું;
    આ->valid_queue   = ખોટું;
    આ->valid_iface   = ખોટું;

    આ->name          = config.name;
    આ->phys_idx      = phys_idx;
    આ->virt_idx      = virt_idx;
    આ->domain_id     = config.domain_id;
    આ->session_id    = 0;
    આ->batch_req_seq = 0;
    આ->batch_rsp_seq = 0;
    આ->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", આ->name.c_str(), આ->domain_id, phys_idx, virt_idx);

    જો (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        ફેંકો std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    અચલ std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    જો (virt_idx > 0) {
        સંરચના remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<અક્ષર *>(dom_name.c_str());
        n.session_name     = const_cast<અક્ષર *>(આ->name.c_str());
        n.session_name_len = આ->name.size();
        n.session_id       = virt_idx;

        પૂર્ણાંક err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (શૂન્ય *) &n, કદ(n));
        જો (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           આ->c_name(), phys_idx, virt_idx, err);
            ફેંકો std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        આ->session_id    = n.session_id;
        આ->domain_id     = n.effective_domain_id;
        આ->valid_session = સાચું;
    } નહીંતર {
        સંરચના remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<અક્ષર *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        પૂર્ણાંક err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (શૂન્ય *) &eff, કદ(eff));
        જો (err == AEE_SUCCESS) {
            આ->domain_id = eff.effective_domain_id;
        } નહીંતર {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           આ->name.c_str(), err, આ->domain_id);
        }
    }

    // Enable unsigned modules
    {
        સંરચના remote_rpc_control_unsigned_module u;
        u.domain = આ->domain_id;
        u.enable = 1;
        પૂર્ણાંક err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (શૂન્ય *) &u, કદ(u));
        જો (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", આ->c_name(), err);
            ફેંકો std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    અક્ષર session_uri[256];
    {
        અક્ષર htp_uri[256];
        snprintf(htp_uri, કદ(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        સંરચના remote_rpc_get_uri u = {};
        u.session_id      = આ->session_id;
        u.domain_name     = const_cast<અક્ષર *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<અક્ષર *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = કદ(session_uri);

        પૂર્ણાંક err = remote_session_control(FASTRPC_GET_URI, (શૂન્ય *) &u, કદ(u));
        જો (err != AEE_SUCCESS) {
            snprintf(session_uri, કદ(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), આ->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          આ->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    પૂર્ણાંક err = htp_iface_open(session_uri, &આ->handle);
    જો (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", આ->c_name(), session_uri, err);
        ફેંકો std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    આ->valid_handle = સાચું;

    // Query HW info and resolve session options
    {
        unsigned પૂર્ણાંક hw_n_threads = 0;
        unsigned પૂર્ણાંક hw_n_hvx     = 0;
        unsigned પૂર્ણાંક hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        પૂર્ણાંક hw_err = htp_iface_hwinfo(આ->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        જો (hw_err == 0) {
            અચલ uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            આ->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_threads) : max_n_threads;
            આ->n_hvx     = આ->n_threads;
            આ->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            આ->vtcm_size = (uint64_t)hw_vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          આ->c_name(), આ->n_threads, આ->n_hvx, આ->n_hmx,
                          (unsigned long long)(આ->vtcm_size / (1024 * 1024)));
        } નહીંતર {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", આ->c_name(), hw_err);
            અચલ uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            આ->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_threads;
            આ->n_hvx     = આ->n_threads;
            આ->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            આ->vtcm_size = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        સંરચના remote_rpc_control_latency l;
        l.enable = 1;

        પૂર્ણાંક err = remote_handle64_control(આ->handle, DSPRPC_CONTROL_LATENCY, (શૂન્ય *) &l, કદ(l));
        જો (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", આ->c_name(),
                  આ->session_id, આ->domain_id, session_uri, (unsigned long) આ->handle);

    અચલ size_t req_q_size = (કદ(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    અચલ size_t rsp_q_size = (કદ(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(આ->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (શૂન્ય *) આ,  // Callback context
                          &queue);
    જો (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", આ->name.c_str(), (unsigned) err);
        ફેંકો std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    આ->valid_queue = સાચું;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &આ->queue_id);
    જો (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        ફેંકો std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    જો (opt_etm) {
        err = htp_iface_etm(આ->handle, 1);
        જો (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    આ->op_queue = નવું ggml_hexagon_opqueue(આ, opt_opbatch, opt_opqueue);

    આ->fence_buf = નવું ggml_hexagon_fence_buffer(આ, &dev_ctx->fence_buffer_type, 64 * 1024);
    જો (આ->mdev.count > 1) {
        આ->mdev_fence_slot = આ->alloc_fence(આ->mdev.count);
    }

    જો (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(આ);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", આ->c_name(), opt_vmem);
    }
    અચલ size_t shm_size = આ->op_queue->shm_size();
    આ->max_vmem = (opt_vmem > shm_size) ? (opt_vmem - shm_size) : opt_vmem;

    આ->op_batch = નવું ggml_hexagon_opbatch(આ, opt_opbatch, આ->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(આ->handle, આ->session_id, આ->queue_id, આ->n_threads, opt_nhmx, આ->max_vmem, (uint32_t) opt_power_mode);
    જો (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", આ->c_name(), (unsigned) err);
        ફેંકો std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    આ->valid_iface = સાચું;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", આ->c_name(), htp_power_mode_to_str(opt_power_mode));

    જો (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(આ->handle, opt_profile, &pmu_conf);
        જો (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }
}

શૂન્ય ggml_hexagon_session::release() noexcept(સાચું) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", આ->name.c_str());

    આ->mdev.sessions.clear();

    પૂર્ણાંક err;

    જો (આ->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(આ->handle);
        જો (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    delete આ->op_batch;
    delete આ->op_queue;
    માટે (auto & it : આ->cpy_fence_slots) {
        free_fence((શૂન્ય *) it.second, 1);
    }
    આ->cpy_fence_slots.clear();

    જો (આ->fence_buf) {
        unclone_buffer(આ->fence_buf);
        delete આ->fence_buf;
        આ->fence_buf = nullptr;
    }
    જ્યારે (!આ->cloned_buffers.empty()) {
        release_buffer(આ->cloned_buffers.begin()->second.get());
    }

    જો (opt_etm) {
        err = htp_iface_etm(આ->handle, 0);
        જો (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    જો (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(આ->handle, 0, &pmu_conf);
        જો (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    જો (આ->valid_queue) {
        err = dspqueue_close(queue);
        જો (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (unsigned) err);
        }
    }

    જો (આ->valid_handle) {
        htp_iface_close(આ->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(અચલ ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_ગણતરી) noexcept(ખોટું) {
    આ->dev        = dev;
    આ->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    આ->mdev.idx   = mdev_idx;
    આ->mdev.count = mdev_ગણતરી > 0 ? mdev_ગણતરી : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)આ) & 0xFFFF;

    પ્રયાસ {
        allocate(config);
        જો (આ->mdev.idx == 0 && !config.mdev_group.empty()) {
            માટે (size_t i = 0; i < config.mdev_group.size(); i++) {
                આ->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], આ->dev, (uint32_t) (i + 1), આ->mdev.count));
            }
        }
    } પકડો (અચલ std::exception & exc) {
        release();
        ફેંકો;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(સાચું) {
    release();
}

// ** backend interface

સ્થિર તાર્કિક ggml_hexagon_flash_attn_is_hmx_eligible(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * q,
    અચલ સંરચના ggml_tensor * k,
    અચલ સંરચના ggml_tensor * v,
    અચલ સંરચના ggml_tensor * sinks
) {
    જો (sess->n_hmx == 0) {
        પાછા ખોટું;
    }

    જો (opt_fa_select < 2) {
        પાછા ખોટું;
    }

    જો ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        પાછા ખોટું;
    }

    અચલ uint32_t DK = q->ne[0];
    અચલ uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    જો (DK % 8 != 0 || DV % 8 != 0) {
        પાછા ખોટું;
    }

    GGML_UNUSED(sinks);

    // Explicit force mode
    જો (opt_fa_select > 2) {
        પાછા સાચું;
    }

    અચલ uint32_t M = q->ne[1];

    // Prefill or batched decode
    જો (M > 1) {
        પાછા સાચું;
    }

    અચલ uint32_t n_head = q->ne[2];
    અચલ uint32_t n_kv_heads = k->ne[2];
    અચલ uint32_t G = n_kv_heads > 0 ? n_head / n_kv_heads : 1;
    અચલ uint32_t S = k->ne[1];
    અચલ uint32_t total_threads = sess->n_threads * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    જો (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < total_threads))) {
        પાછા સાચું;
    }

    // Tile alignment for 32-row HMX tiles
    અચલ તાર્કિક is_tile_aligned = (G > 0 && (32 % G == 0));
    જો (!is_tile_aligned) {
        જો (sess->n_threads >= 6) {
            પાછા ખોટું;
        }
        પાછા S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    જો (DK == 256) {
        s_cross = 4096;
    } નહીંતર જો (DK <= 64) {
        s_cross = (sess->n_threads >= 8) ? 2048 : ((sess->n_threads >= 6) ? 768 : 512);
    } નહીંતર {
        s_cross = (sess->n_threads >= 8) ? 1024 : 512;
    }

    પાછા S >= s_cross;
}

સ્થિર તાર્કિક ggml_hexagon_precompute_flash_attn_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_fa_kernel_params * kparams
) {
    જો (opt_fa_select < 1) {
        પાછા ખોટું;
    }

    memset(kparams, 0, કદ(*kparams));

    અચલ સંરચના ggml_tensor * q    = op->src[0];
    અચલ સંરચના ggml_tensor * k    = op->src[1];
    અચલ સંરચના ggml_tensor * v    = op->src[2];
    અચલ સંરચના ggml_tensor * mask = op->src[3];

    અચલ uint32_t neq0 = q->ne[0];  // head_dim (DK)
    અચલ uint32_t neq1 = q->ne[1];  // n_tokens
    અચલ uint32_t neq2 = q->ne[2];  // n_heads

    અચલ uint32_t nek1 = k->ne[1];  // kv_len

    અચલ uint32_t nev0 = v->ne[0];  // head_dim (DV)

    અચલ uint32_t DK = neq0;
    અચલ uint32_t DV = nev0;

    અચલ uint32_t n_kv_heads = k->ne[2];
    અચલ uint32_t G          = neq2 / n_kv_heads;

    દશાંશ scale         = 1.0f;
    દશાંશ max_bias      = 0.0f;
    દશાંશ logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], કદ(દશાંશ));
    memcpy(&max_bias,      &op->op_params[1], કદ(દશાંશ));
    memcpy(&logit_softcap, &op->op_params[2], કદ(દશાંશ));

    જો (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->head_split = (opt_fa_head_split != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    અચલ uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    અચલ સંરચના ggml_tensor * sinks = op->src[4];
    જો (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        અચલ uint32_t DK_pad = hex_round_up(DK, 64);
        અચલ uint32_t DV_pad = hex_round_up(DV, 64);
        size_t Br = 0, Bc = 0;
        પૂર્ણાંક ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
        જો (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            જો (logit_softcap == 0.0f) {
                kparams->scale = scale * (દશાંશ) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } નહીંતર {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (દશાંશ) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);

            અચલ size_t row_vec_bytes = hex_align_up(Bc * કદ(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            અચલ size_t m_લીટી_bytes = hex_align_up(Bc * કદ(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_લીટી_bytes / કદ(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_કિંમતો(G);
            જો (mask) {
                kparams->src3_div2 = init_fastdiv_કિંમતો(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_કિંમતો(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            પાછા સાચું;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    અચલ size_t size_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    અચલ size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    અચલ size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), mask != nullptr, sinks != nullptr, n_head, sess->n_threads);

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_કિંમતો(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_કિંમતો(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_કિંમતો(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_કિંમતો(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_કિંમતો(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_કિંમતો(q->ne[3]/v->ne[3]);
    જો (mask) {
        kparams->src3_div2 = init_fastdiv_કિંમતો(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_કિંમતો(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_hexagon_supported_flash_attn_ext(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];
    અચલ સંરચના ggml_tensor * src2 = op->src[2];
    અચલ સંરચના ggml_tensor * src3 = op->src[3];
    અચલ સંરચના ggml_tensor * src4 = op->src[4];
    અચલ સંરચના ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    જો ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        પાછા ખોટું;
    }

    જો (src3 && src3->type != GGML_TYPE_F16) {  // mask
        પાછા ખોટું;
    }

    જો (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        પાછા ખોટું;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    જો (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        પાછા ખોટું;
    }

    સંરચના htp_fa_kernel_params kparams;
    જો (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        પાછા ખોટું;
    }

    જો ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        પાછા ખોટું;
    }

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_hexagon_supported_gated_delta_net(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    જો (opt_gdn_select < 1) {
        પાછા ખોટું;
    }

    અચલ સંરચના ggml_tensor * q     = op->src[0];
    અચલ સંરચના ggml_tensor * k     = op->src[1];
    અચલ સંરચના ggml_tensor * v     = op->src[2];
    અચલ સંરચના ggml_tensor * g     = op->src[3];
    અચલ સંરચના ggml_tensor * beta  = op->src[4];
    અચલ સંરચના ggml_tensor * state = op->src[5];
    અચલ સંરચના ggml_tensor * dst   = op;

    જો (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    જો (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        પાછા ખોટું;
    }

    અચલ int64_t S_v      = v->ne[0];
    અચલ int64_t H        = v->ne[1];
    અચલ int64_t n_tokens = v->ne[2];
    અચલ int64_t n_seqs   = v->ne[3];
    અચલ int64_t K        = ggml_get_op_params_i32(op, 0);

    જો (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        પાછા ખોટું;
    }
    જો (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        પાછા ખોટું;
    }
    જો ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        પાછા ખોટું;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    જો (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        પાછા ખોટું;
    }
    જો (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        પાછા ખોટું;
    }

    અચલ uint32_t total_rows = (uint32_t) (H * n_seqs);
    અચલ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    અચલ તાર્કિક can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    જો (can_use_hmx) {
        સંરચના htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        જો (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, સાચું, &n_heads_batch)) {
            પાછા ખોટું;
        }
    } નહીંતર {
        સંરચના htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_threads);
        જો (layout.total_bytes > sess->vtcm_size) {
            પાછા ખોટું;
        }
    }

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_hexagon_matmul_is_hmx_eligible(
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    તાર્કિક is_matmul_id,
    તાર્કિક is_batched
) {
    GGML_UNUSED(dst);

    જો (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        પાછા ખોટું;
    }

    અચલ પૂર્ણાંક ne00  = src0->ne[0];
    અચલ પૂર્ણાંક ne01  = src0->ne[1];
    અચલ પૂર્ણાંક ne11  = src1->ne[1];
    અચલ પૂર્ણાંક ne12  = src1->ne[2];
    અચલ પૂર્ણાંક wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    જો (ne01 % HTP_TENSOR_TILE_N_ROWS != 0 && is_matmul_id) {
        પાછા ખોટું;
    }

    // HMX supports F16, F32, and repack quantized types.
    જો (!htp_is_hmx_weight_type((ggml_type) wtype)) {
        પાછા ખોટું;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    જો (ne00 % HTP_TENSOR_TILE_N_COLS != 0) {
        પાછા ખોટું;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    અચલ તાર્કિક is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    જો (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        પાછા ખોટું;
    }
    જો (!is_matmul_id && is_batched && is_tiled) {
        પાછા ખોટું;
    }

    // HMX assumes contiguous row-major layout.
    જો (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        પાછા ખોટું;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    અચલ પૂર્ણાંક m = is_matmul_id ? ne12 : ne11;
    જો (m <= HTP_MM_HMX_MIN_NROWS) {
        પાછા ખોટું;
    }

    પાછા સાચું;

    GGML_UNUSED(dst);
}

સ્થિર તાર્કિક ggml_hexagon_precompute_hmx_mm_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * act,
    અચલ સંરચના ggml_tensor * dst,
    પૂર્ણાંક wtype,
    પૂર્ણાંક ne00_padded,
    પૂર્ણાંક ne01_padded,
    પૂર્ણાંક ne02,
    પૂર્ણાંક ne11,
    પૂર્ણાંક ne12,
    પૂર્ણાંક ne11_padded,
    તાર્કિક is_matmul_id,
    તાર્કિક is_batched,
    size_t src2_size,
    size_t vtcm_budget,
    સંરચના htp_mm_kernel_params * kparams
) {
    અચલ તાર્કિક is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    અચલ પૂર્ણાંક aligned_tile_size = is_tiled ? htp_tensor_get_tile_padded_size(wtype) : 0;
    અચલ પૂર્ણાંક n_threads = (પૂર્ણાંક)sess->n_threads;
    અચલ પૂર્ણાંક ne10 = act->ne[0];

    પૂર્ણાંક m_for_solver = ne11;
    પૂર્ણાંક m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    જો (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        અચલ તાર્કિક dst_row_split = dst ? ggml_hexagon_can_row_partition(dst) : સાચું;
        અચલ તાર્કિક act_row_split = ggml_hexagon_can_row_partition(act);
        જો (dst_row_split && act_row_split) {
            m_for_solver = (ne11 + (પૂર્ણાંક) sess->mdev.count - 1) / (પૂર્ણાંક) sess->mdev.count;
            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
        }
    }
    અચલ તાર્કિક pipeline = is_matmul_id ? ખોટું : htp_mm_hmx_pipeline(m_for_solver);

    અચલ તાર્કિક is_batched_val = is_matmul_id ? ખોટું : is_batched;
    અચલ પૂર્ણાંક group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    તાર્કિક use_grouped = ખોટું;
    પૂર્ણાંક act_threads_selected = 0;

    જો (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && group_size > 1) {
        // Try grouped path first
        જો (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = સાચું;
        }
    }

    જો (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        પૂર્ણાંક m_id_rows = 0;
        જો (dst && is_matmul_id) {
            અચલ પૂર્ણાંક n_experts = ne02 > 0 ? ne02 : 1;
            અચલ size_t total_expert_rows = (size_t) dst->ne[1] * dst->ne[2];
            પૂર્ણાંક m_per_expert = (પૂર્ણાંક) ((total_expert_rows + n_experts - 1) / n_experts);
            જો (sess->mdev.count > 1 && ggml_hexagon_can_row_partition(dst)) {
                m_per_expert = (m_per_expert + (પૂર્ણાંક) sess->mdev.count - 1) / (પૂર્ણાંક) sess->mdev.count;
            }
            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
        }
        અચલ uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
        જો (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            પાછા ખોટું;
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
    kparams->div_n_act_threads = init_fastdiv_કિંમતો(act_threads_selected);
    kparams->div_ne00_padded   = init_fastdiv_કિંમતો(ne00_padded);
    kparams->vtcm_act_size     = 0;
    kparams->vtcm_bias_size    = (int32_t) src2_size;
    kparams->vtcm_dst_size     = 0;

    જો (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } નહીંતર {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    પાછા સાચું;
}

સ્થિર શૂન્ય ggml_hexagon_precompute_hvx_mm_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    પૂર્ણાંક wtype,
    તાર્કિક is_tiled,
    પૂર્ણાંક ne02,
    પૂર્ણાંક ne03,
    પૂર્ણાંક ne10,
    પૂર્ણાંક ne11,
    પૂર્ણાંક ne12,
    પૂર્ણાંક ne13,
    તાર્કિક is_matmul_id,
    અચલ size_t src2_row_size,
    size_t vtcm_budget,
    સંરચના htp_mm_kernel_params * kparams
) {
    જો (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        પાછા;
    }

    kparams->n_hmx = 0;
    kparams->n_threads = sess->n_threads;

    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);

    અચલ પૂર્ણાંક src1_nrows = ne11 * ne12 * ne13;

    જો (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_size = htp_tensor_get_tile_size(wtype);
        kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);

        અચલ તાર્કિક k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        જો (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (પૂર્ણાંક) sess->n_threads)
                                  ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);

            સંરચના htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            માટે (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], kparams->act_row_size, 0, d, સાચું, ખોટું
                );
                જો (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    તોડો;
                }
            }
            જો (L.total_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                પાછા;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = 0;
            kparams->vtcm_dst_size  = L.dst_bytes;
            પાછા;
        } નહીંતર {
            તાર્કિક try_tiled = (k_align && opt_mm_select >= 1);
            જો (try_tiled) {
                kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                જો (src1_nrows < (પૂર્ણાંક) sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } નહીંતર {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                સંરચના htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                માટે (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, ખોટું, ખોટું
                    );
                    જો (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        તોડો;
                    }
                }

                uint32_t m_chunk = 0;
                જો (htp_mm_hvx_solve_vtcm_params(
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
                    પાછા;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            પાછા;
        }
    } નહીંતર જો (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        પાછા;
    } નહીંતર જો (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        સંરચના htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        જો (htp_mm_hvx_solve_vtcm_params(
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
            પાછા;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        પાછા;
    } નહીંતર જો (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        સંરચના htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        જો (htp_mm_hvx_solve_vtcm_params(
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
            પાછા;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        પાછા;
    } નહીંતર જો (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        સંરચના htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        જો (htp_mm_hvx_solve_vtcm_params(
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
            પાછા;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        પાછા;
    } નહીંતર {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        પાછા;
    }
}

સ્થિર શૂન્ય ggml_hexagon_precompute_matmul_params_impl(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    અચલ size_t src2_row_size,
    અચલ size_t src2_size,
    સંરચના htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));

    અચલ પૂર્ણાંક ne00 = src0->ne[0];
    અચલ પૂર્ણાંક ne01 = src0->ne[1];
    અચલ પૂર્ણાંક ne02 = src0->ne[2];
    અચલ પૂર્ણાંક ne03 = src0->ne[3];

    અચલ પૂર્ણાંક ne10 = src1->ne[0];
    અચલ પૂર્ણાંક ne11 = src1->ne[1];
    અચલ પૂર્ણાંક ne12 = src1->ne[2];
    અચલ પૂર્ણાંક ne13 = src1->ne[3];

    અચલ તાર્કિક is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    અચલ પૂર્ણાંક wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    અચલ પૂર્ણાંક ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    અચલ પૂર્ણાંક ne11_padded = hex_round_up(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    અચલ પૂર્ણાંક ne01_tiled  = hex_round_up(ne01, 32);

    અચલ તાર્કિક is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    અચલ તાર્કિક is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    અચલ size_t vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    તાર્કિક hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    જો (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        જો (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
            જાઓ finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_size, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_કિંમતો(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_કિંમતો(ne11);
    kparams->div_r2       = init_fastdiv_કિંમતો(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_કિંમતો(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_કિંમતો(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
સ્થિર તાર્કિક ggml_hexagon_rows_stride(અચલ int64_t * ne, અચલ size_t * nb, size_t * stride) {
    size_t s = 0, next = 0;
    માટે (પૂર્ણાંક i = 1; i < GGML_MAX_DIMS; i++) {
        જો (ne[i] == 1) ચાલુ;
        જો (s == 0) { s = nb[i]; next = s * ne[i]; ચાલુ; }
        જો (nb[i] != next) પાછા ખોટું;
        next *= ne[i];
    }
    *stride = s ? s : nb[1];
    પાછા સાચું;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
સ્થિર તાર્કિક ggml_hexagon_matmul_can_collapse(અચલ સંરચના ggml_tensor * src0, અચલ સંરચના ggml_tensor * src1, અચલ સંરચના ggml_tensor * dst) {
    size_t s1, sd;
    પાછા (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_size(src1->type) && ggml_hexagon_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_size(dst->type) && ggml_hexagon_rows_stride(dst->ne, dst->nb, &sd);
}

સ્થિર તાર્કિક ggml_hexagon_matmul_add_can_collapse(
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * src2,
    અચલ સંરચના ggml_tensor * dst
) {
    જો (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        પાછા ખોટું;
    }
    જો (!src2) {
        પાછા સાચું;
    }
    અચલ int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    જો (src2_nrows == 1) {
        પાછા સાચું;
    }
    size_t s2;
    પાછા src2->nb[0] == ggml_type_size(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_rows_stride(src2->ne, src2->nb, &s2);
}

સ્થિર ggml_tensor ggml_hexagon_tensor_collapse_rows(અચલ સંરચના ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    પાછા c;
}

સ્થિર શૂન્ય ggml_hexagon_precompute_matmul_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_mm_kernel_params * kparams
) {
    જો (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        અચલ ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        અચલ ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        પાછા;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

સ્થિર શૂન્ય ggml_hexagon_precompute_fused_matmul_add_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * src2,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_mm_kernel_params * kparams
) {
    જો (ggml_hexagon_matmul_add_can_collapse(src0, src1, src2, dst)) {
        અચલ ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        અચલ ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        અચલ ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        અચલ સંરચના ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        અચલ size_t src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
        kparams->collapse = 1;
        પાછા;
    }
    અચલ size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

સ્થિર તાર્કિક ggml_hexagon_precompute_દ્વિઅંકી_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    uint32_t op,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_દ્વિઅંકી_kernel_params * kparams,
    અચલ સંરચના ggml_tensor * src2
) {
    memset(kparams, 0, કદ(*kparams));

    અચલ size_t elem_size = ggml_type_size(src0->type);
    અચલ size_t src0_row_size = src0->ne[0] * elem_size;
    અચલ size_t src1_row_size = src1->ne[0] * elem_size;
    અચલ size_t dst_row_size  = dst->ne[0]  * elem_size;

    અચલ size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    અચલ size_t src1_row_size_aligned = hex_round_up(src1_row_size, 128);
    અચલ size_t dst_row_size_aligned  = hex_round_up(dst_row_size,  128);

    જો (!htp_દ્વિઅંકી_strides_safe(src0, src1, dst, elem_size)) {
        પાછા ખોટું;
    }

    અચલ તાર્કિક is_add_id = op == HTP_OP_ADD_ID;
    અચલ તાર્કિક is_scalar = !is_add_id && src1->ne[0] == 1;
    અચલ તાર્કિક is_row_bcast = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    અચલ તાર્કિક is_same_shape = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    અચલ તાર્કિક is_complex   = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    અચલ તાર્કિક is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    અચલ તાર્કિક is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    જો (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        અચલ uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        અચલ uint32_t n_threads = sess->n_threads;
        અચલ uint32_t max_chunk_elems = 32768 / elem_size;
        અચલ uint32_t min_chunk_elems = 256;
        અચલ uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
        અચલ uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        અચલ uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

        kparams->kernel_type           = HTP_દ્વિઅંકી_KERNEL_CHUNKED;
        kparams->n_threads             = n_threads;
        kparams->rows_per_buffer       = 1;
        kparams->src0_row_size_aligned = src0_row_size_aligned;
        kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
        kparams->dst_row_size_aligned  = dst_row_size_aligned;
        kparams->src1_size             = 0;
        kparams->chunk_size            = chunk_size;
        kparams->chunk_bytes           = chunk_bytes;
        kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

        સંરચના htp_દ્વિઅંકી_vtcm_layout L;
        htp_દ્વિઅંકી_vtcm_layout_build(&L, kparams, sess->vtcm_size);
        જો (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
            પાછા ખોટું;
        }

        kparams->vtcm_size = L.total_bytes;
        પાછા સાચું;
    }

    ગણના htp_દ્વિઅંકી_kernel_type kernel_type;
    size_t src1_size = 0;
    size_t src2_size = 0;

    જો (is_add_id) {
        GGML_ASSERT(src2 != nullptr);
        kernel_type = HTP_દ્વિઅંકી_KERNEL_ADD_ID;
        src1_size = hex_round_up(src1->ne[1] * src1_row_size_aligned, 128);
        src2_size = hex_round_up(ggml_nelements(src2) * કદ(int32_t), 128);
        જો (src1_size + src2_size > sess->vtcm_size) {
            પાછા ખોટું;
        }
    } નહીંતર જો (is_row_bcast) {
        kernel_type = HTP_દ્વિઅંકી_KERNEL_ROW_BCAST;
        src1_size = src1_row_size_aligned;
    } નહીંતર જો (is_scalar) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size, 128);
        જો (src1_size > sess->vtcm_size) {
            પાછા ખોટું;
        }
        kernel_type = HTP_દ્વિઅંકી_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } નહીંતર જો (is_same_shape) {
        kernel_type = HTP_દ્વિઅંકી_KERNEL_SAME_SHAPE;
    } નહીંતર જો (is_complex) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        જો (src1_size > sess->vtcm_size) {
            પાછા ખોટું;
        }
        kernel_type = HTP_દ્વિઅંકી_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } નહીંતર {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        જો (src1_size > sess->vtcm_size) {
            પાછા ખોટું;
        }
        kernel_type = HTP_દ્વિઅંકી_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_threads = sess->n_threads;
    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;
    kparams->src1_size             = src1_size;
    kparams->src2_size             = src2_size;

    અચલ તાર્કિક src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    અચલ તાર્કિક src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    અચલ તાર્કિક dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    અચલ તાર્કિક src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    અચલ તાર્કિક src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    અચલ તાર્કિક dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    તાર્કિક split_at_ne01 = ખોટું;
    જો (src0->ne[2] > 1) {
        જો (kernel_type == HTP_દ્વિઅંકી_KERNEL_SAME_SHAPE) {
            split_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } નહીંતર જો (kernel_type == HTP_દ્વિઅંકી_KERNEL_ROW_BCAST) {
            split_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } નહીંતર {
            split_at_ne01 = સાચું;
        }
    }
    kparams->split_at_ne01 = split_at_ne01 ? 1 : 0;

    તાર્કિક split_at_ne02 = ખોટું;
    જો (src0->ne[3] > 1) {
        જો (kernel_type == HTP_દ્વિઅંકી_KERNEL_SAME_SHAPE) {
            split_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } નહીંતર જો (kernel_type == HTP_દ્વિઅંકી_KERNEL_ROW_BCAST) {
            split_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } નહીંતર {
            split_at_ne02 = સાચું;
        }
    }
    kparams->split_at_ne02 = split_at_ne02 ? 1 : 0;

    સંરચના htp_દ્વિઅંકી_vtcm_layout L;
    htp_દ્વિઅંકી_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    જો (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {
        જો (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            અચલ uint32_t total_elems = (uint32_t) ggml_nelements(src0);
            અચલ uint32_t n_threads = sess->n_threads;
            અચલ uint32_t max_chunk_elems = 32768 / elem_size;
            અચલ uint32_t min_chunk_elems = 256;
            અચલ uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
            અચલ uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
            અચલ uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

            kparams->kernel_type           = HTP_દ્વિઅંકી_KERNEL_CHUNKED;
            kparams->n_threads             = n_threads;
            kparams->rows_per_buffer       = 1;
            kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
            kparams->src1_size             = 0;
            kparams->chunk_size            = chunk_size;
            kparams->chunk_bytes           = chunk_bytes;
            kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

            htp_દ્વિઅંકી_vtcm_layout_build(&L, kparams, sess->vtcm_size);
            જો (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
                પાછા ખોટું;
            }

            kparams->vtcm_size = L.total_bytes;
            પાછા સાચું;
        }
        પાછા ખોટું;
    }

    kparams->rows_per_buffer = L.rows_per_buffer;
    kparams->vtcm_size       = L.total_bytes;

    પાછા સાચું;
}

સ્થિર શૂન્ય ggml_hexagon_precompute_unary_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    uint32_t op,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));

    અચલ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    અચલ uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    અચલ size_t elem_size = ggml_type_size(src0->type);

    અચલ size_t src0_data_row_size = src0->ne[0] * elem_size;
    અચલ size_t dst_data_row_size  = dst->ne[0]  * ggml_type_size(dst->type);

    અચલ size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    અચલ size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = 0;
    તાર્કિક broadcast_weight = ખોટું;

    જો (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_data_row_size = src1->ne[0] * ggml_type_size(src1->type);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->broadcast_weight      = broadcast_weight;

    સંરચના htp_unary_vtcm_layout L;
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

    અચલ uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_કિંમતો(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_કિંમતો(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_કિંમતો(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_કિંમતો(tiles_per_row);
}

સ્થિર શૂન્ય ggml_hexagon_precompute_get_rows_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_get_rows_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));

    અચલ uint32_t ne00 = src0->ne[0];
    અચલ uint32_t ne02 = src0->ne[2];
    અચલ uint32_t ne03 = src0->ne[3];

    અચલ uint32_t ne10 = src1->ne[0];
    અચલ uint32_t ne11 = src1->ne[1];
    અચલ uint32_t ne12 = src1->ne[2];
    અચલ uint32_t nr = ne10 * ne11 * ne12;

    અચલ તાર્કિક tiled = ggml_hexagon_tensor_is_tiled(src0);
    અચલ પૂર્ણાંક wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    જો (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } નહીંતર જો (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } નહીંતર {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    અચલ uint32_t total_tasks = nr;

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, total_tasks);

    સંરચના htp_get_rows_vtcm_layout vtcm_layout = {};
    જ્યારે (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, kparams->n_threads);
        જો (vtcm_layout.total_bytes <= sess->vtcm_size) {
            તોડો;
        }
        --kparams->n_threads;
    }

    જો (kparams->n_threads == 0 && total_tasks > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, 1);
    }

    kparams->vtcm_size = (total_tasks == 0) ? 0 : vtcm_layout.total_bytes;
    kparams->tasks_per_thread = kparams->n_threads > 0 ? (total_tasks + kparams->n_threads - 1) / kparams->n_threads : 0;
    kparams->total_tasks = total_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_કિંમતો(ne10) : fastdiv_કિંમતો{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_કિંમતો(ne10 * ne11) : fastdiv_કિંમતો{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_કિંમતો(ne02) : fastdiv_કિંમતો{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_કિંમતો(ne03) : fastdiv_કિંમતો{0, 0};
}

સ્થિર શૂન્ય ggml_hexagon_precompute_set_rows_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0, //કિંમતો
    અચલ સંરચના ggml_tensor * src1, // indices
    અચલ સંરચના ggml_tensor * dst,  // destination
    સંરચના htp_set_rows_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));

    અચલ uint32_t nr = src0->ne[1];

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, nr);
    kparams->tasks_per_thread = (nr + kparams->n_threads - 1) / kparams->n_threads;
    kparams->total_tasks = nr;

    kparams->div_ne11 = init_fastdiv_કિંમતો(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_કિંમતો(src1->ne[2]);
    kparams->div_tasks_per_thread = init_fastdiv_કિંમતો(kparams->tasks_per_thread);
    kparams->div_ne02 = init_fastdiv_કિંમતો(src0->ne[2]);

    સંરચના htp_set_rows_vtcm_layout vtcm_layout;
    htp_set_rows_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_threads);
    kparams->vtcm_size = vtcm_layout.total_bytes;
}

સ્થિર શૂન્ય ggml_hexagon_precompute_softmax_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));

    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];

    અચલ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    અચલ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);

    દશાંશ scale = 1.0f;
    દશાંશ max_bias = 0.0f;
    memcpy(&scale,    &op->op_params[0], કદ(દશાંશ));
    memcpy(&max_bias, &op->op_params[1], કદ(દશાંશ));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    અચલ uint32_t n_head = src0->ne[2];
    અચલ uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    kparams->n_head      = n_head;
    kparams->n_head_log2 = n_head_log2;

    જો (max_bias > 0.0f && n_head_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_head_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    } નહીંતર {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    અચલ uint32_t ne00 = src0->ne[0];
    અચલ uint32_t ne10 = src1 ? src1->ne[0] : 1;

    સંરચના htp_softmax_vtcm_layout layout;
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
    જો (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } નહીંતર જો (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F16;
    } નહીંતર {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    જો (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_કિંમતો(src0->ne[1]);
    જો (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_કિંમતો(src0->ne[2]);
    અચલ uint32_t ne12 = src1 ? src1->ne[2] : 1;
    અચલ uint32_t ne13 = src1 ? src1->ne[3] : 1;
    જો (ne12 > 0) kparams->div_ne12 = init_fastdiv_કિંમતો(ne12);
    જો (ne13 > 0) kparams->div_ne13 = init_fastdiv_કિંમતો(ne13);
}

સ્થિર શૂન્ય ggml_hexagon_precompute_rope_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));

    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src2 = op->src[2];
    અચલ સંરચના ggml_tensor * dst  = op;

    અચલ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    અચલ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    અચલ uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    સંરચના htp_rope_vtcm_layout layout;
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

    જો (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_કિંમતો(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_કિંમતો(dst->ne[1]);
    }
}

સ્થિર શૂન્ય ggml_hexagon_precompute_ssm_conv_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * src1,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));

    અચલ uint32_t d_conv  = (uint32_t) src1->ne[0];
    અચલ uint32_t d_inner = (uint32_t) src0->ne[1];
    અચલ uint32_t n_t     = (uint32_t) dst->ne[1];
    અચલ uint32_t n_s     = (uint32_t) dst->ne[2];
    અચલ uint32_t ncs     = (uint32_t) src0->ne[0];

    અચલ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    અચલ uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
    અચલ uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);

    અચલ uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * કદ(દશાંશ), 128) + 128;
    અચલ uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * કદ(દશાંશ), 128);
    અચલ uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

    uint32_t vtcm_src0_per_thread = 0;
    uint32_t vtcm_dst_per_thread  = 0;

    જો (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;

        અચલ uint32_t src0_tile_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * કદ(દશાંશ), 128);
        અચલ uint32_t src0_T_bytes        = hex_round_up(d_conv * d_inner_per_thread * કદ(દશાંશ), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw_bytes + src0_T_bytes;

        અચલ uint32_t dst_tile_bytes = hex_round_up(d_inner_per_thread * કદ(દશાંશ), 128);
        vtcm_dst_per_thread = 2 * dst_tile_bytes;
    } નહીંતર {
        અચલ size_t vtcm_budget = (sess->vtcm_size > 0 ? sess->vtcm_size / n_threads : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        અચલ uint32_t src0_block_T    = hex_round_up(ncs * 32 * કદ(દશાંશ), 128);
        અચલ size_t   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
        અચલ size_t   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);

        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
        target_max_tile = (std::max)(target_max_tile, 32u);
        target_max_tile = (std::min)(target_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * કદ(દશાંશ)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        જો (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        જો (d_inner_tile > target_max_tile) {
            d_inner_tile = target_max_tile;
        }
        જો (d_inner_tile > d_inner_per_thread) {
            d_inner_tile = d_inner_per_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        અચલ uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * કદ(દશાંશ), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw + src0_block_T;

        vtcm_dst_per_thread = 2 * hex_round_up(d_inner_tile * n_t * કદ(દશાંશ), 128);
    }

    kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
    kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
    kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;

    kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
    kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
    kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
    kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
}

સ્થિર શૂન્ય ggml_hexagon_precompute_gated_delta_net_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));

    અચલ સંરચના ggml_tensor * q     = op->src[0];
    અચલ સંરચના ggml_tensor * k     = op->src[1];
    અચલ સંરચના ggml_tensor * v     = op->src[2];
    અચલ સંરચના ggml_tensor * g     = op->src[3];
    અચલ સંરચના ggml_tensor * state = op->src[5];

    અચલ uint32_t S_v      = (uint32_t) v->ne[0];
    અચલ uint32_t H        = (uint32_t) v->ne[1];
    અચલ uint32_t n_tokens = (uint32_t) v->ne[2];
    અચલ uint32_t n_seqs   = (uint32_t) v->ne[3];
    અચલ uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    અચલ uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    અચલ uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    અચલ uint32_t total_rows = H * n_seqs;
    અચલ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    અચલ તાર્કિક can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    સંરચના htp_gdn_hmx_vtcm_layout hmx_layout;
    સંરચના htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    જો (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, સાચું, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_size      = HTP_GDN_CHUNK_SIZE;
        kparams->n_chunks        = (n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_size       = (uint32_t) hmx_layout.total_bytes;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_bytes;
        kparams->vtcm_per_thread = (uint32_t) (hmx_layout.total_bytes / (n_threads > 0 ? n_threads : 1));
    } નહીંતર {
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
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / કદ(દશાંશ));
    kparams->state_size_per_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((દશાંશ) S_v);

    જો (H > 0)                  kparams->div_H         = init_fastdiv_કિંમતો(H);
    જો (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_કિંમતો((uint32_t) q->ne[1]);
    જો (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_કિંમતો((uint32_t) k->ne[1]);
    જો (rq3 > 0)                kparams->div_rq3       = init_fastdiv_કિંમતો(rq3);
    જો (rk3 > 0)                kparams->div_rk3       = init_fastdiv_કિંમતો(rk3);
    જો (kparams->n_threads > 0) kparams->div_n_threads = init_fastdiv_કિંમતો(kparams->n_threads);
}

સ્થિર શૂન્ય ggml_hexagon_precompute_sort_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    તાર્કિક is_ટોચ_k,
    સંરચના htp_sort_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));

    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * dst  = op;

    અચલ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    અચલ uint32_t ne00       = src0->ne[0];
    અચલ uint32_t k          = dst->ne[0];

    int32_t order = GGML_SORT_ORDER_DESC;
    જો (!is_ટોચ_k) {
        order = ((અચલ int32_t *) op->op_params)[0];
    }

    અચલ uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    અચલ size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    સંરચના htp_sort_vtcm_layout layout;
    તાર્કિક ok = htp_sort_solve_layout(&layout, ne00, total_rows, k, n_threads_max, vtcm_budget, is_ટોચ_k);
    GGML_ASSERT(ok);

    kparams->n_threads         = (int32_t) layout.n_threads;
    kparams->total_rows        = (int32_t) total_rows;
    kparams->row_start         = 0;
    kparams->row_end           = (int32_t) total_rows;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->order             = order;
    kparams->is_ટોચ_k          = is_ટોચ_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_size         = (int32_t) layout.total_bytes;
    kparams->phase1_slot_size  = (int32_t) layout.phase1_slot_size;
    kparams->merge_કિંમતો_off  = (int32_t) layout.merge_કિંમતો_off;
    kparams->merge_indices_off = (int32_t) layout.merge_indices_off;
    kparams->merge_elems       = (int32_t) layout.merge_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

સ્થિર શૂન્ય ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    અચલ uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    અચલ int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    જો (numer_hi < 0) {
        *ox_hi = 0;
    } નહીંતર {
        અચલ uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    જો (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

સ્થિર તાર્કિક ggml_hexagon_pool_shape_fits(
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * dst
) {
    પાછા ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

સ્થિર શૂન્ય ggml_hexagon_precompute_pool_2d_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0,
    અચલ સંરચના ggml_tensor * dst,
    સંરચના htp_pool_2d_kernel_params * kparams,
    તાર્કિક is_pool_1d
) {
    memset(kparams, 0, કદ(*kparams));

    અચલ uint32_t src_x = (uint32_t) src0->ne[0];
    અચલ uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    અચલ uint32_t dst_x = (uint32_t) dst->ne[0];
    અચલ uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    અચલ uint32_t src_plane_bytes = src_x * src_y * કદ(દશાંશ);
    અચલ uint32_t dst_plane_bytes = dst_x * dst_y * કદ(દશાંશ);
    અચલ uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    અચલ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    સંરચના htp_pool_vtcm_layout layout;
    અચલ તાર્કિક ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, sess->vtcm_size);
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
    kparams->avg_divide_ગણતરી = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    અચલ તાર્કિક narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_path = (kparams->fast_path && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (દશાંશ) (kparams->kernel_x * kparams->kernel_y);
}

સ્થિર તાર્કિક ggml_hexagon_precompute_concat_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (!src0 || !src1 || !dst) {
        પાછા ખોટું;
    }

    પૂર્ણાંક dim = ((અચલ int32_t *) op->op_params)[0];
    જો (dim < 0 || dim >= GGML_MAX_DIMS) {
        પાછા ખોટું;
    }
    kparams->dim = dim;

    જો (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        પાછા ખોટું;
    }
    જો (src0->type != dst->type || src1->type != dst->type) {
        પાછા ખોટું;
    }

    અચલ uint32_t type_size = ggml_type_size(dst->type);

    માટે (પૂર્ણાંક d = 0; d < GGML_MAX_DIMS; d++) {
        અચલ int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        જો (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            પાછા ખોટું;
        }
    }

    અચલ uint32_t max_gather_ne0 = 128 / type_size;
    અચલ uint32_t total_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    અચલ તાર્કિક b_dense = (src1->nb[0] == type_size);
    અચલ તાર્કિક gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_size &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_size
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_size);

    જો (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_size && dst->nb[1] == (size_t) dst->ne[0] * type_size &&
        src0->nb[0] == type_size &&
        (src1->nb[0] == type_size || src1->nb[1] == type_size) &&
        gather_strides_ok &&
        htp_gather_rows_tensor_outer_contiguous(dst) &&
        htp_gather_rows_tensor_outer_contiguous(src0) &&
        htp_gather_rows_tensor_outer_contiguous(src1)) {

        જો (htp_gather_rows_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_size, total_rows,
                                         sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ROWS;
            kparams->n_threads   = (uint8_t) sess->n_threads;
            kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
            પાછા સાચું;
        }
    }

    અચલ તાર્કિક dma_strides_ok = (src0->nb[0] == type_size && src1->nb[0] == type_size && dst->nb[0] == type_size);

    જો (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_threads   = 1;
        પાછા સાચું;
    }

    અચલ તાર્કિક is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    અચલ તાર્કિક is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    અચલ તાર્કિક transposed_rows_ok = (src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size);

    જો (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        અચલ uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 8;
        સંરચના htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_size, n_threads);

        જો (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
            પાછા ખોટું;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_threads                         = n_threads;
        kparams->vtcm_size                         = layout.total_bytes;
        kparams->u.transposed.spad0_size_per_thread = layout.src0_spad_size_per_thread;
        kparams->u.transposed.spad1_size_per_thread = layout.src1_spad_size_per_thread;
        પાછા સાચું;
    }

    પાછા ખોટું;
}

સ્થિર તાર્કિક ggml_hexagon_precompute_cpy_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (!src0 || !dst) {
        પાછા ખોટું;
    }

    જો (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        પાછા ખોટું;
    }
    જો (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        પાછા ખોટું;
    }

    અચલ int64_t nelem_src = ggml_nelements(src0);
    અચલ int64_t nelem_dst = ggml_nelements(dst);
    જો (nelem_src != nelem_dst || nelem_src < 0) {
        પાછા ખોટું;
    }

    અચલ uint32_t src_type_size = ggml_type_size(src0->type);
    અચલ uint32_t dst_type_size = ggml_type_size(dst->type);

    kparams->src0_type_size = (uint8_t) src_type_size;
    kparams->dst_type_size  = (uint8_t) dst_type_size;
    kparams->total_elems    = (uint32_t) nelem_src;

    જો (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        પાછા સાચું;
    }

    જો (nelem_src == 1) {
        જો (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            પાછા સાચું;
        }
        જો ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            પાછા સાચું;
        }
        પાછા ખોટું;
    }

    અચલ તાર્કિક sametype = (src0->type == dst->type);

    તાર્કિક same_extents = સાચું;
    માટે (પૂર્ણાંક d = 0; d < GGML_MAX_DIMS; d++) {
        જો (src0->ne[d] != dst->ne[d]) {
            same_extents = ખોટું;
            તોડો;
        }
    }

    અચલ તાર્કિક transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_size) || (dst->nb[0] != dst_type_size) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_size) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_size);
    અચલ તાર્કિક sameshape  = same_extents && !transposed;

    અચલ તાર્કિક src_is_contiguous = ggml_is_contiguous(src0);
    અચલ તાર્કિક dst_is_contiguous = ggml_is_contiguous(dst);

    જો (sametype) {
        જો (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            પાછા સાચું;
        }

        અચલ uint32_t max_gather_ne0 = 128 / src_type_size;
        અચલ uint32_t total_rows = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        જો ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_size && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_size &&
            total_rows > 1 &&
            htp_gather_rows_tensor_outer_contiguous(src0)) {

            જો (htp_gather_rows_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, ખોટું,
                                             src_type_size, total_rows,
                                             sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ROWS;
                kparams->n_threads   = (uint8_t) sess->n_threads;
                kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
                પાછા સાચું;
            }
        }

        જો (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            પાછા સાચું;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_threads   = sess->n_threads > 0 ? (uint8_t) sess->n_threads : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_કિંમતો((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_કિંમતો((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_કિંમતો((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_કિંમતો((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_કિંમતો((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_કિંમતો((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        પાછા સાચું;
    }

    જો (!sameshape) {
        પાછા ખોટું;
    }

    અચલ તાર્કિક valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    જો (!valid_conversion) {
        પાછા ખોટું;
    }

    અચલ uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 4;
    સંરચના htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_size, dst_type_size, n_threads);

    જો (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
        પાછા ખોટું;
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
    kparams->u.convert.div_ne01      = init_fastdiv_કિંમતો((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_કિંમતો((uint32_t) (src0->ne[2] * src0->ne[1]));

    પાછા સાચું;
}

સ્થિર શૂન્ય ggml_hexagon_precompute_fused_mmnx_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0, // W0
    અચલ સંરચના ggml_tensor * act,  // x
    int32_t n_weights,
    સંરચના htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));
    kparams->n_threads = sess->n_threads;

    અચલ પૂર્ણાંક ne00 = src0->ne[0];
    અચલ પૂર્ણાંક ne01 = src0->ne[1];
    અચલ પૂર્ણાંક ne02 = src0->ne[2];
    અચલ પૂર્ણાંક ne03 = src0->ne[3];

    અચલ પૂર્ણાંક ne10 = act->ne[0];
    અચલ પૂર્ણાંક ne11 = act->ne[1];
    અચલ પૂર્ણાંક ne12 = act->ne[2];
    અચલ પૂર્ણાંક ne13 = act->ne[3];

    અચલ તાર્કિક is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    અચલ પૂર્ણાંક wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    અચલ પૂર્ણાંક ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    અચલ પૂર્ણાંક ne11_padded = hex_round_up(ne11, 32);
    અચલ પૂર્ણાંક ne01_tiled  = hex_round_up(ne01, 32);

    અચલ size_t vtcm_budget = sess->vtcm_size;
    અચલ તાર્કિક is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    તાર્કિક hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    જો (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, nullptr, ખોટું, is_batched)) {
        જો (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, ખોટું, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            જાઓ finalize;
        }
    }

    જો (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        પાછા;
    }

    {
        અચલ પૂર્ણાંક act_nrows = ne11 * ne12 * ne13;
        અચલ size_t act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
        અચલ size_t src0_row_size = src0->nb[1];

        અચલ uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        માટે (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            સંરચના htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
                0, src0_row_size, act_row_size, 0, d, ખોટું, સાચું
            );
            જો (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                તોડો;
            }
        }

        સંરચના htp_mm_hvx_vtcm_layout L;
        તાર્કિક try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
            0, src0_row_size, act_row_size, 0, best_n_prefetch, ખોટું, સાચું
        );

        જો (try_tiled && L.total_bytes <= sess->vtcm_size) {
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
        } નહીંતર {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            પાછા;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_કિંમતો(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_કિંમતો(ne11);
    kparams->div_r2       = init_fastdiv_કિંમતો(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_કિંમતો(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_કિંમતો(ne12);
}

સ્થિર શૂન્ય ggml_hexagon_precompute_fused_mmidnx_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * src0, // W0
    અચલ સંરચના ggml_tensor * act,  // x
    અચલ સંરચના ggml_tensor * dst,  // dst0
    int32_t n_weights,
    સંરચના htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

સ્થિર તાર્કિક ggml_hexagon_op_mul_mat_use_fwht(અચલ સંરચના ggml_tensor * dst) {
    જો (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        પાછા ખોટું;
    }

    અચલ સંરચના ggml_tensor * src0 = dst->src[0];
    અચલ સંરચના ggml_tensor * src1 = dst->src[1];

    અચલ int64_t n = src1->ne[0];
    જો (n < 32 || (n & (n - 1)) != 0) {
        પાછા ખોટું;
    }

    પાછા src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_rows(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

સ્થિર તાર્કિક ggml_hexagon_supported_mul_mat(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * dst) {
    અચલ સંરચના ggml_tensor * src0 = dst->src[0];
    અચલ સંરચના ggml_tensor * src1 = dst->src[1];

    જો (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        જો (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            પાછા ખોટું;
        }
        સંરચના htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, nullptr, dst, &ukparams);
        પાછા ukparams.vtcm_row_per_thread > 0 && (size_t) ukparams.vtcm_size <= sess->vtcm_size;
    }

    જો (dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    જો (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        પાછા ખોટું;
    }

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q2_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_IQ4_NL:
        સ્થિતિ GGML_TYPE_MXFP4:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q2_K: {
            જો (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                પાછા ખોટું;
            }

            અચલ int64_t qk = ggml_blck_size(src0->type);
            જો (src0->ne[0] % qk) {
                પાછા ખોટું;
            }

            જો (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                પાછા ખોટું;
            }
            જો (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                પાછા ખોટું;
            }
            તોડો;
        }

        સ્થિતિ GGML_TYPE_F16:
            જો (src0->nb[1] < src0->nb[0]) {
                પાછા ખોટું;
            }
            જો (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                પાછા ખોટું;
            }
            જો (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                પાછા ખોટું;
            }
            તોડો;

        સ્થિતિ GGML_TYPE_F32:
            જો (src1->type != GGML_TYPE_F32) {
                પાછા ખોટું;
            }
            જો (src0->nb[1] < src0->nb[0]) {
                પાછા ખોટું;
            }
            જો (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                પાછા ખોટું;
            }
            જો (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                પાછા ખોટું;
            }
            તોડો;

        સ્થિતિ GGML_TYPE_BF16:
            જો (src0->nb[1] < src0->nb[0]) {
                પાછા ખોટું;
            }
            જો (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                પાછા ખોટું;
            }
            જો (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                પાછા ખોટું;
            }
            તોડો;

        મૂળભૂત:
            પાછા ખોટું;
    }

    સંરચના htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    જો (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        પાછા ખોટું;
    }

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_hexagon_supported_mul_mat_id(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];
    અચલ સંરચના ggml_tensor * src2 = op->src[2];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        પાછા ખોટું;
    }

    જો (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        પાછા ખોટું;
    }

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q2_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_IQ4_NL:
        સ્થિતિ GGML_TYPE_MXFP4:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q2_K:
            જો (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                પાછા ખોટું;
            }

            જો (src0->ne[0] % ggml_blck_size(src0->type) != 0) {
                પાછા ખોટું;
            }
            તોડો;

        સ્થિતિ GGML_TYPE_F16:
        સ્થિતિ GGML_TYPE_F32:
        સ્થિતિ GGML_TYPE_BF16:
            જો (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                પાછા ખોટું;
            }
            જો (src0->ne[0] % 32 != 0) {
                પાછા ખોટું;
            }
            તોડો;

        મૂળભૂત:
            પાછા ખોટું;
    }

    સંરચના htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    જો (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        પાછા ખોટું;
    }

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_hexagon_supported_દ્વિઅંકી(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src0->type == GGML_TYPE_F32) {
        જો (src1->type != GGML_TYPE_F32) {
            પાછા ખોટું;
        }
        જો (dst->type != GGML_TYPE_F32) {
            પાછા ખોટું;
        }
    }
    નહીંતર જો (src0->type == GGML_TYPE_F16) {
        જો (src1->type != GGML_TYPE_F16) {
            પાછા ખોટું;
        }
        જો (dst->type != GGML_TYPE_F16) {
            પાછા ખોટું;
        }
    }
    નહીંતર {
        પાછા ખોટું;
    }

    જો (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        પાછા ખોટું;
    }
    જો (!ggml_are_same_shape(src0, dst)) {
        પાછા ખોટું;
    }
    જો (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        પાછા ખોટું;
    }

    સંરચના htp_દ્વિઅંકી_kernel_params kparams;
    પાછા ggml_hexagon_precompute_દ્વિઅંકી_params(sess, op_remap_to_htp(op), src0, src1, dst, &kparams);
}

સ્થિર તાર્કિક ggml_hexagon_supported_add_id(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];
    અચલ સંરચના ggml_tensor * src2 = op->src[2];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (!src2) {
        પાછા ખોટું;
    }
    જો (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        પાછા ખોટું;
    }
    જો (!ggml_are_same_shape(src0, dst)) {
        પાછા ખોટું;
    }
    જો (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        પાછા ખોટું;
    }
    જો (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        પાછા ખોટું;
    }
    જો (src0->nb[0] != કદ(દશાંશ) || src1->nb[0] != કદ(દશાંશ) || dst->nb[0] != કદ(દશાંશ) || src2->nb[0] != કદ(int32_t)) {
        પાછા ખોટું;
    }

    // REVISIT: add support for non-contiguous tensors
    જો (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        પાછા ખોટું;
    }
    જો (src2->nb[1] < src2->ne[0] * કદ(int32_t)) {
        પાછા ખોટું;
    }

    સંરચના htp_દ્વિઅંકી_kernel_params kparams;
    પાછા ggml_hexagon_precompute_દ્વિઅંકી_params(sess, HTP_OP_ADD_ID, src0, src1, dst, &kparams, src2);
}

સ્થિર તાર્કિક ggml_hexagon_supported_unary(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        પાછા ખોટું;
    }
    જો (dst->type != src0->type) {
        પાછા ખોટું;
    }
    જો (!ggml_is_contiguous_rows(src0)) {
        પાછા ખોટું;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    જો (src0->type == GGML_TYPE_F16) {
        પસંદગી (op->op) {
            સ્થિતિ GGML_OP_NORM:
            સ્થિતિ GGML_OP_RMS_NORM:
            સ્થિતિ GGML_OP_L2_NORM:
            સ્થિતિ GGML_OP_SCALE:
            સ્થિતિ GGML_OP_CLAMP:
            સ્થિતિ GGML_OP_SQR:
            સ્થિતિ GGML_OP_SQRT:
            સ્થિતિ GGML_OP_LOG:
                તોડો;
            સ્થિતિ GGML_OP_UNARY:
                જો (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_STEP) {
                    પાછા ખોટું;
                }
                તોડો;
            મૂળભૂત:
                પાછા ખોટું;
        }
    }

    જો (!ggml_are_same_shape(src0, dst)) {
        પાછા ખોટું;
    }

    // dst must be contiguous; src0 may be non-contiguous
    જો (!ggml_is_contiguous(dst)) {
        પાછા ખોટું;
    }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_sum(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src0->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }
    જો (dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    જો (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        પાછા ખોટું;
    }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_sum_rows(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src0->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }
    જો (dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    // TODO: add support for non-contigiuos tensors
    જો (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        પાછા ખોટું;
    }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_argmax(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src0->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }
    જો (dst->type != GGML_TYPE_I32) {
        પાછા ખોટું;
    }

    જો (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        પાછા ખોટું;
    }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_pool_2d(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ int32_t * params = op->op_params;

    જો (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        પાછા ખોટું;
    }
    જો (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }
    જો (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        પાછા ખોટું;
    }
    જો (!ggml_hexagon_pool_shape_fits(src0, op)) {
        પાછા ખોટું;
    }

    અચલ int32_t kernel_x = params[1];
    અચલ int32_t kernel_y = params[2];
    અચલ int32_t stride_x = params[3];
    અચલ int32_t stride_y = params[4];
    અચલ int32_t pad_x    = params[5];
    અચલ int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    જો (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        પાછા ખોટું;
    }

    અચલ uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    અચલ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    સંરચના htp_pool_vtcm_layout layout;
    પાછા htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_threads, sess->vtcm_size);
}

સ્થિર તાર્કિક ggml_hexagon_supported_pool_1d(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ int32_t * params = op->op_params;

    જો (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        પાછા ખોટું;
    }
    જો (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }
    જો (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        પાછા ખોટું;
    }
    જો (!ggml_hexagon_pool_shape_fits(src0, op)) {
        પાછા ખોટું;
    }

    અચલ int32_t kernel = params[1];
    અચલ int32_t stride = params[2];
    અચલ int32_t pad    = params[3];
    જો (kernel <= 0 || stride <= 0 || pad < 0) {
        પાછા ખોટું;
    }

    અચલ uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    અચલ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    સંરચના htp_pool_vtcm_layout layout;
    પાછા htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_threads, sess->vtcm_size);
}

સ્થિર તાર્કિક ggml_hexagon_supported_activations(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        પાછા ખોટું;
    }
    જો (dst->type != src0->type) {
        પાછા ખોટું;
    }

    જો (!ggml_is_contiguous_1(src0)) {
        પાછા ખોટું;
    }
    જો (!ggml_is_contiguous(dst)) {
        પાછા ખોટું;
    }

    જો (src1) {
        જો (src1->type != src0->type) {
            પાછા ખોટું;
        }
        જો (!ggml_are_same_shape(src0, src1)) {
            પાછા ખોટું;
        }
        જો (!ggml_is_contiguous_1(src1)) {
            પાછા ખોટું;
        }
    }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_softmax(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];
    અચલ સંરચના ggml_tensor * src2 = op->src[2];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src2) {
        પાછા ખોટું;  // FIXME: add support for sinks
    }

    જો (src0->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }
    જો (dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    જો (src0->ne[2] > 512) {
        પાછા ખોટું;
    }

    જો (src1) {
        જો (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            પાછા ખોટું;
        }
        જો (src0->ne[0] != src1->ne[0]) {
            પાછા ખોટું;
        }
        જો (src1->ne[1] < src0->ne[1]) {
            પાછા ખોટું;
        }
        જો (src0->ne[2] % src1->ne[2] != 0) {
            પાછા ખોટું;
        }
        જો (src0->ne[3] % src1->ne[3] != 0) {
            પાછા ખોટું;
        }
    }

    જો (src1) {
        જો (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            પાછા ખોટું;
        }
    } નહીંતર {
        જો (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            પાછા ખોટું;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    અચલ int64_t ne0 = src0->ne[0];
    જો (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        પાછા ખોટું;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    જો (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        પાછા ખોટું;
    }

    અચલ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    અચલ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    સંરચના htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != nullptr, src1 && src1->type == GGML_TYPE_F16, n_threads);
    જો (layout.total_bytes > sess->vtcm_size) {
        પાછા ખોટું;
    }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_set_rows(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0]; //કિંમતો
    અચલ સંરચના ggml_tensor * src1 = op->src[1]; // indices
    અચલ સંરચના ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    જો (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        પાછા ખોટું;
    }

    જો (src0->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    જો (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        પાછા ખોટું;
    }

    જો (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        પાછા ખોટું;
    }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_get_rows(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0]; //કિંમતો
    અચલ સંરચના ggml_tensor * src1 = op->src[1]; // indices
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        પાછા ખોટું;
    }

    જો (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(dst)) {
        પાછા ખોટું;
    }

    જો (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        પાછા ખોટું;
    }

    જો (src0->type == dst->type) {
        જો (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            પાછા ખોટું;
        }
    } નહીંતર જો (src0->type == GGML_TYPE_I32) {
        પાછા ખોટું;
    } નહીંતર જો (dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    જો (ggml_is_empty(op)) {
        પાછા સાચું;
    }

    સંરચના htp_get_rows_kernel_params kparams;
    ggml_hexagon_precompute_get_rows_params(sess, src0, src1, dst, &kparams);
    જો (kparams.n_threads == 0 || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        પાછા ખોટું;
    }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_argsort(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0]; //કિંમતો
    અચલ સંરચના ggml_tensor * dst  = op;         // indices

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        પાછા ખોટું;
    }

    અચલ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    અચલ uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    અચલ size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    સંરચના htp_sort_vtcm_layout layout;
    જો (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, ખોટું)) {
        પાછા ખોટું;
    }

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_hexagon_supported_ટોચ_k(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0]; //કિંમતો
    અચલ સંરચના ggml_tensor * dst  = op;         // indices

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        પાછા ખોટું;
    }

    અચલ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    અચલ uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    અચલ size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    સંરચના htp_sort_vtcm_layout layout;
    જો (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, સાચું)) {
        પાછા ખોટું;
    }

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_hexagon_supported_rope(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];
    અચલ સંરચના ggml_tensor * src2 = op->src[2];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (!ggml_are_same_shape(src0, dst)) {
        પાછા ખોટું;
    }

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        પાછા ખોટું;
    }

    જો (src0->ne[0] <= 0) {
        પાછા ખોટું;
    }

    અચલ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    જો (src0_nrows == 0) {
        પાછા ખોટું;
    }

    અચલ int32_t * op_params = &op->op_params[0];
    અચલ પૂર્ણાંક n_dims = op_params[1];
    અચલ પૂર્ણાંક mode   = op_params[2];
    અચલ પૂર્ણાંક n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    જો (n_dims < 0 || n_dims % 2 != 0) {
        પાછા ખોટું;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    જો (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        પાછા ખોટું;
    }

    દશાંશ freq_base;
    memcpy(&freq_base, op_params + 5, કદ(દશાંશ));
    જો (freq_base < 0.0f) {
        પાછા ખોટું;
    }

    જો (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        પાછા ખોટું;
    }

    અચલ તાર્કિક is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    જો (mode == GGML_ROPE_TYPE_VISION) {
        જો (n_dims != (પૂર્ણાંક) (src0->ne[0] / 2) || n_offs != 0) {
            પાછા ખોટું;
        }
    }

    જો (is_mrope) {
        અચલ int32_t * sections = op_params + 11;
        જો (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            પાછા ખોટું;
        }
    }

    અચલ int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    જો (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        પાછા ખોટું;
    }

    જો (src2) {
        જો (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            પાછા ખોટું;
        }
        જો (src2->ne[0] < (n_dims / 2)) {
            પાછા ખોટું;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    જો (src0->nb[0] != કદ(દશાંશ) || dst->nb[0] != કદ(દશાંશ)) {
        પાછા ખોટું;
    }
    જો (src0->nb[1] < src0->ne[0] * કદ(દશાંશ) || dst->nb[1] < dst->ne[0] * કદ(દશાંશ)) {
        પાછા ખોટું;
    }

    અચલ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    અચલ uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    સંરચના htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    જો (layout.total_bytes > sess->vtcm_size) {
        પાછા ખોટું;
    }

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_hexagon_supported_ssm_conv(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];
    અચલ સંરચના ggml_tensor * dst  = op;

    // Only support FP32 for now
    જો (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    // Check IO tensor shapes and dims
    જો (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        પાછા ખોટું; // src0 should be effectively 3D
    }

    અચલ પૂર્ણાંક d_conv  = src1->ne[0];
    અચલ પૂર્ણાંક d_inner = src0->ne[1];
    અચલ પૂર્ણાંક n_t = dst->ne[1];
    અચલ પૂર્ણાંક n_s = dst->ne[2];

    જો (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        પાછા ખોટું;
    }
    જો (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        પાછા ખોટું;
    }
    જો (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        પાછા ખોટું;
    }
    જો (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        પાછા ખોટું;
    }
    જો (src0->nb[0] != કદ(દશાંશ) || src1->nb[0] != કદ(દશાંશ) || dst->nb[0] != કદ(દશાંશ)) {
        પાછા ખોટું;
    }
    જો (src0->nb[1] != src0->ne[0] * કદ(દશાંશ) || src1->nb[1] != src1->ne[0] * કદ(દશાંશ)) {
        પાછા ખોટું;
    }

    સંરચના htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    જો ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        પાછા ખોટું;
    }

    પાછા સાચું;
}

સ્થિર તાર્કિક im2col_3d_calc_output_size(uint32_t આગત, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    જો (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        પાછા ખોટું;
    }

    અચલ int64_t numerator = (int64_t) આગત + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    જો (numerator < 0) {
        પાછા ખોટું;
    }

    અચલ uint64_t size = (uint64_t) (numerator / stride) + 1;
    જો (size == 0 || size > UINT32_MAX) {
        પાછા ખોટું;
    }

    *output = (uint32_t) size;
    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_hexagon_precompute_im2col_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (!src0 || !src1 || !dst) {
        પાછા ખોટું;
    }

    જો (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        પાછા ખોટું;
    }

    જો (src1->nb[0] != કદ(દશાંશ) || !ggml_is_contiguous(dst)) {
        પાછા ખોટું;
    }

    અચલ int32_t * op_params = (અચલ int32_t *) op->op_params;
    અચલ int32_t s0 = op_params[0];
    અચલ int32_t s1 = op_params[1];
    અચલ int32_t p0 = op_params[2];
    અચલ int32_t p1 = op_params[3];
    અચલ int32_t d0 = op_params[4];
    અચલ int32_t d1 = op_params[5];
    અચલ તાર્કિક is_2D = (op_params[6] == 1);

    જો (s0 <= 0 || d0 <= 0 || p0 < 0) {
        પાછા ખોટું;
    }
    જો (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        પાછા ખોટું;
    }

    અચલ uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    અચલ uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    અચલ uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    અચલ uint32_t IW = (uint32_t) src1->ne[0];
    અચલ uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    અચલ uint32_t KW = (uint32_t) src0->ne[0];
    અચલ uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    અચલ uint32_t OW = (uint32_t) dst->ne[1];

    જો (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        પાછા ખોટું;
    }

    અચલ uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    અચલ uint64_t total_rows_64   = (uint64_t) N * OH;
    જો (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX) {
        પાછા ખોટું;
    }
    અચલ uint32_t patch_stride = (uint32_t) patch_stride_64;
    અચલ uint32_t total_rows   = (uint32_t) total_rows_64;

    જો (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        પાછા ખોટું;
    }
    જો (is_2D && dst->ne[2] != OH) {
        પાછા ખોટું;
    }

    અચલ uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? કદ(ggml_fp16_t) : કદ(દશાંશ);

    તાર્કિક is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    જો (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    અચલ uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    અચલ uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    અચલ uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    જો (is_patchembed) {
        અચલ uint64_t src_row_bytes_raw = (uint64_t) IC * KH * IW * કદ(દશાંશ);
        અચલ uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        જો (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            અચલ uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            અચલ uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            સંરચના htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            જો (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                પાછા સાચું;
            }
        }
    }

    માટે (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        અચલ uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        જો (Wb > UINT32_MAX) {
            જો (owb == 1) તોડો;
            ચાલુ;
        }

        અચલ uint64_t src_row_bytes_raw = (uint64_t) IC * KH * Wb * કદ(દશાંશ);
        અચલ uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        જો (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            જો (owb == 1) તોડો;
            ચાલુ;
        }

        અચલ uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        અચલ uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        સંરચના htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        જો (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            પાછા સાચું;
        }
        જો (owb == 1) તોડો;
    }

    પાછા ખોટું;
}

સ્થિર તાર્કિક ggml_hexagon_precompute_im2col_3d_params(
    અચલ સંરચના ggml_hexagon_session * sess,
    અચલ સંરચના ggml_tensor * op,
    સંરચના htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, કદ(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * src1 = op->src[1];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (!src0 || !src1 || !dst) {
        પાછા ખોટું;
    }

    જો (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        પાછા ખોટું;
    }

    જો (src1->nb[0] != કદ(દશાંશ) || !ggml_is_contiguous(dst)) {
        પાછા ખોટું;
    }

    અચલ int32_t * params = (અચલ int32_t *) op->op_params;
    અચલ int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    અચલ int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    અચલ int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    અચલ int32_t ic = params[9];

    જો (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        પાછા ખોટું;
    }

    જો (src1->ne[3] % ic != 0) {
        પાછા ખોટું;
    }

    અચલ uint32_t IC = (uint32_t) ic;
    અચલ uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    જો (N == 0) {
        પાછા ખોટું;
    }

    uint32_t OW, OH, OD;
    જો (!im2col_3d_calc_output_size(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_size(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_size(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        પાછા ખોટું;
    }

    અચલ uint32_t KD = (uint32_t) src0->ne[2];
    અચલ uint32_t KH = (uint32_t) src0->ne[1];
    અચલ uint32_t KW = (uint32_t) src0->ne[0];
    અચલ uint32_t ID = (uint32_t) src1->ne[2];
    અચલ uint32_t IH = (uint32_t) src1->ne[1];
    અચલ uint32_t IW = (uint32_t) src1->ne[0];

    અચલ uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    અચલ uint64_t total_rows_64   = (uint64_t) N * OD * OH;
    અચલ uint64_t dst_ne3_64      = (uint64_t) OD * N;

    જો (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        પાછા ખોટું;
    }
    અચલ uint32_t patch_stride = (uint32_t) patch_stride_64;
    અચલ uint32_t total_rows   = (uint32_t) total_rows_64;
    અચલ uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    જો (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        પાછા ખોટું;
    }

    અચલ uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? કદ(ggml_fp16_t) : કદ(દશાંશ);

    અચલ તાર્કિક is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    અચલ uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    અચલ uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    અચલ uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    જો (is_patchembed) {
        અચલ uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * IW * કદ(દશાંશ);
        અચલ uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        જો (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            અચલ uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            અચલ uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            સંરચના htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            જો (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                પાછા સાચું;
            }
        }
    }

    માટે (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        અચલ uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        જો (Wb > UINT32_MAX) {
            જો (owb == 1) તોડો;
            ચાલુ;
        }

        અચલ uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * Wb * કદ(દશાંશ);
        અચલ uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        જો (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            જો (owb == 1) તોડો;
            ચાલુ;
        }

        અચલ uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        અચલ uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        સંરચના htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        જો (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            પાછા સાચું;
        }
        જો (owb == 1) તોડો;
    }

    પાછા ખોટું;
}

સ્થિર તાર્કિક ggml_hexagon_supported_im2col(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    સંરચના htp_im2col_kernel_params kparams;
    પાછા ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

સ્થિર તાર્કિક ggml_hexagon_supported_im2col_3d(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    સંરચના htp_im2col_kernel_params kparams;
    પાછા ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

સ્થિર તાર્કિક ggml_hexagon_supported_pad(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    અચલ int32_t lp0 = ((અચલ int32_t *) op->op_params)[0];
    અચલ int32_t rp0 = ((અચલ int32_t *) op->op_params)[1];
    અચલ int32_t circular = ((અચલ int32_t *) op->op_params)[8];

    જો (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        પાછા ખોટું;
    }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_cumsum(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    જો (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        પાછા ખોટું;
    }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_diag(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * dst  = op;

    // diag only supports F32 currently
    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    // Input must have ne[1] == 1 (vector input)
    જો (src0->ne[1] != 1) {
        પાછા ખોટું;
    }

    // Output must be square in first two dimensions
    જો (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        પાછા ખોટું;
    }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_solve_tri(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * src0 = op->src[0]; // A
    અચલ સંરચના ggml_tensor * src1 = op->src[1]; // B
    અચલ સંરચના ggml_tensor * dst  = op;         // X

    જો (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    જો (src0->ne[0] != src0->ne[1]) {
        પાછા ખોટું;
    }

    જો (src0->ne[1] != src1->ne[1]) {
        પાછા ખોટું;
    }

    જો (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        પાછા ખોટું;
    }

    જો (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        પાછા ખોટું;
    }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_tri(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {

    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src0->type != GGML_TYPE_F32) { પાછા ખોટું; }
    જો (dst->type  != GGML_TYPE_F32) { પાછા ખોટું; }
    જો (!ggml_are_same_shape(src0, dst)) { પાછા ખોટું; }
    જો (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { પાછા ખોટું; }

    પાછા સાચું;

    GGML_UNUSED(sess);
}

સ્થિર અચલ અક્ષર * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    પાછા sess->c_name();
}

સ્થિર શૂન્ય ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

સ્થિર htp_op_code op_remap_to_htp(અચલ ggml_tensor * t) {
    પસંદગી (t->op) {
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:  પાછા HTP_OP_FLASH_ATTN_EXT;
        સ્થિતિ GGML_OP_MUL_MAT:
            જો (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                પાછા HTP_OP_HADAMARD;
            }
            પાછા HTP_OP_MUL_MAT;
        સ્થિતિ GGML_OP_MUL_MAT_ID:      પાછા HTP_OP_MUL_MAT_ID;
        સ્થિતિ GGML_OP_MUL:             પાછા HTP_OP_MUL;
        સ્થિતિ GGML_OP_ADD:             પાછા HTP_OP_ADD;
        સ્થિતિ GGML_OP_ADD_ID:          પાછા HTP_OP_ADD_ID;
        સ્થિતિ GGML_OP_SUB:             પાછા HTP_OP_SUB;
        સ્થિતિ GGML_OP_DIV:             પાછા HTP_OP_DIV;
        સ્થિતિ GGML_OP_DUP:
        સ્થિતિ GGML_OP_CPY:             પાછા HTP_OP_CPY;
        સ્થિતિ GGML_OP_CONT:            પાછા HTP_OP_CPY;
        સ્થિતિ GGML_OP_GET_ROWS:        પાછા HTP_OP_GET_ROWS;
        સ્થિતિ GGML_OP_SET_ROWS:        પાછા HTP_OP_SET_ROWS;
        સ્થિતિ GGML_OP_SUM:             પાછા HTP_OP_SUM;
        સ્થિતિ GGML_OP_SUM_ROWS:        પાછા HTP_OP_SUM_ROWS;
        સ્થિતિ GGML_OP_POOL_2D:         પાછા HTP_OP_POOL_2D;
        સ્થિતિ GGML_OP_POOL_1D:         પાછા HTP_OP_POOL_1D;
        સ્થિતિ GGML_OP_ARGSORT:         પાછા HTP_OP_ARGSORT;
        સ્થિતિ GGML_OP_ટોચ_K:           પાછા HTP_OP_ટોચ_K;
        સ્થિતિ GGML_OP_ARGMAX:          પાછા HTP_OP_ARGMAX;
        સ્થિતિ GGML_OP_NORM:            પાછા HTP_OP_NORM;
        સ્થિતિ GGML_OP_L2_NORM:         પાછા HTP_OP_L2_NORM;
        સ્થિતિ GGML_OP_RMS_NORM:        પાછા HTP_OP_RMS_NORM;
        સ્થિતિ GGML_OP_CONCAT:          પાછા HTP_OP_CONCAT;
        સ્થિતિ GGML_OP_SCALE:           પાછા HTP_OP_SCALE;
        સ્થિતિ GGML_OP_CLAMP:           પાછા HTP_OP_CLAMP;
        સ્થિતિ GGML_OP_LEAKY_RELU:      પાછા HTP_OP_LEAKY_RELU;
        સ્થિતિ GGML_OP_SQR:             પાછા HTP_OP_SQR;
        સ્થિતિ GGML_OP_SQRT:            પાછા HTP_OP_SQRT;
        સ્થિતિ GGML_OP_LOG:             પાછા HTP_OP_UNARY_LOG;
        સ્થિતિ GGML_OP_SOFT_MAX:        પાછા HTP_OP_SOFTMAX;
        સ્થિતિ GGML_OP_SSM_CONV:        પાછા HTP_OP_SSM_CONV;
        સ્થિતિ GGML_OP_GATED_DELTA_NET: પાછા HTP_OP_GATED_DELTA_NET;
        સ્થિતિ GGML_OP_ROPE:            પાછા HTP_OP_ROPE;
        સ્થિતિ GGML_OP_REPEAT:          પાછા HTP_OP_REPEAT;
        સ્થિતિ GGML_OP_CUMSUM:          પાછા HTP_OP_CUMSUM;
        સ્થિતિ GGML_OP_FILL:            પાછા HTP_OP_FILL;
        સ્થિતિ GGML_OP_DIAG:            પાછા HTP_OP_DIAG;
        સ્થિતિ GGML_OP_SOLVE_TRI:       પાછા HTP_OP_SOLVE_TRI;
        સ્થિતિ GGML_OP_TRI:             પાછા HTP_OP_TRI;
        સ્થિતિ GGML_OP_PAD:             પાછા HTP_OP_PAD;
        સ્થિતિ GGML_OP_IM2COL:          પાછા HTP_OP_IM2COL;
        સ્થિતિ GGML_OP_IM2COL_3D:       પાછા HTP_OP_IM2COL_3D;
        સ્થિતિ GGML_OP_ROLL:            પાછા HTP_OP_ROLL;

        સ્થિતિ GGML_OP_UNARY:
            પસંદગી (ggml_get_unary_op(t)) {
                સ્થિતિ GGML_UNARY_OP_SILU:       પાછા HTP_OP_UNARY_SILU;
                સ્થિતિ GGML_UNARY_OP_GELU:       પાછા HTP_OP_UNARY_GELU;
                સ્થિતિ GGML_UNARY_OP_GELU_QUICK: પાછા HTP_OP_UNARY_GELU_QUICK;
                સ્થિતિ GGML_UNARY_OP_GELU_ERF:   પાછા HTP_OP_UNARY_GELU_ERF;
                સ્થિતિ GGML_UNARY_OP_SIGMOID:    પાછા HTP_OP_UNARY_SIGMOID;
                સ્થિતિ GGML_UNARY_OP_NEG:        પાછા HTP_OP_UNARY_NEG;
                સ્થિતિ GGML_UNARY_OP_EXP:        પાછા HTP_OP_UNARY_EXP;
                સ્થિતિ GGML_UNARY_OP_SOFTPLUS:   પાછા HTP_OP_UNARY_SOFTPLUS;
                સ્થિતિ GGML_UNARY_OP_TANH:       પાછા HTP_OP_UNARY_TANH;
                સ્થિતિ GGML_UNARY_OP_ABS:        પાછા HTP_OP_UNARY_ABS;
                સ્થિતિ GGML_UNARY_OP_RELU:       પાછા HTP_OP_UNARY_RELU;
                સ્થિતિ GGML_UNARY_OP_STEP:       પાછા HTP_OP_UNARY_STEP;
            મૂળભૂત:
                તોડો;
            }
            તોડો;

        સ્થિતિ GGML_OP_GLU:
            પસંદગી (ggml_get_glu_op(t)) {
                સ્થિતિ GGML_GLU_OP_SWIGLU:       પાછા HTP_OP_GLU_SWIGLU;
                સ્થિતિ GGML_GLU_OP_SWIGLU_OAI:   પાછા HTP_OP_GLU_SWIGLU_OAI;
                સ્થિતિ GGML_GLU_OP_SWIGLU_CLAMP: પાછા HTP_OP_GLU_SWIGLU_CLAMP;
                સ્થિતિ GGML_GLU_OP_GEGLU:        પાછા HTP_OP_GLU_GEGLU;
                સ્થિતિ GGML_GLU_OP_GEGLU_QUICK:  પાછા HTP_OP_GLU_GEGLU_QUICK;
                સ્થિતિ GGML_GLU_OP_GEGLU_ERF:    પાછા HTP_OP_GLU_GEGLU_ERF;
                મૂળભૂત: તોડો;
            }
            તોડો;

        મૂળભૂત:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    પાછા HTP_OP_INVALID;
}

સ્થિર અંતઃરેખ તાર્કિક op_is_compute(ggml_tensor *ગાંઠ)
{
    પાછા !ggml_op_is_empty(ગાંઠ->op) && !ggml_is_empty(ગાંઠ) && (ગાંઠ->flags & GGML_TENSOR_FLAG_COMPUTE);
}

સ્થિર તાર્કિક mm_is_hmx_eligible(અચલ ggml_tensor * t) {
    જો (opt_nhmx == 0) { પાછા ખોટું; }

    અચલ ggml_tensor * src0 = t->src[0];
    અચલ ggml_tensor * src1 = t->src[1];

    જો (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        અચલ ggml_tensor src1_c = ggml_hexagon_tensor_collapse_rows(src1);
        પાછા ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, ખોટું, ખોટું);
    }

    અચલ તાર્કિક is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    અચલ તાર્કિક is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    પાછા ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

સ્થિર તાર્કિક is_supported_mul_mat_nx_kernel(અચલ ggml_tensor * src0, અચલ સંરચના htp_mm_kernel_params * kparams) {
    જો (kparams->n_hmx) {
        પાછા kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    જો (!ggml_hexagon_tensor_is_tiled(src0)) {
        પાછા ખોટું;
    }

    પાછા kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
}

સ્થિર તાર્કિક is_supported_mul_mat_id_nx_kernel(અચલ ggml_tensor * src0, અચલ સંરચના htp_mm_kernel_params * kparams) {
    જો (kparams->n_hmx) {
        પાછા kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    જો (!ggml_hexagon_tensor_is_tiled(src0)) {
        પાછા ખોટું;
    }

    પાછા kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK;
}

સ્થિર તાર્કિક is_mergeable_mul_mat(અચલ ggml_tensor * t) {
    જો (t->op != GGML_OP_MUL_MAT) પાછા ખોટું;
    જો (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) પાછા ખોટું;

    અચલ ggml_tensor * src0 = t->src[0];
    અચલ ggml_tensor * src1 = t->src[1];
    જો (src0->ne[2] != 1 || src0->ne[3] != 1) પાછા ખોટું;

    જો (mm_is_hmx_eligible(t)) {
        પાછા htp_is_hmx_weight_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    જો (src1->type != GGML_TYPE_F32) પાછા ખોટું;

    પાછા ggml_hexagon_tensor_is_tiled(src0);
}

સ્થિર તાર્કિક is_mergeable_mul_mat_pair(અચલ ggml_tensor * n1, અચલ ggml_tensor * n2) {
    જો (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        પાછા ખોટું;
    }
    જો (n1->src[1] != n2->src[1]) {
        પાછા ખોટું;
    }
    જો (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        પાછા ખોટું;
    }
    જો (n1->src[0]->type != n2->src[0]->type) {
        પાછા ખોટું;
    }
    જો (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        પાછા ખોટું;
    }
    જો (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        પાછા ખોટું;
    }
    પાછા સાચું;
}

સ્થિર તાર્કિક is_mergeable_mul_mat_id(અચલ ggml_tensor * t) {
    જો (t->op != GGML_OP_MUL_MAT_ID) પાછા ખોટું;

    અચલ ggml_tensor * src0 = t->src[0];
    જો (mm_is_hmx_eligible(t)) {
        પાછા htp_is_hmx_weight_type(src0->type);
    }

    પાછા ggml_hexagon_tensor_is_tiled(src0);
}

સ્થિર તાર્કિક is_mergeable_mul_mat_id_pair(અચલ ggml_tensor * n1, અચલ ggml_tensor * n2) {
    જો (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        પાછા ખોટું;
    }
    જો (n1->src[1] != n2->src[1]) {
        પાછા ખોટું;
    }
    જો (n1->src[2] != n2->src[2]) {
        પાછા ખોટું;
    }
    જો (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        પાછા ખોટું;
    }
    જો (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        પાછા ખોટું;
    }
    જો (n1->src[0]->type != n2->src[0]->type) {
        પાછા ખોટું;
    }
    જો (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        પાછા ખોટું;
    }
    જો (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        પાછા ખોટું;
    }
    પાછા સાચું;
}

સ્થિર ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    જો (sess->last_error > HTP_STATUS_OK) {
        પાછા GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    અચલ std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    તાર્કિક cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
    જો (cache_hit) {
        nodes_ptr = &sess->cached_nodes;
    } નહીંતર {
        // Tag fusable tensors in graph
        માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; i++) {
            auto * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            જો (!extra) ચાલુ;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            જો (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } નહીંતર જો ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                જો ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_ગાંઠ_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        માટે (પૂર્ણાંક i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            જો (!op_is_compute(n)) {
                ચાલુ;
            }

            htp_opnode ગાંઠ(HTP_OP_INVALID, n);
            ગાંઠ.opcode = op_remap_to_htp(n);
            જો (ગાંઠ.opcode == HTP_OP_MUL_MAT || ગાંઠ.opcode == HTP_OP_MUL_MAT_ID || ગાંઠ.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = ગાંઠ.ગાંઠ->src[0];
                જો (ggml_hexagon_needs_repack(src0)) {
                    જો (opt_verbose) {
                        અક્ષર dims_buf[64];
                        અક્ષર strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, કદ(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, કદ(strides_buf), src0);
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

            જો (ગાંઠ.opcode == HTP_OP_MUL_MAT || ગાંઠ.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    ગાંઠ.ગાંઠ->src[0], ગાંઠ.ગાંઠ->src[1], ગાંઠ.ગાંઠ,
                    (સંરચના htp_mm_kernel_params *)ગાંઠ.kernel_params
                );
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_MUL || ગાંઠ.opcode == HTP_OP_ADD || ગાંઠ.opcode == HTP_OP_ADD_ID || ગાંઠ.opcode == HTP_OP_SUB || ગાંઠ.opcode == HTP_OP_DIV) {
                અચલ ggml_tensor * src1 = ગાંઠ.ગાંઠ->src[1];
                અચલ ggml_tensor * src2 = (ગાંઠ.opcode == HTP_OP_ADD_ID) ? ગાંઠ.ગાંઠ->src[2] : nullptr;
                GGML_ASSERT(ggml_hexagon_precompute_દ્વિઅંકી_params(sess,
                    ગાંઠ.opcode, ગાંઠ.ગાંઠ->src[0], src1, ગાંઠ.ગાંઠ,
                    (સંરચના htp_દ્વિઅંકી_kernel_params *) ગાંઠ.kernel_params, src2));
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    ગાંઠ.ગાંઠ,
                    (સંરચના htp_fa_kernel_params *)ગાંઠ.kernel_params
                );
            } નહીંતર જો (htp_op_is_unary(ગાંઠ.opcode)) {
                auto inputs = ગાંઠ.get_inputs();
                અચલ સંરચના ggml_tensor * src0 = inputs[0];
                અચલ સંરચના ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    ગાંઠ.opcode, src0, src1, ગાંઠ.dst(),
                    (સંરચના htp_unary_kernel_params *)ગાંઠ.kernel_params
                );
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_GET_ROWS) {
                ggml_hexagon_precompute_get_rows_params(sess,
                    ગાંઠ.ગાંઠ->src[0], ગાંઠ.ગાંઠ->src[1], ગાંઠ.dst(),
                    (સંરચના htp_get_rows_kernel_params *)ગાંઠ.kernel_params
                );
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_SET_ROWS) {
                ggml_hexagon_precompute_set_rows_params(sess,
                    ગાંઠ.ગાંઠ->src[0], ગાંઠ.ગાંઠ->src[1], ગાંઠ.dst(),
                    (સંરચના htp_set_rows_kernel_params *)ગાંઠ.kernel_params
                );
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    ગાંઠ.ગાંઠ,
                    (સંરચના htp_rope_kernel_params *)ગાંઠ.kernel_params
                );
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    ગાંઠ.ગાંઠ->src[0], ગાંઠ.ગાંઠ->src[1], ગાંઠ.dst(),
                    (સંરચના htp_ssm_conv_kernel_params *)ગાંઠ.kernel_params
                );
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    ગાંઠ.ગાંઠ,
                    (સંરચના htp_softmax_kernel_params *)ગાંઠ.kernel_params
                );
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    ગાંઠ.ગાંઠ,
                    (સંરચના htp_gdn_kernel_params *)ગાંઠ.kernel_params
                );
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_ARGSORT || ગાંઠ.opcode == HTP_OP_ટોચ_K) {
                ggml_hexagon_precompute_sort_params(sess,
                    ગાંઠ.ગાંઠ,
                    ગાંઠ.opcode == HTP_OP_ટોચ_K,
                    (સંરચના htp_sort_kernel_params *) ગાંઠ.kernel_params
                );
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_POOL_2D || ગાંઠ.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, ગાંઠ.ગાંઠ->src[0], ગાંઠ.dst(),
                    (સંરચના htp_pool_2d_kernel_params *)ગાંઠ.kernel_params,
                    ગાંઠ.opcode == HTP_OP_POOL_1D);
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    ગાંઠ.ગાંઠ,
                    (સંરચના htp_concat_kernel_params *) ગાંઠ.kernel_params
                );
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_CPY || ગાંઠ.opcode == HTP_OP_CPY_FENCE) {
                અચલ તાર્કિક ok = ggml_hexagon_precompute_cpy_params(sess,
                    ગાંઠ.ગાંઠ,
                    (સંરચના htp_copy_kernel_params *) ગાંઠ.kernel_params
                );
                અચલ auto * kparams = (અચલ સંરચના htp_copy_kernel_params *) ગાંઠ.kernel_params;
                જો (ok && ગાંઠ.opcode == HTP_OP_CPY && kparams->total_elems == 0) {
                    ચાલુ;
                }
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_IM2COL) {
                ggml_hexagon_precompute_im2col_params(sess,
                    ગાંઠ.ગાંઠ,
                    (સંરચના htp_im2col_kernel_params *) ગાંઠ.kernel_params
                );
            } નહીંતર જો (ગાંઠ.opcode == HTP_OP_IM2COL_3D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    ગાંઠ.ગાંઠ,
                    (સંરચના htp_im2col_kernel_params *) ગાંઠ.kernel_params
                );
            }
            computed_nodes.push_back(std::move(ગાંઠ));
        }

        જો (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } નહીંતર {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    માટે (અચલ auto & ગાંઠ : *nodes_ptr) {
        sess->enqueue_op(ગાંઠ);
    }

    જો (sess->last_error > HTP_STATUS_OK) {
        પાછા GGML_STATUS_FAILED;
    }

    પાછા GGML_STATUS_SUCCESS;
}

સ્થિર શૂન્ય ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    જો (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
}

ગણના ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

સંરચના ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

સંરચના ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    શૂન્ય reset() {
        ranges.clear();
    }

    શૂન્ય add(અચલ ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    તાર્કિક check(અચલ ggml_hexagon_mem_range & mr) અચલ {
        માટે (અચલ auto & cmp : ranges) {
            જો (mr.pb != cmp.pb) {
                ચાલુ;
            }
            જો (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                ચાલુ;
            }
            જો (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                પાછા ખોટું;
            }
        }
        પાછા સાચું;
    }
};

સ્થિર ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(અચલ ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    અચલ ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    જો (tensor->buffer) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } નહીંતર {
        mr = {
            /*.pb =*/ (uint64_t) base,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    પાછા mr;
}

સ્થિર શૂન્ય ggml_hexagon_mem_ranges_add_ગાંઠ(ggml_hexagon_mem_ranges & mrs, અચલ htp_opnode & ગાંઠ) {
    જો (ગાંઠ.is_empty()) પાછા;

    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
        જો (ગાંઠ.ગાંઠ->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(ગાંઠ.ગાંઠ->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    માટે (અચલ auto * fused : ગાંઠ.fused) {
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            જો (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(ગાંઠ.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

સ્થિર તાર્કિક ggml_hexagon_mem_ranges_check_ગાંઠ(અચલ ggml_hexagon_mem_ranges & mrs, અચલ htp_opnode & ગાંઠ) {
    જો (ગાંઠ.is_empty()) પાછા સાચું;

    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
        જો (ગાંઠ.ગાંઠ->src[i]) {
            જો (!mrs.check(ggml_hexagon_mem_range_from_tensor(ગાંઠ.ગાંઠ->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                પાછા ખોટું;
            }
        }
    }
    માટે (અચલ auto * fused : ગાંઠ.fused) {
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            જો (fused->src[i]) {
                જો (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    પાછા ખોટું;
                }
            }
        }
    }
    પાછા mrs.check(ggml_hexagon_mem_range_from_tensor(ગાંઠ.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

સ્થિર std::vector<પૂર્ણાંક> ggml_hexagon_graph_optimize_reorder(અચલ std::vector<htp_opnode> & nodes) {
    અચલ પૂર્ણાંક n = nodes.size();

    std::vector<પૂર્ણાંક> res;
    res.reserve(n);

    std::vector<તાર્કિક> used(n, ખોટું);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    માટે (પૂર્ણાંક i0 = 0; i0 < n; i0++) {
        જો (used[i0]) {
            ચાલુ;
        }

        અચલ auto & node0 = nodes[i0];

        જો (!node0.stackable()) {
            res.push_back(i0);
            used[i0] = સાચું;
            ચાલુ;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr પૂર્ણાંક N_FORWARD = 16;

        std::vector<પૂર્ણાંક> ઢગલો;
        ઢગલો.push_back(i0);

        mrs.reset();

        માટે (પૂર્ણાંક i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            જો (used[i1]) {
                ચાલુ;
            }

            અચલ auto & node1 = nodes[i1];

            જો (node1.stackable() && node1.same_આગત(node0) && ggml_hexagon_mem_ranges_check_ગાંઠ(mrs, node1)) {
                ઢગલો.push_back(i1);
            } નહીંતર {
                ggml_hexagon_mem_ranges_add_ગાંઠ(mrs, node1);
            }
        }

        માટે (પૂર્ણાંક idx : ઢગલો) {
            res.push_back(idx);
            used[idx] = સાચું;
        }
    }

    પાછા res;
}

સ્થિર શૂન્ય ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    અચલ પૂર્ણાંક n = gf->n_nodes;

    constexpr પૂર્ણાંક MAX_FUSE = 16;

    ગણના ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        htp_opnode ગાંઠ(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        જો (ગાંઠ.op() == GGML_OP_ADD ||
            ગાંઠ.op() == GGML_OP_NORM ||
            ગાંઠ.op() == GGML_OP_RMS_NORM) {
            ops[0] = ગાંઠ.op();

            પૂર્ણાંક f = i + 1;
            જ્યારે (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                જો (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    તોડો;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            માટે (; f > 1; f--) {
                જો (ggml_can_fuse(gf, i, ops, f)) {
                    તોડો;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            માટે (પૂર્ણાંક k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                ગાંઠ.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(ગાંઠ));
    }

    અચલ auto order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        પૂર્ણાંક j = 0;
        માટે (અચલ auto i : order) {
            અચલ auto & ગાંઠ = nodes[i];

            gf->nodes[j++] = ગાંઠ.ગાંઠ;

            માટે (auto * fused : ગાંઠ.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

સ્થિર uint64_t ggml_hexagon_session_ચાવી(અચલ ggml_hexagon_session * sess) {
    પાછા ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

સ્થિર તાર્કિક ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, અચલ ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    auto sbuf_dst = sess_dst->mmap_tensor(dst);

    જો (!sess_src->clone_buffer(sbuf_dst)) { પાછા ખોટું; }

    અચલ uint64_t src_ચાવી = ggml_hexagon_session_ચાવી(sess_src);
    auto & fence_slot = sess_dst->cpy_fence_slots[src_ચાવી];
    જો (!fence_slot) {
        fence_slot = (અસ્થિર uint32_t *) sess_dst->alloc_fence(1);
    }

    જો (!sess_src->clone_buffer(sess_dst->fence_buf)) { પાછા ખોટું; }

    જો (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    સ્થિર ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.buffer = &sess_dst->fence_buf->backend_buffer;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.data   = (શૂન્ય *) fence_slot;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = કદ(int32_t);
    fence_tensor.nb[1]  = કદ(int32_t);
    fence_tensor.nb[2]  = કદ(int32_t);
    fence_tensor.nb[3]  = કદ(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ સાચું);

    sess_dst->add_peer(sess_src);

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, અચલ ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    auto sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    જો (!sess_dst->clone_buffer(sbuf_src)) { પાછા ખોટું; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, અચલ ggml_tensor * src, ggml_tensor * dst) {
    જો (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        પાછા ખોટું;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    જો (!dst->extra) {
        ggml_backend_buffer_init_tensor(dst->buffer, dst);
    }

    auto * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    અચલ auto * src_extra = static_cast<અચલ ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    જો (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        પાછા સાચું;
    }

    જો (sess_src->phys_idx != sess_dst->phys_idx)
        પાછા ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    પાછા ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

સ્થિર ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = નવું ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_slot = (અસ્થિર uint32_t *) sess->alloc_fence(1);

    સ્થિર ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.buffer = &sess->fence_buf->backend_buffer;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.data   = (શૂન્ય *) hex_event->fence_slot;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = કદ(int32_t);
    hex_event->fence_tensor.nb[1]  = કદ(int32_t);
    hex_event->fence_tensor.nb[2]  = કદ(int32_t);
    hex_event->fence_tensor.nb[3]  = કદ(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (શૂન્ય *)hex_event, (શૂન્ય *)hex_event->fence_slot);

    પાછા નવું ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

સ્થિર શૂન્ય ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    જો (hex_event->seq == 0) {
        પાછા;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (શૂન્ય *)hex_event, hex_event->seq, (શૂન્ય *)hex_event->fence_slot);

    auto * fence = reinterpret_cast<અચલ અસ્થિર std::atomic<uint32_t> *>(hex_event->fence_slot);

    જો ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    જ્યારે (સાચું) {
        જો ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            જો (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            તોડો;
        }
        std::this_thread::yield();
    }
}

સ્થિર શૂન્ય ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (શૂન્ય *)hex_event);
    hex_event->fence_sess->free_fence((શૂન્ય *) hex_event->fence_slot, 1);
    delete hex_event;
    delete event;
}

સ્થિર શૂન્ય ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

સ્થિર શૂન્ય ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    જો (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ ખોટું);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (શૂન્ય *)hex_event, hex_event->seq, (શૂન્ય *)hex_event->fence_slot);
}

સ્થિર શૂન્ય ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    જો (hex_event->seq == 0) {
        પાછા;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (શૂન્ય *)hex_event, hex_event->seq, (શૂન્ય *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    જો (sess->phys_idx == hex_event->sess->phys_idx) {
        જો (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        પાછા;
    }

    sess->clone_buffer(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ સાચું);
}

સ્થિર શૂન્ય ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (પૂર્ણાંક) tensor->buffer->usage : -1);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

સ્થિર શૂન્ય ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, અચલ સંરચના ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (પૂર્ણાંક) tensor->buffer->usage : -1);
    sess->flush_sync();
    જો (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

સ્થિર શૂન્ય ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     સંરચના ggml_tensor * tensor,
                                                     અચલ શૂન્ય * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (પૂર્ણાંક) tensor->buffer->usage : -1);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

સ્થિર શૂન્ય ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     અચલ સંરચના ggml_tensor * tensor,
                                                     શૂન્ય * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (પૂર્ણાંક) tensor->buffer->usage : -1);
    sess->flush_sync();
    જો (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

સ્થિર સંરચના ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ ખાલી,
    /* .graph_plan_free         = */ ખાલી,
    /* .graph_plan_update       = */ ખાલી,
    /* .graph_plan_compute      = */ ખાલી,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_record,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

સ્થિર ggml_guid_t ggml_backend_hexagon_guid() {
    સ્થિર ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    પાછા &guid;
}

તાર્કિક ggml_backend_is_hexagon(ggml_backend_t backend) {
    પાછા backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

સ્થિર ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, અચલ અક્ષર * params) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    પાછા નવું ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

સ્થિર અચલ અક્ષર * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    પાછા dev_ctx->c_name();

    GGML_UNUSED(dev);
}

સ્થિર અચલ અક્ષર * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    પાછા "Hexagon";
    GGML_UNUSED(dev);
}

સ્થિર શૂન્ય ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * મુક્ત, size_t * total) {
    *મુક્ત  = opt_mbuf_total;
    *total = opt_mbuf_total;

    GGML_UNUSED(dev);
}

સ્થિર ગણના ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    પાછા GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

સ્થિર શૂન્ય ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, સંરચના ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ સાચું,
        /* .host_buffer           = */ ખોટું,
        /* .buffer_from_host_ptr  = */ ખોટું,
        /* .events                = */ સાચું,
        /* .mmap_support          = */ ખોટું,
    };
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    પાછા &dev_ctx->buffer_type;
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    જો (!opt_hostbuf) {
        પાછા ખાલી;
    }
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    પાછા &dev_ctx->host_buffer_type;
}

સ્થિર તાર્કિક ggml_hexagon_supported_cpy(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    સંરચના htp_copy_kernel_params kparams;
    પાછા ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

સ્થિર તાર્કિક ggml_hexagon_supported_cont(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    સંરચના htp_copy_kernel_params kparams;
    પાછા ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

સ્થિર તાર્કિક ggml_hexagon_supported_repeat(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    GGML_UNUSED(sess);
    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * dst  = op;

    // Support f32 and f16
    જો (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) પાછા ખોટું;

    // src and dst must be the same type
    જો (src0->type != dst->type) પાછા ખોટું;

    // dst dims must be multiples of src dims
    જો (dst->ne[0] % src0->ne[0] != 0) પાછા ખોટું;
    જો (dst->ne[1] % src0->ne[1] != 0) પાછા ખોટું;
    જો (dst->ne[2] % src0->ne[2] != 0) પાછા ખોટું;
    જો (dst->ne[3] % src0->ne[3] != 0) પાછા ખોટું;

    // require contiguous tensors (no transposition)
    જો (ggml_is_transposed(src0) || ggml_is_transposed(dst)) પાછા ખોટું;

    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_hexagon_supported_concat(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    સંરચના htp_concat_kernel_params kparams;
    પાછા ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

સ્થિર તાર્કિક ggml_hexagon_supported_fill(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    અચલ સંરચના ggml_tensor * dst = op;

    જો (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        પાછા ખોટું;
    }

    પાછા સાચું;
    GGML_UNUSED(sess);
}

સ્થિર તાર્કિક ggml_hexagon_supported_roll(અચલ સંરચના ggml_hexagon_session * sess, અચલ સંરચના ggml_tensor * op) {
    GGML_UNUSED(sess);

    અચલ સંરચના ggml_tensor * src0 = op->src[0];
    અચલ સંરચના ggml_tensor * dst  = op;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    જો (!ggml_are_same_shape(src0, dst)) {
        પાછા ખોટું;
    }

    જો (src0->nb[0] != ggml_type_size(src0->type) || dst->nb[0] != ggml_type_size(dst->type)) {
        પાછા ખોટું;
    }

    જો (!ggml_is_contiguous(dst)) {
        પાછા ખોટું;
    }

    પાછા સાચું;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
સ્થિર તાર્કિક ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, અચલ સંરચના ggml_tensor * op) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    // reject ops that match the filter
    જો (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        પાછા ખોટું;
    }

    તાર્કિક supp = ખોટું;
    પસંદગી (op->op) {
        સ્થિતિ GGML_OP_NONE:
        સ્થિતિ GGML_OP_RESHAPE:
        સ્થિતિ GGML_OP_VIEW:
        સ્થિતિ GGML_OP_PERMUTE:
        સ્થિતિ GGML_OP_TRANSPOSE:
            supp = સાચું;
            તોડો;

        સ્થિતિ GGML_OP_MUL:
        સ્થિતિ GGML_OP_ADD:
        સ્થિતિ GGML_OP_SUB:
        સ્થિતિ GGML_OP_DIV:
            supp = ggml_hexagon_supported_દ્વિઅંકી(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_NORM:
        સ્થિતિ GGML_OP_L2_NORM:
        સ્થિતિ GGML_OP_RMS_NORM:
        સ્થિતિ GGML_OP_SCALE:
        સ્થિતિ GGML_OP_CLAMP:
        સ્થિતિ GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_SQR:
        સ્થિતિ GGML_OP_SQRT:
        સ્થિતિ GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_SUM:
            supp = ggml_hexagon_supported_sum(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_UNARY:
            પસંદગી (ggml_get_unary_op(op)) {
                સ્થિતિ GGML_UNARY_OP_NEG:
                સ્થિતિ GGML_UNARY_OP_EXP:
                સ્થિતિ GGML_UNARY_OP_SIGMOID:
                સ્થિતિ GGML_UNARY_OP_SOFTPLUS:
                સ્થિતિ GGML_UNARY_OP_TANH:
                સ્થિતિ GGML_UNARY_OP_ABS:
                સ્થિતિ GGML_UNARY_OP_SILU:
                સ્થિતિ GGML_UNARY_OP_GELU:
                સ્થિતિ GGML_UNARY_OP_GELU_QUICK:
                સ્થિતિ GGML_UNARY_OP_GELU_ERF:
                સ્થિતિ GGML_UNARY_OP_RELU:
                સ્થિતિ GGML_UNARY_OP_STEP:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    તોડો;
                મૂળભૂત:
                    supp = ખોટું;
                    તોડો;
            }
            તોડો;

        સ્થિતિ GGML_OP_GLU:
            પસંદગી (ggml_get_glu_op(op)) {
                સ્થિતિ GGML_GLU_OP_SWIGLU:
                સ્થિતિ GGML_GLU_OP_SWIGLU_OAI:
                સ્થિતિ GGML_GLU_OP_SWIGLU_CLAMP:
                સ્થિતિ GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_GEGLU_QUICK:
                સ્થિતિ GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    તોડો;
                મૂળભૂત:
                    supp = ખોટું;
                    તોડો;
            }
            તોડો;

        સ્થિતિ GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_DUP:
        સ્થિતિ GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_ટોચ_K:
            supp = ggml_hexagon_supported_ટોચ_k(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_IM2COL_3D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            તોડો;

        સ્થિતિ GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            તોડો;

        મૂળભૂત:
            તોડો;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    પાછા supp;
}

સ્થિર તાર્કિક ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    તાર્કિક supp = (buft == &dev_ctx->host_buffer_type) || (buft == &dev_ctx->buffer_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    પાછા supp;
}

સ્થિર અચલ સંરચના ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ ખાલી,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ ખાલી,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    માટે (size_t i = 0; i < opt_ndev; i++) {
        અચલ auto & cfg = opt_device_configs[i];
        જો (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } નહીંતર {
            std::string peers_str;
            માટે (અચલ auto & p : cfg.mdev_group) {
                જો (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = નવું ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    માટે (size_t i = 0; i < opt_ndev; i++) {
        auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        delete dev_ctx;
    }
}

સ્થિર અચલ અક્ષર * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    પાછા "HTP";
    GGML_UNUSED(reg);
}

સ્થિર size_t ggml_backend_hexagon_reg_get_device_ગણતરી(ggml_backend_reg_t reg) {
    પાછા opt_ndev;
    GGML_UNUSED(reg);
}

સ્થિર ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    જો (index >= opt_ndev || !hreg->devices[index].context) {
        પાછા nullptr;
    }

    પાછા &hreg->devices[index];
}

// ** communication context for tensor-split allreduce

સ્થિર શૂન્ય * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    જો (n_backends < 2 || n_backends > 4) {
        પાછા nullptr;
    }

    માટે (size_t i = 0; i < n_backends; ++i) {
        જો (!ggml_backend_is_hexagon(backends[i])) {
            પાછા nullptr;
        }
    }

    માટે (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        માટે (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            જો (sess_i->phys_idx == sess_j->phys_idx) {
                પાછા nullptr;
            }
        }
    }

    auto * ctx = નવું ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    સ્થિર ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    માટે (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (અસ્થિર uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].buffer = &sess_i->fence_buf->backend_buffer;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].data   = (શૂન્ય *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = કદ(int32_t);
        ctx->fence_tensors[i].nb[1]  = કદ(int32_t);
        ctx->fence_tensors[i].nb[2]  = કદ(int32_t);
        ctx->fence_tensors[i].nb[3]  = કદ(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    પાછા ctx;
}

સ્થિર શૂન્ય ggml_backend_hexagon_comm_free(શૂન્ય * comm_ctx_v) {
    જો (!comm_ctx_v) પાછા;
    auto * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    માટે (size_t i = 0; i < ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((શૂન્ય *) ctx->fence_slots[i], 1);
    }
    delete ctx;
}

સ્થિર તાર્કિક ggml_backend_hexagon_comm_allreduce_tensor(શૂન્ય * comm_ctx_v, સંરચના ggml_tensor ** tensors) {
    જો (opt_ar_select == 0 || !comm_ctx_v) પાછા ખોટું;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    અચલ size_t n_backends = comm_ctx->n_backends;

    જો (n_backends < 2 || n_backends > 4) પાછા ખોટું;

    માટે (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        માટે (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            જો (sess_i->phys_idx == sess_j->phys_idx) {
                પાછા ખોટું;
            }
        }
    }

    માટે (size_t i = 0; i < n_backends; i++) {
        જો (!tensors[i] || !tensors[i]->buffer || !ggml_backend_buffer_is_hexagon(tensors[i]->buffer)) {
            પાછા ખોટું;
        }
        જો (tensors[i]->type != tensors[0]->type) {
            પાછા ખોટું;
        }
        જો (!ggml_is_contiguous(tensors[i])) {
            પાછા ખોટું;
        }
        જો (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            પાછા ખોટું;
        }
    }

    જો (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        પાછા ખોટું;
    }

    માટે (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        સંરચના htp_allreduce_kernel_params kparams;
        જો (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, ખોટું, ખોટું, /*is_shard_ok=*/ ખોટું, &kparams)) {
            પાછા ખોટું;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    માટે (size_t i = 1; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        જો ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    જો (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_entry = max_seq;
    જો (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    માટે (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<અચલ ggml_tensor *> data_tensors(n_backends);
    std::vector<અચલ ggml_tensor *> sync_tensors(n_backends);
    માટે (size_t i = 0; i < n_backends; i++) {
        data_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    માટે (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], data_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        માટે (size_t j = 0; j < n_backends; j++) {
            જો (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    પાછા સાચું;
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_hexagon_split_buffer_type(પૂર્ણાંક main_device, અચલ દશાંશ * tensor_split) {
    GGML_UNUSED(tensor_split);
    auto reg = ggml_backend_hexagon_reg();
    auto dev = ggml_backend_reg_dev_get(reg, main_device);
    જો (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    જો (!dev) પાછા nullptr;
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    પાછા &dev_ctx->buffer_type;
}

સ્થિર શૂન્ય * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, અચલ અક્ષર * name) {
    GGML_UNUSED(reg);
    જો (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        પાછા (શૂન્ય *) ggml_backend_hexagon_split_buffer_type;
    }
    જો (strcmp(name, "ggml_backend_comm_init") == 0) {
        પાછા (શૂન્ય *) ggml_backend_hexagon_comm_init;
    }
    જો (strcmp(name, "ggml_backend_comm_free") == 0) {
        પાછા (શૂન્ય *) ggml_backend_hexagon_comm_free;
    }
    જો (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        પાછા (શૂન્ય *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    પાછા ખાલી;
}

template<typename T> std::vector<T> str_to_vec(અચલ અક્ષર* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    જ્યારે (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    પાછા v;
}

template<typename T, પૂર્ણાંક BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    માટે (auto i : v) { ss << i << ','; }
    auto str = ss.str(); str.pop_back(); // drop last comma
    પાછા str;
}

સ્થિર શૂન્ય ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, તાર્કિક discovery_supported, અચલ std::unordered_map<પૂર્ણાંક, fastrpc_domain> & cdsp_map) {
    જો (discovery_supported) {
        auto it = cdsp_map.find(cfg.physical_idx);
        જો (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } નહીંતર {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } નહીંતર {
        પસંદગી (cfg.physical_idx) {
            સ્થિતિ 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                તોડો;
            સ્થિતિ 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                તોડો;
            મૂળભૂત:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                તોડો;
        }
    }
    માટે (auto & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
સ્થિર શૂન્ય ggml_hexagon_discover_devices() {
    std::unordered_map<પૂર્ણાંક, fastrpc_domain> cdsp_map;
    તાર્કિક discovery_supported = ખોટું;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    પૂર્ણાંક err = remote_system_request(&domain_info);
    જો (err == AEE_SUCCESS && domain_info.sys.અંક_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.અંક_domains);
        domain_info.sys.domains     = domains.data();
        domain_info.sys.max_domains = (પૂર્ણાંક) domains.size();

        err = remote_system_request(&domain_info);
        જો (err == AEE_SUCCESS) {
            discovery_supported = સાચું;
            અચલ પૂર્ણાંક n_domains = std::min(domain_info.sys.અંક_domains, (પૂર્ણાંક) domains.size());
            માટે (પૂર્ણાંક i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (પૂર્ણાંક) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                જો (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (પૂર્ણાંક) domains[i].type);
                    ચાલુ;
                }
                જો (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    ચાલુ;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } નહીંતર {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (unsigned) err);
        }
    } નહીંતર જો (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (unsigned) err);
    }

    // Populate domain IDs and names for all configured devices
    માટે (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

સ્થિર શૂન્ય ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_Q4_0 == (unsigned પૂર્ણાંક) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_Q4_1 == (unsigned પૂર્ણાંક) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_Q8_0 == (unsigned પૂર્ણાંક) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_MXFP4 == (unsigned પૂર્ણાંક) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_IQ4_NL == (unsigned પૂર્ણાંક) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_Q4_K == (unsigned પૂર્ણાંક) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_Q5_K == (unsigned પૂર્ણાંક) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_Q6_K == (unsigned પૂર્ણાંક) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_Q3_K == (unsigned પૂર્ણાંક) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_Q2_K == (unsigned પૂર્ણાંક) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_Q1_0 == (unsigned પૂર્ણાંક) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_Q2_0 == (unsigned પૂર્ણાંક) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned પૂર્ણાંક) HTP_TYPE_BF16 == (unsigned પૂર્ણાંક) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    અચલ અક્ષર * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    અચલ અક્ષર * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    અચલ અક્ષર * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    અચલ અક્ષર * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    અચલ અક્ષર * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    અચલ અક્ષર * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    અચલ અક્ષર * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    અચલ અક્ષર * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    અચલ અક્ષર * str_etm      = getenv("GGML_HEXAGON_ETM");
    અચલ અક્ષર * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    અચલ અક્ષર * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    અચલ અક્ષર * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    અચલ અક્ષર * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    અચલ અક્ષર * str_fa_head_split = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    અચલ અક્ષર * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    અચલ અક્ષર * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    અચલ અક્ષર * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    અચલ અક્ષર * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    અચલ અક્ષર * str_arch     = getenv("GGML_HEXAGON_ARCH");
    અચલ અક્ષર * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    અચલ અક્ષર * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    અચલ અક્ષર * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    અચલ અક્ષર * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    અચલ અક્ષર * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    અચલ અક્ષર * str_power_mode = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    જો (!str_arch) {
        પૂર્ણાંક err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        જો (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } નહીંતર {
            જો (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } નહીંતર જો (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } નહીંતર {
        જો (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, ખાલી, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? નવું std::regex(str_opfilter, RE_ICASE) : ખાલી;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, ખાલી, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, ખાલી, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, ખાલી, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  ખાલી, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    જો (str_power_mode) {
        પૂર્ણાંક m = htp_power_mode_from_str(str_power_mode);
        જો (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_mode);
        } નહીંતર {
            opt_power_mode = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, ખાલી, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_head_split = str_fa_head_split ? atoi(str_fa_head_split)       : opt_fa_head_split;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    જો (str_mbuf) {
        અચલ અક્ષર * p = str_mbuf;
        માટે (પૂર્ણાંક idx = 0; idx < 3 && p && *p; idx++) {
            જ્યારે (*p == ' ') p++;
            અચલ અક્ષર * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : strlen(p);
            જ્યારે (len > 0 && p[len - 1] == ' ') len--;
            જો (len > 0) {
                std::string token(p, len);
                જો (idx == 0) opt_mbuf_dyn    = parse_size(token.c_str());
                જો (idx == 1) opt_mbuf_static = parse_size(token.c_str());
                જો (idx == 2) opt_mbuf_total  = parse_size(token.c_str());
            }
            જો (!comma) તોડો;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_size(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    અચલ અક્ષર * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    જો (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    જો (str_devices && str_devices[0] != '\0') {
        તાર્કિક is_single_સંખ્યા = સાચું;
        માટે (પૂર્ણાંક i = 0; str_devices[i] != '\0'; i++) {
            જો (!isdigit((unsigned અક્ષર)str_devices[i])) {
                is_single_સંખ્યા = ખોટું;
                તોડો;
            }
        }
        જો (is_single_સંખ્યા) {
            પૂર્ણાંક n = atoi(str_devices);
            જો (n < 1) n = 1;
            જો (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            માટે (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (પૂર્ણાંક)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.clear();
            }
        } નહીંતર {
            std::string s_devices(str_devices);
            std::vector<std::string> વસ્તુઓ;
            std::string curr_item;
            પૂર્ણાંક bracket_depth = 0;
            માટે (અક્ષર ch : s_devices) {
                જો (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } નહીંતર જો (ch == ']') {
                    જો (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } નહીંતર જો (ch == ',' && bracket_depth == 0) {
                    size_t s = curr_item.find_first_not_of(" \t\r\n");
                    size_t e = curr_item.find_last_not_of(" \t\r\n");
                    જો (s != std::string::npos) {
                        વસ્તુઓ.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.clear();
                } નહીંતર {
                    curr_item += ch;
                }
            }
            size_t s = curr_item.find_first_not_of(" \t\r\n");
            size_t e = curr_item.find_last_not_of(" \t\r\n");
            જો (s != std::string::npos) {
                વસ્તુઓ.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            માટે (અચલ auto & item : વસ્તુઓ) {
                size_t b_open  = item.find('[');
                size_t b_close = item.rfind(']');

                જો (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    પૂર્ણાંક virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    જો (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        પ્રયાસ {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } પકડો (...) {
                            virt = 0;
                        }
                    } નહીંતર {
                        size_t dev_colon = dev_name.find(':');
                        જો (dev_colon != std::string::npos) {
                            પ્રયાસ {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } પકડો (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<પૂર્ણાંક> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    જ્યારે (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        જો (ps == std::string::npos) ચાલુ;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        જો (dash_pos != std::string::npos) {
                            પ્રયાસ {
                                પૂર્ણાંક p_start = std::stoi(p_part.substr(0, dash_pos));
                                પૂર્ણાંક p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                માટે (પૂર્ણાંક p = p_start; p <= p_end; p++) {
                                    જો (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } પકડો (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } નહીંતર {
                            પ્રયાસ {
                                પૂર્ણાંક p = std::stoi(p_part);
                                જો (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } પકડો (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    જો (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    જો (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        auto & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.clear();

                        માટે (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_group.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } નહીંતર {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } નહીંતર જો (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    size_t colon_pos = rest.find(':');
                    પૂર્ણાંક phys = 0;
                    પૂર્ણાંક virt = 0;
                    પ્રયાસ {
                        જો (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } નહીંતર {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } પકડો (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        ચાલુ;
                    }

                    જો (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_group.clear();
                        opt_ndev++;
                    } નહીંતર {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } નહીંતર {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", item.c_str());
                }
            }
        }
    } નહીંતર {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_group.clear();
    }

#જો defined(__ANDROID__)
    જો (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    જો (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            auto v  = str_to_vec<uint32_t>(str_profile);
            પસંદગી (v.size()) {
                સ્થિતિ 1:  opt_profile = v[0]; પાછા opt_pmu_evt; // mode with default pmu events
                સ્થિતિ 8:  opt_profile = 2;    પાછા v;           // mode with custom  pmu events
                મૂળભૂત: opt_profile = 0;    પાછા {};          // garbage input
            }}();
        જો (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = નવું ggml_hexagon_registry(reg);
}

સ્થિર અચલ સંરચના ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_ગણતરી,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(શૂન્ય) {
    સ્થિર તાર્કિક initialized = ખોટું;

    સ્થિર ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ ખાલી };

    {
        સ્થિર std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        જો (!initialized) {
            auto nErr = htpdrv_init();
            જો (nErr != AEE_SUCCESS) {
                પાછા ખાલી;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = સાચું;
    }

    પાછા &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
