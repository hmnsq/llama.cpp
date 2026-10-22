#define _USE_MATH_DEFINES

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <inttypes.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <time.h>

#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <chrono>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <thread>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <stdexcept>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <sstream>
#ಅನ್ತರ್ಭಾವ <iomanip>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <regex>
#ಅನ್ತರ್ಭಾವ <queue>
#ಅನ್ತರ್ಭಾವ <deque>
#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <initializer_list>

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
#ಅನ್ಯಥಾ
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"

#ಅನ್ತರ್ಭಾವ <AEEStdErr.h>
#ಅನ್ತರ್ಭಾವ <dspqueue.h>
#ಅನ್ತರ್ಭಾವ <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-alloc.h"
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-hexagon.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-quants.h"
#ಅನ್ತರ್ಭಾವ "htp-opnode.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/matmul-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/binary-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/flash-attn-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/unary-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/get-rows-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/set-rows-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/softmax-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/pool-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/rope-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/ssm-conv.h"
#ಅನ್ತರ್ಭಾವ "htp/gated-delta-net-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/argsort-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/concat-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/cpy-ops.h"
#ಅನ್ತರ್ಭಾವ "htp/im2col-ops.h"
#ಅನ್ತರ್ಭಾವ "htp_iface.h"
#ಅನ್ತರ್ಭಾವ "htp-drv.h"
#ಅನ್ತರ್ಭಾವ "htp-repack.h"

using intvec  = std::vector<ಪೂರ್ಣಾಙ್ಕ>;
using uintvec = std::vector<unsigned ಪೂರ್ಣಾಙ್ಕ>;
using u32vec  = std::vector<uint32_t>;

#define GGML_HEXAGON_MAX_SESSIONS          16

#define GGML_HEXAGON_FENCE_SLOT_SIZE       128

ಸಂರಚನಾ ggml_hexagon_device_config {
    ಪೂರ್ಣಾಙ್ಕ         physical_idx = 0;
    ಪೂರ್ಣಾಙ್ಕ         virtual_idx  = 0;
    ಪೂರ್ಣಾಙ್ಕ         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

ಸ್ಥಿರ ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ    opt_arch    = 0; // autodetect
ಸ್ಥಿರ size_t opt_ndev    = 1;
ಸ್ಥಿರ size_t opt_nhvx    = 0; // use all
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
ಸ್ಥಿರ size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ    opt_etm     = 0;
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ    opt_verbose = 0;
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
ಸ್ಥಿರ ತಾರ್ಕಿಕ   opt_hostbuf = ಅಸತ್ಯ;
ಸ್ಥಿರ ತಾರ್ಕಿಕ   opt_dma64   = ಅಸತ್ಯ;
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start

ಸ್ಥಿರ ಧ್ರುವ std::pair<ಧ್ರುವ ವರ್ಣ *, ಪೂರ್ಣಾಙ್ಕ> htp_power_mode_names[] = {
    { "low_power_saver",            HTP_POWER_MODE_ಕೆಳಗಿನ_POWER_SAVER },
    { "power_saver",                HTP_POWER_MODE_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_MODE_ಮೇಲಿನ_POWER_SAVER },
    { "low_balanced",               HTP_POWER_MODE_ಕೆಳಗಿನ_BALANCED },
    { "balanced",                   HTP_POWER_MODE_BALANCED },
    { "high_performance",           HTP_POWER_MODE_ಮೇಲಿನ_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_MODE_SUSTAINED_ಮೇಲಿನ_PERFORMANCE },
    { "burst",                      HTP_POWER_MODE_BURST },
    { "default",                    HTP_POWER_MODE_BURST }, // flip this to change the default mode
};

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ htp_power_mode_from_str(ಧ್ರುವ ವರ್ಣ * s) {
    ಪ್ರತಿ (auto & e : htp_power_mode_names) {
#ಯದಿ defined(_MSC_VER)
        ಯದಿ (_stricmp(s, e.first) == 0) {
#ಅನ್ಯಥಾ
        ಯದಿ (strcasecmp(s, e.first) == 0) {
#endif
            ಪ್ರತ್ಯಾಗಮ e.second;
        }
    }
    ಪ್ರತ್ಯಾಗಮ -1;
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * htp_power_mode_to_str(ಪೂರ್ಣಾಙ್ಕ mode) {
    ಪ್ರತಿ (auto & e : htp_power_mode_names) {
        ಯದಿ (e.second == mode) {
            ಪ್ರತ್ಯಾಗಮ e.first;
        }
    }
    ಪ್ರತ್ಯಾಗಮ "unknown";
}

ಸ್ಥಿರ size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
ಸ್ಥಿರ size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
ಸ್ಥಿರ size_t opt_mbuf_total  = 0;                        // total buffer space limit (0 = unconstrained)

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ    opt_fa_head_split = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
ಸ್ಥಿರ u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ opt_opbatch  = 1280; // max number of ops in a batch
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ opt_opqueue  = 8;   // max number of pending batches
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ opt_optrace  = 0;    // trace buffer size per thread (0 means default)
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ opt_oppoll   = 0;    // polling for batch completions
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ opt_opfusion  = 1;    // enable/disable op fusion
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

ಗಣನಾ ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ADD = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ADD   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_hexagon_is_fusion_enabled(ಪೂರ್ಣಾಙ್ಕ flag) {
    ಯದಿ (opt_opfusion <= 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (opt_opfusion == 1) ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ; // 1 enables all
    ಪ್ರತ್ಯಾಗಮ (opt_opfusion & flag) != 0;
}

ಸ್ಥಿರ std::regex* opt_opfilter = ರಿಕ್ತ; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    ಯದಿ (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * status_to_str(uint32_t status) {
    ವಿಕಲ್ಪ (status) {
        ಪಕ್ಷ HTP_STATUS_OK:
            ಪ್ರತ್ಯಾಗಮ "OK";
        ಪಕ್ಷ HTP_STATUS_NO_SUPPORT:
            ಪ್ರತ್ಯಾಗಮ "NO-SUPPORT";
        ಪಕ್ಷ HTP_STATUS_INVAL_PARAMS:
            ಪ್ರತ್ಯಾಗಮ "INVAL-PARAMS";
        ಪಕ್ಷ HTP_STATUS_VTCM_TOO_SMALL:
            ಪ್ರತ್ಯಾಗಮ "VTCM-TOO-SMALL";
        ಪಕ್ಷ HTP_STATUS_INTERNAL_ERR:
            ಪ್ರತ್ಯಾಗಮ "INTERNAL-ERROR";
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ "UNKNOWN";
    }
}

// ** debug helpers

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_dump_op_exec(ಧ್ರುವ std::string &sess_name, ಧ್ರುವ htp_opnode & ಗಂಟು, ಧ್ರುವ uint32_t req_flags) {
    ಯದಿ (!opt_verbose) ಪ್ರತ್ಯಾಗಮ;

    htp_opformat fmt(ಗಂಟು);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                ಗಂಟು.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_dump_op_supp(ಧ್ರುವ std::string &sess_name, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op, ತಾರ್ಕಿಕ supp) {
    ಯದಿ (!opt_verbose) ಪ್ರತ್ಯಾಗಮ;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * htp_event_name(uint16_t id) {
    ವಿಕಲ್ಪ (id) {
        ಪಕ್ಷ HTP_TRACE_EVT_DMA:            ಪ್ರತ್ಯಾಗಮ "DMA";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_COMP:       ಪ್ರತ್ಯಾಗಮ "HVX_COMP";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_A_QUANT:    ಪ್ರತ್ಯಾಗಮ "HVX_A_QUANT";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_A_PREP:     ಪ್ರತ್ಯಾಗಮ "HVX_A_PREP";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_W_DEQUANT:  ಪ್ರತ್ಯಾಗಮ "HVX_W_DEQUANT";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_W_PREP:     ಪ್ರತ್ಯಾಗಮ "HVX_W_PREP";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_O_PROC:     ಪ್ರತ್ಯಾಗಮ "HVX_O_PROC";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_FA_QK:      ಪ್ರತ್ಯಾಗಮ "HVX_QK_FA";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_FA_SFM:     ಪ್ರತ್ಯಾಗಮ "HVX_SFM_FA";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_FA_Q_PREP:  ಪ್ರತ್ಯಾಗಮ "HVX_Q_PREP";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_FA_K_PREP:  ಪ್ರತ್ಯಾಗಮ "HVX_K_PREP";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_FA_V_PREP:  ಪ್ರತ್ಯಾಗಮ "HVX_V_PREP";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_GDN_PREP:   ಪ್ರತ್ಯಾಗಮ "HVX_GDN_PREP";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_GDN_SOLVE:  ಪ್ರತ್ಯಾಗಮ "HVX_GDN_SOLVE";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_GDN_V_PREP: ಪ್ರತ್ಯಾಗಮ "HVX_GDN_V_PREP";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_GDN_D_PREP: ಪ್ರತ್ಯಾಗಮ "HVX_GDN_D_PREP";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_GDN_OUT:    ಪ್ರತ್ಯಾಗಮ "HVX_GDN_OUT";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_GDN_STATE:  ಪ್ರತ್ಯಾಗಮ "HVX_GDN_STATE";
        ಪಕ್ಷ HTP_TRACE_EVT_HVX_GDN_REM:    ಪ್ರತ್ಯಾಗಮ "HVX_GDN_REM";
        ಪಕ್ಷ HTP_TRACE_EVT_HMX_COMP:       ಪ್ರತ್ಯಾಗಮ "HMX_COMP";
        ಪಕ್ಷ HTP_TRACE_EVT_L2FLUSH:        ಪ್ರತ್ಯಾಗಮ "L2FLUSH";
        ಪಕ್ಷ HTP_TRACE_EVT_INIT:           ಪ್ರತ್ಯಾಗಮ "INIT";
        ಪಕ್ಷ HTP_TRACE_EVT_BUFF:           ಪ್ರತ್ಯಾಗಮ "BUFF";
        ಪಕ್ಷ HTP_TRACE_EVT_FENCE:          ಪ್ರತ್ಯಾಗಮ "FENCE";
        ಸ್ವಭಾವ:                           ಪ್ರತ್ಯಾಗಮ "UNKNOWN";
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_dump_op_prof(ಧ್ರುವ std::string &sess_name, ಧ್ರುವ htp_opnode & ಗಂಟು, ಧ್ರುವ htp_prof_desc & pd) {
    ಯದಿ (!opt_profile) ಪ್ರತ್ಯಾಗಮ;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    ಧ್ರುವ uint32_t * pmu = pd.pmu;

    ವರ್ಣ pmu_str[256] = "";
    ಯದಿ (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, ಪರಿಮಾಣ(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(ಗಂಟು);
    ಭಿನ್ನ mhz = op_usec > 0 ? (ಭಿನ್ನ) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            ಗಂಟು.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_dump_batch_prof(ಧ್ರುವ std::string & sess_name, ಧ್ರುವ htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    ಭಿನ್ನ batch_mhz = rsp.usecs > 0 ? (ಭಿನ್ನ) batch_cycles / rsp.usecs : 0.0f;

    ವರ್ಣ evt_str[256] = "----";
    ಯದಿ (opt_profile == 3) {
        snprintf(evt_str, ಪರಿಮಾಣ(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (unsigned long long) batch_cycles, (unsigned long long) rsp.cycles_start, batch_mhz);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_dump_trace_events(ಧ್ರುವ std::string & sess_name, ಧ್ರುವ htp_opbatch_rsp & rsp,
                                           ಧ್ರುವ htp_trace_desc * trace_events, uint32_t n_traces) {
    ಯದಿ (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        ಪ್ರತಿ (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        ಪ್ರತಿ (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ಪ್ರತಿ (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                ಧ್ರುವ auto & e = trace_events[t * n_traces + idx];
                ತಾರ್ಕಿಕ is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

ಗಣನಾ ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_WEIGHT    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

ಸಂರಚನಾ ggml_hexagon_session;

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_matmul_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_mm_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_fused_matmul_add_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src2,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_mm_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_unary_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    uint32_t op,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_unary_kernel_params * kparams
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_ದ್ವಿಮಾನ_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    uint32_t op,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_ದ್ವಿಮಾನ_kernel_params * kparams,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src2 = nullptr
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_get_rows_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_get_rows_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_set_rows_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_set_rows_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_softmax_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_softmax_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_rope_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_rope_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_ssm_conv_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_ssm_conv_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_gated_delta_net_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_gdn_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_sort_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ತಾರ್ಕಿಕ is_ತುದಿ_k,
    ಸಂರಚನಾ htp_sort_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_pool_2d_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_pool_2d_kernel_params * kparams,
    ತಾರ್ಕಿಕ is_pool_1d
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_concat_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_concat_kernel_params * kparams
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_cpy_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_copy_kernel_params * kparams
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_im2col_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_im2col_kernel_params * kparams
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_im2col_3d_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_im2col_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_fused_mmnx_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * act,
    int32_t n_weights,
    ಸಂರಚನಾ htp_mm_kernel_params * kparams
);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_fused_mmidnx_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * act,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    int32_t n_weights,
    ಸಂರಚನಾ htp_mm_kernel_params * kparams
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_allreduce_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    ತಾರ್ಕಿಕ has_add,
    ತಾರ್ಕಿಕ is_row_bcast,
    ತಾರ್ಕಿಕ is_shard_ok,
    ಸಂರಚನಾ htp_allreduce_kernel_params * kparams
);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_rows_stride(ಧ್ರುವ int64_t * ne, ಧ್ರುವ size_t * nb, size_t * stride);
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_matmul_can_collapse(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst);
ಸ್ಥಿರ ggml_tensor ggml_hexagon_tensor_collapse_rows(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t);

ಸ್ಥಿರ ತಾರ್ಕಿಕ mm_is_hmx_eligible(ಧ್ರುವ ggml_tensor * t);
ಸ್ಥಿರ htp_op_code op_remap_to_htp(ಧ್ರುವ ggml_tensor * t);
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_supported_mul_mat_nx_kernel(ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ಸಂರಚನಾ htp_mm_kernel_params * kparams);
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_supported_mul_mat_id_nx_kernel(ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ಸಂರಚನಾ htp_mm_kernel_params * kparams);
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat(ಧ್ರುವ ggml_tensor * t);
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat_pair(ಧ್ರುವ ggml_tensor * n1, ಧ್ರುವ ggml_tensor * n2);
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat_id(ಧ್ರುವ ggml_tensor * t);
ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat_id_pair(ಧ್ರುವ ggml_tensor * n1, ಧ್ರುವ ggml_tensor * n2);

// ** backend sessions

ಸಂರಚನಾ ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_size { 0 };
    uint32_t             flags { 0 };
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_hexagon_tensor_is_fuseable(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಯದಿ (!t->extra) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    auto extra = (ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_tensor_extra *) t->extra;
    ಪ್ರತ್ಯಾಗಮ (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_hexagon_can_rt_repack(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಯದಿ (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (t->buffer && ggml_backend_buft_is_host(t->buffer->buft)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_hexagon_tensor_is_tiled(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಧ್ರುವ auto * extra = (ಧ್ರುವ ggml_hexagon_tensor_extra *) t->extra;
    ಯದಿ (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_can_rt_repack(t);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_hexagon_needs_repack(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಧ್ರುವ auto * extra = (ಧ್ರುವ ggml_hexagon_tensor_extra *) t->extra;
    ಯದಿ (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_can_rt_repack(t);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_hexagon_tensors_overlap(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * a, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * b) {
    ಧ್ರುವ uintptr_t a0 = (uintptr_t) a->data;
    ಧ್ರುವ uintptr_t b0 = (uintptr_t) b->data;
    ಧ್ರುವ uintptr_t a1 = a0 + ggml_nbytes(a);
    ಧ್ರುವ uintptr_t b1 = b0 + ggml_nbytes(b);

    ಪ್ರತ್ಯಾಗಮ a0 < b1 && b0 < a1;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_hexagon_can_row_partition(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಯದಿ (t->ne[1] > 1 && (t->nb[1] & 127) != 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (t->ne[2] > 1 && (t->nb[2] & 127) != 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (t->ne[3] > 1 && (t->nb[3] & 127) != 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸಂರಚನಾ htp_opnode;

ಸಂರಚನಾ ggml_hexagon_opbatch;
ಸಂರಚನಾ ggml_hexagon_opqueue;
ಸಂರಚನಾ ggml_hexagon_shared_buffer;
ಸಂರಚನಾ ggml_hexagon_fence_buffer;
ಸಂರಚನಾ ggml_hexagon_session;
ಸಂರಚನಾ ggml_backend_hexagon_device_context;

ಸಂರಚನಾ ggml_hexagon_mdev_group {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

ಸಂರಚನಾ ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    ಚಞ್ಚಲ uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

ಸಂರಚನಾ ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    ಚಞ್ಚಲ uint32_t *    fence_slot   = nullptr;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

ಸಂರಚನಾ ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    ಪೂರ್ಣಾಙ್ಕ              phys_idx;
    ಪೂರ್ಣಾಙ್ಕ              virt_idx;
    ತಾರ್ಕಿಕ             valid_session;
    ತಾರ್ಕಿಕ             valid_handle;
    ತಾರ್ಕಿಕ             valid_queue;
    ತಾರ್ಕಿಕ             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<ಪೂರ್ಣಾಙ್ಕ, std::unique_ptr<ggml_hexagon_shared_buffer>> cloned_buffers;
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

    ggml_hexagon_session(ಧ್ರುವ ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_ಎಣಿಕೆ = 0) noexcept(ಅಸತ್ಯ);
    ~ggml_hexagon_session() noexcept(ಸತ್ಯ);

    ಧ್ರುವ ವರ್ಣ* c_name() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ name.c_str(); }

    ಶೂನ್ಯ allocate(ಧ್ರುವ ggml_hexagon_device_config & config) noexcept(ಅಸತ್ಯ);
    ಶೂನ್ಯ release() noexcept(ಸತ್ಯ);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    ಶೂನ್ಯ      free_fence(ಶೂನ್ಯ * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, ಚಞ್ಚಲ uint32_t *> cpy_fence_slots;

    ಶೂನ್ಯ enqueue_mdev_group();
    ಶೂನ್ಯ enqueue_op(ಧ್ರುವ htp_opnode & ಗಂಟು);
    ಶೂನ್ಯ enqueue_cpy(ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst, ಧ್ರುವ ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    ಶೂನ್ಯ enqueue_fence(ಧ್ರುವ ggml_tensor * sync_tensor, uint32_t fence_seq = 0, ತಾರ್ಕಿಕ wait = ಸತ್ಯ);
    ಶೂನ್ಯ enqueue_allreduce(ಧ್ರುವ ggml_tensor * dst, ಧ್ರುವ std::vector<ಧ್ರುವ ggml_tensor *> & src_tensors,
                           ಧ್ರುವ std::vector<ಧ್ರುವ ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);

    ಶೂನ್ಯ start_batch();
    ಶೂನ್ಯ flush_sync(ತಾರ್ಕಿಕ all = ಸತ್ಯ);
    ಶೂನ್ಯ flush_async();
    ಶೂನ್ಯ flush_batch(size_t min_ops = 1);
    ಶೂನ್ಯ flush_peers();
    ಶೂನ್ಯ flush_pending(ತಾರ್ಕಿಕ all = ಸತ್ಯ);

    ggml_hexagon_shared_buffer * mmap_tensor(ಧ್ರುವ ggml_tensor * t);
    ತಾರ್ಕಿಕ clone_buffer(ಧ್ರುವ ggml_hexagon_shared_buffer*);
    ಶೂನ್ಯ release_buffer(ಧ್ರುವ ggml_hexagon_shared_buffer*);
    ಶೂನ್ಯ unclone_buffer(ಧ್ರುವ ggml_hexagon_shared_buffer*);

    ಶೂನ್ಯ add_peer(ggml_hexagon_session * peer) {
        ಯದಿ (ಏತತ್->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } ಅನ್ಯಥಾ {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

ಸಂರಚನಾ ggml_backend_hexagon_device_context {
    ಪೂರ್ಣಾಙ್ಕ                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_buffer_type buffer_type       = {};
    ggml_backend_buffer_type host_buffer_type  = {};
    ggml_backend_buffer_type fence_buffer_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(ಪೂರ್ಣಾಙ್ಕ dev_id, ಧ್ರುವ ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    ಧ್ರುವ ವರ್ಣ * c_name() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ config.name.c_str(); }

    ggml_hexagon_session * session() {
        ಯದಿ (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        ಪ್ರತ್ಯಾಗಮ sess.get();
    }
};

ಸಂರಚನಾ ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(ಧ್ರುವ std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        ಏತತ್->dev_ctx = dev_ctx;
        ಏತತ್->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

ಸಂರಚನಾ ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    ಪೂರ್ಣಾಙ್ಕ       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        ಯದಿ (!base) {
            ಕ್ಷಿಪ std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        ಯದಿ (fd < 0) {
            rpcmem_free(base);
            ಕ್ಷಿಪ std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        ಏತತ್->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        ಯದಿ (base) {
            rpcmem_free(base);
        }
    }
};

ಸಂರಚನಾ ggml_hexagon_shared_buffer {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    ತಾರ್ಕಿಕ     mapped;
    ತಾರ್ಕಿಕ     pinned;
    ತಾರ್ಕಿಕ     extended;

    ಧ್ರುವ ವರ್ಣ * c_name() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ sess->c_name(); }
    uint8_t *    base()   ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ mem ? mem->base : nullptr; }
    size_t       size()   ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ mem ? mem->size : 0;  }
    ಪೂರ್ಣಾಙ್ಕ          fd()     ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ mem ? mem->fd   : -1; }

    ಶೂನ್ಯ mmap(ತಾರ್ಕಿಕ extended = ಅಸತ್ಯ) {
        ಯದಿ (!ಏತತ್->mem)   ಪ್ರತ್ಯಾಗಮ;
        ಯದಿ (ಏತತ್->mapped) ಪ್ರತ್ಯಾಗಮ;

        GGML_ASSERT(!ಏತತ್->pinned || !extended);

        ಏತತ್->extended = extended;
        fastrpc_map_flags flags = ಏತತ್->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        ಪೂರ್ಣಾಙ್ಕ err = fastrpc_mmap(sess->domain_id, fd(), (ಶೂನ್ಯ *) base(), 0, size(), flags);
        ಯದಿ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (unsigned) err);
            ಕ್ಷಿಪ std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (ಶೂನ್ಯ *) base(), size(), fd(), pinned, extended);

        ಏತತ್->mapped = ಸತ್ಯ;
    }

    ಶೂನ್ಯ unmap() {
        ಯದಿ (!ಏತತ್->mapped) ಪ್ರತ್ಯಾಗಮ;

        ಯದಿ (!ಏತತ್->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        ಯದಿ (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (ಶೂನ್ಯ *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (ಶೂನ್ಯ *) base(), size(), fd());

        ಏತತ್->mapped = ಅಸತ್ಯ;
    }

    ಶೂನ್ಯ alloc(size_t size) {
        ಯದಿ (ಏತತ್->mem) ಪ್ರತ್ಯಾಗಮ;

        ಏತತ್->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (ಶೂನ್ಯ *) base(), ಏತತ್->size(), fd(), (ಪೂರ್ಣಾಙ್ಕ) pinned);
        ಯದಿ (ಏತತ್->pinned) {
            mmap();
        }
    }

    ಶೂನ್ಯ ಮುಕ್ತ() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (ಶೂನ್ಯ *) base(), size(), fd());
        ಏತತ್->mem  = nullptr;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, ತಾರ್ಕಿಕ pinned = ಅಸತ್ಯ) {
        ಏತತ್->sess     = sess;
        ಏತತ್->mapped   = ಅಸತ್ಯ;
        ಏತತ್->pinned   = pinned;
        ಏತತ್->extended = ಅಸತ್ಯ;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_offset = (size + 4095) & ~4095;
        size_t total_size   = guard_offset + 4096;
        ಯದಿ (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            total_size = (total_size + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(total_size);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, ಧ್ರುವ ggml_hexagon_shared_buffer & other) {
        ಏತತ್->sess   = sess;
        ಏತತ್->mem    = other.mem;
        ಏತತ್->mapped = ಅಸತ್ಯ;
        ಏತತ್->pinned = other.pinned;
        ಏತತ್->extended = other.extended;
    }

    ~ggml_hexagon_shared_buffer() {
        ಮುಕ್ತ();
        ಪ್ರತಿ (auto * extra : tensor_extra) {
            delete extra;
        }
    }
};

ಸಂರಚನಾ ggml_hexagon_fence_buffer : ಸಾರ್ವಜನಿಕ ggml_hexagon_shared_buffer {
    uint32_t              slot_ಎಣಿಕೆ = 0;
    uint32_t              slot_head  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_buffer   backend_buffer{};

    ggml_hexagon_fence_buffer(ggml_hexagon_session * sess, ggml_backend_buffer_type_t buft, size_t size)
        : ggml_hexagon_shared_buffer(sess, size, ಅಸತ್ಯ /* pinned */),
          slot_ಎಣಿಕೆ(size / GGML_HEXAGON_FENCE_SLOT_SIZE),
          slot_head(0) {
        backend_buffer.buft    = buft;
        backend_buffer.context = static_cast<ggml_hexagon_shared_buffer *>(ಏತತ್);
        backend_buffer.size    = size;
        mmap(ಅಸತ್ಯ);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        ಯದಿ (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        } ಅನ್ಯಥಾ ಯದಿ (slot_head + n_slots <= slot_ಎಣಿಕೆ) {
            uint32_t slot = slot_head;
            slot_head += n_slots;
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        }
        ಯದಿ (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_SLOT_SIZE);
        }
        ಪ್ರತ್ಯಾಗಮ ptr;
    }

    ಶೂನ್ಯ free_slot(ಶೂನ್ಯ * ptr, uint32_t n_slots = 1) {
        ಯದಿ (!ptr) ಪ್ರತ್ಯಾಗಮ;
        uint32_t slot = ((uint8_t *) ptr - base()) / GGML_HEXAGON_FENCE_SLOT_SIZE;
        ಪ್ರತಿ (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

ಅನ್ತಃಸ್ಥ uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    ಪ್ರತ್ಯಾಗಮ ptr;
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_hexagon_session::free_fence(ಶೂನ್ಯ * ptr, uint32_t n_slots) {
    ಯದಿ (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    sbuf->sess->unclone_buffer(sbuf);
    delete sbuf;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    ಪ್ರತ್ಯಾಗಮ sbuf->base();
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (ಶೂನ್ಯ *) sbuf->base(), tensor->data, ggml_nbytes(tensor));

    auto extra = ನೂತನ ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   ಧ್ರುವ ಶೂನ್ಯ *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    ಯದಿ (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        ಯದಿ (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, data, offset, size, (ಪೂರ್ಣಾಙ್ಕ) buffer->usage, extra->flags);

    ಯದಿ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy((ವರ್ಣ *) tensor->data + offset, data, size);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        htp_repack_tensor_tiled(tensor, data, size);
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    memcpy(extra->shadow_buf.data() + offset, data, size);
    extra->shadow_size += size;

    ಯದಿ (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   ಧ್ರುವ ggml_tensor *   tensor,
                                                   ಶೂನ್ಯ *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, data, offset, size, (ಪೂರ್ಣಾಙ್ಕ) buffer->usage, extra->flags);

    ಯದಿ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy(data, (ಧ್ರುವ ವರ್ಣ *) tensor->data + offset, size);
        ಪ್ರತ್ಯಾಗಮ;
    }

    htp_unpack_tensor_tiled(data, tensor, offset, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src,
                                                   ಸಂರಚನಾ ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                          ggml_tensor *         tensor,
                                                          ಧ್ರುವ ಶೂನ್ಯ *          data,
                                                          size_t                offset,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    ಯದಿ (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        ಯದಿ (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (ಪೂರ್ಣಾಙ್ಕ) buffer->usage, extra->flags);

    ಯದಿ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (ಧ್ರುವ uint8_t *) data + i * stride_data, size);
        }
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
        memcpy(extra->shadow_buf.data() + offset + i * stride_tensor, (ಧ್ರುವ uint8_t *) data + i * stride_data, size);
    }
    extra->shadow_size += n_copies * size;

    ಯದಿ (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                      ಧ್ರುವ ggml_tensor *   tensor,
                                                      ಶೂನ್ಯ *                data,
                                                      size_t                offset,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (ಪೂರ್ಣಾಙ್ಕ) buffer->usage);

    ಯದಿ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *)data + i * stride_data, (ಧ್ರುವ uint8_t *)tensor->data + offset + i * stride_tensor, size);
        }
        ಪ್ರತ್ಯಾಗಮ;
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

    ಪ್ರತಿ (size_t i = 0; i < n_copies; i++) {
        memcpy((uint8_t *) data + i * stride_data, temp_buf.data() + i * stride_tensor, size);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (ಶೂನ್ಯ *) sbuf->base(), sbuf->size());
    memset(sbuf->base(), value, sbuf->size());
}

ಸ್ಥಿರ ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ ರಿಕ್ತ,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ ರಿಕ್ತ,
};

// ** backend buffer type

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_host_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                        ggml_tensor *         tensor,
                                                        ಧ್ರುವ ಶೂನ್ಯ *          data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy((ವರ್ಣ *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                        ಧ್ರುವ ggml_tensor *   tensor,
                                                        ಶೂನ್ಯ *                data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy(data, (ಧ್ರುವ ವರ್ಣ *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ ರಿಕ್ತ,
    /* .set_tensor      = */ ggml_backend_hexagon_host_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_buffer_get_tensor,
    /* .set_tensor_2d   = */ ರಿಕ್ತ,
    /* .get_tensor_2d   = */ ರಿಕ್ತ,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ ರಿಕ್ತ,
};

// ** backend buffer type

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    ಪ್ರತ್ಯಾಗಮ static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    ಯದಿ (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಯತ್ನ {
        ggml_hexagon_shared_buffer * sbuf = ನೂತನ ggml_hexagon_shared_buffer(sess, size, ಅಸತ್ಯ);
        ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } ಗೃಹಾಣ (ಧ್ರುವ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    ಯದಿ (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಯತ್ನ {
        ggml_hexagon_shared_buffer * sbuf = ನೂತನ ggml_hexagon_shared_buffer(sess, size, ಅಸತ್ಯ);
        ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_host_buffer_interface, sbuf, size);
    } ಗೃಹಾಣ (ಧ್ರುವ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
}

ಸ್ಥಿರ size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ 128;  // HVX alignment
    GGML_UNUSED(buft);
}

ಸ್ಥಿರ size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಯದಿ (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_size = htp_tiled_row_size(t->type, ne0) * ne1 * ne2 * ne3;
        ಪ್ರತ್ಯಾಗಮ std::max(tiled_size, ggml_nbytes(t));
    }
    ಪ್ರತ್ಯಾಗಮ ggml_nbytes(t);
}

ಸ್ಥಿರ size_t parse_size(ಧ್ರುವ ವರ್ಣ * str, size_t default_unit = 1024 * 1024) {
    ಯದಿ (!str || str[0] == '\0') {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    ವರ್ಣ * end = ರಿಕ್ತ;
    ದ್ವಿಗುಣ val = strtod(str, &end);
    ಯದಿ (val < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    ಯದಿ (end && *end) {
        ಯಾವತ್ (*end == ' ') end++;
        ಯದಿ (*end == 'k' || *end == 'K') {
            ಪ್ರತ್ಯಾಗಮ (size_t) (val * 1024);
        } ಅನ್ಯಥಾ ಯದಿ (*end == 'm' || *end == 'M') {
            ಪ್ರತ್ಯಾಗಮ (size_t) (val * 1024 * 1024);
        } ಅನ್ಯಥಾ ಯದಿ (*end == 'g' || *end == 'G') {
            ಪ್ರತ್ಯಾಗಮ (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    ಪ್ರತ್ಯಾಗಮ (size_t) (val * default_unit);
}

ಸ್ಥಿರ size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ opt_mbuf_dyn;
    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    GGML_UNUSED(buft);
}

ಸಂರಚನಾ ggml_backend_hexagon_alloc_buffer_n_plan_item {
    size_t size;
    ಪೂರ್ಣಾಙ್ಕ    first;
    ಪೂರ್ಣಾಙ್ಕ    last;
};

using ggml_backend_hexagon_alloc_buffer_n_plan_t = std::vector<ggml_backend_hexagon_alloc_buffer_n_plan_item>;

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_hexagon_kv_layer_suffix(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಯದಿ (strncmp(t->name, "cache_", 6) != 0) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಧ್ರುವ ವರ್ಣ * p = strstr(t->name, "_l");
    ಯದಿ (!p || !isdigit((unsigned ವರ್ಣ)p[2])) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    ಪ್ರತ್ಯಾಗಮ p;
}

ಸಂರಚನಾ ggml_backend_hexagon_alloc_unit {
    size_t size;
    ಪೂರ್ಣಾಙ್ಕ    first;
    ಪೂರ್ಣಾಙ್ಕ    last;
};

ಸ್ಥಿರ ggml_backend_hexagon_alloc_buffer_n_plan_t ggml_backend_hexagon_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, ಸಂರಚನಾ ggml_tensor ** tensors, ಪೂರ್ಣಾಙ್ಕ n_tensors) {
    ggml_backend_hexagon_alloc_buffer_n_plan_t plan;

    ಧ್ರುವ size_t alignment = ggml_backend_buft_get_alignment(buft);
    ಧ್ರುವ size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    ಪೂರ್ಣಾಙ್ಕ i = 0;
    ಯಾವತ್ (i < n_tensors) {
        ಸಂರಚನಾ ggml_tensor * t = tensors[i];
        size_t unit_size = 0;
        ಪೂರ್ಣಾಙ್ಕ unit_first = i;
        ಪೂರ್ಣಾಙ್ಕ unit_last = i + 1;

        ಯದಿ (t->data == ರಿಕ್ತ && t->view_src == ರಿಕ್ತ) {
            unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        ಧ್ರುವ ವರ್ಣ * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        ಯಾವತ್ (unit_last < n_tensors) {
            ಸಂರಚನಾ ggml_tensor * next = tensors[unit_last];

            ಯದಿ (next->view_src != ರಿಕ್ತ) {
                unit_last++;
                ಅನುವರ್ತ;
            }

            ಯದಿ (layer_suffix != ರಿಕ್ತ) {
                ಧ್ರುವ ವರ್ಣ * next_suffix = ggml_hexagon_kv_layer_suffix(next);
                ಯದಿ (next_suffix != ರಿಕ್ತ && strcmp(layer_suffix, next_suffix) == 0) {
                    ಯದಿ (next->data == ರಿಕ್ತ) {
                        unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, next), alignment);
                    }
                    unit_last++;
                    ಅನುವರ್ತ;
                }
            }

            ಭಙ್ಗ;
        }

        units.push_back({ unit_size, unit_first, unit_last });
        i = unit_last;
    }

    size_t cur_buf_size  = 0;
    ಪೂರ್ಣಾಙ್ಕ    cur_buf_first = 0;

    ಪ್ರತಿ (ಧ್ರುವ auto & unit : units) {
        ಯದಿ (unit.size == 0) {
            ಅನುವರ್ತ;
        }

        ಯದಿ (cur_buf_size > 0 && (cur_buf_size + unit.size) > max_size) {
            plan.push_back({ cur_buf_size, cur_buf_first, unit.first });
            cur_buf_size  = 0;
            cur_buf_first = unit.first;
        }

        cur_buf_size += unit.size;
    }

    ಯದಿ (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, cur_buf_first, n_tensors });
    }

    ಪ್ರತ್ಯಾಗಮ plan;
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, ಸಂರಚನಾ ggml_tensor ** tensors, ಪೂರ್ಣಾಙ್ಕ n_tensors) {
    ಧ್ರುವ ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    ಪ್ರತಿ (ಧ್ರುವ auto & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        ಯದಿ (buffer == ರಿಕ್ತ) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            ಪ್ರತಿ (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
        }

        ಸಂರಚನಾ ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        ಸಂರಚನಾ ggml_tensor * t_failed = ರಿಕ್ತ;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = item.first; j < item.last; j++) {
            ಸಂರಚನಾ ggml_tensor * t = tensors[j];
            ಯದಿ (t->data == ರಿಕ್ತ) {
                ಯದಿ (t->view_src == ರಿಕ್ತ) {
                    ಯದಿ (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ಭಙ್ಗ;
                    }
                } ಅನ್ಯಥಾ ಯದಿ (t->buffer == ರಿಕ್ತ) {
                    ಯದಿ (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ಭಙ್ಗ;
                    }
                }
            } ಅನ್ಯಥಾ {
                ಯದಿ (t->view_src != ರಿಕ್ತ && t->buffer == ರಿಕ್ತ) {
                    ಯದಿ (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ಭಙ್ಗ;
                    }
                }
            }
        }
        ಯದಿ (t_failed != ರಿಕ್ತ) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            ಪ್ರತಿ (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
        }

        buffers.push_back(buffer);
    }

    ಯದಿ (buffers.empty()) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಯದಿ (buffers.size() == 1) {
        ಪ್ರತ್ಯಾಗಮ buffers[0];
    }

    ಪ್ರತ್ಯಾಗಮ ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

ಸ್ಥಿರ size_t ggml_backend_hexagon_buffer_type_get_alloc_size_n(
        ggml_backend_buffer_type_t buft, ಸಂರಚನಾ ggml_tensor ** tensors, ಪೂರ್ಣಾಙ್ಕ n_tensors) {
    ಧ್ರುವ ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    ಪ್ರತಿ (ಧ್ರುವ auto & item : plan) {
        total += item.size;
    }
    ಪ್ರತ್ಯಾಗಮ total;
}

ಸ್ಥಿರ ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_buffer_type_is_host,
};

ಸ್ಥಿರ ggml_backend_buffer_type_i ggml_backend_hexagon_host_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_host_buffer_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(ಪೂರ್ಣಾಙ್ಕ dev_id, ಧ್ರುವ ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    buffer_type.device  = dev;
    buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    buffer_type.context = ನೂತನ ggml_backend_hexagon_buffer_type_context(config.name, ಏತತ್);

    host_buffer_type.device  = dev;
    host_buffer_type.iface   = ggml_backend_hexagon_host_buffer_type_interface;
    host_buffer_type.context = ನೂತನ ggml_backend_hexagon_buffer_type_context(config.name + "-HOST", ಏತತ್);

    fence_buffer_type.device  = dev;
    fence_buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    fence_buffer_type.context = ನೂತನ ggml_backend_hexagon_buffer_type_context(config.name + "-FENCE", ಏತತ್);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(host_buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(fence_buffer_type.context);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_buffer_is_hexagon(ಧ್ರುವ ಸಂರಚನಾ ggml_backend_buffer * b) {
    ಪ್ರತ್ಯಾಗಮ b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

ಸಂರಚನಾ ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ>                b_map; // buffer fd   to index
    std::unordered_map<ಧ್ರುವ ggml_tensor*, ಪೂರ್ಣಾಙ್ಕ> t_map; // tensor ptr  to index
    std::unordered_multimap<ಶೂನ್ಯ*, ಪೂರ್ಣಾಙ್ಕ>         d_map; // tensor data to index

    unsigned ಪೂರ್ಣಾಙ್ಕ n_bufs;     // num buffers in the batch
    unsigned ಪೂರ್ಣಾಙ್ಕ n_tens;     // num tensors ...
    unsigned ಪೂರ್ಣಾಙ್ಕ n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    unsigned ಪೂರ್ಣಾಙ್ಕ n_bufs_max;
    unsigned ಪೂರ್ಣಾಙ್ಕ n_tens_max;
    unsigned ಪೂರ್ಣಾಙ್ಕ n_ops_max;
    size_t       b_vmem_max;

    ಶೂನ್ಯ reset() {
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
        ಏತತ್->sess = sess;

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

    ತಾರ್ಕಿಕ empty() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ n_ops == 0; }

    // add buffer and return its index
    ಪೂರ್ಣಾಙ್ಕ add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        auto it = b_map.find(sbuf->fd());
        ಯದಿ (it != b_map.end()) { ಪ್ರತ್ಯಾಗಮ it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        ಪೂರ್ಣಾಙ್ಕ bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        ಯದಿ (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (ಶೂನ್ಯ*) sbuf->base(), (size_t) b.size, b_vmem);

        ಪ್ರತ್ಯಾಗಮ bi;
    }

    ತಾರ್ಕಿಕ same_shape(ಧ್ರುವ htp_tensor * h, ಧ್ರುವ ggml_tensor * t) ಧ್ರುವ {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;

        ಧ್ರುವ ತಾರ್ಕಿಕ is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ಧ್ರುವ int64_t ne0_padded = hex_round_up(t->ne[0], 32);
        ಧ್ರುವ int64_t ne1_padded = hex_round_up(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_row_size(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? ಪರಿಮಾಣ(ggml_fp16_t) : t->nb[0];
        ಗಣನಾ ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        ಪ್ರತ್ಯಾಗಮ (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    ಪೂರ್ಣಾಙ್ಕ add_tensor(ಧ್ರುವ ggml_tensor * t) {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;
        auto sbuf  = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        ಪ್ರತಿ (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            ಯದಿ (same_shape(h, t)) { ಪ್ರತ್ಯಾಗಮ it->second; }
        }

        // Lookup by tensor ptr
        auto it = t_map.find(t);
        ಯದಿ (it != t_map.end()) { ಪ್ರತ್ಯಾಗಮ it->second; }

        // Add new tensor to the batch
        ಪೂರ್ಣಾಙ್ಕ ti = n_tens++;
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

        ಧ್ರುವ ತಾರ್ಕಿಕ is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ಯದಿ (is_tiled) {
            ಯದಿ (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = ಪರಿಮಾಣ(ggml_fp16_t);
            } ಅನ್ಯಥಾ {
                h.nb[0] = t->nb[0];
            }
            ಧ್ರುವ int64_t ne0_padded = hex_round_up(t->ne[0], 32);
            ಧ್ರುವ int64_t ne1_padded = hex_round_up(t->ne[1], 32);
            h.nb[1] = htp_tiled_row_size(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_size  = h.size;
        } ಅನ್ಯಥಾ {
            h.size  = t_size;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        ಯದಿ ((extra->flags & GGML_HEXAGON_TENSOR_WEIGHT) != 0) {
            h.flags |= HTP_TENSOR_WEIGHT;
        }
        ಯದಿ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        ಯದಿ ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (ಶೂನ್ಯ*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        ಪ್ರತ್ಯಾಗಮ ti;
    }

    ತಾರ್ಕಿಕ fit_op(ಧ್ರುವ htp_opnode & ಗಂಟು) ಧ್ರುವ {
        ಯದಿ (n_ops >= n_ops_max) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        ಪೂರ್ಣಾಙ್ಕ seen_bufs[HTP_OP_MAX_BUFS];
        ಪೂರ್ಣಾಙ್ಕ n_seen_bufs = 0;

        auto fit_tensor = [&](ಧ್ರುವ ggml_tensor *t) {
            ಯದಿ (!t) ಪ್ರತ್ಯಾಗಮ;
            ಯದಿ (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                ಪೂರ್ಣಾಙ್ಕ fd = sbuf->fd();
                ಯದಿ (!b_map.count(fd)) {
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_seen_bufs; i++) {
                        ಯದಿ (seen_bufs[i] == fd) ಪ್ರತ್ಯಾಗಮ;
                    }
                    ಯದಿ (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    ಯದಿ (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        ಪ್ರತಿ (ಧ್ರುವ auto * src : ಗಂಟು.get_inputs()) {
            fit_tensor(src);
        }
        ಪ್ರತಿ (ಧ್ರುವ auto * output : ಗಂಟು.get_outputs()) {
            fit_tensor(output);
        }

        ಯದಿ ((extra_bufs + n_bufs) > n_bufs_max) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ ((extra_tens + n_tens) > n_tens_max) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ ((extra_vmem + b_vmem) > b_vmem_max) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    // assumes that fit_op() was called first and returned true
    ಶೂನ್ಯ add_op(ಧ್ರುವ htp_opnode & ಗಂಟು) {
        // Add new op

        unsigned ಪೂರ್ಣಾಙ್ಕ n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = ಗಂಟು;

        htp_op_desc &o = h_ops[n];
        memcpy(o.params,        ಗಂಟು.ಗಂಟು->op_params, ಪರಿಮಾಣ(ಗಂಟು.ಗಂಟು->op_params));
        memcpy(o.kernel_params, ಗಂಟು.kernel_params,   ಪರಿಮಾಣ(o.kernel_params));
        o.opcode = ಗಂಟು.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        auto inputs = ಗಂಟು.get_inputs();
        ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        auto outputs = ಗಂಟು.get_outputs();
        ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    ಶೂನ್ಯ sort_buffers() {
        ಯದಿ (n_bufs <= 1) ಪ್ರತ್ಯಾಗಮ;

        std::vector<ಪೂರ್ಣಾಙ್ಕ> order(n_bufs);
        ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_bufs; i++) { order[i] = (ಪೂರ್ಣಾಙ್ಕ) i; }

        std::stable_sort(order.begin(), order.end(), [&](ಪೂರ್ಣಾಙ್ಕ a, ಪೂರ್ಣಾಙ್ಕ b) {
            ಪ್ರತ್ಯಾಗಮ h_bufs[a].size > h_bufs[b].size;
        });

        ತಾರ್ಕಿಕ already_sorted = ಸತ್ಯ;
        ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_bufs; i++) {
            ಯದಿ (order[i] != (ಪೂರ್ಣಾಙ್ಕ) i) {
                already_sorted = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
        ಯದಿ (already_sorted) ಪ್ರತ್ಯಾಗಮ;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ new_bi = 0; new_bi < n_bufs; new_bi++) {
            ಪೂರ್ಣಾಙ್ಕ old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    ಶೂನ್ಯ update_mdev_group(uint32_t mdev_idx) {
        ಯದಿ (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    ತಾರ್ಕಿಕ try_fuse_common(std::initializer_list<ಧ್ರುವ ggml_tensor *> tensors) ಧ್ರುವ {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        ಪೂರ್ಣಾಙ್ಕ seen_bufs[HTP_OP_MAX_BUFS];
        ಪೂರ್ಣಾಙ್ಕ n_seen_bufs = 0;

        ಪ್ರತಿ (ಧ್ರುವ auto * t : tensors) {
            ಯದಿ (!t || t_map.count(t)) {
                ಅನುವರ್ತ;
            }
            extra_tens++;
            auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            ಪೂರ್ಣಾಙ್ಕ fd = sbuf->fd();
            ಯದಿ (!b_map.count(fd)) {
                ತಾರ್ಕಿಕ found = ಅಸತ್ಯ;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_seen_bufs; i++) {
                    ಯದಿ (seen_bufs[i] == fd) {
                        found = ಸತ್ಯ;
                        ಭಙ್ಗ;
                    }
                }
                ಯದಿ (!found) {
                    ಯದಿ (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    ಯದಿ (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        ಯದಿ ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ try_fuse_common(ಧ್ರುವ ggml_tensor * t1, ಧ್ರುವ ggml_tensor * t2) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ try_fuse_common({t1, t2});
    }

    ತಾರ್ಕಿಕ try_fuse_allreduce_add(ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ಯದಿ (n_ops == 0 || opt_ar_select != 2) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ (ಗಂಟು.opcode != HTP_OP_ADD) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        htp_opnode & last_ಗಂಟು = ops[n_ops - 1];
        ಯದಿ (last_ಗಂಟು.opcode != HTP_OP_ALLREDUCE) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        auto * ar_kparams = (ಸಂರಚನಾ htp_allreduce_kernel_params *) last_ಗಂಟು.kernel_params;
        ಧ್ರುವ uint32_t rank    = (uint32_t) ar_kparams->rank;
        ಧ್ರುವ uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        ಧ್ರುವ ggml_tensor * ar_local = last_ಗಂಟು.inputs[rank];
        ಧ್ರುವ ggml_tensor * add_src0 = ಗಂಟು.src0();
        ಧ್ರುವ ggml_tensor * add_src1 = ಗಂಟು.src1();
        ಧ್ರುವ ggml_tensor * add_dst  = ಗಂಟು.dst();

        ಯದಿ (!ggml_hexagon_tensor_is_fuseable(ar_local)) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಧ್ರುವ ggml_tensor * res_tensor;
        ಯದಿ (add_src0 == ar_local || add_src0->data == ar_local->data) {
            res_tensor = add_src1;
        } ಅನ್ಯಥಾ ಯದಿ (add_src1 == ar_local || add_src1->data == ar_local->data) {
            res_tensor = add_src0;
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (ar_local->type != res_tensor->type) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಧ್ರುವ ತಾರ್ಕಿಕ is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        ಧ್ರುವ ತಾರ್ಕಿಕ is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        ಯದಿ (!is_same_shape && !is_row_bcast) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಯದಿ (is_same_shape) {
            ಯದಿ (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
        ಯದಿ (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // scatter is only valid for in-place add within max outputs
        ಧ್ರುವ ತಾರ್ಕಿಕ is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

        ಸಂರಚನಾ htp_allreduce_kernel_params new_kparams;
        ಯದಿ (!ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, ಸತ್ಯ, is_row_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ತಾರ್ಕಿಕ scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

        ಪ್ರತಿ (uint32_t r = 0; r < n_ranks; r++) {
            ಧ್ರುವ ggml_tensor * ar_src = last_ಗಂಟು.inputs[r];
            ಯದಿ (!ggml_hexagon_tensors_overlap(add_dst, ar_src)) {
                ಅನುವರ್ತ;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            ಯದಿ (scatter_ok && r == rank) {
                ಅನುವರ್ತ;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (!try_fuse_common(res_tensor, add_dst)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        last_ಗಂಟು.opcode = HTP_OP_ALLREDUCE_ADD;
        last_ಗಂಟು.name   = "ALLREDUCE+ADD";
        last_ಗಂಟು.inputs.push_back(res_tensor);
        last_ಗಂಟು.outputs.clear();
        last_ಗಂಟು.outputs.push_back(add_dst);
        last_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
        memcpy(last_ಗಂಟು.kernel_params, &new_kparams, ಪರಿಮಾಣ(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ADD;
        memcpy(o.kernel_params, &new_kparams, ಪರಿಮಾಣ(new_kparams));

        o.src[2 * n_ranks] = add_tensor(res_tensor);
        ಯದಿ (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            ಪ್ರತಿ (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            ಪ್ರತಿ (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } ಅನ್ಯಥಾ {
            o.dst[0] = add_tensor(add_dst);
            ಪ್ರತಿ (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (ಪೂರ್ಣಾಙ್ಕ) new_kparams.mode, (ಪೂರ್ಣಾಙ್ಕ) new_kparams.n_dsts);
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ try_fuse_rms_norm_mul(ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ಯದಿ (n_ops == 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ (ಗಂಟು.opcode != HTP_OP_MUL) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        htp_opnode & last_ಗಂಟು = ops[n_ops - 1];
        ಯದಿ (last_ಗಂಟು.opcode != HTP_OP_RMS_NORM) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಧ್ರುವ ggml_tensor * mul_src0 = ಗಂಟು.src0();
        ಧ್ರುವ ggml_tensor * mul_src1 = ಗಂಟು.src1();
        ಧ್ರುವ ggml_tensor * rms_out  = last_ಗಂಟು.dst();

        ಯದಿ (!ggml_hexagon_tensor_is_fuseable(rms_out)) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಧ್ರುವ ggml_tensor * weight;
        ಯದಿ (mul_src0 == rms_out || mul_src0->data == rms_out->data) {
            weight = mul_src1;
        } ಅನ್ಯಥಾ ಯದಿ (mul_src1 == rms_out || mul_src1->data == rms_out->data) {
            weight = mul_src0;
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ggml_tensor * src0 = last_ಗಂಟು.src0();

        ಯದಿ (src0->ne[0] != weight->ne[0] || src0->ne[0] != ಗಂಟು.dst()->ne[0]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ತಾರ್ಕಿಕ is_row_bcast = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
        ಧ್ರುವ ತಾರ್ಕಿಕ is_same_shape = (src0->ne[0] == weight->ne[0] && src0->ne[1] == weight->ne[1] &&
                                    src0->ne[2] == weight->ne[2] && src0->ne[3] == weight->ne[3]);
        ಯದಿ (!is_row_bcast && !is_same_shape) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಯದಿ (!ggml_are_same_shape(src0, ಗಂಟು.dst())) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (ggml_is_contiguous(src0) != ggml_is_contiguous(ಗಂಟು.dst())) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಸಂರಚನಾ htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, weight, ಗಂಟು.dst(), &new_kparams
        );

        ಯದಿ ((size_t) new_kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_size, sess->vtcm_size);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (!try_fuse_common(weight, ಗಂಟು.dst())) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        last_ಗಂಟು.opcode = HTP_OP_RMS_NORM_MUL;
        last_ಗಂಟು.name   = "RMS_NORM+MUL";
        last_ಗಂಟು.inputs.clear();
        last_ಗಂಟು.inputs.push_back(src0);
        last_ಗಂಟು.inputs.push_back(weight);
        last_ಗಂಟು.outputs.clear();
        last_ಗಂಟು.outputs.push_back(ಗಂಟು.dst());
        last_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
        memcpy(last_ಗಂಟು.kernel_params, &new_kparams, ಪರಿಮಾಣ(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        memcpy(o.kernel_params, &new_kparams, ಪರಿಮಾಣ(new_kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(weight);
        ಪ್ರತಿ (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(ಗಂಟು.dst());
        ಪ್ರತಿ (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ try_fuse_mul_mat_add(ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ಯದಿ (n_ops == 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ (ಗಂಟು.opcode != HTP_OP_ADD) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        htp_opnode & last_ಗಂಟು = ops[n_ops - 1];
        ಯದಿ (last_ಗಂಟು.opcode != HTP_OP_MUL_MAT) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಧ್ರುವ ggml_tensor * add_src0 = ಗಂಟು.src0();
        ಧ್ರುವ ggml_tensor * add_src1 = ಗಂಟು.src1();
        ಧ್ರುವ ggml_tensor * mm_out   = last_ಗಂಟು.dst();

        ಯದಿ (!ggml_hexagon_tensor_is_fuseable(mm_out)) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಧ್ರುವ ggml_tensor * src2;
        ಯದಿ (add_src0 == mm_out || add_src0->data == mm_out->data) {
            src2 = add_src1;
        } ಅನ್ಯಥಾ ಯದಿ (add_src1 == mm_out || add_src1->data == mm_out->data) {
            src2 = add_src0;
        } ಅನ್ಯಥಾ {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ggml_tensor * src0 = last_ಗಂಟು.src0();
        ಧ್ರುವ ggml_tensor * src1 = last_ಗಂಟು.src1();

        ಯದಿ (src2->type != GGML_TYPE_F32) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        ಯದಿ (!ggml_are_same_shape(mm_out, ಗಂಟು.dst())) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಧ್ರುವ ಸಂರಚನಾ htp_mm_kernel_params * orig_kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_mm_kernel_params *) last_ಗಂಟು.kernel_params;
        ಸಂರಚನಾ htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_add_params(sess, src0, src1, src2, ಗಂಟು.dst(), &kparams);
        ಯದಿ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (orig_kparams->collapse != kparams.collapse) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        ಧ್ರುವ ತಾರ್ಕಿಕ can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        ಯದಿ (!can_fuse) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಯದಿ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            ಯದಿ (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_threads < orig_kparams->n_act_threads) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_threads, kparams.n_act_threads);
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }

        ಯದಿ (!try_fuse_common(src2, ಗಂಟು.dst())) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        last_ಗಂಟು.opcode = HTP_OP_MUL_MAT_ADD;
        last_ಗಂಟು.name   = "MUL_MAT+ADD";
        last_ಗಂಟು.inputs.clear();
        last_ಗಂಟು.inputs.push_back(src0);
        last_ಗಂಟು.inputs.push_back(src1);
        last_ಗಂಟು.inputs.push_back(src2);
        last_ಗಂಟು.outputs.clear();
        last_ಗಂಟು.outputs.push_back(ಗಂಟು.dst());
        last_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
        memcpy(last_ಗಂಟು.kernel_params, &kparams, ಪರಿಮಾಣ(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ADD;
        memcpy(o.kernel_params, &kparams, ಪರಿಮಾಣ(kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(src1);
        o.src[2] = add_tensor(src2);
        ಪ್ರತಿ (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(ಗಂಟು.dst());
        ಪ್ರತಿ (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ try_fuse_mul_mat_nx(ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ಯದಿ (n_ops == 0 || ಗಂಟು.opcode != HTP_OP_MUL_MAT) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ (!is_mergeable_mul_mat(ಗಂಟು.ಗಂಟು)) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಧ್ರುವ ggml_tensor * w_in = ಗಂಟು.src0();
        ಧ್ರುವ ggml_tensor * x_in = ಗಂಟು.src1();
        ಧ್ರುವ ggml_tensor * d_in = ಗಂಟು.dst();

        htp_opnode & last_ಗಂಟು = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        ಯದಿ (last_ಗಂಟು.opcode == HTP_OP_MUL_MAT_NX) {
            ಧ್ರುವ uint32_t curr_n = (uint32_t) last_ಗಂಟು.outputs.size();
            ಯದಿ (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಧ್ರುವ ggml_tensor * w0 = last_ಗಂಟು.inputs[0];
            ಧ್ರುವ ggml_tensor * x  = last_ಗಂಟು.inputs[curr_n];

            ಯದಿ (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (!last_ಗಂಟು.fused.empty() && (mm_is_hmx_eligible(last_ಗಂಟು.fused[0]) != mm_is_hmx_eligible(ಗಂಟು.ಗಂಟು))) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಧ್ರುವ ಸಂರಚನಾ htp_mm_kernel_params * orig_kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_mm_kernel_params *) last_ಗಂಟು.kernel_params;
            ಧ್ರುವ ತಾರ್ಕಿಕ collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_in, x, d_in);
            ಯದಿ (orig_kparams->collapse && !collapse) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಸಂರಚನಾ htp_mm_kernel_params kparams;
            ಯದಿ (collapse) {
                ಧ್ರುವ ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } ಅನ್ಯಥಾ {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            ಯದಿ (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಯದಿ (!try_fuse_common(w_in, d_in)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            last_ಗಂಟು.inputs[curr_n] = w_in;
            last_ಗಂಟು.inputs.push_back(x);
            last_ಗಂಟು.outputs.push_back(d_in);
            last_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
            memcpy(last_ಗಂಟು.kernel_params, &kparams, ಪರಿಮಾಣ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, ಪರಿಮಾಣ(kparams));

            ಪ್ರತಿ (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = add_tensor(last_ಗಂಟು.inputs[s]);
            }
            ಪ್ರತಿ (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            ಪ್ರತಿ (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_ಗಂಟು.outputs[d]);
            }
            ಪ್ರತಿ (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        // Case 2: last_node is single MUL_MAT
        ಯದಿ (last_ಗಂಟು.opcode == HTP_OP_MUL_MAT) {
            ಯದಿ (!is_mergeable_mul_mat_pair(last_ಗಂಟು.ಗಂಟು, ಗಂಟು.ಗಂಟು)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಧ್ರುವ ggml_tensor * w0 = last_ಗಂಟು.src0();
            ಧ್ರುವ ggml_tensor * x  = last_ಗಂಟು.src1();
            ಧ್ರುವ ggml_tensor * w1 = ಗಂಟು.src0();
            ಧ್ರುವ ggml_tensor * dst_0 = last_ಗಂಟು.dst();
            ಧ್ರುವ ggml_tensor * dst_1 = ಗಂಟು.dst();

            ಧ್ರುವ ಸಂರಚನಾ htp_mm_kernel_params * orig_kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_mm_kernel_params *) last_ಗಂಟು.kernel_params;
            ಧ್ರುವ ತಾರ್ಕಿಕ collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            ಯದಿ (orig_kparams->collapse && !collapse) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಸಂರಚನಾ htp_mm_kernel_params kparams;
            ಯದಿ (collapse) {
                ಧ್ರುವ ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } ಅನ್ಯಥಾ {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            ಯದಿ (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಯದಿ (!try_fuse_common(w1, ಗಂಟು.dst())) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            last_ಗಂಟು.opcode = HTP_OP_MUL_MAT_NX;
            last_ಗಂಟು.name   = "MUL_MAT_NX";
            last_ಗಂಟು.inputs.clear();
            last_ಗಂಟು.inputs.push_back(w0);
            last_ಗಂಟು.inputs.push_back(w1);
            last_ಗಂಟು.inputs.push_back(x);
            last_ಗಂಟು.outputs.clear();
            last_ಗಂಟು.outputs.push_back(dst_0);
            last_ಗಂಟು.outputs.push_back(dst_1);
            last_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
            memcpy(last_ಗಂಟು.kernel_params, &kparams, ಪರಿಮಾಣ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            memcpy(o.kernel_params, &kparams, ಪರಿಮಾಣ(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            ಪ್ರತಿ (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            ಪ್ರತಿ (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ try_fuse_mul_mat_id_nx(ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ಯದಿ (n_ops == 0 || ಗಂಟು.opcode != HTP_OP_MUL_MAT_ID) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ (!is_mergeable_mul_mat_id(ಗಂಟು.ಗಂಟು)) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಧ್ರುವ ggml_tensor * w_in   = ಗಂಟು.src0();
        ಧ್ರುವ ggml_tensor * x_in   = ಗಂಟು.src1();
        ಧ್ರುವ ggml_tensor * ids_in = ಗಂಟು.ಗಂಟು->src[2];
        ಧ್ರುವ ggml_tensor * d_in   = ಗಂಟು.dst();

        htp_opnode & last_ಗಂಟು = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        ಯದಿ (last_ಗಂಟು.opcode == HTP_OP_MUL_MAT_ID_NX) {
            ಧ್ರುವ uint32_t curr_n = (uint32_t) last_ಗಂಟು.outputs.size();
            ಯದಿ (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಧ್ರುವ ggml_tensor * w0  = last_ಗಂಟು.inputs[0];
            ಧ್ರುವ ggml_tensor * x   = last_ಗಂಟು.inputs[curr_n];
            ಧ್ರುವ ggml_tensor * ids = last_ಗಂಟು.inputs[curr_n + 1];

            ಯದಿ (x_in != x || ids_in != ids || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (!last_ಗಂಟು.fused.empty() && (mm_is_hmx_eligible(last_ಗಂಟು.fused[0]) != mm_is_hmx_eligible(ಗಂಟು.ಗಂಟು))) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಸಂರಚನಾ htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
            ಯದಿ (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಯದಿ (!try_fuse_common(w_in, d_in)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            last_ಗಂಟು.inputs[curr_n] = w_in;
            last_ಗಂಟು.inputs[curr_n + 1] = x;
            last_ಗಂಟು.inputs.push_back(ids);
            last_ಗಂಟು.outputs.push_back(d_in);
            last_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
            memcpy(last_ಗಂಟು.kernel_params, &kparams, ಪರಿಮಾಣ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, ಪರಿಮಾಣ(kparams));

            ಪ್ರತಿ (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = add_tensor(last_ಗಂಟು.inputs[s]);
            }
            ಪ್ರತಿ (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            ಪ್ರತಿ (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_ಗಂಟು.outputs[d]);
            }
            ಪ್ರತಿ (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        // Case 2: last_node is single MUL_MAT_ID
        ಯದಿ (last_ಗಂಟು.opcode == HTP_OP_MUL_MAT_ID) {
            ಯದಿ (!is_mergeable_mul_mat_id_pair(last_ಗಂಟು.ಗಂಟು, ಗಂಟು.ಗಂಟು)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಧ್ರುವ ggml_tensor * w0  = last_ಗಂಟು.src0();
            ಧ್ರುವ ggml_tensor * x   = last_ಗಂಟು.src1();
            ಧ್ರುವ ggml_tensor * ids = last_ಗಂಟು.ಗಂಟು->src[2];
            ಧ್ರುವ ggml_tensor * w1  = ಗಂಟು.src0();

            ಸಂರಚನಾ htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, ಗಂಟು.dst(), 2, &kparams);
            ಯದಿ (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಯದಿ (!try_fuse_common(w1, ಗಂಟು.dst())) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಧ್ರುವ ggml_tensor * dst_0 = last_ಗಂಟು.dst();
            ಧ್ರುವ ggml_tensor * dst_1 = ಗಂಟು.dst();

            last_ಗಂಟು.opcode = HTP_OP_MUL_MAT_ID_NX;
            last_ಗಂಟು.name   = "MUL_MAT_ID_NX";
            last_ಗಂಟು.inputs.clear();
            last_ಗಂಟು.inputs.push_back(w0);
            last_ಗಂಟು.inputs.push_back(w1);
            last_ಗಂಟು.inputs.push_back(x);
            last_ಗಂಟು.inputs.push_back(ids);
            last_ಗಂಟು.outputs.clear();
            last_ಗಂಟು.outputs.push_back(dst_0);
            last_ಗಂಟು.outputs.push_back(dst_1);
            last_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);
            memcpy(last_ಗಂಟು.kernel_params, &kparams, ಪರಿಮಾಣ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            memcpy(o.kernel_params, &kparams, ಪರಿಮಾಣ(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            o.src[3] = add_tensor(ids);
            ಪ್ರತಿ (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            ಪ್ರತಿ (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ try_fuse_gdn_cpy(ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ಯದಿ (n_ops == 0 || ಗಂಟು.opcode != HTP_OP_CPY) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        htp_opnode & last_ಗಂಟು = ops[n_ops - 1];
        ಯದಿ (last_ಗಂಟು.opcode != HTP_OP_GATED_DELTA_NET) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ (last_ಗಂಟು.outputs.size() != 1) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಧ್ರುವ ggml_tensor * gdn_out  = last_ಗಂಟು.dst();
        ಧ್ರುವ ggml_tensor * cpy_ಗಂಟು = ಗಂಟು.ಗಂಟು;
        ಧ್ರುವ ggml_tensor * cpy_src  = ಗಂಟು.src0();
        ಧ್ರುವ ggml_tensor * cpy_dst  = ಗಂಟು.dst();

        ಯದಿ (!cpy_src || !cpy_dst || !cpy_dst->data) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_ಗಂಟು->flags & GGML_TENSOR_FLAG_OUTPUT)) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಧ್ರುವ ggml_tensor * v = last_ಗಂಟು.ಗಂಟು->src[2];
        ಯದಿ (!v) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಧ್ರುವ int64_t S_v      = v->ne[0];
        ಧ್ರುವ int64_t H        = v->ne[1];
        ಧ್ರುವ int64_t n_tokens = v->ne[2];
        ಧ್ರುವ int64_t n_seqs   = v->ne[3];
        ಧ್ರುವ int64_t K        = ggml_get_op_params_i32(last_ಗಂಟು.ಗಂಟು, 0);
        ಧ್ರುವ size_t  tail_off = (size_t) S_v * H * n_tokens * n_seqs * ಪರಿಮಾಣ(ಭಿನ್ನ);

        ಧ್ರುವ int64_t D         = S_v * S_v * H;
        ಧ್ರುವ int64_t n_written = std::min<int64_t>(n_tokens, K);

        ಯದಿ (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_out && cpy_src->view_src->data != gdn_out->data) ||
            cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * ಪರಿಮಾಣ(ಭಿನ್ನ)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (n_written > 1) {
            ಯದಿ (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * ಪರಿಮಾಣ(ಭಿನ್ನ)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }

        ಯದಿ (!try_fuse_common({cpy_dst})) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        last_ಗಂಟು.name += "+CPY";
        last_ಗಂಟು.outputs.push_back(cpy_dst);
        last_ಗಂಟು.fused.push_back(ಗಂಟು.ಗಂಟು);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = add_tensor(cpy_dst);
        ಪ್ರತಿ (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ try_fuse(ಧ್ರುವ htp_opnode & ಗಂಟು) {
        ಯದಿ (!opt_opfusion) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ADD) && try_fuse_allreduce_add(ಗಂಟು)) ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಯದಿ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(ಗಂಟು))  ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಯದಿ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ADD)   && try_fuse_mul_mat_add(ಗಂಟು))   ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಯದಿ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(ಗಂಟು))    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಯದಿ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(ಗಂಟು)) ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಯದಿ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(ಗಂಟು))       ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
};

ಸಂರಚನಾ ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

ಸಂರಚನಾ ggml_hexagon_opqueue {
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
        ಯದಿ (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * ಪರಿಮಾಣ(htp_trace_desc);
        }

        shm_blk_size = ಪರಿಮಾಣ(htp_buf_desc)  * n_bufs    +
                       ಪರಿಮಾಣ(htp_tensor)    * n_tensors +
                       ಪರಿಮಾಣ(htp_op_desc)   * n_ops     +
                       ಪರಿಮಾಣ(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = ನೂತನ ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, ಸತ್ಯ /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        ಯದಿ (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size(), shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    size_t shm_size() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ shm_buf ? shm_buf->size() : 0; }

    // push new batch
    ತಾರ್ಕಿಕ ತಳ್ಳು(htp_opbatch_req& req, dspqueue_buffer& dbuf, ಧ್ರುವ ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(ಪರಿಮಾಣ(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(ಪರಿಮಾಣ(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(ಪರಿಮಾಣ(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(ಪರಿಮಾಣ(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(ಪರಿಮಾಣ(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(ಪರಿಮಾಣ(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        ಯದಿ (seq - shm_buf->sess->batch_rsp_seq > depth) { ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; }

        ಧ್ರುವ uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[slot]   = op_batch->ops;
        start_usec[slot] = ggml_time_us();

        ಧ್ರುವ size_t b_size = ಪರಿಮಾಣ(htp_buf_desc)  * req.n_bufs;
        ಧ್ರುವ size_t t_size = ಪರಿಮಾಣ(htp_tensor)    * req.n_tensors;
        ಧ್ರುವ size_t o_size = ಪರಿಮಾಣ(htp_op_desc)   * req.n_ops;
        ಧ್ರುವ size_t p_size = ಪರಿಮಾಣ(htp_prof_desc) * req.n_ops;

        size_t tr_size = 0;
        ಯದಿ (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * ಪರಿಮಾಣ(htp_trace_desc);
        } ಅನ್ಯಥಾ {
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

        memcpy(b_ptr, (ಶೂನ್ಯ *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (ಶೂನ್ಯ *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (ಶೂನ್ಯ *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        ಯದಿ (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (ಶೂನ್ಯ *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            ಪ್ರತಿ (unsigned ಪೂರ್ಣಾಙ್ಕ i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (unsigned long long) t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಶೂನ್ಯ ತೆಗೆ(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        ಧ್ರುವ uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        ಧ್ರುವ size_t b_size = ಪರಿಮಾಣ(htp_buf_desc)  * rsp.n_bufs;
        ಧ್ರುವ size_t t_size = ಪರಿಮಾಣ(htp_tensor)    * rsp.n_tensors;
        ಧ್ರುವ size_t o_size = ಪರಿಮಾಣ(htp_op_desc)   * rsp.n_ops;
        ಧ್ರುವ size_t p_size = ಪರಿಮಾಣ(htp_prof_desc) * rsp.n_ops;

        size_t tr_size = 0;
        uint32_t n_traces = 0;
        ಯದಿ (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * ಪರಿಮಾಣ(htp_trace_desc);
        }

        ಧ್ರುವ size_t m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        ಯದಿ (rsp.n_ops > 0) {
            auto & ops = op_cache[slot];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            ಧ್ರುವ htp_prof_desc * pd = (ಧ್ರುವ htp_prof_desc *) p_ptr;
            ಧ್ರುವ htp_trace_desc * trace_events = nullptr;
            ಯದಿ (opt_profile == 3) {
                trace_events = (ಧ್ರುವ htp_trace_desc *) (p_ptr + p_size);
            }

            ಯದಿ (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            ಪ್ರತಿ (uint32_t i = 0; i < rsp.n_ops; i++) {
                ಯದಿ (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            ಯದಿ (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

ಶೂನ್ಯ ggml_hexagon_session::flush_peers() {
    auto vpeers = std::move(virt_peers);
    virt_peers.clear();
    ಪ್ರತಿ (auto * peer : vpeers) {
        peer->flush_sync();
    }

    auto ppeers = std::move(phys_peers);
    phys_peers.clear();
    ಪ್ರತಿ (auto * peer : ppeers) {
        peer->flush_async();
    }

    ಪ್ರತಿ (auto & sub : ಏತತ್->mdev.sessions) {
        sub->flush_peers();
    }
}

ಶೂನ್ಯ ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

ಶೂನ್ಯ ggml_hexagon_session::flush_pending(ತಾರ್ಕಿಕ all) {
    ಪ್ರತಿ (auto & sub : ಏತತ್->mdev.sessions) {
        sub->flush_pending(all);
        ಯದಿ (sub->last_error > HTP_STATUS_OK) {
            ಏತತ್->last_error = sub->last_error.load();
        }
    }

    ಯಾವತ್ (ಏತತ್->batch_rsp_seq < ಏತತ್->batch_req_seq) {
        ಸಂರಚನಾ htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        ಸಂರಚನಾ dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        ಧ್ರುವ uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        ಪೂರ್ಣಾಙ್ಕ err = dspqueue_read(ಏತತ್->queue, &flags, 1, &n_dbufs, &dbuf, ಪರಿಮಾಣ(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        ಯದಿ (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            ಅನುವರ್ತ;
        }

        ಯದಿ (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        ಯದಿ (rsp_size != ಪರಿಮಾಣ(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", ಏತತ್->c_name(), rsp_size, n_dbufs);
        }

        ಯದಿ (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", ಏತತ್->c_name(), status_to_str(rsp.status));
            ಏತತ್->last_error = rsp.status;
            ಪ್ರತಿ (auto & sub : ಏತತ್->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->ತೆಗೆ(rsp, dbuf);

        GGML_ASSERT(rsp.seq == ಏತತ್->batch_rsp_seq + 1);
        ಏತತ್->batch_rsp_seq = rsp.seq;

        ಯದಿ (!all) ಭಙ್ಗ;
    }
}

ಶೂನ್ಯ ggml_hexagon_session::flush_sync(ತಾರ್ಕಿಕ all) {
    flush_async();
    flush_pending(all);
}

ಶೂನ್ಯ ggml_hexagon_session::start_batch() {
    ಯದಿ (ಏತತ್->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

ಶೂನ್ಯ ggml_hexagon_session::flush_batch(size_t min_ops) {
    ಯದಿ (op_batch->n_ops < min_ops) { ಪ್ರತ್ಯಾಗಮ; }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    ಧ್ರುವ uint64_t seq = ++ಏತತ್->batch_req_seq;

    op_batch->update_mdev_group(ಏತತ್->mdev.idx);

    ಯದಿ (!op_queue->ತಳ್ಳು(req, dbuf, op_batch, seq)) {
        flush_pending(ಅಸತ್ಯ);
        op_queue->ತಳ್ಳು(req, dbuf, op_batch, seq);
    }

    ಪ್ರತಿ (auto & sub : ಏತತ್->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_buffer sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        ಯದಿ (!sub->op_queue->ತಳ್ಳು(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(ಅಸತ್ಯ);
            sub->op_queue->ತಳ್ಳು(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        ಪೂರ್ಣಾಙ್ಕ err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, ಪರಿಮಾಣ(sub_req), (ಧ್ರುವ uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        ಯದಿ (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (unsigned) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", ಏತತ್->c_name(), dbuf.ptr, dbuf.size);

    ಪೂರ್ಣಾಙ್ಕ err = dspqueue_write(ಏತತ್->queue, 0, 1, &dbuf, ಪರಿಮಾಣ(req), (ಧ್ರುವ uint8_t*) &req, DSPQUEUE_TIMEOUT);
    ಯದಿ (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", ಏತತ್->c_name(), (unsigned) err);
    }

    op_batch->reset();
}

ಶೂನ್ಯ ggml_hexagon_session::enqueue_op(ಧ್ರುವ htp_opnode & ಗಂಟು) {
    auto clone_tensor_buffer = [ಏತತ್](ಧ್ರುವ ggml_tensor * t) {
        auto sbuf = ಏತತ್->mmap_tensor(t);
        ಯದಿ (!sbuf) ಪ್ರತ್ಯಾಗಮ;
        ಯದಿ (sbuf->sess != ಏತತ್) {
            ಏತತ್->clone_buffer(sbuf);
        }
        ಪ್ರತಿ (auto & sub : ಏತತ್->mdev.sessions) {
            sub->clone_buffer(sbuf);
        }
    };

    ಪ್ರತಿ (auto t : ಗಂಟು.get_inputs()) {
        clone_tensor_buffer(t);
    }
    ಪ್ರತಿ (auto t : ಗಂಟು.get_outputs()) {
        clone_tensor_buffer(t);
    }

    ಯದಿ (opt_opfusion && op_batch->try_fuse(ಗಂಟು)) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (!op_batch->fit_op(ಗಂಟು)) {
        flush_async();
    }

    ಯದಿ (op_batch->empty()) {
        start_batch();
    }

    ಯದಿ (!op_batch->fit_op(ಗಂಟು)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->add_op(ಗಂಟು);
}

ಶೂನ್ಯ ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_ಗಂಟು(HTP_OP_MDEV_GROUP);

    uint8_t * fence_slot = ಏತತ್->mdev_fence_slot;

    ಸ್ಥಿರ ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.buffer       = &ಏತತ್->fence_buf->backend_buffer;
    dummy_t.extra        = &fence_extra;
    dummy_t.data         = (ಶೂನ್ಯ *) fence_slot;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.ne[1]        = (int64_t) ಏತತ್->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) ಏತತ್->mdev.idx;

    ggml_tensor * ಗಂಟು = group_ಗಂಟು.add_dummy(dummy_t);
    ಗಂಟು->src[0] = ಗಂಟು;
    group_ಗಂಟು.init(ಗಂಟು);
    group_ಗಂಟು.outputs.clear();
    group_ಗಂಟು.name = "MDEV_GROUP";

    ಪ್ರತಿ (auto & sub : ಏತತ್->mdev.sessions) {
        sub->clone_buffer(ಏತತ್->fence_buf);
    }

    op_batch->add_op(group_ಗಂಟು);
}

ಶೂನ್ಯ ggml_hexagon_session::enqueue_cpy(ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst, ಧ್ರುವ ggml_tensor * sync_tensor, uint32_t fence_seq) {
    ಧ್ರುವ ತಾರ್ಕಿಕ with_fence = sync_tensor != nullptr;
    htp_opnode cpy_ಗಂಟು(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* ಗಂಟು = cpy_ಗಂಟು.add_dummy(*dst);
    ಗಂಟು->op     = GGML_OP_CPY;
    ಗಂಟು->src[0] = const_cast<ggml_tensor *>(src);
    ಗಂಟು->src[1] = with_fence ? cpy_ಗಂಟು.add_dummy(*sync_tensor) : nullptr;
    ಯದಿ (with_fence) {
        ಗಂಟು->op_params[0] = (int32_t) fence_seq;
    }

    cpy_ಗಂಟು.init(ಗಂಟು);
    ಯದಿ (with_fence) {
        cpy_ಗಂಟು.name = "CPY+FENCE";
    }
    ಧ್ರುವ ತಾರ್ಕಿಕ ok = ggml_hexagon_precompute_cpy_params(ಏತತ್, ಗಂಟು, (ಸಂರಚನಾ htp_copy_kernel_params *) cpy_ಗಂಟು.kernel_params);
    ಧ್ರುವ auto * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_copy_kernel_params *) cpy_ಗಂಟು.kernel_params;
    ಯದಿ (ok && !with_fence && kparams->total_elems == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಏತತ್->enqueue_op(cpy_ಗಂಟು);
}

ಶೂನ್ಯ ggml_hexagon_session::enqueue_fence(ಧ್ರುವ ggml_tensor * sync_tensor, uint32_t fence_seq, ತಾರ್ಕಿಕ wait) {
    htp_opnode sync_ಗಂಟು(HTP_OP_FENCE);

    ggml_tensor* ಗಂಟು = sync_ಗಂಟು.add_dummy(*sync_tensor);
    ಗಂಟು->op           = GGML_OP_NONE;
    ಗಂಟು->src[0]       = ಗಂಟು;
    ಗಂಟು->op_params[0] = (int32_t) fence_seq;
    ಗಂಟು->op_params[1] = wait ? 0 : 1;

    sync_ಗಂಟು.init(ಗಂಟು);
    sync_ಗಂಟು.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    ಏತತ್->enqueue_op(sync_ಗಂಟು);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_allreduce_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    ತಾರ್ಕಿಕ has_add,
    ತಾರ್ಕಿಕ is_row_bcast,
    ತಾರ್ಕಿಕ is_shard_ok,
    ಸಂರಚನಾ htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    ಧ್ರುವ uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    ಧ್ರುವ uint32_t elem_size = (dst->type == GGML_TYPE_F16) ? ಪರಿಮಾಣ(ggml_fp16_t) : ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ ತಾರ್ಕಿಕ is_contiguous = ggml_is_contiguous(dst);

    ಧ್ರುವ uint32_t ne0 = (uint32_t) dst->ne[0];
    ಧ್ರುವ uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    ಧ್ರುವ ತಾರ್ಕಿಕ use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    ಯದಿ (has_add) {
        // sharded reduce-scatter for contiguous in-place add
        ಯದಿ (opt_ar_scatter && is_shard_ok && use_1d && !is_row_bcast) {
            ಧ್ರುವ uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            ಧ್ರುವ uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            ಧ್ರುವ uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_start = (int32_t) rank_elem_start;
            kparams->rank_nelem      = (int32_t) (rank_elem_end - rank_elem_start);
        } ಅನ್ಯಥಾ ಯದಿ (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } ಅನ್ಯಥಾ {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } ಅನ್ಯಥಾ {
        kparams->n_dsts = (int32_t) n_ranks;
        ಯದಿ (use_1d) {
            ಧ್ರುವ uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            ಧ್ರುವ uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            ಧ್ರುವ uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            ಧ್ರುವ uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } ಅನ್ಯಥಾ {
            ಧ್ರುವ uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            ಧ್ರುವ uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            ಧ್ರುವ uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            ಧ್ರುವ uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    ಯದಿ (use_1d) {
        ಧ್ರುವ uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        ಧ್ರುವ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        ಧ್ರುವ size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_ಎಣಿಕೆ(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        ಯದಿ (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        ಯಾವತ್ ((size_t) kparams->vtcm_size > sess->vtcm_size && block_elems > 128) {
            ಧ್ರುವ size_t max_bytes_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((size_t) (max_bytes_per_buf / elem_size), 128);
            ಯದಿ (block_elems < 128) ಭಙ್ಗ;
            kparams->block_elems          = block_elems;
            kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
        }

        ಯದಿ (sess->vtcm_size < (size_t) kparams->vtcm_size || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    } ಅನ್ಯಥಾ {
        ಧ್ರುವ uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        ಧ್ರುವ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        ಧ್ರುವ size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_ಎಣಿಕೆ(n_ranks, n_threads, has_add, is_row_bcast);

        ಧ್ರುವ uint32_t row_bytes = ne0 * elem_size;
        ಧ್ರುವ uint32_t row_size_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_size_aligned = row_size_aligned;

        ಧ್ರುವ uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        ಯಾವತ್ ((size_t) kparams->vtcm_size > sess->vtcm_size && block_rows > 1) {
            ಧ್ರುವ size_t max_rows_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2 * row_size_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
            ಯದಿ (max_rows_per_buf == 0) ಭಙ್ಗ;
        }

        ಯದಿ (sess->vtcm_size < (size_t) kparams->vtcm_size || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
}

ಶೂನ್ಯ ggml_hexagon_session::enqueue_allreduce(
    ಧ್ರುವ ggml_tensor * dst,
    ಧ್ರುವ std::vector<ಧ್ರುವ ggml_tensor *> & src_tensors,
    ಧ್ರುವ std::vector<ಧ್ರುವ ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_ಗಂಟು(HTP_OP_ALLREDUCE);

    ggml_tensor* ಗಂಟು = ar_ಗಂಟು.add_dummy(*dst);
    ಗಂಟು->op           = GGML_OP_NONE;
    ಗಂಟು->op_params[0] = (int32_t) fence_seq_entry;
    ಗಂಟು->op_params[1] = (int32_t) fence_seq_exit;

    ar_ಗಂಟು.init(ಗಂಟು);

    ar_ಗಂಟು.inputs.clear();
    ಪ್ರತಿ (size_t i = 0; i < src_tensors.size(); i++) {
        ar_ಗಂಟು.inputs.push_back(src_tensors[i]);
    }
    ಪ್ರತಿ (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_ಗಂಟು.inputs.push_back(ar_ಗಂಟು.add_dummy(*sync_tensors[i]));
    }

    ar_ಗಂಟು.outputs.clear();
    ಪ್ರತಿ (size_t i = 0; i < src_tensors.size(); i++) {
        ar_ಗಂಟು.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        ಏತತ್, dst, rank, n_ranks, ಅಸತ್ಯ, ಅಸತ್ಯ, /*is_shard_ok=*/ ಅಸತ್ಯ,
        (ಸಂರಚನಾ htp_allreduce_kernel_params *) ar_ಗಂಟು.kernel_params);

    ar_ಗಂಟು.name = "ALLREDUCE";
    ಏತತ್->enqueue_op(ar_ಗಂಟು);
}

ggml_hexagon_shared_buffer * ggml_hexagon_session::mmap_tensor(ಧ್ರುವ ggml_tensor * t) {
    ಯದಿ (!t) ಪ್ರತ್ಯಾಗಮ nullptr;

    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
    ಯದಿ (!sbuf->mapped) {
        ಧ್ರುವ ತಾರ್ಕಿಕ is_weight = ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        ಧ್ರುವ ತಾರ್ಕಿಕ extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    ಪ್ರತ್ಯಾಗಮ sbuf;
}

ತಾರ್ಕಿಕ ggml_hexagon_session::clone_buffer(ಧ್ರುವ ggml_hexagon_shared_buffer * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    ಯದಿ (sbuf->sess == ಏತತ್) ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    auto mem = sbuf->mem;
    ಪೂರ್ಣಾಙ್ಕ   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    ಯದಿ (ಏತತ್->cloned_buffers.find(fd) != ಏತತ್->cloned_buffers.end()) ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", ಏತತ್->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    auto clone = std::make_unique<ggml_hexagon_shared_buffer>(ಏತತ್, *sbuf);
    ಯತ್ನ {
        clone->mmap(sbuf->extended);
    } ಗೃಹಾಣ (ಧ್ರುವ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", ಏತತ್->c_name(), exc.what());
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಏತತ್->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(ಏತತ್);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಶೂನ್ಯ ggml_hexagon_session::release_buffer(ಧ್ರುವ ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    ಪೂರ್ಣಾಙ್ಕ   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    auto it = ಏತತ್->cloned_buffers.find(fd);
    ಯದಿ (it != ಏತತ್->cloned_buffers.end()) {
        auto clone = std::move(it->second);
        ಏತತ್->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(ಏತತ್);
}

ಶೂನ್ಯ ggml_hexagon_session::unclone_buffer(ಧ್ರುವ ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    ಪ್ರತಿ (auto * sess : sessions) {
        sess->release_buffer(sbuf);
    }
}

ಸ್ಥಿರ size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    ಧ್ರುವ size_t MiB = 1024 * 1024;
    ಧ್ರುವ size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t step = 256u * MiB;

    ಯತ್ನ {
        sbufs.push_back(ನೂತನ ggml_hexagon_shared_buffer(sess, GiB, ಸತ್ಯ)); vmem += GiB;
        sbufs.push_back(ನೂತನ ggml_hexagon_shared_buffer(sess, GiB, ಸತ್ಯ)); vmem += GiB;
        sbufs.push_back(ನೂತನ ggml_hexagon_shared_buffer(sess, GiB, ಸತ್ಯ)); vmem += GiB;

        ಯಾವತ್ (1) {
            sbufs.push_back(ನೂತನ ggml_hexagon_shared_buffer(sess, step, ಸತ್ಯ));
            vmem += step;
        }
    } ಗೃಹಾಣ (...) { }

    ಪ್ರತಿ (auto b : sbufs) { delete b; }

    ಪ್ರತ್ಯಾಗಮ vmem - step; // backoff to account for overhead from internal mappings
}

ಶೂನ್ಯ ggml_hexagon_session::allocate(ಧ್ರುವ ggml_hexagon_device_config & config) noexcept(ಅಸತ್ಯ) {
    ಪೂರ್ಣಾಙ್ಕ phys_idx = config.physical_idx;
    ಪೂರ್ಣಾಙ್ಕ virt_idx = config.virtual_idx;

    ಏತತ್->valid_session = ಅಸತ್ಯ;
    ಏತತ್->valid_handle  = ಅಸತ್ಯ;
    ಏತತ್->valid_queue   = ಅಸತ್ಯ;
    ಏತತ್->valid_iface   = ಅಸತ್ಯ;

    ಏತತ್->name          = config.name;
    ಏತತ್->phys_idx      = phys_idx;
    ಏತತ್->virt_idx      = virt_idx;
    ಏತತ್->domain_id     = config.domain_id;
    ಏತತ್->session_id    = 0;
    ಏತತ್->batch_req_seq = 0;
    ಏತತ್->batch_rsp_seq = 0;
    ಏತತ್->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", ಏತತ್->name.c_str(), ಏತತ್->domain_id, phys_idx, virt_idx);

    ಯದಿ (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        ಕ್ಷಿಪ std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    ಧ್ರುವ std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    ಯದಿ (virt_idx > 0) {
        ಸಂರಚನಾ remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<ವರ್ಣ *>(dom_name.c_str());
        n.session_name     = const_cast<ವರ್ಣ *>(ಏತತ್->name.c_str());
        n.session_name_len = ಏತತ್->name.size();
        n.session_id       = virt_idx;

        ಪೂರ್ಣಾಙ್ಕ err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (ಶೂನ್ಯ *) &n, ಪರಿಮಾಣ(n));
        ಯದಿ (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           ಏತತ್->c_name(), phys_idx, virt_idx, err);
            ಕ್ಷಿಪ std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        ಏತತ್->session_id    = n.session_id;
        ಏತತ್->domain_id     = n.effective_domain_id;
        ಏತತ್->valid_session = ಸತ್ಯ;
    } ಅನ್ಯಥಾ {
        ಸಂರಚನಾ remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<ವರ್ಣ *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        ಪೂರ್ಣಾಙ್ಕ err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (ಶೂನ್ಯ *) &eff, ಪರಿಮಾಣ(eff));
        ಯದಿ (err == AEE_SUCCESS) {
            ಏತತ್->domain_id = eff.effective_domain_id;
        } ಅನ್ಯಥಾ {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           ಏತತ್->name.c_str(), err, ಏತತ್->domain_id);
        }
    }

    // Enable unsigned modules
    {
        ಸಂರಚನಾ remote_rpc_control_unsigned_module u;
        u.domain = ಏತತ್->domain_id;
        u.enable = 1;
        ಪೂರ್ಣಾಙ್ಕ err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (ಶೂನ್ಯ *) &u, ಪರಿಮಾಣ(u));
        ಯದಿ (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", ಏತತ್->c_name(), err);
            ಕ್ಷಿಪ std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    ವರ್ಣ session_uri[256];
    {
        ವರ್ಣ htp_uri[256];
        snprintf(htp_uri, ಪರಿಮಾಣ(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        ಸಂರಚನಾ remote_rpc_get_uri u = {};
        u.session_id      = ಏತತ್->session_id;
        u.domain_name     = const_cast<ವರ್ಣ *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<ವರ್ಣ *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = ಪರಿಮಾಣ(session_uri);

        ಪೂರ್ಣಾಙ್ಕ err = remote_session_control(FASTRPC_GET_URI, (ಶೂನ್ಯ *) &u, ಪರಿಮಾಣ(u));
        ಯದಿ (err != AEE_SUCCESS) {
            snprintf(session_uri, ಪರಿಮಾಣ(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), ಏತತ್->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          ಏತತ್->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    ಪೂರ್ಣಾಙ್ಕ err = htp_iface_open(session_uri, &ಏತತ್->handle);
    ಯದಿ (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", ಏತತ್->c_name(), session_uri, err);
        ಕ್ಷಿಪ std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    ಏತತ್->valid_handle = ಸತ್ಯ;

    // Query HW info and resolve session options
    {
        unsigned ಪೂರ್ಣಾಙ್ಕ hw_n_threads = 0;
        unsigned ಪೂರ್ಣಾಙ್ಕ hw_n_hvx     = 0;
        unsigned ಪೂರ್ಣಾಙ್ಕ hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        ಪೂರ್ಣಾಙ್ಕ hw_err = htp_iface_hwinfo(ಏತತ್->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        ಯದಿ (hw_err == 0) {
            ಧ್ರುವ uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            ಏತತ್->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_threads) : max_n_threads;
            ಏತತ್->n_hvx     = ಏತತ್->n_threads;
            ಏತತ್->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            ಏತತ್->vtcm_size = (uint64_t)hw_vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          ಏತತ್->c_name(), ಏತತ್->n_threads, ಏತತ್->n_hvx, ಏತತ್->n_hmx,
                          (unsigned long long)(ಏತತ್->vtcm_size / (1024 * 1024)));
        } ಅನ್ಯಥಾ {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", ಏತತ್->c_name(), hw_err);
            ಧ್ರುವ uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            ಏತತ್->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_threads;
            ಏತತ್->n_hvx     = ಏತತ್->n_threads;
            ಏತತ್->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            ಏತತ್->vtcm_size = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        ಸಂರಚನಾ remote_rpc_control_latency l;
        l.enable = 1;

        ಪೂರ್ಣಾಙ್ಕ err = remote_handle64_control(ಏತತ್->handle, DSPRPC_CONTROL_LATENCY, (ಶೂನ್ಯ *) &l, ಪರಿಮಾಣ(l));
        ಯದಿ (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", ಏತತ್->c_name(),
                  ಏತತ್->session_id, ಏತತ್->domain_id, session_uri, (unsigned long) ಏತತ್->handle);

    ಧ್ರುವ size_t req_q_size = (ಪರಿಮಾಣ(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    ಧ್ರುವ size_t rsp_q_size = (ಪರಿಮಾಣ(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(ಏತತ್->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (ಶೂನ್ಯ *) ಏತತ್,  // Callback context
                          &queue);
    ಯದಿ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", ಏತತ್->name.c_str(), (unsigned) err);
        ಕ್ಷಿಪ std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    ಏತತ್->valid_queue = ಸತ್ಯ;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &ಏತತ್->queue_id);
    ಯದಿ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        ಕ್ಷಿಪ std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    ಯದಿ (opt_etm) {
        err = htp_iface_etm(ಏತತ್->handle, 1);
        ಯದಿ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    ಏತತ್->op_queue = ನೂತನ ggml_hexagon_opqueue(ಏತತ್, opt_opbatch, opt_opqueue);

    ಏತತ್->fence_buf = ನೂತನ ggml_hexagon_fence_buffer(ಏತತ್, &dev_ctx->fence_buffer_type, 64 * 1024);
    ಯದಿ (ಏತತ್->mdev.count > 1) {
        ಏತತ್->mdev_fence_slot = ಏತತ್->alloc_fence(ಏತತ್->mdev.count);
    }

    ಯದಿ (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(ಏತತ್);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", ಏತತ್->c_name(), opt_vmem);
    }
    ಧ್ರುವ size_t shm_size = ಏತತ್->op_queue->shm_size();
    ಏತತ್->max_vmem = (opt_vmem > shm_size) ? (opt_vmem - shm_size) : opt_vmem;

    ಏತತ್->op_batch = ನೂತನ ggml_hexagon_opbatch(ಏತತ್, opt_opbatch, ಏತತ್->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(ಏತತ್->handle, ಏತತ್->session_id, ಏತತ್->queue_id, ಏತತ್->n_threads, opt_nhmx, ಏತತ್->max_vmem, (uint32_t) opt_power_mode);
    ಯದಿ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", ಏತತ್->c_name(), (unsigned) err);
        ಕ್ಷಿಪ std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    ಏತತ್->valid_iface = ಸತ್ಯ;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", ಏತತ್->c_name(), htp_power_mode_to_str(opt_power_mode));

    ಯದಿ (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(ಏತತ್->handle, opt_profile, &pmu_conf);
        ಯದಿ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }
}

ಶೂನ್ಯ ggml_hexagon_session::release() noexcept(ಸತ್ಯ) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", ಏತತ್->name.c_str());

    ಏತತ್->mdev.sessions.clear();

    ಪೂರ್ಣಾಙ್ಕ err;

    ಯದಿ (ಏತತ್->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(ಏತತ್->handle);
        ಯದಿ (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    delete ಏತತ್->op_batch;
    delete ಏತತ್->op_queue;
    ಪ್ರತಿ (auto & it : ಏತತ್->cpy_fence_slots) {
        free_fence((ಶೂನ್ಯ *) it.second, 1);
    }
    ಏತತ್->cpy_fence_slots.clear();

    ಯದಿ (ಏತತ್->fence_buf) {
        unclone_buffer(ಏತತ್->fence_buf);
        delete ಏತತ್->fence_buf;
        ಏತತ್->fence_buf = nullptr;
    }
    ಯಾವತ್ (!ಏತತ್->cloned_buffers.empty()) {
        release_buffer(ಏತತ್->cloned_buffers.begin()->second.get());
    }

    ಯದಿ (opt_etm) {
        err = htp_iface_etm(ಏತತ್->handle, 0);
        ಯದಿ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    ಯದಿ (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(ಏತತ್->handle, 0, &pmu_conf);
        ಯದಿ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    ಯದಿ (ಏತತ್->valid_queue) {
        err = dspqueue_close(queue);
        ಯದಿ (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (unsigned) err);
        }
    }

    ಯದಿ (ಏತತ್->valid_handle) {
        htp_iface_close(ಏತತ್->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(ಧ್ರುವ ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_ಎಣಿಕೆ) noexcept(ಅಸತ್ಯ) {
    ಏತತ್->dev        = dev;
    ಏತತ್->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಏತತ್->mdev.idx   = mdev_idx;
    ಏತತ್->mdev.count = mdev_ಎಣಿಕೆ > 0 ? mdev_ಎಣಿಕೆ : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)ಏತತ್) & 0xFFFF;

    ಯತ್ನ {
        allocate(config);
        ಯದಿ (ಏತತ್->mdev.idx == 0 && !config.mdev_group.empty()) {
            ಪ್ರತಿ (size_t i = 0; i < config.mdev_group.size(); i++) {
                ಏತತ್->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], ಏತತ್->dev, (uint32_t) (i + 1), ಏತತ್->mdev.count));
            }
        }
    } ಗೃಹಾಣ (ಧ್ರುವ std::exception & exc) {
        release();
        ಕ್ಷಿಪ;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(ಸತ್ಯ) {
    release();
}

// ** backend interface

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_flash_attn_is_hmx_eligible(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * q,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * k,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * v,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * sinks
) {
    ಯದಿ (sess->n_hmx == 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (opt_fa_select < 2) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t DK = q->ne[0];
    ಧ್ರುವ uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    ಯದಿ (DK % 8 != 0 || DV % 8 != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    GGML_UNUSED(sinks);

    // Explicit force mode
    ಯದಿ (opt_fa_select > 2) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t M = q->ne[1];

    // Prefill or batched decode
    ಯದಿ (M > 1) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t n_head = q->ne[2];
    ಧ್ರುವ uint32_t n_kv_heads = k->ne[2];
    ಧ್ರುವ uint32_t G = n_kv_heads > 0 ? n_head / n_kv_heads : 1;
    ಧ್ರುವ uint32_t S = k->ne[1];
    ಧ್ರುವ uint32_t total_threads = sess->n_threads * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    ಯದಿ (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < total_threads))) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    // Tile alignment for 32-row HMX tiles
    ಧ್ರುವ ತಾರ್ಕಿಕ is_tile_aligned = (G > 0 && (32 % G == 0));
    ಯದಿ (!is_tile_aligned) {
        ಯದಿ (sess->n_threads >= 6) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    ಯದಿ (DK == 256) {
        s_cross = 4096;
    } ಅನ್ಯಥಾ ಯದಿ (DK <= 64) {
        s_cross = (sess->n_threads >= 8) ? 2048 : ((sess->n_threads >= 6) ? 768 : 512);
    } ಅನ್ಯಥಾ {
        s_cross = (sess->n_threads >= 8) ? 1024 : 512;
    }

    ಪ್ರತ್ಯಾಗಮ S >= s_cross;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_flash_attn_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_fa_kernel_params * kparams
) {
    ಯದಿ (opt_fa_select < 1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * q    = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * k    = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * v    = op->src[2];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * mask = op->src[3];

    ಧ್ರುವ uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ಧ್ರುವ uint32_t neq1 = q->ne[1];  // n_tokens
    ಧ್ರುವ uint32_t neq2 = q->ne[2];  // n_heads

    ಧ್ರುವ uint32_t nek1 = k->ne[1];  // kv_len

    ಧ್ರುವ uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ಧ್ರುವ uint32_t DK = neq0;
    ಧ್ರುವ uint32_t DV = nev0;

    ಧ್ರುವ uint32_t n_kv_heads = k->ne[2];
    ಧ್ರುವ uint32_t G          = neq2 / n_kv_heads;

    ಭಿನ್ನ scale         = 1.0f;
    ಭಿನ್ನ max_bias      = 0.0f;
    ಭಿನ್ನ logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&max_bias,      &op->op_params[1], ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&logit_softcap, &op->op_params[2], ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಯದಿ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->head_split = (opt_fa_head_split != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    ಧ್ರುವ uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * sinks = op->src[4];
    ಯದಿ (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        ಧ್ರುವ uint32_t DK_pad = hex_round_up(DK, 64);
        ಧ್ರುವ uint32_t DV_pad = hex_round_up(DV, 64);
        size_t Br = 0, Bc = 0;
        ಪೂರ್ಣಾಙ್ಕ ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
        ಯದಿ (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            ಯದಿ (logit_softcap == 0.0f) {
                kparams->scale = scale * (ಭಿನ್ನ) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } ಅನ್ಯಥಾ {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (ಭಿನ್ನ) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);

            ಧ್ರುವ size_t row_vec_bytes = hex_align_up(Bc * ಪರಿಮಾಣ(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            ಧ್ರುವ size_t m_ಸಾಲು_bytes = hex_align_up(Bc * ಪರಿಮಾಣ(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_ಸಾಲು_bytes / ಪರಿಮಾಣ(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_ಮೌಲ್ಯಗಳು(G);
            ಯದಿ (mask) {
                kparams->src3_div2 = init_fastdiv_ಮೌಲ್ಯಗಳು(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_ಮೌಲ್ಯಗಳು(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    ಧ್ರುವ size_t size_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    ಧ್ರುವ size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    ಧ್ರುವ size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), mask != nullptr, sinks != nullptr, n_head, sess->n_threads);

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_ಮೌಲ್ಯಗಳು(q->ne[3]/v->ne[3]);
    ಯದಿ (mask) {
        kparams->src3_div2 = init_fastdiv_ಮೌಲ್ಯಗಳು(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_ಮೌಲ್ಯಗಳು(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_flash_attn_ext(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src2 = op->src[2];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src3 = op->src[3];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src4 = op->src[4];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    ಯದಿ ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src3 && src3->type != GGML_TYPE_F16) {  // mask
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    ಯದಿ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಸಂರಚನಾ htp_fa_kernel_params kparams;
    ಯದಿ (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_gated_delta_net(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಯದಿ (opt_gdn_select < 1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * q     = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * k     = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * v     = op->src[2];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * g     = op->src[3];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * beta  = op->src[4];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * state = op->src[5];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst   = op;

    ಯದಿ (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int64_t S_v      = v->ne[0];
    ಧ್ರುವ int64_t H        = v->ne[1];
    ಧ್ರುವ int64_t n_tokens = v->ne[2];
    ಧ್ರುವ int64_t n_seqs   = v->ne[3];
    ಧ್ರುವ int64_t K        = ggml_get_op_params_i32(op, 0);

    ಯದಿ (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    ಯದಿ (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t total_rows = (uint32_t) (H * n_seqs);
    ಧ್ರುವ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    ಧ್ರುವ ತಾರ್ಕಿಕ can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    ಯದಿ (can_use_hmx) {
        ಸಂರಚನಾ htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        ಯದಿ (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, ಸತ್ಯ, &n_heads_batch)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    } ಅನ್ಯಥಾ {
        ಸಂರಚನಾ htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_threads);
        ಯದಿ (layout.total_bytes > sess->vtcm_size) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_matmul_is_hmx_eligible(
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ತಾರ್ಕಿಕ is_matmul_id,
    ತಾರ್ಕಿಕ is_batched
) {
    GGML_UNUSED(dst);

    ಯದಿ (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00  = src0->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01  = src0->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11  = src1->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12  = src1->ne[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    ಯದಿ (ne01 % HTP_TENSOR_TILE_N_ROWS != 0 && is_matmul_id) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // HMX supports F16, F32, and repack quantized types.
    ಯದಿ (!htp_is_hmx_weight_type((ggml_type) wtype)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    ಯದಿ (ne00 % HTP_TENSOR_TILE_N_COLS != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    ಧ್ರುವ ತಾರ್ಕಿಕ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ಯದಿ (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!is_matmul_id && is_batched && is_tiled) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // HMX assumes contiguous row-major layout.
    ಯದಿ (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ m = is_matmul_id ? ne12 : ne11;
    ಯದಿ (m <= HTP_MM_HMX_MIN_NROWS) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(dst);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_hmx_mm_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * act,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಪೂರ್ಣಾಙ್ಕ wtype,
    ಪೂರ್ಣಾಙ್ಕ ne00_padded,
    ಪೂರ್ಣಾಙ್ಕ ne01_padded,
    ಪೂರ್ಣಾಙ್ಕ ne02,
    ಪೂರ್ಣಾಙ್ಕ ne11,
    ಪೂರ್ಣಾಙ್ಕ ne12,
    ಪೂರ್ಣಾಙ್ಕ ne11_padded,
    ತಾರ್ಕಿಕ is_matmul_id,
    ತಾರ್ಕಿಕ is_batched,
    size_t src2_size,
    size_t vtcm_budget,
    ಸಂರಚನಾ htp_mm_kernel_params * kparams
) {
    ಧ್ರುವ ತಾರ್ಕಿಕ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ aligned_tile_size = is_tiled ? htp_tensor_get_tile_padded_size(wtype) : 0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_threads = (ಪೂರ್ಣಾಙ್ಕ)sess->n_threads;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10 = act->ne[0];

    ಪೂರ್ಣಾಙ್ಕ m_for_solver = ne11;
    ಪೂರ್ಣಾಙ್ಕ m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    ಯದಿ (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        ಧ್ರುವ ತಾರ್ಕಿಕ dst_row_split = dst ? ggml_hexagon_can_row_partition(dst) : ಸತ್ಯ;
        ಧ್ರುವ ತಾರ್ಕಿಕ act_row_split = ggml_hexagon_can_row_partition(act);
        ಯದಿ (dst_row_split && act_row_split) {
            m_for_solver = (ne11 + (ಪೂರ್ಣಾಙ್ಕ) sess->mdev.count - 1) / (ಪೂರ್ಣಾಙ್ಕ) sess->mdev.count;
            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
        }
    }
    ಧ್ರುವ ತಾರ್ಕಿಕ pipeline = is_matmul_id ? ಅಸತ್ಯ : htp_mm_hmx_pipeline(m_for_solver);

    ಧ್ರುವ ತಾರ್ಕಿಕ is_batched_val = is_matmul_id ? ಅಸತ್ಯ : is_batched;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    ತಾರ್ಕಿಕ use_grouped = ಅಸತ್ಯ;
    ಪೂರ್ಣಾಙ್ಕ act_threads_selected = 0;

    ಯದಿ (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && group_size > 1) {
        // Try grouped path first
        ಯದಿ (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = ಸತ್ಯ;
        }
    }

    ಯದಿ (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        ಪೂರ್ಣಾಙ್ಕ m_id_rows = 0;
        ಯದಿ (dst && is_matmul_id) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_experts = ne02 > 0 ? ne02 : 1;
            ಧ್ರುವ size_t total_expert_rows = (size_t) dst->ne[1] * dst->ne[2];
            ಪೂರ್ಣಾಙ್ಕ m_per_expert = (ಪೂರ್ಣಾಙ್ಕ) ((total_expert_rows + n_experts - 1) / n_experts);
            ಯದಿ (sess->mdev.count > 1 && ggml_hexagon_can_row_partition(dst)) {
                m_per_expert = (m_per_expert + (ಪೂರ್ಣಾಙ್ಕ) sess->mdev.count - 1) / (ಪೂರ್ಣಾಙ್ಕ) sess->mdev.count;
            }
            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
        }
        ಧ್ರುವ uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
        ಯದಿ (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
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
    kparams->div_n_act_threads = init_fastdiv_ಮೌಲ್ಯಗಳು(act_threads_selected);
    kparams->div_ne00_padded   = init_fastdiv_ಮೌಲ್ಯಗಳು(ne00_padded);
    kparams->vtcm_act_size     = 0;
    kparams->vtcm_bias_size    = (int32_t) src2_size;
    kparams->vtcm_dst_size     = 0;

    ಯದಿ (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } ಅನ್ಯಥಾ {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_hvx_mm_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಪೂರ್ಣಾಙ್ಕ wtype,
    ತಾರ್ಕಿಕ is_tiled,
    ಪೂರ್ಣಾಙ್ಕ ne02,
    ಪೂರ್ಣಾಙ್ಕ ne03,
    ಪೂರ್ಣಾಙ್ಕ ne10,
    ಪೂರ್ಣಾಙ್ಕ ne11,
    ಪೂರ್ಣಾಙ್ಕ ne12,
    ಪೂರ್ಣಾಙ್ಕ ne13,
    ತಾರ್ಕಿಕ is_matmul_id,
    ಧ್ರುವ size_t src2_row_size,
    size_t vtcm_budget,
    ಸಂರಚನಾ htp_mm_kernel_params * kparams
) {
    ಯದಿ (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಪ್ರತ್ಯಾಗಮ;
    }

    kparams->n_hmx = 0;
    kparams->n_threads = sess->n_threads;

    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ src1_nrows = ne11 * ne12 * ne13;

    ಯದಿ (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_size = htp_tensor_get_tile_size(wtype);
        kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);

        ಧ್ರುವ ತಾರ್ಕಿಕ k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        ಯದಿ (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (ಪೂರ್ಣಾಙ್ಕ) sess->n_threads)
                                  ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);

            ಸಂರಚನಾ htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            ಪ್ರತಿ (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], kparams->act_row_size, 0, d, ಸತ್ಯ, ಅಸತ್ಯ
                );
                ಯದಿ (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    ಭಙ್ಗ;
                }
            }
            ಯದಿ (L.total_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                ಪ್ರತ್ಯಾಗಮ;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = 0;
            kparams->vtcm_dst_size  = L.dst_bytes;
            ಪ್ರತ್ಯಾಗಮ;
        } ಅನ್ಯಥಾ {
            ತಾರ್ಕಿಕ try_tiled = (k_align && opt_mm_select >= 1);
            ಯದಿ (try_tiled) {
                kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                ಯದಿ (src1_nrows < (ಪೂರ್ಣಾಙ್ಕ) sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } ಅನ್ಯಥಾ {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                ಸಂರಚನಾ htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                ಪ್ರತಿ (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, ಅಸತ್ಯ, ಅಸತ್ಯ
                    );
                    ಯದಿ (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        ಭಙ್ಗ;
                    }
                }

                uint32_t m_chunk = 0;
                ಯದಿ (htp_mm_hvx_solve_vtcm_params(
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
                    ಪ್ರತ್ಯಾಗಮ;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            ಪ್ರತ್ಯಾಗಮ;
        }
    } ಅನ್ಯಥಾ ಯದಿ (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಪ್ರತ್ಯಾಗಮ;
    } ಅನ್ಯಥಾ ಯದಿ (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        ಸಂರಚನಾ htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ಯದಿ (htp_mm_hvx_solve_vtcm_params(
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
            ಪ್ರತ್ಯಾಗಮ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಪ್ರತ್ಯಾಗಮ;
    } ಅನ್ಯಥಾ ಯದಿ (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        ಸಂರಚನಾ htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ಯದಿ (htp_mm_hvx_solve_vtcm_params(
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
            ಪ್ರತ್ಯಾಗಮ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಪ್ರತ್ಯಾಗಮ;
    } ಅನ್ಯಥಾ ಯದಿ (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        ಸಂರಚನಾ htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ಯದಿ (htp_mm_hvx_solve_vtcm_params(
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
            ಪ್ರತ್ಯಾಗಮ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಪ್ರತ್ಯಾಗಮ;
    } ಅನ್ಯಥಾ {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಪ್ರತ್ಯಾಗಮ;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_matmul_params_impl(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಧ್ರುವ size_t src2_row_size,
    ಧ್ರುವ size_t src2_size,
    ಸಂರಚನಾ htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00 = src0->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01 = src0->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02 = src0->ne[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne03 = src0->ne[3];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10 = src1->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11 = src1->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12 = src1->ne[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne13 = src1->ne[3];

    ಧ್ರುವ ತಾರ್ಕಿಕ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11_padded = hex_round_up(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01_tiled  = hex_round_up(ne01, 32);

    ಧ್ರುವ ತಾರ್ಕಿಕ is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    ಧ್ರುವ ತಾರ್ಕಿಕ is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ಧ್ರುವ size_t vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    ತಾರ್ಕಿಕ hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    ಯದಿ (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        ಯದಿ (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
            ಗಚ್ಛ finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_size, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_ಮೌಲ್ಯಗಳು(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_ಮೌಲ್ಯಗಳು(ne11);
    kparams->div_r2       = init_fastdiv_ಮೌಲ್ಯಗಳು(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_ಮೌಲ್ಯಗಳು(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_ಮೌಲ್ಯಗಳು(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_rows_stride(ಧ್ರುವ int64_t * ne, ಧ್ರುವ size_t * nb, size_t * stride) {
    size_t s = 0, next = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i < GGML_MAX_DIMS; i++) {
        ಯದಿ (ne[i] == 1) ಅನುವರ್ತ;
        ಯದಿ (s == 0) { s = nb[i]; next = s * ne[i]; ಅನುವರ್ತ; }
        ಯದಿ (nb[i] != next) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        next *= ne[i];
    }
    *stride = s ? s : nb[1];
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_matmul_can_collapse(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst) {
    size_t s1, sd;
    ಪ್ರತ್ಯಾಗಮ (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_size(src1->type) && ggml_hexagon_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_size(dst->type) && ggml_hexagon_rows_stride(dst->ne, dst->nb, &sd);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_matmul_add_can_collapse(
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src2,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst
) {
    ಯದಿ (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!src2) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಧ್ರುವ int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    ಯದಿ (src2_nrows == 1) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    size_t s2;
    ಪ್ರತ್ಯಾಗಮ src2->nb[0] == ggml_type_size(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_rows_stride(src2->ne, src2->nb, &s2);
}

ಸ್ಥಿರ ggml_tensor ggml_hexagon_tensor_collapse_rows(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    ಪ್ರತ್ಯಾಗಮ c;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_matmul_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_mm_kernel_params * kparams
) {
    ಯದಿ (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ಧ್ರುವ ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        ಧ್ರುವ ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        ಪ್ರತ್ಯಾಗಮ;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_fused_matmul_add_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src2,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_mm_kernel_params * kparams
) {
    ಯದಿ (ggml_hexagon_matmul_add_can_collapse(src0, src1, src2, dst)) {
        ಧ್ರುವ ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        ಧ್ರುವ ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ಧ್ರುವ ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        ಧ್ರುವ size_t src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
        kparams->collapse = 1;
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಧ್ರುವ size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_ದ್ವಿಮಾನ_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    uint32_t op,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_ದ್ವಿಮಾನ_kernel_params * kparams,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src2
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));

    ಧ್ರುವ size_t elem_size = ggml_type_size(src0->type);
    ಧ್ರುವ size_t src0_row_size = src0->ne[0] * elem_size;
    ಧ್ರುವ size_t src1_row_size = src1->ne[0] * elem_size;
    ಧ್ರುವ size_t dst_row_size  = dst->ne[0]  * elem_size;

    ಧ್ರುವ size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    ಧ್ರುವ size_t src1_row_size_aligned = hex_round_up(src1_row_size, 128);
    ಧ್ರುವ size_t dst_row_size_aligned  = hex_round_up(dst_row_size,  128);

    ಯದಿ (!htp_ದ್ವಿಮಾನ_strides_safe(src0, src1, dst, elem_size)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ is_add_id = op == HTP_OP_ADD_ID;
    ಧ್ರುವ ತಾರ್ಕಿಕ is_scalar = !is_add_id && src1->ne[0] == 1;
    ಧ್ರುವ ತಾರ್ಕಿಕ is_row_bcast = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    ಧ್ರುವ ತಾರ್ಕಿಕ is_same_shape = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    ಧ್ರುವ ತಾರ್ಕಿಕ is_complex   = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    ಧ್ರುವ ತಾರ್ಕಿಕ is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    ಧ್ರುವ ತಾರ್ಕಿಕ is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    ಯದಿ (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        ಧ್ರುವ uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        ಧ್ರುವ uint32_t n_threads = sess->n_threads;
        ಧ್ರುವ uint32_t max_chunk_elems = 32768 / elem_size;
        ಧ್ರುವ uint32_t min_chunk_elems = 256;
        ಧ್ರುವ uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
        ಧ್ರುವ uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        ಧ್ರುವ uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

        kparams->kernel_type           = HTP_ದ್ವಿಮಾನ_KERNEL_CHUNKED;
        kparams->n_threads             = n_threads;
        kparams->rows_per_buffer       = 1;
        kparams->src0_row_size_aligned = src0_row_size_aligned;
        kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
        kparams->dst_row_size_aligned  = dst_row_size_aligned;
        kparams->src1_size             = 0;
        kparams->chunk_size            = chunk_size;
        kparams->chunk_bytes           = chunk_bytes;
        kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

        ಸಂರಚನಾ htp_ದ್ವಿಮಾನ_vtcm_layout L;
        htp_ದ್ವಿಮಾನ_vtcm_layout_build(&L, kparams, sess->vtcm_size);
        ಯದಿ (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        kparams->vtcm_size = L.total_bytes;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಗಣನಾ htp_ದ್ವಿಮಾನ_kernel_type kernel_type;
    size_t src1_size = 0;
    size_t src2_size = 0;

    ಯದಿ (is_add_id) {
        GGML_ASSERT(src2 != nullptr);
        kernel_type = HTP_ದ್ವಿಮಾನ_KERNEL_ADD_ID;
        src1_size = hex_round_up(src1->ne[1] * src1_row_size_aligned, 128);
        src2_size = hex_round_up(ggml_nelements(src2) * ಪರಿಮಾಣ(int32_t), 128);
        ಯದಿ (src1_size + src2_size > sess->vtcm_size) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    } ಅನ್ಯಥಾ ಯದಿ (is_row_bcast) {
        kernel_type = HTP_ದ್ವಿಮಾನ_KERNEL_ROW_BCAST;
        src1_size = src1_row_size_aligned;
    } ಅನ್ಯಥಾ ಯದಿ (is_scalar) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size, 128);
        ಯದಿ (src1_size > sess->vtcm_size) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        kernel_type = HTP_ದ್ವಿಮಾನ_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } ಅನ್ಯಥಾ ಯದಿ (is_same_shape) {
        kernel_type = HTP_ದ್ವಿಮಾನ_KERNEL_SAME_SHAPE;
    } ಅನ್ಯಥಾ ಯದಿ (is_complex) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        ಯದಿ (src1_size > sess->vtcm_size) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        kernel_type = HTP_ದ್ವಿಮಾನ_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } ಅನ್ಯಥಾ {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        ಯದಿ (src1_size > sess->vtcm_size) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        kernel_type = HTP_ದ್ವಿಮಾನ_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_threads = sess->n_threads;
    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;
    kparams->src1_size             = src1_size;
    kparams->src2_size             = src2_size;

    ಧ್ರುವ ತಾರ್ಕಿಕ src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    ಧ್ರುವ ತಾರ್ಕಿಕ src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    ಧ್ರುವ ತಾರ್ಕಿಕ dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    ಧ್ರುವ ತಾರ್ಕಿಕ src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    ಧ್ರುವ ತಾರ್ಕಿಕ src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    ಧ್ರುವ ತಾರ್ಕಿಕ dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    ತಾರ್ಕಿಕ split_at_ne01 = ಅಸತ್ಯ;
    ಯದಿ (src0->ne[2] > 1) {
        ಯದಿ (kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_SAME_SHAPE) {
            split_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } ಅನ್ಯಥಾ ಯದಿ (kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_ROW_BCAST) {
            split_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } ಅನ್ಯಥಾ {
            split_at_ne01 = ಸತ್ಯ;
        }
    }
    kparams->split_at_ne01 = split_at_ne01 ? 1 : 0;

    ತಾರ್ಕಿಕ split_at_ne02 = ಅಸತ್ಯ;
    ಯದಿ (src0->ne[3] > 1) {
        ಯದಿ (kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_SAME_SHAPE) {
            split_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } ಅನ್ಯಥಾ ಯದಿ (kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_ROW_BCAST) {
            split_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } ಅನ್ಯಥಾ {
            split_at_ne02 = ಸತ್ಯ;
        }
    }
    kparams->split_at_ne02 = split_at_ne02 ? 1 : 0;

    ಸಂರಚನಾ htp_ದ್ವಿಮಾನ_vtcm_layout L;
    htp_ದ್ವಿಮಾನ_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    ಯದಿ (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {
        ಯದಿ (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            ಧ್ರುವ uint32_t total_elems = (uint32_t) ggml_nelements(src0);
            ಧ್ರುವ uint32_t n_threads = sess->n_threads;
            ಧ್ರುವ uint32_t max_chunk_elems = 32768 / elem_size;
            ಧ್ರುವ uint32_t min_chunk_elems = 256;
            ಧ್ರುವ uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
            ಧ್ರುವ uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
            ಧ್ರುವ uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

            kparams->kernel_type           = HTP_ದ್ವಿಮಾನ_KERNEL_CHUNKED;
            kparams->n_threads             = n_threads;
            kparams->rows_per_buffer       = 1;
            kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
            kparams->src1_size             = 0;
            kparams->chunk_size            = chunk_size;
            kparams->chunk_bytes           = chunk_bytes;
            kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

            htp_ದ್ವಿಮಾನ_vtcm_layout_build(&L, kparams, sess->vtcm_size);
            ಯದಿ (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            kparams->vtcm_size = L.total_bytes;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    kparams->rows_per_buffer = L.rows_per_buffer;
    kparams->vtcm_size       = L.total_bytes;

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_unary_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    uint32_t op,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));

    ಧ್ರುವ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಧ್ರುವ uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    ಧ್ರುವ size_t elem_size = ggml_type_size(src0->type);

    ಧ್ರುವ size_t src0_data_row_size = src0->ne[0] * elem_size;
    ಧ್ರುವ size_t dst_data_row_size  = dst->ne[0]  * ggml_type_size(dst->type);

    ಧ್ರುವ size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    ಧ್ರುವ size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = 0;
    ತಾರ್ಕಿಕ broadcast_weight = ಅಸತ್ಯ;

    ಯದಿ (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_data_row_size = src1->ne[0] * ggml_type_size(src1->type);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->broadcast_weight      = broadcast_weight;

    ಸಂರಚನಾ htp_unary_vtcm_layout L;
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

    ಧ್ರುವ uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_ಮೌಲ್ಯಗಳು(tiles_per_row);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_get_rows_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_get_rows_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));

    ಧ್ರುವ uint32_t ne00 = src0->ne[0];
    ಧ್ರುವ uint32_t ne02 = src0->ne[2];
    ಧ್ರುವ uint32_t ne03 = src0->ne[3];

    ಧ್ರುವ uint32_t ne10 = src1->ne[0];
    ಧ್ರುವ uint32_t ne11 = src1->ne[1];
    ಧ್ರುವ uint32_t ne12 = src1->ne[2];
    ಧ್ರುವ uint32_t nr = ne10 * ne11 * ne12;

    ಧ್ರುವ ತಾರ್ಕಿಕ tiled = ggml_hexagon_tensor_is_tiled(src0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    ಯದಿ (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } ಅನ್ಯಥಾ {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    ಧ್ರುವ uint32_t total_tasks = nr;

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, total_tasks);

    ಸಂರಚನಾ htp_get_rows_vtcm_layout vtcm_layout = {};
    ಯಾವತ್ (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, kparams->n_threads);
        ಯದಿ (vtcm_layout.total_bytes <= sess->vtcm_size) {
            ಭಙ್ಗ;
        }
        --kparams->n_threads;
    }

    ಯದಿ (kparams->n_threads == 0 && total_tasks > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, 1);
    }

    kparams->vtcm_size = (total_tasks == 0) ? 0 : vtcm_layout.total_bytes;
    kparams->tasks_per_thread = kparams->n_threads > 0 ? (total_tasks + kparams->n_threads - 1) / kparams->n_threads : 0;
    kparams->total_tasks = total_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_ಮೌಲ್ಯಗಳು(ne10) : fastdiv_ಮೌಲ್ಯಗಳು{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_ಮೌಲ್ಯಗಳು(ne10 * ne11) : fastdiv_ಮೌಲ್ಯಗಳು{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_ಮೌಲ್ಯಗಳು(ne02) : fastdiv_ಮೌಲ್ಯಗಳು{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_ಮೌಲ್ಯಗಳು(ne03) : fastdiv_ಮೌಲ್ಯಗಳು{0, 0};
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_set_rows_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0, //ಮೌಲ್ಯಗಳು
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1, // indices
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,  // destination
    ಸಂರಚನಾ htp_set_rows_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));

    ಧ್ರುವ uint32_t nr = src0->ne[1];

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, nr);
    kparams->tasks_per_thread = (nr + kparams->n_threads - 1) / kparams->n_threads;
    kparams->total_tasks = nr;

    kparams->div_ne11 = init_fastdiv_ಮೌಲ್ಯಗಳು(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_ಮೌಲ್ಯಗಳು(src1->ne[2]);
    kparams->div_tasks_per_thread = init_fastdiv_ಮೌಲ್ಯಗಳು(kparams->tasks_per_thread);
    kparams->div_ne02 = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[2]);

    ಸಂರಚನಾ htp_set_rows_vtcm_layout vtcm_layout;
    htp_set_rows_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_threads);
    kparams->vtcm_size = vtcm_layout.total_bytes;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_softmax_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];

    ಧ್ರುವ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಧ್ರುವ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);

    ಭಿನ್ನ scale = 1.0f;
    ಭಿನ್ನ max_bias = 0.0f;
    memcpy(&scale,    &op->op_params[0], ಪರಿಮಾಣ(ಭಿನ್ನ));
    memcpy(&max_bias, &op->op_params[1], ಪರಿಮಾಣ(ಭಿನ್ನ));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    ಧ್ರುವ uint32_t n_head = src0->ne[2];
    ಧ್ರುವ uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    kparams->n_head      = n_head;
    kparams->n_head_log2 = n_head_log2;

    ಯದಿ (max_bias > 0.0f && n_head_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_head_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    } ಅನ್ಯಥಾ {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    ಧ್ರುವ uint32_t ne00 = src0->ne[0];
    ಧ್ರುವ uint32_t ne10 = src1 ? src1->ne[0] : 1;

    ಸಂರಚನಾ htp_softmax_vtcm_layout layout;
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
    ಯದಿ (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } ಅನ್ಯಥಾ ಯದಿ (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F16;
    } ಅನ್ಯಥಾ {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    ಯದಿ (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[1]);
    ಯದಿ (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_ಮೌಲ್ಯಗಳು(src0->ne[2]);
    ಧ್ರುವ uint32_t ne12 = src1 ? src1->ne[2] : 1;
    ಧ್ರುವ uint32_t ne13 = src1 ? src1->ne[3] : 1;
    ಯದಿ (ne12 > 0) kparams->div_ne12 = init_fastdiv_ಮೌಲ್ಯಗಳು(ne12);
    ಯದಿ (ne13 > 0) kparams->div_ne13 = init_fastdiv_ಮೌಲ್ಯಗಳು(ne13);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_rope_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src2 = op->src[2];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಧ್ರುವ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಧ್ರುವ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ಧ್ರುವ uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ಸಂರಚನಾ htp_rope_vtcm_layout layout;
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

    ಯದಿ (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_ಮೌಲ್ಯಗಳು(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_ಮೌಲ್ಯಗಳು(dst->ne[1]);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_ssm_conv_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));

    ಧ್ರುವ uint32_t d_conv  = (uint32_t) src1->ne[0];
    ಧ್ರುವ uint32_t d_inner = (uint32_t) src0->ne[1];
    ಧ್ರುವ uint32_t n_t     = (uint32_t) dst->ne[1];
    ಧ್ರುವ uint32_t n_s     = (uint32_t) dst->ne[2];
    ಧ್ರುವ uint32_t ncs     = (uint32_t) src0->ne[0];

    ಧ್ರುವ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    ಧ್ರುವ uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
    ಧ್ರುವ uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);

    ಧ್ರುವ uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * ಪರಿಮಾಣ(ಭಿನ್ನ), 128) + 128;
    ಧ್ರುವ uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * ಪರಿಮಾಣ(ಭಿನ್ನ), 128);
    ಧ್ರುವ uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

    uint32_t vtcm_src0_per_thread = 0;
    uint32_t vtcm_dst_per_thread  = 0;

    ಯದಿ (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;

        ಧ್ರುವ uint32_t src0_tile_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * ಪರಿಮಾಣ(ಭಿನ್ನ), 128);
        ಧ್ರುವ uint32_t src0_T_bytes        = hex_round_up(d_conv * d_inner_per_thread * ಪರಿಮಾಣ(ಭಿನ್ನ), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw_bytes + src0_T_bytes;

        ಧ್ರುವ uint32_t dst_tile_bytes = hex_round_up(d_inner_per_thread * ಪರಿಮಾಣ(ಭಿನ್ನ), 128);
        vtcm_dst_per_thread = 2 * dst_tile_bytes;
    } ಅನ್ಯಥಾ {
        ಧ್ರುವ size_t vtcm_budget = (sess->vtcm_size > 0 ? sess->vtcm_size / n_threads : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        ಧ್ರುವ uint32_t src0_block_T    = hex_round_up(ncs * 32 * ಪರಿಮಾಣ(ಭಿನ್ನ), 128);
        ಧ್ರುವ size_t   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
        ಧ್ರುವ size_t   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);

        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
        target_max_tile = (std::max)(target_max_tile, 32u);
        target_max_tile = (std::min)(target_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * ಪರಿಮಾಣ(ಭಿನ್ನ)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        ಯದಿ (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        ಯದಿ (d_inner_tile > target_max_tile) {
            d_inner_tile = target_max_tile;
        }
        ಯದಿ (d_inner_tile > d_inner_per_thread) {
            d_inner_tile = d_inner_per_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        ಧ್ರುವ uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * ಪರಿಮಾಣ(ಭಿನ್ನ), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw + src0_block_T;

        vtcm_dst_per_thread = 2 * hex_round_up(d_inner_tile * n_t * ಪರಿಮಾಣ(ಭಿನ್ನ), 128);
    }

    kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
    kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
    kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;

    kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
    kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
    kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
    kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_gated_delta_net_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * q     = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * k     = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * v     = op->src[2];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * g     = op->src[3];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * state = op->src[5];

    ಧ್ರುವ uint32_t S_v      = (uint32_t) v->ne[0];
    ಧ್ರುವ uint32_t H        = (uint32_t) v->ne[1];
    ಧ್ರುವ uint32_t n_tokens = (uint32_t) v->ne[2];
    ಧ್ರುವ uint32_t n_seqs   = (uint32_t) v->ne[3];
    ಧ್ರುವ uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    ಧ್ರುವ uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    ಧ್ರುವ uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    ಧ್ರುವ uint32_t total_rows = H * n_seqs;
    ಧ್ರುವ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    ಧ್ರುವ ತಾರ್ಕಿಕ can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    ಸಂರಚನಾ htp_gdn_hmx_vtcm_layout hmx_layout;
    ಸಂರಚನಾ htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    ಯದಿ (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, ಸತ್ಯ, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_size      = HTP_GDN_CHUNK_SIZE;
        kparams->n_chunks        = (n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_size       = (uint32_t) hmx_layout.total_bytes;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_bytes;
        kparams->vtcm_per_thread = (uint32_t) (hmx_layout.total_bytes / (n_threads > 0 ? n_threads : 1));
    } ಅನ್ಯಥಾ {
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
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / ಪರಿಮಾಣ(ಭಿನ್ನ));
    kparams->state_size_per_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((ಭಿನ್ನ) S_v);

    ಯದಿ (H > 0)                  kparams->div_H         = init_fastdiv_ಮೌಲ್ಯಗಳು(H);
    ಯದಿ (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) q->ne[1]);
    ಯದಿ (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) k->ne[1]);
    ಯದಿ (rq3 > 0)                kparams->div_rq3       = init_fastdiv_ಮೌಲ್ಯಗಳು(rq3);
    ಯದಿ (rk3 > 0)                kparams->div_rk3       = init_fastdiv_ಮೌಲ್ಯಗಳು(rk3);
    ಯದಿ (kparams->n_threads > 0) kparams->div_n_threads = init_fastdiv_ಮೌಲ್ಯಗಳು(kparams->n_threads);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_sort_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ತಾರ್ಕಿಕ is_ತುದಿ_k,
    ಸಂರಚನಾ htp_sort_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಧ್ರುವ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಧ್ರುವ uint32_t ne00       = src0->ne[0];
    ಧ್ರುವ uint32_t k          = dst->ne[0];

    int32_t order = GGML_SORT_ORDER_DESC;
    ಯದಿ (!is_ತುದಿ_k) {
        order = ((ಧ್ರುವ int32_t *) op->op_params)[0];
    }

    ಧ್ರುವ uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ಧ್ರುವ size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    ಸಂರಚನಾ htp_sort_vtcm_layout layout;
    ತಾರ್ಕಿಕ ok = htp_sort_solve_layout(&layout, ne00, total_rows, k, n_threads_max, vtcm_budget, is_ತುದಿ_k);
    GGML_ASSERT(ok);

    kparams->n_threads         = (int32_t) layout.n_threads;
    kparams->total_rows        = (int32_t) total_rows;
    kparams->row_start         = 0;
    kparams->row_end           = (int32_t) total_rows;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->order             = order;
    kparams->is_ತುದಿ_k          = is_ತುದಿ_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_size         = (int32_t) layout.total_bytes;
    kparams->phase1_slot_size  = (int32_t) layout.phase1_slot_size;
    kparams->merge_ಮೌಲ್ಯಗಳು_off  = (int32_t) layout.merge_ಮೌಲ್ಯಗಳು_off;
    kparams->merge_indices_off = (int32_t) layout.merge_indices_off;
    kparams->merge_elems       = (int32_t) layout.merge_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    ಧ್ರುವ uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    ಧ್ರುವ int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    ಯದಿ (numer_hi < 0) {
        *ox_hi = 0;
    } ಅನ್ಯಥಾ {
        ಧ್ರುವ uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    ಯದಿ (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_pool_shape_fits(
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst
) {
    ಪ್ರತ್ಯಾಗಮ ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_pool_2d_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,
    ಸಂರಚನಾ htp_pool_2d_kernel_params * kparams,
    ತಾರ್ಕಿಕ is_pool_1d
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));

    ಧ್ರುವ uint32_t src_x = (uint32_t) src0->ne[0];
    ಧ್ರುವ uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    ಧ್ರುವ uint32_t dst_x = (uint32_t) dst->ne[0];
    ಧ್ರುವ uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    ಧ್ರುವ uint32_t src_plane_bytes = src_x * src_y * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t dst_plane_bytes = dst_x * dst_y * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    ಧ್ರುವ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    ಸಂರಚನಾ htp_pool_vtcm_layout layout;
    ಧ್ರುವ ತಾರ್ಕಿಕ ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, sess->vtcm_size);
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
    kparams->avg_divide_ಎಣಿಕೆ = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    ಧ್ರುವ ತಾರ್ಕಿಕ narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_path = (kparams->fast_path && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (ಭಿನ್ನ) (kparams->kernel_x * kparams->kernel_y);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_concat_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (!src0 || !src1 || !dst) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪೂರ್ಣಾಙ್ಕ dim = ((ಧ್ರುವ int32_t *) op->op_params)[0];
    ಯದಿ (dim < 0 || dim >= GGML_MAX_DIMS) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    kparams->dim = dim;

    ಯದಿ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src0->type != dst->type || src1->type != dst->type) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t type_size = ggml_type_size(dst->type);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ d = 0; d < GGML_MAX_DIMS; d++) {
        ಧ್ರುವ int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        ಯದಿ (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಧ್ರುವ uint32_t max_gather_ne0 = 128 / type_size;
    ಧ್ರುವ uint32_t total_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    ಧ್ರುವ ತಾರ್ಕಿಕ b_dense = (src1->nb[0] == type_size);
    ಧ್ರುವ ತಾರ್ಕಿಕ gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_size &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_size
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_size);

    ಯದಿ (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_size && dst->nb[1] == (size_t) dst->ne[0] * type_size &&
        src0->nb[0] == type_size &&
        (src1->nb[0] == type_size || src1->nb[1] == type_size) &&
        gather_strides_ok &&
        htp_gather_rows_tensor_outer_contiguous(dst) &&
        htp_gather_rows_tensor_outer_contiguous(src0) &&
        htp_gather_rows_tensor_outer_contiguous(src1)) {

        ಯದಿ (htp_gather_rows_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_size, total_rows,
                                         sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ROWS;
            kparams->n_threads   = (uint8_t) sess->n_threads;
            kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ dma_strides_ok = (src0->nb[0] == type_size && src1->nb[0] == type_size && dst->nb[0] == type_size);

    ಯದಿ (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_threads   = 1;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    ಧ್ರುವ ತಾರ್ಕಿಕ is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    ಧ್ರುವ ತಾರ್ಕಿಕ transposed_rows_ok = (src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size);

    ಯದಿ (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        ಧ್ರುವ uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 8;
        ಸಂರಚನಾ htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_size, n_threads);

        ಯದಿ (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_threads                         = n_threads;
        kparams->vtcm_size                         = layout.total_bytes;
        kparams->u.transposed.spad0_size_per_thread = layout.src0_spad_size_per_thread;
        kparams->u.transposed.spad1_size_per_thread = layout.src1_spad_size_per_thread;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_cpy_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (!src0 || !dst) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int64_t nelem_src = ggml_nelements(src0);
    ಧ್ರುವ int64_t nelem_dst = ggml_nelements(dst);
    ಯದಿ (nelem_src != nelem_dst || nelem_src < 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t src_type_size = ggml_type_size(src0->type);
    ಧ್ರುವ uint32_t dst_type_size = ggml_type_size(dst->type);

    kparams->src0_type_size = (uint8_t) src_type_size;
    kparams->dst_type_size  = (uint8_t) dst_type_size;
    kparams->total_elems    = (uint32_t) nelem_src;

    ಯದಿ (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (nelem_src == 1) {
        ಯದಿ (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಯದಿ ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ sametype = (src0->type == dst->type);

    ತಾರ್ಕಿಕ same_extents = ಸತ್ಯ;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ d = 0; d < GGML_MAX_DIMS; d++) {
        ಯದಿ (src0->ne[d] != dst->ne[d]) {
            same_extents = ಅಸತ್ಯ;
            ಭಙ್ಗ;
        }
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_size) || (dst->nb[0] != dst_type_size) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_size) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_size);
    ಧ್ರುವ ತಾರ್ಕಿಕ sameshape  = same_extents && !transposed;

    ಧ್ರುವ ತಾರ್ಕಿಕ src_is_contiguous = ggml_is_contiguous(src0);
    ಧ್ರುವ ತಾರ್ಕಿಕ dst_is_contiguous = ggml_is_contiguous(dst);

    ಯದಿ (sametype) {
        ಯದಿ (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಧ್ರುವ uint32_t max_gather_ne0 = 128 / src_type_size;
        ಧ್ರುವ uint32_t total_rows = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        ಯದಿ ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_size && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_size &&
            total_rows > 1 &&
            htp_gather_rows_tensor_outer_contiguous(src0)) {

            ಯದಿ (htp_gather_rows_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, ಅಸತ್ಯ,
                                             src_type_size, total_rows,
                                             sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ROWS;
                kparams->n_threads   = (uint8_t) sess->n_threads;
                kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        }

        ಯದಿ (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_threads   = sess->n_threads > 0 ? (uint8_t) sess->n_threads : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (!sameshape) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    ಯದಿ (!valid_conversion) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 4;
    ಸಂರಚನಾ htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_size, dst_type_size, n_threads);

    ಯದಿ (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
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
    kparams->u.convert.div_ne01      = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_ಮೌಲ್ಯಗಳು((uint32_t) (src0->ne[2] * src0->ne[1]));

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_fused_mmnx_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0, // W0
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * act,  // x
    int32_t n_weights,
    ಸಂರಚನಾ htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));
    kparams->n_threads = sess->n_threads;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00 = src0->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01 = src0->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02 = src0->ne[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne03 = src0->ne[3];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10 = act->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11 = act->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12 = act->ne[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne13 = act->ne[3];

    ಧ್ರುವ ತಾರ್ಕಿಕ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11_padded = hex_round_up(ne11, 32);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01_tiled  = hex_round_up(ne01, 32);

    ಧ್ರುವ size_t vtcm_budget = sess->vtcm_size;
    ಧ್ರುವ ತಾರ್ಕಿಕ is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ತಾರ್ಕಿಕ hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    ಯದಿ (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, nullptr, ಅಸತ್ಯ, is_batched)) {
        ಯದಿ (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, ಅಸತ್ಯ, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            ಗಚ್ಛ finalize;
        }
    }

    ಯದಿ (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ಪ್ರತ್ಯಾಗಮ;
    }

    {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ act_nrows = ne11 * ne12 * ne13;
        ಧ್ರುವ size_t act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
        ಧ್ರುವ size_t src0_row_size = src0->nb[1];

        ಧ್ರುವ uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        ಪ್ರತಿ (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            ಸಂರಚನಾ htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
                0, src0_row_size, act_row_size, 0, d, ಅಸತ್ಯ, ಸತ್ಯ
            );
            ಯದಿ (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                ಭಙ್ಗ;
            }
        }

        ಸಂರಚನಾ htp_mm_hvx_vtcm_layout L;
        ತಾರ್ಕಿಕ try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
            0, src0_row_size, act_row_size, 0, best_n_prefetch, ಅಸತ್ಯ, ಸತ್ಯ
        );

        ಯದಿ (try_tiled && L.total_bytes <= sess->vtcm_size) {
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
        } ಅನ್ಯಥಾ {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            ಪ್ರತ್ಯಾಗಮ;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_ಮೌಲ್ಯಗಳು(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_ಮೌಲ್ಯಗಳು(ne11);
    kparams->div_r2       = init_fastdiv_ಮೌಲ್ಯಗಳು(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_ಮೌಲ್ಯಗಳು(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_ಮೌಲ್ಯಗಳು(ne12);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_precompute_fused_mmidnx_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0, // W0
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * act,  // x
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst,  // dst0
    int32_t n_weights,
    ಸಂರಚನಾ htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_op_mul_mat_use_fwht(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst) {
    ಯದಿ (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = dst->src[1];

    ಧ್ರುವ int64_t n = src1->ne[0];
    ಯದಿ (n < 32 || (n & (n - 1)) != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_rows(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_mul_mat(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = dst->src[1];

    ಯದಿ (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        ಯದಿ (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಸಂರಚನಾ htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, nullptr, dst, &ukparams);
        ಪ್ರತ್ಯಾಗಮ ukparams.vtcm_row_per_thread > 0 && (size_t) ukparams.vtcm_size <= sess->vtcm_size;
    }

    ಯದಿ (dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ವಿಕಲ್ಪ (src0->type) {
        ಪಕ್ಷ GGML_TYPE_Q1_0:
        ಪಕ್ಷ GGML_TYPE_Q2_0:
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
        ಪಕ್ಷ GGML_TYPE_MXFP4:
        ಪಕ್ಷ GGML_TYPE_Q4_K:
        ಪಕ್ಷ GGML_TYPE_Q5_K:
        ಪಕ್ಷ GGML_TYPE_Q6_K:
        ಪಕ್ಷ GGML_TYPE_Q3_K:
        ಪಕ್ಷ GGML_TYPE_Q2_K: {
            ಯದಿ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಧ್ರುವ int64_t qk = ggml_blck_size(src0->type);
            ಯದಿ (src0->ne[0] % qk) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಯದಿ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಭಙ್ಗ;
        }

        ಪಕ್ಷ GGML_TYPE_F16:
            ಯದಿ (src0->nb[1] < src0->nb[0]) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_TYPE_F32:
            ಯದಿ (src1->type != GGML_TYPE_F32) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (src0->nb[1] < src0->nb[0]) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_TYPE_BF16:
            ಯದಿ (src0->nb[1] < src0->nb[0]) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಭಙ್ಗ;

        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಸಂರಚನಾ htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    ಯದಿ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_mul_mat_id(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src2 = op->src[2];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ವಿಕಲ್ಪ (src0->type) {
        ಪಕ್ಷ GGML_TYPE_Q1_0:
        ಪಕ್ಷ GGML_TYPE_Q2_0:
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
        ಪಕ್ಷ GGML_TYPE_MXFP4:
        ಪಕ್ಷ GGML_TYPE_Q4_K:
        ಪಕ್ಷ GGML_TYPE_Q5_K:
        ಪಕ್ಷ GGML_TYPE_Q6_K:
        ಪಕ್ಷ GGML_TYPE_Q3_K:
        ಪಕ್ಷ GGML_TYPE_Q2_K:
            ಯದಿ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಯದಿ (src0->ne[0] % ggml_blck_size(src0->type) != 0) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_TYPE_F16:
        ಪಕ್ಷ GGML_TYPE_F32:
        ಪಕ್ಷ GGML_TYPE_BF16:
            ಯದಿ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಯದಿ (src0->ne[0] % 32 != 0) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಭಙ್ಗ;

        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಸಂರಚನಾ htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    ಯದಿ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_ದ್ವಿಮಾನ(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src0->type == GGML_TYPE_F32) {
        ಯದಿ (src1->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (dst->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }
    ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16) {
        ಯದಿ (src1->type != GGML_TYPE_F16) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (dst->type != GGML_TYPE_F16) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }
    ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_are_same_shape(src0, dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಸಂರಚನಾ htp_ದ್ವಿಮಾನ_kernel_params kparams;
    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_precompute_ದ್ವಿಮಾನ_params(sess, op_remap_to_htp(op), src0, src1, dst, &kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_add_id(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src2 = op->src[2];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (!src2) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_are_same_shape(src0, dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src0->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || src1->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || dst->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || src2->nb[0] != ಪರಿಮಾಣ(int32_t)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // REVISIT: add support for non-contiguous tensors
    ಯದಿ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src2->nb[1] < src2->ne[0] * ಪರಿಮಾಣ(int32_t)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಸಂರಚನಾ htp_ದ್ವಿಮಾನ_kernel_params kparams;
    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_precompute_ದ್ವಿಮಾನ_params(sess, HTP_OP_ADD_ID, src0, src1, dst, &kparams, src2);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_unary(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (dst->type != src0->type) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_is_contiguous_rows(src0)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    ಯದಿ (src0->type == GGML_TYPE_F16) {
        ವಿಕಲ್ಪ (op->op) {
            ಪಕ್ಷ GGML_OP_NORM:
            ಪಕ್ಷ GGML_OP_RMS_NORM:
            ಪಕ್ಷ GGML_OP_L2_NORM:
            ಪಕ್ಷ GGML_OP_SCALE:
            ಪಕ್ಷ GGML_OP_CLAMP:
            ಪಕ್ಷ GGML_OP_SQR:
            ಪಕ್ಷ GGML_OP_SQRT:
            ಪಕ್ಷ GGML_OP_LOG:
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_OP_UNARY:
                ಯದಿ (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_STEP) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಯದಿ (!ggml_are_same_shape(src0, dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // dst must be contiguous; src0 may be non-contiguous
    ಯದಿ (!ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_sum(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src0->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_sum_rows(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src0->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // TODO: add support for non-contigiuos tensors
    ಯದಿ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_argmax(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src0->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (dst->type != GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_pool_2d(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ int32_t * params = op->op_params;

    ಯದಿ (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_hexagon_pool_shape_fits(src0, op)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int32_t kernel_x = params[1];
    ಧ್ರುವ int32_t kernel_y = params[2];
    ಧ್ರುವ int32_t stride_x = params[3];
    ಧ್ರುವ int32_t stride_y = params[4];
    ಧ್ರುವ int32_t pad_x    = params[5];
    ಧ್ರುವ int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    ಯದಿ (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    ಧ್ರುವ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    ಸಂರಚನಾ htp_pool_vtcm_layout layout;
    ಪ್ರತ್ಯಾಗಮ htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_threads, sess->vtcm_size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_pool_1d(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ int32_t * params = op->op_params;

    ಯದಿ (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_hexagon_pool_shape_fits(src0, op)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int32_t kernel = params[1];
    ಧ್ರುವ int32_t stride = params[2];
    ಧ್ರುವ int32_t pad    = params[3];
    ಯದಿ (kernel <= 0 || stride <= 0 || pad < 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ಧ್ರುವ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    ಸಂರಚನಾ htp_pool_vtcm_layout layout;
    ಪ್ರತ್ಯಾಗಮ htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_threads, sess->vtcm_size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_activations(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (dst->type != src0->type) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_contiguous_1(src0)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src1) {
        ಯದಿ (src1->type != src0->type) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (!ggml_are_same_shape(src0, src1)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (!ggml_is_contiguous_1(src1)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_softmax(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src2 = op->src[2];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src2) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;  // FIXME: add support for sinks
    }

    ಯದಿ (src0->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src0->ne[2] > 512) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src1) {
        ಯದಿ (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (src0->ne[0] != src1->ne[0]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (src1->ne[1] < src0->ne[1]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (src0->ne[2] % src1->ne[2] != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (src0->ne[3] % src1->ne[3] != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಯದಿ (src1) {
        ಯದಿ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    } ಅನ್ಯಥಾ {
        ಯದಿ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    ಧ್ರುವ int64_t ne0 = src0->ne[0];
    ಯದಿ (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    ಯದಿ (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಧ್ರುವ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ಸಂರಚನಾ htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != nullptr, src1 && src1->type == GGML_TYPE_F16, n_threads);
    ಯದಿ (layout.total_bytes > sess->vtcm_size) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_set_rows(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0]; //ಮೌಲ್ಯಗಳು
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1]; // indices
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    ಯದಿ (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src0->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_get_rows(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0]; //ಮೌಲ್ಯಗಳು
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1]; // indices
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src0->type == dst->type) {
        ಯದಿ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    } ಅನ್ಯಥಾ ಯದಿ (dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    ಯದಿ (ggml_is_empty(op)) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಸಂರಚನಾ htp_get_rows_kernel_params kparams;
    ggml_hexagon_precompute_get_rows_params(sess, src0, src1, dst, &kparams);
    ಯದಿ (kparams.n_threads == 0 || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_argsort(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0]; //ಮೌಲ್ಯಗಳು
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;         // indices

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಧ್ರುವ uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ಧ್ರುವ size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    ಸಂರಚನಾ htp_sort_vtcm_layout layout;
    ಯದಿ (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, ಅಸತ್ಯ)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_ತುದಿ_k(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0]; //ಮೌಲ್ಯಗಳು
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;         // indices

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಧ್ರುವ uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ಧ್ರುವ size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    ಸಂರಚನಾ htp_sort_vtcm_layout layout;
    ಯದಿ (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, ಸತ್ಯ)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_rope(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src2 = op->src[2];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (!ggml_are_same_shape(src0, dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src0->ne[0] <= 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಯದಿ (src0_nrows == 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int32_t * op_params = &op->op_params[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_dims = op_params[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode   = op_params[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    ಯದಿ (n_dims < 0 || n_dims % 2 != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    ಯದಿ (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಭಿನ್ನ freq_base;
    memcpy(&freq_base, op_params + 5, ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಯದಿ (freq_base < 0.0f) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    ಯದಿ (mode == GGML_ROPE_TYPE_VISION) {
        ಯದಿ (n_dims != (ಪೂರ್ಣಾಙ್ಕ) (src0->ne[0] / 2) || n_offs != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಯದಿ (is_mrope) {
        ಧ್ರುವ int32_t * sections = op_params + 11;
        ಯದಿ (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಧ್ರುವ int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    ಯದಿ (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src2) {
        ಯದಿ (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (src2->ne[0] < (n_dims / 2)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    ಯದಿ (src0->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || dst->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src0->nb[1] < src0->ne[0] * ಪರಿಮಾಣ(ಭಿನ್ನ) || dst->nb[1] < dst->ne[0] * ಪರಿಮಾಣ(ಭಿನ್ನ)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ಧ್ರುವ uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ಸಂರಚನಾ htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    ಯದಿ (layout.total_bytes > sess->vtcm_size) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_ssm_conv(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    // Only support FP32 for now
    ಯದಿ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Check IO tensor shapes and dims
    ಯದಿ (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; // src0 should be effectively 3D
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ d_conv  = src1->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ d_inner = src0->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_t = dst->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_s = dst->ne[2];

    ಯದಿ (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src0->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || src1->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || dst->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (src0->nb[1] != src0->ne[0] * ಪರಿಮಾಣ(ಭಿನ್ನ) || src1->nb[1] != src1->ne[0] * ಪರಿಮಾಣ(ಭಿನ್ನ)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಸಂರಚನಾ htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    ಯದಿ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ im2col_3d_calc_output_size(uint32_t ಒಳಹರಿವು, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    ಯದಿ (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int64_t numerator = (int64_t) ಒಳಹರಿವು + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    ಯದಿ (numerator < 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint64_t size = (uint64_t) (numerator / stride) + 1;
    ಯದಿ (size == 0 || size > UINT32_MAX) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    *output = (uint32_t) size;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_im2col_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (!src0 || !src1 || !dst) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src1->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || !ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int32_t * op_params = (ಧ್ರುವ int32_t *) op->op_params;
    ಧ್ರುವ int32_t s0 = op_params[0];
    ಧ್ರುವ int32_t s1 = op_params[1];
    ಧ್ರುವ int32_t p0 = op_params[2];
    ಧ್ರುವ int32_t p1 = op_params[3];
    ಧ್ರುವ int32_t d0 = op_params[4];
    ಧ್ರುವ int32_t d1 = op_params[5];
    ಧ್ರುವ ತಾರ್ಕಿಕ is_2D = (op_params[6] == 1);

    ಯದಿ (s0 <= 0 || d0 <= 0 || p0 < 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    ಧ್ರುವ uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    ಧ್ರುವ uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    ಧ್ರುವ uint32_t IW = (uint32_t) src1->ne[0];
    ಧ್ರುವ uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    ಧ್ರುವ uint32_t KW = (uint32_t) src0->ne[0];
    ಧ್ರುವ uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    ಧ್ರುವ uint32_t OW = (uint32_t) dst->ne[1];

    ಯದಿ (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    ಧ್ರುವ uint64_t total_rows_64   = (uint64_t) N * OH;
    ಯದಿ (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ uint32_t patch_stride = (uint32_t) patch_stride_64;
    ಧ್ರುವ uint32_t total_rows   = (uint32_t) total_rows_64;

    ಯದಿ (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (is_2D && dst->ne[2] != OH) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? ಪರಿಮಾಣ(ggml_fp16_t) : ಪರಿಮಾಣ(ಭಿನ್ನ);

    ತಾರ್ಕಿಕ is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    ಯದಿ (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    ಧ್ರುವ uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    ಧ್ರುವ uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    ಧ್ರುವ uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    ಯದಿ (is_patchembed) {
        ಧ್ರುವ uint64_t src_row_bytes_raw = (uint64_t) IC * KH * IW * ಪರಿಮಾಣ(ಭಿನ್ನ);
        ಧ್ರುವ uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        ಯದಿ (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            ಧ್ರುವ uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            ಧ್ರುವ uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            ಸಂರಚನಾ htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            ಯದಿ (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        }
    }

    ಪ್ರತಿ (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ಧ್ರುವ uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        ಯದಿ (Wb > UINT32_MAX) {
            ಯದಿ (owb == 1) ಭಙ್ಗ;
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ uint64_t src_row_bytes_raw = (uint64_t) IC * KH * Wb * ಪರಿಮಾಣ(ಭಿನ್ನ);
        ಧ್ರುವ uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        ಯದಿ (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            ಯದಿ (owb == 1) ಭಙ್ಗ;
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        ಧ್ರುವ uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        ಸಂರಚನಾ htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        ಯದಿ (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಯದಿ (owb == 1) ಭಙ್ಗ;
    }

    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_precompute_im2col_3d_params(
    ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess,
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op,
    ಸಂರಚನಾ htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, ಪರಿಮಾಣ(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (!src0 || !src1 || !dst) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src1->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || !ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int32_t * params = (ಧ್ರುವ int32_t *) op->op_params;
    ಧ್ರುವ int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    ಧ್ರುವ int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    ಧ್ರುವ int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    ಧ್ರುವ int32_t ic = params[9];

    ಯದಿ (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src1->ne[3] % ic != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t IC = (uint32_t) ic;
    ಧ್ರುವ uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    ಯದಿ (N == 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    uint32_t OW, OH, OD;
    ಯದಿ (!im2col_3d_calc_output_size(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_size(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_size(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t KD = (uint32_t) src0->ne[2];
    ಧ್ರುವ uint32_t KH = (uint32_t) src0->ne[1];
    ಧ್ರುವ uint32_t KW = (uint32_t) src0->ne[0];
    ಧ್ರುವ uint32_t ID = (uint32_t) src1->ne[2];
    ಧ್ರುವ uint32_t IH = (uint32_t) src1->ne[1];
    ಧ್ರುವ uint32_t IW = (uint32_t) src1->ne[0];

    ಧ್ರುವ uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    ಧ್ರುವ uint64_t total_rows_64   = (uint64_t) N * OD * OH;
    ಧ್ರುವ uint64_t dst_ne3_64      = (uint64_t) OD * N;

    ಯದಿ (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ uint32_t patch_stride = (uint32_t) patch_stride_64;
    ಧ್ರುವ uint32_t total_rows   = (uint32_t) total_rows_64;
    ಧ್ರುವ uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    ಯದಿ (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? ಪರಿಮಾಣ(ggml_fp16_t) : ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಧ್ರುವ ತಾರ್ಕಿಕ is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    ಧ್ರುವ uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    ಧ್ರುವ uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    ಧ್ರುವ uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    ಯದಿ (is_patchembed) {
        ಧ್ರುವ uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * IW * ಪರಿಮಾಣ(ಭಿನ್ನ);
        ಧ್ರುವ uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        ಯದಿ (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            ಧ್ರುವ uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            ಧ್ರುವ uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            ಸಂರಚನಾ htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            ಯದಿ (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        }
    }

    ಪ್ರತಿ (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ಧ್ರುವ uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        ಯದಿ (Wb > UINT32_MAX) {
            ಯದಿ (owb == 1) ಭಙ್ಗ;
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * Wb * ಪರಿಮಾಣ(ಭಿನ್ನ);
        ಧ್ರುವ uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        ಯದಿ (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            ಯದಿ (owb == 1) ಭಙ್ಗ;
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        ಧ್ರುವ uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        ಸಂರಚನಾ htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        ಯದಿ (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಯದಿ (owb == 1) ಭಙ್ಗ;
    }

    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_im2col(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಸಂರಚನಾ htp_im2col_kernel_params kparams;
    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_im2col_3d(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಸಂರಚನಾ htp_im2col_kernel_params kparams;
    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_pad(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int32_t lp0 = ((ಧ್ರುವ int32_t *) op->op_params)[0];
    ಧ್ರುವ int32_t rp0 = ((ಧ್ರುವ int32_t *) op->op_params)[1];
    ಧ್ರುವ int32_t circular = ((ಧ್ರುವ int32_t *) op->op_params)[8];

    ಯದಿ (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_cumsum(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_diag(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    // diag only supports F32 currently
    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Input must have ne[1] == 1 (vector input)
    ಯದಿ (src0->ne[1] != 1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Output must be square in first two dimensions
    ಯದಿ (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_solve_tri(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0]; // A
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = op->src[1]; // B
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;         // X

    ಯದಿ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src0->ne[0] != src0->ne[1]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src0->ne[1] != src1->ne[1]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_tri(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src0->type != GGML_TYPE_F32) { ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; }
    ಯದಿ (dst->type  != GGML_TYPE_F32) { ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; }
    ಯದಿ (!ggml_are_same_shape(src0, dst)) { ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; }
    ಯದಿ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    ಪ್ರತ್ಯಾಗಮ sess->c_name();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

ಸ್ಥಿರ htp_op_code op_remap_to_htp(ಧ್ರುವ ggml_tensor * t) {
    ವಿಕಲ್ಪ (t->op) {
        ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:  ಪ್ರತ್ಯಾಗಮ HTP_OP_FLASH_ATTN_EXT;
        ಪಕ್ಷ GGML_OP_MUL_MAT:
            ಯದಿ (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                ಪ್ರತ್ಯಾಗಮ HTP_OP_HADAMARD;
            }
            ಪ್ರತ್ಯಾಗಮ HTP_OP_MUL_MAT;
        ಪಕ್ಷ GGML_OP_MUL_MAT_ID:      ಪ್ರತ್ಯಾಗಮ HTP_OP_MUL_MAT_ID;
        ಪಕ್ಷ GGML_OP_MUL:             ಪ್ರತ್ಯಾಗಮ HTP_OP_MUL;
        ಪಕ್ಷ GGML_OP_ADD:             ಪ್ರತ್ಯಾಗಮ HTP_OP_ADD;
        ಪಕ್ಷ GGML_OP_ADD_ID:          ಪ್ರತ್ಯಾಗಮ HTP_OP_ADD_ID;
        ಪಕ್ಷ GGML_OP_SUB:             ಪ್ರತ್ಯಾಗಮ HTP_OP_SUB;
        ಪಕ್ಷ GGML_OP_DIV:             ಪ್ರತ್ಯಾಗಮ HTP_OP_DIV;
        ಪಕ್ಷ GGML_OP_DUP:
        ಪಕ್ಷ GGML_OP_CPY:             ಪ್ರತ್ಯಾಗಮ HTP_OP_CPY;
        ಪಕ್ಷ GGML_OP_CONT:            ಪ್ರತ್ಯಾಗಮ HTP_OP_CPY;
        ಪಕ್ಷ GGML_OP_GET_ROWS:        ಪ್ರತ್ಯಾಗಮ HTP_OP_GET_ROWS;
        ಪಕ್ಷ GGML_OP_SET_ROWS:        ಪ್ರತ್ಯಾಗಮ HTP_OP_SET_ROWS;
        ಪಕ್ಷ GGML_OP_SUM:             ಪ್ರತ್ಯಾಗಮ HTP_OP_SUM;
        ಪಕ್ಷ GGML_OP_SUM_ROWS:        ಪ್ರತ್ಯಾಗಮ HTP_OP_SUM_ROWS;
        ಪಕ್ಷ GGML_OP_POOL_2D:         ಪ್ರತ್ಯಾಗಮ HTP_OP_POOL_2D;
        ಪಕ್ಷ GGML_OP_POOL_1D:         ಪ್ರತ್ಯಾಗಮ HTP_OP_POOL_1D;
        ಪಕ್ಷ GGML_OP_ARGSORT:         ಪ್ರತ್ಯಾಗಮ HTP_OP_ARGSORT;
        ಪಕ್ಷ GGML_OP_ತುದಿ_K:           ಪ್ರತ್ಯಾಗಮ HTP_OP_ತುದಿ_K;
        ಪಕ್ಷ GGML_OP_ARGMAX:          ಪ್ರತ್ಯಾಗಮ HTP_OP_ARGMAX;
        ಪಕ್ಷ GGML_OP_NORM:            ಪ್ರತ್ಯಾಗಮ HTP_OP_NORM;
        ಪಕ್ಷ GGML_OP_L2_NORM:         ಪ್ರತ್ಯಾಗಮ HTP_OP_L2_NORM;
        ಪಕ್ಷ GGML_OP_RMS_NORM:        ಪ್ರತ್ಯಾಗಮ HTP_OP_RMS_NORM;
        ಪಕ್ಷ GGML_OP_CONCAT:          ಪ್ರತ್ಯಾಗಮ HTP_OP_CONCAT;
        ಪಕ್ಷ GGML_OP_SCALE:           ಪ್ರತ್ಯಾಗಮ HTP_OP_SCALE;
        ಪಕ್ಷ GGML_OP_CLAMP:           ಪ್ರತ್ಯಾಗಮ HTP_OP_CLAMP;
        ಪಕ್ಷ GGML_OP_LEAKY_RELU:      ಪ್ರತ್ಯಾಗಮ HTP_OP_LEAKY_RELU;
        ಪಕ್ಷ GGML_OP_SQR:             ಪ್ರತ್ಯಾಗಮ HTP_OP_SQR;
        ಪಕ್ಷ GGML_OP_SQRT:            ಪ್ರತ್ಯಾಗಮ HTP_OP_SQRT;
        ಪಕ್ಷ GGML_OP_LOG:             ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_LOG;
        ಪಕ್ಷ GGML_OP_SOFT_MAX:        ಪ್ರತ್ಯಾಗಮ HTP_OP_SOFTMAX;
        ಪಕ್ಷ GGML_OP_SSM_CONV:        ಪ್ರತ್ಯಾಗಮ HTP_OP_SSM_CONV;
        ಪಕ್ಷ GGML_OP_GATED_DELTA_NET: ಪ್ರತ್ಯಾಗಮ HTP_OP_GATED_DELTA_NET;
        ಪಕ್ಷ GGML_OP_ROPE:            ಪ್ರತ್ಯಾಗಮ HTP_OP_ROPE;
        ಪಕ್ಷ GGML_OP_REPEAT:          ಪ್ರತ್ಯಾಗಮ HTP_OP_REPEAT;
        ಪಕ್ಷ GGML_OP_CUMSUM:          ಪ್ರತ್ಯಾಗಮ HTP_OP_CUMSUM;
        ಪಕ್ಷ GGML_OP_FILL:            ಪ್ರತ್ಯಾಗಮ HTP_OP_FILL;
        ಪಕ್ಷ GGML_OP_DIAG:            ಪ್ರತ್ಯಾಗಮ HTP_OP_DIAG;
        ಪಕ್ಷ GGML_OP_SOLVE_TRI:       ಪ್ರತ್ಯಾಗಮ HTP_OP_SOLVE_TRI;
        ಪಕ್ಷ GGML_OP_TRI:             ಪ್ರತ್ಯಾಗಮ HTP_OP_TRI;
        ಪಕ್ಷ GGML_OP_PAD:             ಪ್ರತ್ಯಾಗಮ HTP_OP_PAD;
        ಪಕ್ಷ GGML_OP_IM2COL:          ಪ್ರತ್ಯಾಗಮ HTP_OP_IM2COL;
        ಪಕ್ಷ GGML_OP_IM2COL_3D:       ಪ್ರತ್ಯಾಗಮ HTP_OP_IM2COL_3D;
        ಪಕ್ಷ GGML_OP_ROLL:            ಪ್ರತ್ಯಾಗಮ HTP_OP_ROLL;

        ಪಕ್ಷ GGML_OP_UNARY:
            ವಿಕಲ್ಪ (ggml_get_unary_op(t)) {
                ಪಕ್ಷ GGML_UNARY_OP_SILU:       ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_SILU;
                ಪಕ್ಷ GGML_UNARY_OP_GELU:       ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_GELU;
                ಪಕ್ಷ GGML_UNARY_OP_GELU_QUICK: ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_GELU_QUICK;
                ಪಕ್ಷ GGML_UNARY_OP_GELU_ERF:   ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_GELU_ERF;
                ಪಕ್ಷ GGML_UNARY_OP_SIGMOID:    ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_SIGMOID;
                ಪಕ್ಷ GGML_UNARY_OP_NEG:        ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_NEG;
                ಪಕ್ಷ GGML_UNARY_OP_EXP:        ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_EXP;
                ಪಕ್ಷ GGML_UNARY_OP_SOFTPLUS:   ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_SOFTPLUS;
                ಪಕ್ಷ GGML_UNARY_OP_TANH:       ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_TANH;
                ಪಕ್ಷ GGML_UNARY_OP_ABS:        ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_ABS;
                ಪಕ್ಷ GGML_UNARY_OP_RELU:       ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_RELU;
                ಪಕ್ಷ GGML_UNARY_OP_STEP:       ಪ್ರತ್ಯಾಗಮ HTP_OP_UNARY_STEP;
            ಸ್ವಭಾವ:
                ಭಙ್ಗ;
            }
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_GLU:
            ವಿಕಲ್ಪ (ggml_get_glu_op(t)) {
                ಪಕ್ಷ GGML_GLU_OP_SWIGLU:       ಪ್ರತ್ಯಾಗಮ HTP_OP_GLU_SWIGLU;
                ಪಕ್ಷ GGML_GLU_OP_SWIGLU_OAI:   ಪ್ರತ್ಯಾಗಮ HTP_OP_GLU_SWIGLU_OAI;
                ಪಕ್ಷ GGML_GLU_OP_SWIGLU_CLAMP: ಪ್ರತ್ಯಾಗಮ HTP_OP_GLU_SWIGLU_CLAMP;
                ಪಕ್ಷ GGML_GLU_OP_GEGLU:        ಪ್ರತ್ಯಾಗಮ HTP_OP_GLU_GEGLU;
                ಪಕ್ಷ GGML_GLU_OP_GEGLU_QUICK:  ಪ್ರತ್ಯಾಗಮ HTP_OP_GLU_GEGLU_QUICK;
                ಪಕ್ಷ GGML_GLU_OP_GEGLU_ERF:    ಪ್ರತ್ಯಾಗಮ HTP_OP_GLU_GEGLU_ERF;
                ಸ್ವಭಾವ: ಭಙ್ಗ;
            }
            ಭಙ್ಗ;

        ಸ್ವಭಾವ:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    ಪ್ರತ್ಯಾಗಮ HTP_OP_INVALID;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ op_is_compute(ggml_tensor *ಗಂಟು)
{
    ಪ್ರತ್ಯಾಗಮ !ggml_op_is_empty(ಗಂಟು->op) && !ggml_is_empty(ಗಂಟು) && (ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ mm_is_hmx_eligible(ಧ್ರುವ ggml_tensor * t) {
    ಯದಿ (opt_nhmx == 0) { ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; }

    ಧ್ರುವ ggml_tensor * src0 = t->src[0];
    ಧ್ರುವ ggml_tensor * src1 = t->src[1];

    ಯದಿ (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        ಧ್ರುವ ggml_tensor src1_c = ggml_hexagon_tensor_collapse_rows(src1);
        ಪ್ರತ್ಯಾಗಮ ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, ಅಸತ್ಯ, ಅಸತ್ಯ);
    }

    ಧ್ರುವ ತಾರ್ಕಿಕ is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    ಧ್ರುವ ತಾರ್ಕಿಕ is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_supported_mul_mat_nx_kernel(ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ಸಂರಚನಾ htp_mm_kernel_params * kparams) {
    ಯದಿ (kparams->n_hmx) {
        ಪ್ರತ್ಯಾಗಮ kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    ಯದಿ (!ggml_hexagon_tensor_is_tiled(src0)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_supported_mul_mat_id_nx_kernel(ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ಸಂರಚನಾ htp_mm_kernel_params * kparams) {
    ಯದಿ (kparams->n_hmx) {
        ಪ್ರತ್ಯಾಗಮ kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    ಯದಿ (!ggml_hexagon_tensor_is_tiled(src0)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat(ಧ್ರುವ ggml_tensor * t) {
    ಯದಿ (t->op != GGML_OP_MUL_MAT) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    ಧ್ರುವ ggml_tensor * src0 = t->src[0];
    ಧ್ರುವ ggml_tensor * src1 = t->src[1];
    ಯದಿ (src0->ne[2] != 1 || src0->ne[3] != 1) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    ಯದಿ (mm_is_hmx_eligible(t)) {
        ಪ್ರತ್ಯಾಗಮ htp_is_hmx_weight_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    ಯದಿ (src1->type != GGML_TYPE_F32) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_tensor_is_tiled(src0);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat_pair(ಧ್ರುವ ggml_tensor * n1, ಧ್ರುವ ggml_tensor * n2) {
    ಯದಿ (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (n1->src[1] != n2->src[1]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (n1->src[0]->type != n2->src[0]->type) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat_id(ಧ್ರುವ ggml_tensor * t) {
    ಯದಿ (t->op != GGML_OP_MUL_MAT_ID) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    ಧ್ರುವ ggml_tensor * src0 = t->src[0];
    ಯದಿ (mm_is_hmx_eligible(t)) {
        ಪ್ರತ್ಯಾಗಮ htp_is_hmx_weight_type(src0->type);
    }

    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_tensor_is_tiled(src0);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_mergeable_mul_mat_id_pair(ಧ್ರುವ ggml_tensor * n1, ಧ್ರುವ ggml_tensor * n2) {
    ಯದಿ (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (n1->src[1] != n2->src[1]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (n1->src[2] != n2->src[2]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (n1->src[0]->type != n2->src[0]->type) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    ಯದಿ (sess->last_error > HTP_STATUS_OK) {
        ಪ್ರತ್ಯಾಗಮ GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    ಧ್ರುವ std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    ತಾರ್ಕಿಕ cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
    ಯದಿ (cache_hit) {
        nodes_ptr = &sess->cached_nodes;
    } ಅನ್ಯಥಾ {
        // Tag fusable tensors in graph
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; i++) {
            auto * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            ಯದಿ (!extra) ಅನುವರ್ತ;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            ಯದಿ (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } ಅನ್ಯಥಾ ಯದಿ ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                ಯದಿ ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_ಗಂಟು_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            ಯದಿ (!op_is_compute(n)) {
                ಅನುವರ್ತ;
            }

            htp_opnode ಗಂಟು(HTP_OP_INVALID, n);
            ಗಂಟು.opcode = op_remap_to_htp(n);
            ಯದಿ (ಗಂಟು.opcode == HTP_OP_MUL_MAT || ಗಂಟು.opcode == HTP_OP_MUL_MAT_ID || ಗಂಟು.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = ಗಂಟು.ಗಂಟು->src[0];
                ಯದಿ (ggml_hexagon_needs_repack(src0)) {
                    ಯದಿ (opt_verbose) {
                        ವರ್ಣ dims_buf[64];
                        ವರ್ಣ strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, ಪರಿಮಾಣ(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, ಪರಿಮಾಣ(strides_buf), src0);
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

            ಯದಿ (ಗಂಟು.opcode == HTP_OP_MUL_MAT || ಗಂಟು.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    ಗಂಟು.ಗಂಟು->src[0], ಗಂಟು.ಗಂಟು->src[1], ಗಂಟು.ಗಂಟು,
                    (ಸಂರಚನಾ htp_mm_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_MUL || ಗಂಟು.opcode == HTP_OP_ADD || ಗಂಟು.opcode == HTP_OP_ADD_ID || ಗಂಟು.opcode == HTP_OP_SUB || ಗಂಟು.opcode == HTP_OP_DIV) {
                ಧ್ರುವ ggml_tensor * src1 = ಗಂಟು.ಗಂಟು->src[1];
                ಧ್ರುವ ggml_tensor * src2 = (ಗಂಟು.opcode == HTP_OP_ADD_ID) ? ಗಂಟು.ಗಂಟು->src[2] : nullptr;
                GGML_ASSERT(ggml_hexagon_precompute_ದ್ವಿಮಾನ_params(sess,
                    ಗಂಟು.opcode, ಗಂಟು.ಗಂಟು->src[0], src1, ಗಂಟು.ಗಂಟು,
                    (ಸಂರಚನಾ htp_ದ್ವಿಮಾನ_kernel_params *) ಗಂಟು.kernel_params, src2));
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ಸಂರಚನಾ htp_fa_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಅನ್ಯಥಾ ಯದಿ (htp_op_is_unary(ಗಂಟು.opcode)) {
                auto inputs = ಗಂಟು.get_inputs();
                ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = inputs[0];
                ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    ಗಂಟು.opcode, src0, src1, ಗಂಟು.dst(),
                    (ಸಂರಚನಾ htp_unary_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_GET_ROWS) {
                ggml_hexagon_precompute_get_rows_params(sess,
                    ಗಂಟು.ಗಂಟು->src[0], ಗಂಟು.ಗಂಟು->src[1], ಗಂಟು.dst(),
                    (ಸಂರಚನಾ htp_get_rows_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_SET_ROWS) {
                ggml_hexagon_precompute_set_rows_params(sess,
                    ಗಂಟು.ಗಂಟು->src[0], ಗಂಟು.ಗಂಟು->src[1], ಗಂಟು.dst(),
                    (ಸಂರಚನಾ htp_set_rows_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ಸಂರಚನಾ htp_rope_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    ಗಂಟು.ಗಂಟು->src[0], ಗಂಟು.ಗಂಟು->src[1], ಗಂಟು.dst(),
                    (ಸಂರಚನಾ htp_ssm_conv_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ಸಂರಚನಾ htp_softmax_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ಸಂರಚನಾ htp_gdn_kernel_params *)ಗಂಟು.kernel_params
                );
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_ARGSORT || ಗಂಟು.opcode == HTP_OP_ತುದಿ_K) {
                ggml_hexagon_precompute_sort_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    ಗಂಟು.opcode == HTP_OP_ತುದಿ_K,
                    (ಸಂರಚನಾ htp_sort_kernel_params *) ಗಂಟು.kernel_params
                );
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_POOL_2D || ಗಂಟು.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, ಗಂಟು.ಗಂಟು->src[0], ಗಂಟು.dst(),
                    (ಸಂರಚನಾ htp_pool_2d_kernel_params *)ಗಂಟು.kernel_params,
                    ಗಂಟು.opcode == HTP_OP_POOL_1D);
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ಸಂರಚನಾ htp_concat_kernel_params *) ಗಂಟು.kernel_params
                );
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_CPY || ಗಂಟು.opcode == HTP_OP_CPY_FENCE) {
                ಧ್ರುವ ತಾರ್ಕಿಕ ok = ggml_hexagon_precompute_cpy_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ಸಂರಚನಾ htp_copy_kernel_params *) ಗಂಟು.kernel_params
                );
                ಧ್ರುವ auto * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_copy_kernel_params *) ಗಂಟು.kernel_params;
                ಯದಿ (ok && ಗಂಟು.opcode == HTP_OP_CPY && kparams->total_elems == 0) {
                    ಅನುವರ್ತ;
                }
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_IM2COL) {
                ggml_hexagon_precompute_im2col_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ಸಂರಚನಾ htp_im2col_kernel_params *) ಗಂಟು.kernel_params
                );
            } ಅನ್ಯಥಾ ಯದಿ (ಗಂಟು.opcode == HTP_OP_IM2COL_3D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    ಗಂಟು.ಗಂಟು,
                    (ಸಂರಚನಾ htp_im2col_kernel_params *) ಗಂಟು.kernel_params
                );
            }
            computed_nodes.push_back(std::move(ಗಂಟು));
        }

        ಯದಿ (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } ಅನ್ಯಥಾ {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    ಪ್ರತಿ (ಧ್ರುವ auto & ಗಂಟು : *nodes_ptr) {
        sess->enqueue_op(ಗಂಟು);
    }

    ಯದಿ (sess->last_error > HTP_STATUS_OK) {
        ಪ್ರತ್ಯಾಗಮ GGML_STATUS_FAILED;
    }

    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    ಯದಿ (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
}

ಗಣನಾ ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

ಸಂರಚನಾ ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

ಸಂರಚನಾ ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    ಶೂನ್ಯ reset() {
        ranges.clear();
    }

    ಶೂನ್ಯ add(ಧ್ರುವ ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    ತಾರ್ಕಿಕ check(ಧ್ರುವ ggml_hexagon_mem_range & mr) ಧ್ರುವ {
        ಪ್ರತಿ (ಧ್ರುವ auto & cmp : ranges) {
            ಯದಿ (mr.pb != cmp.pb) {
                ಅನುವರ್ತ;
            }
            ಯದಿ (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                ಅನುವರ್ತ;
            }
            ಯದಿ (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
};

ಸ್ಥಿರ ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(ಧ್ರುವ ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    ಧ್ರುವ ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    ಯದಿ (tensor->buffer) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } ಅನ್ಯಥಾ {
        mr = {
            /*.pb =*/ (uint64_t) base,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    ಪ್ರತ್ಯಾಗಮ mr;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_mem_ranges_add_ಗಂಟು(ggml_hexagon_mem_ranges & mrs, ಧ್ರುವ htp_opnode & ಗಂಟು) {
    ಯದಿ (ಗಂಟು.is_empty()) ಪ್ರತ್ಯಾಗಮ;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಯದಿ (ಗಂಟು.ಗಂಟು->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(ಗಂಟು.ಗಂಟು->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    ಪ್ರತಿ (ಧ್ರುವ auto * fused : ಗಂಟು.fused) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಯದಿ (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(ಗಂಟು.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_mem_ranges_check_ಗಂಟು(ಧ್ರುವ ggml_hexagon_mem_ranges & mrs, ಧ್ರುವ htp_opnode & ಗಂಟು) {
    ಯದಿ (ಗಂಟು.is_empty()) ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಯದಿ (ಗಂಟು.ಗಂಟು->src[i]) {
            ಯದಿ (!mrs.check(ggml_hexagon_mem_range_from_tensor(ಗಂಟು.ಗಂಟು->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
    }
    ಪ್ರತಿ (ಧ್ರುವ auto * fused : ಗಂಟು.fused) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಯದಿ (fused->src[i]) {
                ಯದಿ (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ mrs.check(ggml_hexagon_mem_range_from_tensor(ಗಂಟು.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ಸ್ಥಿರ std::vector<ಪೂರ್ಣಾಙ್ಕ> ggml_hexagon_graph_optimize_reorder(ಧ್ರುವ std::vector<htp_opnode> & nodes) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n = nodes.size();

    std::vector<ಪೂರ್ಣಾಙ್ಕ> res;
    res.reserve(n);

    std::vector<ತಾರ್ಕಿಕ> used(n, ಅಸತ್ಯ);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i0 = 0; i0 < n; i0++) {
        ಯದಿ (used[i0]) {
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ auto & node0 = nodes[i0];

        ಯದಿ (!node0.stackable()) {
            res.push_back(i0);
            used[i0] = ಸತ್ಯ;
            ಅನುವರ್ತ;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr ಪೂರ್ಣಾಙ್ಕ N_FORWARD = 16;

        std::vector<ಪೂರ್ಣಾಙ್ಕ> ರಾಶಿ;
        ರಾಶಿ.push_back(i0);

        mrs.reset();

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            ಯದಿ (used[i1]) {
                ಅನುವರ್ತ;
            }

            ಧ್ರುವ auto & node1 = nodes[i1];

            ಯದಿ (node1.stackable() && node1.same_ಒಳಹರಿವು(node0) && ggml_hexagon_mem_ranges_check_ಗಂಟು(mrs, node1)) {
                ರಾಶಿ.push_back(i1);
            } ಅನ್ಯಥಾ {
                ggml_hexagon_mem_ranges_add_ಗಂಟು(mrs, node1);
            }
        }

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx : ರಾಶಿ) {
            res.push_back(idx);
            used[idx] = ಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ res;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n = gf->n_nodes;

    constexpr ಪೂರ್ಣಾಙ್ಕ MAX_FUSE = 16;

    ಗಣನಾ ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        htp_opnode ಗಂಟು(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        ಯದಿ (ಗಂಟು.op() == GGML_OP_ADD ||
            ಗಂಟು.op() == GGML_OP_NORM ||
            ಗಂಟು.op() == GGML_OP_RMS_NORM) {
            ops[0] = ಗಂಟು.op();

            ಪೂರ್ಣಾಙ್ಕ f = i + 1;
            ಯಾವತ್ (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                ಯದಿ (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    ಭಙ್ಗ;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            ಪ್ರತಿ (; f > 1; f--) {
                ಯದಿ (ggml_can_fuse(gf, i, ops, f)) {
                    ಭಙ್ಗ;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                ಗಂಟು.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(ಗಂಟು));
    }

    ಧ್ರುವ auto order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        ಪೂರ್ಣಾಙ್ಕ j = 0;
        ಪ್ರತಿ (ಧ್ರುವ auto i : order) {
            ಧ್ರುವ auto & ಗಂಟು = nodes[i];

            gf->nodes[j++] = ಗಂಟು.ಗಂಟು;

            ಪ್ರತಿ (auto * fused : ಗಂಟು.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

ಸ್ಥಿರ uint64_t ggml_hexagon_session_ಕೀಲಿ(ಧ್ರುವ ggml_hexagon_session * sess) {
    ಪ್ರತ್ಯಾಗಮ ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    auto sbuf_dst = sess_dst->mmap_tensor(dst);

    ಯದಿ (!sess_src->clone_buffer(sbuf_dst)) { ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; }

    ಧ್ರುವ uint64_t src_ಕೀಲಿ = ggml_hexagon_session_ಕೀಲಿ(sess_src);
    auto & fence_slot = sess_dst->cpy_fence_slots[src_ಕೀಲಿ];
    ಯದಿ (!fence_slot) {
        fence_slot = (ಚಞ್ಚಲ uint32_t *) sess_dst->alloc_fence(1);
    }

    ಯದಿ (!sess_src->clone_buffer(sess_dst->fence_buf)) { ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; }

    ಯದಿ (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    ಸ್ಥಿರ ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.buffer = &sess_dst->fence_buf->backend_buffer;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.data   = (ಶೂನ್ಯ *) fence_slot;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = ಪರಿಮಾಣ(int32_t);
    fence_tensor.nb[1]  = ಪರಿಮಾಣ(int32_t);
    fence_tensor.nb[2]  = ಪರಿಮಾಣ(int32_t);
    fence_tensor.nb[3]  = ಪರಿಮಾಣ(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ ಸತ್ಯ);

    sess_dst->add_peer(sess_src);

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    auto sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    ಯದಿ (!sess_dst->clone_buffer(sbuf_src)) { ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಧ್ರುವ ggml_tensor * src, ggml_tensor * dst) {
    ಯದಿ (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    ಯದಿ (!dst->extra) {
        ggml_backend_buffer_init_tensor(dst->buffer, dst);
    }

    auto * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    ಧ್ರುವ auto * src_extra = static_cast<ಧ್ರುವ ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    ಯದಿ (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (sess_src->phys_idx != sess_dst->phys_idx)
        ಪ್ರತ್ಯಾಗಮ ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

ಸ್ಥಿರ ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = ನೂತನ ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_slot = (ಚಞ್ಚಲ uint32_t *) sess->alloc_fence(1);

    ಸ್ಥಿರ ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.buffer = &sess->fence_buf->backend_buffer;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.data   = (ಶೂನ್ಯ *) hex_event->fence_slot;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = ಪರಿಮಾಣ(int32_t);
    hex_event->fence_tensor.nb[1]  = ಪರಿಮಾಣ(int32_t);
    hex_event->fence_tensor.nb[2]  = ಪರಿಮಾಣ(int32_t);
    hex_event->fence_tensor.nb[3]  = ಪರಿಮಾಣ(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (ಶೂನ್ಯ *)hex_event, (ಶೂನ್ಯ *)hex_event->fence_slot);

    ಪ್ರತ್ಯಾಗಮ ನೂತನ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    ಯದಿ (hex_event->seq == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (ಶೂನ್ಯ *)hex_event, hex_event->seq, (ಶೂನ್ಯ *)hex_event->fence_slot);

    auto * fence = reinterpret_cast<ಧ್ರುವ ಚಞ್ಚಲ std::atomic<uint32_t> *>(hex_event->fence_slot);

    ಯದಿ ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    ಯಾವತ್ (ಸತ್ಯ) {
        ಯದಿ ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            ಯದಿ (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            ಭಙ್ಗ;
        }
        std::this_thread::yield();
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (ಶೂನ್ಯ *)hex_event);
    hex_event->fence_sess->free_fence((ಶೂನ್ಯ *) hex_event->fence_slot, 1);
    delete hex_event;
    delete event;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    ಯದಿ (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ ಅಸತ್ಯ);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (ಶೂನ್ಯ *)hex_event, hex_event->seq, (ಶೂನ್ಯ *)hex_event->fence_slot);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    ಯದಿ (hex_event->seq == 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (ಶೂನ್ಯ *)hex_event, hex_event->seq, (ಶೂನ್ಯ *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    ಯದಿ (sess->phys_idx == hex_event->sess->phys_idx) {
        ಯದಿ (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        ಪ್ರತ್ಯಾಗಮ;
    }

    sess->clone_buffer(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ ಸತ್ಯ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (ಪೂರ್ಣಾಙ್ಕ) tensor->buffer->usage : -1);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, ಶೂನ್ಯ * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (ಪೂರ್ಣಾಙ್ಕ) tensor->buffer->usage : -1);
    sess->flush_sync();
    ಯದಿ (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     ಸಂರಚನಾ ggml_tensor * tensor,
                                                     ಧ್ರುವ ಶೂನ್ಯ * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (ಪೂರ್ಣಾಙ್ಕ) tensor->buffer->usage : -1);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor,
                                                     ಶೂನ್ಯ * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (ಪೂರ್ಣಾಙ್ಕ) tensor->buffer->usage : -1);
    sess->flush_sync();
    ಯದಿ (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ ರಿಕ್ತ,
    /* .graph_plan_free         = */ ರಿಕ್ತ,
    /* .graph_plan_update       = */ ರಿಕ್ತ,
    /* .graph_plan_compute      = */ ರಿಕ್ತ,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_record,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_hexagon_guid() {
    ಸ್ಥಿರ ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    ಪ್ರತ್ಯಾಗಮ &guid;
}

ತಾರ್ಕಿಕ ggml_backend_is_hexagon(ggml_backend_t backend) {
    ಪ್ರತ್ಯಾಗಮ backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

ಸ್ಥಿರ ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, ಧ್ರುವ ವರ್ಣ * params) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ಪ್ರತ್ಯಾಗಮ ನೂತನ ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಪ್ರತ್ಯಾಗಮ dev_ctx->c_name();

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ "Hexagon";
    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * ಮುಕ್ತ, size_t * total) {
    *ಮುಕ್ತ  = opt_mbuf_total;
    *total = opt_mbuf_total;

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಗಣನಾ ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    ಪ್ರತ್ಯಾಗಮ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, ಸಂರಚನಾ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ಸತ್ಯ,
        /* .host_buffer           = */ ಅಸತ್ಯ,
        /* .buffer_from_host_ptr  = */ ಅಸತ್ಯ,
        /* .events                = */ ಸತ್ಯ,
        /* .mmap_support          = */ ಅಸತ್ಯ,
    };
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಪ್ರತ್ಯಾಗಮ &dev_ctx->buffer_type;
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    ಯದಿ (!opt_hostbuf) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಪ್ರತ್ಯಾಗಮ &dev_ctx->host_buffer_type;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_cpy(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಸಂರಚನಾ htp_copy_kernel_params kparams;
    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_cont(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಸಂರಚನಾ htp_copy_kernel_params kparams;
    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_repeat(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    GGML_UNUSED(sess);
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    // Support f32 and f16
    ಯದಿ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    // src and dst must be the same type
    ಯದಿ (src0->type != dst->type) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    // dst dims must be multiples of src dims
    ಯದಿ (dst->ne[0] % src0->ne[0] != 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (dst->ne[1] % src0->ne[1] != 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (dst->ne[2] % src0->ne[2] != 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    ಯದಿ (dst->ne[3] % src0->ne[3] != 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    // require contiguous tensors (no transposition)
    ಯದಿ (ggml_is_transposed(src0) || ggml_is_transposed(dst)) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_concat(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಸಂರಚನಾ htp_concat_kernel_params kparams;
    ಪ್ರತ್ಯಾಗಮ ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_fill(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst = op;

    ಯದಿ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    GGML_UNUSED(sess);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_hexagon_supported_roll(ಧ್ರುವ ಸಂರಚನಾ ggml_hexagon_session * sess, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    GGML_UNUSED(sess);

    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * src0 = op->src[0];
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst  = op;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_are_same_shape(src0, dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (src0->nb[0] != ggml_type_size(src0->type) || dst->nb[0] != ggml_type_size(dst->type)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    // reject ops that match the filter
    ಯದಿ (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ supp = ಅಸತ್ಯ;
    ವಿಕಲ್ಪ (op->op) {
        ಪಕ್ಷ GGML_OP_NONE:
        ಪಕ್ಷ GGML_OP_RESHAPE:
        ಪಕ್ಷ GGML_OP_VIEW:
        ಪಕ್ಷ GGML_OP_PERMUTE:
        ಪಕ್ಷ GGML_OP_TRANSPOSE:
            supp = ಸತ್ಯ;
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_MUL:
        ಪಕ್ಷ GGML_OP_ADD:
        ಪಕ್ಷ GGML_OP_SUB:
        ಪಕ್ಷ GGML_OP_DIV:
            supp = ggml_hexagon_supported_ದ್ವಿಮಾನ(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_NORM:
        ಪಕ್ಷ GGML_OP_L2_NORM:
        ಪಕ್ಷ GGML_OP_RMS_NORM:
        ಪಕ್ಷ GGML_OP_SCALE:
        ಪಕ್ಷ GGML_OP_CLAMP:
        ಪಕ್ಷ GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_SQR:
        ಪಕ್ಷ GGML_OP_SQRT:
        ಪಕ್ಷ GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_SUM:
            supp = ggml_hexagon_supported_sum(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_UNARY:
            ವಿಕಲ್ಪ (ggml_get_unary_op(op)) {
                ಪಕ್ಷ GGML_UNARY_OP_NEG:
                ಪಕ್ಷ GGML_UNARY_OP_EXP:
                ಪಕ್ಷ GGML_UNARY_OP_SIGMOID:
                ಪಕ್ಷ GGML_UNARY_OP_SOFTPLUS:
                ಪಕ್ಷ GGML_UNARY_OP_TANH:
                ಪಕ್ಷ GGML_UNARY_OP_ABS:
                ಪಕ್ಷ GGML_UNARY_OP_SILU:
                ಪಕ್ಷ GGML_UNARY_OP_GELU:
                ಪಕ್ಷ GGML_UNARY_OP_GELU_QUICK:
                ಪಕ್ಷ GGML_UNARY_OP_GELU_ERF:
                ಪಕ್ಷ GGML_UNARY_OP_RELU:
                ಪಕ್ಷ GGML_UNARY_OP_STEP:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    ಭಙ್ಗ;
                ಸ್ವಭಾವ:
                    supp = ಅಸತ್ಯ;
                    ಭಙ್ಗ;
            }
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_GLU:
            ವಿಕಲ್ಪ (ggml_get_glu_op(op)) {
                ಪಕ್ಷ GGML_GLU_OP_SWIGLU:
                ಪಕ್ಷ GGML_GLU_OP_SWIGLU_OAI:
                ಪಕ್ಷ GGML_GLU_OP_SWIGLU_CLAMP:
                ಪಕ್ಷ GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    ಭಙ್ಗ;
                ಪಕ್ಷ GGML_GLU_OP_GEGLU_QUICK:
                ಪಕ್ಷ GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    ಭಙ್ಗ;
                ಸ್ವಭಾವ:
                    supp = ಅಸತ್ಯ;
                    ಭಙ್ಗ;
            }
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_DUP:
        ಪಕ್ಷ GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_ತುದಿ_K:
            supp = ggml_hexagon_supported_ತುದಿ_k(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_IM2COL_3D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            ಭಙ್ಗ;

        ಪಕ್ಷ GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            ಭಙ್ಗ;

        ಸ್ವಭಾವ:
            ಭಙ್ಗ;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    ಪ್ರತ್ಯಾಗಮ supp;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    ತಾರ್ಕಿಕ supp = (buft == &dev_ctx->host_buffer_type) || (buft == &dev_ctx->buffer_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    ಪ್ರತ್ಯಾಗಮ supp;
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ ರಿಕ್ತ,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ ರಿಕ್ತ,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    ಪ್ರತಿ (size_t i = 0; i < opt_ndev; i++) {
        ಧ್ರುವ auto & cfg = opt_device_configs[i];
        ಯದಿ (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } ಅನ್ಯಥಾ {
            std::string peers_str;
            ಪ್ರತಿ (ಧ್ರುವ auto & p : cfg.mdev_group) {
                ಯದಿ (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = ನೂತನ ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    ಪ್ರತಿ (size_t i = 0; i < opt_ndev; i++) {
        auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        delete dev_ctx;
    }
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    ಪ್ರತ್ಯಾಗಮ "HTP";
    GGML_UNUSED(reg);
}

ಸ್ಥಿರ size_t ggml_backend_hexagon_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ಪ್ರತ್ಯಾಗಮ opt_ndev;
    GGML_UNUSED(reg);
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    ಯದಿ (index >= opt_ndev || !hreg->devices[index].context) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ಪ್ರತ್ಯಾಗಮ &hreg->devices[index];
}

// ** communication context for tensor-split allreduce

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    ಯದಿ (n_backends < 2 || n_backends > 4) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ಪ್ರತಿ (size_t i = 0; i < n_backends; ++i) {
        ಯದಿ (!ggml_backend_is_hexagon(backends[i])) {
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
    }

    ಪ್ರತಿ (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ಪ್ರತಿ (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            ಯದಿ (sess_i->phys_idx == sess_j->phys_idx) {
                ಪ್ರತ್ಯಾಗಮ nullptr;
            }
        }
    }

    auto * ctx = ನೂತನ ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    ಸ್ಥಿರ ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ಪ್ರತಿ (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (ಚಞ್ಚಲ uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].buffer = &sess_i->fence_buf->backend_buffer;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].data   = (ಶೂನ್ಯ *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = ಪರಿಮಾಣ(int32_t);
        ctx->fence_tensors[i].nb[1]  = ಪರಿಮಾಣ(int32_t);
        ctx->fence_tensors[i].nb[2]  = ಪರಿಮಾಣ(int32_t);
        ctx->fence_tensors[i].nb[3]  = ಪರಿಮಾಣ(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    ಪ್ರತ್ಯಾಗಮ ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_hexagon_comm_free(ಶೂನ್ಯ * comm_ctx_v) {
    ಯದಿ (!comm_ctx_v) ಪ್ರತ್ಯಾಗಮ;
    auto * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ಪ್ರತಿ (size_t i = 0; i < ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((ಶೂನ್ಯ *) ctx->fence_slots[i], 1);
    }
    delete ctx;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_hexagon_comm_allreduce_tensor(ಶೂನ್ಯ * comm_ctx_v, ಸಂರಚನಾ ggml_tensor ** tensors) {
    ಯದಿ (opt_ar_select == 0 || !comm_ctx_v) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ಧ್ರುವ size_t n_backends = comm_ctx->n_backends;

    ಯದಿ (n_backends < 2 || n_backends > 4) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    ಪ್ರತಿ (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        ಪ್ರತಿ (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            ಯದಿ (sess_i->phys_idx == sess_j->phys_idx) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
    }

    ಪ್ರತಿ (size_t i = 0; i < n_backends; i++) {
        ಯದಿ (!tensors[i] || !tensors[i]->buffer || !ggml_backend_buffer_is_hexagon(tensors[i]->buffer)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (tensors[i]->type != tensors[0]->type) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (!ggml_is_contiguous(tensors[i])) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಯದಿ (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತಿ (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        ಸಂರಚನಾ htp_allreduce_kernel_params kparams;
        ಯದಿ (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, ಅಸತ್ಯ, ಅಸತ್ಯ, /*is_shard_ok=*/ ಅಸತ್ಯ, &kparams)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    ಪ್ರತಿ (size_t i = 1; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        ಯದಿ ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    ಯದಿ (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_entry = max_seq;
    ಯದಿ (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    ಪ್ರತಿ (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<ಧ್ರುವ ggml_tensor *> data_tensors(n_backends);
    std::vector<ಧ್ರುವ ggml_tensor *> sync_tensors(n_backends);
    ಪ್ರತಿ (size_t i = 0; i < n_backends; i++) {
        data_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    ಪ್ರತಿ (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], data_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        ಪ್ರತಿ (size_t j = 0; j < n_backends; j++) {
            ಯದಿ (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_hexagon_split_buffer_type(ಪೂರ್ಣಾಙ್ಕ main_device, ಧ್ರುವ ಭಿನ್ನ * tensor_split) {
    GGML_UNUSED(tensor_split);
    auto reg = ggml_backend_hexagon_reg();
    auto dev = ggml_backend_reg_dev_get(reg, main_device);
    ಯದಿ (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    ಯದಿ (!dev) ಪ್ರತ್ಯಾಗಮ nullptr;
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ಪ್ರತ್ಯಾಗಮ &dev_ctx->buffer_type;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, ಧ್ರುವ ವರ್ಣ * name) {
    GGML_UNUSED(reg);
    ಯದಿ (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *) ggml_backend_hexagon_split_buffer_type;
    }
    ಯದಿ (strcmp(name, "ggml_backend_comm_init") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *) ggml_backend_hexagon_comm_init;
    }
    ಯದಿ (strcmp(name, "ggml_backend_comm_free") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *) ggml_backend_hexagon_comm_free;
    }
    ಯದಿ (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
}

template<typename T> std::vector<T> str_to_vec(ಧ್ರುವ ವರ್ಣ* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    ಯಾವತ್ (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    ಪ್ರತ್ಯಾಗಮ v;
}

template<typename T, ಪೂರ್ಣಾಙ್ಕ BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    ಪ್ರತಿ (auto i : v) { ss << i << ','; }
    auto str = ss.str(); str.pop_back(); // drop last comma
    ಪ್ರತ್ಯಾಗಮ str;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, ತಾರ್ಕಿಕ discovery_supported, ಧ್ರುವ std::unordered_map<ಪೂರ್ಣಾಙ್ಕ, fastrpc_domain> & cdsp_map) {
    ಯದಿ (discovery_supported) {
        auto it = cdsp_map.find(cfg.physical_idx);
        ಯದಿ (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } ಅನ್ಯಥಾ {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } ಅನ್ಯಥಾ {
        ವಿಕಲ್ಪ (cfg.physical_idx) {
            ಪಕ್ಷ 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                ಭಙ್ಗ;
            ಪಕ್ಷ 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                ಭಙ್ಗ;
        }
    }
    ಪ್ರತಿ (auto & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_discover_devices() {
    std::unordered_map<ಪೂರ್ಣಾಙ್ಕ, fastrpc_domain> cdsp_map;
    ತಾರ್ಕಿಕ discovery_supported = ಅಸತ್ಯ;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    ಪೂರ್ಣಾಙ್ಕ err = remote_system_request(&domain_info);
    ಯದಿ (err == AEE_SUCCESS && domain_info.sys.ಅಂಕೆ_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.ಅಂಕೆ_domains);
        domain_info.sys.domains     = domains.data();
        domain_info.sys.max_domains = (ಪೂರ್ಣಾಙ್ಕ) domains.size();

        err = remote_system_request(&domain_info);
        ಯದಿ (err == AEE_SUCCESS) {
            discovery_supported = ಸತ್ಯ;
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_domains = std::min(domain_info.sys.ಅಂಕೆ_domains, (ಪೂರ್ಣಾಙ್ಕ) domains.size());
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (ಪೂರ್ಣಾಙ್ಕ) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                ಯದಿ (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (ಪೂರ್ಣಾಙ್ಕ) domains[i].type);
                    ಅನುವರ್ತ;
                }
                ಯದಿ (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    ಅನುವರ್ತ;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } ಅನ್ಯಥಾ {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (unsigned) err);
        }
    } ಅನ್ಯಥಾ ಯದಿ (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (unsigned) err);
    }

    // Populate domain IDs and names for all configured devices
    ಪ್ರತಿ (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_Q4_0 == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_Q4_1 == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_Q8_0 == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_MXFP4 == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_IQ4_NL == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_Q4_K == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_Q5_K == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_Q6_K == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_Q3_K == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_Q2_K == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_Q1_0 == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_Q2_0 == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ಪೂರ್ಣಾಙ್ಕ) HTP_TYPE_BF16 == (unsigned ಪೂರ್ಣಾಙ್ಕ) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    ಧ್ರುವ ವರ್ಣ * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    ಧ್ರುವ ವರ್ಣ * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    ಧ್ರುವ ವರ್ಣ * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    ಧ್ರುವ ವರ್ಣ * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    ಧ್ರುವ ವರ್ಣ * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    ಧ್ರುವ ವರ್ಣ * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    ಧ್ರುವ ವರ್ಣ * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    ಧ್ರುವ ವರ್ಣ * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    ಧ್ರುವ ವರ್ಣ * str_etm      = getenv("GGML_HEXAGON_ETM");
    ಧ್ರುವ ವರ್ಣ * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    ಧ್ರುವ ವರ್ಣ * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    ಧ್ರುವ ವರ್ಣ * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    ಧ್ರುವ ವರ್ಣ * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    ಧ್ರುವ ವರ್ಣ * str_fa_head_split = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    ಧ್ರುವ ವರ್ಣ * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    ಧ್ರುವ ವರ್ಣ * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    ಧ್ರುವ ವರ್ಣ * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    ಧ್ರುವ ವರ್ಣ * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    ಧ್ರುವ ವರ್ಣ * str_arch     = getenv("GGML_HEXAGON_ARCH");
    ಧ್ರುವ ವರ್ಣ * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    ಧ್ರುವ ವರ್ಣ * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    ಧ್ರುವ ವರ್ಣ * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    ಧ್ರುವ ವರ್ಣ * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    ಧ್ರುವ ವರ್ಣ * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    ಧ್ರುವ ವರ್ಣ * str_power_mode = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    ಯದಿ (!str_arch) {
        ಪೂರ್ಣಾಙ್ಕ err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        ಯದಿ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } ಅನ್ಯಥಾ {
            ಯದಿ (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } ಅನ್ಯಥಾ ಯದಿ (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } ಅನ್ಯಥಾ {
        ಯದಿ (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, ರಿಕ್ತ, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? ನೂತನ std::regex(str_opfilter, RE_ICASE) : ರಿಕ್ತ;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, ರಿಕ್ತ, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, ರಿಕ್ತ, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, ರಿಕ್ತ, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  ರಿಕ್ತ, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    ಯದಿ (str_power_mode) {
        ಪೂರ್ಣಾಙ್ಕ m = htp_power_mode_from_str(str_power_mode);
        ಯದಿ (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_mode);
        } ಅನ್ಯಥಾ {
            opt_power_mode = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, ರಿಕ್ತ, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_head_split = str_fa_head_split ? atoi(str_fa_head_split)       : opt_fa_head_split;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    ಯದಿ (str_mbuf) {
        ಧ್ರುವ ವರ್ಣ * p = str_mbuf;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx = 0; idx < 3 && p && *p; idx++) {
            ಯಾವತ್ (*p == ' ') p++;
            ಧ್ರುವ ವರ್ಣ * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : strlen(p);
            ಯಾವತ್ (len > 0 && p[len - 1] == ' ') len--;
            ಯದಿ (len > 0) {
                std::string token(p, len);
                ಯದಿ (idx == 0) opt_mbuf_dyn    = parse_size(token.c_str());
                ಯದಿ (idx == 1) opt_mbuf_static = parse_size(token.c_str());
                ಯದಿ (idx == 2) opt_mbuf_total  = parse_size(token.c_str());
            }
            ಯದಿ (!comma) ಭಙ್ಗ;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_size(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    ಧ್ರುವ ವರ್ಣ * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    ಯದಿ (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    ಯದಿ (str_devices && str_devices[0] != '\0') {
        ತಾರ್ಕಿಕ is_single_ಸಂಖ್ಯೆ = ಸತ್ಯ;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; str_devices[i] != '\0'; i++) {
            ಯದಿ (!isdigit((unsigned ವರ್ಣ)str_devices[i])) {
                is_single_ಸಂಖ್ಯೆ = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
        ಯದಿ (is_single_ಸಂಖ್ಯೆ) {
            ಪೂರ್ಣಾಙ್ಕ n = atoi(str_devices);
            ಯದಿ (n < 1) n = 1;
            ಯದಿ (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            ಪ್ರತಿ (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (ಪೂರ್ಣಾಙ್ಕ)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.clear();
            }
        } ಅನ್ಯಥಾ {
            std::string s_devices(str_devices);
            std::vector<std::string> ವಸ್ತುಗಳು;
            std::string curr_item;
            ಪೂರ್ಣಾಙ್ಕ bracket_depth = 0;
            ಪ್ರತಿ (ವರ್ಣ ch : s_devices) {
                ಯದಿ (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } ಅನ್ಯಥಾ ಯದಿ (ch == ']') {
                    ಯದಿ (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } ಅನ್ಯಥಾ ಯದಿ (ch == ',' && bracket_depth == 0) {
                    size_t s = curr_item.find_first_not_of(" \t\r\n");
                    size_t e = curr_item.find_last_not_of(" \t\r\n");
                    ಯದಿ (s != std::string::npos) {
                        ವಸ್ತುಗಳು.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.clear();
                } ಅನ್ಯಥಾ {
                    curr_item += ch;
                }
            }
            size_t s = curr_item.find_first_not_of(" \t\r\n");
            size_t e = curr_item.find_last_not_of(" \t\r\n");
            ಯದಿ (s != std::string::npos) {
                ವಸ್ತುಗಳು.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            ಪ್ರತಿ (ಧ್ರುವ auto & item : ವಸ್ತುಗಳು) {
                size_t b_open  = item.find('[');
                size_t b_close = item.rfind(']');

                ಯದಿ (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    ಪೂರ್ಣಾಙ್ಕ virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    ಯದಿ (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        ಯತ್ನ {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } ಗೃಹಾಣ (...) {
                            virt = 0;
                        }
                    } ಅನ್ಯಥಾ {
                        size_t dev_colon = dev_name.find(':');
                        ಯದಿ (dev_colon != std::string::npos) {
                            ಯತ್ನ {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } ಗೃಹಾಣ (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<ಪೂರ್ಣಾಙ್ಕ> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    ಯಾವತ್ (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        ಯದಿ (ps == std::string::npos) ಅನುವರ್ತ;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        ಯದಿ (dash_pos != std::string::npos) {
                            ಯತ್ನ {
                                ಪೂರ್ಣಾಙ್ಕ p_start = std::stoi(p_part.substr(0, dash_pos));
                                ಪೂರ್ಣಾಙ್ಕ p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ p = p_start; p <= p_end; p++) {
                                    ಯದಿ (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } ಗೃಹಾಣ (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } ಅನ್ಯಥಾ {
                            ಯತ್ನ {
                                ಪೂರ್ಣಾಙ್ಕ p = std::stoi(p_part);
                                ಯದಿ (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } ಗೃಹಾಣ (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    ಯದಿ (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    ಯದಿ (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        auto & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.clear();

                        ಪ್ರತಿ (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_group.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } ಅನ್ಯಥಾ {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } ಅನ್ಯಥಾ ಯದಿ (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    size_t colon_pos = rest.find(':');
                    ಪೂರ್ಣಾಙ್ಕ phys = 0;
                    ಪೂರ್ಣಾಙ್ಕ virt = 0;
                    ಯತ್ನ {
                        ಯದಿ (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } ಅನ್ಯಥಾ {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } ಗೃಹಾಣ (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        ಅನುವರ್ತ;
                    }

                    ಯದಿ (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_group.clear();
                        opt_ndev++;
                    } ಅನ್ಯಥಾ {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } ಅನ್ಯಥಾ {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", item.c_str());
                }
            }
        }
    } ಅನ್ಯಥಾ {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_group.clear();
    }

#ಯದಿ defined(__ANDROID__)
    ಯದಿ (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    ಯದಿ (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            auto v  = str_to_vec<uint32_t>(str_profile);
            ವಿಕಲ್ಪ (v.size()) {
                ಪಕ್ಷ 1:  opt_profile = v[0]; ಪ್ರತ್ಯಾಗಮ opt_pmu_evt; // mode with default pmu events
                ಪಕ್ಷ 8:  opt_profile = 2;    ಪ್ರತ್ಯಾಗಮ v;           // mode with custom  pmu events
                ಸ್ವಭಾವ: opt_profile = 0;    ಪ್ರತ್ಯಾಗಮ {};          // garbage input
            }}();
        ಯದಿ (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = ನೂತನ ggml_hexagon_registry(reg);
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಅಸತ್ಯ;

    ಸ್ಥಿರ ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ ರಿಕ್ತ };

    {
        ಸ್ಥಿರ std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        ಯದಿ (!initialized) {
            auto nErr = htpdrv_init();
            ಯದಿ (nErr != AEE_SUCCESS) {
                ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = ಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
