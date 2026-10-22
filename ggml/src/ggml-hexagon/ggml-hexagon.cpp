#define _USE_MATH_DEFINES

#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <inttypes.h>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ <time.h>

#ଅନ୍ତର୍ଭାଵ <atomic>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <chrono>
#ଅନ୍ତର୍ଭାଵ <mutex>
#ଅନ୍ତର୍ଭାଵ <thread>
#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <stdexcept>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <sstream>
#ଅନ୍ତର୍ଭାଵ <iomanip>
#ଅନ୍ତର୍ଭାଵ <unordered_set>
#ଅନ୍ତର୍ଭାଵ <unordered_map>
#ଅନ୍ତର୍ଭାଵ <regex>
#ଅନ୍ତର୍ଭାଵ <queue>
#ଅନ୍ତର୍ଭାଵ <deque>
#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <initializer_list>

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
#ଅନ୍ୟଥା
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"

#ଅନ୍ତର୍ଭାଵ <AEEStdErr.h>
#ଅନ୍ତର୍ଭାଵ <dspqueue.h>
#ଅନ୍ତର୍ଭାଵ <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-alloc.h"
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "ggml-hexagon.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-quants.h"
#ଅନ୍ତର୍ଭାଵ "htp-opnode.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/matmul-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/binary-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/flash-attn-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/unary-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/get-rows-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/set-rows-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/softmax-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/pool-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/rope-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/ssm-conv.h"
#ଅନ୍ତର୍ଭାଵ "htp/gated-delta-net-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/argsort-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/concat-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/cpy-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp/im2col-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp_iface.h"
#ଅନ୍ତର୍ଭାଵ "htp-drv.h"
#ଅନ୍ତର୍ଭାଵ "htp-repack.h"

using intvec  = std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>;
using uintvec = std::vector<unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା>;
using u32vec  = std::vector<uint32_t>;

#define GGML_HEXAGON_MAX_SESSIONS          16

#define GGML_HEXAGON_FENCE_SLOT_SIZE       128

ସଂରଚନା ggml_hexagon_device_config {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା         physical_idx = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା         virtual_idx  = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

ସ୍ଥିର ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା    opt_arch    = 0; // autodetect
ସ୍ଥିର size_t opt_ndev    = 1;
ସ୍ଥିର size_t opt_nhvx    = 0; // use all
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
ସ୍ଥିର size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା    opt_etm     = 0;
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା    opt_verbose = 0;
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
ସ୍ଥିର ତାର୍କିକ   opt_hostbuf = ମିଥ୍ୟା;
ସ୍ଥିର ତାର୍କିକ   opt_dma64   = ମିଥ୍ୟା;
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start

ସ୍ଥିର ଅଚଳ std::pair<ଅଚଳ ଅକ୍ଷର *, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> htp_power_mode_names[] = {
    { "low_power_saver",            HTP_POWER_MODE_ତଳ_POWER_SAVER },
    { "power_saver",                HTP_POWER_MODE_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_MODE_ଉପର_POWER_SAVER },
    { "low_balanced",               HTP_POWER_MODE_ତଳ_BALANCED },
    { "balanced",                   HTP_POWER_MODE_BALANCED },
    { "high_performance",           HTP_POWER_MODE_ଉପର_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_MODE_SUSTAINED_ଉପର_PERFORMANCE },
    { "burst",                      HTP_POWER_MODE_BURST },
    { "default",                    HTP_POWER_MODE_BURST }, // flip this to change the default mode
};

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା htp_power_mode_from_str(ଅଚଳ ଅକ୍ଷର * s) {
    ପାଇଁ (auto & e : htp_power_mode_names) {
#ଯଦି defined(_MSC_VER)
        ଯଦି (_stricmp(s, e.first) == 0) {
#ଅନ୍ୟଥା
        ଯଦି (strcasecmp(s, e.first) == 0) {
#endif
            ଫେରନ୍ତୁ e.second;
        }
    }
    ଫେରନ୍ତୁ -1;
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * htp_power_mode_to_str(ପୂର୍ଣ୍ଣସଂଖ୍ୟା mode) {
    ପାଇଁ (auto & e : htp_power_mode_names) {
        ଯଦି (e.second == mode) {
            ଫେରନ୍ତୁ e.first;
        }
    }
    ଫେରନ୍ତୁ "unknown";
}

ସ୍ଥିର size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
ସ୍ଥିର size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
ସ୍ଥିର size_t opt_mbuf_total  = 0;                        // total buffer space limit (0 = unconstrained)

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା    opt_fa_head_split = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
ସ୍ଥିର u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା opt_opbatch  = 1280; // max number of ops in a batch
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା opt_opqueue  = 8;   // max number of pending batches
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା opt_optrace  = 0;    // trace buffer size per thread (0 means default)
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା opt_oppoll   = 0;    // polling for batch completions
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା opt_opfusion  = 1;    // enable/disable op fusion
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

ଗଣନା ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ADD = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ADD   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_hexagon_is_fusion_enabled(ପୂର୍ଣ୍ଣସଂଖ୍ୟା flag) {
    ଯଦି (opt_opfusion <= 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    ଯଦି (opt_opfusion == 1) ଫେରନ୍ତୁ ସତ୍ୟ; // 1 enables all
    ଫେରନ୍ତୁ (opt_opfusion & flag) != 0;
}

ସ୍ଥିର std::regex* opt_opfilter = ଖାଲି; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    ଯଦି (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * status_to_str(uint32_t status) {
    ବାଛ (status) {
        ସ୍ଥିତି HTP_STATUS_OK:
            ଫେରନ୍ତୁ "OK";
        ସ୍ଥିତି HTP_STATUS_NO_SUPPORT:
            ଫେରନ୍ତୁ "NO-SUPPORT";
        ସ୍ଥିତି HTP_STATUS_INVAL_PARAMS:
            ଫେରନ୍ତୁ "INVAL-PARAMS";
        ସ୍ଥିତି HTP_STATUS_VTCM_TOO_SMALL:
            ଫେରନ୍ତୁ "VTCM-TOO-SMALL";
        ସ୍ଥିତି HTP_STATUS_INTERNAL_ERR:
            ଫେରନ୍ତୁ "INTERNAL-ERROR";
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ "UNKNOWN";
    }
}

// ** debug helpers

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_dump_op_exec(ଅଚଳ std::string &sess_name, ଅଚଳ htp_opnode & ଗଣ୍ଠି, ଅଚଳ uint32_t req_flags) {
    ଯଦି (!opt_verbose) ଫେରନ୍ତୁ;

    htp_opformat fmt(ଗଣ୍ଠି);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                ଗଣ୍ଠି.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_dump_op_supp(ଅଚଳ std::string &sess_name, ଅଚଳ ସଂରଚନା ggml_tensor * op, ତାର୍କିକ supp) {
    ଯଦି (!opt_verbose) ଫେରନ୍ତୁ;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * htp_event_name(uint16_t id) {
    ବାଛ (id) {
        ସ୍ଥିତି HTP_TRACE_EVT_DMA:            ଫେରନ୍ତୁ "DMA";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_COMP:       ଫେରନ୍ତୁ "HVX_COMP";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_A_QUANT:    ଫେରନ୍ତୁ "HVX_A_QUANT";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_A_PREP:     ଫେରନ୍ତୁ "HVX_A_PREP";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_W_DEQUANT:  ଫେରନ୍ତୁ "HVX_W_DEQUANT";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_W_PREP:     ଫେରନ୍ତୁ "HVX_W_PREP";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_O_PROC:     ଫେରନ୍ତୁ "HVX_O_PROC";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_FA_QK:      ଫେରନ୍ତୁ "HVX_QK_FA";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_FA_SFM:     ଫେରନ୍ତୁ "HVX_SFM_FA";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_FA_Q_PREP:  ଫେରନ୍ତୁ "HVX_Q_PREP";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_FA_K_PREP:  ଫେରନ୍ତୁ "HVX_K_PREP";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_FA_V_PREP:  ଫେରନ୍ତୁ "HVX_V_PREP";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_GDN_PREP:   ଫେରନ୍ତୁ "HVX_GDN_PREP";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_GDN_SOLVE:  ଫେରନ୍ତୁ "HVX_GDN_SOLVE";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_GDN_V_PREP: ଫେରନ୍ତୁ "HVX_GDN_V_PREP";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_GDN_D_PREP: ଫେରନ୍ତୁ "HVX_GDN_D_PREP";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_GDN_OUT:    ଫେରନ୍ତୁ "HVX_GDN_OUT";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_GDN_STATE:  ଫେରନ୍ତୁ "HVX_GDN_STATE";
        ସ୍ଥିତି HTP_TRACE_EVT_HVX_GDN_REM:    ଫେରନ୍ତୁ "HVX_GDN_REM";
        ସ୍ଥିତି HTP_TRACE_EVT_HMX_COMP:       ଫେରନ୍ତୁ "HMX_COMP";
        ସ୍ଥିତି HTP_TRACE_EVT_L2FLUSH:        ଫେରନ୍ତୁ "L2FLUSH";
        ସ୍ଥିତି HTP_TRACE_EVT_INIT:           ଫେରନ୍ତୁ "INIT";
        ସ୍ଥିତି HTP_TRACE_EVT_BUFF:           ଫେରନ୍ତୁ "BUFF";
        ସ୍ଥିତି HTP_TRACE_EVT_FENCE:          ଫେରନ୍ତୁ "FENCE";
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:                           ଫେରନ୍ତୁ "UNKNOWN";
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_dump_op_prof(ଅଚଳ std::string &sess_name, ଅଚଳ htp_opnode & ଗଣ୍ଠି, ଅଚଳ htp_prof_desc & pd) {
    ଯଦି (!opt_profile) ଫେରନ୍ତୁ;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    ଅଚଳ uint32_t * pmu = pd.pmu;

    ଅକ୍ଷର pmu_str[256] = "";
    ଯଦି (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, ଆକାର(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(ଗଣ୍ଠି);
    ଦଶମିକ mhz = op_usec > 0 ? (ଦଶମିକ) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            ଗଣ୍ଠି.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_dump_batch_prof(ଅଚଳ std::string & sess_name, ଅଚଳ htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    ଦଶମିକ batch_mhz = rsp.usecs > 0 ? (ଦଶମିକ) batch_cycles / rsp.usecs : 0.0f;

    ଅକ୍ଷର evt_str[256] = "----";
    ଯଦି (opt_profile == 3) {
        snprintf(evt_str, ଆକାର(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (unsigned long long) batch_cycles, (unsigned long long) rsp.cycles_start, batch_mhz);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_dump_trace_events(ଅଚଳ std::string & sess_name, ଅଚଳ htp_opbatch_rsp & rsp,
                                           ଅଚଳ htp_trace_desc * trace_events, uint32_t n_traces) {
    ଯଦି (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        ପାଇଁ (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        ପାଇଁ (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ପାଇଁ (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                ଅଚଳ auto & e = trace_events[t * n_traces + idx];
                ତାର୍କିକ is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

ଗଣନା ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_WEIGHT    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

ସଂରଚନା ggml_hexagon_session;

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_matmul_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_mm_kernel_params * kparams
);

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_fused_matmul_add_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * src2,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_mm_kernel_params * kparams
);

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_unary_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    uint32_t op,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_unary_kernel_params * kparams
);

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_ଦ୍ୱିମିକ_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    uint32_t op,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_ଦ୍ୱିମିକ_kernel_params * kparams,
    ଅଚଳ ସଂରଚନା ggml_tensor * src2 = nullptr
);

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_get_rows_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_get_rows_kernel_params * kparams
);

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_set_rows_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_set_rows_kernel_params * kparams
);

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_softmax_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_softmax_kernel_params * kparams
);

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_rope_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_rope_kernel_params * kparams
);

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_ssm_conv_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_ssm_conv_kernel_params * kparams
);

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_gated_delta_net_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_gdn_kernel_params * kparams
);

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_sort_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ତାର୍କିକ is_ଶୀର୍ଷ_k,
    ସଂରଚନା htp_sort_kernel_params * kparams
);

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_pool_2d_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_pool_2d_kernel_params * kparams,
    ତାର୍କିକ is_pool_1d
);

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_concat_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_concat_kernel_params * kparams
);

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_cpy_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_copy_kernel_params * kparams
);

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_im2col_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_im2col_kernel_params * kparams
);

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_im2col_3d_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_im2col_kernel_params * kparams
);

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_fused_mmnx_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * act,
    int32_t n_weights,
    ସଂରଚନା htp_mm_kernel_params * kparams
);

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_fused_mmidnx_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * act,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    int32_t n_weights,
    ସଂରଚନା htp_mm_kernel_params * kparams
);

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_allreduce_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    ତାର୍କିକ has_add,
    ତାର୍କିକ is_row_bcast,
    ତାର୍କିକ is_shard_ok,
    ସଂରଚନା htp_allreduce_kernel_params * kparams
);

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_rows_stride(ଅଚଳ int64_t * ne, ଅଚଳ size_t * nb, size_t * stride);
ସ୍ଥିର ତାର୍କିକ ggml_hexagon_matmul_can_collapse(ଅଚଳ ସଂରଚନା ggml_tensor * src0, ଅଚଳ ସଂରଚନା ggml_tensor * src1, ଅଚଳ ସଂରଚନା ggml_tensor * dst);
ସ୍ଥିର ggml_tensor ggml_hexagon_tensor_collapse_rows(ଅଚଳ ସଂରଚନା ggml_tensor * t);

ସ୍ଥିର ତାର୍କିକ mm_is_hmx_eligible(ଅଚଳ ggml_tensor * t);
ସ୍ଥିର htp_op_code op_remap_to_htp(ଅଚଳ ggml_tensor * t);
ସ୍ଥିର ତାର୍କିକ is_supported_mul_mat_nx_kernel(ଅଚଳ ggml_tensor * src0, ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams);
ସ୍ଥିର ତାର୍କିକ is_supported_mul_mat_id_nx_kernel(ଅଚଳ ggml_tensor * src0, ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams);
ସ୍ଥିର ତାର୍କିକ is_mergeable_mul_mat(ଅଚଳ ggml_tensor * t);
ସ୍ଥିର ତାର୍କିକ is_mergeable_mul_mat_pair(ଅଚଳ ggml_tensor * n1, ଅଚଳ ggml_tensor * n2);
ସ୍ଥିର ତାର୍କିକ is_mergeable_mul_mat_id(ଅଚଳ ggml_tensor * t);
ସ୍ଥିର ତାର୍କିକ is_mergeable_mul_mat_id_pair(ଅଚଳ ggml_tensor * n1, ଅଚଳ ggml_tensor * n2);

// ** backend sessions

ସଂରଚନା ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_size { 0 };
    uint32_t             flags { 0 };
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_hexagon_tensor_is_fuseable(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଯଦି (!t->extra) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    auto extra = (ଅଚଳ ସଂରଚନା ggml_hexagon_tensor_extra *) t->extra;
    ଫେରନ୍ତୁ (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_hexagon_can_rt_repack(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଯଦି (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (t->buffer && ggml_backend_buft_is_host(t->buffer->buft)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_hexagon_tensor_is_tiled(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଅଚଳ auto * extra = (ଅଚଳ ggml_hexagon_tensor_extra *) t->extra;
    ଯଦି (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ଫେରନ୍ତୁ ggml_hexagon_can_rt_repack(t);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_hexagon_needs_repack(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଅଚଳ auto * extra = (ଅଚଳ ggml_hexagon_tensor_extra *) t->extra;
    ଯଦି (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ ggml_hexagon_can_rt_repack(t);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_hexagon_tensors_overlap(ଅଚଳ ସଂରଚନା ggml_tensor * a, ଅଚଳ ସଂରଚନା ggml_tensor * b) {
    ଅଚଳ uintptr_t a0 = (uintptr_t) a->data;
    ଅଚଳ uintptr_t b0 = (uintptr_t) b->data;
    ଅଚଳ uintptr_t a1 = a0 + ggml_nbytes(a);
    ଅଚଳ uintptr_t b1 = b0 + ggml_nbytes(b);

    ଫେରନ୍ତୁ a0 < b1 && b0 < a1;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_hexagon_can_row_partition(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଯଦି (t->ne[1] > 1 && (t->nb[1] & 127) != 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    ଯଦି (t->ne[2] > 1 && (t->nb[2] & 127) != 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    ଯଦି (t->ne[3] > 1 && (t->nb[3] & 127) != 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସଂରଚନା htp_opnode;

ସଂରଚନା ggml_hexagon_opbatch;
ସଂରଚନା ggml_hexagon_opqueue;
ସଂରଚନା ggml_hexagon_shared_buffer;
ସଂରଚନା ggml_hexagon_fence_buffer;
ସଂରଚନା ggml_hexagon_session;
ସଂରଚନା ggml_backend_hexagon_device_context;

ସଂରଚନା ggml_hexagon_mdev_group {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

ସଂରଚନା ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    ଅସ୍ଥିର uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

ସଂରଚନା ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    ଅସ୍ଥିର uint32_t *    fence_slot   = nullptr;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

ସଂରଚନା ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା              phys_idx;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା              virt_idx;
    ତାର୍କିକ             valid_session;
    ତାର୍କିକ             valid_handle;
    ତାର୍କିକ             valid_queue;
    ତାର୍କିକ             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, std::unique_ptr<ggml_hexagon_shared_buffer>> cloned_buffers;
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

    ggml_hexagon_session(ଅଚଳ ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_ଗଣନା = 0) noexcept(ମିଥ୍ୟା);
    ~ggml_hexagon_session() noexcept(ସତ୍ୟ);

    ଅଚଳ ଅକ୍ଷର* c_name() ଅଚଳ { ଫେରନ୍ତୁ name.c_str(); }

    ଶୂନ୍ୟ allocate(ଅଚଳ ggml_hexagon_device_config & config) noexcept(ମିଥ୍ୟା);
    ଶୂନ୍ୟ release() noexcept(ସତ୍ୟ);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    ଶୂନ୍ୟ      free_fence(ଶୂନ୍ୟ * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, ଅସ୍ଥିର uint32_t *> cpy_fence_slots;

    ଶୂନ୍ୟ enqueue_mdev_group();
    ଶୂନ୍ୟ enqueue_op(ଅଚଳ htp_opnode & ଗଣ୍ଠି);
    ଶୂନ୍ୟ enqueue_cpy(ଅଚଳ ggml_tensor * src, ggml_tensor * dst, ଅଚଳ ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    ଶୂନ୍ୟ enqueue_fence(ଅଚଳ ggml_tensor * sync_tensor, uint32_t fence_seq = 0, ତାର୍କିକ wait = ସତ୍ୟ);
    ଶୂନ୍ୟ enqueue_allreduce(ଅଚଳ ggml_tensor * dst, ଅଚଳ std::vector<ଅଚଳ ggml_tensor *> & src_tensors,
                           ଅଚଳ std::vector<ଅଚଳ ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);

    ଶୂନ୍ୟ start_batch();
    ଶୂନ୍ୟ flush_sync(ତାର୍କିକ all = ସତ୍ୟ);
    ଶୂନ୍ୟ flush_async();
    ଶୂନ୍ୟ flush_batch(size_t min_ops = 1);
    ଶୂନ୍ୟ flush_peers();
    ଶୂନ୍ୟ flush_pending(ତାର୍କିକ all = ସତ୍ୟ);

    ggml_hexagon_shared_buffer * mmap_tensor(ଅଚଳ ggml_tensor * t);
    ତାର୍କିକ clone_buffer(ଅଚଳ ggml_hexagon_shared_buffer*);
    ଶୂନ୍ୟ release_buffer(ଅଚଳ ggml_hexagon_shared_buffer*);
    ଶୂନ୍ୟ unclone_buffer(ଅଚଳ ggml_hexagon_shared_buffer*);

    ଶୂନ୍ୟ add_peer(ggml_hexagon_session * peer) {
        ଯଦି (ଏହି->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } ଅନ୍ୟଥା {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

ସଂରଚନା ggml_backend_hexagon_device_context {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_buffer_type buffer_type       = {};
    ggml_backend_buffer_type host_buffer_type  = {};
    ggml_backend_buffer_type fence_buffer_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(ପୂର୍ଣ୍ଣସଂଖ୍ୟା dev_id, ଅଚଳ ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    ଅଚଳ ଅକ୍ଷର * c_name() ଅଚଳ { ଫେରନ୍ତୁ config.name.c_str(); }

    ggml_hexagon_session * session() {
        ଯଦି (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        ଫେରନ୍ତୁ sess.get();
    }
};

ସଂରଚନା ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(ଅଚଳ std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        ଏହି->dev_ctx = dev_ctx;
        ଏହି->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

ସଂରଚନା ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        ଯଦି (!base) {
            ଫିଙ୍ଗ std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        ଯଦି (fd < 0) {
            rpcmem_free(base);
            ଫିଙ୍ଗ std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        ଏହି->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        ଯଦି (base) {
            rpcmem_free(base);
        }
    }
};

ସଂରଚନା ggml_hexagon_shared_buffer {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    ତାର୍କିକ     mapped;
    ତାର୍କିକ     pinned;
    ତାର୍କିକ     extended;

    ଅଚଳ ଅକ୍ଷର * c_name() ଅଚଳ { ଫେରନ୍ତୁ sess->c_name(); }
    uint8_t *    base()   ଅଚଳ { ଫେରନ୍ତୁ mem ? mem->base : nullptr; }
    size_t       size()   ଅଚଳ { ଫେରନ୍ତୁ mem ? mem->size : 0;  }
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା          fd()     ଅଚଳ { ଫେରନ୍ତୁ mem ? mem->fd   : -1; }

    ଶୂନ୍ୟ mmap(ତାର୍କିକ extended = ମିଥ୍ୟା) {
        ଯଦି (!ଏହି->mem)   ଫେରନ୍ତୁ;
        ଯଦି (ଏହି->mapped) ଫେରନ୍ତୁ;

        GGML_ASSERT(!ଏହି->pinned || !extended);

        ଏହି->extended = extended;
        fastrpc_map_flags flags = ଏହି->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = fastrpc_mmap(sess->domain_id, fd(), (ଶୂନ୍ୟ *) base(), 0, size(), flags);
        ଯଦି (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (unsigned) err);
            ଫିଙ୍ଗ std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (ଶୂନ୍ୟ *) base(), size(), fd(), pinned, extended);

        ଏହି->mapped = ସତ୍ୟ;
    }

    ଶୂନ୍ୟ unmap() {
        ଯଦି (!ଏହି->mapped) ଫେରନ୍ତୁ;

        ଯଦି (!ଏହି->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        ଯଦି (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (ଶୂନ୍ୟ *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (ଶୂନ୍ୟ *) base(), size(), fd());

        ଏହି->mapped = ମିଥ୍ୟା;
    }

    ଶୂନ୍ୟ alloc(size_t size) {
        ଯଦି (ଏହି->mem) ଫେରନ୍ତୁ;

        ଏହି->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (ଶୂନ୍ୟ *) base(), ଏହି->size(), fd(), (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) pinned);
        ଯଦି (ଏହି->pinned) {
            mmap();
        }
    }

    ଶୂନ୍ୟ ମୁକ୍ତ() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (ଶୂନ୍ୟ *) base(), size(), fd());
        ଏହି->mem  = nullptr;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, ତାର୍କିକ pinned = ମିଥ୍ୟା) {
        ଏହି->sess     = sess;
        ଏହି->mapped   = ମିଥ୍ୟା;
        ଏହି->pinned   = pinned;
        ଏହି->extended = ମିଥ୍ୟା;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_offset = (size + 4095) & ~4095;
        size_t total_size   = guard_offset + 4096;
        ଯଦି (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            total_size = (total_size + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(total_size);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, ଅଚଳ ggml_hexagon_shared_buffer & other) {
        ଏହି->sess   = sess;
        ଏହି->mem    = other.mem;
        ଏହି->mapped = ମିଥ୍ୟା;
        ଏହି->pinned = other.pinned;
        ଏହି->extended = other.extended;
    }

    ~ggml_hexagon_shared_buffer() {
        ମୁକ୍ତ();
        ପାଇଁ (auto * extra : tensor_extra) {
            delete extra;
        }
    }
};

ସଂରଚନା ggml_hexagon_fence_buffer : ସର୍ବସାଧାରଣ ggml_hexagon_shared_buffer {
    uint32_t              slot_ଗଣନା = 0;
    uint32_t              slot_head  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_buffer   backend_buffer{};

    ggml_hexagon_fence_buffer(ggml_hexagon_session * sess, ggml_backend_buffer_type_t buft, size_t size)
        : ggml_hexagon_shared_buffer(sess, size, ମିଥ୍ୟା /* pinned */),
          slot_ଗଣନା(size / GGML_HEXAGON_FENCE_SLOT_SIZE),
          slot_head(0) {
        backend_buffer.buft    = buft;
        backend_buffer.context = static_cast<ggml_hexagon_shared_buffer *>(ଏହି);
        backend_buffer.size    = size;
        mmap(ମିଥ୍ୟା);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        ଯଦି (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        } ଅନ୍ୟଥା ଯଦି (slot_head + n_slots <= slot_ଗଣନା) {
            uint32_t slot = slot_head;
            slot_head += n_slots;
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        }
        ଯଦି (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_SLOT_SIZE);
        }
        ଫେରନ୍ତୁ ptr;
    }

    ଶୂନ୍ୟ free_slot(ଶୂନ୍ୟ * ptr, uint32_t n_slots = 1) {
        ଯଦି (!ptr) ଫେରନ୍ତୁ;
        uint32_t slot = ((uint8_t *) ptr - base()) / GGML_HEXAGON_FENCE_SLOT_SIZE;
        ପାଇଁ (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

ଧାଡ଼ିଭିତରେ uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    ଫେରନ୍ତୁ ptr;
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_hexagon_session::free_fence(ଶୂନ୍ୟ * ptr, uint32_t n_slots) {
    ଯଦି (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    sbuf->sess->unclone_buffer(sbuf);
    delete sbuf;
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    ଫେରନ୍ତୁ sbuf->base();
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (ଶୂନ୍ୟ *) sbuf->base(), tensor->data, ggml_nbytes(tensor));

    auto extra = ନୂତନ ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   ଅଚଳ ଶୂନ୍ୟ *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    ଯଦି (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        ଯଦି (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, data, offset, size, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) buffer->usage, extra->flags);

    ଯଦି ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy((ଅକ୍ଷର *) tensor->data + offset, data, size);
        ଫେରନ୍ତୁ;
    }

    ଯଦି (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        htp_repack_tensor_tiled(tensor, data, size);
        ଫେରନ୍ତୁ;
    }

    ଯଦି (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    memcpy(extra->shadow_buf.data() + offset, data, size);
    extra->shadow_size += size;

    ଯଦି (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   ଅଚଳ ggml_tensor *   tensor,
                                                   ଶୂନ୍ୟ *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, data, offset, size, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) buffer->usage, extra->flags);

    ଯଦି ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy(data, (ଅଚଳ ଅକ୍ଷର *) tensor->data + offset, size);
        ଫେରନ୍ତୁ;
    }

    htp_unpack_tensor_tiled(data, tensor, offset, size);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   ଅଚଳ ସଂରଚନା ggml_tensor * src,
                                                   ସଂରଚନା ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    ଫେରନ୍ତୁ ମିଥ୍ୟା;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                          ggml_tensor *         tensor,
                                                          ଅଚଳ ଶୂନ୍ୟ *          data,
                                                          size_t                offset,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    ଯଦି (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        ଯଦି (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) buffer->usage, extra->flags);

    ଯଦି ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        ପାଇଁ (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (ଅଚଳ uint8_t *) data + i * stride_data, size);
        }
        ଫେରନ୍ତୁ;
    }

    ଯଦି (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    ପାଇଁ (size_t i = 0; i < n_copies; i++) {
        memcpy(extra->shadow_buf.data() + offset + i * stride_tensor, (ଅଚଳ uint8_t *) data + i * stride_data, size);
    }
    extra->shadow_size += n_copies * size;

    ଯଦି (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                      ଅଚଳ ggml_tensor *   tensor,
                                                      ଶୂନ୍ୟ *                data,
                                                      size_t                offset,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) buffer->usage);

    ଯଦି ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        ପାଇଁ (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *)data + i * stride_data, (ଅଚଳ uint8_t *)tensor->data + offset + i * stride_tensor, size);
        }
        ଫେରନ୍ତୁ;
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

    ପାଇଁ (size_t i = 0; i < n_copies; i++) {
        memcpy((uint8_t *) data + i * stride_data, temp_buf.data() + i * stride_tensor, size);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (ଶୂନ୍ୟ *) sbuf->base(), sbuf->size());
    memset(sbuf->base(), value, sbuf->size());
}

ସ୍ଥିର ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ ଖାଲି,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ ଖାଲି,
};

// ** backend buffer type

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_host_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                        ggml_tensor *         tensor,
                                                        ଅଚଳ ଶୂନ୍ୟ *          data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy((ଅକ୍ଷର *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                        ଅଚଳ ggml_tensor *   tensor,
                                                        ଶୂନ୍ୟ *                data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy(data, (ଅଚଳ ଅକ୍ଷର *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

ସ୍ଥିର ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ ଖାଲି,
    /* .set_tensor      = */ ggml_backend_hexagon_host_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_buffer_get_tensor,
    /* .set_tensor_2d   = */ ଖାଲି,
    /* .get_tensor_2d   = */ ଖାଲି,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ ଖାଲି,
};

// ** backend buffer type

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    ଫେରନ୍ତୁ static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    ଯଦି (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        ଫେରନ୍ତୁ nullptr;
    }
    ଚେଷ୍ଟା {
        ggml_hexagon_shared_buffer * sbuf = ନୂତନ ggml_hexagon_shared_buffer(sess, size, ମିଥ୍ୟା);
        ଫେରନ୍ତୁ ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } ଧର (ଅଚଳ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        ଫେରନ୍ତୁ nullptr;
    }
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    ଯଦି (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        ଫେରନ୍ତୁ nullptr;
    }
    ଚେଷ୍ଟା {
        ggml_hexagon_shared_buffer * sbuf = ନୂତନ ggml_hexagon_shared_buffer(sess, size, ମିଥ୍ୟା);
        ଫେରନ୍ତୁ ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_host_buffer_interface, sbuf, size);
    } ଧର (ଅଚଳ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        ଫେରନ୍ତୁ nullptr;
    }
}

ସ୍ଥିର size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ 128;  // HVX alignment
    GGML_UNUSED(buft);
}

ସ୍ଥିର size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଯଦି (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_size = htp_tiled_row_size(t->type, ne0) * ne1 * ne2 * ne3;
        ଫେରନ୍ତୁ std::max(tiled_size, ggml_nbytes(t));
    }
    ଫେରନ୍ତୁ ggml_nbytes(t);
}

ସ୍ଥିର size_t parse_size(ଅଚଳ ଅକ୍ଷର * str, size_t default_unit = 1024 * 1024) {
    ଯଦି (!str || str[0] == '\0') {
        ଫେରନ୍ତୁ 0;
    }
    ଅକ୍ଷର * end = ଖାଲି;
    ଦ୍ୱିଗୁଣ val = strtod(str, &end);
    ଯଦି (val < 0) {
        ଫେରନ୍ତୁ 0;
    }
    ଯଦି (end && *end) {
        ଯେତେବେଳେ (*end == ' ') end++;
        ଯଦି (*end == 'k' || *end == 'K') {
            ଫେରନ୍ତୁ (size_t) (val * 1024);
        } ଅନ୍ୟଥା ଯଦି (*end == 'm' || *end == 'M') {
            ଫେରନ୍ତୁ (size_t) (val * 1024 * 1024);
        } ଅନ୍ୟଥା ଯଦି (*end == 'g' || *end == 'G') {
            ଫେରନ୍ତୁ (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    ଫେରନ୍ତୁ (size_t) (val * default_unit);
}

ସ୍ଥିର size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ opt_mbuf_dyn;
    GGML_UNUSED(buft);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
    GGML_UNUSED(buft);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ ସତ୍ୟ;
    GGML_UNUSED(buft);
}

ସଂରଚନା ggml_backend_hexagon_alloc_buffer_n_plan_item {
    size_t size;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା    first;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା    last;
};

using ggml_backend_hexagon_alloc_buffer_n_plan_t = std::vector<ggml_backend_hexagon_alloc_buffer_n_plan_item>;

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_hexagon_kv_layer_suffix(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଯଦି (strncmp(t->name, "cache_", 6) != 0) {
        ଫେରନ୍ତୁ ଖାଲି;
    }
    ଅଚଳ ଅକ୍ଷର * p = strstr(t->name, "_l");
    ଯଦି (!p || !isdigit((unsigned ଅକ୍ଷର)p[2])) {
        ଫେରନ୍ତୁ ଖାଲି;
    }
    ଫେରନ୍ତୁ p;
}

ସଂରଚନା ggml_backend_hexagon_alloc_unit {
    size_t size;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା    first;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା    last;
};

ସ୍ଥିର ggml_backend_hexagon_alloc_buffer_n_plan_t ggml_backend_hexagon_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, ସଂରଚନା ggml_tensor ** tensors, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tensors) {
    ggml_backend_hexagon_alloc_buffer_n_plan_t plan;

    ଅଚଳ size_t alignment = ggml_backend_buft_get_alignment(buft);
    ଅଚଳ size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0;
    ଯେତେବେଳେ (i < n_tensors) {
        ସଂରଚନା ggml_tensor * t = tensors[i];
        size_t unit_size = 0;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା unit_first = i;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା unit_last = i + 1;

        ଯଦି (t->data == ଖାଲି && t->view_src == ଖାଲି) {
            unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        ଅଚଳ ଅକ୍ଷର * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        ଯେତେବେଳେ (unit_last < n_tensors) {
            ସଂରଚନା ggml_tensor * next = tensors[unit_last];

            ଯଦି (next->view_src != ଖାଲି) {
                unit_last++;
                ଜାରି;
            }

            ଯଦି (layer_suffix != ଖାଲି) {
                ଅଚଳ ଅକ୍ଷର * next_suffix = ggml_hexagon_kv_layer_suffix(next);
                ଯଦି (next_suffix != ଖାଲି && strcmp(layer_suffix, next_suffix) == 0) {
                    ଯଦି (next->data == ଖାଲି) {
                        unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, next), alignment);
                    }
                    unit_last++;
                    ଜାରି;
                }
            }

            ଭାଙ୍ଗ;
        }

        units.push_back({ unit_size, unit_first, unit_last });
        i = unit_last;
    }

    size_t cur_buf_size  = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା    cur_buf_first = 0;

    ପାଇଁ (ଅଚଳ auto & unit : units) {
        ଯଦି (unit.size == 0) {
            ଜାରି;
        }

        ଯଦି (cur_buf_size > 0 && (cur_buf_size + unit.size) > max_size) {
            plan.push_back({ cur_buf_size, cur_buf_first, unit.first });
            cur_buf_size  = 0;
            cur_buf_first = unit.first;
        }

        cur_buf_size += unit.size;
    }

    ଯଦି (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, cur_buf_first, n_tensors });
    }

    ଫେରନ୍ତୁ plan;
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, ସଂରଚନା ggml_tensor ** tensors, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tensors) {
    ଅଚଳ ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    ପାଇଁ (ଅଚଳ auto & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        ଯଦି (buffer == ଖାଲି) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            ପାଇଁ (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ଫେରନ୍ତୁ ଖାଲି;
        }

        ସଂରଚନା ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        ସଂରଚନା ggml_tensor * t_failed = ଖାଲି;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = item.first; j < item.last; j++) {
            ସଂରଚନା ggml_tensor * t = tensors[j];
            ଯଦି (t->data == ଖାଲି) {
                ଯଦି (t->view_src == ଖାଲି) {
                    ଯଦି (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ଭାଙ୍ଗ;
                    }
                } ଅନ୍ୟଥା ଯଦି (t->buffer == ଖାଲି) {
                    ଯଦି (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ଭାଙ୍ଗ;
                    }
                }
            } ଅନ୍ୟଥା {
                ଯଦି (t->view_src != ଖାଲି && t->buffer == ଖାଲି) {
                    ଯଦି (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        ଭାଙ୍ଗ;
                    }
                }
            }
        }
        ଯଦି (t_failed != ଖାଲି) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            ପାଇଁ (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            ଫେରନ୍ତୁ ଖାଲି;
        }

        buffers.push_back(buffer);
    }

    ଯଦି (buffers.empty()) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଯଦି (buffers.size() == 1) {
        ଫେରନ୍ତୁ buffers[0];
    }

    ଫେରନ୍ତୁ ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

ସ୍ଥିର size_t ggml_backend_hexagon_buffer_type_get_alloc_size_n(
        ggml_backend_buffer_type_t buft, ସଂରଚନା ggml_tensor ** tensors, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tensors) {
    ଅଚଳ ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    ପାଇଁ (ଅଚଳ auto & item : plan) {
        total += item.size;
    }
    ଫେରନ୍ତୁ total;
}

ସ୍ଥିର ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_buffer_type_is_host,
};

ସ୍ଥିର ggml_backend_buffer_type_i ggml_backend_hexagon_host_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_host_buffer_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(ପୂର୍ଣ୍ଣସଂଖ୍ୟା dev_id, ଅଚଳ ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    buffer_type.device  = dev;
    buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    buffer_type.context = ନୂତନ ggml_backend_hexagon_buffer_type_context(config.name, ଏହି);

    host_buffer_type.device  = dev;
    host_buffer_type.iface   = ggml_backend_hexagon_host_buffer_type_interface;
    host_buffer_type.context = ନୂତନ ggml_backend_hexagon_buffer_type_context(config.name + "-HOST", ଏହି);

    fence_buffer_type.device  = dev;
    fence_buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    fence_buffer_type.context = ନୂତନ ggml_backend_hexagon_buffer_type_context(config.name + "-FENCE", ଏହି);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(host_buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(fence_buffer_type.context);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_buffer_is_hexagon(ଅଚଳ ସଂରଚନା ggml_backend_buffer * b) {
    ଫେରନ୍ତୁ b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

ସଂରଚନା ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା>                b_map; // buffer fd   to index
    std::unordered_map<ଅଚଳ ggml_tensor*, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> t_map; // tensor ptr  to index
    std::unordered_multimap<ଶୂନ୍ୟ*, ପୂର୍ଣ୍ଣସଂଖ୍ୟା>         d_map; // tensor data to index

    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_bufs;     // num buffers in the batch
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tens;     // num tensors ...
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_bufs_max;
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tens_max;
    unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ops_max;
    size_t       b_vmem_max;

    ଶୂନ୍ୟ reset() {
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
        ଏହି->sess = sess;

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

    ତାର୍କିକ empty() ଅଚଳ { ଫେରନ୍ତୁ n_ops == 0; }

    // add buffer and return its index
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        auto it = b_map.find(sbuf->fd());
        ଯଦି (it != b_map.end()) { ଫେରନ୍ତୁ it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        ଯଦି (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (ଶୂନ୍ୟ*) sbuf->base(), (size_t) b.size, b_vmem);

        ଫେରନ୍ତୁ bi;
    }

    ତାର୍କିକ same_shape(ଅଚଳ htp_tensor * h, ଅଚଳ ggml_tensor * t) ଅଚଳ {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;

        ଅଚଳ ତାର୍କିକ is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ଅଚଳ int64_t ne0_padded = hex_round_up(t->ne[0], 32);
        ଅଚଳ int64_t ne1_padded = hex_round_up(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_row_size(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? ଆକାର(ggml_fp16_t) : t->nb[0];
        ଗଣନା ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        ଫେରନ୍ତୁ (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା add_tensor(ଅଚଳ ggml_tensor * t) {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;
        auto sbuf  = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        ପାଇଁ (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            ଯଦି (same_shape(h, t)) { ଫେରନ୍ତୁ it->second; }
        }

        // Lookup by tensor ptr
        auto it = t_map.find(t);
        ଯଦି (it != t_map.end()) { ଫେରନ୍ତୁ it->second; }

        // Add new tensor to the batch
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ti = n_tens++;
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

        ଅଚଳ ତାର୍କିକ is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ଯଦି (is_tiled) {
            ଯଦି (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = ଆକାର(ggml_fp16_t);
            } ଅନ୍ୟଥା {
                h.nb[0] = t->nb[0];
            }
            ଅଚଳ int64_t ne0_padded = hex_round_up(t->ne[0], 32);
            ଅଚଳ int64_t ne1_padded = hex_round_up(t->ne[1], 32);
            h.nb[1] = htp_tiled_row_size(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_size  = h.size;
        } ଅନ୍ୟଥା {
            h.size  = t_size;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        ଯଦି ((extra->flags & GGML_HEXAGON_TENSOR_WEIGHT) != 0) {
            h.flags |= HTP_TENSOR_WEIGHT;
        }
        ଯଦି ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        ଯଦି ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (ଶୂନ୍ୟ*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        ଫେରନ୍ତୁ ti;
    }

    ତାର୍କିକ fit_op(ଅଚଳ htp_opnode & ଗଣ୍ଠି) ଅଚଳ {
        ଯଦି (n_ops >= n_ops_max) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା seen_bufs[HTP_OP_MAX_BUFS];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_seen_bufs = 0;

        auto fit_tensor = [&](ଅଚଳ ggml_tensor *t) {
            ଯଦି (!t) ଫେରନ୍ତୁ;
            ଯଦି (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd = sbuf->fd();
                ଯଦି (!b_map.count(fd)) {
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_seen_bufs; i++) {
                        ଯଦି (seen_bufs[i] == fd) ଫେରନ୍ତୁ;
                    }
                    ଯଦି (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    ଯଦି (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        ପାଇଁ (ଅଚଳ auto * src : ଗଣ୍ଠି.get_inputs()) {
            fit_tensor(src);
        }
        ପାଇଁ (ଅଚଳ auto * output : ଗଣ୍ଠି.get_outputs()) {
            fit_tensor(output);
        }

        ଯଦି ((extra_bufs + n_bufs) > n_bufs_max) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି ((extra_tens + n_tens) > n_tens_max) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି ((extra_vmem + b_vmem) > b_vmem_max) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    // assumes that fit_op() was called first and returned true
    ଶୂନ୍ୟ add_op(ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        // Add new op

        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = ଗଣ୍ଠି;

        htp_op_desc &o = h_ops[n];
        memcpy(o.params,        ଗଣ୍ଠି.ଗଣ୍ଠି->op_params, ଆକାର(ଗଣ୍ଠି.ଗଣ୍ଠି->op_params));
        memcpy(o.kernel_params, ଗଣ୍ଠି.kernel_params,   ଆକାର(o.kernel_params));
        o.opcode = ଗଣ୍ଠି.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        auto inputs = ଗଣ୍ଠି.get_inputs();
        ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        auto outputs = ଗଣ୍ଠି.get_outputs();
        ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    ଶୂନ୍ୟ sort_buffers() {
        ଯଦି (n_bufs <= 1) ଫେରନ୍ତୁ;

        std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> order(n_bufs);
        ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_bufs; i++) { order[i] = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) i; }

        std::stable_sort(order.begin(), order.end(), [&](ପୂର୍ଣ୍ଣସଂଖ୍ୟା a, ପୂର୍ଣ୍ଣସଂଖ୍ୟା b) {
            ଫେରନ୍ତୁ h_bufs[a].size > h_bufs[b].size;
        });

        ତାର୍କିକ already_sorted = ସତ୍ୟ;
        ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_bufs; i++) {
            ଯଦି (order[i] != (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) i) {
                already_sorted = ମିଥ୍ୟା;
                ଭାଙ୍ଗ;
            }
        }
        ଯଦି (already_sorted) ଫେରନ୍ତୁ;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା new_bi = 0; new_bi < n_bufs; new_bi++) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    ଶୂନ୍ୟ update_mdev_group(uint32_t mdev_idx) {
        ଯଦି (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    ତାର୍କିକ try_fuse_common(std::initializer_list<ଅଚଳ ggml_tensor *> tensors) ଅଚଳ {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା seen_bufs[HTP_OP_MAX_BUFS];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_seen_bufs = 0;

        ପାଇଁ (ଅଚଳ auto * t : tensors) {
            ଯଦି (!t || t_map.count(t)) {
                ଜାରି;
            }
            extra_tens++;
            auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd = sbuf->fd();
            ଯଦି (!b_map.count(fd)) {
                ତାର୍କିକ found = ମିଥ୍ୟା;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_seen_bufs; i++) {
                    ଯଦି (seen_bufs[i] == fd) {
                        found = ସତ୍ୟ;
                        ଭାଙ୍ଗ;
                    }
                }
                ଯଦି (!found) {
                    ଯଦି (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    ଯଦି (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        ଯଦି ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ତାର୍କିକ try_fuse_common(ଅଚଳ ggml_tensor * t1, ଅଚଳ ggml_tensor * t2) ଅଚଳ {
        ଫେରନ୍ତୁ try_fuse_common({t1, t2});
    }

    ତାର୍କିକ try_fuse_allreduce_add(ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଯଦି (n_ops == 0 || opt_ar_select != 2) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି (ଗଣ୍ଠି.opcode != HTP_OP_ADD) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        htp_opnode & last_ଗଣ୍ଠି = ops[n_ops - 1];
        ଯଦି (last_ଗଣ୍ଠି.opcode != HTP_OP_ALLREDUCE) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        auto * ar_kparams = (ସଂରଚନା htp_allreduce_kernel_params *) last_ଗଣ୍ଠି.kernel_params;
        ଅଚଳ uint32_t rank    = (uint32_t) ar_kparams->rank;
        ଅଚଳ uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        ଅଚଳ ggml_tensor * ar_local = last_ଗଣ୍ଠି.inputs[rank];
        ଅଚଳ ggml_tensor * add_src0 = ଗଣ୍ଠି.src0();
        ଅଚଳ ggml_tensor * add_src1 = ଗଣ୍ଠି.src1();
        ଅଚଳ ggml_tensor * add_dst  = ଗଣ୍ଠି.dst();

        ଯଦି (!ggml_hexagon_tensor_is_fuseable(ar_local)) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଅଚଳ ggml_tensor * res_tensor;
        ଯଦି (add_src0 == ar_local || add_src0->data == ar_local->data) {
            res_tensor = add_src1;
        } ଅନ୍ୟଥା ଯଦି (add_src1 == ar_local || add_src1->data == ar_local->data) {
            res_tensor = add_src0;
        } ଅନ୍ୟଥା {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (ar_local->type != res_tensor->type) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଅଚଳ ତାର୍କିକ is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        ଅଚଳ ତାର୍କିକ is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        ଯଦି (!is_same_shape && !is_row_bcast) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଯଦି (is_same_shape) {
            ଯଦି (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }
        ଯଦି (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // scatter is only valid for in-place add within max outputs
        ଅଚଳ ତାର୍କିକ is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

        ସଂରଚନା htp_allreduce_kernel_params new_kparams;
        ଯଦି (!ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, ସତ୍ୟ, is_row_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ତାର୍କିକ scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

        ପାଇଁ (uint32_t r = 0; r < n_ranks; r++) {
            ଅଚଳ ggml_tensor * ar_src = last_ଗଣ୍ଠି.inputs[r];
            ଯଦି (!ggml_hexagon_tensors_overlap(add_dst, ar_src)) {
                ଜାରି;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            ଯଦି (scatter_ok && r == rank) {
                ଜାରି;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (!try_fuse_common(res_tensor, add_dst)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        last_ଗଣ୍ଠି.opcode = HTP_OP_ALLREDUCE_ADD;
        last_ଗଣ୍ଠି.name   = "ALLREDUCE+ADD";
        last_ଗଣ୍ଠି.inputs.push_back(res_tensor);
        last_ଗଣ୍ଠି.outputs.clear();
        last_ଗଣ୍ଠି.outputs.push_back(add_dst);
        last_ଗଣ୍ଠି.fused.push_back(ଗଣ୍ଠି.ଗଣ୍ଠି);
        memcpy(last_ଗଣ୍ଠି.kernel_params, &new_kparams, ଆକାର(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ADD;
        memcpy(o.kernel_params, &new_kparams, ଆକାର(new_kparams));

        o.src[2 * n_ranks] = add_tensor(res_tensor);
        ଯଦି (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            ପାଇଁ (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            ପାଇଁ (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } ଅନ୍ୟଥା {
            o.dst[0] = add_tensor(add_dst);
            ପାଇଁ (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) new_kparams.mode, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) new_kparams.n_dsts);
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ତାର୍କିକ try_fuse_rms_norm_mul(ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଯଦି (n_ops == 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି (ଗଣ୍ଠି.opcode != HTP_OP_MUL) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        htp_opnode & last_ଗଣ୍ଠି = ops[n_ops - 1];
        ଯଦି (last_ଗଣ୍ଠି.opcode != HTP_OP_RMS_NORM) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଅଚଳ ggml_tensor * mul_src0 = ଗଣ୍ଠି.src0();
        ଅଚଳ ggml_tensor * mul_src1 = ଗଣ୍ଠି.src1();
        ଅଚଳ ggml_tensor * rms_out  = last_ଗଣ୍ଠି.dst();

        ଯଦି (!ggml_hexagon_tensor_is_fuseable(rms_out)) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଅଚଳ ggml_tensor * weight;
        ଯଦି (mul_src0 == rms_out || mul_src0->data == rms_out->data) {
            weight = mul_src1;
        } ଅନ୍ୟଥା ଯଦି (mul_src1 == rms_out || mul_src1->data == rms_out->data) {
            weight = mul_src0;
        } ଅନ୍ୟଥା {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ggml_tensor * src0 = last_ଗଣ୍ଠି.src0();

        ଯଦି (src0->ne[0] != weight->ne[0] || src0->ne[0] != ଗଣ୍ଠି.dst()->ne[0]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ତାର୍କିକ is_row_bcast = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
        ଅଚଳ ତାର୍କିକ is_same_shape = (src0->ne[0] == weight->ne[0] && src0->ne[1] == weight->ne[1] &&
                                    src0->ne[2] == weight->ne[2] && src0->ne[3] == weight->ne[3]);
        ଯଦି (!is_row_bcast && !is_same_shape) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଯଦି (!ggml_are_same_shape(src0, ଗଣ୍ଠି.dst())) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (ggml_is_contiguous(src0) != ggml_is_contiguous(ଗଣ୍ଠି.dst())) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ସଂରଚନା htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, weight, ଗଣ୍ଠି.dst(), &new_kparams
        );

        ଯଦି ((size_t) new_kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_size, sess->vtcm_size);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (!try_fuse_common(weight, ଗଣ୍ଠି.dst())) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        last_ଗଣ୍ଠି.opcode = HTP_OP_RMS_NORM_MUL;
        last_ଗଣ୍ଠି.name   = "RMS_NORM+MUL";
        last_ଗଣ୍ଠି.inputs.clear();
        last_ଗଣ୍ଠି.inputs.push_back(src0);
        last_ଗଣ୍ଠି.inputs.push_back(weight);
        last_ଗଣ୍ଠି.outputs.clear();
        last_ଗଣ୍ଠି.outputs.push_back(ଗଣ୍ଠି.dst());
        last_ଗଣ୍ଠି.fused.push_back(ଗଣ୍ଠି.ଗଣ୍ଠି);
        memcpy(last_ଗଣ୍ଠି.kernel_params, &new_kparams, ଆକାର(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        memcpy(o.kernel_params, &new_kparams, ଆକାର(new_kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(weight);
        ପାଇଁ (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(ଗଣ୍ଠି.dst());
        ପାଇଁ (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ତାର୍କିକ try_fuse_mul_mat_add(ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଯଦି (n_ops == 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି (ଗଣ୍ଠି.opcode != HTP_OP_ADD) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        htp_opnode & last_ଗଣ୍ଠି = ops[n_ops - 1];
        ଯଦି (last_ଗଣ୍ଠି.opcode != HTP_OP_MUL_MAT) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଅଚଳ ggml_tensor * add_src0 = ଗଣ୍ଠି.src0();
        ଅଚଳ ggml_tensor * add_src1 = ଗଣ୍ଠି.src1();
        ଅଚଳ ggml_tensor * mm_out   = last_ଗଣ୍ଠି.dst();

        ଯଦି (!ggml_hexagon_tensor_is_fuseable(mm_out)) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଅଚଳ ggml_tensor * src2;
        ଯଦି (add_src0 == mm_out || add_src0->data == mm_out->data) {
            src2 = add_src1;
        } ଅନ୍ୟଥା ଯଦି (add_src1 == mm_out || add_src1->data == mm_out->data) {
            src2 = add_src0;
        } ଅନ୍ୟଥା {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ggml_tensor * src0 = last_ଗଣ୍ଠି.src0();
        ଅଚଳ ggml_tensor * src1 = last_ଗଣ୍ଠି.src1();

        ଯଦି (src2->type != GGML_TYPE_F32) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        ଯଦି (!ggml_are_same_shape(mm_out, ଗଣ୍ଠି.dst())) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଅଚଳ ସଂରଚନା htp_mm_kernel_params * orig_kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) last_ଗଣ୍ଠି.kernel_params;
        ସଂରଚନା htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_add_params(sess, src0, src1, src2, ଗଣ୍ଠି.dst(), &kparams);
        ଯଦି (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (orig_kparams->collapse != kparams.collapse) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        ଅଚଳ ତାର୍କିକ can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        ଯଦି (!can_fuse) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଯଦି ((size_t) kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            ଯଦି (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_threads < orig_kparams->n_act_threads) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_threads, kparams.n_act_threads);
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }

        ଯଦି (!try_fuse_common(src2, ଗଣ୍ଠି.dst())) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        last_ଗଣ୍ଠି.opcode = HTP_OP_MUL_MAT_ADD;
        last_ଗଣ୍ଠି.name   = "MUL_MAT+ADD";
        last_ଗଣ୍ଠି.inputs.clear();
        last_ଗଣ୍ଠି.inputs.push_back(src0);
        last_ଗଣ୍ଠି.inputs.push_back(src1);
        last_ଗଣ୍ଠି.inputs.push_back(src2);
        last_ଗଣ୍ଠି.outputs.clear();
        last_ଗଣ୍ଠି.outputs.push_back(ଗଣ୍ଠି.dst());
        last_ଗଣ୍ଠି.fused.push_back(ଗଣ୍ଠି.ଗଣ୍ଠି);
        memcpy(last_ଗଣ୍ଠି.kernel_params, &kparams, ଆକାର(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ADD;
        memcpy(o.kernel_params, &kparams, ଆକାର(kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(src1);
        o.src[2] = add_tensor(src2);
        ପାଇଁ (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(ଗଣ୍ଠି.dst());
        ପାଇଁ (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ତାର୍କିକ try_fuse_mul_mat_nx(ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଯଦି (n_ops == 0 || ଗଣ୍ଠି.opcode != HTP_OP_MUL_MAT) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି (!is_mergeable_mul_mat(ଗଣ୍ଠି.ଗଣ୍ଠି)) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଅଚଳ ggml_tensor * w_in = ଗଣ୍ଠି.src0();
        ଅଚଳ ggml_tensor * x_in = ଗଣ୍ଠି.src1();
        ଅଚଳ ggml_tensor * d_in = ଗଣ୍ଠି.dst();

        htp_opnode & last_ଗଣ୍ଠି = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        ଯଦି (last_ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT_NX) {
            ଅଚଳ uint32_t curr_n = (uint32_t) last_ଗଣ୍ଠି.outputs.size();
            ଯଦି (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଅଚଳ ggml_tensor * w0 = last_ଗଣ୍ଠି.inputs[0];
            ଅଚଳ ggml_tensor * x  = last_ଗଣ୍ଠି.inputs[curr_n];

            ଯଦି (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (!last_ଗଣ୍ଠି.fused.empty() && (mm_is_hmx_eligible(last_ଗଣ୍ଠି.fused[0]) != mm_is_hmx_eligible(ଗଣ୍ଠି.ଗଣ୍ଠି))) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଅଚଳ ସଂରଚନା htp_mm_kernel_params * orig_kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) last_ଗଣ୍ଠି.kernel_params;
            ଅଚଳ ତାର୍କିକ collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_in, x, d_in);
            ଯଦି (orig_kparams->collapse && !collapse) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ସଂରଚନା htp_mm_kernel_params kparams;
            ଯଦି (collapse) {
                ଅଚଳ ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } ଅନ୍ୟଥା {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            ଯଦି (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଯଦି (!try_fuse_common(w_in, d_in)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            last_ଗଣ୍ଠି.inputs[curr_n] = w_in;
            last_ଗଣ୍ଠି.inputs.push_back(x);
            last_ଗଣ୍ଠି.outputs.push_back(d_in);
            last_ଗଣ୍ଠି.fused.push_back(ଗଣ୍ଠି.ଗଣ୍ଠି);
            memcpy(last_ଗଣ୍ଠି.kernel_params, &kparams, ଆକାର(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, ଆକାର(kparams));

            ପାଇଁ (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = add_tensor(last_ଗଣ୍ଠି.inputs[s]);
            }
            ପାଇଁ (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            ପାଇଁ (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_ଗଣ୍ଠି.outputs[d]);
            }
            ପାଇଁ (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        // Case 2: last_node is single MUL_MAT
        ଯଦି (last_ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT) {
            ଯଦି (!is_mergeable_mul_mat_pair(last_ଗଣ୍ଠି.ଗଣ୍ଠି, ଗଣ୍ଠି.ଗଣ୍ଠି)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଅଚଳ ggml_tensor * w0 = last_ଗଣ୍ଠି.src0();
            ଅଚଳ ggml_tensor * x  = last_ଗଣ୍ଠି.src1();
            ଅଚଳ ggml_tensor * w1 = ଗଣ୍ଠି.src0();
            ଅଚଳ ggml_tensor * dst_0 = last_ଗଣ୍ଠି.dst();
            ଅଚଳ ggml_tensor * dst_1 = ଗଣ୍ଠି.dst();

            ଅଚଳ ସଂରଚନା htp_mm_kernel_params * orig_kparams = (ଅଚଳ ସଂରଚନା htp_mm_kernel_params *) last_ଗଣ୍ଠି.kernel_params;
            ଅଚଳ ତାର୍କିକ collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            ଯଦି (orig_kparams->collapse && !collapse) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ସଂରଚନା htp_mm_kernel_params kparams;
            ଯଦି (collapse) {
                ଅଚଳ ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } ଅନ୍ୟଥା {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            ଯଦି (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଯଦି (!try_fuse_common(w1, ଗଣ୍ଠି.dst())) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            last_ଗଣ୍ଠି.opcode = HTP_OP_MUL_MAT_NX;
            last_ଗଣ୍ଠି.name   = "MUL_MAT_NX";
            last_ଗଣ୍ଠି.inputs.clear();
            last_ଗଣ୍ଠି.inputs.push_back(w0);
            last_ଗଣ୍ଠି.inputs.push_back(w1);
            last_ଗଣ୍ଠି.inputs.push_back(x);
            last_ଗଣ୍ଠି.outputs.clear();
            last_ଗଣ୍ଠି.outputs.push_back(dst_0);
            last_ଗଣ୍ଠି.outputs.push_back(dst_1);
            last_ଗଣ୍ଠି.fused.push_back(ଗଣ୍ଠି.ଗଣ୍ଠି);
            memcpy(last_ଗଣ୍ଠି.kernel_params, &kparams, ଆକାର(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            memcpy(o.kernel_params, &kparams, ଆକାର(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            ପାଇଁ (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            ପାଇଁ (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ତାର୍କିକ try_fuse_mul_mat_id_nx(ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଯଦି (n_ops == 0 || ଗଣ୍ଠି.opcode != HTP_OP_MUL_MAT_ID) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି (!is_mergeable_mul_mat_id(ଗଣ୍ଠି.ଗଣ୍ଠି)) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଅଚଳ ggml_tensor * w_in   = ଗଣ୍ଠି.src0();
        ଅଚଳ ggml_tensor * x_in   = ଗଣ୍ଠି.src1();
        ଅଚଳ ggml_tensor * ids_in = ଗଣ୍ଠି.ଗଣ୍ଠି->src[2];
        ଅଚଳ ggml_tensor * d_in   = ଗଣ୍ଠି.dst();

        htp_opnode & last_ଗଣ୍ଠି = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        ଯଦି (last_ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT_ID_NX) {
            ଅଚଳ uint32_t curr_n = (uint32_t) last_ଗଣ୍ଠି.outputs.size();
            ଯଦି (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଅଚଳ ggml_tensor * w0  = last_ଗଣ୍ଠି.inputs[0];
            ଅଚଳ ggml_tensor * x   = last_ଗଣ୍ଠି.inputs[curr_n];
            ଅଚଳ ggml_tensor * ids = last_ଗଣ୍ଠି.inputs[curr_n + 1];

            ଯଦି (x_in != x || ids_in != ids || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (!last_ଗଣ୍ଠି.fused.empty() && (mm_is_hmx_eligible(last_ଗଣ୍ଠି.fused[0]) != mm_is_hmx_eligible(ଗଣ୍ଠି.ଗଣ୍ଠି))) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ସଂରଚନା htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
            ଯଦି (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଯଦି (!try_fuse_common(w_in, d_in)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            last_ଗଣ୍ଠି.inputs[curr_n] = w_in;
            last_ଗଣ୍ଠି.inputs[curr_n + 1] = x;
            last_ଗଣ୍ଠି.inputs.push_back(ids);
            last_ଗଣ୍ଠି.outputs.push_back(d_in);
            last_ଗଣ୍ଠି.fused.push_back(ଗଣ୍ଠି.ଗଣ୍ଠି);
            memcpy(last_ଗଣ୍ଠି.kernel_params, &kparams, ଆକାର(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, ଆକାର(kparams));

            ପାଇଁ (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = add_tensor(last_ଗଣ୍ଠି.inputs[s]);
            }
            ପାଇଁ (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            ପାଇଁ (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_ଗଣ୍ଠି.outputs[d]);
            }
            ପାଇଁ (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        // Case 2: last_node is single MUL_MAT_ID
        ଯଦି (last_ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT_ID) {
            ଯଦି (!is_mergeable_mul_mat_id_pair(last_ଗଣ୍ଠି.ଗଣ୍ଠି, ଗଣ୍ଠି.ଗଣ୍ଠି)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଅଚଳ ggml_tensor * w0  = last_ଗଣ୍ଠି.src0();
            ଅଚଳ ggml_tensor * x   = last_ଗଣ୍ଠି.src1();
            ଅଚଳ ggml_tensor * ids = last_ଗଣ୍ଠି.ଗଣ୍ଠି->src[2];
            ଅଚଳ ggml_tensor * w1  = ଗଣ୍ଠି.src0();

            ସଂରଚନା htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, ଗଣ୍ଠି.dst(), 2, &kparams);
            ଯଦି (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଯଦି (!try_fuse_common(w1, ଗଣ୍ଠି.dst())) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଅଚଳ ggml_tensor * dst_0 = last_ଗଣ୍ଠି.dst();
            ଅଚଳ ggml_tensor * dst_1 = ଗଣ୍ଠି.dst();

            last_ଗଣ୍ଠି.opcode = HTP_OP_MUL_MAT_ID_NX;
            last_ଗଣ୍ଠି.name   = "MUL_MAT_ID_NX";
            last_ଗଣ୍ଠି.inputs.clear();
            last_ଗଣ୍ଠି.inputs.push_back(w0);
            last_ଗଣ୍ଠି.inputs.push_back(w1);
            last_ଗଣ୍ଠି.inputs.push_back(x);
            last_ଗଣ୍ଠି.inputs.push_back(ids);
            last_ଗଣ୍ଠି.outputs.clear();
            last_ଗଣ୍ଠି.outputs.push_back(dst_0);
            last_ଗଣ୍ଠି.outputs.push_back(dst_1);
            last_ଗଣ୍ଠି.fused.push_back(ଗଣ୍ଠି.ଗଣ୍ଠି);
            memcpy(last_ଗଣ୍ଠି.kernel_params, &kparams, ଆକାର(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            memcpy(o.kernel_params, &kparams, ଆକାର(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            o.src[3] = add_tensor(ids);
            ପାଇଁ (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            ପାଇଁ (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ତାର୍କିକ try_fuse_gdn_cpy(ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଯଦି (n_ops == 0 || ଗଣ୍ଠି.opcode != HTP_OP_CPY) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        htp_opnode & last_ଗଣ୍ଠି = ops[n_ops - 1];
        ଯଦି (last_ଗଣ୍ଠି.opcode != HTP_OP_GATED_DELTA_NET) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି (last_ଗଣ୍ଠି.outputs.size() != 1) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଅଚଳ ggml_tensor * gdn_out  = last_ଗଣ୍ଠି.dst();
        ଅଚଳ ggml_tensor * cpy_ଗଣ୍ଠି = ଗଣ୍ଠି.ଗଣ୍ଠି;
        ଅଚଳ ggml_tensor * cpy_src  = ଗଣ୍ଠି.src0();
        ଅଚଳ ggml_tensor * cpy_dst  = ଗଣ୍ଠି.dst();

        ଯଦି (!cpy_src || !cpy_dst || !cpy_dst->data) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_OUTPUT)) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଅଚଳ ggml_tensor * v = last_ଗଣ୍ଠି.ଗଣ୍ଠି->src[2];
        ଯଦି (!v) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ଅଚଳ int64_t S_v      = v->ne[0];
        ଅଚଳ int64_t H        = v->ne[1];
        ଅଚଳ int64_t n_tokens = v->ne[2];
        ଅଚଳ int64_t n_seqs   = v->ne[3];
        ଅଚଳ int64_t K        = ggml_get_op_params_i32(last_ଗଣ୍ଠି.ଗଣ୍ଠି, 0);
        ଅଚଳ size_t  tail_off = (size_t) S_v * H * n_tokens * n_seqs * ଆକାର(ଦଶମିକ);

        ଅଚଳ int64_t D         = S_v * S_v * H;
        ଅଚଳ int64_t n_written = std::min<int64_t>(n_tokens, K);

        ଯଦି (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_out && cpy_src->view_src->data != gdn_out->data) ||
            cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != ଆକାର(ଦଶମିକ)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * ଆକାର(ଦଶମିକ)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (n_written > 1) {
            ଯଦି (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * ଆକାର(ଦଶମିକ)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }

        ଯଦି (!try_fuse_common({cpy_dst})) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        last_ଗଣ୍ଠି.name += "+CPY";
        last_ଗଣ୍ଠି.outputs.push_back(cpy_dst);
        last_ଗଣ୍ଠି.fused.push_back(ଗଣ୍ଠି.ଗଣ୍ଠି);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = add_tensor(cpy_dst);
        ପାଇଁ (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ତାର୍କିକ try_fuse(ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
        ଯଦି (!opt_opfusion) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ADD) && try_fuse_allreduce_add(ଗଣ୍ଠି)) ଫେରନ୍ତୁ ସତ୍ୟ;
        ଯଦି (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(ଗଣ୍ଠି))  ଫେରନ୍ତୁ ସତ୍ୟ;
        ଯଦି (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ADD)   && try_fuse_mul_mat_add(ଗଣ୍ଠି))   ଫେରନ୍ତୁ ସତ୍ୟ;
        ଯଦି (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(ଗଣ୍ଠି))    ଫେରନ୍ତୁ ସତ୍ୟ;
        ଯଦି (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(ଗଣ୍ଠି)) ଫେରନ୍ତୁ ସତ୍ୟ;
        ଯଦି (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(ଗଣ୍ଠି))       ଫେରନ୍ତୁ ସତ୍ୟ;
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
};

ସଂରଚନା ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

ସଂରଚନା ggml_hexagon_opqueue {
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
        ଯଦି (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * ଆକାର(htp_trace_desc);
        }

        shm_blk_size = ଆକାର(htp_buf_desc)  * n_bufs    +
                       ଆକାର(htp_tensor)    * n_tensors +
                       ଆକାର(htp_op_desc)   * n_ops     +
                       ଆକାର(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = ନୂତନ ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, ସତ୍ୟ /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        ଯଦି (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size(), shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    size_t shm_size() ଅଚଳ { ଫେରନ୍ତୁ shm_buf ? shm_buf->size() : 0; }

    // push new batch
    ତାର୍କିକ ଠେଲ(htp_opbatch_req& req, dspqueue_buffer& dbuf, ଅଚଳ ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(ଆକାର(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(ଆକାର(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(ଆକାର(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(ଆକାର(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(ଆକାର(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(ଆକାର(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        ଯଦି (seq - shm_buf->sess->batch_rsp_seq > depth) { ଫେରନ୍ତୁ ମିଥ୍ୟା; }

        ଅଚଳ uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[slot]   = op_batch->ops;
        start_usec[slot] = ggml_time_us();

        ଅଚଳ size_t b_size = ଆକାର(htp_buf_desc)  * req.n_bufs;
        ଅଚଳ size_t t_size = ଆକାର(htp_tensor)    * req.n_tensors;
        ଅଚଳ size_t o_size = ଆକାର(htp_op_desc)   * req.n_ops;
        ଅଚଳ size_t p_size = ଆକାର(htp_prof_desc) * req.n_ops;

        size_t tr_size = 0;
        ଯଦି (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * ଆକାର(htp_trace_desc);
        } ଅନ୍ୟଥା {
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

        memcpy(b_ptr, (ଶୂନ୍ୟ *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (ଶୂନ୍ୟ *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (ଶୂନ୍ୟ *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        ଯଦି (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (ଶୂନ୍ୟ *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (unsigned long long) t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଶୂନ୍ୟ କାଢ଼(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        ଅଚଳ uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        ଅଚଳ size_t b_size = ଆକାର(htp_buf_desc)  * rsp.n_bufs;
        ଅଚଳ size_t t_size = ଆକାର(htp_tensor)    * rsp.n_tensors;
        ଅଚଳ size_t o_size = ଆକାର(htp_op_desc)   * rsp.n_ops;
        ଅଚଳ size_t p_size = ଆକାର(htp_prof_desc) * rsp.n_ops;

        size_t tr_size = 0;
        uint32_t n_traces = 0;
        ଯଦି (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * ଆକାର(htp_trace_desc);
        }

        ଅଚଳ size_t m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        ଯଦି (rsp.n_ops > 0) {
            auto & ops = op_cache[slot];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            ଅଚଳ htp_prof_desc * pd = (ଅଚଳ htp_prof_desc *) p_ptr;
            ଅଚଳ htp_trace_desc * trace_events = nullptr;
            ଯଦି (opt_profile == 3) {
                trace_events = (ଅଚଳ htp_trace_desc *) (p_ptr + p_size);
            }

            ଯଦି (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            ପାଇଁ (uint32_t i = 0; i < rsp.n_ops; i++) {
                ଯଦି (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            ଯଦି (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

ଶୂନ୍ୟ ggml_hexagon_session::flush_peers() {
    auto vpeers = std::move(virt_peers);
    virt_peers.clear();
    ପାଇଁ (auto * peer : vpeers) {
        peer->flush_sync();
    }

    auto ppeers = std::move(phys_peers);
    phys_peers.clear();
    ପାଇଁ (auto * peer : ppeers) {
        peer->flush_async();
    }

    ପାଇଁ (auto & sub : ଏହି->mdev.sessions) {
        sub->flush_peers();
    }
}

ଶୂନ୍ୟ ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

ଶୂନ୍ୟ ggml_hexagon_session::flush_pending(ତାର୍କିକ all) {
    ପାଇଁ (auto & sub : ଏହି->mdev.sessions) {
        sub->flush_pending(all);
        ଯଦି (sub->last_error > HTP_STATUS_OK) {
            ଏହି->last_error = sub->last_error.load();
        }
    }

    ଯେତେବେଳେ (ଏହି->batch_rsp_seq < ଏହି->batch_req_seq) {
        ସଂରଚନା htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        ସଂରଚନା dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        ଅଚଳ uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = dspqueue_read(ଏହି->queue, &flags, 1, &n_dbufs, &dbuf, ଆକାର(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        ଯଦି (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            ଜାରି;
        }

        ଯଦି (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        ଯଦି (rsp_size != ଆକାର(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", ଏହି->c_name(), rsp_size, n_dbufs);
        }

        ଯଦି (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", ଏହି->c_name(), status_to_str(rsp.status));
            ଏହି->last_error = rsp.status;
            ପାଇଁ (auto & sub : ଏହି->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->କାଢ଼(rsp, dbuf);

        GGML_ASSERT(rsp.seq == ଏହି->batch_rsp_seq + 1);
        ଏହି->batch_rsp_seq = rsp.seq;

        ଯଦି (!all) ଭାଙ୍ଗ;
    }
}

ଶୂନ୍ୟ ggml_hexagon_session::flush_sync(ତାର୍କିକ all) {
    flush_async();
    flush_pending(all);
}

ଶୂନ୍ୟ ggml_hexagon_session::start_batch() {
    ଯଦି (ଏହି->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

ଶୂନ୍ୟ ggml_hexagon_session::flush_batch(size_t min_ops) {
    ଯଦି (op_batch->n_ops < min_ops) { ଫେରନ୍ତୁ; }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    ଅଚଳ uint64_t seq = ++ଏହି->batch_req_seq;

    op_batch->update_mdev_group(ଏହି->mdev.idx);

    ଯଦି (!op_queue->ଠେଲ(req, dbuf, op_batch, seq)) {
        flush_pending(ମିଥ୍ୟା);
        op_queue->ଠେଲ(req, dbuf, op_batch, seq);
    }

    ପାଇଁ (auto & sub : ଏହି->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_buffer sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        ଯଦି (!sub->op_queue->ଠେଲ(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(ମିଥ୍ୟା);
            sub->op_queue->ଠେଲ(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, ଆକାର(sub_req), (ଅଚଳ uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        ଯଦି (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (unsigned) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", ଏହି->c_name(), dbuf.ptr, dbuf.size);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = dspqueue_write(ଏହି->queue, 0, 1, &dbuf, ଆକାର(req), (ଅଚଳ uint8_t*) &req, DSPQUEUE_TIMEOUT);
    ଯଦି (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", ଏହି->c_name(), (unsigned) err);
    }

    op_batch->reset();
}

ଶୂନ୍ୟ ggml_hexagon_session::enqueue_op(ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
    auto clone_tensor_buffer = [ଏହି](ଅଚଳ ggml_tensor * t) {
        auto sbuf = ଏହି->mmap_tensor(t);
        ଯଦି (!sbuf) ଫେରନ୍ତୁ;
        ଯଦି (sbuf->sess != ଏହି) {
            ଏହି->clone_buffer(sbuf);
        }
        ପାଇଁ (auto & sub : ଏହି->mdev.sessions) {
            sub->clone_buffer(sbuf);
        }
    };

    ପାଇଁ (auto t : ଗଣ୍ଠି.get_inputs()) {
        clone_tensor_buffer(t);
    }
    ପାଇଁ (auto t : ଗଣ୍ଠି.get_outputs()) {
        clone_tensor_buffer(t);
    }

    ଯଦି (opt_opfusion && op_batch->try_fuse(ଗଣ୍ଠି)) {
        ଫେରନ୍ତୁ;
    }

    ଯଦି (!op_batch->fit_op(ଗଣ୍ଠି)) {
        flush_async();
    }

    ଯଦି (op_batch->empty()) {
        start_batch();
    }

    ଯଦି (!op_batch->fit_op(ଗଣ୍ଠି)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->add_op(ଗଣ୍ଠି);
}

ଶୂନ୍ୟ ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_ଗଣ୍ଠି(HTP_OP_MDEV_GROUP);

    uint8_t * fence_slot = ଏହି->mdev_fence_slot;

    ସ୍ଥିର ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.buffer       = &ଏହି->fence_buf->backend_buffer;
    dummy_t.extra        = &fence_extra;
    dummy_t.data         = (ଶୂନ୍ୟ *) fence_slot;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.ne[1]        = (int64_t) ଏହି->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) ଏହି->mdev.idx;

    ggml_tensor * ଗଣ୍ଠି = group_ଗଣ୍ଠି.add_dummy(dummy_t);
    ଗଣ୍ଠି->src[0] = ଗଣ୍ଠି;
    group_ଗଣ୍ଠି.init(ଗଣ୍ଠି);
    group_ଗଣ୍ଠି.outputs.clear();
    group_ଗଣ୍ଠି.name = "MDEV_GROUP";

    ପାଇଁ (auto & sub : ଏହି->mdev.sessions) {
        sub->clone_buffer(ଏହି->fence_buf);
    }

    op_batch->add_op(group_ଗଣ୍ଠି);
}

ଶୂନ୍ୟ ggml_hexagon_session::enqueue_cpy(ଅଚଳ ggml_tensor * src, ggml_tensor * dst, ଅଚଳ ggml_tensor * sync_tensor, uint32_t fence_seq) {
    ଅଚଳ ତାର୍କିକ with_fence = sync_tensor != nullptr;
    htp_opnode cpy_ଗଣ୍ଠି(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* ଗଣ୍ଠି = cpy_ଗଣ୍ଠି.add_dummy(*dst);
    ଗଣ୍ଠି->op     = GGML_OP_CPY;
    ଗଣ୍ଠି->src[0] = const_cast<ggml_tensor *>(src);
    ଗଣ୍ଠି->src[1] = with_fence ? cpy_ଗଣ୍ଠି.add_dummy(*sync_tensor) : nullptr;
    ଯଦି (with_fence) {
        ଗଣ୍ଠି->op_params[0] = (int32_t) fence_seq;
    }

    cpy_ଗଣ୍ଠି.init(ଗଣ୍ଠି);
    ଯଦି (with_fence) {
        cpy_ଗଣ୍ଠି.name = "CPY+FENCE";
    }
    ଅଚଳ ତାର୍କିକ ok = ggml_hexagon_precompute_cpy_params(ଏହି, ଗଣ୍ଠି, (ସଂରଚନା htp_copy_kernel_params *) cpy_ଗଣ୍ଠି.kernel_params);
    ଅଚଳ auto * kparams = (ଅଚଳ ସଂରଚନା htp_copy_kernel_params *) cpy_ଗଣ୍ଠି.kernel_params;
    ଯଦି (ok && !with_fence && kparams->total_elems == 0) {
        ଫେରନ୍ତୁ;
    }
    ଏହି->enqueue_op(cpy_ଗଣ୍ଠି);
}

ଶୂନ୍ୟ ggml_hexagon_session::enqueue_fence(ଅଚଳ ggml_tensor * sync_tensor, uint32_t fence_seq, ତାର୍କିକ wait) {
    htp_opnode sync_ଗଣ୍ଠି(HTP_OP_FENCE);

    ggml_tensor* ଗଣ୍ଠି = sync_ଗଣ୍ଠି.add_dummy(*sync_tensor);
    ଗଣ୍ଠି->op           = GGML_OP_NONE;
    ଗଣ୍ଠି->src[0]       = ଗଣ୍ଠି;
    ଗଣ୍ଠି->op_params[0] = (int32_t) fence_seq;
    ଗଣ୍ଠି->op_params[1] = wait ? 0 : 1;

    sync_ଗଣ୍ଠି.init(ଗଣ୍ଠି);
    sync_ଗଣ୍ଠି.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    ଏହି->enqueue_op(sync_ଗଣ୍ଠି);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_allreduce_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    ତାର୍କିକ has_add,
    ତାର୍କିକ is_row_bcast,
    ତାର୍କିକ is_shard_ok,
    ସଂରଚନା htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    ଅଚଳ uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    ଅଚଳ uint32_t elem_size = (dst->type == GGML_TYPE_F16) ? ଆକାର(ggml_fp16_t) : ଆକାର(ଦଶମିକ);
    ଅଚଳ ତାର୍କିକ is_contiguous = ggml_is_contiguous(dst);

    ଅଚଳ uint32_t ne0 = (uint32_t) dst->ne[0];
    ଅଚଳ uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    ଅଚଳ ତାର୍କିକ use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    ଯଦି (has_add) {
        // sharded reduce-scatter for contiguous in-place add
        ଯଦି (opt_ar_scatter && is_shard_ok && use_1d && !is_row_bcast) {
            ଅଚଳ uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            ଅଚଳ uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            ଅଚଳ uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_start = (int32_t) rank_elem_start;
            kparams->rank_nelem      = (int32_t) (rank_elem_end - rank_elem_start);
        } ଅନ୍ୟଥା ଯଦି (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } ଅନ୍ୟଥା {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } ଅନ୍ୟଥା {
        kparams->n_dsts = (int32_t) n_ranks;
        ଯଦି (use_1d) {
            ଅଚଳ uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            ଅଚଳ uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            ଅଚଳ uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            ଅଚଳ uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } ଅନ୍ୟଥା {
            ଅଚଳ uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            ଅଚଳ uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            ଅଚଳ uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            ଅଚଳ uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    ଯଦି (use_1d) {
        ଅଚଳ uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        ଅଚଳ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        ଅଚଳ size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_ଗଣନା(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        ଯଦି (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        ଯେତେବେଳେ ((size_t) kparams->vtcm_size > sess->vtcm_size && block_elems > 128) {
            ଅଚଳ size_t max_bytes_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((size_t) (max_bytes_per_buf / elem_size), 128);
            ଯଦି (block_elems < 128) ଭାଙ୍ଗ;
            kparams->block_elems          = block_elems;
            kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
        }

        ଯଦି (sess->vtcm_size < (size_t) kparams->vtcm_size || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    } ଅନ୍ୟଥା {
        ଅଚଳ uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        ଅଚଳ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        ଅଚଳ size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_ଗଣନା(n_ranks, n_threads, has_add, is_row_bcast);

        ଅଚଳ uint32_t row_bytes = ne0 * elem_size;
        ଅଚଳ uint32_t row_size_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_size_aligned = row_size_aligned;

        ଅଚଳ uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        ଯେତେବେଳେ ((size_t) kparams->vtcm_size > sess->vtcm_size && block_rows > 1) {
            ଅଚଳ size_t max_rows_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2 * row_size_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
            ଯଦି (max_rows_per_buf == 0) ଭାଙ୍ଗ;
        }

        ଯଦି (sess->vtcm_size < (size_t) kparams->vtcm_size || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
}

ଶୂନ୍ୟ ggml_hexagon_session::enqueue_allreduce(
    ଅଚଳ ggml_tensor * dst,
    ଅଚଳ std::vector<ଅଚଳ ggml_tensor *> & src_tensors,
    ଅଚଳ std::vector<ଅଚଳ ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_ଗଣ୍ଠି(HTP_OP_ALLREDUCE);

    ggml_tensor* ଗଣ୍ଠି = ar_ଗଣ୍ଠି.add_dummy(*dst);
    ଗଣ୍ଠି->op           = GGML_OP_NONE;
    ଗଣ୍ଠି->op_params[0] = (int32_t) fence_seq_entry;
    ଗଣ୍ଠି->op_params[1] = (int32_t) fence_seq_exit;

    ar_ଗଣ୍ଠି.init(ଗଣ୍ଠି);

    ar_ଗଣ୍ଠି.inputs.clear();
    ପାଇଁ (size_t i = 0; i < src_tensors.size(); i++) {
        ar_ଗଣ୍ଠି.inputs.push_back(src_tensors[i]);
    }
    ପାଇଁ (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_ଗଣ୍ଠି.inputs.push_back(ar_ଗଣ୍ଠି.add_dummy(*sync_tensors[i]));
    }

    ar_ଗଣ୍ଠି.outputs.clear();
    ପାଇଁ (size_t i = 0; i < src_tensors.size(); i++) {
        ar_ଗଣ୍ଠି.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        ଏହି, dst, rank, n_ranks, ମିଥ୍ୟା, ମିଥ୍ୟା, /*is_shard_ok=*/ ମିଥ୍ୟା,
        (ସଂରଚନା htp_allreduce_kernel_params *) ar_ଗଣ୍ଠି.kernel_params);

    ar_ଗଣ୍ଠି.name = "ALLREDUCE";
    ଏହି->enqueue_op(ar_ଗଣ୍ଠି);
}

ggml_hexagon_shared_buffer * ggml_hexagon_session::mmap_tensor(ଅଚଳ ggml_tensor * t) {
    ଯଦି (!t) ଫେରନ୍ତୁ nullptr;

    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
    ଯଦି (!sbuf->mapped) {
        ଅଚଳ ତାର୍କିକ is_weight = ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        ଅଚଳ ତାର୍କିକ extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    ଫେରନ୍ତୁ sbuf;
}

ତାର୍କିକ ggml_hexagon_session::clone_buffer(ଅଚଳ ggml_hexagon_shared_buffer * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    ଯଦି (sbuf->sess == ଏହି) ଫେରନ୍ତୁ ସତ୍ୟ;

    auto mem = sbuf->mem;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    ଯଦି (ଏହି->cloned_buffers.find(fd) != ଏହି->cloned_buffers.end()) ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", ଏହି->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    auto clone = std::make_unique<ggml_hexagon_shared_buffer>(ଏହି, *sbuf);
    ଚେଷ୍ଟା {
        clone->mmap(sbuf->extended);
    } ଧର (ଅଚଳ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", ଏହି->c_name(), exc.what());
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଏହି->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(ଏହି);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ଶୂନ୍ୟ ggml_hexagon_session::release_buffer(ଅଚଳ ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    auto it = ଏହି->cloned_buffers.find(fd);
    ଯଦି (it != ଏହି->cloned_buffers.end()) {
        auto clone = std::move(it->second);
        ଏହି->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(ଏହି);
}

ଶୂନ୍ୟ ggml_hexagon_session::unclone_buffer(ଅଚଳ ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    ପାଇଁ (auto * sess : sessions) {
        sess->release_buffer(sbuf);
    }
}

ସ୍ଥିର size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    ଅଚଳ size_t MiB = 1024 * 1024;
    ଅଚଳ size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t step = 256u * MiB;

    ଚେଷ୍ଟା {
        sbufs.push_back(ନୂତନ ggml_hexagon_shared_buffer(sess, GiB, ସତ୍ୟ)); vmem += GiB;
        sbufs.push_back(ନୂତନ ggml_hexagon_shared_buffer(sess, GiB, ସତ୍ୟ)); vmem += GiB;
        sbufs.push_back(ନୂତନ ggml_hexagon_shared_buffer(sess, GiB, ସତ୍ୟ)); vmem += GiB;

        ଯେତେବେଳେ (1) {
            sbufs.push_back(ନୂତନ ggml_hexagon_shared_buffer(sess, step, ସତ୍ୟ));
            vmem += step;
        }
    } ଧର (...) { }

    ପାଇଁ (auto b : sbufs) { delete b; }

    ଫେରନ୍ତୁ vmem - step; // backoff to account for overhead from internal mappings
}

ଶୂନ୍ୟ ggml_hexagon_session::allocate(ଅଚଳ ggml_hexagon_device_config & config) noexcept(ମିଥ୍ୟା) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା phys_idx = config.physical_idx;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା virt_idx = config.virtual_idx;

    ଏହି->valid_session = ମିଥ୍ୟା;
    ଏହି->valid_handle  = ମିଥ୍ୟା;
    ଏହି->valid_queue   = ମିଥ୍ୟା;
    ଏହି->valid_iface   = ମିଥ୍ୟା;

    ଏହି->name          = config.name;
    ଏହି->phys_idx      = phys_idx;
    ଏହି->virt_idx      = virt_idx;
    ଏହି->domain_id     = config.domain_id;
    ଏହି->session_id    = 0;
    ଏହି->batch_req_seq = 0;
    ଏହି->batch_rsp_seq = 0;
    ଏହି->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", ଏହି->name.c_str(), ଏହି->domain_id, phys_idx, virt_idx);

    ଯଦି (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        ଫିଙ୍ଗ std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    ଅଚଳ std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    ଯଦି (virt_idx > 0) {
        ସଂରଚନା remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<ଅକ୍ଷର *>(dom_name.c_str());
        n.session_name     = const_cast<ଅକ୍ଷର *>(ଏହି->name.c_str());
        n.session_name_len = ଏହି->name.size();
        n.session_id       = virt_idx;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (ଶୂନ୍ୟ *) &n, ଆକାର(n));
        ଯଦି (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           ଏହି->c_name(), phys_idx, virt_idx, err);
            ଫିଙ୍ଗ std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        ଏହି->session_id    = n.session_id;
        ଏହି->domain_id     = n.effective_domain_id;
        ଏହି->valid_session = ସତ୍ୟ;
    } ଅନ୍ୟଥା {
        ସଂରଚନା remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<ଅକ୍ଷର *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (ଶୂନ୍ୟ *) &eff, ଆକାର(eff));
        ଯଦି (err == AEE_SUCCESS) {
            ଏହି->domain_id = eff.effective_domain_id;
        } ଅନ୍ୟଥା {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           ଏହି->name.c_str(), err, ଏହି->domain_id);
        }
    }

    // Enable unsigned modules
    {
        ସଂରଚନା remote_rpc_control_unsigned_module u;
        u.domain = ଏହି->domain_id;
        u.enable = 1;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (ଶୂନ୍ୟ *) &u, ଆକାର(u));
        ଯଦି (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", ଏହି->c_name(), err);
            ଫିଙ୍ଗ std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    ଅକ୍ଷର session_uri[256];
    {
        ଅକ୍ଷର htp_uri[256];
        snprintf(htp_uri, ଆକାର(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        ସଂରଚନା remote_rpc_get_uri u = {};
        u.session_id      = ଏହି->session_id;
        u.domain_name     = const_cast<ଅକ୍ଷର *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<ଅକ୍ଷର *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = ଆକାର(session_uri);

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = remote_session_control(FASTRPC_GET_URI, (ଶୂନ୍ୟ *) &u, ଆକାର(u));
        ଯଦି (err != AEE_SUCCESS) {
            snprintf(session_uri, ଆକାର(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), ଏହି->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          ଏହି->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = htp_iface_open(session_uri, &ଏହି->handle);
    ଯଦି (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", ଏହି->c_name(), session_uri, err);
        ଫିଙ୍ଗ std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    ଏହି->valid_handle = ସତ୍ୟ;

    // Query HW info and resolve session options
    {
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା hw_n_threads = 0;
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା hw_n_hvx     = 0;
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା hw_err = htp_iface_hwinfo(ଏହି->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        ଯଦି (hw_err == 0) {
            ଅଚଳ uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            ଏହି->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_threads) : max_n_threads;
            ଏହି->n_hvx     = ଏହି->n_threads;
            ଏହି->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            ଏହି->vtcm_size = (uint64_t)hw_vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          ଏହି->c_name(), ଏହି->n_threads, ଏହି->n_hvx, ଏହି->n_hmx,
                          (unsigned long long)(ଏହି->vtcm_size / (1024 * 1024)));
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", ଏହି->c_name(), hw_err);
            ଅଚଳ uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            ଏହି->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_threads;
            ଏହି->n_hvx     = ଏହି->n_threads;
            ଏହି->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            ଏହି->vtcm_size = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        ସଂରଚନା remote_rpc_control_latency l;
        l.enable = 1;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = remote_handle64_control(ଏହି->handle, DSPRPC_CONTROL_LATENCY, (ଶୂନ୍ୟ *) &l, ଆକାର(l));
        ଯଦି (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", ଏହି->c_name(),
                  ଏହି->session_id, ଏହି->domain_id, session_uri, (unsigned long) ଏହି->handle);

    ଅଚଳ size_t req_q_size = (ଆକାର(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    ଅଚଳ size_t rsp_q_size = (ଆକାର(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(ଏହି->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (ଶୂନ୍ୟ *) ଏହି,  // Callback context
                          &queue);
    ଯଦି (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", ଏହି->name.c_str(), (unsigned) err);
        ଫିଙ୍ଗ std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    ଏହି->valid_queue = ସତ୍ୟ;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &ଏହି->queue_id);
    ଯଦି (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        ଫିଙ୍ଗ std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    ଯଦି (opt_etm) {
        err = htp_iface_etm(ଏହି->handle, 1);
        ଯଦି (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    ଏହି->op_queue = ନୂତନ ggml_hexagon_opqueue(ଏହି, opt_opbatch, opt_opqueue);

    ଏହି->fence_buf = ନୂତନ ggml_hexagon_fence_buffer(ଏହି, &dev_ctx->fence_buffer_type, 64 * 1024);
    ଯଦି (ଏହି->mdev.count > 1) {
        ଏହି->mdev_fence_slot = ଏହି->alloc_fence(ଏହି->mdev.count);
    }

    ଯଦି (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(ଏହି);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", ଏହି->c_name(), opt_vmem);
    }
    ଅଚଳ size_t shm_size = ଏହି->op_queue->shm_size();
    ଏହି->max_vmem = (opt_vmem > shm_size) ? (opt_vmem - shm_size) : opt_vmem;

    ଏହି->op_batch = ନୂତନ ggml_hexagon_opbatch(ଏହି, opt_opbatch, ଏହି->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(ଏହି->handle, ଏହି->session_id, ଏହି->queue_id, ଏହି->n_threads, opt_nhmx, ଏହି->max_vmem, (uint32_t) opt_power_mode);
    ଯଦି (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", ଏହି->c_name(), (unsigned) err);
        ଫିଙ୍ଗ std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    ଏହି->valid_iface = ସତ୍ୟ;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", ଏହି->c_name(), htp_power_mode_to_str(opt_power_mode));

    ଯଦି (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(ଏହି->handle, opt_profile, &pmu_conf);
        ଯଦି (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }
}

ଶୂନ୍ୟ ggml_hexagon_session::release() noexcept(ସତ୍ୟ) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", ଏହି->name.c_str());

    ଏହି->mdev.sessions.clear();

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err;

    ଯଦି (ଏହି->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(ଏହି->handle);
        ଯଦି (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    delete ଏହି->op_batch;
    delete ଏହି->op_queue;
    ପାଇଁ (auto & it : ଏହି->cpy_fence_slots) {
        free_fence((ଶୂନ୍ୟ *) it.second, 1);
    }
    ଏହି->cpy_fence_slots.clear();

    ଯଦି (ଏହି->fence_buf) {
        unclone_buffer(ଏହି->fence_buf);
        delete ଏହି->fence_buf;
        ଏହି->fence_buf = nullptr;
    }
    ଯେତେବେଳେ (!ଏହି->cloned_buffers.empty()) {
        release_buffer(ଏହି->cloned_buffers.begin()->second.get());
    }

    ଯଦି (opt_etm) {
        err = htp_iface_etm(ଏହି->handle, 0);
        ଯଦି (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    ଯଦି (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(ଏହି->handle, 0, &pmu_conf);
        ଯଦି (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    ଯଦି (ଏହି->valid_queue) {
        err = dspqueue_close(queue);
        ଯଦି (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (unsigned) err);
        }
    }

    ଯଦି (ଏହି->valid_handle) {
        htp_iface_close(ଏହି->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(ଅଚଳ ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_ଗଣନା) noexcept(ମିଥ୍ୟା) {
    ଏହି->dev        = dev;
    ଏହି->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ଏହି->mdev.idx   = mdev_idx;
    ଏହି->mdev.count = mdev_ଗଣନା > 0 ? mdev_ଗଣନା : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)ଏହି) & 0xFFFF;

    ଚେଷ୍ଟା {
        allocate(config);
        ଯଦି (ଏହି->mdev.idx == 0 && !config.mdev_group.empty()) {
            ପାଇଁ (size_t i = 0; i < config.mdev_group.size(); i++) {
                ଏହି->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], ଏହି->dev, (uint32_t) (i + 1), ଏହି->mdev.count));
            }
        }
    } ଧର (ଅଚଳ std::exception & exc) {
        release();
        ଫିଙ୍ଗ;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(ସତ୍ୟ) {
    release();
}

// ** backend interface

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_flash_attn_is_hmx_eligible(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * q,
    ଅଚଳ ସଂରଚନା ggml_tensor * k,
    ଅଚଳ ସଂରଚନା ggml_tensor * v,
    ଅଚଳ ସଂରଚନା ggml_tensor * sinks
) {
    ଯଦି (sess->n_hmx == 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (opt_fa_select < 2) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t DK = q->ne[0];
    ଅଚଳ uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    ଯଦି (DK % 8 != 0 || DV % 8 != 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    GGML_UNUSED(sinks);

    // Explicit force mode
    ଯଦି (opt_fa_select > 2) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଅଚଳ uint32_t M = q->ne[1];

    // Prefill or batched decode
    ଯଦି (M > 1) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଅଚଳ uint32_t n_head = q->ne[2];
    ଅଚଳ uint32_t n_kv_heads = k->ne[2];
    ଅଚଳ uint32_t G = n_kv_heads > 0 ? n_head / n_kv_heads : 1;
    ଅଚଳ uint32_t S = k->ne[1];
    ଅଚଳ uint32_t total_threads = sess->n_threads * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    ଯଦି (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < total_threads))) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    // Tile alignment for 32-row HMX tiles
    ଅଚଳ ତାର୍କିକ is_tile_aligned = (G > 0 && (32 % G == 0));
    ଯଦି (!is_tile_aligned) {
        ଯଦି (sess->n_threads >= 6) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଫେରନ୍ତୁ S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    ଯଦି (DK == 256) {
        s_cross = 4096;
    } ଅନ୍ୟଥା ଯଦି (DK <= 64) {
        s_cross = (sess->n_threads >= 8) ? 2048 : ((sess->n_threads >= 6) ? 768 : 512);
    } ଅନ୍ୟଥା {
        s_cross = (sess->n_threads >= 8) ? 1024 : 512;
    }

    ଫେରନ୍ତୁ S >= s_cross;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_flash_attn_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_fa_kernel_params * kparams
) {
    ଯଦି (opt_fa_select < 1) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    memset(kparams, 0, ଆକାର(*kparams));

    ଅଚଳ ସଂରଚନା ggml_tensor * q    = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * k    = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * v    = op->src[2];
    ଅଚଳ ସଂରଚନା ggml_tensor * mask = op->src[3];

    ଅଚଳ uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ଅଚଳ uint32_t neq1 = q->ne[1];  // n_tokens
    ଅଚଳ uint32_t neq2 = q->ne[2];  // n_heads

    ଅଚଳ uint32_t nek1 = k->ne[1];  // kv_len

    ଅଚଳ uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ଅଚଳ uint32_t DK = neq0;
    ଅଚଳ uint32_t DV = nev0;

    ଅଚଳ uint32_t n_kv_heads = k->ne[2];
    ଅଚଳ uint32_t G          = neq2 / n_kv_heads;

    ଦଶମିକ scale         = 1.0f;
    ଦଶମିକ max_bias      = 0.0f;
    ଦଶମିକ logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], ଆକାର(ଦଶମିକ));
    memcpy(&max_bias,      &op->op_params[1], ଆକାର(ଦଶମିକ));
    memcpy(&logit_softcap, &op->op_params[2], ଆକାର(ଦଶମିକ));

    ଯଦି (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->head_split = (opt_fa_head_split != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    ଅଚଳ uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    ଅଚଳ ସଂରଚନା ggml_tensor * sinks = op->src[4];
    ଯଦି (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        ଅଚଳ uint32_t DK_pad = hex_round_up(DK, 64);
        ଅଚଳ uint32_t DV_pad = hex_round_up(DV, 64);
        size_t Br = 0, Bc = 0;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
        ଯଦି (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            ଯଦି (logit_softcap == 0.0f) {
                kparams->scale = scale * (ଦଶମିକ) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } ଅନ୍ୟଥା {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (ଦଶମିକ) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);

            ଅଚଳ size_t row_vec_bytes = hex_align_up(Bc * ଆକାର(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            ଅଚଳ size_t m_ଧାଡ଼ି_bytes = hex_align_up(Bc * ଆକାର(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_ଧାଡ଼ି_bytes / ଆକାର(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(G);
            ଯଦି (mask) {
                kparams->src3_div2 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    ଅଚଳ size_t size_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    ଅଚଳ size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    ଅଚଳ size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), mask != nullptr, sinks != nullptr, n_head, sess->n_threads);

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(q->ne[3]/v->ne[3]);
    ଯଦି (mask) {
        kparams->src3_div2 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_flash_attn_ext(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * src2 = op->src[2];
    ଅଚଳ ସଂରଚନା ggml_tensor * src3 = op->src[3];
    ଅଚଳ ସଂରଚନା ggml_tensor * src4 = op->src[4];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    ଯଦି ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src3 && src3->type != GGML_TYPE_F16) {  // mask
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    ଯଦି (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ସଂରଚନା htp_fa_kernel_params kparams;
    ଯଦି (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_gated_delta_net(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଯଦି (opt_gdn_select < 1) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ସଂରଚନା ggml_tensor * q     = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * k     = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * v     = op->src[2];
    ଅଚଳ ସଂରଚନା ggml_tensor * g     = op->src[3];
    ଅଚଳ ସଂରଚନା ggml_tensor * beta  = op->src[4];
    ଅଚଳ ସଂରଚନା ggml_tensor * state = op->src[5];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst   = op;

    ଯଦି (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int64_t S_v      = v->ne[0];
    ଅଚଳ int64_t H        = v->ne[1];
    ଅଚଳ int64_t n_tokens = v->ne[2];
    ଅଚଳ int64_t n_seqs   = v->ne[3];
    ଅଚଳ int64_t K        = ggml_get_op_params_i32(op, 0);

    ଯଦି (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    ଯଦି (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t total_rows = (uint32_t) (H * n_seqs);
    ଅଚଳ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    ଅଚଳ ତାର୍କିକ can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    ଯଦି (can_use_hmx) {
        ସଂରଚନା htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        ଯଦି (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, ସତ୍ୟ, &n_heads_batch)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    } ଅନ୍ୟଥା {
        ସଂରଚନା htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_threads);
        ଯଦି (layout.total_bytes > sess->vtcm_size) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_matmul_is_hmx_eligible(
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ତାର୍କିକ is_matmul_id,
    ତାର୍କିକ is_batched
) {
    GGML_UNUSED(dst);

    ଯଦି (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00  = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01  = src0->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11  = src1->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12  = src1->ne[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    ଯଦି (ne01 % HTP_TENSOR_TILE_N_ROWS != 0 && is_matmul_id) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // HMX supports F16, F32, and repack quantized types.
    ଯଦି (!htp_is_hmx_weight_type((ggml_type) wtype)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    ଯଦି (ne00 % HTP_TENSOR_TILE_N_COLS != 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    ଅଚଳ ତାର୍କିକ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ଯଦି (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!is_matmul_id && is_batched && is_tiled) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // HMX assumes contiguous row-major layout.
    ଯଦି (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା m = is_matmul_id ? ne12 : ne11;
    ଯଦି (m <= HTP_MM_HMX_MIN_NROWS) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(dst);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_hmx_mm_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * act,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00_padded,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01_padded,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11_padded,
    ତାର୍କିକ is_matmul_id,
    ତାର୍କିକ is_batched,
    size_t src2_size,
    size_t vtcm_budget,
    ସଂରଚନା htp_mm_kernel_params * kparams
) {
    ଅଚଳ ତାର୍କିକ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା aligned_tile_size = is_tiled ? htp_tensor_get_tile_padded_size(wtype) : 0;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)sess->n_threads;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10 = act->ne[0];

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_for_solver = ne11;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    ଯଦି (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        ଅଚଳ ତାର୍କିକ dst_row_split = dst ? ggml_hexagon_can_row_partition(dst) : ସତ୍ୟ;
        ଅଚଳ ତାର୍କିକ act_row_split = ggml_hexagon_can_row_partition(act);
        ଯଦି (dst_row_split && act_row_split) {
            m_for_solver = (ne11 + (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) sess->mdev.count - 1) / (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) sess->mdev.count;
            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
        }
    }
    ଅଚଳ ତାର୍କିକ pipeline = is_matmul_id ? ମିଥ୍ୟା : htp_mm_hmx_pipeline(m_for_solver);

    ଅଚଳ ତାର୍କିକ is_batched_val = is_matmul_id ? ମିଥ୍ୟା : is_batched;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    ତାର୍କିକ use_grouped = ମିଥ୍ୟା;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_threads_selected = 0;

    ଯଦି (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && group_size > 1) {
        // Try grouped path first
        ଯଦି (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = ସତ୍ୟ;
        }
    }

    ଯଦି (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_id_rows = 0;
        ଯଦି (dst && is_matmul_id) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_experts = ne02 > 0 ? ne02 : 1;
            ଅଚଳ size_t total_expert_rows = (size_t) dst->ne[1] * dst->ne[2];
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_per_expert = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ((total_expert_rows + n_experts - 1) / n_experts);
            ଯଦି (sess->mdev.count > 1 && ggml_hexagon_can_row_partition(dst)) {
                m_per_expert = (m_per_expert + (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) sess->mdev.count - 1) / (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) sess->mdev.count;
            }
            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
        }
        ଅଚଳ uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
        ଯଦି (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
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
    kparams->div_n_act_threads = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(act_threads_selected);
    kparams->div_ne00_padded   = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne00_padded);
    kparams->vtcm_act_size     = 0;
    kparams->vtcm_bias_size    = (int32_t) src2_size;
    kparams->vtcm_dst_size     = 0;

    ଯଦି (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } ଅନ୍ୟଥା {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_hvx_mm_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype,
    ତାର୍କିକ is_tiled,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne03,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne13,
    ତାର୍କିକ is_matmul_id,
    ଅଚଳ size_t src2_row_size,
    size_t vtcm_budget,
    ସଂରଚନା htp_mm_kernel_params * kparams
) {
    ଯଦି (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ଫେରନ୍ତୁ;
    }

    kparams->n_hmx = 0;
    kparams->n_threads = sess->n_threads;

    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src1_nrows = ne11 * ne12 * ne13;

    ଯଦି (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_size = htp_tensor_get_tile_size(wtype);
        kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);

        ଅଚଳ ତାର୍କିକ k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        ଯଦି (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) sess->n_threads)
                                  ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);

            ସଂରଚନା htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            ପାଇଁ (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], kparams->act_row_size, 0, d, ସତ୍ୟ, ମିଥ୍ୟା
                );
                ଯଦି (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    ଭାଙ୍ଗ;
                }
            }
            ଯଦି (L.total_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                ଫେରନ୍ତୁ;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = 0;
            kparams->vtcm_dst_size  = L.dst_bytes;
            ଫେରନ୍ତୁ;
        } ଅନ୍ୟଥା {
            ତାର୍କିକ try_tiled = (k_align && opt_mm_select >= 1);
            ଯଦି (try_tiled) {
                kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                ଯଦି (src1_nrows < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } ଅନ୍ୟଥା {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                ସଂରଚନା htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                ପାଇଁ (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, ମିଥ୍ୟା, ମିଥ୍ୟା
                    );
                    ଯଦି (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        ଭାଙ୍ଗ;
                    }
                }

                uint32_t m_chunk = 0;
                ଯଦି (htp_mm_hvx_solve_vtcm_params(
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
                    ଫେରନ୍ତୁ;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            ଫେରନ୍ତୁ;
        }
    } ଅନ୍ୟଥା ଯଦି (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ଫେରନ୍ତୁ;
    } ଅନ୍ୟଥା ଯଦି (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        ସଂରଚନା htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ଯଦି (htp_mm_hvx_solve_vtcm_params(
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
            ଫେରନ୍ତୁ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ଫେରନ୍ତୁ;
    } ଅନ୍ୟଥା ଯଦି (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        ସଂରଚନା htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ଯଦି (htp_mm_hvx_solve_vtcm_params(
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
            ଫେରନ୍ତୁ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ଫେରନ୍ତୁ;
    } ଅନ୍ୟଥା ଯଦି (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        ସଂରଚନା htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ଯଦି (htp_mm_hvx_solve_vtcm_params(
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
            ଫେରନ୍ତୁ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ଫେରନ୍ତୁ;
    } ଅନ୍ୟଥା {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ଫେରନ୍ତୁ;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_matmul_params_impl(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ଅଚଳ size_t src2_row_size,
    ଅଚଳ size_t src2_size,
    ସଂରଚନା htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00 = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01 = src0->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02 = src0->ne[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne03 = src0->ne[3];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10 = src1->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11 = src1->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12 = src1->ne[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne13 = src1->ne[3];

    ଅଚଳ ତାର୍କିକ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11_padded = hex_round_up(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01_tiled  = hex_round_up(ne01, 32);

    ଅଚଳ ତାର୍କିକ is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    ଅଚଳ ତାର୍କିକ is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ଅଚଳ size_t vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    ତାର୍କିକ hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    ଯଦି (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        ଯଦି (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
            ଯାଅ finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_size, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne11);
    kparams->div_r2       = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
ସ୍ଥିର ତାର୍କିକ ggml_hexagon_rows_stride(ଅଚଳ int64_t * ne, ଅଚଳ size_t * nb, size_t * stride) {
    size_t s = 0, next = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i < GGML_MAX_DIMS; i++) {
        ଯଦି (ne[i] == 1) ଜାରି;
        ଯଦି (s == 0) { s = nb[i]; next = s * ne[i]; ଜାରି; }
        ଯଦି (nb[i] != next) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        next *= ne[i];
    }
    *stride = s ? s : nb[1];
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
ସ୍ଥିର ତାର୍କିକ ggml_hexagon_matmul_can_collapse(ଅଚଳ ସଂରଚନା ggml_tensor * src0, ଅଚଳ ସଂରଚନା ggml_tensor * src1, ଅଚଳ ସଂରଚନା ggml_tensor * dst) {
    size_t s1, sd;
    ଫେରନ୍ତୁ (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_size(src1->type) && ggml_hexagon_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_size(dst->type) && ggml_hexagon_rows_stride(dst->ne, dst->nb, &sd);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_matmul_add_can_collapse(
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * src2,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst
) {
    ଯଦି (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!src2) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ଅଚଳ int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    ଯଦି (src2_nrows == 1) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    size_t s2;
    ଫେରନ୍ତୁ src2->nb[0] == ggml_type_size(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_rows_stride(src2->ne, src2->nb, &s2);
}

ସ୍ଥିର ggml_tensor ggml_hexagon_tensor_collapse_rows(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    ଫେରନ୍ତୁ c;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_matmul_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_mm_kernel_params * kparams
) {
    ଯଦି (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ଅଚଳ ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        ଅଚଳ ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        ଫେରନ୍ତୁ;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_fused_matmul_add_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * src2,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_mm_kernel_params * kparams
) {
    ଯଦି (ggml_hexagon_matmul_add_can_collapse(src0, src1, src2, dst)) {
        ଅଚଳ ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        ଅଚଳ ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ଅଚଳ ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        ଅଚଳ ସଂରଚନା ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        ଅଚଳ size_t src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
        kparams->collapse = 1;
        ଫେରନ୍ତୁ;
    }
    ଅଚଳ size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_ଦ୍ୱିମିକ_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    uint32_t op,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_ଦ୍ୱିମିକ_kernel_params * kparams,
    ଅଚଳ ସଂରଚନା ggml_tensor * src2
) {
    memset(kparams, 0, ଆକାର(*kparams));

    ଅଚଳ size_t elem_size = ggml_type_size(src0->type);
    ଅଚଳ size_t src0_row_size = src0->ne[0] * elem_size;
    ଅଚଳ size_t src1_row_size = src1->ne[0] * elem_size;
    ଅଚଳ size_t dst_row_size  = dst->ne[0]  * elem_size;

    ଅଚଳ size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    ଅଚଳ size_t src1_row_size_aligned = hex_round_up(src1_row_size, 128);
    ଅଚଳ size_t dst_row_size_aligned  = hex_round_up(dst_row_size,  128);

    ଯଦି (!htp_ଦ୍ୱିମିକ_strides_safe(src0, src1, dst, elem_size)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ତାର୍କିକ is_add_id = op == HTP_OP_ADD_ID;
    ଅଚଳ ତାର୍କିକ is_scalar = !is_add_id && src1->ne[0] == 1;
    ଅଚଳ ତାର୍କିକ is_row_bcast = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    ଅଚଳ ତାର୍କିକ is_same_shape = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    ଅଚଳ ତାର୍କିକ is_complex   = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    ଅଚଳ ତାର୍କିକ is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    ଅଚଳ ତାର୍କିକ is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    ଯଦି (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        ଅଚଳ uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        ଅଚଳ uint32_t n_threads = sess->n_threads;
        ଅଚଳ uint32_t max_chunk_elems = 32768 / elem_size;
        ଅଚଳ uint32_t min_chunk_elems = 256;
        ଅଚଳ uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
        ଅଚଳ uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        ଅଚଳ uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

        kparams->kernel_type           = HTP_ଦ୍ୱିମିକ_KERNEL_CHUNKED;
        kparams->n_threads             = n_threads;
        kparams->rows_per_buffer       = 1;
        kparams->src0_row_size_aligned = src0_row_size_aligned;
        kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
        kparams->dst_row_size_aligned  = dst_row_size_aligned;
        kparams->src1_size             = 0;
        kparams->chunk_size            = chunk_size;
        kparams->chunk_bytes           = chunk_bytes;
        kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

        ସଂରଚନା htp_ଦ୍ୱିମିକ_vtcm_layout L;
        htp_ଦ୍ୱିମିକ_vtcm_layout_build(&L, kparams, sess->vtcm_size);
        ଯଦି (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        kparams->vtcm_size = L.total_bytes;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଗଣନା htp_ଦ୍ୱିମିକ_kernel_type kernel_type;
    size_t src1_size = 0;
    size_t src2_size = 0;

    ଯଦି (is_add_id) {
        GGML_ASSERT(src2 != nullptr);
        kernel_type = HTP_ଦ୍ୱିମିକ_KERNEL_ADD_ID;
        src1_size = hex_round_up(src1->ne[1] * src1_row_size_aligned, 128);
        src2_size = hex_round_up(ggml_nelements(src2) * ଆକାର(int32_t), 128);
        ଯଦି (src1_size + src2_size > sess->vtcm_size) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    } ଅନ୍ୟଥା ଯଦି (is_row_bcast) {
        kernel_type = HTP_ଦ୍ୱିମିକ_KERNEL_ROW_BCAST;
        src1_size = src1_row_size_aligned;
    } ଅନ୍ୟଥା ଯଦି (is_scalar) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size, 128);
        ଯଦି (src1_size > sess->vtcm_size) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        kernel_type = HTP_ଦ୍ୱିମିକ_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } ଅନ୍ୟଥା ଯଦି (is_same_shape) {
        kernel_type = HTP_ଦ୍ୱିମିକ_KERNEL_SAME_SHAPE;
    } ଅନ୍ୟଥା ଯଦି (is_complex) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        ଯଦି (src1_size > sess->vtcm_size) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        kernel_type = HTP_ଦ୍ୱିମିକ_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } ଅନ୍ୟଥା {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        ଯଦି (src1_size > sess->vtcm_size) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        kernel_type = HTP_ଦ୍ୱିମିକ_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_threads = sess->n_threads;
    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;
    kparams->src1_size             = src1_size;
    kparams->src2_size             = src2_size;

    ଅଚଳ ତାର୍କିକ src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    ଅଚଳ ତାର୍କିକ src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    ଅଚଳ ତାର୍କିକ dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    ଅଚଳ ତାର୍କିକ src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    ଅଚଳ ତାର୍କିକ src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    ଅଚଳ ତାର୍କିକ dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    ତାର୍କିକ split_at_ne01 = ମିଥ୍ୟା;
    ଯଦି (src0->ne[2] > 1) {
        ଯଦି (kernel_type == HTP_ଦ୍ୱିମିକ_KERNEL_SAME_SHAPE) {
            split_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } ଅନ୍ୟଥା ଯଦି (kernel_type == HTP_ଦ୍ୱିମିକ_KERNEL_ROW_BCAST) {
            split_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } ଅନ୍ୟଥା {
            split_at_ne01 = ସତ୍ୟ;
        }
    }
    kparams->split_at_ne01 = split_at_ne01 ? 1 : 0;

    ତାର୍କିକ split_at_ne02 = ମିଥ୍ୟା;
    ଯଦି (src0->ne[3] > 1) {
        ଯଦି (kernel_type == HTP_ଦ୍ୱିମିକ_KERNEL_SAME_SHAPE) {
            split_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } ଅନ୍ୟଥା ଯଦି (kernel_type == HTP_ଦ୍ୱିମିକ_KERNEL_ROW_BCAST) {
            split_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } ଅନ୍ୟଥା {
            split_at_ne02 = ସତ୍ୟ;
        }
    }
    kparams->split_at_ne02 = split_at_ne02 ? 1 : 0;

    ସଂରଚନା htp_ଦ୍ୱିମିକ_vtcm_layout L;
    htp_ଦ୍ୱିମିକ_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    ଯଦି (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {
        ଯଦି (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            ଅଚଳ uint32_t total_elems = (uint32_t) ggml_nelements(src0);
            ଅଚଳ uint32_t n_threads = sess->n_threads;
            ଅଚଳ uint32_t max_chunk_elems = 32768 / elem_size;
            ଅଚଳ uint32_t min_chunk_elems = 256;
            ଅଚଳ uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
            ଅଚଳ uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
            ଅଚଳ uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

            kparams->kernel_type           = HTP_ଦ୍ୱିମିକ_KERNEL_CHUNKED;
            kparams->n_threads             = n_threads;
            kparams->rows_per_buffer       = 1;
            kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
            kparams->src1_size             = 0;
            kparams->chunk_size            = chunk_size;
            kparams->chunk_bytes           = chunk_bytes;
            kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

            htp_ଦ୍ୱିମିକ_vtcm_layout_build(&L, kparams, sess->vtcm_size);
            ଯଦି (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            kparams->vtcm_size = L.total_bytes;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    kparams->rows_per_buffer = L.rows_per_buffer;
    kparams->vtcm_size       = L.total_bytes;

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_unary_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    uint32_t op,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));

    ଅଚଳ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଅଚଳ uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    ଅଚଳ size_t elem_size = ggml_type_size(src0->type);

    ଅଚଳ size_t src0_data_row_size = src0->ne[0] * elem_size;
    ଅଚଳ size_t dst_data_row_size  = dst->ne[0]  * ggml_type_size(dst->type);

    ଅଚଳ size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    ଅଚଳ size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = 0;
    ତାର୍କିକ broadcast_weight = ମିଥ୍ୟା;

    ଯଦି (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_data_row_size = src1->ne[0] * ggml_type_size(src1->type);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->broadcast_weight      = broadcast_weight;

    ସଂରଚନା htp_unary_vtcm_layout L;
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

    ଅଚଳ uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(tiles_per_row);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_get_rows_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_get_rows_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));

    ଅଚଳ uint32_t ne00 = src0->ne[0];
    ଅଚଳ uint32_t ne02 = src0->ne[2];
    ଅଚଳ uint32_t ne03 = src0->ne[3];

    ଅଚଳ uint32_t ne10 = src1->ne[0];
    ଅଚଳ uint32_t ne11 = src1->ne[1];
    ଅଚଳ uint32_t ne12 = src1->ne[2];
    ଅଚଳ uint32_t nr = ne10 * ne11 * ne12;

    ଅଚଳ ତାର୍କିକ tiled = ggml_hexagon_tensor_is_tiled(src0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    ଯଦି (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } ଅନ୍ୟଥା ଯଦି (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } ଅନ୍ୟଥା {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    ଅଚଳ uint32_t total_tasks = nr;

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, total_tasks);

    ସଂରଚନା htp_get_rows_vtcm_layout vtcm_layout = {};
    ଯେତେବେଳେ (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, kparams->n_threads);
        ଯଦି (vtcm_layout.total_bytes <= sess->vtcm_size) {
            ଭାଙ୍ଗ;
        }
        --kparams->n_threads;
    }

    ଯଦି (kparams->n_threads == 0 && total_tasks > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, 1);
    }

    kparams->vtcm_size = (total_tasks == 0) ? 0 : vtcm_layout.total_bytes;
    kparams->tasks_per_thread = kparams->n_threads > 0 ? (total_tasks + kparams->n_threads - 1) / kparams->n_threads : 0;
    kparams->total_tasks = total_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne10) : fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne10 * ne11) : fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne02) : fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne03) : fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ{0, 0};
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_set_rows_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0, //ମୂଲ୍ୟଗୁଡ଼ିକ
    ଅଚଳ ସଂରଚନା ggml_tensor * src1, // indices
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,  // destination
    ସଂରଚନା htp_set_rows_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));

    ଅଚଳ uint32_t nr = src0->ne[1];

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, nr);
    kparams->tasks_per_thread = (nr + kparams->n_threads - 1) / kparams->n_threads;
    kparams->total_tasks = nr;

    kparams->div_ne11 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(src1->ne[2]);
    kparams->div_tasks_per_thread = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(kparams->tasks_per_thread);
    kparams->div_ne02 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(src0->ne[2]);

    ସଂରଚନା htp_set_rows_vtcm_layout vtcm_layout;
    htp_set_rows_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_threads);
    kparams->vtcm_size = vtcm_layout.total_bytes;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_softmax_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];

    ଅଚଳ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଅଚଳ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);

    ଦଶମିକ scale = 1.0f;
    ଦଶମିକ max_bias = 0.0f;
    memcpy(&scale,    &op->op_params[0], ଆକାର(ଦଶମିକ));
    memcpy(&max_bias, &op->op_params[1], ଆକାର(ଦଶମିକ));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    ଅଚଳ uint32_t n_head = src0->ne[2];
    ଅଚଳ uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    kparams->n_head      = n_head;
    kparams->n_head_log2 = n_head_log2;

    ଯଦି (max_bias > 0.0f && n_head_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_head_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    } ଅନ୍ୟଥା {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    ଅଚଳ uint32_t ne00 = src0->ne[0];
    ଅଚଳ uint32_t ne10 = src1 ? src1->ne[0] : 1;

    ସଂରଚନା htp_softmax_vtcm_layout layout;
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
    ଯଦି (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } ଅନ୍ୟଥା ଯଦି (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F16;
    } ଅନ୍ୟଥା {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    ଯଦି (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(src0->ne[1]);
    ଯଦି (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(src0->ne[2]);
    ଅଚଳ uint32_t ne12 = src1 ? src1->ne[2] : 1;
    ଅଚଳ uint32_t ne13 = src1 ? src1->ne[3] : 1;
    ଯଦି (ne12 > 0) kparams->div_ne12 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne12);
    ଯଦି (ne13 > 0) kparams->div_ne13 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne13);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_rope_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src2 = op->src[2];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଅଚଳ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଅଚଳ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ଅଚଳ uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ସଂରଚନା htp_rope_vtcm_layout layout;
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

    ଯଦି (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(dst->ne[1]);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_ssm_conv_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * src1,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));

    ଅଚଳ uint32_t d_conv  = (uint32_t) src1->ne[0];
    ଅଚଳ uint32_t d_inner = (uint32_t) src0->ne[1];
    ଅଚଳ uint32_t n_t     = (uint32_t) dst->ne[1];
    ଅଚଳ uint32_t n_s     = (uint32_t) dst->ne[2];
    ଅଚଳ uint32_t ncs     = (uint32_t) src0->ne[0];

    ଅଚଳ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    ଅଚଳ uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
    ଅଚଳ uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);

    ଅଚଳ uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * ଆକାର(ଦଶମିକ), 128) + 128;
    ଅଚଳ uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * ଆକାର(ଦଶମିକ), 128);
    ଅଚଳ uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

    uint32_t vtcm_src0_per_thread = 0;
    uint32_t vtcm_dst_per_thread  = 0;

    ଯଦି (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;

        ଅଚଳ uint32_t src0_tile_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * ଆକାର(ଦଶମିକ), 128);
        ଅଚଳ uint32_t src0_T_bytes        = hex_round_up(d_conv * d_inner_per_thread * ଆକାର(ଦଶମିକ), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw_bytes + src0_T_bytes;

        ଅଚଳ uint32_t dst_tile_bytes = hex_round_up(d_inner_per_thread * ଆକାର(ଦଶମିକ), 128);
        vtcm_dst_per_thread = 2 * dst_tile_bytes;
    } ଅନ୍ୟଥା {
        ଅଚଳ size_t vtcm_budget = (sess->vtcm_size > 0 ? sess->vtcm_size / n_threads : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        ଅଚଳ uint32_t src0_block_T    = hex_round_up(ncs * 32 * ଆକାର(ଦଶମିକ), 128);
        ଅଚଳ size_t   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
        ଅଚଳ size_t   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);

        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
        target_max_tile = (std::max)(target_max_tile, 32u);
        target_max_tile = (std::min)(target_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * ଆକାର(ଦଶମିକ)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        ଯଦି (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        ଯଦି (d_inner_tile > target_max_tile) {
            d_inner_tile = target_max_tile;
        }
        ଯଦି (d_inner_tile > d_inner_per_thread) {
            d_inner_tile = d_inner_per_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        ଅଚଳ uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * ଆକାର(ଦଶମିକ), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw + src0_block_T;

        vtcm_dst_per_thread = 2 * hex_round_up(d_inner_tile * n_t * ଆକାର(ଦଶମିକ), 128);
    }

    kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
    kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
    kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;

    kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
    kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
    kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
    kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_gated_delta_net_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));

    ଅଚଳ ସଂରଚନା ggml_tensor * q     = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * k     = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * v     = op->src[2];
    ଅଚଳ ସଂରଚନା ggml_tensor * g     = op->src[3];
    ଅଚଳ ସଂରଚନା ggml_tensor * state = op->src[5];

    ଅଚଳ uint32_t S_v      = (uint32_t) v->ne[0];
    ଅଚଳ uint32_t H        = (uint32_t) v->ne[1];
    ଅଚଳ uint32_t n_tokens = (uint32_t) v->ne[2];
    ଅଚଳ uint32_t n_seqs   = (uint32_t) v->ne[3];
    ଅଚଳ uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    ଅଚଳ uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    ଅଚଳ uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    ଅଚଳ uint32_t total_rows = H * n_seqs;
    ଅଚଳ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    ଅଚଳ ତାର୍କିକ can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    ସଂରଚନା htp_gdn_hmx_vtcm_layout hmx_layout;
    ସଂରଚନା htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    ଯଦି (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, ସତ୍ୟ, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_size      = HTP_GDN_CHUNK_SIZE;
        kparams->n_chunks        = (n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_size       = (uint32_t) hmx_layout.total_bytes;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_bytes;
        kparams->vtcm_per_thread = (uint32_t) (hmx_layout.total_bytes / (n_threads > 0 ? n_threads : 1));
    } ଅନ୍ୟଥା {
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
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / ଆକାର(ଦଶମିକ));
    kparams->state_size_per_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((ଦଶମିକ) S_v);

    ଯଦି (H > 0)                  kparams->div_H         = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(H);
    ଯଦି (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) q->ne[1]);
    ଯଦି (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) k->ne[1]);
    ଯଦି (rq3 > 0)                kparams->div_rq3       = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(rq3);
    ଯଦି (rk3 > 0)                kparams->div_rk3       = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(rk3);
    ଯଦି (kparams->n_threads > 0) kparams->div_n_threads = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(kparams->n_threads);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_sort_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ତାର୍କିକ is_ଶୀର୍ଷ_k,
    ସଂରଚନା htp_sort_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଅଚଳ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଅଚଳ uint32_t ne00       = src0->ne[0];
    ଅଚଳ uint32_t k          = dst->ne[0];

    int32_t order = GGML_SORT_ORDER_DESC;
    ଯଦି (!is_ଶୀର୍ଷ_k) {
        order = ((ଅଚଳ int32_t *) op->op_params)[0];
    }

    ଅଚଳ uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ଅଚଳ size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    ସଂରଚନା htp_sort_vtcm_layout layout;
    ତାର୍କିକ ok = htp_sort_solve_layout(&layout, ne00, total_rows, k, n_threads_max, vtcm_budget, is_ଶୀର୍ଷ_k);
    GGML_ASSERT(ok);

    kparams->n_threads         = (int32_t) layout.n_threads;
    kparams->total_rows        = (int32_t) total_rows;
    kparams->row_start         = 0;
    kparams->row_end           = (int32_t) total_rows;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->order             = order;
    kparams->is_ଶୀର୍ଷ_k          = is_ଶୀର୍ଷ_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_size         = (int32_t) layout.total_bytes;
    kparams->phase1_slot_size  = (int32_t) layout.phase1_slot_size;
    kparams->merge_ମୂଲ୍ୟଗୁଡ଼ିକ_off  = (int32_t) layout.merge_ମୂଲ୍ୟଗୁଡ଼ିକ_off;
    kparams->merge_indices_off = (int32_t) layout.merge_indices_off;
    kparams->merge_elems       = (int32_t) layout.merge_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    ଅଚଳ uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    ଅଚଳ int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    ଯଦି (numer_hi < 0) {
        *ox_hi = 0;
    } ଅନ୍ୟଥା {
        ଅଚଳ uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    ଯଦି (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_pool_shape_fits(
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst
) {
    ଫେରନ୍ତୁ ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_pool_2d_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0,
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,
    ସଂରଚନା htp_pool_2d_kernel_params * kparams,
    ତାର୍କିକ is_pool_1d
) {
    memset(kparams, 0, ଆକାର(*kparams));

    ଅଚଳ uint32_t src_x = (uint32_t) src0->ne[0];
    ଅଚଳ uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    ଅଚଳ uint32_t dst_x = (uint32_t) dst->ne[0];
    ଅଚଳ uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    ଅଚଳ uint32_t src_plane_bytes = src_x * src_y * ଆକାର(ଦଶମିକ);
    ଅଚଳ uint32_t dst_plane_bytes = dst_x * dst_y * ଆକାର(ଦଶମିକ);
    ଅଚଳ uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    ଅଚଳ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    ସଂରଚନା htp_pool_vtcm_layout layout;
    ଅଚଳ ତାର୍କିକ ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, sess->vtcm_size);
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
    kparams->avg_divide_ଗଣନା = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    ଅଚଳ ତାର୍କିକ narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_path = (kparams->fast_path && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (ଦଶମିକ) (kparams->kernel_x * kparams->kernel_y);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_concat_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (!src0 || !src1 || !dst) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dim = ((ଅଚଳ int32_t *) op->op_params)[0];
    ଯଦି (dim < 0 || dim >= GGML_MAX_DIMS) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    kparams->dim = dim;

    ଯଦି (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src0->type != dst->type || src1->type != dst->type) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t type_size = ggml_type_size(dst->type);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା d = 0; d < GGML_MAX_DIMS; d++) {
        ଅଚଳ int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        ଯଦି (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଅଚଳ uint32_t max_gather_ne0 = 128 / type_size;
    ଅଚଳ uint32_t total_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    ଅଚଳ ତାର୍କିକ b_dense = (src1->nb[0] == type_size);
    ଅଚଳ ତାର୍କିକ gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_size &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_size
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_size);

    ଯଦି (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_size && dst->nb[1] == (size_t) dst->ne[0] * type_size &&
        src0->nb[0] == type_size &&
        (src1->nb[0] == type_size || src1->nb[1] == type_size) &&
        gather_strides_ok &&
        htp_gather_rows_tensor_outer_contiguous(dst) &&
        htp_gather_rows_tensor_outer_contiguous(src0) &&
        htp_gather_rows_tensor_outer_contiguous(src1)) {

        ଯଦି (htp_gather_rows_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_size, total_rows,
                                         sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ROWS;
            kparams->n_threads   = (uint8_t) sess->n_threads;
            kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
    }

    ଅଚଳ ତାର୍କିକ dma_strides_ok = (src0->nb[0] == type_size && src1->nb[0] == type_size && dst->nb[0] == type_size);

    ଯଦି (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_threads   = 1;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଅଚଳ ତାର୍କିକ is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    ଅଚଳ ତାର୍କିକ is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    ଅଚଳ ତାର୍କିକ transposed_rows_ok = (src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size);

    ଯଦି (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        ଅଚଳ uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 8;
        ସଂରଚନା htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_size, n_threads);

        ଯଦି (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_threads                         = n_threads;
        kparams->vtcm_size                         = layout.total_bytes;
        kparams->u.transposed.spad0_size_per_thread = layout.src0_spad_size_per_thread;
        kparams->u.transposed.spad1_size_per_thread = layout.src1_spad_size_per_thread;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_cpy_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (!src0 || !dst) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int64_t nelem_src = ggml_nelements(src0);
    ଅଚଳ int64_t nelem_dst = ggml_nelements(dst);
    ଯଦି (nelem_src != nelem_dst || nelem_src < 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t src_type_size = ggml_type_size(src0->type);
    ଅଚଳ uint32_t dst_type_size = ggml_type_size(dst->type);

    kparams->src0_type_size = (uint8_t) src_type_size;
    kparams->dst_type_size  = (uint8_t) dst_type_size;
    kparams->total_elems    = (uint32_t) nelem_src;

    ଯଦି (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଯଦି (nelem_src == 1) {
        ଯଦି (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଯଦି ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ତାର୍କିକ sametype = (src0->type == dst->type);

    ତାର୍କିକ same_extents = ସତ୍ୟ;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା d = 0; d < GGML_MAX_DIMS; d++) {
        ଯଦି (src0->ne[d] != dst->ne[d]) {
            same_extents = ମିଥ୍ୟା;
            ଭାଙ୍ଗ;
        }
    }

    ଅଚଳ ତାର୍କିକ transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_size) || (dst->nb[0] != dst_type_size) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_size) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_size);
    ଅଚଳ ତାର୍କିକ sameshape  = same_extents && !transposed;

    ଅଚଳ ତାର୍କିକ src_is_contiguous = ggml_is_contiguous(src0);
    ଅଚଳ ତାର୍କିକ dst_is_contiguous = ggml_is_contiguous(dst);

    ଯଦି (sametype) {
        ଯଦି (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        ଅଚଳ uint32_t max_gather_ne0 = 128 / src_type_size;
        ଅଚଳ uint32_t total_rows = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        ଯଦି ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_size && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_size &&
            total_rows > 1 &&
            htp_gather_rows_tensor_outer_contiguous(src0)) {

            ଯଦି (htp_gather_rows_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, ମିଥ୍ୟା,
                                             src_type_size, total_rows,
                                             sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ROWS;
                kparams->n_threads   = (uint8_t) sess->n_threads;
                kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        }

        ଯଦି (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_threads   = sess->n_threads > 0 ? (uint8_t) sess->n_threads : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଯଦି (!sameshape) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ତାର୍କିକ valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    ଯଦି (!valid_conversion) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 4;
    ସଂରଚନା htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_size, dst_type_size, n_threads);

    ଯଦି (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
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
    kparams->u.convert.div_ne01      = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ((uint32_t) (src0->ne[2] * src0->ne[1]));

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_fused_mmnx_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0, // W0
    ଅଚଳ ସଂରଚନା ggml_tensor * act,  // x
    int32_t n_weights,
    ସଂରଚନା htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));
    kparams->n_threads = sess->n_threads;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00 = src0->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01 = src0->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02 = src0->ne[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne03 = src0->ne[3];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10 = act->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11 = act->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12 = act->ne[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne13 = act->ne[3];

    ଅଚଳ ତାର୍କିକ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11_padded = hex_round_up(ne11, 32);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01_tiled  = hex_round_up(ne01, 32);

    ଅଚଳ size_t vtcm_budget = sess->vtcm_size;
    ଅଚଳ ତାର୍କିକ is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ତାର୍କିକ hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    ଯଦି (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, nullptr, ମିଥ୍ୟା, is_batched)) {
        ଯଦି (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, ମିଥ୍ୟା, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            ଯାଅ finalize;
        }
    }

    ଯଦି (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ଫେରନ୍ତୁ;
    }

    {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_nrows = ne11 * ne12 * ne13;
        ଅଚଳ size_t act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
        ଅଚଳ size_t src0_row_size = src0->nb[1];

        ଅଚଳ uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        ପାଇଁ (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            ସଂରଚନା htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
                0, src0_row_size, act_row_size, 0, d, ମିଥ୍ୟା, ସତ୍ୟ
            );
            ଯଦି (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                ଭାଙ୍ଗ;
            }
        }

        ସଂରଚନା htp_mm_hvx_vtcm_layout L;
        ତାର୍କିକ try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
            0, src0_row_size, act_row_size, 0, best_n_prefetch, ମିଥ୍ୟା, ସତ୍ୟ
        );

        ଯଦି (try_tiled && L.total_bytes <= sess->vtcm_size) {
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
        } ଅନ୍ୟଥା {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            ଫେରନ୍ତୁ;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne11);
    kparams->div_r2       = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(ne12);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_precompute_fused_mmidnx_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * src0, // W0
    ଅଚଳ ସଂରଚନା ggml_tensor * act,  // x
    ଅଚଳ ସଂରଚନା ggml_tensor * dst,  // dst0
    int32_t n_weights,
    ସଂରଚନା htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_op_mul_mat_use_fwht(ଅଚଳ ସଂରଚନା ggml_tensor * dst) {
    ଯଦି (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = dst->src[1];

    ଅଚଳ int64_t n = src1->ne[0];
    ଯଦି (n < 32 || (n & (n - 1)) != 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_rows(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_mul_mat(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * dst) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = dst->src[1];

    ଯଦି (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        ଯଦି (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ସଂରଚନା htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, nullptr, dst, &ukparams);
        ଫେରନ୍ତୁ ukparams.vtcm_row_per_thread > 0 && (size_t) ukparams.vtcm_size <= sess->vtcm_size;
    }

    ଯଦି (dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q2_K: {
            ଯଦି (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଅଚଳ int64_t qk = ggml_blck_size(src0->type);
            ଯଦି (src0->ne[0] % qk) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଯଦି (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଭାଙ୍ଗ;
        }

        ସ୍ଥିତି GGML_TYPE_F16:
            ଯଦି (src0->nb[1] < src0->nb[0]) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_TYPE_F32:
            ଯଦି (src1->type != GGML_TYPE_F32) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (src0->nb[1] < src0->nb[0]) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_TYPE_BF16:
            ଯଦି (src0->nb[1] < src0->nb[0]) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଭାଙ୍ଗ;

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ସଂରଚନା htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    ଯଦି (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_mul_mat_id(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * src2 = op->src[2];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
            ଯଦି (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଯଦି (src0->ne[0] % ggml_blck_size(src0->type) != 0) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_TYPE_F16:
        ସ୍ଥିତି GGML_TYPE_F32:
        ସ୍ଥିତି GGML_TYPE_BF16:
            ଯଦି (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଯଦି (src0->ne[0] % 32 != 0) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଭାଙ୍ଗ;

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ସଂରଚନା htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    ଯଦି (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_ଦ୍ୱିମିକ(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src0->type == GGML_TYPE_F32) {
        ଯଦି (src1->type != GGML_TYPE_F32) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (dst->type != GGML_TYPE_F32) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }
    ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16) {
        ଯଦି (src1->type != GGML_TYPE_F16) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (dst->type != GGML_TYPE_F16) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }
    ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ggml_are_same_shape(src0, dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ସଂରଚନା htp_ଦ୍ୱିମିକ_kernel_params kparams;
    ଫେରନ୍ତୁ ggml_hexagon_precompute_ଦ୍ୱିମିକ_params(sess, op_remap_to_htp(op), src0, src1, dst, &kparams);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_add_id(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * src2 = op->src[2];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (!src2) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ggml_are_same_shape(src0, dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src0->nb[0] != ଆକାର(ଦଶମିକ) || src1->nb[0] != ଆକାର(ଦଶମିକ) || dst->nb[0] != ଆକାର(ଦଶମିକ) || src2->nb[0] != ଆକାର(int32_t)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // REVISIT: add support for non-contiguous tensors
    ଯଦି (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src2->nb[1] < src2->ne[0] * ଆକାର(int32_t)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ସଂରଚନା htp_ଦ୍ୱିମିକ_kernel_params kparams;
    ଫେରନ୍ତୁ ggml_hexagon_precompute_ଦ୍ୱିମିକ_params(sess, HTP_OP_ADD_ID, src0, src1, dst, &kparams, src2);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_unary(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (dst->type != src0->type) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ggml_is_contiguous_rows(src0)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    ଯଦି (src0->type == GGML_TYPE_F16) {
        ବାଛ (op->op) {
            ସ୍ଥିତି GGML_OP_NORM:
            ସ୍ଥିତି GGML_OP_RMS_NORM:
            ସ୍ଥିତି GGML_OP_L2_NORM:
            ସ୍ଥିତି GGML_OP_SCALE:
            ସ୍ଥିତି GGML_OP_CLAMP:
            ସ୍ଥିତି GGML_OP_SQR:
            ସ୍ଥିତି GGML_OP_SQRT:
            ସ୍ଥିତି GGML_OP_LOG:
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_UNARY:
                ଯଦି (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_STEP) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଯଦି (!ggml_are_same_shape(src0, dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // dst must be contiguous; src0 may be non-contiguous
    ଯଦି (!ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_sum(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src0->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_sum_rows(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src0->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // TODO: add support for non-contigiuos tensors
    ଯଦି (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_argmax(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src0->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (dst->type != GGML_TYPE_I32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_pool_2d(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ int32_t * params = op->op_params;

    ଯଦି (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ggml_hexagon_pool_shape_fits(src0, op)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int32_t kernel_x = params[1];
    ଅଚଳ int32_t kernel_y = params[2];
    ଅଚଳ int32_t stride_x = params[3];
    ଅଚଳ int32_t stride_y = params[4];
    ଅଚଳ int32_t pad_x    = params[5];
    ଅଚଳ int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    ଯଦି (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    ଅଚଳ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    ସଂରଚନା htp_pool_vtcm_layout layout;
    ଫେରନ୍ତୁ htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_threads, sess->vtcm_size);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_pool_1d(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ int32_t * params = op->op_params;

    ଯଦି (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ggml_hexagon_pool_shape_fits(src0, op)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int32_t kernel = params[1];
    ଅଚଳ int32_t stride = params[2];
    ଅଚଳ int32_t pad    = params[3];
    ଯଦି (kernel <= 0 || stride <= 0 || pad < 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ଅଚଳ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    ସଂରଚନା htp_pool_vtcm_layout layout;
    ଫେରନ୍ତୁ htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_threads, sess->vtcm_size);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_activations(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (dst->type != src0->type) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous_1(src0)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src1) {
        ଯଦି (src1->type != src0->type) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (!ggml_are_same_shape(src0, src1)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (!ggml_is_contiguous_1(src1)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_softmax(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * src2 = op->src[2];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src2) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;  // FIXME: add support for sinks
    }

    ଯଦି (src0->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src0->ne[2] > 512) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src1) {
        ଯଦି (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (src0->ne[0] != src1->ne[0]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (src1->ne[1] < src0->ne[1]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (src0->ne[2] % src1->ne[2] != 0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (src0->ne[3] % src1->ne[3] != 0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଯଦି (src1) {
        ଯଦି (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    } ଅନ୍ୟଥା {
        ଯଦି (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    ଅଚଳ int64_t ne0 = src0->ne[0];
    ଯଦି (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    ଯଦି (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଅଚଳ uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ସଂରଚନା htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != nullptr, src1 && src1->type == GGML_TYPE_F16, n_threads);
    ଯଦି (layout.total_bytes > sess->vtcm_size) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_set_rows(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0]; //ମୂଲ୍ୟଗୁଡ଼ିକ
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1]; // indices
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    ଯଦି (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src0->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_get_rows(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0]; //ମୂଲ୍ୟଗୁଡ଼ିକ
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1]; // indices
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src0->type == dst->type) {
        ଯଦି (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_I32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    } ଅନ୍ୟଥା ଯଦି (dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    ଯଦି (ggml_is_empty(op)) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ସଂରଚନା htp_get_rows_kernel_params kparams;
    ggml_hexagon_precompute_get_rows_params(sess, src0, src1, dst, &kparams);
    ଯଦି (kparams.n_threads == 0 || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_argsort(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0]; //ମୂଲ୍ୟଗୁଡ଼ିକ
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;         // indices

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଅଚଳ uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ଅଚଳ size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    ସଂରଚନା htp_sort_vtcm_layout layout;
    ଯଦି (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, ମିଥ୍ୟା)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_ଶୀର୍ଷ_k(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0]; //ମୂଲ୍ୟଗୁଡ଼ିକ
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;         // indices

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଅଚଳ uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ଅଚଳ size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    ସଂରଚନା htp_sort_vtcm_layout layout;
    ଯଦି (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, ସତ୍ୟ)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_rope(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * src2 = op->src[2];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (!ggml_are_same_shape(src0, dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src0->ne[0] <= 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଯଦି (src0_nrows == 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int32_t * op_params = &op->op_params[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims = op_params[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mode   = op_params[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    ଯଦି (n_dims < 0 || n_dims % 2 != 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    ଯଦି (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଦଶମିକ freq_base;
    memcpy(&freq_base, op_params + 5, ଆକାର(ଦଶମିକ));
    ଯଦି (freq_base < 0.0f) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ତାର୍କିକ is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    ଯଦି (mode == GGML_ROPE_TYPE_VISION) {
        ଯଦି (n_dims != (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (src0->ne[0] / 2) || n_offs != 0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଯଦି (is_mrope) {
        ଅଚଳ int32_t * sections = op_params + 11;
        ଯଦି (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଅଚଳ int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    ଯଦି (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src2) {
        ଯଦି (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (src2->ne[0] < (n_dims / 2)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    ଯଦି (src0->nb[0] != ଆକାର(ଦଶମିକ) || dst->nb[0] != ଆକାର(ଦଶମିକ)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src0->nb[1] < src0->ne[0] * ଆକାର(ଦଶମିକ) || dst->nb[1] < dst->ne[0] * ଆକାର(ଦଶମିକ)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ଅଚଳ uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ସଂରଚନା htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    ଯଦି (layout.total_bytes > sess->vtcm_size) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_ssm_conv(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    // Only support FP32 for now
    ଯଦି (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Check IO tensor shapes and dims
    ଯଦି (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା; // src0 should be effectively 3D
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_conv  = src1->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା d_inner = src0->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_t = dst->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_s = dst->ne[2];

    ଯଦି (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src0->nb[0] != ଆକାର(ଦଶମିକ) || src1->nb[0] != ଆକାର(ଦଶମିକ) || dst->nb[0] != ଆକାର(ଦଶମିକ)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (src0->nb[1] != src0->ne[0] * ଆକାର(ଦଶମିକ) || src1->nb[1] != src1->ne[0] * ଆକାର(ଦଶମିକ)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ସଂରଚନା htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    ଯଦି ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ im2col_3d_calc_output_size(uint32_t ନିବେଶ, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    ଯଦି (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int64_t numerator = (int64_t) ନିବେଶ + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    ଯଦି (numerator < 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint64_t size = (uint64_t) (numerator / stride) + 1;
    ଯଦି (size == 0 || size > UINT32_MAX) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    *output = (uint32_t) size;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_im2col_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (!src0 || !src1 || !dst) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src1->nb[0] != ଆକାର(ଦଶମିକ) || !ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int32_t * op_params = (ଅଚଳ int32_t *) op->op_params;
    ଅଚଳ int32_t s0 = op_params[0];
    ଅଚଳ int32_t s1 = op_params[1];
    ଅଚଳ int32_t p0 = op_params[2];
    ଅଚଳ int32_t p1 = op_params[3];
    ଅଚଳ int32_t d0 = op_params[4];
    ଅଚଳ int32_t d1 = op_params[5];
    ଅଚଳ ତାର୍କିକ is_2D = (op_params[6] == 1);

    ଯଦି (s0 <= 0 || d0 <= 0 || p0 < 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    ଅଚଳ uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    ଅଚଳ uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    ଅଚଳ uint32_t IW = (uint32_t) src1->ne[0];
    ଅଚଳ uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    ଅଚଳ uint32_t KW = (uint32_t) src0->ne[0];
    ଅଚଳ uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    ଅଚଳ uint32_t OW = (uint32_t) dst->ne[1];

    ଯଦି (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    ଅଚଳ uint64_t total_rows_64   = (uint64_t) N * OH;
    ଯଦି (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ uint32_t patch_stride = (uint32_t) patch_stride_64;
    ଅଚଳ uint32_t total_rows   = (uint32_t) total_rows_64;

    ଯଦି (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (is_2D && dst->ne[2] != OH) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? ଆକାର(ggml_fp16_t) : ଆକାର(ଦଶମିକ);

    ତାର୍କିକ is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    ଯଦି (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    ଅଚଳ uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    ଅଚଳ uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    ଅଚଳ uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    ଯଦି (is_patchembed) {
        ଅଚଳ uint64_t src_row_bytes_raw = (uint64_t) IC * KH * IW * ଆକାର(ଦଶମିକ);
        ଅଚଳ uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        ଯଦି (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            ଅଚଳ uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            ଅଚଳ uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            ସଂରଚନା htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            ଯଦି (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        }
    }

    ପାଇଁ (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ଅଚଳ uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        ଯଦି (Wb > UINT32_MAX) {
            ଯଦି (owb == 1) ଭାଙ୍ଗ;
            ଜାରି;
        }

        ଅଚଳ uint64_t src_row_bytes_raw = (uint64_t) IC * KH * Wb * ଆକାର(ଦଶମିକ);
        ଅଚଳ uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        ଯଦି (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            ଯଦି (owb == 1) ଭାଙ୍ଗ;
            ଜାରି;
        }

        ଅଚଳ uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        ଅଚଳ uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        ସଂରଚନା htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        ଯଦି (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଯଦି (owb == 1) ଭାଙ୍ଗ;
    }

    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_precompute_im2col_3d_params(
    ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess,
    ଅଚଳ ସଂରଚନା ggml_tensor * op,
    ସଂରଚନା htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, ଆକାର(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (!src0 || !src1 || !dst) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src1->nb[0] != ଆକାର(ଦଶମିକ) || !ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int32_t * params = (ଅଚଳ int32_t *) op->op_params;
    ଅଚଳ int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    ଅଚଳ int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    ଅଚଳ int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    ଅଚଳ int32_t ic = params[9];

    ଯଦି (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src1->ne[3] % ic != 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t IC = (uint32_t) ic;
    ଅଚଳ uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    ଯଦି (N == 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    uint32_t OW, OH, OD;
    ଯଦି (!im2col_3d_calc_output_size(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_size(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_size(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t KD = (uint32_t) src0->ne[2];
    ଅଚଳ uint32_t KH = (uint32_t) src0->ne[1];
    ଅଚଳ uint32_t KW = (uint32_t) src0->ne[0];
    ଅଚଳ uint32_t ID = (uint32_t) src1->ne[2];
    ଅଚଳ uint32_t IH = (uint32_t) src1->ne[1];
    ଅଚଳ uint32_t IW = (uint32_t) src1->ne[0];

    ଅଚଳ uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    ଅଚଳ uint64_t total_rows_64   = (uint64_t) N * OD * OH;
    ଅଚଳ uint64_t dst_ne3_64      = (uint64_t) OD * N;

    ଯଦି (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ uint32_t patch_stride = (uint32_t) patch_stride_64;
    ଅଚଳ uint32_t total_rows   = (uint32_t) total_rows_64;
    ଅଚଳ uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    ଯଦି (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? ଆକାର(ggml_fp16_t) : ଆକାର(ଦଶମିକ);

    ଅଚଳ ତାର୍କିକ is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    ଅଚଳ uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    ଅଚଳ uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    ଅଚଳ uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    ଯଦି (is_patchembed) {
        ଅଚଳ uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * IW * ଆକାର(ଦଶମିକ);
        ଅଚଳ uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        ଯଦି (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            ଅଚଳ uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            ଅଚଳ uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            ସଂରଚନା htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            ଯଦି (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        }
    }

    ପାଇଁ (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ଅଚଳ uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        ଯଦି (Wb > UINT32_MAX) {
            ଯଦି (owb == 1) ଭାଙ୍ଗ;
            ଜାରି;
        }

        ଅଚଳ uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * Wb * ଆକାର(ଦଶମିକ);
        ଅଚଳ uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        ଯଦି (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            ଯଦି (owb == 1) ଭାଙ୍ଗ;
            ଜାରି;
        }

        ଅଚଳ uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        ଅଚଳ uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        ସଂରଚନା htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        ଯଦି (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଯଦି (owb == 1) ଭାଙ୍ଗ;
    }

    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_im2col(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ସଂରଚନା htp_im2col_kernel_params kparams;
    ଫେରନ୍ତୁ ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_im2col_3d(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ସଂରଚନା htp_im2col_kernel_params kparams;
    ଫେରନ୍ତୁ ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_pad(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int32_t lp0 = ((ଅଚଳ int32_t *) op->op_params)[0];
    ଅଚଳ int32_t rp0 = ((ଅଚଳ int32_t *) op->op_params)[1];
    ଅଚଳ int32_t circular = ((ଅଚଳ int32_t *) op->op_params)[8];

    ଯଦି (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_cumsum(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_diag(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    // diag only supports F32 currently
    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Input must have ne[1] == 1 (vector input)
    ଯଦି (src0->ne[1] != 1) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Output must be square in first two dimensions
    ଯଦି (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_solve_tri(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0]; // A
    ଅଚଳ ସଂରଚନା ggml_tensor * src1 = op->src[1]; // B
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;         // X

    ଯଦି (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src0->ne[0] != src0->ne[1]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src0->ne[1] != src1->ne[1]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_tri(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src0->type != GGML_TYPE_F32) { ଫେରନ୍ତୁ ମିଥ୍ୟା; }
    ଯଦି (dst->type  != GGML_TYPE_F32) { ଫେରନ୍ତୁ ମିଥ୍ୟା; }
    ଯଦି (!ggml_are_same_shape(src0, dst)) { ଫେରନ୍ତୁ ମିଥ୍ୟା; }
    ଯଦି (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { ଫେରନ୍ତୁ ମିଥ୍ୟା; }

    ଫେରନ୍ତୁ ସତ୍ୟ;

    GGML_UNUSED(sess);
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    ଫେରନ୍ତୁ sess->c_name();
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

ସ୍ଥିର htp_op_code op_remap_to_htp(ଅଚଳ ggml_tensor * t) {
    ବାଛ (t->op) {
        ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:  ଫେରନ୍ତୁ HTP_OP_FLASH_ATTN_EXT;
        ସ୍ଥିତି GGML_OP_MUL_MAT:
            ଯଦି (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                ଫେରନ୍ତୁ HTP_OP_HADAMARD;
            }
            ଫେରନ୍ତୁ HTP_OP_MUL_MAT;
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:      ଫେରନ୍ତୁ HTP_OP_MUL_MAT_ID;
        ସ୍ଥିତି GGML_OP_MUL:             ଫେରନ୍ତୁ HTP_OP_MUL;
        ସ୍ଥିତି GGML_OP_ADD:             ଫେରନ୍ତୁ HTP_OP_ADD;
        ସ୍ଥିତି GGML_OP_ADD_ID:          ଫେରନ୍ତୁ HTP_OP_ADD_ID;
        ସ୍ଥିତି GGML_OP_SUB:             ଫେରନ୍ତୁ HTP_OP_SUB;
        ସ୍ଥିତି GGML_OP_DIV:             ଫେରନ୍ତୁ HTP_OP_DIV;
        ସ୍ଥିତି GGML_OP_DUP:
        ସ୍ଥିତି GGML_OP_CPY:             ଫେରନ୍ତୁ HTP_OP_CPY;
        ସ୍ଥିତି GGML_OP_CONT:            ଫେରନ୍ତୁ HTP_OP_CPY;
        ସ୍ଥିତି GGML_OP_GET_ROWS:        ଫେରନ୍ତୁ HTP_OP_GET_ROWS;
        ସ୍ଥିତି GGML_OP_SET_ROWS:        ଫେରନ୍ତୁ HTP_OP_SET_ROWS;
        ସ୍ଥିତି GGML_OP_SUM:             ଫେରନ୍ତୁ HTP_OP_SUM;
        ସ୍ଥିତି GGML_OP_SUM_ROWS:        ଫେରନ୍ତୁ HTP_OP_SUM_ROWS;
        ସ୍ଥିତି GGML_OP_POOL_2D:         ଫେରନ୍ତୁ HTP_OP_POOL_2D;
        ସ୍ଥିତି GGML_OP_POOL_1D:         ଫେରନ୍ତୁ HTP_OP_POOL_1D;
        ସ୍ଥିତି GGML_OP_ARGSORT:         ଫେରନ୍ତୁ HTP_OP_ARGSORT;
        ସ୍ଥିତି GGML_OP_ଶୀର୍ଷ_K:           ଫେରନ୍ତୁ HTP_OP_ଶୀର୍ଷ_K;
        ସ୍ଥିତି GGML_OP_ARGMAX:          ଫେରନ୍ତୁ HTP_OP_ARGMAX;
        ସ୍ଥିତି GGML_OP_NORM:            ଫେରନ୍ତୁ HTP_OP_NORM;
        ସ୍ଥିତି GGML_OP_L2_NORM:         ଫେରନ୍ତୁ HTP_OP_L2_NORM;
        ସ୍ଥିତି GGML_OP_RMS_NORM:        ଫେରନ୍ତୁ HTP_OP_RMS_NORM;
        ସ୍ଥିତି GGML_OP_CONCAT:          ଫେରନ୍ତୁ HTP_OP_CONCAT;
        ସ୍ଥିତି GGML_OP_SCALE:           ଫେରନ୍ତୁ HTP_OP_SCALE;
        ସ୍ଥିତି GGML_OP_CLAMP:           ଫେରନ୍ତୁ HTP_OP_CLAMP;
        ସ୍ଥିତି GGML_OP_LEAKY_RELU:      ଫେରନ୍ତୁ HTP_OP_LEAKY_RELU;
        ସ୍ଥିତି GGML_OP_SQR:             ଫେରନ୍ତୁ HTP_OP_SQR;
        ସ୍ଥିତି GGML_OP_SQRT:            ଫେରନ୍ତୁ HTP_OP_SQRT;
        ସ୍ଥିତି GGML_OP_LOG:             ଫେରନ୍ତୁ HTP_OP_UNARY_LOG;
        ସ୍ଥିତି GGML_OP_SOFT_MAX:        ଫେରନ୍ତୁ HTP_OP_SOFTMAX;
        ସ୍ଥିତି GGML_OP_SSM_CONV:        ଫେରନ୍ତୁ HTP_OP_SSM_CONV;
        ସ୍ଥିତି GGML_OP_GATED_DELTA_NET: ଫେରନ୍ତୁ HTP_OP_GATED_DELTA_NET;
        ସ୍ଥିତି GGML_OP_ROPE:            ଫେରନ୍ତୁ HTP_OP_ROPE;
        ସ୍ଥିତି GGML_OP_REPEAT:          ଫେରନ୍ତୁ HTP_OP_REPEAT;
        ସ୍ଥିତି GGML_OP_CUMSUM:          ଫେରନ୍ତୁ HTP_OP_CUMSUM;
        ସ୍ଥିତି GGML_OP_FILL:            ଫେରନ୍ତୁ HTP_OP_FILL;
        ସ୍ଥିତି GGML_OP_DIAG:            ଫେରନ୍ତୁ HTP_OP_DIAG;
        ସ୍ଥିତି GGML_OP_SOLVE_TRI:       ଫେରନ୍ତୁ HTP_OP_SOLVE_TRI;
        ସ୍ଥିତି GGML_OP_TRI:             ଫେରନ୍ତୁ HTP_OP_TRI;
        ସ୍ଥିତି GGML_OP_PAD:             ଫେରନ୍ତୁ HTP_OP_PAD;
        ସ୍ଥିତି GGML_OP_IM2COL:          ଫେରନ୍ତୁ HTP_OP_IM2COL;
        ସ୍ଥିତି GGML_OP_IM2COL_3D:       ଫେରନ୍ତୁ HTP_OP_IM2COL_3D;
        ସ୍ଥିତି GGML_OP_ROLL:            ଫେରନ୍ତୁ HTP_OP_ROLL;

        ସ୍ଥିତି GGML_OP_UNARY:
            ବାଛ (ggml_get_unary_op(t)) {
                ସ୍ଥିତି GGML_UNARY_OP_SILU:       ଫେରନ୍ତୁ HTP_OP_UNARY_SILU;
                ସ୍ଥିତି GGML_UNARY_OP_GELU:       ଫେରନ୍ତୁ HTP_OP_UNARY_GELU;
                ସ୍ଥିତି GGML_UNARY_OP_GELU_QUICK: ଫେରନ୍ତୁ HTP_OP_UNARY_GELU_QUICK;
                ସ୍ଥିତି GGML_UNARY_OP_GELU_ERF:   ଫେରନ୍ତୁ HTP_OP_UNARY_GELU_ERF;
                ସ୍ଥିତି GGML_UNARY_OP_SIGMOID:    ଫେରନ୍ତୁ HTP_OP_UNARY_SIGMOID;
                ସ୍ଥିତି GGML_UNARY_OP_NEG:        ଫେରନ୍ତୁ HTP_OP_UNARY_NEG;
                ସ୍ଥିତି GGML_UNARY_OP_EXP:        ଫେରନ୍ତୁ HTP_OP_UNARY_EXP;
                ସ୍ଥିତି GGML_UNARY_OP_SOFTPLUS:   ଫେରନ୍ତୁ HTP_OP_UNARY_SOFTPLUS;
                ସ୍ଥିତି GGML_UNARY_OP_TANH:       ଫେରନ୍ତୁ HTP_OP_UNARY_TANH;
                ସ୍ଥିତି GGML_UNARY_OP_ABS:        ଫେରନ୍ତୁ HTP_OP_UNARY_ABS;
                ସ୍ଥିତି GGML_UNARY_OP_RELU:       ଫେରନ୍ତୁ HTP_OP_UNARY_RELU;
                ସ୍ଥିତି GGML_UNARY_OP_STEP:       ଫେରନ୍ତୁ HTP_OP_UNARY_STEP;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଭାଙ୍ଗ;
            }
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_GLU:
            ବାଛ (ggml_get_glu_op(t)) {
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU:       ଫେରନ୍ତୁ HTP_OP_GLU_SWIGLU;
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU_OAI:   ଫେରନ୍ତୁ HTP_OP_GLU_SWIGLU_OAI;
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU_CLAMP: ଫେରନ୍ତୁ HTP_OP_GLU_SWIGLU_CLAMP;
                ସ୍ଥିତି GGML_GLU_OP_GEGLU:        ଫେରନ୍ତୁ HTP_OP_GLU_GEGLU;
                ସ୍ଥିତି GGML_GLU_OP_GEGLU_QUICK:  ଫେରନ୍ତୁ HTP_OP_GLU_GEGLU_QUICK;
                ସ୍ଥିତି GGML_GLU_OP_GEGLU_ERF:    ଫେରନ୍ତୁ HTP_OP_GLU_GEGLU_ERF;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ: ଭାଙ୍ଗ;
            }
            ଭାଙ୍ଗ;

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    ଫେରନ୍ତୁ HTP_OP_INVALID;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ op_is_compute(ggml_tensor *ଗଣ୍ଠି)
{
    ଫେରନ୍ତୁ !ggml_op_is_empty(ଗଣ୍ଠି->op) && !ggml_is_empty(ଗଣ୍ଠି) && (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_COMPUTE);
}

ସ୍ଥିର ତାର୍କିକ mm_is_hmx_eligible(ଅଚଳ ggml_tensor * t) {
    ଯଦି (opt_nhmx == 0) { ଫେରନ୍ତୁ ମିଥ୍ୟା; }

    ଅଚଳ ggml_tensor * src0 = t->src[0];
    ଅଚଳ ggml_tensor * src1 = t->src[1];

    ଯଦି (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        ଅଚଳ ggml_tensor src1_c = ggml_hexagon_tensor_collapse_rows(src1);
        ଫେରନ୍ତୁ ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, ମିଥ୍ୟା, ମିଥ୍ୟା);
    }

    ଅଚଳ ତାର୍କିକ is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    ଅଚଳ ତାର୍କିକ is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    ଫେରନ୍ତୁ ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

ସ୍ଥିର ତାର୍କିକ is_supported_mul_mat_nx_kernel(ଅଚଳ ggml_tensor * src0, ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams) {
    ଯଦି (kparams->n_hmx) {
        ଫେରନ୍ତୁ kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    ଯଦି (!ggml_hexagon_tensor_is_tiled(src0)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
}

ସ୍ଥିର ତାର୍କିକ is_supported_mul_mat_id_nx_kernel(ଅଚଳ ggml_tensor * src0, ଅଚଳ ସଂରଚନା htp_mm_kernel_params * kparams) {
    ଯଦି (kparams->n_hmx) {
        ଫେରନ୍ତୁ kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    ଯଦି (!ggml_hexagon_tensor_is_tiled(src0)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK;
}

ସ୍ଥିର ତାର୍କିକ is_mergeable_mul_mat(ଅଚଳ ggml_tensor * t) {
    ଯଦି (t->op != GGML_OP_MUL_MAT) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    ଯଦି (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    ଅଚଳ ggml_tensor * src0 = t->src[0];
    ଅଚଳ ggml_tensor * src1 = t->src[1];
    ଯଦି (src0->ne[2] != 1 || src0->ne[3] != 1) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    ଯଦି (mm_is_hmx_eligible(t)) {
        ଫେରନ୍ତୁ htp_is_hmx_weight_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    ଯଦି (src1->type != GGML_TYPE_F32) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    ଫେରନ୍ତୁ ggml_hexagon_tensor_is_tiled(src0);
}

ସ୍ଥିର ତାର୍କିକ is_mergeable_mul_mat_pair(ଅଚଳ ggml_tensor * n1, ଅଚଳ ggml_tensor * n2) {
    ଯଦି (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (n1->src[1] != n2->src[1]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (n1->src[0]->type != n2->src[0]->type) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ is_mergeable_mul_mat_id(ଅଚଳ ggml_tensor * t) {
    ଯଦି (t->op != GGML_OP_MUL_MAT_ID) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    ଅଚଳ ggml_tensor * src0 = t->src[0];
    ଯଦି (mm_is_hmx_eligible(t)) {
        ଫେରନ୍ତୁ htp_is_hmx_weight_type(src0->type);
    }

    ଫେରନ୍ତୁ ggml_hexagon_tensor_is_tiled(src0);
}

ସ୍ଥିର ତାର୍କିକ is_mergeable_mul_mat_id_pair(ଅଚଳ ggml_tensor * n1, ଅଚଳ ggml_tensor * n2) {
    ଯଦି (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (n1->src[1] != n2->src[1]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (n1->src[2] != n2->src[2]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (n1->src[0]->type != n2->src[0]->type) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    ଯଦି (sess->last_error > HTP_STATUS_OK) {
        ଫେରନ୍ତୁ GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    ଅଚଳ std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    ତାର୍କିକ cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
    ଯଦି (cache_hit) {
        nodes_ptr = &sess->cached_nodes;
    } ଅନ୍ୟଥା {
        // Tag fusable tensors in graph
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < graph->n_nodes; i++) {
            auto * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            ଯଦି (!extra) ଜାରି;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            ଯଦି (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } ଅନ୍ୟଥା ଯଦି ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                ଯଦି ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_ଗଣ୍ଠି_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            ଯଦି (!op_is_compute(n)) {
                ଜାରି;
            }

            htp_opnode ଗଣ୍ଠି(HTP_OP_INVALID, n);
            ଗଣ୍ଠି.opcode = op_remap_to_htp(n);
            ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT || ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT_ID || ଗଣ୍ଠି.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = ଗଣ୍ଠି.ଗଣ୍ଠି->src[0];
                ଯଦି (ggml_hexagon_needs_repack(src0)) {
                    ଯଦି (opt_verbose) {
                        ଅକ୍ଷର dims_buf[64];
                        ଅକ୍ଷର strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, ଆକାର(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, ଆକାର(strides_buf), src0);
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

            ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT || ଗଣ୍ଠି.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି->src[0], ଗଣ୍ଠି.ଗଣ୍ଠି->src[1], ଗଣ୍ଠି.ଗଣ୍ଠି,
                    (ସଂରଚନା htp_mm_kernel_params *)ଗଣ୍ଠି.kernel_params
                );
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_MUL || ଗଣ୍ଠି.opcode == HTP_OP_ADD || ଗଣ୍ଠି.opcode == HTP_OP_ADD_ID || ଗଣ୍ଠି.opcode == HTP_OP_SUB || ଗଣ୍ଠି.opcode == HTP_OP_DIV) {
                ଅଚଳ ggml_tensor * src1 = ଗଣ୍ଠି.ଗଣ୍ଠି->src[1];
                ଅଚଳ ggml_tensor * src2 = (ଗଣ୍ଠି.opcode == HTP_OP_ADD_ID) ? ଗଣ୍ଠି.ଗଣ୍ଠି->src[2] : nullptr;
                GGML_ASSERT(ggml_hexagon_precompute_ଦ୍ୱିମିକ_params(sess,
                    ଗଣ୍ଠି.opcode, ଗଣ୍ଠି.ଗଣ୍ଠି->src[0], src1, ଗଣ୍ଠି.ଗଣ୍ଠି,
                    (ସଂରଚନା htp_ଦ୍ୱିମିକ_kernel_params *) ଗଣ୍ଠି.kernel_params, src2));
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି,
                    (ସଂରଚନା htp_fa_kernel_params *)ଗଣ୍ଠି.kernel_params
                );
            } ଅନ୍ୟଥା ଯଦି (htp_op_is_unary(ଗଣ୍ଠି.opcode)) {
                auto inputs = ଗଣ୍ଠି.get_inputs();
                ଅଚଳ ସଂରଚନା ggml_tensor * src0 = inputs[0];
                ଅଚଳ ସଂରଚନା ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    ଗଣ୍ଠି.opcode, src0, src1, ଗଣ୍ଠି.dst(),
                    (ସଂରଚନା htp_unary_kernel_params *)ଗଣ୍ଠି.kernel_params
                );
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_GET_ROWS) {
                ggml_hexagon_precompute_get_rows_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି->src[0], ଗଣ୍ଠି.ଗଣ୍ଠି->src[1], ଗଣ୍ଠି.dst(),
                    (ସଂରଚନା htp_get_rows_kernel_params *)ଗଣ୍ଠି.kernel_params
                );
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_SET_ROWS) {
                ggml_hexagon_precompute_set_rows_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି->src[0], ଗଣ୍ଠି.ଗଣ୍ଠି->src[1], ଗଣ୍ଠି.dst(),
                    (ସଂରଚନା htp_set_rows_kernel_params *)ଗଣ୍ଠି.kernel_params
                );
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି,
                    (ସଂରଚନା htp_rope_kernel_params *)ଗଣ୍ଠି.kernel_params
                );
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି->src[0], ଗଣ୍ଠି.ଗଣ୍ଠି->src[1], ଗଣ୍ଠି.dst(),
                    (ସଂରଚନା htp_ssm_conv_kernel_params *)ଗଣ୍ଠି.kernel_params
                );
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି,
                    (ସଂରଚନା htp_softmax_kernel_params *)ଗଣ୍ଠି.kernel_params
                );
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି,
                    (ସଂରଚନା htp_gdn_kernel_params *)ଗଣ୍ଠି.kernel_params
                );
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_ARGSORT || ଗଣ୍ଠି.opcode == HTP_OP_ଶୀର୍ଷ_K) {
                ggml_hexagon_precompute_sort_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି,
                    ଗଣ୍ଠି.opcode == HTP_OP_ଶୀର୍ଷ_K,
                    (ସଂରଚନା htp_sort_kernel_params *) ଗଣ୍ଠି.kernel_params
                );
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_POOL_2D || ଗଣ୍ଠି.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, ଗଣ୍ଠି.ଗଣ୍ଠି->src[0], ଗଣ୍ଠି.dst(),
                    (ସଂରଚନା htp_pool_2d_kernel_params *)ଗଣ୍ଠି.kernel_params,
                    ଗଣ୍ଠି.opcode == HTP_OP_POOL_1D);
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି,
                    (ସଂରଚନା htp_concat_kernel_params *) ଗଣ୍ଠି.kernel_params
                );
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_CPY || ଗଣ୍ଠି.opcode == HTP_OP_CPY_FENCE) {
                ଅଚଳ ତାର୍କିକ ok = ggml_hexagon_precompute_cpy_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି,
                    (ସଂରଚନା htp_copy_kernel_params *) ଗଣ୍ଠି.kernel_params
                );
                ଅଚଳ auto * kparams = (ଅଚଳ ସଂରଚନା htp_copy_kernel_params *) ଗଣ୍ଠି.kernel_params;
                ଯଦି (ok && ଗଣ୍ଠି.opcode == HTP_OP_CPY && kparams->total_elems == 0) {
                    ଜାରି;
                }
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_IM2COL) {
                ggml_hexagon_precompute_im2col_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି,
                    (ସଂରଚନା htp_im2col_kernel_params *) ଗଣ୍ଠି.kernel_params
                );
            } ଅନ୍ୟଥା ଯଦି (ଗଣ୍ଠି.opcode == HTP_OP_IM2COL_3D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    ଗଣ୍ଠି.ଗଣ୍ଠି,
                    (ସଂରଚନା htp_im2col_kernel_params *) ଗଣ୍ଠି.kernel_params
                );
            }
            computed_nodes.push_back(std::move(ଗଣ୍ଠି));
        }

        ଯଦି (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } ଅନ୍ୟଥା {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    ପାଇଁ (ଅଚଳ auto & ଗଣ୍ଠି : *nodes_ptr) {
        sess->enqueue_op(ଗଣ୍ଠି);
    }

    ଯଦି (sess->last_error > HTP_STATUS_OK) {
        ଫେରନ୍ତୁ GGML_STATUS_FAILED;
    }

    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    ଯଦି (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
}

ଗଣନା ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

ସଂରଚନା ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

ସଂରଚନା ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    ଶୂନ୍ୟ reset() {
        ranges.clear();
    }

    ଶୂନ୍ୟ add(ଅଚଳ ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    ତାର୍କିକ check(ଅଚଳ ggml_hexagon_mem_range & mr) ଅଚଳ {
        ପାଇଁ (ଅଚଳ auto & cmp : ranges) {
            ଯଦି (mr.pb != cmp.pb) {
                ଜାରି;
            }
            ଯଦି (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                ଜାରି;
            }
            ଯଦି (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
};

ସ୍ଥିର ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(ଅଚଳ ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    ଅଚଳ ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    ଯଦି (tensor->buffer) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } ଅନ୍ୟଥା {
        mr = {
            /*.pb =*/ (uint64_t) base,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    ଫେରନ୍ତୁ mr;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_mem_ranges_add_ଗଣ୍ଠି(ggml_hexagon_mem_ranges & mrs, ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
    ଯଦି (ଗଣ୍ଠି.is_empty()) ଫେରନ୍ତୁ;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
        ଯଦି (ଗଣ୍ଠି.ଗଣ୍ଠି->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(ଗଣ୍ଠି.ଗଣ୍ଠି->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    ପାଇଁ (ଅଚଳ auto * fused : ଗଣ୍ଠି.fused) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
            ଯଦି (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(ଗଣ୍ଠି.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_mem_ranges_check_ଗଣ୍ଠି(ଅଚଳ ggml_hexagon_mem_ranges & mrs, ଅଚଳ htp_opnode & ଗଣ୍ଠି) {
    ଯଦି (ଗଣ୍ଠି.is_empty()) ଫେରନ୍ତୁ ସତ୍ୟ;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
        ଯଦି (ଗଣ୍ଠି.ଗଣ୍ଠି->src[i]) {
            ଯଦି (!mrs.check(ggml_hexagon_mem_range_from_tensor(ଗଣ୍ଠି.ଗଣ୍ଠି->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }
    }
    ପାଇଁ (ଅଚଳ auto * fused : ଗଣ୍ଠି.fused) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
            ଯଦି (fused->src[i]) {
                ଯଦି (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
            }
        }
    }
    ଫେରନ୍ତୁ mrs.check(ggml_hexagon_mem_range_from_tensor(ଗଣ୍ଠି.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ସ୍ଥିର std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> ggml_hexagon_graph_optimize_reorder(ଅଚଳ std::vector<htp_opnode> & nodes) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = nodes.size();

    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> res;
    res.reserve(n);

    std::vector<ତାର୍କିକ> used(n, ମିଥ୍ୟା);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0 = 0; i0 < n; i0++) {
        ଯଦି (used[i0]) {
            ଜାରି;
        }

        ଅଚଳ auto & node0 = nodes[i0];

        ଯଦି (!node0.stackable()) {
            res.push_back(i0);
            used[i0] = ସତ୍ୟ;
            ଜାରି;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା N_FORWARD = 16;

        std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> ଗଦା;
        ଗଦା.push_back(i0);

        mrs.reset();

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            ଯଦି (used[i1]) {
                ଜାରି;
            }

            ଅଚଳ auto & node1 = nodes[i1];

            ଯଦି (node1.stackable() && node1.same_ନିବେଶ(node0) && ggml_hexagon_mem_ranges_check_ଗଣ୍ଠି(mrs, node1)) {
                ଗଦା.push_back(i1);
            } ଅନ୍ୟଥା {
                ggml_hexagon_mem_ranges_add_ଗଣ୍ଠି(mrs, node1);
            }
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx : ଗଦା) {
            res.push_back(idx);
            used[idx] = ସତ୍ୟ;
        }
    }

    ଫେରନ୍ତୁ res;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = gf->n_nodes;

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା MAX_FUSE = 16;

    ଗଣନା ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        htp_opnode ଗଣ୍ଠି(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        ଯଦି (ଗଣ୍ଠି.op() == GGML_OP_ADD ||
            ଗଣ୍ଠି.op() == GGML_OP_NORM ||
            ଗଣ୍ଠି.op() == GGML_OP_RMS_NORM) {
            ops[0] = ଗଣ୍ଠି.op();

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା f = i + 1;
            ଯେତେବେଳେ (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                ଯଦି (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    ଭାଙ୍ଗ;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            ପାଇଁ (; f > 1; f--) {
                ଯଦି (ggml_can_fuse(gf, i, ops, f)) {
                    ଭାଙ୍ଗ;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                ଗଣ୍ଠି.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(ଗଣ୍ଠି));
    }

    ଅଚଳ auto order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0;
        ପାଇଁ (ଅଚଳ auto i : order) {
            ଅଚଳ auto & ଗଣ୍ଠି = nodes[i];

            gf->nodes[j++] = ଗଣ୍ଠି.ଗଣ୍ଠି;

            ପାଇଁ (auto * fused : ଗଣ୍ଠି.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

ସ୍ଥିର uint64_t ggml_hexagon_session_ଚାବି(ଅଚଳ ggml_hexagon_session * sess) {
    ଫେରନ୍ତୁ ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, ଅଚଳ ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    auto sbuf_dst = sess_dst->mmap_tensor(dst);

    ଯଦି (!sess_src->clone_buffer(sbuf_dst)) { ଫେରନ୍ତୁ ମିଥ୍ୟା; }

    ଅଚଳ uint64_t src_ଚାବି = ggml_hexagon_session_ଚାବି(sess_src);
    auto & fence_slot = sess_dst->cpy_fence_slots[src_ଚାବି];
    ଯଦି (!fence_slot) {
        fence_slot = (ଅସ୍ଥିର uint32_t *) sess_dst->alloc_fence(1);
    }

    ଯଦି (!sess_src->clone_buffer(sess_dst->fence_buf)) { ଫେରନ୍ତୁ ମିଥ୍ୟା; }

    ଯଦି (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    ସ୍ଥିର ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.buffer = &sess_dst->fence_buf->backend_buffer;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.data   = (ଶୂନ୍ୟ *) fence_slot;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = ଆକାର(int32_t);
    fence_tensor.nb[1]  = ଆକାର(int32_t);
    fence_tensor.nb[2]  = ଆକାର(int32_t);
    fence_tensor.nb[3]  = ଆକାର(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ ସତ୍ୟ);

    sess_dst->add_peer(sess_src);

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, ଅଚଳ ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    auto sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    ଯଦି (!sess_dst->clone_buffer(sbuf_src)) { ଫେରନ୍ତୁ ମିଥ୍ୟା; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ଅଚଳ ggml_tensor * src, ggml_tensor * dst) {
    ଯଦି (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    ଯଦି (!dst->extra) {
        ggml_backend_buffer_init_tensor(dst->buffer, dst);
    }

    auto * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    ଅଚଳ auto * src_extra = static_cast<ଅଚଳ ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    ଯଦି (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଯଦି (sess_src->phys_idx != sess_dst->phys_idx)
        ଫେରନ୍ତୁ ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    ଫେରନ୍ତୁ ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

ସ୍ଥିର ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = ନୂତନ ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_slot = (ଅସ୍ଥିର uint32_t *) sess->alloc_fence(1);

    ସ୍ଥିର ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.buffer = &sess->fence_buf->backend_buffer;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.data   = (ଶୂନ୍ୟ *) hex_event->fence_slot;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = ଆକାର(int32_t);
    hex_event->fence_tensor.nb[1]  = ଆକାର(int32_t);
    hex_event->fence_tensor.nb[2]  = ଆକାର(int32_t);
    hex_event->fence_tensor.nb[3]  = ଆକାର(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (ଶୂନ୍ୟ *)hex_event, (ଶୂନ୍ୟ *)hex_event->fence_slot);

    ଫେରନ୍ତୁ ନୂତନ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    ଯଦି (hex_event->seq == 0) {
        ଫେରନ୍ତୁ;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (ଶୂନ୍ୟ *)hex_event, hex_event->seq, (ଶୂନ୍ୟ *)hex_event->fence_slot);

    auto * fence = reinterpret_cast<ଅଚଳ ଅସ୍ଥିର std::atomic<uint32_t> *>(hex_event->fence_slot);

    ଯଦି ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    ଯେତେବେଳେ (ସତ୍ୟ) {
        ଯଦି ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            ଯଦି (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            ଭାଙ୍ଗ;
        }
        std::this_thread::yield();
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (ଶୂନ୍ୟ *)hex_event);
    hex_event->fence_sess->free_fence((ଶୂନ୍ୟ *) hex_event->fence_slot, 1);
    delete hex_event;
    delete event;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    ଯଦି (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ ମିଥ୍ୟା);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (ଶୂନ୍ୟ *)hex_event, hex_event->seq, (ଶୂନ୍ୟ *)hex_event->fence_slot);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    ଯଦି (hex_event->seq == 0) {
        ଫେରନ୍ତୁ;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (ଶୂନ୍ୟ *)hex_event, hex_event->seq, (ଶୂନ୍ୟ *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    ଯଦି (sess->phys_idx == hex_event->sess->phys_idx) {
        ଯଦି (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        ଫେରନ୍ତୁ;
    }

    sess->clone_buffer(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ ସତ୍ୟ);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, ସଂରଚନା ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) tensor->buffer->usage : -1);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) tensor->buffer->usage : -1);
    sess->flush_sync();
    ଯଦି (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     ସଂରଚନା ggml_tensor * tensor,
                                                     ଅଚଳ ଶୂନ୍ୟ * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) tensor->buffer->usage : -1);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     ଅଚଳ ସଂରଚନା ggml_tensor * tensor,
                                                     ଶୂନ୍ୟ * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) tensor->buffer->usage : -1);
    sess->flush_sync();
    ଯଦି (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ସ୍ଥିର ସଂରଚନା ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ ଖାଲି,
    /* .graph_plan_free         = */ ଖାଲି,
    /* .graph_plan_update       = */ ଖାଲି,
    /* .graph_plan_compute      = */ ଖାଲି,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_record,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

ସ୍ଥିର ggml_guid_t ggml_backend_hexagon_guid() {
    ସ୍ଥିର ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    ଫେରନ୍ତୁ &guid;
}

ତାର୍କିକ ggml_backend_is_hexagon(ggml_backend_t backend) {
    ଫେରନ୍ତୁ backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

ସ୍ଥିର ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, ଅଚଳ ଅକ୍ଷର * params) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ଫେରନ୍ତୁ ନୂତନ ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ଫେରନ୍ତୁ dev_ctx->c_name();

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ "Hexagon";
    GGML_UNUSED(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * ମୁକ୍ତ, size_t * total) {
    *ମୁକ୍ତ  = opt_mbuf_total;
    *total = opt_mbuf_total;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଗଣନା ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    ଫେରନ୍ତୁ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, ସଂରଚନା ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ସତ୍ୟ,
        /* .host_buffer           = */ ମିଥ୍ୟା,
        /* .buffer_from_host_ptr  = */ ମିଥ୍ୟା,
        /* .events                = */ ସତ୍ୟ,
        /* .mmap_support          = */ ମିଥ୍ୟା,
    };
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ଫେରନ୍ତୁ &dev_ctx->buffer_type;
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    ଯଦି (!opt_hostbuf) {
        ଫେରନ୍ତୁ ଖାଲି;
    }
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ଫେରନ୍ତୁ &dev_ctx->host_buffer_type;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_cpy(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ସଂରଚନା htp_copy_kernel_params kparams;
    ଫେରନ୍ତୁ ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_cont(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ସଂରଚନା htp_copy_kernel_params kparams;
    ଫେରନ୍ତୁ ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_repeat(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    GGML_UNUSED(sess);
    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    // Support f32 and f16
    ଯଦି (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    // src and dst must be the same type
    ଯଦି (src0->type != dst->type) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    // dst dims must be multiples of src dims
    ଯଦି (dst->ne[0] % src0->ne[0] != 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    ଯଦି (dst->ne[1] % src0->ne[1] != 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    ଯଦି (dst->ne[2] % src0->ne[2] != 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    ଯଦି (dst->ne[3] % src0->ne[3] != 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    // require contiguous tensors (no transposition)
    ଯଦି (ggml_is_transposed(src0) || ggml_is_transposed(dst)) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_concat(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ସଂରଚନା htp_concat_kernel_params kparams;
    ଫେରନ୍ତୁ ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_fill(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    ଅଚଳ ସଂରଚନା ggml_tensor * dst = op;

    ଯଦି (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
    GGML_UNUSED(sess);
}

ସ୍ଥିର ତାର୍କିକ ggml_hexagon_supported_roll(ଅଚଳ ସଂରଚନା ggml_hexagon_session * sess, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    GGML_UNUSED(sess);

    ଅଚଳ ସଂରଚନା ggml_tensor * src0 = op->src[0];
    ଅଚଳ ସଂରଚନା ggml_tensor * dst  = op;

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_are_same_shape(src0, dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (src0->nb[0] != ggml_type_size(src0->type) || dst->nb[0] != ggml_type_size(dst->type)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
ସ୍ଥିର ତାର୍କିକ ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    // reject ops that match the filter
    ଯଦି (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ତାର୍କିକ supp = ମିଥ୍ୟା;
    ବାଛ (op->op) {
        ସ୍ଥିତି GGML_OP_NONE:
        ସ୍ଥିତି GGML_OP_RESHAPE:
        ସ୍ଥିତି GGML_OP_VIEW:
        ସ୍ଥିତି GGML_OP_PERMUTE:
        ସ୍ଥିତି GGML_OP_TRANSPOSE:
            supp = ସତ୍ୟ;
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_MUL:
        ସ୍ଥିତି GGML_OP_ADD:
        ସ୍ଥିତି GGML_OP_SUB:
        ସ୍ଥିତି GGML_OP_DIV:
            supp = ggml_hexagon_supported_ଦ୍ୱିମିକ(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_NORM:
        ସ୍ଥିତି GGML_OP_L2_NORM:
        ସ୍ଥିତି GGML_OP_RMS_NORM:
        ସ୍ଥିତି GGML_OP_SCALE:
        ସ୍ଥିତି GGML_OP_CLAMP:
        ସ୍ଥିତି GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_SQR:
        ସ୍ଥିତି GGML_OP_SQRT:
        ସ୍ଥିତି GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_SUM:
            supp = ggml_hexagon_supported_sum(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_UNARY:
            ବାଛ (ggml_get_unary_op(op)) {
                ସ୍ଥିତି GGML_UNARY_OP_NEG:
                ସ୍ଥିତି GGML_UNARY_OP_EXP:
                ସ୍ଥିତି GGML_UNARY_OP_SIGMOID:
                ସ୍ଥିତି GGML_UNARY_OP_SOFTPLUS:
                ସ୍ଥିତି GGML_UNARY_OP_TANH:
                ସ୍ଥିତି GGML_UNARY_OP_ABS:
                ସ୍ଥିତି GGML_UNARY_OP_SILU:
                ସ୍ଥିତି GGML_UNARY_OP_GELU:
                ସ୍ଥିତି GGML_UNARY_OP_GELU_QUICK:
                ସ୍ଥିତି GGML_UNARY_OP_GELU_ERF:
                ସ୍ଥିତି GGML_UNARY_OP_RELU:
                ସ୍ଥିତି GGML_UNARY_OP_STEP:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    supp = ମିଥ୍ୟା;
                    ଭାଙ୍ଗ;
            }
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_GLU:
            ବାଛ (ggml_get_glu_op(op)) {
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU:
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU_OAI:
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU_CLAMP:
                ସ୍ଥିତି GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_GEGLU_QUICK:
                ସ୍ଥିତି GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    supp = ମିଥ୍ୟା;
                    ଭାଙ୍ଗ;
            }
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_DUP:
        ସ୍ଥିତି GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_ଶୀର୍ଷ_K:
            supp = ggml_hexagon_supported_ଶୀର୍ଷ_k(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_IM2COL_3D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            ଭାଙ୍ଗ;

        ସ୍ଥିତି GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            ଭାଙ୍ଗ;

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଭାଙ୍ଗ;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    ଫେରନ୍ତୁ supp;
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    ତାର୍କିକ supp = (buft == &dev_ctx->host_buffer_type) || (buft == &dev_ctx->buffer_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    ଫେରନ୍ତୁ supp;
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ ଖାଲି,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ ଖାଲି,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    ପାଇଁ (size_t i = 0; i < opt_ndev; i++) {
        ଅଚଳ auto & cfg = opt_device_configs[i];
        ଯଦି (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } ଅନ୍ୟଥା {
            std::string peers_str;
            ପାଇଁ (ଅଚଳ auto & p : cfg.mdev_group) {
                ଯଦି (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = ନୂତନ ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    ପାଇଁ (size_t i = 0; i < opt_ndev; i++) {
        auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        delete dev_ctx;
    }
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    ଫେରନ୍ତୁ "HTP";
    GGML_UNUSED(reg);
}

ସ୍ଥିର size_t ggml_backend_hexagon_reg_get_device_ଗଣନା(ggml_backend_reg_t reg) {
    ଫେରନ୍ତୁ opt_ndev;
    GGML_UNUSED(reg);
}

ସ୍ଥିର ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    ଯଦି (index >= opt_ndev || !hreg->devices[index].context) {
        ଫେରନ୍ତୁ nullptr;
    }

    ଫେରନ୍ତୁ &hreg->devices[index];
}

// ** communication context for tensor-split allreduce

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    ଯଦି (n_backends < 2 || n_backends > 4) {
        ଫେରନ୍ତୁ nullptr;
    }

    ପାଇଁ (size_t i = 0; i < n_backends; ++i) {
        ଯଦି (!ggml_backend_is_hexagon(backends[i])) {
            ଫେରନ୍ତୁ nullptr;
        }
    }

    ପାଇଁ (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ପାଇଁ (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            ଯଦି (sess_i->phys_idx == sess_j->phys_idx) {
                ଫେରନ୍ତୁ nullptr;
            }
        }
    }

    auto * ctx = ନୂତନ ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    ସ୍ଥିର ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ପାଇଁ (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (ଅସ୍ଥିର uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].buffer = &sess_i->fence_buf->backend_buffer;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].data   = (ଶୂନ୍ୟ *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = ଆକାର(int32_t);
        ctx->fence_tensors[i].nb[1]  = ଆକାର(int32_t);
        ctx->fence_tensors[i].nb[2]  = ଆକାର(int32_t);
        ctx->fence_tensors[i].nb[3]  = ଆକାର(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    ଫେରନ୍ତୁ ctx;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_hexagon_comm_free(ଶୂନ୍ୟ * comm_ctx_v) {
    ଯଦି (!comm_ctx_v) ଫେରନ୍ତୁ;
    auto * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ପାଇଁ (size_t i = 0; i < ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((ଶୂନ୍ୟ *) ctx->fence_slots[i], 1);
    }
    delete ctx;
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_hexagon_comm_allreduce_tensor(ଶୂନ୍ୟ * comm_ctx_v, ସଂରଚନା ggml_tensor ** tensors) {
    ଯଦି (opt_ar_select == 0 || !comm_ctx_v) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ଅଚଳ size_t n_backends = comm_ctx->n_backends;

    ଯଦି (n_backends < 2 || n_backends > 4) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    ପାଇଁ (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        ପାଇଁ (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            ଯଦି (sess_i->phys_idx == sess_j->phys_idx) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }
    }

    ପାଇଁ (size_t i = 0; i < n_backends; i++) {
        ଯଦି (!tensors[i] || !tensors[i]->buffer || !ggml_backend_buffer_is_hexagon(tensors[i]->buffer)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (tensors[i]->type != tensors[0]->type) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (!ggml_is_contiguous(tensors[i])) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଯଦି (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ପାଇଁ (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        ସଂରଚନା htp_allreduce_kernel_params kparams;
        ଯଦି (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, ମିଥ୍ୟା, ମିଥ୍ୟା, /*is_shard_ok=*/ ମିଥ୍ୟା, &kparams)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    ପାଇଁ (size_t i = 1; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        ଯଦି ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    ଯଦି (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_entry = max_seq;
    ଯଦି (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    ପାଇଁ (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<ଅଚଳ ggml_tensor *> data_tensors(n_backends);
    std::vector<ଅଚଳ ggml_tensor *> sync_tensors(n_backends);
    ପାଇଁ (size_t i = 0; i < n_backends; i++) {
        data_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    ପାଇଁ (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], data_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        ପାଇଁ (size_t j = 0; j < n_backends; j++) {
            ଯଦି (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_hexagon_split_buffer_type(ପୂର୍ଣ୍ଣସଂଖ୍ୟା main_device, ଅଚଳ ଦଶମିକ * tensor_split) {
    GGML_UNUSED(tensor_split);
    auto reg = ggml_backend_hexagon_reg();
    auto dev = ggml_backend_reg_dev_get(reg, main_device);
    ଯଦି (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    ଯଦି (!dev) ଫେରନ୍ତୁ nullptr;
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ଫେରନ୍ତୁ &dev_ctx->buffer_type;
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, ଅଚଳ ଅକ୍ଷର * name) {
    GGML_UNUSED(reg);
    ଯଦି (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *) ggml_backend_hexagon_split_buffer_type;
    }
    ଯଦି (strcmp(name, "ggml_backend_comm_init") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *) ggml_backend_hexagon_comm_init;
    }
    ଯଦି (strcmp(name, "ggml_backend_comm_free") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *) ggml_backend_hexagon_comm_free;
    }
    ଯଦି (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    ଫେରନ୍ତୁ ଖାଲି;
}

template<typename T> std::vector<T> str_to_vec(ଅଚଳ ଅକ୍ଷର* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    ଯେତେବେଳେ (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    ଫେରନ୍ତୁ v;
}

template<typename T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    ପାଇଁ (auto i : v) { ss << i << ','; }
    auto str = ss.str(); str.pop_back(); // drop last comma
    ଫେରନ୍ତୁ str;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, ତାର୍କିକ discovery_supported, ଅଚଳ std::unordered_map<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, fastrpc_domain> & cdsp_map) {
    ଯଦି (discovery_supported) {
        auto it = cdsp_map.find(cfg.physical_idx);
        ଯଦି (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } ଅନ୍ୟଥା {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } ଅନ୍ୟଥା {
        ବାଛ (cfg.physical_idx) {
            ସ୍ଥିତି 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                ଭାଙ୍ଗ;
            ସ୍ଥିତି 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                ଭାଙ୍ଗ;
        }
    }
    ପାଇଁ (auto & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_discover_devices() {
    std::unordered_map<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, fastrpc_domain> cdsp_map;
    ତାର୍କିକ discovery_supported = ମିଥ୍ୟା;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = remote_system_request(&domain_info);
    ଯଦି (err == AEE_SUCCESS && domain_info.sys.ଅଙ୍କ_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.ଅଙ୍କ_domains);
        domain_info.sys.domains     = domains.data();
        domain_info.sys.max_domains = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) domains.size();

        err = remote_system_request(&domain_info);
        ଯଦି (err == AEE_SUCCESS) {
            discovery_supported = ସତ୍ୟ;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_domains = std::min(domain_info.sys.ଅଙ୍କ_domains, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) domains.size());
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                ଯଦି (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) domains[i].type);
                    ଜାରି;
                }
                ଯଦି (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    ଜାରି;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } ଅନ୍ୟଥା {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (unsigned) err);
        }
    } ଅନ୍ୟଥା ଯଦି (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (unsigned) err);
    }

    // Populate domain IDs and names for all configured devices
    ପାଇଁ (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_Q4_0 == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_Q4_1 == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_Q8_0 == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_MXFP4 == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_IQ4_NL == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_Q4_K == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_Q5_K == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_Q6_K == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_Q3_K == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_Q2_K == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_Q1_0 == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_Q2_0 == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) HTP_TYPE_BF16 == (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    ଅଚଳ ଅକ୍ଷର * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    ଅଚଳ ଅକ୍ଷର * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    ଅଚଳ ଅକ୍ଷର * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    ଅଚଳ ଅକ୍ଷର * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    ଅଚଳ ଅକ୍ଷର * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    ଅଚଳ ଅକ୍ଷର * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    ଅଚଳ ଅକ୍ଷର * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    ଅଚଳ ଅକ୍ଷର * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    ଅଚଳ ଅକ୍ଷର * str_etm      = getenv("GGML_HEXAGON_ETM");
    ଅଚଳ ଅକ୍ଷର * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    ଅଚଳ ଅକ୍ଷର * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    ଅଚଳ ଅକ୍ଷର * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    ଅଚଳ ଅକ୍ଷର * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    ଅଚଳ ଅକ୍ଷର * str_fa_head_split = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    ଅଚଳ ଅକ୍ଷର * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    ଅଚଳ ଅକ୍ଷର * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    ଅଚଳ ଅକ୍ଷର * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    ଅଚଳ ଅକ୍ଷର * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    ଅଚଳ ଅକ୍ଷର * str_arch     = getenv("GGML_HEXAGON_ARCH");
    ଅଚଳ ଅକ୍ଷର * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    ଅଚଳ ଅକ୍ଷର * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    ଅଚଳ ଅକ୍ଷର * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    ଅଚଳ ଅକ୍ଷର * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    ଅଚଳ ଅକ୍ଷର * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    ଅଚଳ ଅକ୍ଷର * str_power_mode = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    ଯଦି (!str_arch) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        ଯଦି (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } ଅନ୍ୟଥା {
            ଯଦି (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } ଅନ୍ୟଥା ଯଦି (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } ଅନ୍ୟଥା {
        ଯଦି (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, ଖାଲି, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? ନୂତନ std::regex(str_opfilter, RE_ICASE) : ଖାଲି;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, ଖାଲି, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, ଖାଲି, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, ଖାଲି, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  ଖାଲି, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    ଯଦି (str_power_mode) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା m = htp_power_mode_from_str(str_power_mode);
        ଯଦି (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_mode);
        } ଅନ୍ୟଥା {
            opt_power_mode = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, ଖାଲି, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_head_split = str_fa_head_split ? atoi(str_fa_head_split)       : opt_fa_head_split;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    ଯଦି (str_mbuf) {
        ଅଚଳ ଅକ୍ଷର * p = str_mbuf;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx = 0; idx < 3 && p && *p; idx++) {
            ଯେତେବେଳେ (*p == ' ') p++;
            ଅଚଳ ଅକ୍ଷର * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : strlen(p);
            ଯେତେବେଳେ (len > 0 && p[len - 1] == ' ') len--;
            ଯଦି (len > 0) {
                std::string token(p, len);
                ଯଦି (idx == 0) opt_mbuf_dyn    = parse_size(token.c_str());
                ଯଦି (idx == 1) opt_mbuf_static = parse_size(token.c_str());
                ଯଦି (idx == 2) opt_mbuf_total  = parse_size(token.c_str());
            }
            ଯଦି (!comma) ଭାଙ୍ଗ;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_size(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    ଅଚଳ ଅକ୍ଷର * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    ଯଦି (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    ଯଦି (str_devices && str_devices[0] != '\0') {
        ତାର୍କିକ is_single_ସଂଖ୍ୟା = ସତ୍ୟ;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; str_devices[i] != '\0'; i++) {
            ଯଦି (!isdigit((unsigned ଅକ୍ଷର)str_devices[i])) {
                is_single_ସଂଖ୍ୟା = ମିଥ୍ୟା;
                ଭାଙ୍ଗ;
            }
        }
        ଯଦି (is_single_ସଂଖ୍ୟା) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = atoi(str_devices);
            ଯଦି (n < 1) n = 1;
            ଯଦି (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            ପାଇଁ (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.clear();
            }
        } ଅନ୍ୟଥା {
            std::string s_devices(str_devices);
            std::vector<std::string> ବସ୍ତୁଗୁଡ଼ିକ;
            std::string curr_item;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା bracket_depth = 0;
            ପାଇଁ (ଅକ୍ଷର ch : s_devices) {
                ଯଦି (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } ଅନ୍ୟଥା ଯଦି (ch == ']') {
                    ଯଦି (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } ଅନ୍ୟଥା ଯଦି (ch == ',' && bracket_depth == 0) {
                    size_t s = curr_item.find_first_not_of(" \t\r\n");
                    size_t e = curr_item.find_last_not_of(" \t\r\n");
                    ଯଦି (s != std::string::npos) {
                        ବସ୍ତୁଗୁଡ଼ିକ.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.clear();
                } ଅନ୍ୟଥା {
                    curr_item += ch;
                }
            }
            size_t s = curr_item.find_first_not_of(" \t\r\n");
            size_t e = curr_item.find_last_not_of(" \t\r\n");
            ଯଦି (s != std::string::npos) {
                ବସ୍ତୁଗୁଡ଼ିକ.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            ପାଇଁ (ଅଚଳ auto & item : ବସ୍ତୁଗୁଡ଼ିକ) {
                size_t b_open  = item.find('[');
                size_t b_close = item.rfind(']');

                ଯଦି (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    ଯଦି (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        ଚେଷ୍ଟା {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } ଧର (...) {
                            virt = 0;
                        }
                    } ଅନ୍ୟଥା {
                        size_t dev_colon = dev_name.find(':');
                        ଯଦି (dev_colon != std::string::npos) {
                            ଚେଷ୍ଟା {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } ଧର (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    ଯେତେବେଳେ (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        ଯଦି (ps == std::string::npos) ଜାରି;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        ଯଦି (dash_pos != std::string::npos) {
                            ଚେଷ୍ଟା {
                                ପୂର୍ଣ୍ଣସଂଖ୍ୟା p_start = std::stoi(p_part.substr(0, dash_pos));
                                ପୂର୍ଣ୍ଣସଂଖ୍ୟା p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା p = p_start; p <= p_end; p++) {
                                    ଯଦି (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } ଧର (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } ଅନ୍ୟଥା {
                            ଚେଷ୍ଟା {
                                ପୂର୍ଣ୍ଣସଂଖ୍ୟା p = std::stoi(p_part);
                                ଯଦି (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } ଧର (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    ଯଦି (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    ଯଦି (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        auto & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.clear();

                        ପାଇଁ (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_group.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } ଅନ୍ୟଥା {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } ଅନ୍ୟଥା ଯଦି (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    size_t colon_pos = rest.find(':');
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା phys = 0;
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା virt = 0;
                    ଚେଷ୍ଟା {
                        ଯଦି (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } ଅନ୍ୟଥା {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } ଧର (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        ଜାରି;
                    }

                    ଯଦି (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_group.clear();
                        opt_ndev++;
                    } ଅନ୍ୟଥା {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } ଅନ୍ୟଥା {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", item.c_str());
                }
            }
        }
    } ଅନ୍ୟଥା {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_group.clear();
    }

#ଯଦି defined(__ANDROID__)
    ଯଦି (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    ଯଦି (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            auto v  = str_to_vec<uint32_t>(str_profile);
            ବାଛ (v.size()) {
                ସ୍ଥିତି 1:  opt_profile = v[0]; ଫେରନ୍ତୁ opt_pmu_evt; // mode with default pmu events
                ସ୍ଥିତି 8:  opt_profile = 2;    ଫେରନ୍ତୁ v;           // mode with custom  pmu events
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ: opt_profile = 0;    ଫେରନ୍ତୁ {};          // garbage input
            }}();
        ଯଦି (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = ନୂତନ ggml_hexagon_registry(reg);
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_ଗଣନା,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(ଶୂନ୍ୟ) {
    ସ୍ଥିର ତାର୍କିକ initialized = ମିଥ୍ୟା;

    ସ୍ଥିର ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ ଖାଲି };

    {
        ସ୍ଥିର std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        ଯଦି (!initialized) {
            auto nErr = htpdrv_init();
            ଯଦି (nErr != AEE_SUCCESS) {
                ଫେରନ୍ତୁ ଖାଲି;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = ସତ୍ୟ;
    }

    ଫେରନ୍ତୁ &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
