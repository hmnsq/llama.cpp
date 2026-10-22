#define _USE_MATH_DEFINES

#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <inttypes.h>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <string.h>
#അന്തര്ഭാവ <time.h>

#അന്തര്ഭാവ <atomic>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <chrono>
#അന്തര്ഭാവ <mutex>
#അന്തര്ഭാവ <thread>
#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <stdexcept>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <sstream>
#അന്തര്ഭാവ <iomanip>
#അന്തര്ഭാവ <unordered_set>
#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <regex>
#അന്തര്ഭാവ <queue>
#അന്തര്ഭാവ <deque>
#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <initializer_list>

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
#അല്ലെങ്കിൽ
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"

#അന്തര്ഭാവ <AEEStdErr.h>
#അന്തര്ഭാവ <dspqueue.h>
#അന്തര്ഭാവ <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-alloc.h"
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "ggml-hexagon.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-quants.h"
#അന്തര്ഭാവ "htp-opnode.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp/matmul-ops.h"
#അന്തര്ഭാവ "htp/binary-ops.h"
#അന്തര്ഭാവ "htp/flash-attn-ops.h"
#അന്തര്ഭാവ "htp/unary-ops.h"
#അന്തര്ഭാവ "htp/get-rows-ops.h"
#അന്തര്ഭാവ "htp/set-rows-ops.h"
#അന്തര്ഭാവ "htp/softmax-ops.h"
#അന്തര്ഭാവ "htp/pool-ops.h"
#അന്തര്ഭാവ "htp/rope-ops.h"
#അന്തര്ഭാവ "htp/ssm-conv.h"
#അന്തര്ഭാവ "htp/gated-delta-net-ops.h"
#അന്തര്ഭാവ "htp/argsort-ops.h"
#അന്തര്ഭാവ "htp/concat-ops.h"
#അന്തര്ഭാവ "htp/cpy-ops.h"
#അന്തര്ഭാവ "htp/im2col-ops.h"
#അന്തര്ഭാവ "htp_iface.h"
#അന്തര്ഭാവ "htp-drv.h"
#അന്തര്ഭാവ "htp-repack.h"

using intvec  = std::vector<പൂർണ്ണസംഖ്യ>;
using uintvec = std::vector<unsigned പൂർണ്ണസംഖ്യ>;
using u32vec  = std::vector<uint32_t>;

#define GGML_HEXAGON_MAX_SESSIONS          16

#define GGML_HEXAGON_FENCE_SLOT_SIZE       128

ഘടന ggml_hexagon_device_config {
    പൂർണ്ണസംഖ്യ         physical_idx = 0;
    പൂർണ്ണസംഖ്യ         virtual_idx  = 0;
    പൂർണ്ണസംഖ്യ         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

സ്ഥിതം ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

സ്ഥിതം പൂർണ്ണസംഖ്യ    opt_arch    = 0; // autodetect
സ്ഥിതം size_t opt_ndev    = 1;
സ്ഥിതം size_t opt_nhvx    = 0; // use all
സ്ഥിതം പൂർണ്ണസംഖ്യ    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
സ്ഥിതം size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
സ്ഥിതം പൂർണ്ണസംഖ്യ    opt_etm     = 0;
സ്ഥിതം പൂർണ്ണസംഖ്യ    opt_verbose = 0;
സ്ഥിതം പൂർണ്ണസംഖ്യ    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
സ്ഥിതം തർക്കം   opt_hostbuf = അസത്യം;
സ്ഥിതം തർക്കം   opt_dma64   = അസത്യം;
സ്ഥിതം പൂർണ്ണസംഖ്യ    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start

സ്ഥിതം സ്ഥിരം std::pair<സ്ഥിരം അക്ഷരം *, പൂർണ്ണസംഖ്യ> htp_power_mode_names[] = {
    { "low_power_saver",            HTP_POWER_MODE_താഴെയുള്ള_POWER_SAVER },
    { "power_saver",                HTP_POWER_MODE_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_MODE_മുകളിലെ_POWER_SAVER },
    { "low_balanced",               HTP_POWER_MODE_താഴെയുള്ള_BALANCED },
    { "balanced",                   HTP_POWER_MODE_BALANCED },
    { "high_performance",           HTP_POWER_MODE_മുകളിലെ_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_MODE_SUSTAINED_മുകളിലെ_PERFORMANCE },
    { "burst",                      HTP_POWER_MODE_BURST },
    { "default",                    HTP_POWER_MODE_BURST }, // flip this to change the default mode
};

സ്ഥിതം പൂർണ്ണസംഖ്യ htp_power_mode_from_str(സ്ഥിരം അക്ഷരം * s) {
    വേണ്ടി (auto & e : htp_power_mode_names) {
#എങ്കിൽ defined(_MSC_VER)
        എങ്കിൽ (_stricmp(s, e.first) == 0) {
#അല്ലെങ്കിൽ
        എങ്കിൽ (strcasecmp(s, e.first) == 0) {
#endif
            തിരികെ e.second;
        }
    }
    തിരികെ -1;
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * htp_power_mode_to_str(പൂർണ്ണസംഖ്യ mode) {
    വേണ്ടി (auto & e : htp_power_mode_names) {
        എങ്കിൽ (e.second == mode) {
            തിരികെ e.first;
        }
    }
    തിരികെ "unknown";
}

സ്ഥിതം size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
സ്ഥിതം size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
സ്ഥിതം size_t opt_mbuf_total  = 0;                        // total buffer space limit (0 = unconstrained)

സ്ഥിതം പൂർണ്ണസംഖ്യ    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
സ്ഥിതം പൂർണ്ണസംഖ്യ    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
സ്ഥിതം പൂർണ്ണസംഖ്യ    opt_fa_head_split = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
സ്ഥിതം പൂർണ്ണസംഖ്യ    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
സ്ഥിതം പൂർണ്ണസംഖ്യ    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
സ്ഥിതം പൂർണ്ണസംഖ്യ    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
സ്ഥിതം u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

സ്ഥിതം പൂർണ്ണസംഖ്യ opt_opbatch  = 1280; // max number of ops in a batch
സ്ഥിതം പൂർണ്ണസംഖ്യ opt_opqueue  = 8;   // max number of pending batches
സ്ഥിതം പൂർണ്ണസംഖ്യ opt_optrace  = 0;    // trace buffer size per thread (0 means default)
സ്ഥിതം പൂർണ്ണസംഖ്യ opt_oppoll   = 0;    // polling for batch completions
സ്ഥിതം പൂർണ്ണസംഖ്യ opt_opfusion  = 1;    // enable/disable op fusion
സ്ഥിതം പൂർണ്ണസംഖ്യ opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

എണ്ണൽ ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ADD = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ADD   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

സ്ഥിതം വരിയിൽ തർക്കം ggml_hexagon_is_fusion_enabled(പൂർണ്ണസംഖ്യ flag) {
    എങ്കിൽ (opt_opfusion <= 0) തിരികെ അസത്യം;
    എങ്കിൽ (opt_opfusion == 1) തിരികെ സത്യം; // 1 enables all
    തിരികെ (opt_opfusion & flag) != 0;
}

സ്ഥിതം std::regex* opt_opfilter = ശൂന്യ; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    എങ്കിൽ (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

സ്ഥിതം സ്ഥിരം അക്ഷരം * status_to_str(uint32_t status) {
    തിരഞ്ഞെടുപ്പ് (status) {
        അവസ്ഥ HTP_STATUS_OK:
            തിരികെ "OK";
        അവസ്ഥ HTP_STATUS_NO_SUPPORT:
            തിരികെ "NO-SUPPORT";
        അവസ്ഥ HTP_STATUS_INVAL_PARAMS:
            തിരികെ "INVAL-PARAMS";
        അവസ്ഥ HTP_STATUS_VTCM_TOO_SMALL:
            തിരികെ "VTCM-TOO-SMALL";
        അവസ്ഥ HTP_STATUS_INTERNAL_ERR:
            തിരികെ "INTERNAL-ERROR";
        സ്ഥിരസ്ഥിതി:
            തിരികെ "UNKNOWN";
    }
}

// ** debug helpers

സ്ഥിതം ശൂന്യം ggml_hexagon_dump_op_exec(സ്ഥിരം std::string &sess_name, സ്ഥിരം htp_opnode & കണ്ണി, സ്ഥിരം uint32_t req_flags) {
    എങ്കിൽ (!opt_verbose) തിരികെ;

    htp_opformat fmt(കണ്ണി);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                കണ്ണി.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

സ്ഥിതം ശൂന്യം ggml_hexagon_dump_op_supp(സ്ഥിരം std::string &sess_name, സ്ഥിരം ഘടന ggml_tensor * op, തർക്കം supp) {
    എങ്കിൽ (!opt_verbose) തിരികെ;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * htp_event_name(uint16_t id) {
    തിരഞ്ഞെടുപ്പ് (id) {
        അവസ്ഥ HTP_TRACE_EVT_DMA:            തിരികെ "DMA";
        അവസ്ഥ HTP_TRACE_EVT_HVX_COMP:       തിരികെ "HVX_COMP";
        അവസ്ഥ HTP_TRACE_EVT_HVX_A_QUANT:    തിരികെ "HVX_A_QUANT";
        അവസ്ഥ HTP_TRACE_EVT_HVX_A_PREP:     തിരികെ "HVX_A_PREP";
        അവസ്ഥ HTP_TRACE_EVT_HVX_W_DEQUANT:  തിരികെ "HVX_W_DEQUANT";
        അവസ്ഥ HTP_TRACE_EVT_HVX_W_PREP:     തിരികെ "HVX_W_PREP";
        അവസ്ഥ HTP_TRACE_EVT_HVX_O_PROC:     തിരികെ "HVX_O_PROC";
        അവസ്ഥ HTP_TRACE_EVT_HVX_FA_QK:      തിരികെ "HVX_QK_FA";
        അവസ്ഥ HTP_TRACE_EVT_HVX_FA_SFM:     തിരികെ "HVX_SFM_FA";
        അവസ്ഥ HTP_TRACE_EVT_HVX_FA_Q_PREP:  തിരികെ "HVX_Q_PREP";
        അവസ്ഥ HTP_TRACE_EVT_HVX_FA_K_PREP:  തിരികെ "HVX_K_PREP";
        അവസ്ഥ HTP_TRACE_EVT_HVX_FA_V_PREP:  തിരികെ "HVX_V_PREP";
        അവസ്ഥ HTP_TRACE_EVT_HVX_GDN_PREP:   തിരികെ "HVX_GDN_PREP";
        അവസ്ഥ HTP_TRACE_EVT_HVX_GDN_SOLVE:  തിരികെ "HVX_GDN_SOLVE";
        അവസ്ഥ HTP_TRACE_EVT_HVX_GDN_V_PREP: തിരികെ "HVX_GDN_V_PREP";
        അവസ്ഥ HTP_TRACE_EVT_HVX_GDN_D_PREP: തിരികെ "HVX_GDN_D_PREP";
        അവസ്ഥ HTP_TRACE_EVT_HVX_GDN_OUT:    തിരികെ "HVX_GDN_OUT";
        അവസ്ഥ HTP_TRACE_EVT_HVX_GDN_STATE:  തിരികെ "HVX_GDN_STATE";
        അവസ്ഥ HTP_TRACE_EVT_HVX_GDN_REM:    തിരികെ "HVX_GDN_REM";
        അവസ്ഥ HTP_TRACE_EVT_HMX_COMP:       തിരികെ "HMX_COMP";
        അവസ്ഥ HTP_TRACE_EVT_L2FLUSH:        തിരികെ "L2FLUSH";
        അവസ്ഥ HTP_TRACE_EVT_INIT:           തിരികെ "INIT";
        അവസ്ഥ HTP_TRACE_EVT_BUFF:           തിരികെ "BUFF";
        അവസ്ഥ HTP_TRACE_EVT_FENCE:          തിരികെ "FENCE";
        സ്ഥിരസ്ഥിതി:                           തിരികെ "UNKNOWN";
    }
}

സ്ഥിതം ശൂന്യം ggml_hexagon_dump_op_prof(സ്ഥിരം std::string &sess_name, സ്ഥിരം htp_opnode & കണ്ണി, സ്ഥിരം htp_prof_desc & pd) {
    എങ്കിൽ (!opt_profile) തിരികെ;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    സ്ഥിരം uint32_t * pmu = pd.pmu;

    അക്ഷരം pmu_str[256] = "";
    എങ്കിൽ (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, വലുപ്പം(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(കണ്ണി);
    ദശാംശം mhz = op_usec > 0 ? (ദശാംശം) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            കണ്ണി.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

സ്ഥിതം ശൂന്യം ggml_hexagon_dump_batch_prof(സ്ഥിരം std::string & sess_name, സ്ഥിരം htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    ദശാംശം batch_mhz = rsp.usecs > 0 ? (ദശാംശം) batch_cycles / rsp.usecs : 0.0f;

    അക്ഷരം evt_str[256] = "----";
    എങ്കിൽ (opt_profile == 3) {
        snprintf(evt_str, വലുപ്പം(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (unsigned long long) batch_cycles, (unsigned long long) rsp.cycles_start, batch_mhz);
}

സ്ഥിതം ശൂന്യം ggml_hexagon_dump_trace_events(സ്ഥിരം std::string & sess_name, സ്ഥിരം htp_opbatch_rsp & rsp,
                                           സ്ഥിരം htp_trace_desc * trace_events, uint32_t n_traces) {
    എങ്കിൽ (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        വേണ്ടി (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        വേണ്ടി (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            വേണ്ടി (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                സ്ഥിരം auto & e = trace_events[t * n_traces + idx];
                തർക്കം is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

എണ്ണൽ ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_WEIGHT    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

ഘടന ggml_hexagon_session;

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_matmul_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_mm_kernel_params * kparams
);

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_fused_matmul_add_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * src2,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_mm_kernel_params * kparams
);

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_unary_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    uint32_t op,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_unary_kernel_params * kparams
);

സ്ഥിതം തർക്കം ggml_hexagon_precompute_ദ്വിമാന_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    uint32_t op,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_ദ്വിമാന_kernel_params * kparams,
    സ്ഥിരം ഘടന ggml_tensor * src2 = nullptr
);

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_get_rows_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_get_rows_kernel_params * kparams
);

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_set_rows_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_set_rows_kernel_params * kparams
);

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_softmax_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_softmax_kernel_params * kparams
);

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_rope_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_rope_kernel_params * kparams
);

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_ssm_conv_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_ssm_conv_kernel_params * kparams
);

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_gated_delta_net_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_gdn_kernel_params * kparams
);

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_sort_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    തർക്കം is_മുകൾ_k,
    ഘടന htp_sort_kernel_params * kparams
);

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_pool_2d_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_pool_2d_kernel_params * kparams,
    തർക്കം is_pool_1d
);

സ്ഥിതം തർക്കം ggml_hexagon_precompute_concat_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_concat_kernel_params * kparams
);

സ്ഥിതം തർക്കം ggml_hexagon_precompute_cpy_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_copy_kernel_params * kparams
);

സ്ഥിതം തർക്കം ggml_hexagon_precompute_im2col_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_im2col_kernel_params * kparams
);

സ്ഥിതം തർക്കം ggml_hexagon_precompute_im2col_3d_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_im2col_kernel_params * kparams
);

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_fused_mmnx_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * act,
    int32_t n_weights,
    ഘടന htp_mm_kernel_params * kparams
);

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_fused_mmidnx_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * act,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    int32_t n_weights,
    ഘടന htp_mm_kernel_params * kparams
);

സ്ഥിതം തർക്കം ggml_hexagon_precompute_allreduce_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    തർക്കം has_add,
    തർക്കം is_row_bcast,
    തർക്കം is_shard_ok,
    ഘടന htp_allreduce_kernel_params * kparams
);

സ്ഥിതം തർക്കം ggml_hexagon_rows_stride(സ്ഥിരം int64_t * ne, സ്ഥിരം size_t * nb, size_t * stride);
സ്ഥിതം തർക്കം ggml_hexagon_matmul_can_collapse(സ്ഥിരം ഘടന ggml_tensor * src0, സ്ഥിരം ഘടന ggml_tensor * src1, സ്ഥിരം ഘടന ggml_tensor * dst);
സ്ഥിതം ggml_tensor ggml_hexagon_tensor_collapse_rows(സ്ഥിരം ഘടന ggml_tensor * t);

സ്ഥിതം തർക്കം mm_is_hmx_eligible(സ്ഥിരം ggml_tensor * t);
സ്ഥിതം htp_op_code op_remap_to_htp(സ്ഥിരം ggml_tensor * t);
സ്ഥിതം തർക്കം is_supported_mul_mat_nx_kernel(സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ഘടന htp_mm_kernel_params * kparams);
സ്ഥിതം തർക്കം is_supported_mul_mat_id_nx_kernel(സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ഘടന htp_mm_kernel_params * kparams);
സ്ഥിതം തർക്കം is_mergeable_mul_mat(സ്ഥിരം ggml_tensor * t);
സ്ഥിതം തർക്കം is_mergeable_mul_mat_pair(സ്ഥിരം ggml_tensor * n1, സ്ഥിരം ggml_tensor * n2);
സ്ഥിതം തർക്കം is_mergeable_mul_mat_id(സ്ഥിരം ggml_tensor * t);
സ്ഥിതം തർക്കം is_mergeable_mul_mat_id_pair(സ്ഥിരം ggml_tensor * n1, സ്ഥിരം ggml_tensor * n2);

// ** backend sessions

ഘടന ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_size { 0 };
    uint32_t             flags { 0 };
};

സ്ഥിതം വരിയിൽ തർക്കം ggml_hexagon_tensor_is_fuseable(സ്ഥിരം ഘടന ggml_tensor * t) {
    എങ്കിൽ (!t->extra) തിരികെ അസത്യം;
    auto extra = (സ്ഥിരം ഘടന ggml_hexagon_tensor_extra *) t->extra;
    തിരികെ (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

സ്ഥിതം വരിയിൽ തർക്കം ggml_hexagon_can_rt_repack(സ്ഥിരം ഘടന ggml_tensor * t) {
    എങ്കിൽ (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (t->buffer && ggml_backend_buft_is_host(t->buffer->buft)) {
        തിരികെ അസത്യം;
    }
    തിരികെ opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

സ്ഥിതം വരിയിൽ തർക്കം ggml_hexagon_tensor_is_tiled(സ്ഥിരം ഘടന ggml_tensor * t) {
    സ്ഥിരം auto * extra = (സ്ഥിരം ggml_hexagon_tensor_extra *) t->extra;
    എങ്കിൽ (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        തിരികെ സത്യം;
    }
    തിരികെ ggml_hexagon_can_rt_repack(t);
}

സ്ഥിതം വരിയിൽ തർക്കം ggml_hexagon_needs_repack(സ്ഥിരം ഘടന ggml_tensor * t) {
    സ്ഥിരം auto * extra = (സ്ഥിരം ggml_hexagon_tensor_extra *) t->extra;
    എങ്കിൽ (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        തിരികെ അസത്യം;
    }
    തിരികെ ggml_hexagon_can_rt_repack(t);
}

സ്ഥിതം വരിയിൽ തർക്കം ggml_hexagon_tensors_overlap(സ്ഥിരം ഘടന ggml_tensor * a, സ്ഥിരം ഘടന ggml_tensor * b) {
    സ്ഥിരം uintptr_t a0 = (uintptr_t) a->data;
    സ്ഥിരം uintptr_t b0 = (uintptr_t) b->data;
    സ്ഥിരം uintptr_t a1 = a0 + ggml_nbytes(a);
    സ്ഥിരം uintptr_t b1 = b0 + ggml_nbytes(b);

    തിരികെ a0 < b1 && b0 < a1;
}

സ്ഥിതം വരിയിൽ തർക്കം ggml_hexagon_can_row_partition(സ്ഥിരം ഘടന ggml_tensor * t) {
    എങ്കിൽ (t->ne[1] > 1 && (t->nb[1] & 127) != 0) തിരികെ അസത്യം;
    എങ്കിൽ (t->ne[2] > 1 && (t->nb[2] & 127) != 0) തിരികെ അസത്യം;
    എങ്കിൽ (t->ne[3] > 1 && (t->nb[3] & 127) != 0) തിരികെ അസത്യം;
    തിരികെ സത്യം;
}

ഘടന htp_opnode;

ഘടന ggml_hexagon_opbatch;
ഘടന ggml_hexagon_opqueue;
ഘടന ggml_hexagon_shared_buffer;
ഘടന ggml_hexagon_fence_buffer;
ഘടന ggml_hexagon_session;
ഘടന ggml_backend_hexagon_device_context;

ഘടന ggml_hexagon_mdev_group {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

ഘടന ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    അസ്ഥിര uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

ഘടന ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    അസ്ഥിര uint32_t *    fence_slot   = nullptr;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

ഘടന ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    പൂർണ്ണസംഖ്യ              phys_idx;
    പൂർണ്ണസംഖ്യ              virt_idx;
    തർക്കം             valid_session;
    തർക്കം             valid_handle;
    തർക്കം             valid_queue;
    തർക്കം             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<പൂർണ്ണസംഖ്യ, std::unique_ptr<ggml_hexagon_shared_buffer>> cloned_buffers;
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

    ggml_hexagon_session(സ്ഥിരം ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_എണ്ണം = 0) noexcept(അസത്യം);
    ~ggml_hexagon_session() noexcept(സത്യം);

    സ്ഥിരം അക്ഷരം* c_name() സ്ഥിരം { തിരികെ name.c_str(); }

    ശൂന്യം allocate(സ്ഥിരം ggml_hexagon_device_config & config) noexcept(അസത്യം);
    ശൂന്യം release() noexcept(സത്യം);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    ശൂന്യം      free_fence(ശൂന്യം * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, അസ്ഥിര uint32_t *> cpy_fence_slots;

    ശൂന്യം enqueue_mdev_group();
    ശൂന്യം enqueue_op(സ്ഥിരം htp_opnode & കണ്ണി);
    ശൂന്യം enqueue_cpy(സ്ഥിരം ggml_tensor * src, ggml_tensor * dst, സ്ഥിരം ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    ശൂന്യം enqueue_fence(സ്ഥിരം ggml_tensor * sync_tensor, uint32_t fence_seq = 0, തർക്കം wait = സത്യം);
    ശൂന്യം enqueue_allreduce(സ്ഥിരം ggml_tensor * dst, സ്ഥിരം std::vector<സ്ഥിരം ggml_tensor *> & src_tensors,
                           സ്ഥിരം std::vector<സ്ഥിരം ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);

    ശൂന്യം start_batch();
    ശൂന്യം flush_sync(തർക്കം all = സത്യം);
    ശൂന്യം flush_async();
    ശൂന്യം flush_batch(size_t min_ops = 1);
    ശൂന്യം flush_peers();
    ശൂന്യം flush_pending(തർക്കം all = സത്യം);

    ggml_hexagon_shared_buffer * mmap_tensor(സ്ഥിരം ggml_tensor * t);
    തർക്കം clone_buffer(സ്ഥിരം ggml_hexagon_shared_buffer*);
    ശൂന്യം release_buffer(സ്ഥിരം ggml_hexagon_shared_buffer*);
    ശൂന്യം unclone_buffer(സ്ഥിരം ggml_hexagon_shared_buffer*);

    ശൂന്യം add_peer(ggml_hexagon_session * peer) {
        എങ്കിൽ (ഇത്->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } അല്ലെങ്കിൽ {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

ഘടന ggml_backend_hexagon_device_context {
    പൂർണ്ണസംഖ്യ                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_buffer_type buffer_type       = {};
    ggml_backend_buffer_type host_buffer_type  = {};
    ggml_backend_buffer_type fence_buffer_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(പൂർണ്ണസംഖ്യ dev_id, സ്ഥിരം ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    സ്ഥിരം അക്ഷരം * c_name() സ്ഥിരം { തിരികെ config.name.c_str(); }

    ggml_hexagon_session * session() {
        എങ്കിൽ (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        തിരികെ sess.get();
    }
};

ഘടന ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(സ്ഥിരം std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        ഇത്->dev_ctx = dev_ctx;
        ഇത്->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

ഘടന ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    പൂർണ്ണസംഖ്യ       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        എങ്കിൽ (!base) {
            എറിയുക std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        എങ്കിൽ (fd < 0) {
            rpcmem_free(base);
            എറിയുക std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        ഇത്->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        എങ്കിൽ (base) {
            rpcmem_free(base);
        }
    }
};

ഘടന ggml_hexagon_shared_buffer {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    തർക്കം     mapped;
    തർക്കം     pinned;
    തർക്കം     extended;

    സ്ഥിരം അക്ഷരം * c_name() സ്ഥിരം { തിരികെ sess->c_name(); }
    uint8_t *    base()   സ്ഥിരം { തിരികെ mem ? mem->base : nullptr; }
    size_t       size()   സ്ഥിരം { തിരികെ mem ? mem->size : 0;  }
    പൂർണ്ണസംഖ്യ          fd()     സ്ഥിരം { തിരികെ mem ? mem->fd   : -1; }

    ശൂന്യം mmap(തർക്കം extended = അസത്യം) {
        എങ്കിൽ (!ഇത്->mem)   തിരികെ;
        എങ്കിൽ (ഇത്->mapped) തിരികെ;

        GGML_ASSERT(!ഇത്->pinned || !extended);

        ഇത്->extended = extended;
        fastrpc_map_flags flags = ഇത്->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        പൂർണ്ണസംഖ്യ err = fastrpc_mmap(sess->domain_id, fd(), (ശൂന്യം *) base(), 0, size(), flags);
        എങ്കിൽ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (unsigned) err);
            എറിയുക std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (ശൂന്യം *) base(), size(), fd(), pinned, extended);

        ഇത്->mapped = സത്യം;
    }

    ശൂന്യം unmap() {
        എങ്കിൽ (!ഇത്->mapped) തിരികെ;

        എങ്കിൽ (!ഇത്->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        എങ്കിൽ (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (ശൂന്യം *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (ശൂന്യം *) base(), size(), fd());

        ഇത്->mapped = അസത്യം;
    }

    ശൂന്യം alloc(size_t size) {
        എങ്കിൽ (ഇത്->mem) തിരികെ;

        ഇത്->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (ശൂന്യം *) base(), ഇത്->size(), fd(), (പൂർണ്ണസംഖ്യ) pinned);
        എങ്കിൽ (ഇത്->pinned) {
            mmap();
        }
    }

    ശൂന്യം സ്വതന്ത്രം() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (ശൂന്യം *) base(), size(), fd());
        ഇത്->mem  = nullptr;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, തർക്കം pinned = അസത്യം) {
        ഇത്->sess     = sess;
        ഇത്->mapped   = അസത്യം;
        ഇത്->pinned   = pinned;
        ഇത്->extended = അസത്യം;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_offset = (size + 4095) & ~4095;
        size_t total_size   = guard_offset + 4096;
        എങ്കിൽ (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            total_size = (total_size + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(total_size);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, സ്ഥിരം ggml_hexagon_shared_buffer & other) {
        ഇത്->sess   = sess;
        ഇത്->mem    = other.mem;
        ഇത്->mapped = അസത്യം;
        ഇത്->pinned = other.pinned;
        ഇത്->extended = other.extended;
    }

    ~ggml_hexagon_shared_buffer() {
        സ്വതന്ത്രം();
        വേണ്ടി (auto * extra : tensor_extra) {
            delete extra;
        }
    }
};

ഘടന ggml_hexagon_fence_buffer : പൊതു ggml_hexagon_shared_buffer {
    uint32_t              slot_എണ്ണം = 0;
    uint32_t              slot_head  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_buffer   backend_buffer{};

    ggml_hexagon_fence_buffer(ggml_hexagon_session * sess, ggml_backend_buffer_type_t buft, size_t size)
        : ggml_hexagon_shared_buffer(sess, size, അസത്യം /* pinned */),
          slot_എണ്ണം(size / GGML_HEXAGON_FENCE_SLOT_SIZE),
          slot_head(0) {
        backend_buffer.buft    = buft;
        backend_buffer.context = static_cast<ggml_hexagon_shared_buffer *>(ഇത്);
        backend_buffer.size    = size;
        mmap(അസത്യം);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        എങ്കിൽ (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        } അല്ലെങ്കിൽ എങ്കിൽ (slot_head + n_slots <= slot_എണ്ണം) {
            uint32_t slot = slot_head;
            slot_head += n_slots;
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        }
        എങ്കിൽ (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_SLOT_SIZE);
        }
        തിരികെ ptr;
    }

    ശൂന്യം free_slot(ശൂന്യം * ptr, uint32_t n_slots = 1) {
        എങ്കിൽ (!ptr) തിരികെ;
        uint32_t slot = ((uint8_t *) ptr - base()) / GGML_HEXAGON_FENCE_SLOT_SIZE;
        വേണ്ടി (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

വരിയിൽ uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    തിരികെ ptr;
}

വരിയിൽ ശൂന്യം ggml_hexagon_session::free_fence(ശൂന്യം * ptr, uint32_t n_slots) {
    എങ്കിൽ (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    sbuf->sess->unclone_buffer(sbuf);
    delete sbuf;
}

സ്ഥിതം ശൂന്യം * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    തിരികെ sbuf->base();
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (ശൂന്യം *) sbuf->base(), tensor->data, ggml_nbytes(tensor));

    auto extra = പുതിയ ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    തിരികെ GGML_STATUS_SUCCESS;
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   സ്ഥിരം ശൂന്യം *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    എങ്കിൽ (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        എങ്കിൽ (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, data, offset, size, (പൂർണ്ണസംഖ്യ) buffer->usage, extra->flags);

    എങ്കിൽ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy((അക്ഷരം *) tensor->data + offset, data, size);
        തിരികെ;
    }

    എങ്കിൽ (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        htp_repack_tensor_tiled(tensor, data, size);
        തിരികെ;
    }

    എങ്കിൽ (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    memcpy(extra->shadow_buf.data() + offset, data, size);
    extra->shadow_size += size;

    എങ്കിൽ (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   സ്ഥിരം ggml_tensor *   tensor,
                                                   ശൂന്യം *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, data, offset, size, (പൂർണ്ണസംഖ്യ) buffer->usage, extra->flags);

    എങ്കിൽ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy(data, (സ്ഥിരം അക്ഷരം *) tensor->data + offset, size);
        തിരികെ;
    }

    htp_unpack_tensor_tiled(data, tensor, offset, size);
}

സ്ഥിതം തർക്കം ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   സ്ഥിരം ഘടന ggml_tensor * src,
                                                   ഘടന ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    തിരികെ അസത്യം;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                          ggml_tensor *         tensor,
                                                          സ്ഥിരം ശൂന്യം *          data,
                                                          size_t                offset,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    എങ്കിൽ (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        എങ്കിൽ (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (പൂർണ്ണസംഖ്യ) buffer->usage, extra->flags);

    എങ്കിൽ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        വേണ്ടി (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (സ്ഥിരം uint8_t *) data + i * stride_data, size);
        }
        തിരികെ;
    }

    എങ്കിൽ (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    വേണ്ടി (size_t i = 0; i < n_copies; i++) {
        memcpy(extra->shadow_buf.data() + offset + i * stride_tensor, (സ്ഥിരം uint8_t *) data + i * stride_data, size);
    }
    extra->shadow_size += n_copies * size;

    എങ്കിൽ (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                      സ്ഥിരം ggml_tensor *   tensor,
                                                      ശൂന്യം *                data,
                                                      size_t                offset,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (പൂർണ്ണസംഖ്യ) buffer->usage);

    എങ്കിൽ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        വേണ്ടി (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *)data + i * stride_data, (സ്ഥിരം uint8_t *)tensor->data + offset + i * stride_tensor, size);
        }
        തിരികെ;
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

    വേണ്ടി (size_t i = 0; i < n_copies; i++) {
        memcpy((uint8_t *) data + i * stride_data, temp_buf.data() + i * stride_tensor, size);
    }
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (ശൂന്യം *) sbuf->base(), sbuf->size());
    memset(sbuf->base(), value, sbuf->size());
}

സ്ഥിതം ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ ശൂന്യ,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ ശൂന്യ,
};

// ** backend buffer type

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_host_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                        ggml_tensor *         tensor,
                                                        സ്ഥിരം ശൂന്യം *          data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy((അക്ഷരം *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                        സ്ഥിരം ggml_tensor *   tensor,
                                                        ശൂന്യം *                data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy(data, (സ്ഥിരം അക്ഷരം *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

സ്ഥിതം ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ ശൂന്യ,
    /* .set_tensor      = */ ggml_backend_hexagon_host_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_buffer_get_tensor,
    /* .set_tensor_2d   = */ ശൂന്യ,
    /* .get_tensor_2d   = */ ശൂന്യ,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ ശൂന്യ,
};

// ** backend buffer type

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    തിരികെ static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    എങ്കിൽ (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        തിരികെ nullptr;
    }
    ശ്രമിക്കുക {
        ggml_hexagon_shared_buffer * sbuf = പുതിയ ggml_hexagon_shared_buffer(sess, size, അസത്യം);
        തിരികെ ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } പിടിക്കുക (സ്ഥിരം std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        തിരികെ nullptr;
    }
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    എങ്കിൽ (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        തിരികെ nullptr;
    }
    ശ്രമിക്കുക {
        ggml_hexagon_shared_buffer * sbuf = പുതിയ ggml_hexagon_shared_buffer(sess, size, അസത്യം);
        തിരികെ ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_host_buffer_interface, sbuf, size);
    } പിടിക്കുക (സ്ഥിരം std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        തിരികെ nullptr;
    }
}

സ്ഥിതം size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    തിരികെ 128;  // HVX alignment
    GGML_UNUSED(buft);
}

സ്ഥിതം size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, സ്ഥിരം ഘടന ggml_tensor * t) {
    എങ്കിൽ (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_size = htp_tiled_row_size(t->type, ne0) * ne1 * ne2 * ne3;
        തിരികെ std::max(tiled_size, ggml_nbytes(t));
    }
    തിരികെ ggml_nbytes(t);
}

സ്ഥിതം size_t parse_size(സ്ഥിരം അക്ഷരം * str, size_t default_unit = 1024 * 1024) {
    എങ്കിൽ (!str || str[0] == '\0') {
        തിരികെ 0;
    }
    അക്ഷരം * end = ശൂന്യ;
    ഇരട്ട val = strtod(str, &end);
    എങ്കിൽ (val < 0) {
        തിരികെ 0;
    }
    എങ്കിൽ (end && *end) {
        ആകുമ്പോൾ (*end == ' ') end++;
        എങ്കിൽ (*end == 'k' || *end == 'K') {
            തിരികെ (size_t) (val * 1024);
        } അല്ലെങ്കിൽ എങ്കിൽ (*end == 'm' || *end == 'M') {
            തിരികെ (size_t) (val * 1024 * 1024);
        } അല്ലെങ്കിൽ എങ്കിൽ (*end == 'g' || *end == 'G') {
            തിരികെ (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    തിരികെ (size_t) (val * default_unit);
}

സ്ഥിതം size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    തിരികെ opt_mbuf_dyn;
    GGML_UNUSED(buft);
}

സ്ഥിതം തർക്കം ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    തിരികെ അസത്യം;
    GGML_UNUSED(buft);
}

സ്ഥിതം തർക്കം ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    തിരികെ സത്യം;
    GGML_UNUSED(buft);
}

ഘടന ggml_backend_hexagon_alloc_buffer_n_plan_item {
    size_t size;
    പൂർണ്ണസംഖ്യ    first;
    പൂർണ്ണസംഖ്യ    last;
};

using ggml_backend_hexagon_alloc_buffer_n_plan_t = std::vector<ggml_backend_hexagon_alloc_buffer_n_plan_item>;

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_hexagon_kv_layer_suffix(സ്ഥിരം ഘടന ggml_tensor * t) {
    എങ്കിൽ (strncmp(t->name, "cache_", 6) != 0) {
        തിരികെ ശൂന്യ;
    }
    സ്ഥിരം അക്ഷരം * p = strstr(t->name, "_l");
    എങ്കിൽ (!p || !isdigit((unsigned അക്ഷരം)p[2])) {
        തിരികെ ശൂന്യ;
    }
    തിരികെ p;
}

ഘടന ggml_backend_hexagon_alloc_unit {
    size_t size;
    പൂർണ്ണസംഖ്യ    first;
    പൂർണ്ണസംഖ്യ    last;
};

സ്ഥിതം ggml_backend_hexagon_alloc_buffer_n_plan_t ggml_backend_hexagon_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, ഘടന ggml_tensor ** tensors, പൂർണ്ണസംഖ്യ n_tensors) {
    ggml_backend_hexagon_alloc_buffer_n_plan_t plan;

    സ്ഥിരം size_t alignment = ggml_backend_buft_get_alignment(buft);
    സ്ഥിരം size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    പൂർണ്ണസംഖ്യ i = 0;
    ആകുമ്പോൾ (i < n_tensors) {
        ഘടന ggml_tensor * t = tensors[i];
        size_t unit_size = 0;
        പൂർണ്ണസംഖ്യ unit_first = i;
        പൂർണ്ണസംഖ്യ unit_last = i + 1;

        എങ്കിൽ (t->data == ശൂന്യ && t->view_src == ശൂന്യ) {
            unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        സ്ഥിരം അക്ഷരം * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        ആകുമ്പോൾ (unit_last < n_tensors) {
            ഘടന ggml_tensor * next = tensors[unit_last];

            എങ്കിൽ (next->view_src != ശൂന്യ) {
                unit_last++;
                തുടരുക;
            }

            എങ്കിൽ (layer_suffix != ശൂന്യ) {
                സ്ഥിരം അക്ഷരം * next_suffix = ggml_hexagon_kv_layer_suffix(next);
                എങ്കിൽ (next_suffix != ശൂന്യ && strcmp(layer_suffix, next_suffix) == 0) {
                    എങ്കിൽ (next->data == ശൂന്യ) {
                        unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, next), alignment);
                    }
                    unit_last++;
                    തുടരുക;
                }
            }

            നിർത്തുക;
        }

        units.push_back({ unit_size, unit_first, unit_last });
        i = unit_last;
    }

    size_t cur_buf_size  = 0;
    പൂർണ്ണസംഖ്യ    cur_buf_first = 0;

    വേണ്ടി (സ്ഥിരം auto & unit : units) {
        എങ്കിൽ (unit.size == 0) {
            തുടരുക;
        }

        എങ്കിൽ (cur_buf_size > 0 && (cur_buf_size + unit.size) > max_size) {
            plan.push_back({ cur_buf_size, cur_buf_first, unit.first });
            cur_buf_size  = 0;
            cur_buf_first = unit.first;
        }

        cur_buf_size += unit.size;
    }

    എങ്കിൽ (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, cur_buf_first, n_tensors });
    }

    തിരികെ plan;
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, ഘടന ggml_tensor ** tensors, പൂർണ്ണസംഖ്യ n_tensors) {
    സ്ഥിരം ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    വേണ്ടി (സ്ഥിരം auto & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        എങ്കിൽ (buffer == ശൂന്യ) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            വേണ്ടി (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            തിരികെ ശൂന്യ;
        }

        ഘടന ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        ഘടന ggml_tensor * t_failed = ശൂന്യ;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = item.first; j < item.last; j++) {
            ഘടന ggml_tensor * t = tensors[j];
            എങ്കിൽ (t->data == ശൂന്യ) {
                എങ്കിൽ (t->view_src == ശൂന്യ) {
                    എങ്കിൽ (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        നിർത്തുക;
                    }
                } അല്ലെങ്കിൽ എങ്കിൽ (t->buffer == ശൂന്യ) {
                    എങ്കിൽ (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        നിർത്തുക;
                    }
                }
            } അല്ലെങ്കിൽ {
                എങ്കിൽ (t->view_src != ശൂന്യ && t->buffer == ശൂന്യ) {
                    എങ്കിൽ (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        നിർത്തുക;
                    }
                }
            }
        }
        എങ്കിൽ (t_failed != ശൂന്യ) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            വേണ്ടി (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            തിരികെ ശൂന്യ;
        }

        buffers.push_back(buffer);
    }

    എങ്കിൽ (buffers.empty()) {
        തിരികെ ശൂന്യ;
    }

    എങ്കിൽ (buffers.size() == 1) {
        തിരികെ buffers[0];
    }

    തിരികെ ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

സ്ഥിതം size_t ggml_backend_hexagon_buffer_type_get_alloc_size_n(
        ggml_backend_buffer_type_t buft, ഘടന ggml_tensor ** tensors, പൂർണ്ണസംഖ്യ n_tensors) {
    സ്ഥിരം ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    വേണ്ടി (സ്ഥിരം auto & item : plan) {
        total += item.size;
    }
    തിരികെ total;
}

സ്ഥിതം ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_buffer_type_is_host,
};

സ്ഥിതം ggml_backend_buffer_type_i ggml_backend_hexagon_host_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_host_buffer_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(പൂർണ്ണസംഖ്യ dev_id, സ്ഥിരം ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    buffer_type.device  = dev;
    buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    buffer_type.context = പുതിയ ggml_backend_hexagon_buffer_type_context(config.name, ഇത്);

    host_buffer_type.device  = dev;
    host_buffer_type.iface   = ggml_backend_hexagon_host_buffer_type_interface;
    host_buffer_type.context = പുതിയ ggml_backend_hexagon_buffer_type_context(config.name + "-HOST", ഇത്);

    fence_buffer_type.device  = dev;
    fence_buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    fence_buffer_type.context = പുതിയ ggml_backend_hexagon_buffer_type_context(config.name + "-FENCE", ഇത്);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(host_buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(fence_buffer_type.context);
}

സ്ഥിതം തർക്കം ggml_backend_buffer_is_hexagon(സ്ഥിരം ഘടന ggml_backend_buffer * b) {
    തിരികെ b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

ഘടന ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<പൂർണ്ണസംഖ്യ, പൂർണ്ണസംഖ്യ>                b_map; // buffer fd   to index
    std::unordered_map<സ്ഥിരം ggml_tensor*, പൂർണ്ണസംഖ്യ> t_map; // tensor ptr  to index
    std::unordered_multimap<ശൂന്യം*, പൂർണ്ണസംഖ്യ>         d_map; // tensor data to index

    unsigned പൂർണ്ണസംഖ്യ n_bufs;     // num buffers in the batch
    unsigned പൂർണ്ണസംഖ്യ n_tens;     // num tensors ...
    unsigned പൂർണ്ണസംഖ്യ n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    unsigned പൂർണ്ണസംഖ്യ n_bufs_max;
    unsigned പൂർണ്ണസംഖ്യ n_tens_max;
    unsigned പൂർണ്ണസംഖ്യ n_ops_max;
    size_t       b_vmem_max;

    ശൂന്യം reset() {
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
        ഇത്->sess = sess;

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

    തർക്കം empty() സ്ഥിരം { തിരികെ n_ops == 0; }

    // add buffer and return its index
    പൂർണ്ണസംഖ്യ add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        auto it = b_map.find(sbuf->fd());
        എങ്കിൽ (it != b_map.end()) { തിരികെ it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        പൂർണ്ണസംഖ്യ bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        എങ്കിൽ (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (ശൂന്യം*) sbuf->base(), (size_t) b.size, b_vmem);

        തിരികെ bi;
    }

    തർക്കം same_shape(സ്ഥിരം htp_tensor * h, സ്ഥിരം ggml_tensor * t) സ്ഥിരം {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;

        സ്ഥിരം തർക്കം is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        സ്ഥിരം int64_t ne0_padded = hex_round_up(t->ne[0], 32);
        സ്ഥിരം int64_t ne1_padded = hex_round_up(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_row_size(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? വലുപ്പം(ggml_fp16_t) : t->nb[0];
        എണ്ണൽ ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        തിരികെ (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    പൂർണ്ണസംഖ്യ add_tensor(സ്ഥിരം ggml_tensor * t) {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;
        auto sbuf  = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        വേണ്ടി (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            എങ്കിൽ (same_shape(h, t)) { തിരികെ it->second; }
        }

        // Lookup by tensor ptr
        auto it = t_map.find(t);
        എങ്കിൽ (it != t_map.end()) { തിരികെ it->second; }

        // Add new tensor to the batch
        പൂർണ്ണസംഖ്യ ti = n_tens++;
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

        സ്ഥിരം തർക്കം is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        എങ്കിൽ (is_tiled) {
            എങ്കിൽ (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = വലുപ്പം(ggml_fp16_t);
            } അല്ലെങ്കിൽ {
                h.nb[0] = t->nb[0];
            }
            സ്ഥിരം int64_t ne0_padded = hex_round_up(t->ne[0], 32);
            സ്ഥിരം int64_t ne1_padded = hex_round_up(t->ne[1], 32);
            h.nb[1] = htp_tiled_row_size(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_size  = h.size;
        } അല്ലെങ്കിൽ {
            h.size  = t_size;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        എങ്കിൽ ((extra->flags & GGML_HEXAGON_TENSOR_WEIGHT) != 0) {
            h.flags |= HTP_TENSOR_WEIGHT;
        }
        എങ്കിൽ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        എങ്കിൽ ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (ശൂന്യം*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        തിരികെ ti;
    }

    തർക്കം fit_op(സ്ഥിരം htp_opnode & കണ്ണി) സ്ഥിരം {
        എങ്കിൽ (n_ops >= n_ops_max) തിരികെ അസത്യം;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        പൂർണ്ണസംഖ്യ seen_bufs[HTP_OP_MAX_BUFS];
        പൂർണ്ണസംഖ്യ n_seen_bufs = 0;

        auto fit_tensor = [&](സ്ഥിരം ggml_tensor *t) {
            എങ്കിൽ (!t) തിരികെ;
            എങ്കിൽ (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                പൂർണ്ണസംഖ്യ fd = sbuf->fd();
                എങ്കിൽ (!b_map.count(fd)) {
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_seen_bufs; i++) {
                        എങ്കിൽ (seen_bufs[i] == fd) തിരികെ;
                    }
                    എങ്കിൽ (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    എങ്കിൽ (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        വേണ്ടി (സ്ഥിരം auto * src : കണ്ണി.get_inputs()) {
            fit_tensor(src);
        }
        വേണ്ടി (സ്ഥിരം auto * output : കണ്ണി.get_outputs()) {
            fit_tensor(output);
        }

        എങ്കിൽ ((extra_bufs + n_bufs) > n_bufs_max) തിരികെ അസത്യം;
        എങ്കിൽ ((extra_tens + n_tens) > n_tens_max) തിരികെ അസത്യം;
        എങ്കിൽ ((extra_vmem + b_vmem) > b_vmem_max) തിരികെ അസത്യം;

        തിരികെ സത്യം;
    }

    // assumes that fit_op() was called first and returned true
    ശൂന്യം add_op(സ്ഥിരം htp_opnode & കണ്ണി) {
        // Add new op

        unsigned പൂർണ്ണസംഖ്യ n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = കണ്ണി;

        htp_op_desc &o = h_ops[n];
        memcpy(o.params,        കണ്ണി.കണ്ണി->op_params, വലുപ്പം(കണ്ണി.കണ്ണി->op_params));
        memcpy(o.kernel_params, കണ്ണി.kernel_params,   വലുപ്പം(o.kernel_params));
        o.opcode = കണ്ണി.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        auto inputs = കണ്ണി.get_inputs();
        വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        auto outputs = കണ്ണി.get_outputs();
        വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    ശൂന്യം sort_buffers() {
        എങ്കിൽ (n_bufs <= 1) തിരികെ;

        std::vector<പൂർണ്ണസംഖ്യ> order(n_bufs);
        വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i = 0; i < n_bufs; i++) { order[i] = (പൂർണ്ണസംഖ്യ) i; }

        std::stable_sort(order.begin(), order.end(), [&](പൂർണ്ണസംഖ്യ a, പൂർണ്ണസംഖ്യ b) {
            തിരികെ h_bufs[a].size > h_bufs[b].size;
        });

        തർക്കം already_sorted = സത്യം;
        വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i = 0; i < n_bufs; i++) {
            എങ്കിൽ (order[i] != (പൂർണ്ണസംഖ്യ) i) {
                already_sorted = അസത്യം;
                നിർത്തുക;
            }
        }
        എങ്കിൽ (already_sorted) തിരികെ;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ new_bi = 0; new_bi < n_bufs; new_bi++) {
            പൂർണ്ണസംഖ്യ old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    ശൂന്യം update_mdev_group(uint32_t mdev_idx) {
        എങ്കിൽ (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    തർക്കം try_fuse_common(std::initializer_list<സ്ഥിരം ggml_tensor *> tensors) സ്ഥിരം {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        പൂർണ്ണസംഖ്യ seen_bufs[HTP_OP_MAX_BUFS];
        പൂർണ്ണസംഖ്യ n_seen_bufs = 0;

        വേണ്ടി (സ്ഥിരം auto * t : tensors) {
            എങ്കിൽ (!t || t_map.count(t)) {
                തുടരുക;
            }
            extra_tens++;
            auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            പൂർണ്ണസംഖ്യ fd = sbuf->fd();
            എങ്കിൽ (!b_map.count(fd)) {
                തർക്കം found = അസത്യം;
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_seen_bufs; i++) {
                    എങ്കിൽ (seen_bufs[i] == fd) {
                        found = സത്യം;
                        നിർത്തുക;
                    }
                }
                എങ്കിൽ (!found) {
                    എങ്കിൽ (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    എങ്കിൽ (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        എങ്കിൽ ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            തിരികെ അസത്യം;
        }

        തിരികെ സത്യം;
    }

    തർക്കം try_fuse_common(സ്ഥിരം ggml_tensor * t1, സ്ഥിരം ggml_tensor * t2) സ്ഥിരം {
        തിരികെ try_fuse_common({t1, t2});
    }

    തർക്കം try_fuse_allreduce_add(സ്ഥിരം htp_opnode & കണ്ണി) {
        എങ്കിൽ (n_ops == 0 || opt_ar_select != 2) തിരികെ അസത്യം;
        എങ്കിൽ (കണ്ണി.opcode != HTP_OP_ADD) തിരികെ അസത്യം;

        htp_opnode & last_കണ്ണി = ops[n_ops - 1];
        എങ്കിൽ (last_കണ്ണി.opcode != HTP_OP_ALLREDUCE) തിരികെ അസത്യം;

        auto * ar_kparams = (ഘടന htp_allreduce_kernel_params *) last_കണ്ണി.kernel_params;
        സ്ഥിരം uint32_t rank    = (uint32_t) ar_kparams->rank;
        സ്ഥിരം uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        സ്ഥിരം ggml_tensor * ar_local = last_കണ്ണി.inputs[rank];
        സ്ഥിരം ggml_tensor * add_src0 = കണ്ണി.src0();
        സ്ഥിരം ggml_tensor * add_src1 = കണ്ണി.src1();
        സ്ഥിരം ggml_tensor * add_dst  = കണ്ണി.dst();

        എങ്കിൽ (!ggml_hexagon_tensor_is_fuseable(ar_local)) തിരികെ അസത്യം;

        സ്ഥിരം ggml_tensor * res_tensor;
        എങ്കിൽ (add_src0 == ar_local || add_src0->data == ar_local->data) {
            res_tensor = add_src1;
        } അല്ലെങ്കിൽ എങ്കിൽ (add_src1 == ar_local || add_src1->data == ar_local->data) {
            res_tensor = add_src0;
        } അല്ലെങ്കിൽ {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (ar_local->type != res_tensor->type) തിരികെ അസത്യം;

        സ്ഥിരം തർക്കം is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        സ്ഥിരം തർക്കം is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        എങ്കിൽ (!is_same_shape && !is_row_bcast) തിരികെ അസത്യം;

        എങ്കിൽ (is_same_shape) {
            എങ്കിൽ (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                തിരികെ അസത്യം;
            }
        }
        എങ്കിൽ (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) {
            തിരികെ അസത്യം;
        }

        // scatter is only valid for in-place add within max outputs
        സ്ഥിരം തർക്കം is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

        ഘടന htp_allreduce_kernel_params new_kparams;
        എങ്കിൽ (!ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, സത്യം, is_row_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            തിരികെ അസത്യം;
        }

        സ്ഥിരം തർക്കം scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

        വേണ്ടി (uint32_t r = 0; r < n_ranks; r++) {
            സ്ഥിരം ggml_tensor * ar_src = last_കണ്ണി.inputs[r];
            എങ്കിൽ (!ggml_hexagon_tensors_overlap(add_dst, ar_src)) {
                തുടരുക;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            എങ്കിൽ (scatter_ok && r == rank) {
                തുടരുക;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (!try_fuse_common(res_tensor, add_dst)) {
            തിരികെ അസത്യം;
        }

        last_കണ്ണി.opcode = HTP_OP_ALLREDUCE_ADD;
        last_കണ്ണി.name   = "ALLREDUCE+ADD";
        last_കണ്ണി.inputs.push_back(res_tensor);
        last_കണ്ണി.outputs.clear();
        last_കണ്ണി.outputs.push_back(add_dst);
        last_കണ്ണി.fused.push_back(കണ്ണി.കണ്ണി);
        memcpy(last_കണ്ണി.kernel_params, &new_kparams, വലുപ്പം(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ADD;
        memcpy(o.kernel_params, &new_kparams, വലുപ്പം(new_kparams));

        o.src[2 * n_ranks] = add_tensor(res_tensor);
        എങ്കിൽ (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            വേണ്ടി (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            വേണ്ടി (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } അല്ലെങ്കിൽ {
            o.dst[0] = add_tensor(add_dst);
            വേണ്ടി (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (പൂർണ്ണസംഖ്യ) new_kparams.mode, (പൂർണ്ണസംഖ്യ) new_kparams.n_dsts);
        തിരികെ സത്യം;
    }

    തർക്കം try_fuse_rms_norm_mul(സ്ഥിരം htp_opnode & കണ്ണി) {
        എങ്കിൽ (n_ops == 0) തിരികെ അസത്യം;
        എങ്കിൽ (കണ്ണി.opcode != HTP_OP_MUL) തിരികെ അസത്യം;

        htp_opnode & last_കണ്ണി = ops[n_ops - 1];
        എങ്കിൽ (last_കണ്ണി.opcode != HTP_OP_RMS_NORM) തിരികെ അസത്യം;

        സ്ഥിരം ggml_tensor * mul_src0 = കണ്ണി.src0();
        സ്ഥിരം ggml_tensor * mul_src1 = കണ്ണി.src1();
        സ്ഥിരം ggml_tensor * rms_out  = last_കണ്ണി.dst();

        എങ്കിൽ (!ggml_hexagon_tensor_is_fuseable(rms_out)) തിരികെ അസത്യം;

        സ്ഥിരം ggml_tensor * weight;
        എങ്കിൽ (mul_src0 == rms_out || mul_src0->data == rms_out->data) {
            weight = mul_src1;
        } അല്ലെങ്കിൽ എങ്കിൽ (mul_src1 == rms_out || mul_src1->data == rms_out->data) {
            weight = mul_src0;
        } അല്ലെങ്കിൽ {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം ggml_tensor * src0 = last_കണ്ണി.src0();

        എങ്കിൽ (src0->ne[0] != weight->ne[0] || src0->ne[0] != കണ്ണി.dst()->ne[0]) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം തർക്കം is_row_bcast = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
        സ്ഥിരം തർക്കം is_same_shape = (src0->ne[0] == weight->ne[0] && src0->ne[1] == weight->ne[1] &&
                                    src0->ne[2] == weight->ne[2] && src0->ne[3] == weight->ne[3]);
        എങ്കിൽ (!is_row_bcast && !is_same_shape) തിരികെ അസത്യം;

        എങ്കിൽ (!ggml_are_same_shape(src0, കണ്ണി.dst())) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (ggml_is_contiguous(src0) != ggml_is_contiguous(കണ്ണി.dst())) {
            തിരികെ അസത്യം;
        }

        ഘടന htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, weight, കണ്ണി.dst(), &new_kparams
        );

        എങ്കിൽ ((size_t) new_kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_size, sess->vtcm_size);
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (!try_fuse_common(weight, കണ്ണി.dst())) {
            തിരികെ അസത്യം;
        }

        last_കണ്ണി.opcode = HTP_OP_RMS_NORM_MUL;
        last_കണ്ണി.name   = "RMS_NORM+MUL";
        last_കണ്ണി.inputs.clear();
        last_കണ്ണി.inputs.push_back(src0);
        last_കണ്ണി.inputs.push_back(weight);
        last_കണ്ണി.outputs.clear();
        last_കണ്ണി.outputs.push_back(കണ്ണി.dst());
        last_കണ്ണി.fused.push_back(കണ്ണി.കണ്ണി);
        memcpy(last_കണ്ണി.kernel_params, &new_kparams, വലുപ്പം(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        memcpy(o.kernel_params, &new_kparams, വലുപ്പം(new_kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(weight);
        വേണ്ടി (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(കണ്ണി.dst());
        വേണ്ടി (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        തിരികെ സത്യം;
    }

    തർക്കം try_fuse_mul_mat_add(സ്ഥിരം htp_opnode & കണ്ണി) {
        എങ്കിൽ (n_ops == 0) തിരികെ അസത്യം;
        എങ്കിൽ (കണ്ണി.opcode != HTP_OP_ADD) തിരികെ അസത്യം;

        htp_opnode & last_കണ്ണി = ops[n_ops - 1];
        എങ്കിൽ (last_കണ്ണി.opcode != HTP_OP_MUL_MAT) തിരികെ അസത്യം;

        സ്ഥിരം ggml_tensor * add_src0 = കണ്ണി.src0();
        സ്ഥിരം ggml_tensor * add_src1 = കണ്ണി.src1();
        സ്ഥിരം ggml_tensor * mm_out   = last_കണ്ണി.dst();

        എങ്കിൽ (!ggml_hexagon_tensor_is_fuseable(mm_out)) തിരികെ അസത്യം;

        സ്ഥിരം ggml_tensor * src2;
        എങ്കിൽ (add_src0 == mm_out || add_src0->data == mm_out->data) {
            src2 = add_src1;
        } അല്ലെങ്കിൽ എങ്കിൽ (add_src1 == mm_out || add_src1->data == mm_out->data) {
            src2 = add_src0;
        } അല്ലെങ്കിൽ {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം ggml_tensor * src0 = last_കണ്ണി.src0();
        സ്ഥിരം ggml_tensor * src1 = last_കണ്ണി.src1();

        എങ്കിൽ (src2->type != GGML_TYPE_F32) തിരികെ അസത്യം;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        എങ്കിൽ (!ggml_are_same_shape(mm_out, കണ്ണി.dst())) തിരികെ അസത്യം;

        സ്ഥിരം ഘടന htp_mm_kernel_params * orig_kparams = (സ്ഥിരം ഘടന htp_mm_kernel_params *) last_കണ്ണി.kernel_params;
        ഘടന htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_add_params(sess, src0, src1, src2, കണ്ണി.dst(), &kparams);
        എങ്കിൽ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (orig_kparams->collapse != kparams.collapse) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        സ്ഥിരം തർക്കം can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        എങ്കിൽ (!can_fuse) തിരികെ അസത്യം;

        എങ്കിൽ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            എങ്കിൽ (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_threads < orig_kparams->n_act_threads) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_threads, kparams.n_act_threads);
                തിരികെ അസത്യം;
            }
        }

        എങ്കിൽ (!try_fuse_common(src2, കണ്ണി.dst())) {
            തിരികെ അസത്യം;
        }

        last_കണ്ണി.opcode = HTP_OP_MUL_MAT_ADD;
        last_കണ്ണി.name   = "MUL_MAT+ADD";
        last_കണ്ണി.inputs.clear();
        last_കണ്ണി.inputs.push_back(src0);
        last_കണ്ണി.inputs.push_back(src1);
        last_കണ്ണി.inputs.push_back(src2);
        last_കണ്ണി.outputs.clear();
        last_കണ്ണി.outputs.push_back(കണ്ണി.dst());
        last_കണ്ണി.fused.push_back(കണ്ണി.കണ്ണി);
        memcpy(last_കണ്ണി.kernel_params, &kparams, വലുപ്പം(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ADD;
        memcpy(o.kernel_params, &kparams, വലുപ്പം(kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(src1);
        o.src[2] = add_tensor(src2);
        വേണ്ടി (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(കണ്ണി.dst());
        വേണ്ടി (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        തിരികെ സത്യം;
    }

    തർക്കം try_fuse_mul_mat_nx(സ്ഥിരം htp_opnode & കണ്ണി) {
        എങ്കിൽ (n_ops == 0 || കണ്ണി.opcode != HTP_OP_MUL_MAT) തിരികെ അസത്യം;
        എങ്കിൽ (!is_mergeable_mul_mat(കണ്ണി.കണ്ണി)) തിരികെ അസത്യം;

        സ്ഥിരം ggml_tensor * w_in = കണ്ണി.src0();
        സ്ഥിരം ggml_tensor * x_in = കണ്ണി.src1();
        സ്ഥിരം ggml_tensor * d_in = കണ്ണി.dst();

        htp_opnode & last_കണ്ണി = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        എങ്കിൽ (last_കണ്ണി.opcode == HTP_OP_MUL_MAT_NX) {
            സ്ഥിരം uint32_t curr_n = (uint32_t) last_കണ്ണി.outputs.size();
            എങ്കിൽ (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                തിരികെ അസത്യം;
            }

            സ്ഥിരം ggml_tensor * w0 = last_കണ്ണി.inputs[0];
            സ്ഥിരം ggml_tensor * x  = last_കണ്ണി.inputs[curr_n];

            എങ്കിൽ (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (!last_കണ്ണി.fused.empty() && (mm_is_hmx_eligible(last_കണ്ണി.fused[0]) != mm_is_hmx_eligible(കണ്ണി.കണ്ണി))) {
                തിരികെ അസത്യം;
            }

            സ്ഥിരം ഘടന htp_mm_kernel_params * orig_kparams = (സ്ഥിരം ഘടന htp_mm_kernel_params *) last_കണ്ണി.kernel_params;
            സ്ഥിരം തർക്കം collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_in, x, d_in);
            എങ്കിൽ (orig_kparams->collapse && !collapse) {
                തിരികെ അസത്യം;
            }

            ഘടന htp_mm_kernel_params kparams;
            എങ്കിൽ (collapse) {
                സ്ഥിരം ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } അല്ലെങ്കിൽ {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            എങ്കിൽ (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                തിരികെ അസത്യം;
            }

            എങ്കിൽ (!try_fuse_common(w_in, d_in)) {
                തിരികെ അസത്യം;
            }

            last_കണ്ണി.inputs[curr_n] = w_in;
            last_കണ്ണി.inputs.push_back(x);
            last_കണ്ണി.outputs.push_back(d_in);
            last_കണ്ണി.fused.push_back(കണ്ണി.കണ്ണി);
            memcpy(last_കണ്ണി.kernel_params, &kparams, വലുപ്പം(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, വലുപ്പം(kparams));

            വേണ്ടി (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = add_tensor(last_കണ്ണി.inputs[s]);
            }
            വേണ്ടി (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            വേണ്ടി (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_കണ്ണി.outputs[d]);
            }
            വേണ്ടി (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            തിരികെ സത്യം;
        }

        // Case 2: last_node is single MUL_MAT
        എങ്കിൽ (last_കണ്ണി.opcode == HTP_OP_MUL_MAT) {
            എങ്കിൽ (!is_mergeable_mul_mat_pair(last_കണ്ണി.കണ്ണി, കണ്ണി.കണ്ണി)) {
                തിരികെ അസത്യം;
            }

            സ്ഥിരം ggml_tensor * w0 = last_കണ്ണി.src0();
            സ്ഥിരം ggml_tensor * x  = last_കണ്ണി.src1();
            സ്ഥിരം ggml_tensor * w1 = കണ്ണി.src0();
            സ്ഥിരം ggml_tensor * dst_0 = last_കണ്ണി.dst();
            സ്ഥിരം ggml_tensor * dst_1 = കണ്ണി.dst();

            സ്ഥിരം ഘടന htp_mm_kernel_params * orig_kparams = (സ്ഥിരം ഘടന htp_mm_kernel_params *) last_കണ്ണി.kernel_params;
            സ്ഥിരം തർക്കം collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            എങ്കിൽ (orig_kparams->collapse && !collapse) {
                തിരികെ അസത്യം;
            }

            ഘടന htp_mm_kernel_params kparams;
            എങ്കിൽ (collapse) {
                സ്ഥിരം ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } അല്ലെങ്കിൽ {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            എങ്കിൽ (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                തിരികെ അസത്യം;
            }

            എങ്കിൽ (!try_fuse_common(w1, കണ്ണി.dst())) {
                തിരികെ അസത്യം;
            }

            last_കണ്ണി.opcode = HTP_OP_MUL_MAT_NX;
            last_കണ്ണി.name   = "MUL_MAT_NX";
            last_കണ്ണി.inputs.clear();
            last_കണ്ണി.inputs.push_back(w0);
            last_കണ്ണി.inputs.push_back(w1);
            last_കണ്ണി.inputs.push_back(x);
            last_കണ്ണി.outputs.clear();
            last_കണ്ണി.outputs.push_back(dst_0);
            last_കണ്ണി.outputs.push_back(dst_1);
            last_കണ്ണി.fused.push_back(കണ്ണി.കണ്ണി);
            memcpy(last_കണ്ണി.kernel_params, &kparams, വലുപ്പം(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            memcpy(o.kernel_params, &kparams, വലുപ്പം(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            വേണ്ടി (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            വേണ്ടി (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            തിരികെ സത്യം;
        }

        തിരികെ അസത്യം;
    }

    തർക്കം try_fuse_mul_mat_id_nx(സ്ഥിരം htp_opnode & കണ്ണി) {
        എങ്കിൽ (n_ops == 0 || കണ്ണി.opcode != HTP_OP_MUL_MAT_ID) തിരികെ അസത്യം;
        എങ്കിൽ (!is_mergeable_mul_mat_id(കണ്ണി.കണ്ണി)) തിരികെ അസത്യം;

        സ്ഥിരം ggml_tensor * w_in   = കണ്ണി.src0();
        സ്ഥിരം ggml_tensor * x_in   = കണ്ണി.src1();
        സ്ഥിരം ggml_tensor * ids_in = കണ്ണി.കണ്ണി->src[2];
        സ്ഥിരം ggml_tensor * d_in   = കണ്ണി.dst();

        htp_opnode & last_കണ്ണി = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        എങ്കിൽ (last_കണ്ണി.opcode == HTP_OP_MUL_MAT_ID_NX) {
            സ്ഥിരം uint32_t curr_n = (uint32_t) last_കണ്ണി.outputs.size();
            എങ്കിൽ (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                തിരികെ അസത്യം;
            }

            സ്ഥിരം ggml_tensor * w0  = last_കണ്ണി.inputs[0];
            സ്ഥിരം ggml_tensor * x   = last_കണ്ണി.inputs[curr_n];
            സ്ഥിരം ggml_tensor * ids = last_കണ്ണി.inputs[curr_n + 1];

            എങ്കിൽ (x_in != x || ids_in != ids || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (!last_കണ്ണി.fused.empty() && (mm_is_hmx_eligible(last_കണ്ണി.fused[0]) != mm_is_hmx_eligible(കണ്ണി.കണ്ണി))) {
                തിരികെ അസത്യം;
            }

            ഘടന htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
            എങ്കിൽ (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                തിരികെ അസത്യം;
            }

            എങ്കിൽ (!try_fuse_common(w_in, d_in)) {
                തിരികെ അസത്യം;
            }

            last_കണ്ണി.inputs[curr_n] = w_in;
            last_കണ്ണി.inputs[curr_n + 1] = x;
            last_കണ്ണി.inputs.push_back(ids);
            last_കണ്ണി.outputs.push_back(d_in);
            last_കണ്ണി.fused.push_back(കണ്ണി.കണ്ണി);
            memcpy(last_കണ്ണി.kernel_params, &kparams, വലുപ്പം(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, വലുപ്പം(kparams));

            വേണ്ടി (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = add_tensor(last_കണ്ണി.inputs[s]);
            }
            വേണ്ടി (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            വേണ്ടി (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_കണ്ണി.outputs[d]);
            }
            വേണ്ടി (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            തിരികെ സത്യം;
        }

        // Case 2: last_node is single MUL_MAT_ID
        എങ്കിൽ (last_കണ്ണി.opcode == HTP_OP_MUL_MAT_ID) {
            എങ്കിൽ (!is_mergeable_mul_mat_id_pair(last_കണ്ണി.കണ്ണി, കണ്ണി.കണ്ണി)) {
                തിരികെ അസത്യം;
            }

            സ്ഥിരം ggml_tensor * w0  = last_കണ്ണി.src0();
            സ്ഥിരം ggml_tensor * x   = last_കണ്ണി.src1();
            സ്ഥിരം ggml_tensor * ids = last_കണ്ണി.കണ്ണി->src[2];
            സ്ഥിരം ggml_tensor * w1  = കണ്ണി.src0();

            ഘടന htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, കണ്ണി.dst(), 2, &kparams);
            എങ്കിൽ (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                തിരികെ അസത്യം;
            }

            എങ്കിൽ (!try_fuse_common(w1, കണ്ണി.dst())) {
                തിരികെ അസത്യം;
            }

            സ്ഥിരം ggml_tensor * dst_0 = last_കണ്ണി.dst();
            സ്ഥിരം ggml_tensor * dst_1 = കണ്ണി.dst();

            last_കണ്ണി.opcode = HTP_OP_MUL_MAT_ID_NX;
            last_കണ്ണി.name   = "MUL_MAT_ID_NX";
            last_കണ്ണി.inputs.clear();
            last_കണ്ണി.inputs.push_back(w0);
            last_കണ്ണി.inputs.push_back(w1);
            last_കണ്ണി.inputs.push_back(x);
            last_കണ്ണി.inputs.push_back(ids);
            last_കണ്ണി.outputs.clear();
            last_കണ്ണി.outputs.push_back(dst_0);
            last_കണ്ണി.outputs.push_back(dst_1);
            last_കണ്ണി.fused.push_back(കണ്ണി.കണ്ണി);
            memcpy(last_കണ്ണി.kernel_params, &kparams, വലുപ്പം(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            memcpy(o.kernel_params, &kparams, വലുപ്പം(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            o.src[3] = add_tensor(ids);
            വേണ്ടി (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            വേണ്ടി (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            തിരികെ സത്യം;
        }

        തിരികെ അസത്യം;
    }

    തർക്കം try_fuse_gdn_cpy(സ്ഥിരം htp_opnode & കണ്ണി) {
        എങ്കിൽ (n_ops == 0 || കണ്ണി.opcode != HTP_OP_CPY) തിരികെ അസത്യം;

        htp_opnode & last_കണ്ണി = ops[n_ops - 1];
        എങ്കിൽ (last_കണ്ണി.opcode != HTP_OP_GATED_DELTA_NET) തിരികെ അസത്യം;
        എങ്കിൽ (last_കണ്ണി.outputs.size() != 1) തിരികെ അസത്യം;

        സ്ഥിരം ggml_tensor * gdn_out  = last_കണ്ണി.dst();
        സ്ഥിരം ggml_tensor * cpy_കണ്ണി = കണ്ണി.കണ്ണി;
        സ്ഥിരം ggml_tensor * cpy_src  = കണ്ണി.src0();
        സ്ഥിരം ggml_tensor * cpy_dst  = കണ്ണി.dst();

        എങ്കിൽ (!cpy_src || !cpy_dst || !cpy_dst->data) തിരികെ അസത്യം;
        എങ്കിൽ (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) തിരികെ അസത്യം;
        എങ്കിൽ ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_കണ്ണി->flags & GGML_TENSOR_FLAG_OUTPUT)) തിരികെ അസത്യം;

        സ്ഥിരം ggml_tensor * v = last_കണ്ണി.കണ്ണി->src[2];
        എങ്കിൽ (!v) തിരികെ അസത്യം;

        സ്ഥിരം int64_t S_v      = v->ne[0];
        സ്ഥിരം int64_t H        = v->ne[1];
        സ്ഥിരം int64_t n_tokens = v->ne[2];
        സ്ഥിരം int64_t n_seqs   = v->ne[3];
        സ്ഥിരം int64_t K        = ggml_get_op_params_i32(last_കണ്ണി.കണ്ണി, 0);
        സ്ഥിരം size_t  tail_off = (size_t) S_v * H * n_tokens * n_seqs * വലുപ്പം(ദശാംശം);

        സ്ഥിരം int64_t D         = S_v * S_v * H;
        സ്ഥിരം int64_t n_written = std::min<int64_t>(n_tokens, K);

        എങ്കിൽ (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_out && cpy_src->view_src->data != gdn_out->data) ||
            cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != വലുപ്പം(ദശാംശം)) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * വലുപ്പം(ദശാംശം)) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (n_written > 1) {
            എങ്കിൽ (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * വലുപ്പം(ദശാംശം)) {
                തിരികെ അസത്യം;
            }
        }

        എങ്കിൽ (!try_fuse_common({cpy_dst})) {
            തിരികെ അസത്യം;
        }

        last_കണ്ണി.name += "+CPY";
        last_കണ്ണി.outputs.push_back(cpy_dst);
        last_കണ്ണി.fused.push_back(കണ്ണി.കണ്ണി);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = add_tensor(cpy_dst);
        വേണ്ടി (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        തിരികെ സത്യം;
    }

    തർക്കം try_fuse(സ്ഥിരം htp_opnode & കണ്ണി) {
        എങ്കിൽ (!opt_opfusion) തിരികെ അസത്യം;
        എങ്കിൽ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ADD) && try_fuse_allreduce_add(കണ്ണി)) തിരികെ സത്യം;
        എങ്കിൽ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(കണ്ണി))  തിരികെ സത്യം;
        എങ്കിൽ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ADD)   && try_fuse_mul_mat_add(കണ്ണി))   തിരികെ സത്യം;
        എങ്കിൽ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(കണ്ണി))    തിരികെ സത്യം;
        എങ്കിൽ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(കണ്ണി)) തിരികെ സത്യം;
        എങ്കിൽ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(കണ്ണി))       തിരികെ സത്യം;
        തിരികെ അസത്യം;
    }
};

ഘടന ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

ഘടന ggml_hexagon_opqueue {
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
        എങ്കിൽ (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * വലുപ്പം(htp_trace_desc);
        }

        shm_blk_size = വലുപ്പം(htp_buf_desc)  * n_bufs    +
                       വലുപ്പം(htp_tensor)    * n_tensors +
                       വലുപ്പം(htp_op_desc)   * n_ops     +
                       വലുപ്പം(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = പുതിയ ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, സത്യം /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        എങ്കിൽ (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size(), shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    size_t shm_size() സ്ഥിരം { തിരികെ shm_buf ? shm_buf->size() : 0; }

    // push new batch
    തർക്കം തള്ളുക(htp_opbatch_req& req, dspqueue_buffer& dbuf, സ്ഥിരം ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(വലുപ്പം(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(വലുപ്പം(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(വലുപ്പം(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(വലുപ്പം(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(വലുപ്പം(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(വലുപ്പം(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        എങ്കിൽ (seq - shm_buf->sess->batch_rsp_seq > depth) { തിരികെ അസത്യം; }

        സ്ഥിരം uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[slot]   = op_batch->ops;
        start_usec[slot] = ggml_time_us();

        സ്ഥിരം size_t b_size = വലുപ്പം(htp_buf_desc)  * req.n_bufs;
        സ്ഥിരം size_t t_size = വലുപ്പം(htp_tensor)    * req.n_tensors;
        സ്ഥിരം size_t o_size = വലുപ്പം(htp_op_desc)   * req.n_ops;
        സ്ഥിരം size_t p_size = വലുപ്പം(htp_prof_desc) * req.n_ops;

        size_t tr_size = 0;
        എങ്കിൽ (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * വലുപ്പം(htp_trace_desc);
        } അല്ലെങ്കിൽ {
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

        memcpy(b_ptr, (ശൂന്യം *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (ശൂന്യം *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (ശൂന്യം *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        എങ്കിൽ (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (ശൂന്യം *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (unsigned long long) t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        തിരികെ സത്യം;
    }

    ശൂന്യം എടുക്കുക(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        സ്ഥിരം uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        സ്ഥിരം size_t b_size = വലുപ്പം(htp_buf_desc)  * rsp.n_bufs;
        സ്ഥിരം size_t t_size = വലുപ്പം(htp_tensor)    * rsp.n_tensors;
        സ്ഥിരം size_t o_size = വലുപ്പം(htp_op_desc)   * rsp.n_ops;
        സ്ഥിരം size_t p_size = വലുപ്പം(htp_prof_desc) * rsp.n_ops;

        size_t tr_size = 0;
        uint32_t n_traces = 0;
        എങ്കിൽ (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * വലുപ്പം(htp_trace_desc);
        }

        സ്ഥിരം size_t m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        എങ്കിൽ (rsp.n_ops > 0) {
            auto & ops = op_cache[slot];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            സ്ഥിരം htp_prof_desc * pd = (സ്ഥിരം htp_prof_desc *) p_ptr;
            സ്ഥിരം htp_trace_desc * trace_events = nullptr;
            എങ്കിൽ (opt_profile == 3) {
                trace_events = (സ്ഥിരം htp_trace_desc *) (p_ptr + p_size);
            }

            എങ്കിൽ (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            വേണ്ടി (uint32_t i = 0; i < rsp.n_ops; i++) {
                എങ്കിൽ (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            എങ്കിൽ (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

ശൂന്യം ggml_hexagon_session::flush_peers() {
    auto vpeers = std::move(virt_peers);
    virt_peers.clear();
    വേണ്ടി (auto * peer : vpeers) {
        peer->flush_sync();
    }

    auto ppeers = std::move(phys_peers);
    phys_peers.clear();
    വേണ്ടി (auto * peer : ppeers) {
        peer->flush_async();
    }

    വേണ്ടി (auto & sub : ഇത്->mdev.sessions) {
        sub->flush_peers();
    }
}

ശൂന്യം ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

ശൂന്യം ggml_hexagon_session::flush_pending(തർക്കം all) {
    വേണ്ടി (auto & sub : ഇത്->mdev.sessions) {
        sub->flush_pending(all);
        എങ്കിൽ (sub->last_error > HTP_STATUS_OK) {
            ഇത്->last_error = sub->last_error.load();
        }
    }

    ആകുമ്പോൾ (ഇത്->batch_rsp_seq < ഇത്->batch_req_seq) {
        ഘടന htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        ഘടന dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        സ്ഥിരം uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        പൂർണ്ണസംഖ്യ err = dspqueue_read(ഇത്->queue, &flags, 1, &n_dbufs, &dbuf, വലുപ്പം(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        എങ്കിൽ (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            തുടരുക;
        }

        എങ്കിൽ (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        എങ്കിൽ (rsp_size != വലുപ്പം(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", ഇത്->c_name(), rsp_size, n_dbufs);
        }

        എങ്കിൽ (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", ഇത്->c_name(), status_to_str(rsp.status));
            ഇത്->last_error = rsp.status;
            വേണ്ടി (auto & sub : ഇത്->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->എടുക്കുക(rsp, dbuf);

        GGML_ASSERT(rsp.seq == ഇത്->batch_rsp_seq + 1);
        ഇത്->batch_rsp_seq = rsp.seq;

        എങ്കിൽ (!all) നിർത്തുക;
    }
}

ശൂന്യം ggml_hexagon_session::flush_sync(തർക്കം all) {
    flush_async();
    flush_pending(all);
}

ശൂന്യം ggml_hexagon_session::start_batch() {
    എങ്കിൽ (ഇത്->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

ശൂന്യം ggml_hexagon_session::flush_batch(size_t min_ops) {
    എങ്കിൽ (op_batch->n_ops < min_ops) { തിരികെ; }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    സ്ഥിരം uint64_t seq = ++ഇത്->batch_req_seq;

    op_batch->update_mdev_group(ഇത്->mdev.idx);

    എങ്കിൽ (!op_queue->തള്ളുക(req, dbuf, op_batch, seq)) {
        flush_pending(അസത്യം);
        op_queue->തള്ളുക(req, dbuf, op_batch, seq);
    }

    വേണ്ടി (auto & sub : ഇത്->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_buffer sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        എങ്കിൽ (!sub->op_queue->തള്ളുക(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(അസത്യം);
            sub->op_queue->തള്ളുക(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        പൂർണ്ണസംഖ്യ err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, വലുപ്പം(sub_req), (സ്ഥിരം uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        എങ്കിൽ (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (unsigned) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", ഇത്->c_name(), dbuf.ptr, dbuf.size);

    പൂർണ്ണസംഖ്യ err = dspqueue_write(ഇത്->queue, 0, 1, &dbuf, വലുപ്പം(req), (സ്ഥിരം uint8_t*) &req, DSPQUEUE_TIMEOUT);
    എങ്കിൽ (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", ഇത്->c_name(), (unsigned) err);
    }

    op_batch->reset();
}

ശൂന്യം ggml_hexagon_session::enqueue_op(സ്ഥിരം htp_opnode & കണ്ണി) {
    auto clone_tensor_buffer = [ഇത്](സ്ഥിരം ggml_tensor * t) {
        auto sbuf = ഇത്->mmap_tensor(t);
        എങ്കിൽ (!sbuf) തിരികെ;
        എങ്കിൽ (sbuf->sess != ഇത്) {
            ഇത്->clone_buffer(sbuf);
        }
        വേണ്ടി (auto & sub : ഇത്->mdev.sessions) {
            sub->clone_buffer(sbuf);
        }
    };

    വേണ്ടി (auto t : കണ്ണി.get_inputs()) {
        clone_tensor_buffer(t);
    }
    വേണ്ടി (auto t : കണ്ണി.get_outputs()) {
        clone_tensor_buffer(t);
    }

    എങ്കിൽ (opt_opfusion && op_batch->try_fuse(കണ്ണി)) {
        തിരികെ;
    }

    എങ്കിൽ (!op_batch->fit_op(കണ്ണി)) {
        flush_async();
    }

    എങ്കിൽ (op_batch->empty()) {
        start_batch();
    }

    എങ്കിൽ (!op_batch->fit_op(കണ്ണി)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->add_op(കണ്ണി);
}

ശൂന്യം ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_കണ്ണി(HTP_OP_MDEV_GROUP);

    uint8_t * fence_slot = ഇത്->mdev_fence_slot;

    സ്ഥിതം ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.buffer       = &ഇത്->fence_buf->backend_buffer;
    dummy_t.extra        = &fence_extra;
    dummy_t.data         = (ശൂന്യം *) fence_slot;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.ne[1]        = (int64_t) ഇത്->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) ഇത്->mdev.idx;

    ggml_tensor * കണ്ണി = group_കണ്ണി.add_dummy(dummy_t);
    കണ്ണി->src[0] = കണ്ണി;
    group_കണ്ണി.init(കണ്ണി);
    group_കണ്ണി.outputs.clear();
    group_കണ്ണി.name = "MDEV_GROUP";

    വേണ്ടി (auto & sub : ഇത്->mdev.sessions) {
        sub->clone_buffer(ഇത്->fence_buf);
    }

    op_batch->add_op(group_കണ്ണി);
}

ശൂന്യം ggml_hexagon_session::enqueue_cpy(സ്ഥിരം ggml_tensor * src, ggml_tensor * dst, സ്ഥിരം ggml_tensor * sync_tensor, uint32_t fence_seq) {
    സ്ഥിരം തർക്കം with_fence = sync_tensor != nullptr;
    htp_opnode cpy_കണ്ണി(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* കണ്ണി = cpy_കണ്ണി.add_dummy(*dst);
    കണ്ണി->op     = GGML_OP_CPY;
    കണ്ണി->src[0] = const_cast<ggml_tensor *>(src);
    കണ്ണി->src[1] = with_fence ? cpy_കണ്ണി.add_dummy(*sync_tensor) : nullptr;
    എങ്കിൽ (with_fence) {
        കണ്ണി->op_params[0] = (int32_t) fence_seq;
    }

    cpy_കണ്ണി.init(കണ്ണി);
    എങ്കിൽ (with_fence) {
        cpy_കണ്ണി.name = "CPY+FENCE";
    }
    സ്ഥിരം തർക്കം ok = ggml_hexagon_precompute_cpy_params(ഇത്, കണ്ണി, (ഘടന htp_copy_kernel_params *) cpy_കണ്ണി.kernel_params);
    സ്ഥിരം auto * kparams = (സ്ഥിരം ഘടന htp_copy_kernel_params *) cpy_കണ്ണി.kernel_params;
    എങ്കിൽ (ok && !with_fence && kparams->total_elems == 0) {
        തിരികെ;
    }
    ഇത്->enqueue_op(cpy_കണ്ണി);
}

ശൂന്യം ggml_hexagon_session::enqueue_fence(സ്ഥിരം ggml_tensor * sync_tensor, uint32_t fence_seq, തർക്കം wait) {
    htp_opnode sync_കണ്ണി(HTP_OP_FENCE);

    ggml_tensor* കണ്ണി = sync_കണ്ണി.add_dummy(*sync_tensor);
    കണ്ണി->op           = GGML_OP_NONE;
    കണ്ണി->src[0]       = കണ്ണി;
    കണ്ണി->op_params[0] = (int32_t) fence_seq;
    കണ്ണി->op_params[1] = wait ? 0 : 1;

    sync_കണ്ണി.init(കണ്ണി);
    sync_കണ്ണി.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    ഇത്->enqueue_op(sync_കണ്ണി);
}

സ്ഥിതം തർക്കം ggml_hexagon_precompute_allreduce_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    തർക്കം has_add,
    തർക്കം is_row_bcast,
    തർക്കം is_shard_ok,
    ഘടന htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    സ്ഥിരം uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    സ്ഥിരം uint32_t elem_size = (dst->type == GGML_TYPE_F16) ? വലുപ്പം(ggml_fp16_t) : വലുപ്പം(ദശാംശം);
    സ്ഥിരം തർക്കം is_contiguous = ggml_is_contiguous(dst);

    സ്ഥിരം uint32_t ne0 = (uint32_t) dst->ne[0];
    സ്ഥിരം uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    സ്ഥിരം തർക്കം use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    എങ്കിൽ (has_add) {
        // sharded reduce-scatter for contiguous in-place add
        എങ്കിൽ (opt_ar_scatter && is_shard_ok && use_1d && !is_row_bcast) {
            സ്ഥിരം uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            സ്ഥിരം uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            സ്ഥിരം uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_start = (int32_t) rank_elem_start;
            kparams->rank_nelem      = (int32_t) (rank_elem_end - rank_elem_start);
        } അല്ലെങ്കിൽ എങ്കിൽ (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } അല്ലെങ്കിൽ {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } അല്ലെങ്കിൽ {
        kparams->n_dsts = (int32_t) n_ranks;
        എങ്കിൽ (use_1d) {
            സ്ഥിരം uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            സ്ഥിരം uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            സ്ഥിരം uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            സ്ഥിരം uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } അല്ലെങ്കിൽ {
            സ്ഥിരം uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            സ്ഥിരം uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            സ്ഥിരം uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            സ്ഥിരം uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    എങ്കിൽ (use_1d) {
        സ്ഥിരം uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        സ്ഥിരം uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        സ്ഥിരം size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_എണ്ണം(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        എങ്കിൽ (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        ആകുമ്പോൾ ((size_t) kparams->vtcm_size > sess->vtcm_size && block_elems > 128) {
            സ്ഥിരം size_t max_bytes_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((size_t) (max_bytes_per_buf / elem_size), 128);
            എങ്കിൽ (block_elems < 128) നിർത്തുക;
            kparams->block_elems          = block_elems;
            kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
        }

        എങ്കിൽ (sess->vtcm_size < (size_t) kparams->vtcm_size || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            തിരികെ അസത്യം;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        തിരികെ സത്യം;
    } അല്ലെങ്കിൽ {
        സ്ഥിരം uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        സ്ഥിരം uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        സ്ഥിരം size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_എണ്ണം(n_ranks, n_threads, has_add, is_row_bcast);

        സ്ഥിരം uint32_t row_bytes = ne0 * elem_size;
        സ്ഥിരം uint32_t row_size_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_size_aligned = row_size_aligned;

        സ്ഥിരം uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        ആകുമ്പോൾ ((size_t) kparams->vtcm_size > sess->vtcm_size && block_rows > 1) {
            സ്ഥിരം size_t max_rows_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2 * row_size_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
            എങ്കിൽ (max_rows_per_buf == 0) നിർത്തുക;
        }

        എങ്കിൽ (sess->vtcm_size < (size_t) kparams->vtcm_size || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            തിരികെ അസത്യം;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        തിരികെ സത്യം;
    }
}

ശൂന്യം ggml_hexagon_session::enqueue_allreduce(
    സ്ഥിരം ggml_tensor * dst,
    സ്ഥിരം std::vector<സ്ഥിരം ggml_tensor *> & src_tensors,
    സ്ഥിരം std::vector<സ്ഥിരം ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_കണ്ണി(HTP_OP_ALLREDUCE);

    ggml_tensor* കണ്ണി = ar_കണ്ണി.add_dummy(*dst);
    കണ്ണി->op           = GGML_OP_NONE;
    കണ്ണി->op_params[0] = (int32_t) fence_seq_entry;
    കണ്ണി->op_params[1] = (int32_t) fence_seq_exit;

    ar_കണ്ണി.init(കണ്ണി);

    ar_കണ്ണി.inputs.clear();
    വേണ്ടി (size_t i = 0; i < src_tensors.size(); i++) {
        ar_കണ്ണി.inputs.push_back(src_tensors[i]);
    }
    വേണ്ടി (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_കണ്ണി.inputs.push_back(ar_കണ്ണി.add_dummy(*sync_tensors[i]));
    }

    ar_കണ്ണി.outputs.clear();
    വേണ്ടി (size_t i = 0; i < src_tensors.size(); i++) {
        ar_കണ്ണി.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        ഇത്, dst, rank, n_ranks, അസത്യം, അസത്യം, /*is_shard_ok=*/ അസത്യം,
        (ഘടന htp_allreduce_kernel_params *) ar_കണ്ണി.kernel_params);

    ar_കണ്ണി.name = "ALLREDUCE";
    ഇത്->enqueue_op(ar_കണ്ണി);
}

ggml_hexagon_shared_buffer * ggml_hexagon_session::mmap_tensor(സ്ഥിരം ggml_tensor * t) {
    എങ്കിൽ (!t) തിരികെ nullptr;

    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
    എങ്കിൽ (!sbuf->mapped) {
        സ്ഥിരം തർക്കം is_weight = ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        സ്ഥിരം തർക്കം extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    തിരികെ sbuf;
}

തർക്കം ggml_hexagon_session::clone_buffer(സ്ഥിരം ggml_hexagon_shared_buffer * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    എങ്കിൽ (sbuf->sess == ഇത്) തിരികെ സത്യം;

    auto mem = sbuf->mem;
    പൂർണ്ണസംഖ്യ   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    എങ്കിൽ (ഇത്->cloned_buffers.find(fd) != ഇത്->cloned_buffers.end()) തിരികെ സത്യം;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", ഇത്->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    auto clone = std::make_unique<ggml_hexagon_shared_buffer>(ഇത്, *sbuf);
    ശ്രമിക്കുക {
        clone->mmap(sbuf->extended);
    } പിടിക്കുക (സ്ഥിരം std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", ഇത്->c_name(), exc.what());
        തിരികെ അസത്യം;
    }

    ഇത്->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(ഇത്);
    തിരികെ സത്യം;
}

ശൂന്യം ggml_hexagon_session::release_buffer(സ്ഥിരം ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    പൂർണ്ണസംഖ്യ   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    auto it = ഇത്->cloned_buffers.find(fd);
    എങ്കിൽ (it != ഇത്->cloned_buffers.end()) {
        auto clone = std::move(it->second);
        ഇത്->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(ഇത്);
}

ശൂന്യം ggml_hexagon_session::unclone_buffer(സ്ഥിരം ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    വേണ്ടി (auto * sess : sessions) {
        sess->release_buffer(sbuf);
    }
}

സ്ഥിതം size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    സ്ഥിരം size_t MiB = 1024 * 1024;
    സ്ഥിരം size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t step = 256u * MiB;

    ശ്രമിക്കുക {
        sbufs.push_back(പുതിയ ggml_hexagon_shared_buffer(sess, GiB, സത്യം)); vmem += GiB;
        sbufs.push_back(പുതിയ ggml_hexagon_shared_buffer(sess, GiB, സത്യം)); vmem += GiB;
        sbufs.push_back(പുതിയ ggml_hexagon_shared_buffer(sess, GiB, സത്യം)); vmem += GiB;

        ആകുമ്പോൾ (1) {
            sbufs.push_back(പുതിയ ggml_hexagon_shared_buffer(sess, step, സത്യം));
            vmem += step;
        }
    } പിടിക്കുക (...) { }

    വേണ്ടി (auto b : sbufs) { delete b; }

    തിരികെ vmem - step; // backoff to account for overhead from internal mappings
}

ശൂന്യം ggml_hexagon_session::allocate(സ്ഥിരം ggml_hexagon_device_config & config) noexcept(അസത്യം) {
    പൂർണ്ണസംഖ്യ phys_idx = config.physical_idx;
    പൂർണ്ണസംഖ്യ virt_idx = config.virtual_idx;

    ഇത്->valid_session = അസത്യം;
    ഇത്->valid_handle  = അസത്യം;
    ഇത്->valid_queue   = അസത്യം;
    ഇത്->valid_iface   = അസത്യം;

    ഇത്->name          = config.name;
    ഇത്->phys_idx      = phys_idx;
    ഇത്->virt_idx      = virt_idx;
    ഇത്->domain_id     = config.domain_id;
    ഇത്->session_id    = 0;
    ഇത്->batch_req_seq = 0;
    ഇത്->batch_rsp_seq = 0;
    ഇത്->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", ഇത്->name.c_str(), ഇത്->domain_id, phys_idx, virt_idx);

    എങ്കിൽ (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        എറിയുക std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    സ്ഥിരം std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    എങ്കിൽ (virt_idx > 0) {
        ഘടന remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<അക്ഷരം *>(dom_name.c_str());
        n.session_name     = const_cast<അക്ഷരം *>(ഇത്->name.c_str());
        n.session_name_len = ഇത്->name.size();
        n.session_id       = virt_idx;

        പൂർണ്ണസംഖ്യ err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (ശൂന്യം *) &n, വലുപ്പം(n));
        എങ്കിൽ (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           ഇത്->c_name(), phys_idx, virt_idx, err);
            എറിയുക std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        ഇത്->session_id    = n.session_id;
        ഇത്->domain_id     = n.effective_domain_id;
        ഇത്->valid_session = സത്യം;
    } അല്ലെങ്കിൽ {
        ഘടന remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<അക്ഷരം *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        പൂർണ്ണസംഖ്യ err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (ശൂന്യം *) &eff, വലുപ്പം(eff));
        എങ്കിൽ (err == AEE_SUCCESS) {
            ഇത്->domain_id = eff.effective_domain_id;
        } അല്ലെങ്കിൽ {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           ഇത്->name.c_str(), err, ഇത്->domain_id);
        }
    }

    // Enable unsigned modules
    {
        ഘടന remote_rpc_control_unsigned_module u;
        u.domain = ഇത്->domain_id;
        u.enable = 1;
        പൂർണ്ണസംഖ്യ err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (ശൂന്യം *) &u, വലുപ്പം(u));
        എങ്കിൽ (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", ഇത്->c_name(), err);
            എറിയുക std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    അക്ഷരം session_uri[256];
    {
        അക്ഷരം htp_uri[256];
        snprintf(htp_uri, വലുപ്പം(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        ഘടന remote_rpc_get_uri u = {};
        u.session_id      = ഇത്->session_id;
        u.domain_name     = const_cast<അക്ഷരം *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<അക്ഷരം *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = വലുപ്പം(session_uri);

        പൂർണ്ണസംഖ്യ err = remote_session_control(FASTRPC_GET_URI, (ശൂന്യം *) &u, വലുപ്പം(u));
        എങ്കിൽ (err != AEE_SUCCESS) {
            snprintf(session_uri, വലുപ്പം(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), ഇത്->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          ഇത്->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    പൂർണ്ണസംഖ്യ err = htp_iface_open(session_uri, &ഇത്->handle);
    എങ്കിൽ (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", ഇത്->c_name(), session_uri, err);
        എറിയുക std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    ഇത്->valid_handle = സത്യം;

    // Query HW info and resolve session options
    {
        unsigned പൂർണ്ണസംഖ്യ hw_n_threads = 0;
        unsigned പൂർണ്ണസംഖ്യ hw_n_hvx     = 0;
        unsigned പൂർണ്ണസംഖ്യ hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        പൂർണ്ണസംഖ്യ hw_err = htp_iface_hwinfo(ഇത്->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        എങ്കിൽ (hw_err == 0) {
            സ്ഥിരം uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            ഇത്->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_threads) : max_n_threads;
            ഇത്->n_hvx     = ഇത്->n_threads;
            ഇത്->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            ഇത്->vtcm_size = (uint64_t)hw_vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          ഇത്->c_name(), ഇത്->n_threads, ഇത്->n_hvx, ഇത്->n_hmx,
                          (unsigned long long)(ഇത്->vtcm_size / (1024 * 1024)));
        } അല്ലെങ്കിൽ {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", ഇത്->c_name(), hw_err);
            സ്ഥിരം uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            ഇത്->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_threads;
            ഇത്->n_hvx     = ഇത്->n_threads;
            ഇത്->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            ഇത്->vtcm_size = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        ഘടന remote_rpc_control_latency l;
        l.enable = 1;

        പൂർണ്ണസംഖ്യ err = remote_handle64_control(ഇത്->handle, DSPRPC_CONTROL_LATENCY, (ശൂന്യം *) &l, വലുപ്പം(l));
        എങ്കിൽ (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", ഇത്->c_name(),
                  ഇത്->session_id, ഇത്->domain_id, session_uri, (unsigned long) ഇത്->handle);

    സ്ഥിരം size_t req_q_size = (വലുപ്പം(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    സ്ഥിരം size_t rsp_q_size = (വലുപ്പം(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(ഇത്->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (ശൂന്യം *) ഇത്,  // Callback context
                          &queue);
    എങ്കിൽ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", ഇത്->name.c_str(), (unsigned) err);
        എറിയുക std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    ഇത്->valid_queue = സത്യം;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &ഇത്->queue_id);
    എങ്കിൽ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        എറിയുക std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    എങ്കിൽ (opt_etm) {
        err = htp_iface_etm(ഇത്->handle, 1);
        എങ്കിൽ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    ഇത്->op_queue = പുതിയ ggml_hexagon_opqueue(ഇത്, opt_opbatch, opt_opqueue);

    ഇത്->fence_buf = പുതിയ ggml_hexagon_fence_buffer(ഇത്, &dev_ctx->fence_buffer_type, 64 * 1024);
    എങ്കിൽ (ഇത്->mdev.count > 1) {
        ഇത്->mdev_fence_slot = ഇത്->alloc_fence(ഇത്->mdev.count);
    }

    എങ്കിൽ (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(ഇത്);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", ഇത്->c_name(), opt_vmem);
    }
    സ്ഥിരം size_t shm_size = ഇത്->op_queue->shm_size();
    ഇത്->max_vmem = (opt_vmem > shm_size) ? (opt_vmem - shm_size) : opt_vmem;

    ഇത്->op_batch = പുതിയ ggml_hexagon_opbatch(ഇത്, opt_opbatch, ഇത്->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(ഇത്->handle, ഇത്->session_id, ഇത്->queue_id, ഇത്->n_threads, opt_nhmx, ഇത്->max_vmem, (uint32_t) opt_power_mode);
    എങ്കിൽ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", ഇത്->c_name(), (unsigned) err);
        എറിയുക std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    ഇത്->valid_iface = സത്യം;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", ഇത്->c_name(), htp_power_mode_to_str(opt_power_mode));

    എങ്കിൽ (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(ഇത്->handle, opt_profile, &pmu_conf);
        എങ്കിൽ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }
}

ശൂന്യം ggml_hexagon_session::release() noexcept(സത്യം) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", ഇത്->name.c_str());

    ഇത്->mdev.sessions.clear();

    പൂർണ്ണസംഖ്യ err;

    എങ്കിൽ (ഇത്->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(ഇത്->handle);
        എങ്കിൽ (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    delete ഇത്->op_batch;
    delete ഇത്->op_queue;
    വേണ്ടി (auto & it : ഇത്->cpy_fence_slots) {
        free_fence((ശൂന്യം *) it.second, 1);
    }
    ഇത്->cpy_fence_slots.clear();

    എങ്കിൽ (ഇത്->fence_buf) {
        unclone_buffer(ഇത്->fence_buf);
        delete ഇത്->fence_buf;
        ഇത്->fence_buf = nullptr;
    }
    ആകുമ്പോൾ (!ഇത്->cloned_buffers.empty()) {
        release_buffer(ഇത്->cloned_buffers.begin()->second.get());
    }

    എങ്കിൽ (opt_etm) {
        err = htp_iface_etm(ഇത്->handle, 0);
        എങ്കിൽ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    എങ്കിൽ (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(ഇത്->handle, 0, &pmu_conf);
        എങ്കിൽ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    എങ്കിൽ (ഇത്->valid_queue) {
        err = dspqueue_close(queue);
        എങ്കിൽ (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (unsigned) err);
        }
    }

    എങ്കിൽ (ഇത്->valid_handle) {
        htp_iface_close(ഇത്->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(സ്ഥിരം ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_എണ്ണം) noexcept(അസത്യം) {
    ഇത്->dev        = dev;
    ഇത്->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ഇത്->mdev.idx   = mdev_idx;
    ഇത്->mdev.count = mdev_എണ്ണം > 0 ? mdev_എണ്ണം : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)ഇത്) & 0xFFFF;

    ശ്രമിക്കുക {
        allocate(config);
        എങ്കിൽ (ഇത്->mdev.idx == 0 && !config.mdev_group.empty()) {
            വേണ്ടി (size_t i = 0; i < config.mdev_group.size(); i++) {
                ഇത്->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], ഇത്->dev, (uint32_t) (i + 1), ഇത്->mdev.count));
            }
        }
    } പിടിക്കുക (സ്ഥിരം std::exception & exc) {
        release();
        എറിയുക;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(സത്യം) {
    release();
}

// ** backend interface

സ്ഥിതം തർക്കം ggml_hexagon_flash_attn_is_hmx_eligible(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * q,
    സ്ഥിരം ഘടന ggml_tensor * k,
    സ്ഥിരം ഘടന ggml_tensor * v,
    സ്ഥിരം ഘടന ggml_tensor * sinks
) {
    എങ്കിൽ (sess->n_hmx == 0) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (opt_fa_select < 2) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t DK = q->ne[0];
    സ്ഥിരം uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    എങ്കിൽ (DK % 8 != 0 || DV % 8 != 0) {
        തിരികെ അസത്യം;
    }

    GGML_UNUSED(sinks);

    // Explicit force mode
    എങ്കിൽ (opt_fa_select > 2) {
        തിരികെ സത്യം;
    }

    സ്ഥിരം uint32_t M = q->ne[1];

    // Prefill or batched decode
    എങ്കിൽ (M > 1) {
        തിരികെ സത്യം;
    }

    സ്ഥിരം uint32_t n_head = q->ne[2];
    സ്ഥിരം uint32_t n_kv_heads = k->ne[2];
    സ്ഥിരം uint32_t G = n_kv_heads > 0 ? n_head / n_kv_heads : 1;
    സ്ഥിരം uint32_t S = k->ne[1];
    സ്ഥിരം uint32_t total_threads = sess->n_threads * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    എങ്കിൽ (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < total_threads))) {
        തിരികെ സത്യം;
    }

    // Tile alignment for 32-row HMX tiles
    സ്ഥിരം തർക്കം is_tile_aligned = (G > 0 && (32 % G == 0));
    എങ്കിൽ (!is_tile_aligned) {
        എങ്കിൽ (sess->n_threads >= 6) {
            തിരികെ അസത്യം;
        }
        തിരികെ S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    എങ്കിൽ (DK == 256) {
        s_cross = 4096;
    } അല്ലെങ്കിൽ എങ്കിൽ (DK <= 64) {
        s_cross = (sess->n_threads >= 8) ? 2048 : ((sess->n_threads >= 6) ? 768 : 512);
    } അല്ലെങ്കിൽ {
        s_cross = (sess->n_threads >= 8) ? 1024 : 512;
    }

    തിരികെ S >= s_cross;
}

സ്ഥിതം തർക്കം ggml_hexagon_precompute_flash_attn_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_fa_kernel_params * kparams
) {
    എങ്കിൽ (opt_fa_select < 1) {
        തിരികെ അസത്യം;
    }

    memset(kparams, 0, വലുപ്പം(*kparams));

    സ്ഥിരം ഘടന ggml_tensor * q    = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * k    = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * v    = op->src[2];
    സ്ഥിരം ഘടന ggml_tensor * mask = op->src[3];

    സ്ഥിരം uint32_t neq0 = q->ne[0];  // head_dim (DK)
    സ്ഥിരം uint32_t neq1 = q->ne[1];  // n_tokens
    സ്ഥിരം uint32_t neq2 = q->ne[2];  // n_heads

    സ്ഥിരം uint32_t nek1 = k->ne[1];  // kv_len

    സ്ഥിരം uint32_t nev0 = v->ne[0];  // head_dim (DV)

    സ്ഥിരം uint32_t DK = neq0;
    സ്ഥിരം uint32_t DV = nev0;

    സ്ഥിരം uint32_t n_kv_heads = k->ne[2];
    സ്ഥിരം uint32_t G          = neq2 / n_kv_heads;

    ദശാംശം scale         = 1.0f;
    ദശാംശം max_bias      = 0.0f;
    ദശാംശം logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], വലുപ്പം(ദശാംശം));
    memcpy(&max_bias,      &op->op_params[1], വലുപ്പം(ദശാംശം));
    memcpy(&logit_softcap, &op->op_params[2], വലുപ്പം(ദശാംശം));

    എങ്കിൽ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->head_split = (opt_fa_head_split != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    സ്ഥിരം uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    സ്ഥിരം ഘടന ggml_tensor * sinks = op->src[4];
    എങ്കിൽ (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        സ്ഥിരം uint32_t DK_pad = hex_round_up(DK, 64);
        സ്ഥിരം uint32_t DV_pad = hex_round_up(DV, 64);
        size_t Br = 0, Bc = 0;
        പൂർണ്ണസംഖ്യ ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
        എങ്കിൽ (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            എങ്കിൽ (logit_softcap == 0.0f) {
                kparams->scale = scale * (ദശാംശം) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } അല്ലെങ്കിൽ {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (ദശാംശം) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);

            സ്ഥിരം size_t row_vec_bytes = hex_align_up(Bc * വലുപ്പം(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            സ്ഥിരം size_t m_വരി_bytes = hex_align_up(Bc * വലുപ്പം(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_വരി_bytes / വലുപ്പം(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_മൂല്യങ്ങൾ(G);
            എങ്കിൽ (mask) {
                kparams->src3_div2 = init_fastdiv_മൂല്യങ്ങൾ(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_മൂല്യങ്ങൾ(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            തിരികെ സത്യം;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    സ്ഥിരം size_t size_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    സ്ഥിരം size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    സ്ഥിരം size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), mask != nullptr, sinks != nullptr, n_head, sess->n_threads);

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_മൂല്യങ്ങൾ(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_മൂല്യങ്ങൾ(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_മൂല്യങ്ങൾ(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_മൂല്യങ്ങൾ(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_മൂല്യങ്ങൾ(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_മൂല്യങ്ങൾ(q->ne[3]/v->ne[3]);
    എങ്കിൽ (mask) {
        kparams->src3_div2 = init_fastdiv_മൂല്യങ്ങൾ(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_മൂല്യങ്ങൾ(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_flash_attn_ext(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * src2 = op->src[2];
    സ്ഥിരം ഘടന ggml_tensor * src3 = op->src[3];
    സ്ഥിരം ഘടന ggml_tensor * src4 = op->src[4];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    എങ്കിൽ ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src3 && src3->type != GGML_TYPE_F16) {  // mask
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        തിരികെ അസത്യം;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    എങ്കിൽ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        തിരികെ അസത്യം;
    }

    ഘടന htp_fa_kernel_params kparams;
    എങ്കിൽ (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_gated_delta_net(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    എങ്കിൽ (opt_gdn_select < 1) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം ഘടന ggml_tensor * q     = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * k     = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * v     = op->src[2];
    സ്ഥിരം ഘടന ggml_tensor * g     = op->src[3];
    സ്ഥിരം ഘടന ggml_tensor * beta  = op->src[4];
    സ്ഥിരം ഘടന ggml_tensor * state = op->src[5];
    സ്ഥിരം ഘടന ggml_tensor * dst   = op;

    എങ്കിൽ (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int64_t S_v      = v->ne[0];
    സ്ഥിരം int64_t H        = v->ne[1];
    സ്ഥിരം int64_t n_tokens = v->ne[2];
    സ്ഥിരം int64_t n_seqs   = v->ne[3];
    സ്ഥിരം int64_t K        = ggml_get_op_params_i32(op, 0);

    എങ്കിൽ (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        തിരികെ അസത്യം;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    എങ്കിൽ (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t total_rows = (uint32_t) (H * n_seqs);
    സ്ഥിരം uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    സ്ഥിരം തർക്കം can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    എങ്കിൽ (can_use_hmx) {
        ഘടന htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        എങ്കിൽ (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, സത്യം, &n_heads_batch)) {
            തിരികെ അസത്യം;
        }
    } അല്ലെങ്കിൽ {
        ഘടന htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_threads);
        എങ്കിൽ (layout.total_bytes > sess->vtcm_size) {
            തിരികെ അസത്യം;
        }
    }

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_matmul_is_hmx_eligible(
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    തർക്കം is_matmul_id,
    തർക്കം is_batched
) {
    GGML_UNUSED(dst);

    എങ്കിൽ (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ ne00  = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne01  = src0->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne11  = src1->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne12  = src1->ne[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    എങ്കിൽ (ne01 % HTP_TENSOR_TILE_N_ROWS != 0 && is_matmul_id) {
        തിരികെ അസത്യം;
    }

    // HMX supports F16, F32, and repack quantized types.
    എങ്കിൽ (!htp_is_hmx_weight_type((ggml_type) wtype)) {
        തിരികെ അസത്യം;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    എങ്കിൽ (ne00 % HTP_TENSOR_TILE_N_COLS != 0) {
        തിരികെ അസത്യം;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    സ്ഥിരം തർക്കം is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    എങ്കിൽ (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!is_matmul_id && is_batched && is_tiled) {
        തിരികെ അസത്യം;
    }

    // HMX assumes contiguous row-major layout.
    എങ്കിൽ (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        തിരികെ അസത്യം;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    സ്ഥിരം പൂർണ്ണസംഖ്യ m = is_matmul_id ? ne12 : ne11;
    എങ്കിൽ (m <= HTP_MM_HMX_MIN_NROWS) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;

    GGML_UNUSED(dst);
}

സ്ഥിതം തർക്കം ggml_hexagon_precompute_hmx_mm_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * act,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    പൂർണ്ണസംഖ്യ wtype,
    പൂർണ്ണസംഖ്യ ne00_padded,
    പൂർണ്ണസംഖ്യ ne01_padded,
    പൂർണ്ണസംഖ്യ ne02,
    പൂർണ്ണസംഖ്യ ne11,
    പൂർണ്ണസംഖ്യ ne12,
    പൂർണ്ണസംഖ്യ ne11_padded,
    തർക്കം is_matmul_id,
    തർക്കം is_batched,
    size_t src2_size,
    size_t vtcm_budget,
    ഘടന htp_mm_kernel_params * kparams
) {
    സ്ഥിരം തർക്കം is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ aligned_tile_size = is_tiled ? htp_tensor_get_tile_padded_size(wtype) : 0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_threads = (പൂർണ്ണസംഖ്യ)sess->n_threads;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne10 = act->ne[0];

    പൂർണ്ണസംഖ്യ m_for_solver = ne11;
    പൂർണ്ണസംഖ്യ m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    എങ്കിൽ (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        സ്ഥിരം തർക്കം dst_row_split = dst ? ggml_hexagon_can_row_partition(dst) : സത്യം;
        സ്ഥിരം തർക്കം act_row_split = ggml_hexagon_can_row_partition(act);
        എങ്കിൽ (dst_row_split && act_row_split) {
            m_for_solver = (ne11 + (പൂർണ്ണസംഖ്യ) sess->mdev.count - 1) / (പൂർണ്ണസംഖ്യ) sess->mdev.count;
            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
        }
    }
    സ്ഥിരം തർക്കം pipeline = is_matmul_id ? അസത്യം : htp_mm_hmx_pipeline(m_for_solver);

    സ്ഥിരം തർക്കം is_batched_val = is_matmul_id ? അസത്യം : is_batched;
    സ്ഥിരം പൂർണ്ണസംഖ്യ group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    തർക്കം use_grouped = അസത്യം;
    പൂർണ്ണസംഖ്യ act_threads_selected = 0;

    എങ്കിൽ (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && group_size > 1) {
        // Try grouped path first
        എങ്കിൽ (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = സത്യം;
        }
    }

    എങ്കിൽ (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        പൂർണ്ണസംഖ്യ m_id_rows = 0;
        എങ്കിൽ (dst && is_matmul_id) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ n_experts = ne02 > 0 ? ne02 : 1;
            സ്ഥിരം size_t total_expert_rows = (size_t) dst->ne[1] * dst->ne[2];
            പൂർണ്ണസംഖ്യ m_per_expert = (പൂർണ്ണസംഖ്യ) ((total_expert_rows + n_experts - 1) / n_experts);
            എങ്കിൽ (sess->mdev.count > 1 && ggml_hexagon_can_row_partition(dst)) {
                m_per_expert = (m_per_expert + (പൂർണ്ണസംഖ്യ) sess->mdev.count - 1) / (പൂർണ്ണസംഖ്യ) sess->mdev.count;
            }
            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
        }
        സ്ഥിരം uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
        എങ്കിൽ (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            തിരികെ അസത്യം;
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
    kparams->div_n_act_threads = init_fastdiv_മൂല്യങ്ങൾ(act_threads_selected);
    kparams->div_ne00_padded   = init_fastdiv_മൂല്യങ്ങൾ(ne00_padded);
    kparams->vtcm_act_size     = 0;
    kparams->vtcm_bias_size    = (int32_t) src2_size;
    kparams->vtcm_dst_size     = 0;

    എങ്കിൽ (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } അല്ലെങ്കിൽ {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    തിരികെ സത്യം;
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_hvx_mm_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    പൂർണ്ണസംഖ്യ wtype,
    തർക്കം is_tiled,
    പൂർണ്ണസംഖ്യ ne02,
    പൂർണ്ണസംഖ്യ ne03,
    പൂർണ്ണസംഖ്യ ne10,
    പൂർണ്ണസംഖ്യ ne11,
    പൂർണ്ണസംഖ്യ ne12,
    പൂർണ്ണസംഖ്യ ne13,
    തർക്കം is_matmul_id,
    സ്ഥിരം size_t src2_row_size,
    size_t vtcm_budget,
    ഘടന htp_mm_kernel_params * kparams
) {
    എങ്കിൽ (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        തിരികെ;
    }

    kparams->n_hmx = 0;
    kparams->n_threads = sess->n_threads;

    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);

    സ്ഥിരം പൂർണ്ണസംഖ്യ src1_nrows = ne11 * ne12 * ne13;

    എങ്കിൽ (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_size = htp_tensor_get_tile_size(wtype);
        kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);

        സ്ഥിരം തർക്കം k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        എങ്കിൽ (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (പൂർണ്ണസംഖ്യ) sess->n_threads)
                                  ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);

            ഘടന htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            വേണ്ടി (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], kparams->act_row_size, 0, d, സത്യം, അസത്യം
                );
                എങ്കിൽ (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    നിർത്തുക;
                }
            }
            എങ്കിൽ (L.total_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                തിരികെ;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = 0;
            kparams->vtcm_dst_size  = L.dst_bytes;
            തിരികെ;
        } അല്ലെങ്കിൽ {
            തർക്കം try_tiled = (k_align && opt_mm_select >= 1);
            എങ്കിൽ (try_tiled) {
                kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                എങ്കിൽ (src1_nrows < (പൂർണ്ണസംഖ്യ) sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } അല്ലെങ്കിൽ {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                ഘടന htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                വേണ്ടി (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, അസത്യം, അസത്യം
                    );
                    എങ്കിൽ (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        നിർത്തുക;
                    }
                }

                uint32_t m_chunk = 0;
                എങ്കിൽ (htp_mm_hvx_solve_vtcm_params(
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
                    തിരികെ;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            തിരികെ;
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        തിരികെ;
    } അല്ലെങ്കിൽ എങ്കിൽ (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        ഘടന htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        എങ്കിൽ (htp_mm_hvx_solve_vtcm_params(
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
            തിരികെ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        തിരികെ;
    } അല്ലെങ്കിൽ എങ്കിൽ (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        ഘടന htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        എങ്കിൽ (htp_mm_hvx_solve_vtcm_params(
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
            തിരികെ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        തിരികെ;
    } അല്ലെങ്കിൽ എങ്കിൽ (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        ഘടന htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        എങ്കിൽ (htp_mm_hvx_solve_vtcm_params(
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
            തിരികെ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        തിരികെ;
    } അല്ലെങ്കിൽ {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        തിരികെ;
    }
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_matmul_params_impl(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    സ്ഥിരം size_t src2_row_size,
    സ്ഥിരം size_t src2_size,
    ഘടന htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));

    സ്ഥിരം പൂർണ്ണസംഖ്യ ne00 = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne01 = src0->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne02 = src0->ne[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne03 = src0->ne[3];

    സ്ഥിരം പൂർണ്ണസംഖ്യ ne10 = src1->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne11 = src1->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne12 = src1->ne[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne13 = src1->ne[3];

    സ്ഥിരം തർക്കം is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne11_padded = hex_round_up(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne01_tiled  = hex_round_up(ne01, 32);

    സ്ഥിരം തർക്കം is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    സ്ഥിരം തർക്കം is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    സ്ഥിരം size_t vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    തർക്കം hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    എങ്കിൽ (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        എങ്കിൽ (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
            പോകുക finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_size, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_മൂല്യങ്ങൾ(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_മൂല്യങ്ങൾ(ne11);
    kparams->div_r2       = init_fastdiv_മൂല്യങ്ങൾ(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_മൂല്യങ്ങൾ(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_മൂല്യങ്ങൾ(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
സ്ഥിതം തർക്കം ggml_hexagon_rows_stride(സ്ഥിരം int64_t * ne, സ്ഥിരം size_t * nb, size_t * stride) {
    size_t s = 0, next = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i < GGML_MAX_DIMS; i++) {
        എങ്കിൽ (ne[i] == 1) തുടരുക;
        എങ്കിൽ (s == 0) { s = nb[i]; next = s * ne[i]; തുടരുക; }
        എങ്കിൽ (nb[i] != next) തിരികെ അസത്യം;
        next *= ne[i];
    }
    *stride = s ? s : nb[1];
    തിരികെ സത്യം;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
സ്ഥിതം തർക്കം ggml_hexagon_matmul_can_collapse(സ്ഥിരം ഘടന ggml_tensor * src0, സ്ഥിരം ഘടന ggml_tensor * src1, സ്ഥിരം ഘടന ggml_tensor * dst) {
    size_t s1, sd;
    തിരികെ (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_size(src1->type) && ggml_hexagon_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_size(dst->type) && ggml_hexagon_rows_stride(dst->ne, dst->nb, &sd);
}

സ്ഥിതം തർക്കം ggml_hexagon_matmul_add_can_collapse(
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * src2,
    സ്ഥിരം ഘടന ggml_tensor * dst
) {
    എങ്കിൽ (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!src2) {
        തിരികെ സത്യം;
    }
    സ്ഥിരം int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    എങ്കിൽ (src2_nrows == 1) {
        തിരികെ സത്യം;
    }
    size_t s2;
    തിരികെ src2->nb[0] == ggml_type_size(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_rows_stride(src2->ne, src2->nb, &s2);
}

സ്ഥിതം ggml_tensor ggml_hexagon_tensor_collapse_rows(സ്ഥിരം ഘടന ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    തിരികെ c;
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_matmul_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_mm_kernel_params * kparams
) {
    എങ്കിൽ (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        സ്ഥിരം ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        സ്ഥിരം ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        തിരികെ;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_fused_matmul_add_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * src2,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_mm_kernel_params * kparams
) {
    എങ്കിൽ (ggml_hexagon_matmul_add_can_collapse(src0, src1, src2, dst)) {
        സ്ഥിരം ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        സ്ഥിരം ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        സ്ഥിരം ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        സ്ഥിരം ഘടന ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        സ്ഥിരം size_t src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
        kparams->collapse = 1;
        തിരികെ;
    }
    സ്ഥിരം size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

സ്ഥിതം തർക്കം ggml_hexagon_precompute_ദ്വിമാന_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    uint32_t op,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_ദ്വിമാന_kernel_params * kparams,
    സ്ഥിരം ഘടന ggml_tensor * src2
) {
    memset(kparams, 0, വലുപ്പം(*kparams));

    സ്ഥിരം size_t elem_size = ggml_type_size(src0->type);
    സ്ഥിരം size_t src0_row_size = src0->ne[0] * elem_size;
    സ്ഥിരം size_t src1_row_size = src1->ne[0] * elem_size;
    സ്ഥിരം size_t dst_row_size  = dst->ne[0]  * elem_size;

    സ്ഥിരം size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    സ്ഥിരം size_t src1_row_size_aligned = hex_round_up(src1_row_size, 128);
    സ്ഥിരം size_t dst_row_size_aligned  = hex_round_up(dst_row_size,  128);

    എങ്കിൽ (!htp_ദ്വിമാന_strides_safe(src0, src1, dst, elem_size)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം തർക്കം is_add_id = op == HTP_OP_ADD_ID;
    സ്ഥിരം തർക്കം is_scalar = !is_add_id && src1->ne[0] == 1;
    സ്ഥിരം തർക്കം is_row_bcast = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    സ്ഥിരം തർക്കം is_same_shape = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    സ്ഥിരം തർക്കം is_complex   = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    സ്ഥിരം തർക്കം is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    സ്ഥിരം തർക്കം is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    എങ്കിൽ (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        സ്ഥിരം uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        സ്ഥിരം uint32_t n_threads = sess->n_threads;
        സ്ഥിരം uint32_t max_chunk_elems = 32768 / elem_size;
        സ്ഥിരം uint32_t min_chunk_elems = 256;
        സ്ഥിരം uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
        സ്ഥിരം uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        സ്ഥിരം uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

        kparams->kernel_type           = HTP_ദ്വിമാന_KERNEL_CHUNKED;
        kparams->n_threads             = n_threads;
        kparams->rows_per_buffer       = 1;
        kparams->src0_row_size_aligned = src0_row_size_aligned;
        kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
        kparams->dst_row_size_aligned  = dst_row_size_aligned;
        kparams->src1_size             = 0;
        kparams->chunk_size            = chunk_size;
        kparams->chunk_bytes           = chunk_bytes;
        kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

        ഘടന htp_ദ്വിമാന_vtcm_layout L;
        htp_ദ്വിമാന_vtcm_layout_build(&L, kparams, sess->vtcm_size);
        എങ്കിൽ (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
            തിരികെ അസത്യം;
        }

        kparams->vtcm_size = L.total_bytes;
        തിരികെ സത്യം;
    }

    എണ്ണൽ htp_ദ്വിമാന_kernel_type kernel_type;
    size_t src1_size = 0;
    size_t src2_size = 0;

    എങ്കിൽ (is_add_id) {
        GGML_ASSERT(src2 != nullptr);
        kernel_type = HTP_ദ്വിമാന_KERNEL_ADD_ID;
        src1_size = hex_round_up(src1->ne[1] * src1_row_size_aligned, 128);
        src2_size = hex_round_up(ggml_nelements(src2) * വലുപ്പം(int32_t), 128);
        എങ്കിൽ (src1_size + src2_size > sess->vtcm_size) {
            തിരികെ അസത്യം;
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (is_row_bcast) {
        kernel_type = HTP_ദ്വിമാന_KERNEL_ROW_BCAST;
        src1_size = src1_row_size_aligned;
    } അല്ലെങ്കിൽ എങ്കിൽ (is_scalar) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size, 128);
        എങ്കിൽ (src1_size > sess->vtcm_size) {
            തിരികെ അസത്യം;
        }
        kernel_type = HTP_ദ്വിമാന_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } അല്ലെങ്കിൽ എങ്കിൽ (is_same_shape) {
        kernel_type = HTP_ദ്വിമാന_KERNEL_SAME_SHAPE;
    } അല്ലെങ്കിൽ എങ്കിൽ (is_complex) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        എങ്കിൽ (src1_size > sess->vtcm_size) {
            തിരികെ അസത്യം;
        }
        kernel_type = HTP_ദ്വിമാന_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } അല്ലെങ്കിൽ {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        എങ്കിൽ (src1_size > sess->vtcm_size) {
            തിരികെ അസത്യം;
        }
        kernel_type = HTP_ദ്വിമാന_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_threads = sess->n_threads;
    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;
    kparams->src1_size             = src1_size;
    kparams->src2_size             = src2_size;

    സ്ഥിരം തർക്കം src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    സ്ഥിരം തർക്കം src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    സ്ഥിരം തർക്കം dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    സ്ഥിരം തർക്കം src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    സ്ഥിരം തർക്കം src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    സ്ഥിരം തർക്കം dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    തർക്കം split_at_ne01 = അസത്യം;
    എങ്കിൽ (src0->ne[2] > 1) {
        എങ്കിൽ (kernel_type == HTP_ദ്വിമാന_KERNEL_SAME_SHAPE) {
            split_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } അല്ലെങ്കിൽ എങ്കിൽ (kernel_type == HTP_ദ്വിമാന_KERNEL_ROW_BCAST) {
            split_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } അല്ലെങ്കിൽ {
            split_at_ne01 = സത്യം;
        }
    }
    kparams->split_at_ne01 = split_at_ne01 ? 1 : 0;

    തർക്കം split_at_ne02 = അസത്യം;
    എങ്കിൽ (src0->ne[3] > 1) {
        എങ്കിൽ (kernel_type == HTP_ദ്വിമാന_KERNEL_SAME_SHAPE) {
            split_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } അല്ലെങ്കിൽ എങ്കിൽ (kernel_type == HTP_ദ്വിമാന_KERNEL_ROW_BCAST) {
            split_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } അല്ലെങ്കിൽ {
            split_at_ne02 = സത്യം;
        }
    }
    kparams->split_at_ne02 = split_at_ne02 ? 1 : 0;

    ഘടന htp_ദ്വിമാന_vtcm_layout L;
    htp_ദ്വിമാന_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    എങ്കിൽ (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {
        എങ്കിൽ (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            സ്ഥിരം uint32_t total_elems = (uint32_t) ggml_nelements(src0);
            സ്ഥിരം uint32_t n_threads = sess->n_threads;
            സ്ഥിരം uint32_t max_chunk_elems = 32768 / elem_size;
            സ്ഥിരം uint32_t min_chunk_elems = 256;
            സ്ഥിരം uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
            സ്ഥിരം uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
            സ്ഥിരം uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

            kparams->kernel_type           = HTP_ദ്വിമാന_KERNEL_CHUNKED;
            kparams->n_threads             = n_threads;
            kparams->rows_per_buffer       = 1;
            kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
            kparams->src1_size             = 0;
            kparams->chunk_size            = chunk_size;
            kparams->chunk_bytes           = chunk_bytes;
            kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

            htp_ദ്വിമാന_vtcm_layout_build(&L, kparams, sess->vtcm_size);
            എങ്കിൽ (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
                തിരികെ അസത്യം;
            }

            kparams->vtcm_size = L.total_bytes;
            തിരികെ സത്യം;
        }
        തിരികെ അസത്യം;
    }

    kparams->rows_per_buffer = L.rows_per_buffer;
    kparams->vtcm_size       = L.total_bytes;

    തിരികെ സത്യം;
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_unary_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    uint32_t op,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));

    സ്ഥിരം uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    സ്ഥിരം uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    സ്ഥിരം size_t elem_size = ggml_type_size(src0->type);

    സ്ഥിരം size_t src0_data_row_size = src0->ne[0] * elem_size;
    സ്ഥിരം size_t dst_data_row_size  = dst->ne[0]  * ggml_type_size(dst->type);

    സ്ഥിരം size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    സ്ഥിരം size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = 0;
    തർക്കം broadcast_weight = അസത്യം;

    എങ്കിൽ (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_data_row_size = src1->ne[0] * ggml_type_size(src1->type);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->broadcast_weight      = broadcast_weight;

    ഘടന htp_unary_vtcm_layout L;
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

    സ്ഥിരം uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_മൂല്യങ്ങൾ(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_മൂല്യങ്ങൾ(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_മൂല്യങ്ങൾ(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_മൂല്യങ്ങൾ(tiles_per_row);
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_get_rows_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_get_rows_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));

    സ്ഥിരം uint32_t ne00 = src0->ne[0];
    സ്ഥിരം uint32_t ne02 = src0->ne[2];
    സ്ഥിരം uint32_t ne03 = src0->ne[3];

    സ്ഥിരം uint32_t ne10 = src1->ne[0];
    സ്ഥിരം uint32_t ne11 = src1->ne[1];
    സ്ഥിരം uint32_t ne12 = src1->ne[2];
    സ്ഥിരം uint32_t nr = ne10 * ne11 * ne12;

    സ്ഥിരം തർക്കം tiled = ggml_hexagon_tensor_is_tiled(src0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    എങ്കിൽ (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } അല്ലെങ്കിൽ {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    സ്ഥിരം uint32_t total_tasks = nr;

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, total_tasks);

    ഘടന htp_get_rows_vtcm_layout vtcm_layout = {};
    ആകുമ്പോൾ (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, kparams->n_threads);
        എങ്കിൽ (vtcm_layout.total_bytes <= sess->vtcm_size) {
            നിർത്തുക;
        }
        --kparams->n_threads;
    }

    എങ്കിൽ (kparams->n_threads == 0 && total_tasks > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, 1);
    }

    kparams->vtcm_size = (total_tasks == 0) ? 0 : vtcm_layout.total_bytes;
    kparams->tasks_per_thread = kparams->n_threads > 0 ? (total_tasks + kparams->n_threads - 1) / kparams->n_threads : 0;
    kparams->total_tasks = total_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_മൂല്യങ്ങൾ(ne10) : fastdiv_മൂല്യങ്ങൾ{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_മൂല്യങ്ങൾ(ne10 * ne11) : fastdiv_മൂല്യങ്ങൾ{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_മൂല്യങ്ങൾ(ne02) : fastdiv_മൂല്യങ്ങൾ{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_മൂല്യങ്ങൾ(ne03) : fastdiv_മൂല്യങ്ങൾ{0, 0};
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_set_rows_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0, //മൂല്യങ്ങൾ
    സ്ഥിരം ഘടന ggml_tensor * src1, // indices
    സ്ഥിരം ഘടന ggml_tensor * dst,  // destination
    ഘടന htp_set_rows_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));

    സ്ഥിരം uint32_t nr = src0->ne[1];

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, nr);
    kparams->tasks_per_thread = (nr + kparams->n_threads - 1) / kparams->n_threads;
    kparams->total_tasks = nr;

    kparams->div_ne11 = init_fastdiv_മൂല്യങ്ങൾ(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_മൂല്യങ്ങൾ(src1->ne[2]);
    kparams->div_tasks_per_thread = init_fastdiv_മൂല്യങ്ങൾ(kparams->tasks_per_thread);
    kparams->div_ne02 = init_fastdiv_മൂല്യങ്ങൾ(src0->ne[2]);

    ഘടന htp_set_rows_vtcm_layout vtcm_layout;
    htp_set_rows_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_threads);
    kparams->vtcm_size = vtcm_layout.total_bytes;
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_softmax_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));

    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];

    സ്ഥിരം uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    സ്ഥിരം uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);

    ദശാംശം scale = 1.0f;
    ദശാംശം max_bias = 0.0f;
    memcpy(&scale,    &op->op_params[0], വലുപ്പം(ദശാംശം));
    memcpy(&max_bias, &op->op_params[1], വലുപ്പം(ദശാംശം));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    സ്ഥിരം uint32_t n_head = src0->ne[2];
    സ്ഥിരം uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    kparams->n_head      = n_head;
    kparams->n_head_log2 = n_head_log2;

    എങ്കിൽ (max_bias > 0.0f && n_head_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_head_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    } അല്ലെങ്കിൽ {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    സ്ഥിരം uint32_t ne00 = src0->ne[0];
    സ്ഥിരം uint32_t ne10 = src1 ? src1->ne[0] : 1;

    ഘടന htp_softmax_vtcm_layout layout;
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
    എങ്കിൽ (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } അല്ലെങ്കിൽ എങ്കിൽ (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F16;
    } അല്ലെങ്കിൽ {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    എങ്കിൽ (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_മൂല്യങ്ങൾ(src0->ne[1]);
    എങ്കിൽ (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_മൂല്യങ്ങൾ(src0->ne[2]);
    സ്ഥിരം uint32_t ne12 = src1 ? src1->ne[2] : 1;
    സ്ഥിരം uint32_t ne13 = src1 ? src1->ne[3] : 1;
    എങ്കിൽ (ne12 > 0) kparams->div_ne12 = init_fastdiv_മൂല്യങ്ങൾ(ne12);
    എങ്കിൽ (ne13 > 0) kparams->div_ne13 = init_fastdiv_മൂല്യങ്ങൾ(ne13);
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_rope_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));

    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src2 = op->src[2];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    സ്ഥിരം uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    സ്ഥിരം uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    സ്ഥിരം uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ഘടന htp_rope_vtcm_layout layout;
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

    എങ്കിൽ (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_മൂല്യങ്ങൾ(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_മൂല്യങ്ങൾ(dst->ne[1]);
    }
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_ssm_conv_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * src1,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));

    സ്ഥിരം uint32_t d_conv  = (uint32_t) src1->ne[0];
    സ്ഥിരം uint32_t d_inner = (uint32_t) src0->ne[1];
    സ്ഥിരം uint32_t n_t     = (uint32_t) dst->ne[1];
    സ്ഥിരം uint32_t n_s     = (uint32_t) dst->ne[2];
    സ്ഥിരം uint32_t ncs     = (uint32_t) src0->ne[0];

    സ്ഥിരം uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    സ്ഥിരം uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
    സ്ഥിരം uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);

    സ്ഥിരം uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * വലുപ്പം(ദശാംശം), 128) + 128;
    സ്ഥിരം uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * വലുപ്പം(ദശാംശം), 128);
    സ്ഥിരം uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

    uint32_t vtcm_src0_per_thread = 0;
    uint32_t vtcm_dst_per_thread  = 0;

    എങ്കിൽ (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;

        സ്ഥിരം uint32_t src0_tile_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * വലുപ്പം(ദശാംശം), 128);
        സ്ഥിരം uint32_t src0_T_bytes        = hex_round_up(d_conv * d_inner_per_thread * വലുപ്പം(ദശാംശം), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw_bytes + src0_T_bytes;

        സ്ഥിരം uint32_t dst_tile_bytes = hex_round_up(d_inner_per_thread * വലുപ്പം(ദശാംശം), 128);
        vtcm_dst_per_thread = 2 * dst_tile_bytes;
    } അല്ലെങ്കിൽ {
        സ്ഥിരം size_t vtcm_budget = (sess->vtcm_size > 0 ? sess->vtcm_size / n_threads : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        സ്ഥിരം uint32_t src0_block_T    = hex_round_up(ncs * 32 * വലുപ്പം(ദശാംശം), 128);
        സ്ഥിരം size_t   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
        സ്ഥിരം size_t   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);

        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
        target_max_tile = (std::max)(target_max_tile, 32u);
        target_max_tile = (std::min)(target_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * വലുപ്പം(ദശാംശം)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        എങ്കിൽ (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        എങ്കിൽ (d_inner_tile > target_max_tile) {
            d_inner_tile = target_max_tile;
        }
        എങ്കിൽ (d_inner_tile > d_inner_per_thread) {
            d_inner_tile = d_inner_per_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        സ്ഥിരം uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * വലുപ്പം(ദശാംശം), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw + src0_block_T;

        vtcm_dst_per_thread = 2 * hex_round_up(d_inner_tile * n_t * വലുപ്പം(ദശാംശം), 128);
    }

    kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
    kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
    kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;

    kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
    kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
    kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
    kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_gated_delta_net_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));

    സ്ഥിരം ഘടന ggml_tensor * q     = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * k     = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * v     = op->src[2];
    സ്ഥിരം ഘടന ggml_tensor * g     = op->src[3];
    സ്ഥിരം ഘടന ggml_tensor * state = op->src[5];

    സ്ഥിരം uint32_t S_v      = (uint32_t) v->ne[0];
    സ്ഥിരം uint32_t H        = (uint32_t) v->ne[1];
    സ്ഥിരം uint32_t n_tokens = (uint32_t) v->ne[2];
    സ്ഥിരം uint32_t n_seqs   = (uint32_t) v->ne[3];
    സ്ഥിരം uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    സ്ഥിരം uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    സ്ഥിരം uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    സ്ഥിരം uint32_t total_rows = H * n_seqs;
    സ്ഥിരം uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    സ്ഥിരം തർക്കം can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    ഘടന htp_gdn_hmx_vtcm_layout hmx_layout;
    ഘടന htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    എങ്കിൽ (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, സത്യം, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_size      = HTP_GDN_CHUNK_SIZE;
        kparams->n_chunks        = (n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_size       = (uint32_t) hmx_layout.total_bytes;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_bytes;
        kparams->vtcm_per_thread = (uint32_t) (hmx_layout.total_bytes / (n_threads > 0 ? n_threads : 1));
    } അല്ലെങ്കിൽ {
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
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / വലുപ്പം(ദശാംശം));
    kparams->state_size_per_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((ദശാംശം) S_v);

    എങ്കിൽ (H > 0)                  kparams->div_H         = init_fastdiv_മൂല്യങ്ങൾ(H);
    എങ്കിൽ (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) q->ne[1]);
    എങ്കിൽ (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) k->ne[1]);
    എങ്കിൽ (rq3 > 0)                kparams->div_rq3       = init_fastdiv_മൂല്യങ്ങൾ(rq3);
    എങ്കിൽ (rk3 > 0)                kparams->div_rk3       = init_fastdiv_മൂല്യങ്ങൾ(rk3);
    എങ്കിൽ (kparams->n_threads > 0) kparams->div_n_threads = init_fastdiv_മൂല്യങ്ങൾ(kparams->n_threads);
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_sort_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    തർക്കം is_മുകൾ_k,
    ഘടന htp_sort_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));

    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    സ്ഥിരം uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    സ്ഥിരം uint32_t ne00       = src0->ne[0];
    സ്ഥിരം uint32_t k          = dst->ne[0];

    int32_t order = GGML_SORT_ORDER_DESC;
    എങ്കിൽ (!is_മുകൾ_k) {
        order = ((സ്ഥിരം int32_t *) op->op_params)[0];
    }

    സ്ഥിരം uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    സ്ഥിരം size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    ഘടന htp_sort_vtcm_layout layout;
    തർക്കം ok = htp_sort_solve_layout(&layout, ne00, total_rows, k, n_threads_max, vtcm_budget, is_മുകൾ_k);
    GGML_ASSERT(ok);

    kparams->n_threads         = (int32_t) layout.n_threads;
    kparams->total_rows        = (int32_t) total_rows;
    kparams->row_start         = 0;
    kparams->row_end           = (int32_t) total_rows;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->order             = order;
    kparams->is_മുകൾ_k          = is_മുകൾ_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_size         = (int32_t) layout.total_bytes;
    kparams->phase1_slot_size  = (int32_t) layout.phase1_slot_size;
    kparams->merge_മൂല്യങ്ങൾ_off  = (int32_t) layout.merge_മൂല്യങ്ങൾ_off;
    kparams->merge_indices_off = (int32_t) layout.merge_indices_off;
    kparams->merge_elems       = (int32_t) layout.merge_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

സ്ഥിതം ശൂന്യം ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    സ്ഥിരം uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    സ്ഥിരം int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    എങ്കിൽ (numer_hi < 0) {
        *ox_hi = 0;
    } അല്ലെങ്കിൽ {
        സ്ഥിരം uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    എങ്കിൽ (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

സ്ഥിതം തർക്കം ggml_hexagon_pool_shape_fits(
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * dst
) {
    തിരികെ ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_pool_2d_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0,
    സ്ഥിരം ഘടന ggml_tensor * dst,
    ഘടന htp_pool_2d_kernel_params * kparams,
    തർക്കം is_pool_1d
) {
    memset(kparams, 0, വലുപ്പം(*kparams));

    സ്ഥിരം uint32_t src_x = (uint32_t) src0->ne[0];
    സ്ഥിരം uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    സ്ഥിരം uint32_t dst_x = (uint32_t) dst->ne[0];
    സ്ഥിരം uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    സ്ഥിരം uint32_t src_plane_bytes = src_x * src_y * വലുപ്പം(ദശാംശം);
    സ്ഥിരം uint32_t dst_plane_bytes = dst_x * dst_y * വലുപ്പം(ദശാംശം);
    സ്ഥിരം uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    സ്ഥിരം uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    ഘടന htp_pool_vtcm_layout layout;
    സ്ഥിരം തർക്കം ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, sess->vtcm_size);
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
    kparams->avg_divide_എണ്ണം = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    സ്ഥിരം തർക്കം narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_path = (kparams->fast_path && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (ദശാംശം) (kparams->kernel_x * kparams->kernel_y);
}

സ്ഥിതം തർക്കം ggml_hexagon_precompute_concat_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (!src0 || !src1 || !dst) {
        തിരികെ അസത്യം;
    }

    പൂർണ്ണസംഖ്യ dim = ((സ്ഥിരം int32_t *) op->op_params)[0];
    എങ്കിൽ (dim < 0 || dim >= GGML_MAX_DIMS) {
        തിരികെ അസത്യം;
    }
    kparams->dim = dim;

    എങ്കിൽ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src0->type != dst->type || src1->type != dst->type) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t type_size = ggml_type_size(dst->type);

    വേണ്ടി (പൂർണ്ണസംഖ്യ d = 0; d < GGML_MAX_DIMS; d++) {
        സ്ഥിരം int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        എങ്കിൽ (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            തിരികെ അസത്യം;
        }
    }

    സ്ഥിരം uint32_t max_gather_ne0 = 128 / type_size;
    സ്ഥിരം uint32_t total_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    സ്ഥിരം തർക്കം b_dense = (src1->nb[0] == type_size);
    സ്ഥിരം തർക്കം gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_size &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_size
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_size);

    എങ്കിൽ (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_size && dst->nb[1] == (size_t) dst->ne[0] * type_size &&
        src0->nb[0] == type_size &&
        (src1->nb[0] == type_size || src1->nb[1] == type_size) &&
        gather_strides_ok &&
        htp_gather_rows_tensor_outer_contiguous(dst) &&
        htp_gather_rows_tensor_outer_contiguous(src0) &&
        htp_gather_rows_tensor_outer_contiguous(src1)) {

        എങ്കിൽ (htp_gather_rows_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_size, total_rows,
                                         sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ROWS;
            kparams->n_threads   = (uint8_t) sess->n_threads;
            kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
            തിരികെ സത്യം;
        }
    }

    സ്ഥിരം തർക്കം dma_strides_ok = (src0->nb[0] == type_size && src1->nb[0] == type_size && dst->nb[0] == type_size);

    എങ്കിൽ (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_threads   = 1;
        തിരികെ സത്യം;
    }

    സ്ഥിരം തർക്കം is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    സ്ഥിരം തർക്കം is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    സ്ഥിരം തർക്കം transposed_rows_ok = (src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size);

    എങ്കിൽ (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        സ്ഥിരം uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 8;
        ഘടന htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_size, n_threads);

        എങ്കിൽ (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
            തിരികെ അസത്യം;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_threads                         = n_threads;
        kparams->vtcm_size                         = layout.total_bytes;
        kparams->u.transposed.spad0_size_per_thread = layout.src0_spad_size_per_thread;
        kparams->u.transposed.spad1_size_per_thread = layout.src1_spad_size_per_thread;
        തിരികെ സത്യം;
    }

    തിരികെ അസത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_precompute_cpy_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (!src0 || !dst) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int64_t nelem_src = ggml_nelements(src0);
    സ്ഥിരം int64_t nelem_dst = ggml_nelements(dst);
    എങ്കിൽ (nelem_src != nelem_dst || nelem_src < 0) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t src_type_size = ggml_type_size(src0->type);
    സ്ഥിരം uint32_t dst_type_size = ggml_type_size(dst->type);

    kparams->src0_type_size = (uint8_t) src_type_size;
    kparams->dst_type_size  = (uint8_t) dst_type_size;
    kparams->total_elems    = (uint32_t) nelem_src;

    എങ്കിൽ (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        തിരികെ സത്യം;
    }

    എങ്കിൽ (nelem_src == 1) {
        എങ്കിൽ (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            തിരികെ സത്യം;
        }
        എങ്കിൽ ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            തിരികെ സത്യം;
        }
        തിരികെ അസത്യം;
    }

    സ്ഥിരം തർക്കം sametype = (src0->type == dst->type);

    തർക്കം same_extents = സത്യം;
    വേണ്ടി (പൂർണ്ണസംഖ്യ d = 0; d < GGML_MAX_DIMS; d++) {
        എങ്കിൽ (src0->ne[d] != dst->ne[d]) {
            same_extents = അസത്യം;
            നിർത്തുക;
        }
    }

    സ്ഥിരം തർക്കം transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_size) || (dst->nb[0] != dst_type_size) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_size) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_size);
    സ്ഥിരം തർക്കം sameshape  = same_extents && !transposed;

    സ്ഥിരം തർക്കം src_is_contiguous = ggml_is_contiguous(src0);
    സ്ഥിരം തർക്കം dst_is_contiguous = ggml_is_contiguous(dst);

    എങ്കിൽ (sametype) {
        എങ്കിൽ (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            തിരികെ സത്യം;
        }

        സ്ഥിരം uint32_t max_gather_ne0 = 128 / src_type_size;
        സ്ഥിരം uint32_t total_rows = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        എങ്കിൽ ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_size && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_size &&
            total_rows > 1 &&
            htp_gather_rows_tensor_outer_contiguous(src0)) {

            എങ്കിൽ (htp_gather_rows_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, അസത്യം,
                                             src_type_size, total_rows,
                                             sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ROWS;
                kparams->n_threads   = (uint8_t) sess->n_threads;
                kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
                തിരികെ സത്യം;
            }
        }

        എങ്കിൽ (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            തിരികെ സത്യം;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_threads   = sess->n_threads > 0 ? (uint8_t) sess->n_threads : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        തിരികെ സത്യം;
    }

    എങ്കിൽ (!sameshape) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം തർക്കം valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    എങ്കിൽ (!valid_conversion) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 4;
    ഘടന htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_size, dst_type_size, n_threads);

    എങ്കിൽ (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
        തിരികെ അസത്യം;
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
    kparams->u.convert.div_ne01      = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_മൂല്യങ്ങൾ((uint32_t) (src0->ne[2] * src0->ne[1]));

    തിരികെ സത്യം;
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_fused_mmnx_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0, // W0
    സ്ഥിരം ഘടന ggml_tensor * act,  // x
    int32_t n_weights,
    ഘടന htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));
    kparams->n_threads = sess->n_threads;

    സ്ഥിരം പൂർണ്ണസംഖ്യ ne00 = src0->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne01 = src0->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne02 = src0->ne[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne03 = src0->ne[3];

    സ്ഥിരം പൂർണ്ണസംഖ്യ ne10 = act->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne11 = act->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne12 = act->ne[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne13 = act->ne[3];

    സ്ഥിരം തർക്കം is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne11_padded = hex_round_up(ne11, 32);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ne01_tiled  = hex_round_up(ne01, 32);

    സ്ഥിരം size_t vtcm_budget = sess->vtcm_size;
    സ്ഥിരം തർക്കം is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    തർക്കം hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    എങ്കിൽ (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, nullptr, അസത്യം, is_batched)) {
        എങ്കിൽ (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, അസത്യം, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            പോകുക finalize;
        }
    }

    എങ്കിൽ (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        തിരികെ;
    }

    {
        സ്ഥിരം പൂർണ്ണസംഖ്യ act_nrows = ne11 * ne12 * ne13;
        സ്ഥിരം size_t act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
        സ്ഥിരം size_t src0_row_size = src0->nb[1];

        സ്ഥിരം uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        വേണ്ടി (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            ഘടന htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
                0, src0_row_size, act_row_size, 0, d, അസത്യം, സത്യം
            );
            എങ്കിൽ (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                നിർത്തുക;
            }
        }

        ഘടന htp_mm_hvx_vtcm_layout L;
        തർക്കം try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
            0, src0_row_size, act_row_size, 0, best_n_prefetch, അസത്യം, സത്യം
        );

        എങ്കിൽ (try_tiled && L.total_bytes <= sess->vtcm_size) {
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
        } അല്ലെങ്കിൽ {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            തിരികെ;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_മൂല്യങ്ങൾ(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_മൂല്യങ്ങൾ(ne11);
    kparams->div_r2       = init_fastdiv_മൂല്യങ്ങൾ(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_മൂല്യങ്ങൾ(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_മൂല്യങ്ങൾ(ne12);
}

സ്ഥിതം ശൂന്യം ggml_hexagon_precompute_fused_mmidnx_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * src0, // W0
    സ്ഥിരം ഘടന ggml_tensor * act,  // x
    സ്ഥിരം ഘടന ggml_tensor * dst,  // dst0
    int32_t n_weights,
    ഘടന htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

സ്ഥിതം തർക്കം ggml_hexagon_op_mul_mat_use_fwht(സ്ഥിരം ഘടന ggml_tensor * dst) {
    എങ്കിൽ (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം ഘടന ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = dst->src[1];

    സ്ഥിരം int64_t n = src1->ne[0];
    എങ്കിൽ (n < 32 || (n & (n - 1)) != 0) {
        തിരികെ അസത്യം;
    }

    തിരികെ src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_rows(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_mul_mat(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * dst) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = dst->src[1];

    എങ്കിൽ (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        എങ്കിൽ (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            തിരികെ അസത്യം;
        }
        ഘടന htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, nullptr, dst, &ukparams);
        തിരികെ ukparams.vtcm_row_per_thread > 0 && (size_t) ukparams.vtcm_size <= sess->vtcm_size;
    }

    എങ്കിൽ (dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        തിരികെ അസത്യം;
    }

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q2_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q4_1:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_IQ4_NL:
        അവസ്ഥ GGML_TYPE_MXFP4:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q2_K: {
            എങ്കിൽ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                തിരികെ അസത്യം;
            }

            സ്ഥിരം int64_t qk = ggml_blck_size(src0->type);
            എങ്കിൽ (src0->ne[0] % qk) {
                തിരികെ അസത്യം;
            }

            എങ്കിൽ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                തിരികെ അസത്യം;
            }
            നിർത്തുക;
        }

        അവസ്ഥ GGML_TYPE_F16:
            എങ്കിൽ (src0->nb[1] < src0->nb[0]) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                തിരികെ അസത്യം;
            }
            നിർത്തുക;

        അവസ്ഥ GGML_TYPE_F32:
            എങ്കിൽ (src1->type != GGML_TYPE_F32) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (src0->nb[1] < src0->nb[0]) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                തിരികെ അസത്യം;
            }
            നിർത്തുക;

        അവസ്ഥ GGML_TYPE_BF16:
            എങ്കിൽ (src0->nb[1] < src0->nb[0]) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                തിരികെ അസത്യം;
            }
            നിർത്തുക;

        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }

    ഘടന htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    എങ്കിൽ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_mul_mat_id(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * src2 = op->src[2];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        തിരികെ അസത്യം;
    }

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q2_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q4_1:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_IQ4_NL:
        അവസ്ഥ GGML_TYPE_MXFP4:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q2_K:
            എങ്കിൽ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                തിരികെ അസത്യം;
            }

            എങ്കിൽ (src0->ne[0] % ggml_blck_size(src0->type) != 0) {
                തിരികെ അസത്യം;
            }
            നിർത്തുക;

        അവസ്ഥ GGML_TYPE_F16:
        അവസ്ഥ GGML_TYPE_F32:
        അവസ്ഥ GGML_TYPE_BF16:
            എങ്കിൽ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                തിരികെ അസത്യം;
            }
            എങ്കിൽ (src0->ne[0] % 32 != 0) {
                തിരികെ അസത്യം;
            }
            നിർത്തുക;

        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }

    ഘടന htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    എങ്കിൽ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_ദ്വിമാന(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src0->type == GGML_TYPE_F32) {
        എങ്കിൽ (src1->type != GGML_TYPE_F32) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (dst->type != GGML_TYPE_F32) {
            തിരികെ അസത്യം;
        }
    }
    അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_F16) {
        എങ്കിൽ (src1->type != GGML_TYPE_F16) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (dst->type != GGML_TYPE_F16) {
            തിരികെ അസത്യം;
        }
    }
    അല്ലെങ്കിൽ {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ggml_are_same_shape(src0, dst)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        തിരികെ അസത്യം;
    }

    ഘടന htp_ദ്വിമാന_kernel_params kparams;
    തിരികെ ggml_hexagon_precompute_ദ്വിമാന_params(sess, op_remap_to_htp(op), src0, src1, dst, &kparams);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_add_id(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * src2 = op->src[2];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (!src2) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ggml_are_same_shape(src0, dst)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src0->nb[0] != വലുപ്പം(ദശാംശം) || src1->nb[0] != വലുപ്പം(ദശാംശം) || dst->nb[0] != വലുപ്പം(ദശാംശം) || src2->nb[0] != വലുപ്പം(int32_t)) {
        തിരികെ അസത്യം;
    }

    // REVISIT: add support for non-contiguous tensors
    എങ്കിൽ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src2->nb[1] < src2->ne[0] * വലുപ്പം(int32_t)) {
        തിരികെ അസത്യം;
    }

    ഘടന htp_ദ്വിമാന_kernel_params kparams;
    തിരികെ ggml_hexagon_precompute_ദ്വിമാന_params(sess, HTP_OP_ADD_ID, src0, src1, dst, &kparams, src2);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_unary(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (dst->type != src0->type) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ggml_is_contiguous_rows(src0)) {
        തിരികെ അസത്യം;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    എങ്കിൽ (src0->type == GGML_TYPE_F16) {
        തിരഞ്ഞെടുപ്പ് (op->op) {
            അവസ്ഥ GGML_OP_NORM:
            അവസ്ഥ GGML_OP_RMS_NORM:
            അവസ്ഥ GGML_OP_L2_NORM:
            അവസ്ഥ GGML_OP_SCALE:
            അവസ്ഥ GGML_OP_CLAMP:
            അവസ്ഥ GGML_OP_SQR:
            അവസ്ഥ GGML_OP_SQRT:
            അവസ്ഥ GGML_OP_LOG:
                നിർത്തുക;
            അവസ്ഥ GGML_OP_UNARY:
                എങ്കിൽ (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_STEP) {
                    തിരികെ അസത്യം;
                }
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                തിരികെ അസത്യം;
        }
    }

    എങ്കിൽ (!ggml_are_same_shape(src0, dst)) {
        തിരികെ അസത്യം;
    }

    // dst must be contiguous; src0 may be non-contiguous
    എങ്കിൽ (!ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_sum(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src0->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_sum_rows(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src0->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    // TODO: add support for non-contigiuos tensors
    എങ്കിൽ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_argmax(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src0->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (dst->type != GGML_TYPE_I32) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_pool_2d(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം int32_t * params = op->op_params;

    എങ്കിൽ (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ggml_hexagon_pool_shape_fits(src0, op)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int32_t kernel_x = params[1];
    സ്ഥിരം int32_t kernel_y = params[2];
    സ്ഥിരം int32_t stride_x = params[3];
    സ്ഥിരം int32_t stride_y = params[4];
    സ്ഥിരം int32_t pad_x    = params[5];
    സ്ഥിരം int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    എങ്കിൽ (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    സ്ഥിരം uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    ഘടന htp_pool_vtcm_layout layout;
    തിരികെ htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_threads, sess->vtcm_size);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_pool_1d(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം int32_t * params = op->op_params;

    എങ്കിൽ (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ggml_hexagon_pool_shape_fits(src0, op)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int32_t kernel = params[1];
    സ്ഥിരം int32_t stride = params[2];
    സ്ഥിരം int32_t pad    = params[3];
    എങ്കിൽ (kernel <= 0 || stride <= 0 || pad < 0) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    സ്ഥിരം uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    ഘടന htp_pool_vtcm_layout layout;
    തിരികെ htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_threads, sess->vtcm_size);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_activations(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (dst->type != src0->type) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ggml_is_contiguous_1(src0)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src1) {
        എങ്കിൽ (src1->type != src0->type) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (!ggml_are_same_shape(src0, src1)) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (!ggml_is_contiguous_1(src1)) {
            തിരികെ അസത്യം;
        }
    }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_softmax(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * src2 = op->src[2];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src2) {
        തിരികെ അസത്യം;  // FIXME: add support for sinks
    }

    എങ്കിൽ (src0->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src0->ne[2] > 512) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src1) {
        എങ്കിൽ (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (src0->ne[0] != src1->ne[0]) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (src1->ne[1] < src0->ne[1]) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (src0->ne[2] % src1->ne[2] != 0) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (src0->ne[3] % src1->ne[3] != 0) {
            തിരികെ അസത്യം;
        }
    }

    എങ്കിൽ (src1) {
        എങ്കിൽ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            തിരികെ അസത്യം;
        }
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            തിരികെ അസത്യം;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    സ്ഥിരം int64_t ne0 = src0->ne[0];
    എങ്കിൽ (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        തിരികെ അസത്യം;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    എങ്കിൽ (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    സ്ഥിരം uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ഘടന htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != nullptr, src1 && src1->type == GGML_TYPE_F16, n_threads);
    എങ്കിൽ (layout.total_bytes > sess->vtcm_size) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_set_rows(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0]; //മൂല്യങ്ങൾ
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1]; // indices
    സ്ഥിരം ഘടന ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    എങ്കിൽ (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src0->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_get_rows(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0]; //മൂല്യങ്ങൾ
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1]; // indices
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(dst)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src0->type == dst->type) {
        എങ്കിൽ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            തിരികെ അസത്യം;
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (src0->type == GGML_TYPE_I32) {
        തിരികെ അസത്യം;
    } അല്ലെങ്കിൽ എങ്കിൽ (dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    എങ്കിൽ (ggml_is_empty(op)) {
        തിരികെ സത്യം;
    }

    ഘടന htp_get_rows_kernel_params kparams;
    ggml_hexagon_precompute_get_rows_params(sess, src0, src1, dst, &kparams);
    എങ്കിൽ (kparams.n_threads == 0 || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_argsort(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0]; //മൂല്യങ്ങൾ
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;         // indices

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    സ്ഥിരം uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    സ്ഥിരം size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    ഘടന htp_sort_vtcm_layout layout;
    എങ്കിൽ (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, അസത്യം)) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_മുകൾ_k(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0]; //മൂല്യങ്ങൾ
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;         // indices

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    സ്ഥിരം uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    സ്ഥിരം size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    ഘടന htp_sort_vtcm_layout layout;
    എങ്കിൽ (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, സത്യം)) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_rope(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * src2 = op->src[2];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (!ggml_are_same_shape(src0, dst)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src0->ne[0] <= 0) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    എങ്കിൽ (src0_nrows == 0) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int32_t * op_params = &op->op_params[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_dims = op_params[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ mode   = op_params[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    എങ്കിൽ (n_dims < 0 || n_dims % 2 != 0) {
        തിരികെ അസത്യം;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    എങ്കിൽ (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        തിരികെ അസത്യം;
    }

    ദശാംശം freq_base;
    memcpy(&freq_base, op_params + 5, വലുപ്പം(ദശാംശം));
    എങ്കിൽ (freq_base < 0.0f) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം തർക്കം is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    എങ്കിൽ (mode == GGML_ROPE_TYPE_VISION) {
        എങ്കിൽ (n_dims != (പൂർണ്ണസംഖ്യ) (src0->ne[0] / 2) || n_offs != 0) {
            തിരികെ അസത്യം;
        }
    }

    എങ്കിൽ (is_mrope) {
        സ്ഥിരം int32_t * sections = op_params + 11;
        എങ്കിൽ (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            തിരികെ അസത്യം;
        }
    }

    സ്ഥിരം int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    എങ്കിൽ (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src2) {
        എങ്കിൽ (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (src2->ne[0] < (n_dims / 2)) {
            തിരികെ അസത്യം;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    എങ്കിൽ (src0->nb[0] != വലുപ്പം(ദശാംശം) || dst->nb[0] != വലുപ്പം(ദശാംശം)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src0->nb[1] < src0->ne[0] * വലുപ്പം(ദശാംശം) || dst->nb[1] < dst->ne[0] * വലുപ്പം(ദശാംശം)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    സ്ഥിരം uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ഘടന htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    എങ്കിൽ (layout.total_bytes > sess->vtcm_size) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_ssm_conv(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    // Only support FP32 for now
    എങ്കിൽ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    // Check IO tensor shapes and dims
    എങ്കിൽ (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        തിരികെ അസത്യം; // src0 should be effectively 3D
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ d_conv  = src1->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ d_inner = src0->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_t = dst->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_s = dst->ne[2];

    എങ്കിൽ (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src0->nb[0] != വലുപ്പം(ദശാംശം) || src1->nb[0] != വലുപ്പം(ദശാംശം) || dst->nb[0] != വലുപ്പം(ദശാംശം)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (src0->nb[1] != src0->ne[0] * വലുപ്പം(ദശാംശം) || src1->nb[1] != src1->ne[0] * വലുപ്പം(ദശാംശം)) {
        തിരികെ അസത്യം;
    }

    ഘടന htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    എങ്കിൽ ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം im2col_3d_calc_output_size(uint32_t നിവേശം, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    എങ്കിൽ (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int64_t numerator = (int64_t) നിവേശം + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    എങ്കിൽ (numerator < 0) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint64_t size = (uint64_t) (numerator / stride) + 1;
    എങ്കിൽ (size == 0 || size > UINT32_MAX) {
        തിരികെ അസത്യം;
    }

    *output = (uint32_t) size;
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_precompute_im2col_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (!src0 || !src1 || !dst) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src1->nb[0] != വലുപ്പം(ദശാംശം) || !ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int32_t * op_params = (സ്ഥിരം int32_t *) op->op_params;
    സ്ഥിരം int32_t s0 = op_params[0];
    സ്ഥിരം int32_t s1 = op_params[1];
    സ്ഥിരം int32_t p0 = op_params[2];
    സ്ഥിരം int32_t p1 = op_params[3];
    സ്ഥിരം int32_t d0 = op_params[4];
    സ്ഥിരം int32_t d1 = op_params[5];
    സ്ഥിരം തർക്കം is_2D = (op_params[6] == 1);

    എങ്കിൽ (s0 <= 0 || d0 <= 0 || p0 < 0) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    സ്ഥിരം uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    സ്ഥിരം uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    സ്ഥിരം uint32_t IW = (uint32_t) src1->ne[0];
    സ്ഥിരം uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    സ്ഥിരം uint32_t KW = (uint32_t) src0->ne[0];
    സ്ഥിരം uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    സ്ഥിരം uint32_t OW = (uint32_t) dst->ne[1];

    എങ്കിൽ (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    സ്ഥിരം uint64_t total_rows_64   = (uint64_t) N * OH;
    എങ്കിൽ (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX) {
        തിരികെ അസത്യം;
    }
    സ്ഥിരം uint32_t patch_stride = (uint32_t) patch_stride_64;
    സ്ഥിരം uint32_t total_rows   = (uint32_t) total_rows_64;

    എങ്കിൽ (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (is_2D && dst->ne[2] != OH) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? വലുപ്പം(ggml_fp16_t) : വലുപ്പം(ദശാംശം);

    തർക്കം is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    എങ്കിൽ (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    സ്ഥിരം uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    സ്ഥിരം uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    സ്ഥിരം uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    എങ്കിൽ (is_patchembed) {
        സ്ഥിരം uint64_t src_row_bytes_raw = (uint64_t) IC * KH * IW * വലുപ്പം(ദശാംശം);
        സ്ഥിരം uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        എങ്കിൽ (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            സ്ഥിരം uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            സ്ഥിരം uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            ഘടന htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            എങ്കിൽ (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                തിരികെ സത്യം;
            }
        }
    }

    വേണ്ടി (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        സ്ഥിരം uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        എങ്കിൽ (Wb > UINT32_MAX) {
            എങ്കിൽ (owb == 1) നിർത്തുക;
            തുടരുക;
        }

        സ്ഥിരം uint64_t src_row_bytes_raw = (uint64_t) IC * KH * Wb * വലുപ്പം(ദശാംശം);
        സ്ഥിരം uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        എങ്കിൽ (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            എങ്കിൽ (owb == 1) നിർത്തുക;
            തുടരുക;
        }

        സ്ഥിരം uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        സ്ഥിരം uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        ഘടന htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        എങ്കിൽ (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            തിരികെ സത്യം;
        }
        എങ്കിൽ (owb == 1) നിർത്തുക;
    }

    തിരികെ അസത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_precompute_im2col_3d_params(
    സ്ഥിരം ഘടന ggml_hexagon_session * sess,
    സ്ഥിരം ഘടന ggml_tensor * op,
    ഘടന htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, വലുപ്പം(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (!src0 || !src1 || !dst) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src1->nb[0] != വലുപ്പം(ദശാംശം) || !ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int32_t * params = (സ്ഥിരം int32_t *) op->op_params;
    സ്ഥിരം int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    സ്ഥിരം int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    സ്ഥിരം int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    സ്ഥിരം int32_t ic = params[9];

    എങ്കിൽ (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src1->ne[3] % ic != 0) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t IC = (uint32_t) ic;
    സ്ഥിരം uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    എങ്കിൽ (N == 0) {
        തിരികെ അസത്യം;
    }

    uint32_t OW, OH, OD;
    എങ്കിൽ (!im2col_3d_calc_output_size(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_size(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_size(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t KD = (uint32_t) src0->ne[2];
    സ്ഥിരം uint32_t KH = (uint32_t) src0->ne[1];
    സ്ഥിരം uint32_t KW = (uint32_t) src0->ne[0];
    സ്ഥിരം uint32_t ID = (uint32_t) src1->ne[2];
    സ്ഥിരം uint32_t IH = (uint32_t) src1->ne[1];
    സ്ഥിരം uint32_t IW = (uint32_t) src1->ne[0];

    സ്ഥിരം uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    സ്ഥിരം uint64_t total_rows_64   = (uint64_t) N * OD * OH;
    സ്ഥിരം uint64_t dst_ne3_64      = (uint64_t) OD * N;

    എങ്കിൽ (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        തിരികെ അസത്യം;
    }
    സ്ഥിരം uint32_t patch_stride = (uint32_t) patch_stride_64;
    സ്ഥിരം uint32_t total_rows   = (uint32_t) total_rows_64;
    സ്ഥിരം uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    എങ്കിൽ (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? വലുപ്പം(ggml_fp16_t) : വലുപ്പം(ദശാംശം);

    സ്ഥിരം തർക്കം is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    സ്ഥിരം uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    സ്ഥിരം uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    സ്ഥിരം uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    എങ്കിൽ (is_patchembed) {
        സ്ഥിരം uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * IW * വലുപ്പം(ദശാംശം);
        സ്ഥിരം uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        എങ്കിൽ (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            സ്ഥിരം uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            സ്ഥിരം uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            ഘടന htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            എങ്കിൽ (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                തിരികെ സത്യം;
            }
        }
    }

    വേണ്ടി (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        സ്ഥിരം uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        എങ്കിൽ (Wb > UINT32_MAX) {
            എങ്കിൽ (owb == 1) നിർത്തുക;
            തുടരുക;
        }

        സ്ഥിരം uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * Wb * വലുപ്പം(ദശാംശം);
        സ്ഥിരം uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        എങ്കിൽ (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            എങ്കിൽ (owb == 1) നിർത്തുക;
            തുടരുക;
        }

        സ്ഥിരം uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        സ്ഥിരം uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        ഘടന htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        എങ്കിൽ (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            തിരികെ സത്യം;
        }
        എങ്കിൽ (owb == 1) നിർത്തുക;
    }

    തിരികെ അസത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_im2col(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    ഘടന htp_im2col_kernel_params kparams;
    തിരികെ ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_im2col_3d(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    ഘടന htp_im2col_kernel_params kparams;
    തിരികെ ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_pad(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int32_t lp0 = ((സ്ഥിരം int32_t *) op->op_params)[0];
    സ്ഥിരം int32_t rp0 = ((സ്ഥിരം int32_t *) op->op_params)[1];
    സ്ഥിരം int32_t circular = ((സ്ഥിരം int32_t *) op->op_params)[8];

    എങ്കിൽ (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_cumsum(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_diag(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    // diag only supports F32 currently
    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    // Input must have ne[1] == 1 (vector input)
    എങ്കിൽ (src0->ne[1] != 1) {
        തിരികെ അസത്യം;
    }

    // Output must be square in first two dimensions
    എങ്കിൽ (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_solve_tri(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0]; // A
    സ്ഥിരം ഘടന ggml_tensor * src1 = op->src[1]; // B
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;         // X

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src0->ne[0] != src0->ne[1]) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src0->ne[1] != src1->ne[1]) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_tri(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {

    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src0->type != GGML_TYPE_F32) { തിരികെ അസത്യം; }
    എങ്കിൽ (dst->type  != GGML_TYPE_F32) { തിരികെ അസത്യം; }
    എങ്കിൽ (!ggml_are_same_shape(src0, dst)) { തിരികെ അസത്യം; }
    എങ്കിൽ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { തിരികെ അസത്യം; }

    തിരികെ സത്യം;

    GGML_UNUSED(sess);
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    തിരികെ sess->c_name();
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

സ്ഥിതം htp_op_code op_remap_to_htp(സ്ഥിരം ggml_tensor * t) {
    തിരഞ്ഞെടുപ്പ് (t->op) {
        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:  തിരികെ HTP_OP_FLASH_ATTN_EXT;
        അവസ്ഥ GGML_OP_MUL_MAT:
            എങ്കിൽ (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                തിരികെ HTP_OP_HADAMARD;
            }
            തിരികെ HTP_OP_MUL_MAT;
        അവസ്ഥ GGML_OP_MUL_MAT_ID:      തിരികെ HTP_OP_MUL_MAT_ID;
        അവസ്ഥ GGML_OP_MUL:             തിരികെ HTP_OP_MUL;
        അവസ്ഥ GGML_OP_ADD:             തിരികെ HTP_OP_ADD;
        അവസ്ഥ GGML_OP_ADD_ID:          തിരികെ HTP_OP_ADD_ID;
        അവസ്ഥ GGML_OP_SUB:             തിരികെ HTP_OP_SUB;
        അവസ്ഥ GGML_OP_DIV:             തിരികെ HTP_OP_DIV;
        അവസ്ഥ GGML_OP_DUP:
        അവസ്ഥ GGML_OP_CPY:             തിരികെ HTP_OP_CPY;
        അവസ്ഥ GGML_OP_CONT:            തിരികെ HTP_OP_CPY;
        അവസ്ഥ GGML_OP_GET_ROWS:        തിരികെ HTP_OP_GET_ROWS;
        അവസ്ഥ GGML_OP_SET_ROWS:        തിരികെ HTP_OP_SET_ROWS;
        അവസ്ഥ GGML_OP_SUM:             തിരികെ HTP_OP_SUM;
        അവസ്ഥ GGML_OP_SUM_ROWS:        തിരികെ HTP_OP_SUM_ROWS;
        അവസ്ഥ GGML_OP_POOL_2D:         തിരികെ HTP_OP_POOL_2D;
        അവസ്ഥ GGML_OP_POOL_1D:         തിരികെ HTP_OP_POOL_1D;
        അവസ്ഥ GGML_OP_ARGSORT:         തിരികെ HTP_OP_ARGSORT;
        അവസ്ഥ GGML_OP_മുകൾ_K:           തിരികെ HTP_OP_മുകൾ_K;
        അവസ്ഥ GGML_OP_ARGMAX:          തിരികെ HTP_OP_ARGMAX;
        അവസ്ഥ GGML_OP_NORM:            തിരികെ HTP_OP_NORM;
        അവസ്ഥ GGML_OP_L2_NORM:         തിരികെ HTP_OP_L2_NORM;
        അവസ്ഥ GGML_OP_RMS_NORM:        തിരികെ HTP_OP_RMS_NORM;
        അവസ്ഥ GGML_OP_CONCAT:          തിരികെ HTP_OP_CONCAT;
        അവസ്ഥ GGML_OP_SCALE:           തിരികെ HTP_OP_SCALE;
        അവസ്ഥ GGML_OP_CLAMP:           തിരികെ HTP_OP_CLAMP;
        അവസ്ഥ GGML_OP_LEAKY_RELU:      തിരികെ HTP_OP_LEAKY_RELU;
        അവസ്ഥ GGML_OP_SQR:             തിരികെ HTP_OP_SQR;
        അവസ്ഥ GGML_OP_SQRT:            തിരികെ HTP_OP_SQRT;
        അവസ്ഥ GGML_OP_LOG:             തിരികെ HTP_OP_UNARY_LOG;
        അവസ്ഥ GGML_OP_SOFT_MAX:        തിരികെ HTP_OP_SOFTMAX;
        അവസ്ഥ GGML_OP_SSM_CONV:        തിരികെ HTP_OP_SSM_CONV;
        അവസ്ഥ GGML_OP_GATED_DELTA_NET: തിരികെ HTP_OP_GATED_DELTA_NET;
        അവസ്ഥ GGML_OP_ROPE:            തിരികെ HTP_OP_ROPE;
        അവസ്ഥ GGML_OP_REPEAT:          തിരികെ HTP_OP_REPEAT;
        അവസ്ഥ GGML_OP_CUMSUM:          തിരികെ HTP_OP_CUMSUM;
        അവസ്ഥ GGML_OP_FILL:            തിരികെ HTP_OP_FILL;
        അവസ്ഥ GGML_OP_DIAG:            തിരികെ HTP_OP_DIAG;
        അവസ്ഥ GGML_OP_SOLVE_TRI:       തിരികെ HTP_OP_SOLVE_TRI;
        അവസ്ഥ GGML_OP_TRI:             തിരികെ HTP_OP_TRI;
        അവസ്ഥ GGML_OP_PAD:             തിരികെ HTP_OP_PAD;
        അവസ്ഥ GGML_OP_IM2COL:          തിരികെ HTP_OP_IM2COL;
        അവസ്ഥ GGML_OP_IM2COL_3D:       തിരികെ HTP_OP_IM2COL_3D;
        അവസ്ഥ GGML_OP_ROLL:            തിരികെ HTP_OP_ROLL;

        അവസ്ഥ GGML_OP_UNARY:
            തിരഞ്ഞെടുപ്പ് (ggml_get_unary_op(t)) {
                അവസ്ഥ GGML_UNARY_OP_SILU:       തിരികെ HTP_OP_UNARY_SILU;
                അവസ്ഥ GGML_UNARY_OP_GELU:       തിരികെ HTP_OP_UNARY_GELU;
                അവസ്ഥ GGML_UNARY_OP_GELU_QUICK: തിരികെ HTP_OP_UNARY_GELU_QUICK;
                അവസ്ഥ GGML_UNARY_OP_GELU_ERF:   തിരികെ HTP_OP_UNARY_GELU_ERF;
                അവസ്ഥ GGML_UNARY_OP_SIGMOID:    തിരികെ HTP_OP_UNARY_SIGMOID;
                അവസ്ഥ GGML_UNARY_OP_NEG:        തിരികെ HTP_OP_UNARY_NEG;
                അവസ്ഥ GGML_UNARY_OP_EXP:        തിരികെ HTP_OP_UNARY_EXP;
                അവസ്ഥ GGML_UNARY_OP_SOFTPLUS:   തിരികെ HTP_OP_UNARY_SOFTPLUS;
                അവസ്ഥ GGML_UNARY_OP_TANH:       തിരികെ HTP_OP_UNARY_TANH;
                അവസ്ഥ GGML_UNARY_OP_ABS:        തിരികെ HTP_OP_UNARY_ABS;
                അവസ്ഥ GGML_UNARY_OP_RELU:       തിരികെ HTP_OP_UNARY_RELU;
                അവസ്ഥ GGML_UNARY_OP_STEP:       തിരികെ HTP_OP_UNARY_STEP;
            സ്ഥിരസ്ഥിതി:
                നിർത്തുക;
            }
            നിർത്തുക;

        അവസ്ഥ GGML_OP_GLU:
            തിരഞ്ഞെടുപ്പ് (ggml_get_glu_op(t)) {
                അവസ്ഥ GGML_GLU_OP_SWIGLU:       തിരികെ HTP_OP_GLU_SWIGLU;
                അവസ്ഥ GGML_GLU_OP_SWIGLU_OAI:   തിരികെ HTP_OP_GLU_SWIGLU_OAI;
                അവസ്ഥ GGML_GLU_OP_SWIGLU_CLAMP: തിരികെ HTP_OP_GLU_SWIGLU_CLAMP;
                അവസ്ഥ GGML_GLU_OP_GEGLU:        തിരികെ HTP_OP_GLU_GEGLU;
                അവസ്ഥ GGML_GLU_OP_GEGLU_QUICK:  തിരികെ HTP_OP_GLU_GEGLU_QUICK;
                അവസ്ഥ GGML_GLU_OP_GEGLU_ERF:    തിരികെ HTP_OP_GLU_GEGLU_ERF;
                സ്ഥിരസ്ഥിതി: നിർത്തുക;
            }
            നിർത്തുക;

        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    തിരികെ HTP_OP_INVALID;
}

സ്ഥിതം വരിയിൽ തർക്കം op_is_compute(ggml_tensor *കണ്ണി)
{
    തിരികെ !ggml_op_is_empty(കണ്ണി->op) && !ggml_is_empty(കണ്ണി) && (കണ്ണി->flags & GGML_TENSOR_FLAG_COMPUTE);
}

സ്ഥിതം തർക്കം mm_is_hmx_eligible(സ്ഥിരം ggml_tensor * t) {
    എങ്കിൽ (opt_nhmx == 0) { തിരികെ അസത്യം; }

    സ്ഥിരം ggml_tensor * src0 = t->src[0];
    സ്ഥിരം ggml_tensor * src1 = t->src[1];

    എങ്കിൽ (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        സ്ഥിരം ggml_tensor src1_c = ggml_hexagon_tensor_collapse_rows(src1);
        തിരികെ ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, അസത്യം, അസത്യം);
    }

    സ്ഥിരം തർക്കം is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    സ്ഥിരം തർക്കം is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    തിരികെ ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

സ്ഥിതം തർക്കം is_supported_mul_mat_nx_kernel(സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ഘടന htp_mm_kernel_params * kparams) {
    എങ്കിൽ (kparams->n_hmx) {
        തിരികെ kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    എങ്കിൽ (!ggml_hexagon_tensor_is_tiled(src0)) {
        തിരികെ അസത്യം;
    }

    തിരികെ kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
}

സ്ഥിതം തർക്കം is_supported_mul_mat_id_nx_kernel(സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ഘടന htp_mm_kernel_params * kparams) {
    എങ്കിൽ (kparams->n_hmx) {
        തിരികെ kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    എങ്കിൽ (!ggml_hexagon_tensor_is_tiled(src0)) {
        തിരികെ അസത്യം;
    }

    തിരികെ kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK;
}

സ്ഥിതം തർക്കം is_mergeable_mul_mat(സ്ഥിരം ggml_tensor * t) {
    എങ്കിൽ (t->op != GGML_OP_MUL_MAT) തിരികെ അസത്യം;
    എങ്കിൽ (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) തിരികെ അസത്യം;

    സ്ഥിരം ggml_tensor * src0 = t->src[0];
    സ്ഥിരം ggml_tensor * src1 = t->src[1];
    എങ്കിൽ (src0->ne[2] != 1 || src0->ne[3] != 1) തിരികെ അസത്യം;

    എങ്കിൽ (mm_is_hmx_eligible(t)) {
        തിരികെ htp_is_hmx_weight_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    എങ്കിൽ (src1->type != GGML_TYPE_F32) തിരികെ അസത്യം;

    തിരികെ ggml_hexagon_tensor_is_tiled(src0);
}

സ്ഥിതം തർക്കം is_mergeable_mul_mat_pair(സ്ഥിരം ggml_tensor * n1, സ്ഥിരം ggml_tensor * n2) {
    എങ്കിൽ (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (n1->src[1] != n2->src[1]) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (n1->src[0]->type != n2->src[0]->type) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        തിരികെ അസത്യം;
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം is_mergeable_mul_mat_id(സ്ഥിരം ggml_tensor * t) {
    എങ്കിൽ (t->op != GGML_OP_MUL_MAT_ID) തിരികെ അസത്യം;

    സ്ഥിരം ggml_tensor * src0 = t->src[0];
    എങ്കിൽ (mm_is_hmx_eligible(t)) {
        തിരികെ htp_is_hmx_weight_type(src0->type);
    }

    തിരികെ ggml_hexagon_tensor_is_tiled(src0);
}

സ്ഥിതം തർക്കം is_mergeable_mul_mat_id_pair(സ്ഥിരം ggml_tensor * n1, സ്ഥിരം ggml_tensor * n2) {
    എങ്കിൽ (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (n1->src[1] != n2->src[1]) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (n1->src[2] != n2->src[2]) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (n1->src[0]->type != n2->src[0]->type) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        തിരികെ അസത്യം;
    }
    തിരികെ സത്യം;
}

സ്ഥിതം ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    എങ്കിൽ (sess->last_error > HTP_STATUS_OK) {
        തിരികെ GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    സ്ഥിരം std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    തർക്കം cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
    എങ്കിൽ (cache_hit) {
        nodes_ptr = &sess->cached_nodes;
    } അല്ലെങ്കിൽ {
        // Tag fusable tensors in graph
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; i++) {
            auto * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            എങ്കിൽ (!extra) തുടരുക;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            എങ്കിൽ (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } അല്ലെങ്കിൽ എങ്കിൽ ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                എങ്കിൽ ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_കണ്ണി_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            എങ്കിൽ (!op_is_compute(n)) {
                തുടരുക;
            }

            htp_opnode കണ്ണി(HTP_OP_INVALID, n);
            കണ്ണി.opcode = op_remap_to_htp(n);
            എങ്കിൽ (കണ്ണി.opcode == HTP_OP_MUL_MAT || കണ്ണി.opcode == HTP_OP_MUL_MAT_ID || കണ്ണി.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = കണ്ണി.കണ്ണി->src[0];
                എങ്കിൽ (ggml_hexagon_needs_repack(src0)) {
                    എങ്കിൽ (opt_verbose) {
                        അക്ഷരം dims_buf[64];
                        അക്ഷരം strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, വലുപ്പം(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, വലുപ്പം(strides_buf), src0);
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

            എങ്കിൽ (കണ്ണി.opcode == HTP_OP_MUL_MAT || കണ്ണി.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    കണ്ണി.കണ്ണി->src[0], കണ്ണി.കണ്ണി->src[1], കണ്ണി.കണ്ണി,
                    (ഘടന htp_mm_kernel_params *)കണ്ണി.kernel_params
                );
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_MUL || കണ്ണി.opcode == HTP_OP_ADD || കണ്ണി.opcode == HTP_OP_ADD_ID || കണ്ണി.opcode == HTP_OP_SUB || കണ്ണി.opcode == HTP_OP_DIV) {
                സ്ഥിരം ggml_tensor * src1 = കണ്ണി.കണ്ണി->src[1];
                സ്ഥിരം ggml_tensor * src2 = (കണ്ണി.opcode == HTP_OP_ADD_ID) ? കണ്ണി.കണ്ണി->src[2] : nullptr;
                GGML_ASSERT(ggml_hexagon_precompute_ദ്വിമാന_params(sess,
                    കണ്ണി.opcode, കണ്ണി.കണ്ണി->src[0], src1, കണ്ണി.കണ്ണി,
                    (ഘടന htp_ദ്വിമാന_kernel_params *) കണ്ണി.kernel_params, src2));
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    കണ്ണി.കണ്ണി,
                    (ഘടന htp_fa_kernel_params *)കണ്ണി.kernel_params
                );
            } അല്ലെങ്കിൽ എങ്കിൽ (htp_op_is_unary(കണ്ണി.opcode)) {
                auto inputs = കണ്ണി.get_inputs();
                സ്ഥിരം ഘടന ggml_tensor * src0 = inputs[0];
                സ്ഥിരം ഘടന ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    കണ്ണി.opcode, src0, src1, കണ്ണി.dst(),
                    (ഘടന htp_unary_kernel_params *)കണ്ണി.kernel_params
                );
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_GET_ROWS) {
                ggml_hexagon_precompute_get_rows_params(sess,
                    കണ്ണി.കണ്ണി->src[0], കണ്ണി.കണ്ണി->src[1], കണ്ണി.dst(),
                    (ഘടന htp_get_rows_kernel_params *)കണ്ണി.kernel_params
                );
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_SET_ROWS) {
                ggml_hexagon_precompute_set_rows_params(sess,
                    കണ്ണി.കണ്ണി->src[0], കണ്ണി.കണ്ണി->src[1], കണ്ണി.dst(),
                    (ഘടന htp_set_rows_kernel_params *)കണ്ണി.kernel_params
                );
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    കണ്ണി.കണ്ണി,
                    (ഘടന htp_rope_kernel_params *)കണ്ണി.kernel_params
                );
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    കണ്ണി.കണ്ണി->src[0], കണ്ണി.കണ്ണി->src[1], കണ്ണി.dst(),
                    (ഘടന htp_ssm_conv_kernel_params *)കണ്ണി.kernel_params
                );
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    കണ്ണി.കണ്ണി,
                    (ഘടന htp_softmax_kernel_params *)കണ്ണി.kernel_params
                );
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    കണ്ണി.കണ്ണി,
                    (ഘടന htp_gdn_kernel_params *)കണ്ണി.kernel_params
                );
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_ARGSORT || കണ്ണി.opcode == HTP_OP_മുകൾ_K) {
                ggml_hexagon_precompute_sort_params(sess,
                    കണ്ണി.കണ്ണി,
                    കണ്ണി.opcode == HTP_OP_മുകൾ_K,
                    (ഘടന htp_sort_kernel_params *) കണ്ണി.kernel_params
                );
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_POOL_2D || കണ്ണി.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, കണ്ണി.കണ്ണി->src[0], കണ്ണി.dst(),
                    (ഘടന htp_pool_2d_kernel_params *)കണ്ണി.kernel_params,
                    കണ്ണി.opcode == HTP_OP_POOL_1D);
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    കണ്ണി.കണ്ണി,
                    (ഘടന htp_concat_kernel_params *) കണ്ണി.kernel_params
                );
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_CPY || കണ്ണി.opcode == HTP_OP_CPY_FENCE) {
                സ്ഥിരം തർക്കം ok = ggml_hexagon_precompute_cpy_params(sess,
                    കണ്ണി.കണ്ണി,
                    (ഘടന htp_copy_kernel_params *) കണ്ണി.kernel_params
                );
                സ്ഥിരം auto * kparams = (സ്ഥിരം ഘടന htp_copy_kernel_params *) കണ്ണി.kernel_params;
                എങ്കിൽ (ok && കണ്ണി.opcode == HTP_OP_CPY && kparams->total_elems == 0) {
                    തുടരുക;
                }
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_IM2COL) {
                ggml_hexagon_precompute_im2col_params(sess,
                    കണ്ണി.കണ്ണി,
                    (ഘടന htp_im2col_kernel_params *) കണ്ണി.kernel_params
                );
            } അല്ലെങ്കിൽ എങ്കിൽ (കണ്ണി.opcode == HTP_OP_IM2COL_3D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    കണ്ണി.കണ്ണി,
                    (ഘടന htp_im2col_kernel_params *) കണ്ണി.kernel_params
                );
            }
            computed_nodes.push_back(std::move(കണ്ണി));
        }

        എങ്കിൽ (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } അല്ലെങ്കിൽ {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    വേണ്ടി (സ്ഥിരം auto & കണ്ണി : *nodes_ptr) {
        sess->enqueue_op(കണ്ണി);
    }

    എങ്കിൽ (sess->last_error > HTP_STATUS_OK) {
        തിരികെ GGML_STATUS_FAILED;
    }

    തിരികെ GGML_STATUS_SUCCESS;
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    എങ്കിൽ (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
}

എണ്ണൽ ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

ഘടന ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

ഘടന ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    ശൂന്യം reset() {
        ranges.clear();
    }

    ശൂന്യം add(സ്ഥിരം ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    തർക്കം check(സ്ഥിരം ggml_hexagon_mem_range & mr) സ്ഥിരം {
        വേണ്ടി (സ്ഥിരം auto & cmp : ranges) {
            എങ്കിൽ (mr.pb != cmp.pb) {
                തുടരുക;
            }
            എങ്കിൽ (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                തുടരുക;
            }
            എങ്കിൽ (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                തിരികെ അസത്യം;
            }
        }
        തിരികെ സത്യം;
    }
};

സ്ഥിതം ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(സ്ഥിരം ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    സ്ഥിരം ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    എങ്കിൽ (tensor->buffer) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } അല്ലെങ്കിൽ {
        mr = {
            /*.pb =*/ (uint64_t) base,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    തിരികെ mr;
}

സ്ഥിതം ശൂന്യം ggml_hexagon_mem_ranges_add_കണ്ണി(ggml_hexagon_mem_ranges & mrs, സ്ഥിരം htp_opnode & കണ്ണി) {
    എങ്കിൽ (കണ്ണി.is_empty()) തിരികെ;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
        എങ്കിൽ (കണ്ണി.കണ്ണി->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(കണ്ണി.കണ്ണി->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    വേണ്ടി (സ്ഥിരം auto * fused : കണ്ണി.fused) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
            എങ്കിൽ (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(കണ്ണി.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

സ്ഥിതം തർക്കം ggml_hexagon_mem_ranges_check_കണ്ണി(സ്ഥിരം ggml_hexagon_mem_ranges & mrs, സ്ഥിരം htp_opnode & കണ്ണി) {
    എങ്കിൽ (കണ്ണി.is_empty()) തിരികെ സത്യം;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
        എങ്കിൽ (കണ്ണി.കണ്ണി->src[i]) {
            എങ്കിൽ (!mrs.check(ggml_hexagon_mem_range_from_tensor(കണ്ണി.കണ്ണി->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                തിരികെ അസത്യം;
            }
        }
    }
    വേണ്ടി (സ്ഥിരം auto * fused : കണ്ണി.fused) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
            എങ്കിൽ (fused->src[i]) {
                എങ്കിൽ (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    തിരികെ അസത്യം;
                }
            }
        }
    }
    തിരികെ mrs.check(ggml_hexagon_mem_range_from_tensor(കണ്ണി.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

സ്ഥിതം std::vector<പൂർണ്ണസംഖ്യ> ggml_hexagon_graph_optimize_reorder(സ്ഥിരം std::vector<htp_opnode> & nodes) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ n = nodes.size();

    std::vector<പൂർണ്ണസംഖ്യ> res;
    res.reserve(n);

    std::vector<തർക്കം> used(n, അസത്യം);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    വേണ്ടി (പൂർണ്ണസംഖ്യ i0 = 0; i0 < n; i0++) {
        എങ്കിൽ (used[i0]) {
            തുടരുക;
        }

        സ്ഥിരം auto & node0 = nodes[i0];

        എങ്കിൽ (!node0.stackable()) {
            res.push_back(i0);
            used[i0] = സത്യം;
            തുടരുക;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr പൂർണ്ണസംഖ്യ N_FORWARD = 16;

        std::vector<പൂർണ്ണസംഖ്യ> അട്ടി;
        അട്ടി.push_back(i0);

        mrs.reset();

        വേണ്ടി (പൂർണ്ണസംഖ്യ i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            എങ്കിൽ (used[i1]) {
                തുടരുക;
            }

            സ്ഥിരം auto & node1 = nodes[i1];

            എങ്കിൽ (node1.stackable() && node1.same_നിവേശം(node0) && ggml_hexagon_mem_ranges_check_കണ്ണി(mrs, node1)) {
                അട്ടി.push_back(i1);
            } അല്ലെങ്കിൽ {
                ggml_hexagon_mem_ranges_add_കണ്ണി(mrs, node1);
            }
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ idx : അട്ടി) {
            res.push_back(idx);
            used[idx] = സത്യം;
        }
    }

    തിരികെ res;
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    സ്ഥിരം പൂർണ്ണസംഖ്യ n = gf->n_nodes;

    constexpr പൂർണ്ണസംഖ്യ MAX_FUSE = 16;

    എണ്ണൽ ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        htp_opnode കണ്ണി(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        എങ്കിൽ (കണ്ണി.op() == GGML_OP_ADD ||
            കണ്ണി.op() == GGML_OP_NORM ||
            കണ്ണി.op() == GGML_OP_RMS_NORM) {
            ops[0] = കണ്ണി.op();

            പൂർണ്ണസംഖ്യ f = i + 1;
            ആകുമ്പോൾ (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                എങ്കിൽ (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    നിർത്തുക;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            വേണ്ടി (; f > 1; f--) {
                എങ്കിൽ (ggml_can_fuse(gf, i, ops, f)) {
                    നിർത്തുക;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            വേണ്ടി (പൂർണ്ണസംഖ്യ k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                കണ്ണി.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(കണ്ണി));
    }

    സ്ഥിരം auto order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        പൂർണ്ണസംഖ്യ j = 0;
        വേണ്ടി (സ്ഥിരം auto i : order) {
            സ്ഥിരം auto & കണ്ണി = nodes[i];

            gf->nodes[j++] = കണ്ണി.കണ്ണി;

            വേണ്ടി (auto * fused : കണ്ണി.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

സ്ഥിതം uint64_t ggml_hexagon_session_താക്കോൽ(സ്ഥിരം ggml_hexagon_session * sess) {
    തിരികെ ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

സ്ഥിതം തർക്കം ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, സ്ഥിരം ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    auto sbuf_dst = sess_dst->mmap_tensor(dst);

    എങ്കിൽ (!sess_src->clone_buffer(sbuf_dst)) { തിരികെ അസത്യം; }

    സ്ഥിരം uint64_t src_താക്കോൽ = ggml_hexagon_session_താക്കോൽ(sess_src);
    auto & fence_slot = sess_dst->cpy_fence_slots[src_താക്കോൽ];
    എങ്കിൽ (!fence_slot) {
        fence_slot = (അസ്ഥിര uint32_t *) sess_dst->alloc_fence(1);
    }

    എങ്കിൽ (!sess_src->clone_buffer(sess_dst->fence_buf)) { തിരികെ അസത്യം; }

    എങ്കിൽ (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    സ്ഥിതം ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.buffer = &sess_dst->fence_buf->backend_buffer;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.data   = (ശൂന്യം *) fence_slot;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = വലുപ്പം(int32_t);
    fence_tensor.nb[1]  = വലുപ്പം(int32_t);
    fence_tensor.nb[2]  = വലുപ്പം(int32_t);
    fence_tensor.nb[3]  = വലുപ്പം(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ സത്യം);

    sess_dst->add_peer(sess_src);

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, സ്ഥിരം ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    auto sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    എങ്കിൽ (!sess_dst->clone_buffer(sbuf_src)) { തിരികെ അസത്യം; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, സ്ഥിരം ggml_tensor * src, ggml_tensor * dst) {
    എങ്കിൽ (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        തിരികെ അസത്യം;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    എങ്കിൽ (!dst->extra) {
        ggml_backend_buffer_init_tensor(dst->buffer, dst);
    }

    auto * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    സ്ഥിരം auto * src_extra = static_cast<സ്ഥിരം ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    എങ്കിൽ (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        തിരികെ സത്യം;
    }

    എങ്കിൽ (sess_src->phys_idx != sess_dst->phys_idx)
        തിരികെ ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    തിരികെ ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

സ്ഥിതം ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = പുതിയ ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_slot = (അസ്ഥിര uint32_t *) sess->alloc_fence(1);

    സ്ഥിതം ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.buffer = &sess->fence_buf->backend_buffer;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.data   = (ശൂന്യം *) hex_event->fence_slot;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = വലുപ്പം(int32_t);
    hex_event->fence_tensor.nb[1]  = വലുപ്പം(int32_t);
    hex_event->fence_tensor.nb[2]  = വലുപ്പം(int32_t);
    hex_event->fence_tensor.nb[3]  = വലുപ്പം(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (ശൂന്യം *)hex_event, (ശൂന്യം *)hex_event->fence_slot);

    തിരികെ പുതിയ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

സ്ഥിതം ശൂന്യം ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    എങ്കിൽ (hex_event->seq == 0) {
        തിരികെ;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (ശൂന്യം *)hex_event, hex_event->seq, (ശൂന്യം *)hex_event->fence_slot);

    auto * fence = reinterpret_cast<സ്ഥിരം അസ്ഥിര std::atomic<uint32_t> *>(hex_event->fence_slot);

    എങ്കിൽ ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    ആകുമ്പോൾ (സത്യം) {
        എങ്കിൽ ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            എങ്കിൽ (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            നിർത്തുക;
        }
        std::this_thread::yield();
    }
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (ശൂന്യം *)hex_event);
    hex_event->fence_sess->free_fence((ശൂന്യം *) hex_event->fence_slot, 1);
    delete hex_event;
    delete event;
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    എങ്കിൽ (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ അസത്യം);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (ശൂന്യം *)hex_event, hex_event->seq, (ശൂന്യം *)hex_event->fence_slot);
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    എങ്കിൽ (hex_event->seq == 0) {
        തിരികെ;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (ശൂന്യം *)hex_event, hex_event->seq, (ശൂന്യം *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    എങ്കിൽ (sess->phys_idx == hex_event->sess->phys_idx) {
        എങ്കിൽ (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        തിരികെ;
    }

    sess->clone_buffer(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ സത്യം);
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, ഘടന ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (പൂർണ്ണസംഖ്യ) tensor->buffer->usage : -1);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, സ്ഥിരം ഘടന ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (പൂർണ്ണസംഖ്യ) tensor->buffer->usage : -1);
    sess->flush_sync();
    എങ്കിൽ (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     ഘടന ggml_tensor * tensor,
                                                     സ്ഥിരം ശൂന്യം * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (പൂർണ്ണസംഖ്യ) tensor->buffer->usage : -1);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     സ്ഥിരം ഘടന ggml_tensor * tensor,
                                                     ശൂന്യം * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (പൂർണ്ണസംഖ്യ) tensor->buffer->usage : -1);
    sess->flush_sync();
    എങ്കിൽ (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

സ്ഥിതം ഘടന ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ ശൂന്യ,
    /* .graph_plan_free         = */ ശൂന്യ,
    /* .graph_plan_update       = */ ശൂന്യ,
    /* .graph_plan_compute      = */ ശൂന്യ,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_record,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

സ്ഥിതം ggml_guid_t ggml_backend_hexagon_guid() {
    സ്ഥിതം ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    തിരികെ &guid;
}

തർക്കം ggml_backend_is_hexagon(ggml_backend_t backend) {
    തിരികെ backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

സ്ഥിതം ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, സ്ഥിരം അക്ഷരം * params) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    തിരികെ പുതിയ ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    തിരികെ dev_ctx->c_name();

    GGML_UNUSED(dev);
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    തിരികെ "Hexagon";
    GGML_UNUSED(dev);
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * സ്വതന്ത്രം, size_t * total) {
    *സ്വതന്ത്രം  = opt_mbuf_total;
    *total = opt_mbuf_total;

    GGML_UNUSED(dev);
}

സ്ഥിതം എണ്ണൽ ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    തിരികെ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, ഘടന ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ സത്യം,
        /* .host_buffer           = */ അസത്യം,
        /* .buffer_from_host_ptr  = */ അസത്യം,
        /* .events                = */ സത്യം,
        /* .mmap_support          = */ അസത്യം,
    };
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    തിരികെ &dev_ctx->buffer_type;
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    എങ്കിൽ (!opt_hostbuf) {
        തിരികെ ശൂന്യ;
    }
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    തിരികെ &dev_ctx->host_buffer_type;
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_cpy(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    ഘടന htp_copy_kernel_params kparams;
    തിരികെ ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_cont(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    ഘടന htp_copy_kernel_params kparams;
    തിരികെ ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_repeat(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    GGML_UNUSED(sess);
    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    // Support f32 and f16
    എങ്കിൽ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) തിരികെ അസത്യം;

    // src and dst must be the same type
    എങ്കിൽ (src0->type != dst->type) തിരികെ അസത്യം;

    // dst dims must be multiples of src dims
    എങ്കിൽ (dst->ne[0] % src0->ne[0] != 0) തിരികെ അസത്യം;
    എങ്കിൽ (dst->ne[1] % src0->ne[1] != 0) തിരികെ അസത്യം;
    എങ്കിൽ (dst->ne[2] % src0->ne[2] != 0) തിരികെ അസത്യം;
    എങ്കിൽ (dst->ne[3] % src0->ne[3] != 0) തിരികെ അസത്യം;

    // require contiguous tensors (no transposition)
    എങ്കിൽ (ggml_is_transposed(src0) || ggml_is_transposed(dst)) തിരികെ അസത്യം;

    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_concat(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    ഘടന htp_concat_kernel_params kparams;
    തിരികെ ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_fill(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    സ്ഥിരം ഘടന ggml_tensor * dst = op;

    എങ്കിൽ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
    GGML_UNUSED(sess);
}

സ്ഥിതം തർക്കം ggml_hexagon_supported_roll(സ്ഥിരം ഘടന ggml_hexagon_session * sess, സ്ഥിരം ഘടന ggml_tensor * op) {
    GGML_UNUSED(sess);

    സ്ഥിരം ഘടന ggml_tensor * src0 = op->src[0];
    സ്ഥിരം ഘടന ggml_tensor * dst  = op;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ggml_are_same_shape(src0, dst)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (src0->nb[0] != ggml_type_size(src0->type) || dst->nb[0] != ggml_type_size(dst->type)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
സ്ഥിതം തർക്കം ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, സ്ഥിരം ഘടന ggml_tensor * op) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    // reject ops that match the filter
    എങ്കിൽ (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        തിരികെ അസത്യം;
    }

    തർക്കം supp = അസത്യം;
    തിരഞ്ഞെടുപ്പ് (op->op) {
        അവസ്ഥ GGML_OP_NONE:
        അവസ്ഥ GGML_OP_RESHAPE:
        അവസ്ഥ GGML_OP_VIEW:
        അവസ്ഥ GGML_OP_PERMUTE:
        അവസ്ഥ GGML_OP_TRANSPOSE:
            supp = സത്യം;
            നിർത്തുക;

        അവസ്ഥ GGML_OP_MUL:
        അവസ്ഥ GGML_OP_ADD:
        അവസ്ഥ GGML_OP_SUB:
        അവസ്ഥ GGML_OP_DIV:
            supp = ggml_hexagon_supported_ദ്വിമാന(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_NORM:
        അവസ്ഥ GGML_OP_L2_NORM:
        അവസ്ഥ GGML_OP_RMS_NORM:
        അവസ്ഥ GGML_OP_SCALE:
        അവസ്ഥ GGML_OP_CLAMP:
        അവസ്ഥ GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_SQR:
        അവസ്ഥ GGML_OP_SQRT:
        അവസ്ഥ GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_SUM:
            supp = ggml_hexagon_supported_sum(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_UNARY:
            തിരഞ്ഞെടുപ്പ് (ggml_get_unary_op(op)) {
                അവസ്ഥ GGML_UNARY_OP_NEG:
                അവസ്ഥ GGML_UNARY_OP_EXP:
                അവസ്ഥ GGML_UNARY_OP_SIGMOID:
                അവസ്ഥ GGML_UNARY_OP_SOFTPLUS:
                അവസ്ഥ GGML_UNARY_OP_TANH:
                അവസ്ഥ GGML_UNARY_OP_ABS:
                അവസ്ഥ GGML_UNARY_OP_SILU:
                അവസ്ഥ GGML_UNARY_OP_GELU:
                അവസ്ഥ GGML_UNARY_OP_GELU_QUICK:
                അവസ്ഥ GGML_UNARY_OP_GELU_ERF:
                അവസ്ഥ GGML_UNARY_OP_RELU:
                അവസ്ഥ GGML_UNARY_OP_STEP:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    supp = അസത്യം;
                    നിർത്തുക;
            }
            നിർത്തുക;

        അവസ്ഥ GGML_OP_GLU:
            തിരഞ്ഞെടുപ്പ് (ggml_get_glu_op(op)) {
                അവസ്ഥ GGML_GLU_OP_SWIGLU:
                അവസ്ഥ GGML_GLU_OP_SWIGLU_OAI:
                അവസ്ഥ GGML_GLU_OP_SWIGLU_CLAMP:
                അവസ്ഥ GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    നിർത്തുക;
                അവസ്ഥ GGML_GLU_OP_GEGLU_QUICK:
                അവസ്ഥ GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    supp = അസത്യം;
                    നിർത്തുക;
            }
            നിർത്തുക;

        അവസ്ഥ GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_DUP:
        അവസ്ഥ GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_മുകൾ_K:
            supp = ggml_hexagon_supported_മുകൾ_k(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_IM2COL_3D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            നിർത്തുക;

        അവസ്ഥ GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            നിർത്തുക;

        സ്ഥിരസ്ഥിതി:
            നിർത്തുക;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    തിരികെ supp;
}

സ്ഥിതം തർക്കം ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    തർക്കം supp = (buft == &dev_ctx->host_buffer_type) || (buft == &dev_ctx->buffer_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    തിരികെ supp;
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ ശൂന്യ,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ ശൂന്യ,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    വേണ്ടി (size_t i = 0; i < opt_ndev; i++) {
        സ്ഥിരം auto & cfg = opt_device_configs[i];
        എങ്കിൽ (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } അല്ലെങ്കിൽ {
            std::string peers_str;
            വേണ്ടി (സ്ഥിരം auto & p : cfg.mdev_group) {
                എങ്കിൽ (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = പുതിയ ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    വേണ്ടി (size_t i = 0; i < opt_ndev; i++) {
        auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        delete dev_ctx;
    }
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    തിരികെ "HTP";
    GGML_UNUSED(reg);
}

സ്ഥിതം size_t ggml_backend_hexagon_reg_get_device_എണ്ണം(ggml_backend_reg_t reg) {
    തിരികെ opt_ndev;
    GGML_UNUSED(reg);
}

സ്ഥിതം ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    എങ്കിൽ (index >= opt_ndev || !hreg->devices[index].context) {
        തിരികെ nullptr;
    }

    തിരികെ &hreg->devices[index];
}

// ** communication context for tensor-split allreduce

സ്ഥിതം ശൂന്യം * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    എങ്കിൽ (n_backends < 2 || n_backends > 4) {
        തിരികെ nullptr;
    }

    വേണ്ടി (size_t i = 0; i < n_backends; ++i) {
        എങ്കിൽ (!ggml_backend_is_hexagon(backends[i])) {
            തിരികെ nullptr;
        }
    }

    വേണ്ടി (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        വേണ്ടി (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            എങ്കിൽ (sess_i->phys_idx == sess_j->phys_idx) {
                തിരികെ nullptr;
            }
        }
    }

    auto * ctx = പുതിയ ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    സ്ഥിതം ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    വേണ്ടി (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (അസ്ഥിര uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].buffer = &sess_i->fence_buf->backend_buffer;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].data   = (ശൂന്യം *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = വലുപ്പം(int32_t);
        ctx->fence_tensors[i].nb[1]  = വലുപ്പം(int32_t);
        ctx->fence_tensors[i].nb[2]  = വലുപ്പം(int32_t);
        ctx->fence_tensors[i].nb[3]  = വലുപ്പം(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    തിരികെ ctx;
}

സ്ഥിതം ശൂന്യം ggml_backend_hexagon_comm_free(ശൂന്യം * comm_ctx_v) {
    എങ്കിൽ (!comm_ctx_v) തിരികെ;
    auto * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    വേണ്ടി (size_t i = 0; i < ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((ശൂന്യം *) ctx->fence_slots[i], 1);
    }
    delete ctx;
}

സ്ഥിതം തർക്കം ggml_backend_hexagon_comm_allreduce_tensor(ശൂന്യം * comm_ctx_v, ഘടന ggml_tensor ** tensors) {
    എങ്കിൽ (opt_ar_select == 0 || !comm_ctx_v) തിരികെ അസത്യം;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    സ്ഥിരം size_t n_backends = comm_ctx->n_backends;

    എങ്കിൽ (n_backends < 2 || n_backends > 4) തിരികെ അസത്യം;

    വേണ്ടി (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        വേണ്ടി (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            എങ്കിൽ (sess_i->phys_idx == sess_j->phys_idx) {
                തിരികെ അസത്യം;
            }
        }
    }

    വേണ്ടി (size_t i = 0; i < n_backends; i++) {
        എങ്കിൽ (!tensors[i] || !tensors[i]->buffer || !ggml_backend_buffer_is_hexagon(tensors[i]->buffer)) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (tensors[i]->type != tensors[0]->type) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (!ggml_is_contiguous(tensors[i])) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            തിരികെ അസത്യം;
        }
    }

    എങ്കിൽ (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    വേണ്ടി (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        ഘടന htp_allreduce_kernel_params kparams;
        എങ്കിൽ (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, അസത്യം, അസത്യം, /*is_shard_ok=*/ അസത്യം, &kparams)) {
            തിരികെ അസത്യം;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    വേണ്ടി (size_t i = 1; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        എങ്കിൽ ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    എങ്കിൽ (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_entry = max_seq;
    എങ്കിൽ (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    വേണ്ടി (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<സ്ഥിരം ggml_tensor *> data_tensors(n_backends);
    std::vector<സ്ഥിരം ggml_tensor *> sync_tensors(n_backends);
    വേണ്ടി (size_t i = 0; i < n_backends; i++) {
        data_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    വേണ്ടി (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], data_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        വേണ്ടി (size_t j = 0; j < n_backends; j++) {
            എങ്കിൽ (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    തിരികെ സത്യം;
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_hexagon_split_buffer_type(പൂർണ്ണസംഖ്യ main_device, സ്ഥിരം ദശാംശം * tensor_split) {
    GGML_UNUSED(tensor_split);
    auto reg = ggml_backend_hexagon_reg();
    auto dev = ggml_backend_reg_dev_get(reg, main_device);
    എങ്കിൽ (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    എങ്കിൽ (!dev) തിരികെ nullptr;
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    തിരികെ &dev_ctx->buffer_type;
}

സ്ഥിതം ശൂന്യം * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, സ്ഥിരം അക്ഷരം * name) {
    GGML_UNUSED(reg);
    എങ്കിൽ (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        തിരികെ (ശൂന്യം *) ggml_backend_hexagon_split_buffer_type;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_comm_init") == 0) {
        തിരികെ (ശൂന്യം *) ggml_backend_hexagon_comm_init;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_comm_free") == 0) {
        തിരികെ (ശൂന്യം *) ggml_backend_hexagon_comm_free;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        തിരികെ (ശൂന്യം *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    തിരികെ ശൂന്യ;
}

template<typename T> std::vector<T> str_to_vec(സ്ഥിരം അക്ഷരം* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    ആകുമ്പോൾ (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    തിരികെ v;
}

template<typename T, പൂർണ്ണസംഖ്യ BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    വേണ്ടി (auto i : v) { ss << i << ','; }
    auto str = ss.str(); str.pop_back(); // drop last comma
    തിരികെ str;
}

സ്ഥിതം ശൂന്യം ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, തർക്കം discovery_supported, സ്ഥിരം std::unordered_map<പൂർണ്ണസംഖ്യ, fastrpc_domain> & cdsp_map) {
    എങ്കിൽ (discovery_supported) {
        auto it = cdsp_map.find(cfg.physical_idx);
        എങ്കിൽ (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } അല്ലെങ്കിൽ {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } അല്ലെങ്കിൽ {
        തിരഞ്ഞെടുപ്പ് (cfg.physical_idx) {
            അവസ്ഥ 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                നിർത്തുക;
            അവസ്ഥ 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                നിർത്തുക;
        }
    }
    വേണ്ടി (auto & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
സ്ഥിതം ശൂന്യം ggml_hexagon_discover_devices() {
    std::unordered_map<പൂർണ്ണസംഖ്യ, fastrpc_domain> cdsp_map;
    തർക്കം discovery_supported = അസത്യം;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    പൂർണ്ണസംഖ്യ err = remote_system_request(&domain_info);
    എങ്കിൽ (err == AEE_SUCCESS && domain_info.sys.അക്കം_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.അക്കം_domains);
        domain_info.sys.domains     = domains.data();
        domain_info.sys.max_domains = (പൂർണ്ണസംഖ്യ) domains.size();

        err = remote_system_request(&domain_info);
        എങ്കിൽ (err == AEE_SUCCESS) {
            discovery_supported = സത്യം;
            സ്ഥിരം പൂർണ്ണസംഖ്യ n_domains = std::min(domain_info.sys.അക്കം_domains, (പൂർണ്ണസംഖ്യ) domains.size());
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (പൂർണ്ണസംഖ്യ) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                എങ്കിൽ (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (പൂർണ്ണസംഖ്യ) domains[i].type);
                    തുടരുക;
                }
                എങ്കിൽ (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    തുടരുക;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } അല്ലെങ്കിൽ {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (unsigned) err);
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (unsigned) err);
    }

    // Populate domain IDs and names for all configured devices
    വേണ്ടി (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

സ്ഥിതം ശൂന്യം ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_Q4_0 == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_Q4_1 == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_Q8_0 == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_MXFP4 == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_IQ4_NL == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_Q4_K == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_Q5_K == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_Q6_K == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_Q3_K == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_Q2_K == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_Q1_0 == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_Q2_0 == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned പൂർണ്ണസംഖ്യ) HTP_TYPE_BF16 == (unsigned പൂർണ്ണസംഖ്യ) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    സ്ഥിരം അക്ഷരം * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    സ്ഥിരം അക്ഷരം * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    സ്ഥിരം അക്ഷരം * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    സ്ഥിരം അക്ഷരം * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    സ്ഥിരം അക്ഷരം * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    സ്ഥിരം അക്ഷരം * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    സ്ഥിരം അക്ഷരം * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    സ്ഥിരം അക്ഷരം * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    സ്ഥിരം അക്ഷരം * str_etm      = getenv("GGML_HEXAGON_ETM");
    സ്ഥിരം അക്ഷരം * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    സ്ഥിരം അക്ഷരം * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    സ്ഥിരം അക്ഷരം * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    സ്ഥിരം അക്ഷരം * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    സ്ഥിരം അക്ഷരം * str_fa_head_split = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    സ്ഥിരം അക്ഷരം * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    സ്ഥിരം അക്ഷരം * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    സ്ഥിരം അക്ഷരം * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    സ്ഥിരം അക്ഷരം * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    സ്ഥിരം അക്ഷരം * str_arch     = getenv("GGML_HEXAGON_ARCH");
    സ്ഥിരം അക്ഷരം * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    സ്ഥിരം അക്ഷരം * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    സ്ഥിരം അക്ഷരം * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    സ്ഥിരം അക്ഷരം * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    സ്ഥിരം അക്ഷരം * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    സ്ഥിരം അക്ഷരം * str_power_mode = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    എങ്കിൽ (!str_arch) {
        പൂർണ്ണസംഖ്യ err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        എങ്കിൽ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } അല്ലെങ്കിൽ {
            എങ്കിൽ (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } അല്ലെങ്കിൽ എങ്കിൽ (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, ശൂന്യ, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? പുതിയ std::regex(str_opfilter, RE_ICASE) : ശൂന്യ;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, ശൂന്യ, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, ശൂന്യ, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, ശൂന്യ, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  ശൂന്യ, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    എങ്കിൽ (str_power_mode) {
        പൂർണ്ണസംഖ്യ m = htp_power_mode_from_str(str_power_mode);
        എങ്കിൽ (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_mode);
        } അല്ലെങ്കിൽ {
            opt_power_mode = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, ശൂന്യ, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_head_split = str_fa_head_split ? atoi(str_fa_head_split)       : opt_fa_head_split;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    എങ്കിൽ (str_mbuf) {
        സ്ഥിരം അക്ഷരം * p = str_mbuf;
        വേണ്ടി (പൂർണ്ണസംഖ്യ idx = 0; idx < 3 && p && *p; idx++) {
            ആകുമ്പോൾ (*p == ' ') p++;
            സ്ഥിരം അക്ഷരം * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : strlen(p);
            ആകുമ്പോൾ (len > 0 && p[len - 1] == ' ') len--;
            എങ്കിൽ (len > 0) {
                std::string token(p, len);
                എങ്കിൽ (idx == 0) opt_mbuf_dyn    = parse_size(token.c_str());
                എങ്കിൽ (idx == 1) opt_mbuf_static = parse_size(token.c_str());
                എങ്കിൽ (idx == 2) opt_mbuf_total  = parse_size(token.c_str());
            }
            എങ്കിൽ (!comma) നിർത്തുക;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_size(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    സ്ഥിരം അക്ഷരം * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    എങ്കിൽ (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    എങ്കിൽ (str_devices && str_devices[0] != '\0') {
        തർക്കം is_single_സംഖ്യ = സത്യം;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; str_devices[i] != '\0'; i++) {
            എങ്കിൽ (!isdigit((unsigned അക്ഷരം)str_devices[i])) {
                is_single_സംഖ്യ = അസത്യം;
                നിർത്തുക;
            }
        }
        എങ്കിൽ (is_single_സംഖ്യ) {
            പൂർണ്ണസംഖ്യ n = atoi(str_devices);
            എങ്കിൽ (n < 1) n = 1;
            എങ്കിൽ (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            വേണ്ടി (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (പൂർണ്ണസംഖ്യ)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.clear();
            }
        } അല്ലെങ്കിൽ {
            std::string s_devices(str_devices);
            std::vector<std::string> ഇനങ്ങൾ;
            std::string curr_item;
            പൂർണ്ണസംഖ്യ bracket_depth = 0;
            വേണ്ടി (അക്ഷരം ch : s_devices) {
                എങ്കിൽ (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } അല്ലെങ്കിൽ എങ്കിൽ (ch == ']') {
                    എങ്കിൽ (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } അല്ലെങ്കിൽ എങ്കിൽ (ch == ',' && bracket_depth == 0) {
                    size_t s = curr_item.find_first_not_of(" \t\r\n");
                    size_t e = curr_item.find_last_not_of(" \t\r\n");
                    എങ്കിൽ (s != std::string::npos) {
                        ഇനങ്ങൾ.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.clear();
                } അല്ലെങ്കിൽ {
                    curr_item += ch;
                }
            }
            size_t s = curr_item.find_first_not_of(" \t\r\n");
            size_t e = curr_item.find_last_not_of(" \t\r\n");
            എങ്കിൽ (s != std::string::npos) {
                ഇനങ്ങൾ.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            വേണ്ടി (സ്ഥിരം auto & item : ഇനങ്ങൾ) {
                size_t b_open  = item.find('[');
                size_t b_close = item.rfind(']');

                എങ്കിൽ (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    പൂർണ്ണസംഖ്യ virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    എങ്കിൽ (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        ശ്രമിക്കുക {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } പിടിക്കുക (...) {
                            virt = 0;
                        }
                    } അല്ലെങ്കിൽ {
                        size_t dev_colon = dev_name.find(':');
                        എങ്കിൽ (dev_colon != std::string::npos) {
                            ശ്രമിക്കുക {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } പിടിക്കുക (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<പൂർണ്ണസംഖ്യ> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    ആകുമ്പോൾ (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        എങ്കിൽ (ps == std::string::npos) തുടരുക;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        എങ്കിൽ (dash_pos != std::string::npos) {
                            ശ്രമിക്കുക {
                                പൂർണ്ണസംഖ്യ p_start = std::stoi(p_part.substr(0, dash_pos));
                                പൂർണ്ണസംഖ്യ p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                വേണ്ടി (പൂർണ്ണസംഖ്യ p = p_start; p <= p_end; p++) {
                                    എങ്കിൽ (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } പിടിക്കുക (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } അല്ലെങ്കിൽ {
                            ശ്രമിക്കുക {
                                പൂർണ്ണസംഖ്യ p = std::stoi(p_part);
                                എങ്കിൽ (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } പിടിക്കുക (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    എങ്കിൽ (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    എങ്കിൽ (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        auto & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.clear();

                        വേണ്ടി (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_group.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } അല്ലെങ്കിൽ {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } അല്ലെങ്കിൽ എങ്കിൽ (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    size_t colon_pos = rest.find(':');
                    പൂർണ്ണസംഖ്യ phys = 0;
                    പൂർണ്ണസംഖ്യ virt = 0;
                    ശ്രമിക്കുക {
                        എങ്കിൽ (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } അല്ലെങ്കിൽ {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } പിടിക്കുക (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        തുടരുക;
                    }

                    എങ്കിൽ (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_group.clear();
                        opt_ndev++;
                    } അല്ലെങ്കിൽ {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } അല്ലെങ്കിൽ {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", item.c_str());
                }
            }
        }
    } അല്ലെങ്കിൽ {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_group.clear();
    }

#എങ്കിൽ defined(__ANDROID__)
    എങ്കിൽ (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    എങ്കിൽ (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            auto v  = str_to_vec<uint32_t>(str_profile);
            തിരഞ്ഞെടുപ്പ് (v.size()) {
                അവസ്ഥ 1:  opt_profile = v[0]; തിരികെ opt_pmu_evt; // mode with default pmu events
                അവസ്ഥ 8:  opt_profile = 2;    തിരികെ v;           // mode with custom  pmu events
                സ്ഥിരസ്ഥിതി: opt_profile = 0;    തിരികെ {};          // garbage input
            }}();
        എങ്കിൽ (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = പുതിയ ggml_hexagon_registry(reg);
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_എണ്ണം,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(ശൂന്യം) {
    സ്ഥിതം തർക്കം initialized = അസത്യം;

    സ്ഥിതം ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ ശൂന്യ };

    {
        സ്ഥിതം std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        എങ്കിൽ (!initialized) {
            auto nErr = htpdrv_init();
            എങ്കിൽ (nErr != AEE_SUCCESS) {
                തിരികെ ശൂന്യ;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = സത്യം;
    }

    തിരികെ &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
