#define _USE_MATH_DEFINES

#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <inttypes.h>
#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <string.h>
#அந்தர்பாவ <time.h>

#அந்தர்பாவ <atomic>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <chrono>
#அந்தர்பாவ <mutex>
#அந்தர்பாவ <thread>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <stdexcept>
#அந்தர்பாவ <string>
#அந்தர்பாவ <sstream>
#அந்தர்பாவ <iomanip>
#அந்தர்பாவ <unordered_set>
#அந்தர்பாவ <unordered_map>
#அந்தர்பாவ <regex>
#அந்தர்பாவ <queue>
#அந்தர்பாவ <deque>
#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cmath>
#அந்தர்பாவ <initializer_list>

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
#இல்லையெனில்
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"

#அந்தர்பாவ <AEEStdErr.h>
#அந்தர்பாவ <dspqueue.h>
#அந்தர்பாவ <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-alloc.h"
#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "ggml-hexagon.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-quants.h"
#அந்தர்பாவ "htp-opnode.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp/matmul-ops.h"
#அந்தர்பாவ "htp/binary-ops.h"
#அந்தர்பாவ "htp/flash-attn-ops.h"
#அந்தர்பாவ "htp/unary-ops.h"
#அந்தர்பாவ "htp/get-rows-ops.h"
#அந்தர்பாவ "htp/set-rows-ops.h"
#அந்தர்பாவ "htp/softmax-ops.h"
#அந்தர்பாவ "htp/pool-ops.h"
#அந்தர்பாவ "htp/rope-ops.h"
#அந்தர்பாவ "htp/ssm-conv.h"
#அந்தர்பாவ "htp/gated-delta-net-ops.h"
#அந்தர்பாவ "htp/argsort-ops.h"
#அந்தர்பாவ "htp/concat-ops.h"
#அந்தர்பாவ "htp/cpy-ops.h"
#அந்தர்பாவ "htp/im2col-ops.h"
#அந்தர்பாவ "htp_iface.h"
#அந்தர்பாவ "htp-drv.h"
#அந்தர்பாவ "htp-repack.h"

using intvec  = std::vector<முழு>;
using uintvec = std::vector<unsigned முழு>;
using u32vec  = std::vector<uint32_t>;

#define GGML_HEXAGON_MAX_SESSIONS          16

#define GGML_HEXAGON_FENCE_SLOT_SIZE       128

கட்டமைப்பு ggml_hexagon_device_config {
    முழு         physical_idx = 0;
    முழு         virtual_idx  = 0;
    முழு         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

நிலையான ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

நிலையான முழு    opt_arch    = 0; // autodetect
நிலையான size_t opt_ndev    = 1;
நிலையான size_t opt_nhvx    = 0; // use all
நிலையான முழு    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
நிலையான size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
நிலையான முழு    opt_etm     = 0;
நிலையான முழு    opt_verbose = 0;
நிலையான முழு    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
நிலையான தருக்க   opt_hostbuf = பொய்;
நிலையான தருக்க   opt_dma64   = பொய்;
நிலையான முழு    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start

நிலையான மாறிலி std::pair<மாறிலி எழுத்து *, முழு> htp_power_mode_names[] = {
    { "low_power_saver",            HTP_POWER_MODE_கீழ்_POWER_SAVER },
    { "power_saver",                HTP_POWER_MODE_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_MODE_உயர்வு_POWER_SAVER },
    { "low_balanced",               HTP_POWER_MODE_கீழ்_BALANCED },
    { "balanced",                   HTP_POWER_MODE_BALANCED },
    { "high_performance",           HTP_POWER_MODE_உயர்வு_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_MODE_SUSTAINED_உயர்வு_PERFORMANCE },
    { "burst",                      HTP_POWER_MODE_BURST },
    { "default",                    HTP_POWER_MODE_BURST }, // flip this to change the default mode
};

நிலையான முழு htp_power_mode_from_str(மாறிலி எழுத்து * s) {
    க்கு (auto & e : htp_power_mode_names) {
#எனில் defined(_MSC_VER)
        எனில் (_stricmp(s, e.first) == 0) {
#இல்லையெனில்
        எனில் (strcasecmp(s, e.first) == 0) {
#endif
            திரும்பு e.second;
        }
    }
    திரும்பு -1;
}

நிலையான மாறிலி எழுத்து * htp_power_mode_to_str(முழு mode) {
    க்கு (auto & e : htp_power_mode_names) {
        எனில் (e.second == mode) {
            திரும்பு e.first;
        }
    }
    திரும்பு "unknown";
}

நிலையான size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
நிலையான size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
நிலையான size_t opt_mbuf_total  = 0;                        // total buffer space limit (0 = unconstrained)

நிலையான முழு    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
நிலையான முழு    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
நிலையான முழு    opt_fa_head_split = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
நிலையான முழு    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
நிலையான முழு    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
நிலையான முழு    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
நிலையான u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

நிலையான முழு opt_opbatch  = 1280; // max number of ops in a batch
நிலையான முழு opt_opqueue  = 8;   // max number of pending batches
நிலையான முழு opt_optrace  = 0;    // trace buffer size per thread (0 means default)
நிலையான முழு opt_oppoll   = 0;    // polling for batch completions
நிலையான முழு opt_opfusion  = 1;    // enable/disable op fusion
நிலையான முழு opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

எண்ணிக்கை ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ADD = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ADD   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

நிலையான உள்வரி தருக்க ggml_hexagon_is_fusion_enabled(முழு flag) {
    எனில் (opt_opfusion <= 0) திரும்பு பொய்;
    எனில் (opt_opfusion == 1) திரும்பு உண்மை; // 1 enables all
    திரும்பு (opt_opfusion & flag) != 0;
}

நிலையான std::regex* opt_opfilter = வெற்று; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    எனில் (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

நிலையான மாறிலி எழுத்து * status_to_str(uint32_t status) {
    தேர்வு (status) {
        நிலை HTP_STATUS_OK:
            திரும்பு "OK";
        நிலை HTP_STATUS_NO_SUPPORT:
            திரும்பு "NO-SUPPORT";
        நிலை HTP_STATUS_INVAL_PARAMS:
            திரும்பு "INVAL-PARAMS";
        நிலை HTP_STATUS_VTCM_TOO_SMALL:
            திரும்பு "VTCM-TOO-SMALL";
        நிலை HTP_STATUS_INTERNAL_ERR:
            திரும்பு "INTERNAL-ERROR";
        இயல்பு:
            திரும்பு "UNKNOWN";
    }
}

// ** debug helpers

நிலையான வெறுமை ggml_hexagon_dump_op_exec(மாறிலி std::string &sess_name, மாறிலி htp_opnode & முனை, மாறிலி uint32_t req_flags) {
    எனில் (!opt_verbose) திரும்பு;

    htp_opformat fmt(முனை);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                முனை.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

நிலையான வெறுமை ggml_hexagon_dump_op_supp(மாறிலி std::string &sess_name, மாறிலி கட்டமைப்பு ggml_tensor * op, தருக்க supp) {
    எனில் (!opt_verbose) திரும்பு;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

நிலையான மாறிலி எழுத்து * htp_event_name(uint16_t id) {
    தேர்வு (id) {
        நிலை HTP_TRACE_EVT_DMA:            திரும்பு "DMA";
        நிலை HTP_TRACE_EVT_HVX_COMP:       திரும்பு "HVX_COMP";
        நிலை HTP_TRACE_EVT_HVX_A_QUANT:    திரும்பு "HVX_A_QUANT";
        நிலை HTP_TRACE_EVT_HVX_A_PREP:     திரும்பு "HVX_A_PREP";
        நிலை HTP_TRACE_EVT_HVX_W_DEQUANT:  திரும்பு "HVX_W_DEQUANT";
        நிலை HTP_TRACE_EVT_HVX_W_PREP:     திரும்பு "HVX_W_PREP";
        நிலை HTP_TRACE_EVT_HVX_O_PROC:     திரும்பு "HVX_O_PROC";
        நிலை HTP_TRACE_EVT_HVX_FA_QK:      திரும்பு "HVX_QK_FA";
        நிலை HTP_TRACE_EVT_HVX_FA_SFM:     திரும்பு "HVX_SFM_FA";
        நிலை HTP_TRACE_EVT_HVX_FA_Q_PREP:  திரும்பு "HVX_Q_PREP";
        நிலை HTP_TRACE_EVT_HVX_FA_K_PREP:  திரும்பு "HVX_K_PREP";
        நிலை HTP_TRACE_EVT_HVX_FA_V_PREP:  திரும்பு "HVX_V_PREP";
        நிலை HTP_TRACE_EVT_HVX_GDN_PREP:   திரும்பு "HVX_GDN_PREP";
        நிலை HTP_TRACE_EVT_HVX_GDN_SOLVE:  திரும்பு "HVX_GDN_SOLVE";
        நிலை HTP_TRACE_EVT_HVX_GDN_V_PREP: திரும்பு "HVX_GDN_V_PREP";
        நிலை HTP_TRACE_EVT_HVX_GDN_D_PREP: திரும்பு "HVX_GDN_D_PREP";
        நிலை HTP_TRACE_EVT_HVX_GDN_OUT:    திரும்பு "HVX_GDN_OUT";
        நிலை HTP_TRACE_EVT_HVX_GDN_STATE:  திரும்பு "HVX_GDN_STATE";
        நிலை HTP_TRACE_EVT_HVX_GDN_REM:    திரும்பு "HVX_GDN_REM";
        நிலை HTP_TRACE_EVT_HMX_COMP:       திரும்பு "HMX_COMP";
        நிலை HTP_TRACE_EVT_L2FLUSH:        திரும்பு "L2FLUSH";
        நிலை HTP_TRACE_EVT_INIT:           திரும்பு "INIT";
        நிலை HTP_TRACE_EVT_BUFF:           திரும்பு "BUFF";
        நிலை HTP_TRACE_EVT_FENCE:          திரும்பு "FENCE";
        இயல்பு:                           திரும்பு "UNKNOWN";
    }
}

நிலையான வெறுமை ggml_hexagon_dump_op_prof(மாறிலி std::string &sess_name, மாறிலி htp_opnode & முனை, மாறிலி htp_prof_desc & pd) {
    எனில் (!opt_profile) திரும்பு;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    மாறிலி uint32_t * pmu = pd.pmu;

    எழுத்து pmu_str[256] = "";
    எனில் (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, அளவு(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(முனை);
    தசம mhz = op_usec > 0 ? (தசம) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            முனை.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

நிலையான வெறுமை ggml_hexagon_dump_batch_prof(மாறிலி std::string & sess_name, மாறிலி htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    தசம batch_mhz = rsp.usecs > 0 ? (தசம) batch_cycles / rsp.usecs : 0.0f;

    எழுத்து evt_str[256] = "----";
    எனில் (opt_profile == 3) {
        snprintf(evt_str, அளவு(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (unsigned long long) batch_cycles, (unsigned long long) rsp.cycles_start, batch_mhz);
}

நிலையான வெறுமை ggml_hexagon_dump_trace_events(மாறிலி std::string & sess_name, மாறிலி htp_opbatch_rsp & rsp,
                                           மாறிலி htp_trace_desc * trace_events, uint32_t n_traces) {
    எனில் (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        க்கு (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        க்கு (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            க்கு (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                மாறிலி auto & e = trace_events[t * n_traces + idx];
                தருக்க is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

எண்ணிக்கை ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_WEIGHT    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

கட்டமைப்பு ggml_hexagon_session;

நிலையான வெறுமை ggml_hexagon_precompute_matmul_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_mm_kernel_params * kparams
);

நிலையான வெறுமை ggml_hexagon_precompute_fused_matmul_add_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * src2,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_mm_kernel_params * kparams
);

நிலையான வெறுமை ggml_hexagon_precompute_unary_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    uint32_t op,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_unary_kernel_params * kparams
);

நிலையான தருக்க ggml_hexagon_precompute_இருமம்_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    uint32_t op,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_இருமம்_kernel_params * kparams,
    மாறிலி கட்டமைப்பு ggml_tensor * src2 = nullptr
);

நிலையான வெறுமை ggml_hexagon_precompute_get_rows_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_get_rows_kernel_params * kparams
);

நிலையான வெறுமை ggml_hexagon_precompute_set_rows_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_set_rows_kernel_params * kparams
);

நிலையான வெறுமை ggml_hexagon_precompute_softmax_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_softmax_kernel_params * kparams
);

நிலையான வெறுமை ggml_hexagon_precompute_rope_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_rope_kernel_params * kparams
);

நிலையான வெறுமை ggml_hexagon_precompute_ssm_conv_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_ssm_conv_kernel_params * kparams
);

நிலையான வெறுமை ggml_hexagon_precompute_gated_delta_net_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_gdn_kernel_params * kparams
);

நிலையான வெறுமை ggml_hexagon_precompute_sort_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    தருக்க is_உச்சி_k,
    கட்டமைப்பு htp_sort_kernel_params * kparams
);

நிலையான வெறுமை ggml_hexagon_precompute_pool_2d_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_pool_2d_kernel_params * kparams,
    தருக்க is_pool_1d
);

நிலையான தருக்க ggml_hexagon_precompute_concat_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_concat_kernel_params * kparams
);

நிலையான தருக்க ggml_hexagon_precompute_cpy_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_copy_kernel_params * kparams
);

நிலையான தருக்க ggml_hexagon_precompute_im2col_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_im2col_kernel_params * kparams
);

நிலையான தருக்க ggml_hexagon_precompute_im2col_3d_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_im2col_kernel_params * kparams
);

நிலையான வெறுமை ggml_hexagon_precompute_fused_mmnx_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * act,
    int32_t n_weights,
    கட்டமைப்பு htp_mm_kernel_params * kparams
);

நிலையான வெறுமை ggml_hexagon_precompute_fused_mmidnx_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * act,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    int32_t n_weights,
    கட்டமைப்பு htp_mm_kernel_params * kparams
);

நிலையான தருக்க ggml_hexagon_precompute_allreduce_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    தருக்க has_add,
    தருக்க is_row_bcast,
    தருக்க is_shard_ok,
    கட்டமைப்பு htp_allreduce_kernel_params * kparams
);

நிலையான தருக்க ggml_hexagon_rows_stride(மாறிலி int64_t * ne, மாறிலி size_t * nb, size_t * stride);
நிலையான தருக்க ggml_hexagon_matmul_can_collapse(மாறிலி கட்டமைப்பு ggml_tensor * src0, மாறிலி கட்டமைப்பு ggml_tensor * src1, மாறிலி கட்டமைப்பு ggml_tensor * dst);
நிலையான ggml_tensor ggml_hexagon_tensor_collapse_rows(மாறிலி கட்டமைப்பு ggml_tensor * t);

நிலையான தருக்க mm_is_hmx_eligible(மாறிலி ggml_tensor * t);
நிலையான htp_op_code op_remap_to_htp(மாறிலி ggml_tensor * t);
நிலையான தருக்க is_supported_mul_mat_nx_kernel(மாறிலி ggml_tensor * src0, மாறிலி கட்டமைப்பு htp_mm_kernel_params * kparams);
நிலையான தருக்க is_supported_mul_mat_id_nx_kernel(மாறிலி ggml_tensor * src0, மாறிலி கட்டமைப்பு htp_mm_kernel_params * kparams);
நிலையான தருக்க is_mergeable_mul_mat(மாறிலி ggml_tensor * t);
நிலையான தருக்க is_mergeable_mul_mat_pair(மாறிலி ggml_tensor * n1, மாறிலி ggml_tensor * n2);
நிலையான தருக்க is_mergeable_mul_mat_id(மாறிலி ggml_tensor * t);
நிலையான தருக்க is_mergeable_mul_mat_id_pair(மாறிலி ggml_tensor * n1, மாறிலி ggml_tensor * n2);

// ** backend sessions

கட்டமைப்பு ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_size { 0 };
    uint32_t             flags { 0 };
};

நிலையான உள்வரி தருக்க ggml_hexagon_tensor_is_fuseable(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    எனில் (!t->extra) திரும்பு பொய்;
    auto extra = (மாறிலி கட்டமைப்பு ggml_hexagon_tensor_extra *) t->extra;
    திரும்பு (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

நிலையான உள்வரி தருக்க ggml_hexagon_can_rt_repack(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    எனில் (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
        திரும்பு பொய்;
    }
    எனில் (t->buffer && ggml_backend_buft_is_host(t->buffer->buft)) {
        திரும்பு பொய்;
    }
    திரும்பு opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

நிலையான உள்வரி தருக்க ggml_hexagon_tensor_is_tiled(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    மாறிலி auto * extra = (மாறிலி ggml_hexagon_tensor_extra *) t->extra;
    எனில் (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        திரும்பு உண்மை;
    }
    திரும்பு ggml_hexagon_can_rt_repack(t);
}

நிலையான உள்வரி தருக்க ggml_hexagon_needs_repack(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    மாறிலி auto * extra = (மாறிலி ggml_hexagon_tensor_extra *) t->extra;
    எனில் (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        திரும்பு பொய்;
    }
    திரும்பு ggml_hexagon_can_rt_repack(t);
}

நிலையான உள்வரி தருக்க ggml_hexagon_tensors_overlap(மாறிலி கட்டமைப்பு ggml_tensor * a, மாறிலி கட்டமைப்பு ggml_tensor * b) {
    மாறிலி uintptr_t a0 = (uintptr_t) a->data;
    மாறிலி uintptr_t b0 = (uintptr_t) b->data;
    மாறிலி uintptr_t a1 = a0 + ggml_nbytes(a);
    மாறிலி uintptr_t b1 = b0 + ggml_nbytes(b);

    திரும்பு a0 < b1 && b0 < a1;
}

நிலையான உள்வரி தருக்க ggml_hexagon_can_row_partition(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    எனில் (t->ne[1] > 1 && (t->nb[1] & 127) != 0) திரும்பு பொய்;
    எனில் (t->ne[2] > 1 && (t->nb[2] & 127) != 0) திரும்பு பொய்;
    எனில் (t->ne[3] > 1 && (t->nb[3] & 127) != 0) திரும்பு பொய்;
    திரும்பு உண்மை;
}

கட்டமைப்பு htp_opnode;

கட்டமைப்பு ggml_hexagon_opbatch;
கட்டமைப்பு ggml_hexagon_opqueue;
கட்டமைப்பு ggml_hexagon_shared_buffer;
கட்டமைப்பு ggml_hexagon_fence_buffer;
கட்டமைப்பு ggml_hexagon_session;
கட்டமைப்பு ggml_backend_hexagon_device_context;

கட்டமைப்பு ggml_hexagon_mdev_group {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

கட்டமைப்பு ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    நிலையற்ற uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

கட்டமைப்பு ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    நிலையற்ற uint32_t *    fence_slot   = nullptr;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

கட்டமைப்பு ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    முழு              phys_idx;
    முழு              virt_idx;
    தருக்க             valid_session;
    தருக்க             valid_handle;
    தருக்க             valid_queue;
    தருக்க             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<முழு, std::unique_ptr<ggml_hexagon_shared_buffer>> cloned_buffers;
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

    ggml_hexagon_session(மாறிலி ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_மொத்தம் = 0) noexcept(பொய்);
    ~ggml_hexagon_session() noexcept(உண்மை);

    மாறிலி எழுத்து* c_name() மாறிலி { திரும்பு name.c_str(); }

    வெறுமை allocate(மாறிலி ggml_hexagon_device_config & config) noexcept(பொய்);
    வெறுமை release() noexcept(உண்மை);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    வெறுமை      free_fence(வெறுமை * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, நிலையற்ற uint32_t *> cpy_fence_slots;

    வெறுமை enqueue_mdev_group();
    வெறுமை enqueue_op(மாறிலி htp_opnode & முனை);
    வெறுமை enqueue_cpy(மாறிலி ggml_tensor * src, ggml_tensor * dst, மாறிலி ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    வெறுமை enqueue_fence(மாறிலி ggml_tensor * sync_tensor, uint32_t fence_seq = 0, தருக்க wait = உண்மை);
    வெறுமை enqueue_allreduce(மாறிலி ggml_tensor * dst, மாறிலி std::vector<மாறிலி ggml_tensor *> & src_tensors,
                           மாறிலி std::vector<மாறிலி ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);

    வெறுமை start_batch();
    வெறுமை flush_sync(தருக்க all = உண்மை);
    வெறுமை flush_async();
    வெறுமை flush_batch(size_t min_ops = 1);
    வெறுமை flush_peers();
    வெறுமை flush_pending(தருக்க all = உண்மை);

    ggml_hexagon_shared_buffer * mmap_tensor(மாறிலி ggml_tensor * t);
    தருக்க clone_buffer(மாறிலி ggml_hexagon_shared_buffer*);
    வெறுமை release_buffer(மாறிலி ggml_hexagon_shared_buffer*);
    வெறுமை unclone_buffer(மாறிலி ggml_hexagon_shared_buffer*);

    வெறுமை add_peer(ggml_hexagon_session * peer) {
        எனில் (இது->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } இல்லையெனில் {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

கட்டமைப்பு ggml_backend_hexagon_device_context {
    முழு                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_buffer_type buffer_type       = {};
    ggml_backend_buffer_type host_buffer_type  = {};
    ggml_backend_buffer_type fence_buffer_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(முழு dev_id, மாறிலி ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    மாறிலி எழுத்து * c_name() மாறிலி { திரும்பு config.name.c_str(); }

    ggml_hexagon_session * session() {
        எனில் (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        திரும்பு sess.get();
    }
};

கட்டமைப்பு ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(மாறிலி std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        இது->dev_ctx = dev_ctx;
        இது->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

கட்டமைப்பு ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    முழு       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        எனில் (!base) {
            எறி std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        எனில் (fd < 0) {
            rpcmem_free(base);
            எறி std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        இது->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        எனில் (base) {
            rpcmem_free(base);
        }
    }
};

கட்டமைப்பு ggml_hexagon_shared_buffer {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    தருக்க     mapped;
    தருக்க     pinned;
    தருக்க     extended;

    மாறிலி எழுத்து * c_name() மாறிலி { திரும்பு sess->c_name(); }
    uint8_t *    base()   மாறிலி { திரும்பு mem ? mem->base : nullptr; }
    size_t       size()   மாறிலி { திரும்பு mem ? mem->size : 0;  }
    முழு          fd()     மாறிலி { திரும்பு mem ? mem->fd   : -1; }

    வெறுமை mmap(தருக்க extended = பொய்) {
        எனில் (!இது->mem)   திரும்பு;
        எனில் (இது->mapped) திரும்பு;

        GGML_ASSERT(!இது->pinned || !extended);

        இது->extended = extended;
        fastrpc_map_flags flags = இது->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        முழு err = fastrpc_mmap(sess->domain_id, fd(), (வெறுமை *) base(), 0, size(), flags);
        எனில் (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (unsigned) err);
            எறி std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (வெறுமை *) base(), size(), fd(), pinned, extended);

        இது->mapped = உண்மை;
    }

    வெறுமை unmap() {
        எனில் (!இது->mapped) திரும்பு;

        எனில் (!இது->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        எனில் (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (வெறுமை *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (வெறுமை *) base(), size(), fd());

        இது->mapped = பொய்;
    }

    வெறுமை alloc(size_t size) {
        எனில் (இது->mem) திரும்பு;

        இது->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (வெறுமை *) base(), இது->size(), fd(), (முழு) pinned);
        எனில் (இது->pinned) {
            mmap();
        }
    }

    வெறுமை விடுவி() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (வெறுமை *) base(), size(), fd());
        இது->mem  = nullptr;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, தருக்க pinned = பொய்) {
        இது->sess     = sess;
        இது->mapped   = பொய்;
        இது->pinned   = pinned;
        இது->extended = பொய்;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_offset = (size + 4095) & ~4095;
        size_t total_size   = guard_offset + 4096;
        எனில் (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            total_size = (total_size + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(total_size);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, மாறிலி ggml_hexagon_shared_buffer & other) {
        இது->sess   = sess;
        இது->mem    = other.mem;
        இது->mapped = பொய்;
        இது->pinned = other.pinned;
        இது->extended = other.extended;
    }

    ~ggml_hexagon_shared_buffer() {
        விடுவி();
        க்கு (auto * extra : tensor_extra) {
            delete extra;
        }
    }
};

கட்டமைப்பு ggml_hexagon_fence_buffer : பொது ggml_hexagon_shared_buffer {
    uint32_t              slot_மொத்தம் = 0;
    uint32_t              slot_head  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_buffer   backend_buffer{};

    ggml_hexagon_fence_buffer(ggml_hexagon_session * sess, ggml_backend_buffer_type_t buft, size_t size)
        : ggml_hexagon_shared_buffer(sess, size, பொய் /* pinned */),
          slot_மொத்தம்(size / GGML_HEXAGON_FENCE_SLOT_SIZE),
          slot_head(0) {
        backend_buffer.buft    = buft;
        backend_buffer.context = static_cast<ggml_hexagon_shared_buffer *>(இது);
        backend_buffer.size    = size;
        mmap(பொய்);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        எனில் (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        } இல்லையெனில் எனில் (slot_head + n_slots <= slot_மொத்தம்) {
            uint32_t slot = slot_head;
            slot_head += n_slots;
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        }
        எனில் (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_SLOT_SIZE);
        }
        திரும்பு ptr;
    }

    வெறுமை free_slot(வெறுமை * ptr, uint32_t n_slots = 1) {
        எனில் (!ptr) திரும்பு;
        uint32_t slot = ((uint8_t *) ptr - base()) / GGML_HEXAGON_FENCE_SLOT_SIZE;
        க்கு (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

உள்வரி uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    திரும்பு ptr;
}

உள்வரி வெறுமை ggml_hexagon_session::free_fence(வெறுமை * ptr, uint32_t n_slots) {
    எனில் (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

நிலையான வெறுமை ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    sbuf->sess->unclone_buffer(sbuf);
    delete sbuf;
}

நிலையான வெறுமை * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    திரும்பு sbuf->base();
}

நிலையான எண்ணிக்கை ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (வெறுமை *) sbuf->base(), tensor->data, ggml_nbytes(tensor));

    auto extra = புதிய ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    திரும்பு GGML_STATUS_SUCCESS;
}

நிலையான வெறுமை ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   மாறிலி வெறுமை *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    எனில் (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        எனில் (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, data, offset, size, (முழு) buffer->usage, extra->flags);

    எனில் ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy((எழுத்து *) tensor->data + offset, data, size);
        திரும்பு;
    }

    எனில் (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        htp_repack_tensor_tiled(tensor, data, size);
        திரும்பு;
    }

    எனில் (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    memcpy(extra->shadow_buf.data() + offset, data, size);
    extra->shadow_size += size;

    எனில் (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

நிலையான வெறுமை ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   மாறிலி ggml_tensor *   tensor,
                                                   வெறுமை *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, data, offset, size, (முழு) buffer->usage, extra->flags);

    எனில் ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy(data, (மாறிலி எழுத்து *) tensor->data + offset, size);
        திரும்பு;
    }

    htp_unpack_tensor_tiled(data, tensor, offset, size);
}

நிலையான தருக்க ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   மாறிலி கட்டமைப்பு ggml_tensor * src,
                                                   கட்டமைப்பு ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    திரும்பு பொய்;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

நிலையான வெறுமை ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                          ggml_tensor *         tensor,
                                                          மாறிலி வெறுமை *          data,
                                                          size_t                offset,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    எனில் (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        எனில் (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (முழு) buffer->usage, extra->flags);

    எனில் ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        க்கு (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (மாறிலி uint8_t *) data + i * stride_data, size);
        }
        திரும்பு;
    }

    எனில் (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    க்கு (size_t i = 0; i < n_copies; i++) {
        memcpy(extra->shadow_buf.data() + offset + i * stride_tensor, (மாறிலி uint8_t *) data + i * stride_data, size);
    }
    extra->shadow_size += n_copies * size;

    எனில் (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

நிலையான வெறுமை ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                      மாறிலி ggml_tensor *   tensor,
                                                      வெறுமை *                data,
                                                      size_t                offset,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (முழு) buffer->usage);

    எனில் ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        க்கு (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *)data + i * stride_data, (மாறிலி uint8_t *)tensor->data + offset + i * stride_tensor, size);
        }
        திரும்பு;
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

    க்கு (size_t i = 0; i < n_copies; i++) {
        memcpy((uint8_t *) data + i * stride_data, temp_buf.data() + i * stride_tensor, size);
    }
}

நிலையான வெறுமை ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (வெறுமை *) sbuf->base(), sbuf->size());
    memset(sbuf->base(), value, sbuf->size());
}

நிலையான ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ வெற்று,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ வெற்று,
};

// ** backend buffer type

நிலையான வெறுமை ggml_backend_hexagon_host_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                        ggml_tensor *         tensor,
                                                        மாறிலி வெறுமை *          data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy((எழுத்து *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

நிலையான வெறுமை ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                        மாறிலி ggml_tensor *   tensor,
                                                        வெறுமை *                data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy(data, (மாறிலி எழுத்து *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

நிலையான ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ வெற்று,
    /* .set_tensor      = */ ggml_backend_hexagon_host_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_buffer_get_tensor,
    /* .set_tensor_2d   = */ வெற்று,
    /* .get_tensor_2d   = */ வெற்று,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ வெற்று,
};

// ** backend buffer type

நிலையான மாறிலி எழுத்து * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    திரும்பு static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

நிலையான ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    எனில் (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        திரும்பு nullptr;
    }
    முயல் {
        ggml_hexagon_shared_buffer * sbuf = புதிய ggml_hexagon_shared_buffer(sess, size, பொய்);
        திரும்பு ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } பிடி (மாறிலி std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        திரும்பு nullptr;
    }
}

நிலையான ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    எனில் (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        திரும்பு nullptr;
    }
    முயல் {
        ggml_hexagon_shared_buffer * sbuf = புதிய ggml_hexagon_shared_buffer(sess, size, பொய்);
        திரும்பு ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_host_buffer_interface, sbuf, size);
    } பிடி (மாறிலி std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        திரும்பு nullptr;
    }
}

நிலையான size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    திரும்பு 128;  // HVX alignment
    GGML_UNUSED(buft);
}

நிலையான size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, மாறிலி கட்டமைப்பு ggml_tensor * t) {
    எனில் (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_size = htp_tiled_row_size(t->type, ne0) * ne1 * ne2 * ne3;
        திரும்பு std::max(tiled_size, ggml_nbytes(t));
    }
    திரும்பு ggml_nbytes(t);
}

நிலையான size_t parse_size(மாறிலி எழுத்து * str, size_t default_unit = 1024 * 1024) {
    எனில் (!str || str[0] == '\0') {
        திரும்பு 0;
    }
    எழுத்து * end = வெற்று;
    இரட்டை val = strtod(str, &end);
    எனில் (val < 0) {
        திரும்பு 0;
    }
    எனில் (end && *end) {
        போது (*end == ' ') end++;
        எனில் (*end == 'k' || *end == 'K') {
            திரும்பு (size_t) (val * 1024);
        } இல்லையெனில் எனில் (*end == 'm' || *end == 'M') {
            திரும்பு (size_t) (val * 1024 * 1024);
        } இல்லையெனில் எனில் (*end == 'g' || *end == 'G') {
            திரும்பு (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    திரும்பு (size_t) (val * default_unit);
}

நிலையான size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    திரும்பு opt_mbuf_dyn;
    GGML_UNUSED(buft);
}

நிலையான தருக்க ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    திரும்பு பொய்;
    GGML_UNUSED(buft);
}

நிலையான தருக்க ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    திரும்பு உண்மை;
    GGML_UNUSED(buft);
}

கட்டமைப்பு ggml_backend_hexagon_alloc_buffer_n_plan_item {
    size_t size;
    முழு    first;
    முழு    last;
};

using ggml_backend_hexagon_alloc_buffer_n_plan_t = std::vector<ggml_backend_hexagon_alloc_buffer_n_plan_item>;

நிலையான மாறிலி எழுத்து * ggml_hexagon_kv_layer_suffix(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    எனில் (strncmp(t->name, "cache_", 6) != 0) {
        திரும்பு வெற்று;
    }
    மாறிலி எழுத்து * p = strstr(t->name, "_l");
    எனில் (!p || !isdigit((unsigned எழுத்து)p[2])) {
        திரும்பு வெற்று;
    }
    திரும்பு p;
}

கட்டமைப்பு ggml_backend_hexagon_alloc_unit {
    size_t size;
    முழு    first;
    முழு    last;
};

நிலையான ggml_backend_hexagon_alloc_buffer_n_plan_t ggml_backend_hexagon_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, கட்டமைப்பு ggml_tensor ** tensors, முழு n_tensors) {
    ggml_backend_hexagon_alloc_buffer_n_plan_t plan;

    மாறிலி size_t alignment = ggml_backend_buft_get_alignment(buft);
    மாறிலி size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    முழு i = 0;
    போது (i < n_tensors) {
        கட்டமைப்பு ggml_tensor * t = tensors[i];
        size_t unit_size = 0;
        முழு unit_first = i;
        முழு unit_last = i + 1;

        எனில் (t->data == வெற்று && t->view_src == வெற்று) {
            unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        மாறிலி எழுத்து * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        போது (unit_last < n_tensors) {
            கட்டமைப்பு ggml_tensor * next = tensors[unit_last];

            எனில் (next->view_src != வெற்று) {
                unit_last++;
                தொடர்;
            }

            எனில் (layer_suffix != வெற்று) {
                மாறிலி எழுத்து * next_suffix = ggml_hexagon_kv_layer_suffix(next);
                எனில் (next_suffix != வெற்று && strcmp(layer_suffix, next_suffix) == 0) {
                    எனில் (next->data == வெற்று) {
                        unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, next), alignment);
                    }
                    unit_last++;
                    தொடர்;
                }
            }

            நிறுத்து;
        }

        units.push_back({ unit_size, unit_first, unit_last });
        i = unit_last;
    }

    size_t cur_buf_size  = 0;
    முழு    cur_buf_first = 0;

    க்கு (மாறிலி auto & unit : units) {
        எனில் (unit.size == 0) {
            தொடர்;
        }

        எனில் (cur_buf_size > 0 && (cur_buf_size + unit.size) > max_size) {
            plan.push_back({ cur_buf_size, cur_buf_first, unit.first });
            cur_buf_size  = 0;
            cur_buf_first = unit.first;
        }

        cur_buf_size += unit.size;
    }

    எனில் (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, cur_buf_first, n_tensors });
    }

    திரும்பு plan;
}

நிலையான ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, கட்டமைப்பு ggml_tensor ** tensors, முழு n_tensors) {
    மாறிலி ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    க்கு (மாறிலி auto & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        எனில் (buffer == வெற்று) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            க்கு (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            திரும்பு வெற்று;
        }

        கட்டமைப்பு ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        கட்டமைப்பு ggml_tensor * t_failed = வெற்று;
        க்கு (முழு j = item.first; j < item.last; j++) {
            கட்டமைப்பு ggml_tensor * t = tensors[j];
            எனில் (t->data == வெற்று) {
                எனில் (t->view_src == வெற்று) {
                    எனில் (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        நிறுத்து;
                    }
                } இல்லையெனில் எனில் (t->buffer == வெற்று) {
                    எனில் (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        நிறுத்து;
                    }
                }
            } இல்லையெனில் {
                எனில் (t->view_src != வெற்று && t->buffer == வெற்று) {
                    எனில் (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        நிறுத்து;
                    }
                }
            }
        }
        எனில் (t_failed != வெற்று) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            க்கு (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            திரும்பு வெற்று;
        }

        buffers.push_back(buffer);
    }

    எனில் (buffers.empty()) {
        திரும்பு வெற்று;
    }

    எனில் (buffers.size() == 1) {
        திரும்பு buffers[0];
    }

    திரும்பு ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

நிலையான size_t ggml_backend_hexagon_buffer_type_get_alloc_size_n(
        ggml_backend_buffer_type_t buft, கட்டமைப்பு ggml_tensor ** tensors, முழு n_tensors) {
    மாறிலி ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    க்கு (மாறிலி auto & item : plan) {
        total += item.size;
    }
    திரும்பு total;
}

நிலையான ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_buffer_type_is_host,
};

நிலையான ggml_backend_buffer_type_i ggml_backend_hexagon_host_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_host_buffer_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(முழு dev_id, மாறிலி ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    buffer_type.device  = dev;
    buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    buffer_type.context = புதிய ggml_backend_hexagon_buffer_type_context(config.name, இது);

    host_buffer_type.device  = dev;
    host_buffer_type.iface   = ggml_backend_hexagon_host_buffer_type_interface;
    host_buffer_type.context = புதிய ggml_backend_hexagon_buffer_type_context(config.name + "-HOST", இது);

    fence_buffer_type.device  = dev;
    fence_buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    fence_buffer_type.context = புதிய ggml_backend_hexagon_buffer_type_context(config.name + "-FENCE", இது);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(host_buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(fence_buffer_type.context);
}

நிலையான தருக்க ggml_backend_buffer_is_hexagon(மாறிலி கட்டமைப்பு ggml_backend_buffer * b) {
    திரும்பு b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

கட்டமைப்பு ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<முழு, முழு>                b_map; // buffer fd   to index
    std::unordered_map<மாறிலி ggml_tensor*, முழு> t_map; // tensor ptr  to index
    std::unordered_multimap<வெறுமை*, முழு>         d_map; // tensor data to index

    unsigned முழு n_bufs;     // num buffers in the batch
    unsigned முழு n_tens;     // num tensors ...
    unsigned முழு n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    unsigned முழு n_bufs_max;
    unsigned முழு n_tens_max;
    unsigned முழு n_ops_max;
    size_t       b_vmem_max;

    வெறுமை reset() {
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
        இது->sess = sess;

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

    தருக்க empty() மாறிலி { திரும்பு n_ops == 0; }

    // add buffer and return its index
    முழு add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        auto it = b_map.find(sbuf->fd());
        எனில் (it != b_map.end()) { திரும்பு it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        முழு bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        எனில் (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (வெறுமை*) sbuf->base(), (size_t) b.size, b_vmem);

        திரும்பு bi;
    }

    தருக்க same_shape(மாறிலி htp_tensor * h, மாறிலி ggml_tensor * t) மாறிலி {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;

        மாறிலி தருக்க is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        மாறிலி int64_t ne0_padded = hex_round_up(t->ne[0], 32);
        மாறிலி int64_t ne1_padded = hex_round_up(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_row_size(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? அளவு(ggml_fp16_t) : t->nb[0];
        எண்ணிக்கை ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        திரும்பு (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    முழு add_tensor(மாறிலி ggml_tensor * t) {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;
        auto sbuf  = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        க்கு (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            எனில் (same_shape(h, t)) { திரும்பு it->second; }
        }

        // Lookup by tensor ptr
        auto it = t_map.find(t);
        எனில் (it != t_map.end()) { திரும்பு it->second; }

        // Add new tensor to the batch
        முழு ti = n_tens++;
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

        மாறிலி தருக்க is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        எனில் (is_tiled) {
            எனில் (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = அளவு(ggml_fp16_t);
            } இல்லையெனில் {
                h.nb[0] = t->nb[0];
            }
            மாறிலி int64_t ne0_padded = hex_round_up(t->ne[0], 32);
            மாறிலி int64_t ne1_padded = hex_round_up(t->ne[1], 32);
            h.nb[1] = htp_tiled_row_size(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_size  = h.size;
        } இல்லையெனில் {
            h.size  = t_size;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        எனில் ((extra->flags & GGML_HEXAGON_TENSOR_WEIGHT) != 0) {
            h.flags |= HTP_TENSOR_WEIGHT;
        }
        எனில் ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        எனில் ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (வெறுமை*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        திரும்பு ti;
    }

    தருக்க fit_op(மாறிலி htp_opnode & முனை) மாறிலி {
        எனில் (n_ops >= n_ops_max) திரும்பு பொய்;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        முழு seen_bufs[HTP_OP_MAX_BUFS];
        முழு n_seen_bufs = 0;

        auto fit_tensor = [&](மாறிலி ggml_tensor *t) {
            எனில் (!t) திரும்பு;
            எனில் (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                முழு fd = sbuf->fd();
                எனில் (!b_map.count(fd)) {
                    க்கு (முழு i = 0; i < n_seen_bufs; i++) {
                        எனில் (seen_bufs[i] == fd) திரும்பு;
                    }
                    எனில் (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    எனில் (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        க்கு (மாறிலி auto * src : முனை.get_inputs()) {
            fit_tensor(src);
        }
        க்கு (மாறிலி auto * output : முனை.get_outputs()) {
            fit_tensor(output);
        }

        எனில் ((extra_bufs + n_bufs) > n_bufs_max) திரும்பு பொய்;
        எனில் ((extra_tens + n_tens) > n_tens_max) திரும்பு பொய்;
        எனில் ((extra_vmem + b_vmem) > b_vmem_max) திரும்பு பொய்;

        திரும்பு உண்மை;
    }

    // assumes that fit_op() was called first and returned true
    வெறுமை add_op(மாறிலி htp_opnode & முனை) {
        // Add new op

        unsigned முழு n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = முனை;

        htp_op_desc &o = h_ops[n];
        memcpy(o.params,        முனை.முனை->op_params, அளவு(முனை.முனை->op_params));
        memcpy(o.kernel_params, முனை.kernel_params,   அளவு(o.kernel_params));
        o.opcode = முனை.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        auto inputs = முனை.get_inputs();
        க்கு (unsigned முழு i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        auto outputs = முனை.get_outputs();
        க்கு (unsigned முழு i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    வெறுமை sort_buffers() {
        எனில் (n_bufs <= 1) திரும்பு;

        std::vector<முழு> order(n_bufs);
        க்கு (unsigned முழு i = 0; i < n_bufs; i++) { order[i] = (முழு) i; }

        std::stable_sort(order.begin(), order.end(), [&](முழு a, முழு b) {
            திரும்பு h_bufs[a].size > h_bufs[b].size;
        });

        தருக்க already_sorted = உண்மை;
        க்கு (unsigned முழு i = 0; i < n_bufs; i++) {
            எனில் (order[i] != (முழு) i) {
                already_sorted = பொய்;
                நிறுத்து;
            }
        }
        எனில் (already_sorted) திரும்பு;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        க்கு (unsigned முழு new_bi = 0; new_bi < n_bufs; new_bi++) {
            முழு old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        க்கு (unsigned முழு i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        க்கு (unsigned முழு i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    வெறுமை update_mdev_group(uint32_t mdev_idx) {
        எனில் (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    தருக்க try_fuse_common(std::initializer_list<மாறிலி ggml_tensor *> tensors) மாறிலி {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        முழு seen_bufs[HTP_OP_MAX_BUFS];
        முழு n_seen_bufs = 0;

        க்கு (மாறிலி auto * t : tensors) {
            எனில் (!t || t_map.count(t)) {
                தொடர்;
            }
            extra_tens++;
            auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            முழு fd = sbuf->fd();
            எனில் (!b_map.count(fd)) {
                தருக்க found = பொய்;
                க்கு (முழு i = 0; i < n_seen_bufs; i++) {
                    எனில் (seen_bufs[i] == fd) {
                        found = உண்மை;
                        நிறுத்து;
                    }
                }
                எனில் (!found) {
                    எனில் (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    எனில் (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        எனில் ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            திரும்பு பொய்;
        }

        திரும்பு உண்மை;
    }

    தருக்க try_fuse_common(மாறிலி ggml_tensor * t1, மாறிலி ggml_tensor * t2) மாறிலி {
        திரும்பு try_fuse_common({t1, t2});
    }

    தருக்க try_fuse_allreduce_add(மாறிலி htp_opnode & முனை) {
        எனில் (n_ops == 0 || opt_ar_select != 2) திரும்பு பொய்;
        எனில் (முனை.opcode != HTP_OP_ADD) திரும்பு பொய்;

        htp_opnode & last_முனை = ops[n_ops - 1];
        எனில் (last_முனை.opcode != HTP_OP_ALLREDUCE) திரும்பு பொய்;

        auto * ar_kparams = (கட்டமைப்பு htp_allreduce_kernel_params *) last_முனை.kernel_params;
        மாறிலி uint32_t rank    = (uint32_t) ar_kparams->rank;
        மாறிலி uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        மாறிலி ggml_tensor * ar_local = last_முனை.inputs[rank];
        மாறிலி ggml_tensor * add_src0 = முனை.src0();
        மாறிலி ggml_tensor * add_src1 = முனை.src1();
        மாறிலி ggml_tensor * add_dst  = முனை.dst();

        எனில் (!ggml_hexagon_tensor_is_fuseable(ar_local)) திரும்பு பொய்;

        மாறிலி ggml_tensor * res_tensor;
        எனில் (add_src0 == ar_local || add_src0->data == ar_local->data) {
            res_tensor = add_src1;
        } இல்லையெனில் எனில் (add_src1 == ar_local || add_src1->data == ar_local->data) {
            res_tensor = add_src0;
        } இல்லையெனில் {
            திரும்பு பொய்;
        }

        எனில் (ar_local->type != res_tensor->type) திரும்பு பொய்;

        மாறிலி தருக்க is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        மாறிலி தருக்க is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        எனில் (!is_same_shape && !is_row_bcast) திரும்பு பொய்;

        எனில் (is_same_shape) {
            எனில் (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                திரும்பு பொய்;
            }
            எனில் (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                திரும்பு பொய்;
            }
        }
        எனில் (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) {
            திரும்பு பொய்;
        }

        // scatter is only valid for in-place add within max outputs
        மாறிலி தருக்க is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

        கட்டமைப்பு htp_allreduce_kernel_params new_kparams;
        எனில் (!ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, உண்மை, is_row_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            திரும்பு பொய்;
        }

        மாறிலி தருக்க scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

        க்கு (uint32_t r = 0; r < n_ranks; r++) {
            மாறிலி ggml_tensor * ar_src = last_முனை.inputs[r];
            எனில் (!ggml_hexagon_tensors_overlap(add_dst, ar_src)) {
                தொடர்;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            எனில் (scatter_ok && r == rank) {
                தொடர்;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            திரும்பு பொய்;
        }

        எனில் (!try_fuse_common(res_tensor, add_dst)) {
            திரும்பு பொய்;
        }

        last_முனை.opcode = HTP_OP_ALLREDUCE_ADD;
        last_முனை.name   = "ALLREDUCE+ADD";
        last_முனை.inputs.push_back(res_tensor);
        last_முனை.outputs.clear();
        last_முனை.outputs.push_back(add_dst);
        last_முனை.fused.push_back(முனை.முனை);
        memcpy(last_முனை.kernel_params, &new_kparams, அளவு(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ADD;
        memcpy(o.kernel_params, &new_kparams, அளவு(new_kparams));

        o.src[2 * n_ranks] = add_tensor(res_tensor);
        எனில் (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            க்கு (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            க்கு (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } இல்லையெனில் {
            o.dst[0] = add_tensor(add_dst);
            க்கு (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (முழு) new_kparams.mode, (முழு) new_kparams.n_dsts);
        திரும்பு உண்மை;
    }

    தருக்க try_fuse_rms_norm_mul(மாறிலி htp_opnode & முனை) {
        எனில் (n_ops == 0) திரும்பு பொய்;
        எனில் (முனை.opcode != HTP_OP_MUL) திரும்பு பொய்;

        htp_opnode & last_முனை = ops[n_ops - 1];
        எனில் (last_முனை.opcode != HTP_OP_RMS_NORM) திரும்பு பொய்;

        மாறிலி ggml_tensor * mul_src0 = முனை.src0();
        மாறிலி ggml_tensor * mul_src1 = முனை.src1();
        மாறிலி ggml_tensor * rms_out  = last_முனை.dst();

        எனில் (!ggml_hexagon_tensor_is_fuseable(rms_out)) திரும்பு பொய்;

        மாறிலி ggml_tensor * weight;
        எனில் (mul_src0 == rms_out || mul_src0->data == rms_out->data) {
            weight = mul_src1;
        } இல்லையெனில் எனில் (mul_src1 == rms_out || mul_src1->data == rms_out->data) {
            weight = mul_src0;
        } இல்லையெனில் {
            திரும்பு பொய்;
        }

        மாறிலி ggml_tensor * src0 = last_முனை.src0();

        எனில் (src0->ne[0] != weight->ne[0] || src0->ne[0] != முனை.dst()->ne[0]) {
            திரும்பு பொய்;
        }

        மாறிலி தருக்க is_row_bcast = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
        மாறிலி தருக்க is_same_shape = (src0->ne[0] == weight->ne[0] && src0->ne[1] == weight->ne[1] &&
                                    src0->ne[2] == weight->ne[2] && src0->ne[3] == weight->ne[3]);
        எனில் (!is_row_bcast && !is_same_shape) திரும்பு பொய்;

        எனில் (!ggml_are_same_shape(src0, முனை.dst())) {
            திரும்பு பொய்;
        }
        எனில் (ggml_is_contiguous(src0) != ggml_is_contiguous(முனை.dst())) {
            திரும்பு பொய்;
        }

        கட்டமைப்பு htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, weight, முனை.dst(), &new_kparams
        );

        எனில் ((size_t) new_kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_size, sess->vtcm_size);
            திரும்பு பொய்;
        }

        எனில் (!try_fuse_common(weight, முனை.dst())) {
            திரும்பு பொய்;
        }

        last_முனை.opcode = HTP_OP_RMS_NORM_MUL;
        last_முனை.name   = "RMS_NORM+MUL";
        last_முனை.inputs.clear();
        last_முனை.inputs.push_back(src0);
        last_முனை.inputs.push_back(weight);
        last_முனை.outputs.clear();
        last_முனை.outputs.push_back(முனை.dst());
        last_முனை.fused.push_back(முனை.முனை);
        memcpy(last_முனை.kernel_params, &new_kparams, அளவு(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        memcpy(o.kernel_params, &new_kparams, அளவு(new_kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(weight);
        க்கு (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(முனை.dst());
        க்கு (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        திரும்பு உண்மை;
    }

    தருக்க try_fuse_mul_mat_add(மாறிலி htp_opnode & முனை) {
        எனில் (n_ops == 0) திரும்பு பொய்;
        எனில் (முனை.opcode != HTP_OP_ADD) திரும்பு பொய்;

        htp_opnode & last_முனை = ops[n_ops - 1];
        எனில் (last_முனை.opcode != HTP_OP_MUL_MAT) திரும்பு பொய்;

        மாறிலி ggml_tensor * add_src0 = முனை.src0();
        மாறிலி ggml_tensor * add_src1 = முனை.src1();
        மாறிலி ggml_tensor * mm_out   = last_முனை.dst();

        எனில் (!ggml_hexagon_tensor_is_fuseable(mm_out)) திரும்பு பொய்;

        மாறிலி ggml_tensor * src2;
        எனில் (add_src0 == mm_out || add_src0->data == mm_out->data) {
            src2 = add_src1;
        } இல்லையெனில் எனில் (add_src1 == mm_out || add_src1->data == mm_out->data) {
            src2 = add_src0;
        } இல்லையெனில் {
            திரும்பு பொய்;
        }

        மாறிலி ggml_tensor * src0 = last_முனை.src0();
        மாறிலி ggml_tensor * src1 = last_முனை.src1();

        எனில் (src2->type != GGML_TYPE_F32) திரும்பு பொய்;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        எனில் (!ggml_are_same_shape(mm_out, முனை.dst())) திரும்பு பொய்;

        மாறிலி கட்டமைப்பு htp_mm_kernel_params * orig_kparams = (மாறிலி கட்டமைப்பு htp_mm_kernel_params *) last_முனை.kernel_params;
        கட்டமைப்பு htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_add_params(sess, src0, src1, src2, முனை.dst(), &kparams);
        எனில் (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            திரும்பு பொய்;
        }

        எனில் (orig_kparams->collapse != kparams.collapse) {
            திரும்பு பொய்;
        }

        மாறிலி முழு src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        மாறிலி தருக்க can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        எனில் (!can_fuse) திரும்பு பொய்;

        எனில் ((size_t) kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
            திரும்பு பொய்;
        }

        எனில் (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            எனில் (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_threads < orig_kparams->n_act_threads) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_threads, kparams.n_act_threads);
                திரும்பு பொய்;
            }
        }

        எனில் (!try_fuse_common(src2, முனை.dst())) {
            திரும்பு பொய்;
        }

        last_முனை.opcode = HTP_OP_MUL_MAT_ADD;
        last_முனை.name   = "MUL_MAT+ADD";
        last_முனை.inputs.clear();
        last_முனை.inputs.push_back(src0);
        last_முனை.inputs.push_back(src1);
        last_முனை.inputs.push_back(src2);
        last_முனை.outputs.clear();
        last_முனை.outputs.push_back(முனை.dst());
        last_முனை.fused.push_back(முனை.முனை);
        memcpy(last_முனை.kernel_params, &kparams, அளவு(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ADD;
        memcpy(o.kernel_params, &kparams, அளவு(kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(src1);
        o.src[2] = add_tensor(src2);
        க்கு (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(முனை.dst());
        க்கு (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        திரும்பு உண்மை;
    }

    தருக்க try_fuse_mul_mat_nx(மாறிலி htp_opnode & முனை) {
        எனில் (n_ops == 0 || முனை.opcode != HTP_OP_MUL_MAT) திரும்பு பொய்;
        எனில் (!is_mergeable_mul_mat(முனை.முனை)) திரும்பு பொய்;

        மாறிலி ggml_tensor * w_in = முனை.src0();
        மாறிலி ggml_tensor * x_in = முனை.src1();
        மாறிலி ggml_tensor * d_in = முனை.dst();

        htp_opnode & last_முனை = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        எனில் (last_முனை.opcode == HTP_OP_MUL_MAT_NX) {
            மாறிலி uint32_t curr_n = (uint32_t) last_முனை.outputs.size();
            எனில் (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                திரும்பு பொய்;
            }

            மாறிலி ggml_tensor * w0 = last_முனை.inputs[0];
            மாறிலி ggml_tensor * x  = last_முனை.inputs[curr_n];

            எனில் (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                திரும்பு பொய்;
            }
            எனில் (!last_முனை.fused.empty() && (mm_is_hmx_eligible(last_முனை.fused[0]) != mm_is_hmx_eligible(முனை.முனை))) {
                திரும்பு பொய்;
            }

            மாறிலி கட்டமைப்பு htp_mm_kernel_params * orig_kparams = (மாறிலி கட்டமைப்பு htp_mm_kernel_params *) last_முனை.kernel_params;
            மாறிலி தருக்க collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_in, x, d_in);
            எனில் (orig_kparams->collapse && !collapse) {
                திரும்பு பொய்;
            }

            கட்டமைப்பு htp_mm_kernel_params kparams;
            எனில் (collapse) {
                மாறிலி ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } இல்லையெனில் {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            எனில் (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                திரும்பு பொய்;
            }
            எனில் ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                திரும்பு பொய்;
            }

            எனில் (!try_fuse_common(w_in, d_in)) {
                திரும்பு பொய்;
            }

            last_முனை.inputs[curr_n] = w_in;
            last_முனை.inputs.push_back(x);
            last_முனை.outputs.push_back(d_in);
            last_முனை.fused.push_back(முனை.முனை);
            memcpy(last_முனை.kernel_params, &kparams, அளவு(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, அளவு(kparams));

            க்கு (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = add_tensor(last_முனை.inputs[s]);
            }
            க்கு (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            க்கு (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_முனை.outputs[d]);
            }
            க்கு (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            திரும்பு உண்மை;
        }

        // Case 2: last_node is single MUL_MAT
        எனில் (last_முனை.opcode == HTP_OP_MUL_MAT) {
            எனில் (!is_mergeable_mul_mat_pair(last_முனை.முனை, முனை.முனை)) {
                திரும்பு பொய்;
            }

            மாறிலி ggml_tensor * w0 = last_முனை.src0();
            மாறிலி ggml_tensor * x  = last_முனை.src1();
            மாறிலி ggml_tensor * w1 = முனை.src0();
            மாறிலி ggml_tensor * dst_0 = last_முனை.dst();
            மாறிலி ggml_tensor * dst_1 = முனை.dst();

            மாறிலி கட்டமைப்பு htp_mm_kernel_params * orig_kparams = (மாறிலி கட்டமைப்பு htp_mm_kernel_params *) last_முனை.kernel_params;
            மாறிலி தருக்க collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            எனில் (orig_kparams->collapse && !collapse) {
                திரும்பு பொய்;
            }

            கட்டமைப்பு htp_mm_kernel_params kparams;
            எனில் (collapse) {
                மாறிலி ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } இல்லையெனில் {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            எனில் (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                திரும்பு பொய்;
            }
            எனில் ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                திரும்பு பொய்;
            }

            எனில் (!try_fuse_common(w1, முனை.dst())) {
                திரும்பு பொய்;
            }

            last_முனை.opcode = HTP_OP_MUL_MAT_NX;
            last_முனை.name   = "MUL_MAT_NX";
            last_முனை.inputs.clear();
            last_முனை.inputs.push_back(w0);
            last_முனை.inputs.push_back(w1);
            last_முனை.inputs.push_back(x);
            last_முனை.outputs.clear();
            last_முனை.outputs.push_back(dst_0);
            last_முனை.outputs.push_back(dst_1);
            last_முனை.fused.push_back(முனை.முனை);
            memcpy(last_முனை.kernel_params, &kparams, அளவு(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            memcpy(o.kernel_params, &kparams, அளவு(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            க்கு (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            க்கு (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            திரும்பு உண்மை;
        }

        திரும்பு பொய்;
    }

    தருக்க try_fuse_mul_mat_id_nx(மாறிலி htp_opnode & முனை) {
        எனில் (n_ops == 0 || முனை.opcode != HTP_OP_MUL_MAT_ID) திரும்பு பொய்;
        எனில் (!is_mergeable_mul_mat_id(முனை.முனை)) திரும்பு பொய்;

        மாறிலி ggml_tensor * w_in   = முனை.src0();
        மாறிலி ggml_tensor * x_in   = முனை.src1();
        மாறிலி ggml_tensor * ids_in = முனை.முனை->src[2];
        மாறிலி ggml_tensor * d_in   = முனை.dst();

        htp_opnode & last_முனை = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        எனில் (last_முனை.opcode == HTP_OP_MUL_MAT_ID_NX) {
            மாறிலி uint32_t curr_n = (uint32_t) last_முனை.outputs.size();
            எனில் (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                திரும்பு பொய்;
            }

            மாறிலி ggml_tensor * w0  = last_முனை.inputs[0];
            மாறிலி ggml_tensor * x   = last_முனை.inputs[curr_n];
            மாறிலி ggml_tensor * ids = last_முனை.inputs[curr_n + 1];

            எனில் (x_in != x || ids_in != ids || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                திரும்பு பொய்;
            }
            எனில் (!last_முனை.fused.empty() && (mm_is_hmx_eligible(last_முனை.fused[0]) != mm_is_hmx_eligible(முனை.முனை))) {
                திரும்பு பொய்;
            }

            கட்டமைப்பு htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
            எனில் (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                திரும்பு பொய்;
            }
            எனில் ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                திரும்பு பொய்;
            }

            எனில் (!try_fuse_common(w_in, d_in)) {
                திரும்பு பொய்;
            }

            last_முனை.inputs[curr_n] = w_in;
            last_முனை.inputs[curr_n + 1] = x;
            last_முனை.inputs.push_back(ids);
            last_முனை.outputs.push_back(d_in);
            last_முனை.fused.push_back(முனை.முனை);
            memcpy(last_முனை.kernel_params, &kparams, அளவு(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, அளவு(kparams));

            க்கு (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = add_tensor(last_முனை.inputs[s]);
            }
            க்கு (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            க்கு (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_முனை.outputs[d]);
            }
            க்கு (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            திரும்பு உண்மை;
        }

        // Case 2: last_node is single MUL_MAT_ID
        எனில் (last_முனை.opcode == HTP_OP_MUL_MAT_ID) {
            எனில் (!is_mergeable_mul_mat_id_pair(last_முனை.முனை, முனை.முனை)) {
                திரும்பு பொய்;
            }

            மாறிலி ggml_tensor * w0  = last_முனை.src0();
            மாறிலி ggml_tensor * x   = last_முனை.src1();
            மாறிலி ggml_tensor * ids = last_முனை.முனை->src[2];
            மாறிலி ggml_tensor * w1  = முனை.src0();

            கட்டமைப்பு htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, முனை.dst(), 2, &kparams);
            எனில் (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                திரும்பு பொய்;
            }
            எனில் ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                திரும்பு பொய்;
            }

            எனில் (!try_fuse_common(w1, முனை.dst())) {
                திரும்பு பொய்;
            }

            மாறிலி ggml_tensor * dst_0 = last_முனை.dst();
            மாறிலி ggml_tensor * dst_1 = முனை.dst();

            last_முனை.opcode = HTP_OP_MUL_MAT_ID_NX;
            last_முனை.name   = "MUL_MAT_ID_NX";
            last_முனை.inputs.clear();
            last_முனை.inputs.push_back(w0);
            last_முனை.inputs.push_back(w1);
            last_முனை.inputs.push_back(x);
            last_முனை.inputs.push_back(ids);
            last_முனை.outputs.clear();
            last_முனை.outputs.push_back(dst_0);
            last_முனை.outputs.push_back(dst_1);
            last_முனை.fused.push_back(முனை.முனை);
            memcpy(last_முனை.kernel_params, &kparams, அளவு(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            memcpy(o.kernel_params, &kparams, அளவு(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            o.src[3] = add_tensor(ids);
            க்கு (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            க்கு (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            திரும்பு உண்மை;
        }

        திரும்பு பொய்;
    }

    தருக்க try_fuse_gdn_cpy(மாறிலி htp_opnode & முனை) {
        எனில் (n_ops == 0 || முனை.opcode != HTP_OP_CPY) திரும்பு பொய்;

        htp_opnode & last_முனை = ops[n_ops - 1];
        எனில் (last_முனை.opcode != HTP_OP_GATED_DELTA_NET) திரும்பு பொய்;
        எனில் (last_முனை.outputs.size() != 1) திரும்பு பொய்;

        மாறிலி ggml_tensor * gdn_out  = last_முனை.dst();
        மாறிலி ggml_tensor * cpy_முனை = முனை.முனை;
        மாறிலி ggml_tensor * cpy_src  = முனை.src0();
        மாறிலி ggml_tensor * cpy_dst  = முனை.dst();

        எனில் (!cpy_src || !cpy_dst || !cpy_dst->data) திரும்பு பொய்;
        எனில் (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) திரும்பு பொய்;
        எனில் ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_முனை->flags & GGML_TENSOR_FLAG_OUTPUT)) திரும்பு பொய்;

        மாறிலி ggml_tensor * v = last_முனை.முனை->src[2];
        எனில் (!v) திரும்பு பொய்;

        மாறிலி int64_t S_v      = v->ne[0];
        மாறிலி int64_t H        = v->ne[1];
        மாறிலி int64_t n_tokens = v->ne[2];
        மாறிலி int64_t n_seqs   = v->ne[3];
        மாறிலி int64_t K        = ggml_get_op_params_i32(last_முனை.முனை, 0);
        மாறிலி size_t  tail_off = (size_t) S_v * H * n_tokens * n_seqs * அளவு(தசம);

        மாறிலி int64_t D         = S_v * S_v * H;
        மாறிலி int64_t n_written = std::min<int64_t>(n_tokens, K);

        எனில் (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_out && cpy_src->view_src->data != gdn_out->data) ||
            cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
            திரும்பு பொய்;
        }

        எனில் (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != அளவு(தசம)) {
            திரும்பு பொய்;
        }
        எனில் (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * அளவு(தசம)) {
            திரும்பு பொய்;
        }
        எனில் (n_written > 1) {
            எனில் (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * அளவு(தசம)) {
                திரும்பு பொய்;
            }
        }

        எனில் (!try_fuse_common({cpy_dst})) {
            திரும்பு பொய்;
        }

        last_முனை.name += "+CPY";
        last_முனை.outputs.push_back(cpy_dst);
        last_முனை.fused.push_back(முனை.முனை);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = add_tensor(cpy_dst);
        க்கு (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        திரும்பு உண்மை;
    }

    தருக்க try_fuse(மாறிலி htp_opnode & முனை) {
        எனில் (!opt_opfusion) திரும்பு பொய்;
        எனில் (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ADD) && try_fuse_allreduce_add(முனை)) திரும்பு உண்மை;
        எனில் (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(முனை))  திரும்பு உண்மை;
        எனில் (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ADD)   && try_fuse_mul_mat_add(முனை))   திரும்பு உண்மை;
        எனில் (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(முனை))    திரும்பு உண்மை;
        எனில் (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(முனை)) திரும்பு உண்மை;
        எனில் (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(முனை))       திரும்பு உண்மை;
        திரும்பு பொய்;
    }
};

கட்டமைப்பு ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

கட்டமைப்பு ggml_hexagon_opqueue {
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
        எனில் (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * அளவு(htp_trace_desc);
        }

        shm_blk_size = அளவு(htp_buf_desc)  * n_bufs    +
                       அளவு(htp_tensor)    * n_tensors +
                       அளவு(htp_op_desc)   * n_ops     +
                       அளவு(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = புதிய ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, உண்மை /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        எனில் (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size(), shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    size_t shm_size() மாறிலி { திரும்பு shm_buf ? shm_buf->size() : 0; }

    // push new batch
    தருக்க தள்ளு(htp_opbatch_req& req, dspqueue_buffer& dbuf, மாறிலி ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(அளவு(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(அளவு(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(அளவு(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(அளவு(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(அளவு(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(அளவு(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        எனில் (seq - shm_buf->sess->batch_rsp_seq > depth) { திரும்பு பொய்; }

        மாறிலி uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[slot]   = op_batch->ops;
        start_usec[slot] = ggml_time_us();

        மாறிலி size_t b_size = அளவு(htp_buf_desc)  * req.n_bufs;
        மாறிலி size_t t_size = அளவு(htp_tensor)    * req.n_tensors;
        மாறிலி size_t o_size = அளவு(htp_op_desc)   * req.n_ops;
        மாறிலி size_t p_size = அளவு(htp_prof_desc) * req.n_ops;

        size_t tr_size = 0;
        எனில் (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * அளவு(htp_trace_desc);
        } இல்லையெனில் {
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

        memcpy(b_ptr, (வெறுமை *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (வெறுமை *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (வெறுமை *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        எனில் (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            க்கு (unsigned முழு i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (வெறுமை *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            க்கு (unsigned முழு i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (unsigned long long) t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        திரும்பு உண்மை;
    }

    வெறுமை எடு(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        மாறிலி uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        மாறிலி size_t b_size = அளவு(htp_buf_desc)  * rsp.n_bufs;
        மாறிலி size_t t_size = அளவு(htp_tensor)    * rsp.n_tensors;
        மாறிலி size_t o_size = அளவு(htp_op_desc)   * rsp.n_ops;
        மாறிலி size_t p_size = அளவு(htp_prof_desc) * rsp.n_ops;

        size_t tr_size = 0;
        uint32_t n_traces = 0;
        எனில் (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * அளவு(htp_trace_desc);
        }

        மாறிலி size_t m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        எனில் (rsp.n_ops > 0) {
            auto & ops = op_cache[slot];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            மாறிலி htp_prof_desc * pd = (மாறிலி htp_prof_desc *) p_ptr;
            மாறிலி htp_trace_desc * trace_events = nullptr;
            எனில் (opt_profile == 3) {
                trace_events = (மாறிலி htp_trace_desc *) (p_ptr + p_size);
            }

            எனில் (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            க்கு (uint32_t i = 0; i < rsp.n_ops; i++) {
                எனில் (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            எனில் (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

வெறுமை ggml_hexagon_session::flush_peers() {
    auto vpeers = std::move(virt_peers);
    virt_peers.clear();
    க்கு (auto * peer : vpeers) {
        peer->flush_sync();
    }

    auto ppeers = std::move(phys_peers);
    phys_peers.clear();
    க்கு (auto * peer : ppeers) {
        peer->flush_async();
    }

    க்கு (auto & sub : இது->mdev.sessions) {
        sub->flush_peers();
    }
}

வெறுமை ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

வெறுமை ggml_hexagon_session::flush_pending(தருக்க all) {
    க்கு (auto & sub : இது->mdev.sessions) {
        sub->flush_pending(all);
        எனில் (sub->last_error > HTP_STATUS_OK) {
            இது->last_error = sub->last_error.load();
        }
    }

    போது (இது->batch_rsp_seq < இது->batch_req_seq) {
        கட்டமைப்பு htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        கட்டமைப்பு dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        மாறிலி uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        முழு err = dspqueue_read(இது->queue, &flags, 1, &n_dbufs, &dbuf, அளவு(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        எனில் (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            தொடர்;
        }

        எனில் (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        எனில் (rsp_size != அளவு(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", இது->c_name(), rsp_size, n_dbufs);
        }

        எனில் (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", இது->c_name(), status_to_str(rsp.status));
            இது->last_error = rsp.status;
            க்கு (auto & sub : இது->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->எடு(rsp, dbuf);

        GGML_ASSERT(rsp.seq == இது->batch_rsp_seq + 1);
        இது->batch_rsp_seq = rsp.seq;

        எனில் (!all) நிறுத்து;
    }
}

வெறுமை ggml_hexagon_session::flush_sync(தருக்க all) {
    flush_async();
    flush_pending(all);
}

வெறுமை ggml_hexagon_session::start_batch() {
    எனில் (இது->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

வெறுமை ggml_hexagon_session::flush_batch(size_t min_ops) {
    எனில் (op_batch->n_ops < min_ops) { திரும்பு; }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    மாறிலி uint64_t seq = ++இது->batch_req_seq;

    op_batch->update_mdev_group(இது->mdev.idx);

    எனில் (!op_queue->தள்ளு(req, dbuf, op_batch, seq)) {
        flush_pending(பொய்);
        op_queue->தள்ளு(req, dbuf, op_batch, seq);
    }

    க்கு (auto & sub : இது->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_buffer sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        எனில் (!sub->op_queue->தள்ளு(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(பொய்);
            sub->op_queue->தள்ளு(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        முழு err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, அளவு(sub_req), (மாறிலி uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        எனில் (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (unsigned) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", இது->c_name(), dbuf.ptr, dbuf.size);

    முழு err = dspqueue_write(இது->queue, 0, 1, &dbuf, அளவு(req), (மாறிலி uint8_t*) &req, DSPQUEUE_TIMEOUT);
    எனில் (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", இது->c_name(), (unsigned) err);
    }

    op_batch->reset();
}

வெறுமை ggml_hexagon_session::enqueue_op(மாறிலி htp_opnode & முனை) {
    auto clone_tensor_buffer = [இது](மாறிலி ggml_tensor * t) {
        auto sbuf = இது->mmap_tensor(t);
        எனில் (!sbuf) திரும்பு;
        எனில் (sbuf->sess != இது) {
            இது->clone_buffer(sbuf);
        }
        க்கு (auto & sub : இது->mdev.sessions) {
            sub->clone_buffer(sbuf);
        }
    };

    க்கு (auto t : முனை.get_inputs()) {
        clone_tensor_buffer(t);
    }
    க்கு (auto t : முனை.get_outputs()) {
        clone_tensor_buffer(t);
    }

    எனில் (opt_opfusion && op_batch->try_fuse(முனை)) {
        திரும்பு;
    }

    எனில் (!op_batch->fit_op(முனை)) {
        flush_async();
    }

    எனில் (op_batch->empty()) {
        start_batch();
    }

    எனில் (!op_batch->fit_op(முனை)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->add_op(முனை);
}

வெறுமை ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_முனை(HTP_OP_MDEV_GROUP);

    uint8_t * fence_slot = இது->mdev_fence_slot;

    நிலையான ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.buffer       = &இது->fence_buf->backend_buffer;
    dummy_t.extra        = &fence_extra;
    dummy_t.data         = (வெறுமை *) fence_slot;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.ne[1]        = (int64_t) இது->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) இது->mdev.idx;

    ggml_tensor * முனை = group_முனை.add_dummy(dummy_t);
    முனை->src[0] = முனை;
    group_முனை.init(முனை);
    group_முனை.outputs.clear();
    group_முனை.name = "MDEV_GROUP";

    க்கு (auto & sub : இது->mdev.sessions) {
        sub->clone_buffer(இது->fence_buf);
    }

    op_batch->add_op(group_முனை);
}

வெறுமை ggml_hexagon_session::enqueue_cpy(மாறிலி ggml_tensor * src, ggml_tensor * dst, மாறிலி ggml_tensor * sync_tensor, uint32_t fence_seq) {
    மாறிலி தருக்க with_fence = sync_tensor != nullptr;
    htp_opnode cpy_முனை(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* முனை = cpy_முனை.add_dummy(*dst);
    முனை->op     = GGML_OP_CPY;
    முனை->src[0] = const_cast<ggml_tensor *>(src);
    முனை->src[1] = with_fence ? cpy_முனை.add_dummy(*sync_tensor) : nullptr;
    எனில் (with_fence) {
        முனை->op_params[0] = (int32_t) fence_seq;
    }

    cpy_முனை.init(முனை);
    எனில் (with_fence) {
        cpy_முனை.name = "CPY+FENCE";
    }
    மாறிலி தருக்க ok = ggml_hexagon_precompute_cpy_params(இது, முனை, (கட்டமைப்பு htp_copy_kernel_params *) cpy_முனை.kernel_params);
    மாறிலி auto * kparams = (மாறிலி கட்டமைப்பு htp_copy_kernel_params *) cpy_முனை.kernel_params;
    எனில் (ok && !with_fence && kparams->total_elems == 0) {
        திரும்பு;
    }
    இது->enqueue_op(cpy_முனை);
}

வெறுமை ggml_hexagon_session::enqueue_fence(மாறிலி ggml_tensor * sync_tensor, uint32_t fence_seq, தருக்க wait) {
    htp_opnode sync_முனை(HTP_OP_FENCE);

    ggml_tensor* முனை = sync_முனை.add_dummy(*sync_tensor);
    முனை->op           = GGML_OP_NONE;
    முனை->src[0]       = முனை;
    முனை->op_params[0] = (int32_t) fence_seq;
    முனை->op_params[1] = wait ? 0 : 1;

    sync_முனை.init(முனை);
    sync_முனை.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    இது->enqueue_op(sync_முனை);
}

நிலையான தருக்க ggml_hexagon_precompute_allreduce_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    தருக்க has_add,
    தருக்க is_row_bcast,
    தருக்க is_shard_ok,
    கட்டமைப்பு htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    மாறிலி uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    மாறிலி uint32_t elem_size = (dst->type == GGML_TYPE_F16) ? அளவு(ggml_fp16_t) : அளவு(தசம);
    மாறிலி தருக்க is_contiguous = ggml_is_contiguous(dst);

    மாறிலி uint32_t ne0 = (uint32_t) dst->ne[0];
    மாறிலி uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    மாறிலி தருக்க use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    எனில் (has_add) {
        // sharded reduce-scatter for contiguous in-place add
        எனில் (opt_ar_scatter && is_shard_ok && use_1d && !is_row_bcast) {
            மாறிலி uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            மாறிலி uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            மாறிலி uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_start = (int32_t) rank_elem_start;
            kparams->rank_nelem      = (int32_t) (rank_elem_end - rank_elem_start);
        } இல்லையெனில் எனில் (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } இல்லையெனில் {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } இல்லையெனில் {
        kparams->n_dsts = (int32_t) n_ranks;
        எனில் (use_1d) {
            மாறிலி uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            மாறிலி uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            மாறிலி uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            மாறிலி uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } இல்லையெனில் {
            மாறிலி uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            மாறிலி uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            மாறிலி uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            மாறிலி uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    எனில் (use_1d) {
        மாறிலி uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        மாறிலி uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        மாறிலி size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_மொத்தம்(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        எனில் (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        போது ((size_t) kparams->vtcm_size > sess->vtcm_size && block_elems > 128) {
            மாறிலி size_t max_bytes_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((size_t) (max_bytes_per_buf / elem_size), 128);
            எனில் (block_elems < 128) நிறுத்து;
            kparams->block_elems          = block_elems;
            kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
        }

        எனில் (sess->vtcm_size < (size_t) kparams->vtcm_size || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            திரும்பு பொய்;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        திரும்பு உண்மை;
    } இல்லையெனில் {
        மாறிலி uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        மாறிலி uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        மாறிலி size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_மொத்தம்(n_ranks, n_threads, has_add, is_row_bcast);

        மாறிலி uint32_t row_bytes = ne0 * elem_size;
        மாறிலி uint32_t row_size_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_size_aligned = row_size_aligned;

        மாறிலி uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        போது ((size_t) kparams->vtcm_size > sess->vtcm_size && block_rows > 1) {
            மாறிலி size_t max_rows_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2 * row_size_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
            எனில் (max_rows_per_buf == 0) நிறுத்து;
        }

        எனில் (sess->vtcm_size < (size_t) kparams->vtcm_size || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            திரும்பு பொய்;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        திரும்பு உண்மை;
    }
}

வெறுமை ggml_hexagon_session::enqueue_allreduce(
    மாறிலி ggml_tensor * dst,
    மாறிலி std::vector<மாறிலி ggml_tensor *> & src_tensors,
    மாறிலி std::vector<மாறிலி ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_முனை(HTP_OP_ALLREDUCE);

    ggml_tensor* முனை = ar_முனை.add_dummy(*dst);
    முனை->op           = GGML_OP_NONE;
    முனை->op_params[0] = (int32_t) fence_seq_entry;
    முனை->op_params[1] = (int32_t) fence_seq_exit;

    ar_முனை.init(முனை);

    ar_முனை.inputs.clear();
    க்கு (size_t i = 0; i < src_tensors.size(); i++) {
        ar_முனை.inputs.push_back(src_tensors[i]);
    }
    க்கு (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_முனை.inputs.push_back(ar_முனை.add_dummy(*sync_tensors[i]));
    }

    ar_முனை.outputs.clear();
    க்கு (size_t i = 0; i < src_tensors.size(); i++) {
        ar_முனை.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        இது, dst, rank, n_ranks, பொய், பொய், /*is_shard_ok=*/ பொய்,
        (கட்டமைப்பு htp_allreduce_kernel_params *) ar_முனை.kernel_params);

    ar_முனை.name = "ALLREDUCE";
    இது->enqueue_op(ar_முனை);
}

ggml_hexagon_shared_buffer * ggml_hexagon_session::mmap_tensor(மாறிலி ggml_tensor * t) {
    எனில் (!t) திரும்பு nullptr;

    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
    எனில் (!sbuf->mapped) {
        மாறிலி தருக்க is_weight = ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        மாறிலி தருக்க extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    திரும்பு sbuf;
}

தருக்க ggml_hexagon_session::clone_buffer(மாறிலி ggml_hexagon_shared_buffer * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    எனில் (sbuf->sess == இது) திரும்பு உண்மை;

    auto mem = sbuf->mem;
    முழு   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    எனில் (இது->cloned_buffers.find(fd) != இது->cloned_buffers.end()) திரும்பு உண்மை;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", இது->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    auto clone = std::make_unique<ggml_hexagon_shared_buffer>(இது, *sbuf);
    முயல் {
        clone->mmap(sbuf->extended);
    } பிடி (மாறிலி std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", இது->c_name(), exc.what());
        திரும்பு பொய்;
    }

    இது->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(இது);
    திரும்பு உண்மை;
}

வெறுமை ggml_hexagon_session::release_buffer(மாறிலி ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    முழு   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    auto it = இது->cloned_buffers.find(fd);
    எனில் (it != இது->cloned_buffers.end()) {
        auto clone = std::move(it->second);
        இது->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(இது);
}

வெறுமை ggml_hexagon_session::unclone_buffer(மாறிலி ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    க்கு (auto * sess : sessions) {
        sess->release_buffer(sbuf);
    }
}

நிலையான size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    மாறிலி size_t MiB = 1024 * 1024;
    மாறிலி size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t step = 256u * MiB;

    முயல் {
        sbufs.push_back(புதிய ggml_hexagon_shared_buffer(sess, GiB, உண்மை)); vmem += GiB;
        sbufs.push_back(புதிய ggml_hexagon_shared_buffer(sess, GiB, உண்மை)); vmem += GiB;
        sbufs.push_back(புதிய ggml_hexagon_shared_buffer(sess, GiB, உண்மை)); vmem += GiB;

        போது (1) {
            sbufs.push_back(புதிய ggml_hexagon_shared_buffer(sess, step, உண்மை));
            vmem += step;
        }
    } பிடி (...) { }

    க்கு (auto b : sbufs) { delete b; }

    திரும்பு vmem - step; // backoff to account for overhead from internal mappings
}

வெறுமை ggml_hexagon_session::allocate(மாறிலி ggml_hexagon_device_config & config) noexcept(பொய்) {
    முழு phys_idx = config.physical_idx;
    முழு virt_idx = config.virtual_idx;

    இது->valid_session = பொய்;
    இது->valid_handle  = பொய்;
    இது->valid_queue   = பொய்;
    இது->valid_iface   = பொய்;

    இது->name          = config.name;
    இது->phys_idx      = phys_idx;
    இது->virt_idx      = virt_idx;
    இது->domain_id     = config.domain_id;
    இது->session_id    = 0;
    இது->batch_req_seq = 0;
    இது->batch_rsp_seq = 0;
    இது->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", இது->name.c_str(), இது->domain_id, phys_idx, virt_idx);

    எனில் (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        எறி std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    மாறிலி std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    எனில் (virt_idx > 0) {
        கட்டமைப்பு remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<எழுத்து *>(dom_name.c_str());
        n.session_name     = const_cast<எழுத்து *>(இது->name.c_str());
        n.session_name_len = இது->name.size();
        n.session_id       = virt_idx;

        முழு err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (வெறுமை *) &n, அளவு(n));
        எனில் (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           இது->c_name(), phys_idx, virt_idx, err);
            எறி std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        இது->session_id    = n.session_id;
        இது->domain_id     = n.effective_domain_id;
        இது->valid_session = உண்மை;
    } இல்லையெனில் {
        கட்டமைப்பு remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<எழுத்து *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        முழு err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (வெறுமை *) &eff, அளவு(eff));
        எனில் (err == AEE_SUCCESS) {
            இது->domain_id = eff.effective_domain_id;
        } இல்லையெனில் {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           இது->name.c_str(), err, இது->domain_id);
        }
    }

    // Enable unsigned modules
    {
        கட்டமைப்பு remote_rpc_control_unsigned_module u;
        u.domain = இது->domain_id;
        u.enable = 1;
        முழு err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (வெறுமை *) &u, அளவு(u));
        எனில் (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", இது->c_name(), err);
            எறி std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    எழுத்து session_uri[256];
    {
        எழுத்து htp_uri[256];
        snprintf(htp_uri, அளவு(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        கட்டமைப்பு remote_rpc_get_uri u = {};
        u.session_id      = இது->session_id;
        u.domain_name     = const_cast<எழுத்து *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<எழுத்து *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = அளவு(session_uri);

        முழு err = remote_session_control(FASTRPC_GET_URI, (வெறுமை *) &u, அளவு(u));
        எனில் (err != AEE_SUCCESS) {
            snprintf(session_uri, அளவு(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), இது->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          இது->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    முழு err = htp_iface_open(session_uri, &இது->handle);
    எனில் (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", இது->c_name(), session_uri, err);
        எறி std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    இது->valid_handle = உண்மை;

    // Query HW info and resolve session options
    {
        unsigned முழு hw_n_threads = 0;
        unsigned முழு hw_n_hvx     = 0;
        unsigned முழு hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        முழு hw_err = htp_iface_hwinfo(இது->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        எனில் (hw_err == 0) {
            மாறிலி uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            இது->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_threads) : max_n_threads;
            இது->n_hvx     = இது->n_threads;
            இது->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            இது->vtcm_size = (uint64_t)hw_vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          இது->c_name(), இது->n_threads, இது->n_hvx, இது->n_hmx,
                          (unsigned long long)(இது->vtcm_size / (1024 * 1024)));
        } இல்லையெனில் {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", இது->c_name(), hw_err);
            மாறிலி uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            இது->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_threads;
            இது->n_hvx     = இது->n_threads;
            இது->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            இது->vtcm_size = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        கட்டமைப்பு remote_rpc_control_latency l;
        l.enable = 1;

        முழு err = remote_handle64_control(இது->handle, DSPRPC_CONTROL_LATENCY, (வெறுமை *) &l, அளவு(l));
        எனில் (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", இது->c_name(),
                  இது->session_id, இது->domain_id, session_uri, (unsigned long) இது->handle);

    மாறிலி size_t req_q_size = (அளவு(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    மாறிலி size_t rsp_q_size = (அளவு(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(இது->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (வெறுமை *) இது,  // Callback context
                          &queue);
    எனில் (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", இது->name.c_str(), (unsigned) err);
        எறி std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    இது->valid_queue = உண்மை;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &இது->queue_id);
    எனில் (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        எறி std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    எனில் (opt_etm) {
        err = htp_iface_etm(இது->handle, 1);
        எனில் (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    இது->op_queue = புதிய ggml_hexagon_opqueue(இது, opt_opbatch, opt_opqueue);

    இது->fence_buf = புதிய ggml_hexagon_fence_buffer(இது, &dev_ctx->fence_buffer_type, 64 * 1024);
    எனில் (இது->mdev.count > 1) {
        இது->mdev_fence_slot = இது->alloc_fence(இது->mdev.count);
    }

    எனில் (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(இது);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", இது->c_name(), opt_vmem);
    }
    மாறிலி size_t shm_size = இது->op_queue->shm_size();
    இது->max_vmem = (opt_vmem > shm_size) ? (opt_vmem - shm_size) : opt_vmem;

    இது->op_batch = புதிய ggml_hexagon_opbatch(இது, opt_opbatch, இது->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(இது->handle, இது->session_id, இது->queue_id, இது->n_threads, opt_nhmx, இது->max_vmem, (uint32_t) opt_power_mode);
    எனில் (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", இது->c_name(), (unsigned) err);
        எறி std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    இது->valid_iface = உண்மை;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", இது->c_name(), htp_power_mode_to_str(opt_power_mode));

    எனில் (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(இது->handle, opt_profile, &pmu_conf);
        எனில் (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }
}

வெறுமை ggml_hexagon_session::release() noexcept(உண்மை) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", இது->name.c_str());

    இது->mdev.sessions.clear();

    முழு err;

    எனில் (இது->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(இது->handle);
        எனில் (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    delete இது->op_batch;
    delete இது->op_queue;
    க்கு (auto & it : இது->cpy_fence_slots) {
        free_fence((வெறுமை *) it.second, 1);
    }
    இது->cpy_fence_slots.clear();

    எனில் (இது->fence_buf) {
        unclone_buffer(இது->fence_buf);
        delete இது->fence_buf;
        இது->fence_buf = nullptr;
    }
    போது (!இது->cloned_buffers.empty()) {
        release_buffer(இது->cloned_buffers.begin()->second.get());
    }

    எனில் (opt_etm) {
        err = htp_iface_etm(இது->handle, 0);
        எனில் (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    எனில் (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(இது->handle, 0, &pmu_conf);
        எனில் (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    எனில் (இது->valid_queue) {
        err = dspqueue_close(queue);
        எனில் (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (unsigned) err);
        }
    }

    எனில் (இது->valid_handle) {
        htp_iface_close(இது->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(மாறிலி ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_மொத்தம்) noexcept(பொய்) {
    இது->dev        = dev;
    இது->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    இது->mdev.idx   = mdev_idx;
    இது->mdev.count = mdev_மொத்தம் > 0 ? mdev_மொத்தம் : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)இது) & 0xFFFF;

    முயல் {
        allocate(config);
        எனில் (இது->mdev.idx == 0 && !config.mdev_group.empty()) {
            க்கு (size_t i = 0; i < config.mdev_group.size(); i++) {
                இது->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], இது->dev, (uint32_t) (i + 1), இது->mdev.count));
            }
        }
    } பிடி (மாறிலி std::exception & exc) {
        release();
        எறி;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(உண்மை) {
    release();
}

// ** backend interface

நிலையான தருக்க ggml_hexagon_flash_attn_is_hmx_eligible(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * q,
    மாறிலி கட்டமைப்பு ggml_tensor * k,
    மாறிலி கட்டமைப்பு ggml_tensor * v,
    மாறிலி கட்டமைப்பு ggml_tensor * sinks
) {
    எனில் (sess->n_hmx == 0) {
        திரும்பு பொய்;
    }

    எனில் (opt_fa_select < 2) {
        திரும்பு பொய்;
    }

    எனில் ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t DK = q->ne[0];
    மாறிலி uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    எனில் (DK % 8 != 0 || DV % 8 != 0) {
        திரும்பு பொய்;
    }

    GGML_UNUSED(sinks);

    // Explicit force mode
    எனில் (opt_fa_select > 2) {
        திரும்பு உண்மை;
    }

    மாறிலி uint32_t M = q->ne[1];

    // Prefill or batched decode
    எனில் (M > 1) {
        திரும்பு உண்மை;
    }

    மாறிலி uint32_t n_head = q->ne[2];
    மாறிலி uint32_t n_kv_heads = k->ne[2];
    மாறிலி uint32_t G = n_kv_heads > 0 ? n_head / n_kv_heads : 1;
    மாறிலி uint32_t S = k->ne[1];
    மாறிலி uint32_t total_threads = sess->n_threads * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    எனில் (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < total_threads))) {
        திரும்பு உண்மை;
    }

    // Tile alignment for 32-row HMX tiles
    மாறிலி தருக்க is_tile_aligned = (G > 0 && (32 % G == 0));
    எனில் (!is_tile_aligned) {
        எனில் (sess->n_threads >= 6) {
            திரும்பு பொய்;
        }
        திரும்பு S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    எனில் (DK == 256) {
        s_cross = 4096;
    } இல்லையெனில் எனில் (DK <= 64) {
        s_cross = (sess->n_threads >= 8) ? 2048 : ((sess->n_threads >= 6) ? 768 : 512);
    } இல்லையெனில் {
        s_cross = (sess->n_threads >= 8) ? 1024 : 512;
    }

    திரும்பு S >= s_cross;
}

நிலையான தருக்க ggml_hexagon_precompute_flash_attn_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_fa_kernel_params * kparams
) {
    எனில் (opt_fa_select < 1) {
        திரும்பு பொய்;
    }

    memset(kparams, 0, அளவு(*kparams));

    மாறிலி கட்டமைப்பு ggml_tensor * q    = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * k    = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * v    = op->src[2];
    மாறிலி கட்டமைப்பு ggml_tensor * mask = op->src[3];

    மாறிலி uint32_t neq0 = q->ne[0];  // head_dim (DK)
    மாறிலி uint32_t neq1 = q->ne[1];  // n_tokens
    மாறிலி uint32_t neq2 = q->ne[2];  // n_heads

    மாறிலி uint32_t nek1 = k->ne[1];  // kv_len

    மாறிலி uint32_t nev0 = v->ne[0];  // head_dim (DV)

    மாறிலி uint32_t DK = neq0;
    மாறிலி uint32_t DV = nev0;

    மாறிலி uint32_t n_kv_heads = k->ne[2];
    மாறிலி uint32_t G          = neq2 / n_kv_heads;

    தசம scale         = 1.0f;
    தசம max_bias      = 0.0f;
    தசம logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], அளவு(தசம));
    memcpy(&max_bias,      &op->op_params[1], அளவு(தசம));
    memcpy(&logit_softcap, &op->op_params[2], அளவு(தசம));

    எனில் (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->head_split = (opt_fa_head_split != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    மாறிலி uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    மாறிலி கட்டமைப்பு ggml_tensor * sinks = op->src[4];
    எனில் (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        மாறிலி uint32_t DK_pad = hex_round_up(DK, 64);
        மாறிலி uint32_t DV_pad = hex_round_up(DV, 64);
        size_t Br = 0, Bc = 0;
        முழு ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
        எனில் (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            எனில் (logit_softcap == 0.0f) {
                kparams->scale = scale * (தசம) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } இல்லையெனில் {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (தசம) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);

            மாறிலி size_t row_vec_bytes = hex_align_up(Bc * அளவு(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            மாறிலி size_t m_வரி_bytes = hex_align_up(Bc * அளவு(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_வரி_bytes / அளவு(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_மதிப்புகள்(G);
            எனில் (mask) {
                kparams->src3_div2 = init_fastdiv_மதிப்புகள்(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_மதிப்புகள்(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            திரும்பு உண்மை;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    மாறிலி size_t size_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    மாறிலி size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    மாறிலி size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), mask != nullptr, sinks != nullptr, n_head, sess->n_threads);

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_மதிப்புகள்(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_மதிப்புகள்(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_மதிப்புகள்(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_மதிப்புகள்(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_மதிப்புகள்(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_மதிப்புகள்(q->ne[3]/v->ne[3]);
    எனில் (mask) {
        kparams->src3_div2 = init_fastdiv_மதிப்புகள்(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_மதிப்புகள்(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_hexagon_supported_flash_attn_ext(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * src2 = op->src[2];
    மாறிலி கட்டமைப்பு ggml_tensor * src3 = op->src[3];
    மாறிலி கட்டமைப்பு ggml_tensor * src4 = op->src[4];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    எனில் ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        திரும்பு பொய்;
    }

    எனில் (src3 && src3->type != GGML_TYPE_F16) {  // mask
        திரும்பு பொய்;
    }

    எனில் (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        திரும்பு பொய்;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    எனில் (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        திரும்பு பொய்;
    }

    கட்டமைப்பு htp_fa_kernel_params kparams;
    எனில் (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        திரும்பு பொய்;
    }

    எனில் ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_hexagon_supported_gated_delta_net(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    எனில் (opt_gdn_select < 1) {
        திரும்பு பொய்;
    }

    மாறிலி கட்டமைப்பு ggml_tensor * q     = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * k     = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * v     = op->src[2];
    மாறிலி கட்டமைப்பு ggml_tensor * g     = op->src[3];
    மாறிலி கட்டமைப்பு ggml_tensor * beta  = op->src[4];
    மாறிலி கட்டமைப்பு ggml_tensor * state = op->src[5];
    மாறிலி கட்டமைப்பு ggml_tensor * dst   = op;

    எனில் (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }

    மாறிலி int64_t S_v      = v->ne[0];
    மாறிலி int64_t H        = v->ne[1];
    மாறிலி int64_t n_tokens = v->ne[2];
    மாறிலி int64_t n_seqs   = v->ne[3];
    மாறிலி int64_t K        = ggml_get_op_params_i32(op, 0);

    எனில் (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        திரும்பு பொய்;
    }
    எனில் (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        திரும்பு பொய்;
    }
    எனில் ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        திரும்பு பொய்;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    எனில் (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        திரும்பு பொய்;
    }
    எனில் (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t total_rows = (uint32_t) (H * n_seqs);
    மாறிலி uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    மாறிலி தருக்க can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    எனில் (can_use_hmx) {
        கட்டமைப்பு htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        எனில் (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, உண்மை, &n_heads_batch)) {
            திரும்பு பொய்;
        }
    } இல்லையெனில் {
        கட்டமைப்பு htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_threads);
        எனில் (layout.total_bytes > sess->vtcm_size) {
            திரும்பு பொய்;
        }
    }

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_hexagon_matmul_is_hmx_eligible(
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    தருக்க is_matmul_id,
    தருக்க is_batched
) {
    GGML_UNUSED(dst);

    எனில் (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        திரும்பு பொய்;
    }

    மாறிலி முழு ne00  = src0->ne[0];
    மாறிலி முழு ne01  = src0->ne[1];
    மாறிலி முழு ne11  = src1->ne[1];
    மாறிலி முழு ne12  = src1->ne[2];
    மாறிலி முழு wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    எனில் (ne01 % HTP_TENSOR_TILE_N_ROWS != 0 && is_matmul_id) {
        திரும்பு பொய்;
    }

    // HMX supports F16, F32, and repack quantized types.
    எனில் (!htp_is_hmx_weight_type((ggml_type) wtype)) {
        திரும்பு பொய்;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    எனில் (ne00 % HTP_TENSOR_TILE_N_COLS != 0) {
        திரும்பு பொய்;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    மாறிலி தருக்க is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    எனில் (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        திரும்பு பொய்;
    }
    எனில் (!is_matmul_id && is_batched && is_tiled) {
        திரும்பு பொய்;
    }

    // HMX assumes contiguous row-major layout.
    எனில் (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        திரும்பு பொய்;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    மாறிலி முழு m = is_matmul_id ? ne12 : ne11;
    எனில் (m <= HTP_MM_HMX_MIN_NROWS) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;

    GGML_UNUSED(dst);
}

நிலையான தருக்க ggml_hexagon_precompute_hmx_mm_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * act,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    முழு wtype,
    முழு ne00_padded,
    முழு ne01_padded,
    முழு ne02,
    முழு ne11,
    முழு ne12,
    முழு ne11_padded,
    தருக்க is_matmul_id,
    தருக்க is_batched,
    size_t src2_size,
    size_t vtcm_budget,
    கட்டமைப்பு htp_mm_kernel_params * kparams
) {
    மாறிலி தருக்க is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    மாறிலி முழு aligned_tile_size = is_tiled ? htp_tensor_get_tile_padded_size(wtype) : 0;
    மாறிலி முழு n_threads = (முழு)sess->n_threads;
    மாறிலி முழு ne10 = act->ne[0];

    முழு m_for_solver = ne11;
    முழு m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    எனில் (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        மாறிலி தருக்க dst_row_split = dst ? ggml_hexagon_can_row_partition(dst) : உண்மை;
        மாறிலி தருக்க act_row_split = ggml_hexagon_can_row_partition(act);
        எனில் (dst_row_split && act_row_split) {
            m_for_solver = (ne11 + (முழு) sess->mdev.count - 1) / (முழு) sess->mdev.count;
            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
        }
    }
    மாறிலி தருக்க pipeline = is_matmul_id ? பொய் : htp_mm_hmx_pipeline(m_for_solver);

    மாறிலி தருக்க is_batched_val = is_matmul_id ? பொய் : is_batched;
    மாறிலி முழு group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    தருக்க use_grouped = பொய்;
    முழு act_threads_selected = 0;

    எனில் (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && group_size > 1) {
        // Try grouped path first
        எனில் (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = உண்மை;
        }
    }

    எனில் (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        முழு m_id_rows = 0;
        எனில் (dst && is_matmul_id) {
            மாறிலி முழு n_experts = ne02 > 0 ? ne02 : 1;
            மாறிலி size_t total_expert_rows = (size_t) dst->ne[1] * dst->ne[2];
            முழு m_per_expert = (முழு) ((total_expert_rows + n_experts - 1) / n_experts);
            எனில் (sess->mdev.count > 1 && ggml_hexagon_can_row_partition(dst)) {
                m_per_expert = (m_per_expert + (முழு) sess->mdev.count - 1) / (முழு) sess->mdev.count;
            }
            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
        }
        மாறிலி uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
        எனில் (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            திரும்பு பொய்;
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
    kparams->div_n_act_threads = init_fastdiv_மதிப்புகள்(act_threads_selected);
    kparams->div_ne00_padded   = init_fastdiv_மதிப்புகள்(ne00_padded);
    kparams->vtcm_act_size     = 0;
    kparams->vtcm_bias_size    = (int32_t) src2_size;
    kparams->vtcm_dst_size     = 0;

    எனில் (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } இல்லையெனில் {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    திரும்பு உண்மை;
}

நிலையான வெறுமை ggml_hexagon_precompute_hvx_mm_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    முழு wtype,
    தருக்க is_tiled,
    முழு ne02,
    முழு ne03,
    முழு ne10,
    முழு ne11,
    முழு ne12,
    முழு ne13,
    தருக்க is_matmul_id,
    மாறிலி size_t src2_row_size,
    size_t vtcm_budget,
    கட்டமைப்பு htp_mm_kernel_params * kparams
) {
    எனில் (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        திரும்பு;
    }

    kparams->n_hmx = 0;
    kparams->n_threads = sess->n_threads;

    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);

    மாறிலி முழு src1_nrows = ne11 * ne12 * ne13;

    எனில் (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_size = htp_tensor_get_tile_size(wtype);
        kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);

        மாறிலி தருக்க k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        எனில் (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (முழு) sess->n_threads)
                                  ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);

            கட்டமைப்பு htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            க்கு (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], kparams->act_row_size, 0, d, உண்மை, பொய்
                );
                எனில் (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    நிறுத்து;
                }
            }
            எனில் (L.total_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                திரும்பு;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = 0;
            kparams->vtcm_dst_size  = L.dst_bytes;
            திரும்பு;
        } இல்லையெனில் {
            தருக்க try_tiled = (k_align && opt_mm_select >= 1);
            எனில் (try_tiled) {
                kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                எனில் (src1_nrows < (முழு) sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } இல்லையெனில் {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                கட்டமைப்பு htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                க்கு (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, பொய், பொய்
                    );
                    எனில் (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        நிறுத்து;
                    }
                }

                uint32_t m_chunk = 0;
                எனில் (htp_mm_hvx_solve_vtcm_params(
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
                    திரும்பு;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            திரும்பு;
        }
    } இல்லையெனில் எனில் (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        திரும்பு;
    } இல்லையெனில் எனில் (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        கட்டமைப்பு htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        எனில் (htp_mm_hvx_solve_vtcm_params(
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
            திரும்பு;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        திரும்பு;
    } இல்லையெனில் எனில் (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        கட்டமைப்பு htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        எனில் (htp_mm_hvx_solve_vtcm_params(
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
            திரும்பு;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        திரும்பு;
    } இல்லையெனில் எனில் (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        கட்டமைப்பு htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        எனில் (htp_mm_hvx_solve_vtcm_params(
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
            திரும்பு;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        திரும்பு;
    } இல்லையெனில் {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        திரும்பு;
    }
}

நிலையான வெறுமை ggml_hexagon_precompute_matmul_params_impl(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    மாறிலி size_t src2_row_size,
    மாறிலி size_t src2_size,
    கட்டமைப்பு htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));

    மாறிலி முழு ne00 = src0->ne[0];
    மாறிலி முழு ne01 = src0->ne[1];
    மாறிலி முழு ne02 = src0->ne[2];
    மாறிலி முழு ne03 = src0->ne[3];

    மாறிலி முழு ne10 = src1->ne[0];
    மாறிலி முழு ne11 = src1->ne[1];
    மாறிலி முழு ne12 = src1->ne[2];
    மாறிலி முழு ne13 = src1->ne[3];

    மாறிலி தருக்க is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    மாறிலி முழு wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    மாறிலி முழு ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    மாறிலி முழு ne11_padded = hex_round_up(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    மாறிலி முழு ne01_tiled  = hex_round_up(ne01, 32);

    மாறிலி தருக்க is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    மாறிலி தருக்க is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    மாறிலி size_t vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    தருக்க hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    எனில் (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        எனில் (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
            செல் finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_size, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_மதிப்புகள்(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_மதிப்புகள்(ne11);
    kparams->div_r2       = init_fastdiv_மதிப்புகள்(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_மதிப்புகள்(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_மதிப்புகள்(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
நிலையான தருக்க ggml_hexagon_rows_stride(மாறிலி int64_t * ne, மாறிலி size_t * nb, size_t * stride) {
    size_t s = 0, next = 0;
    க்கு (முழு i = 1; i < GGML_MAX_DIMS; i++) {
        எனில் (ne[i] == 1) தொடர்;
        எனில் (s == 0) { s = nb[i]; next = s * ne[i]; தொடர்; }
        எனில் (nb[i] != next) திரும்பு பொய்;
        next *= ne[i];
    }
    *stride = s ? s : nb[1];
    திரும்பு உண்மை;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
நிலையான தருக்க ggml_hexagon_matmul_can_collapse(மாறிலி கட்டமைப்பு ggml_tensor * src0, மாறிலி கட்டமைப்பு ggml_tensor * src1, மாறிலி கட்டமைப்பு ggml_tensor * dst) {
    size_t s1, sd;
    திரும்பு (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_size(src1->type) && ggml_hexagon_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_size(dst->type) && ggml_hexagon_rows_stride(dst->ne, dst->nb, &sd);
}

நிலையான தருக்க ggml_hexagon_matmul_add_can_collapse(
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * src2,
    மாறிலி கட்டமைப்பு ggml_tensor * dst
) {
    எனில் (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        திரும்பு பொய்;
    }
    எனில் (!src2) {
        திரும்பு உண்மை;
    }
    மாறிலி int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    எனில் (src2_nrows == 1) {
        திரும்பு உண்மை;
    }
    size_t s2;
    திரும்பு src2->nb[0] == ggml_type_size(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_rows_stride(src2->ne, src2->nb, &s2);
}

நிலையான ggml_tensor ggml_hexagon_tensor_collapse_rows(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    திரும்பு c;
}

நிலையான வெறுமை ggml_hexagon_precompute_matmul_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_mm_kernel_params * kparams
) {
    எனில் (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        மாறிலி ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        மாறிலி ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        திரும்பு;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

நிலையான வெறுமை ggml_hexagon_precompute_fused_matmul_add_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * src2,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_mm_kernel_params * kparams
) {
    எனில் (ggml_hexagon_matmul_add_can_collapse(src0, src1, src2, dst)) {
        மாறிலி ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        மாறிலி ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        மாறிலி ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        மாறிலி கட்டமைப்பு ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        மாறிலி size_t src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
        kparams->collapse = 1;
        திரும்பு;
    }
    மாறிலி size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

நிலையான தருக்க ggml_hexagon_precompute_இருமம்_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    uint32_t op,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_இருமம்_kernel_params * kparams,
    மாறிலி கட்டமைப்பு ggml_tensor * src2
) {
    memset(kparams, 0, அளவு(*kparams));

    மாறிலி size_t elem_size = ggml_type_size(src0->type);
    மாறிலி size_t src0_row_size = src0->ne[0] * elem_size;
    மாறிலி size_t src1_row_size = src1->ne[0] * elem_size;
    மாறிலி size_t dst_row_size  = dst->ne[0]  * elem_size;

    மாறிலி size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    மாறிலி size_t src1_row_size_aligned = hex_round_up(src1_row_size, 128);
    மாறிலி size_t dst_row_size_aligned  = hex_round_up(dst_row_size,  128);

    எனில் (!htp_இருமம்_strides_safe(src0, src1, dst, elem_size)) {
        திரும்பு பொய்;
    }

    மாறிலி தருக்க is_add_id = op == HTP_OP_ADD_ID;
    மாறிலி தருக்க is_scalar = !is_add_id && src1->ne[0] == 1;
    மாறிலி தருக்க is_row_bcast = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    மாறிலி தருக்க is_same_shape = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    மாறிலி தருக்க is_complex   = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    மாறிலி தருக்க is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    மாறிலி தருக்க is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    எனில் (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        மாறிலி uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        மாறிலி uint32_t n_threads = sess->n_threads;
        மாறிலி uint32_t max_chunk_elems = 32768 / elem_size;
        மாறிலி uint32_t min_chunk_elems = 256;
        மாறிலி uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
        மாறிலி uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        மாறிலி uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

        kparams->kernel_type           = HTP_இருமம்_KERNEL_CHUNKED;
        kparams->n_threads             = n_threads;
        kparams->rows_per_buffer       = 1;
        kparams->src0_row_size_aligned = src0_row_size_aligned;
        kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
        kparams->dst_row_size_aligned  = dst_row_size_aligned;
        kparams->src1_size             = 0;
        kparams->chunk_size            = chunk_size;
        kparams->chunk_bytes           = chunk_bytes;
        kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

        கட்டமைப்பு htp_இருமம்_vtcm_layout L;
        htp_இருமம்_vtcm_layout_build(&L, kparams, sess->vtcm_size);
        எனில் (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
            திரும்பு பொய்;
        }

        kparams->vtcm_size = L.total_bytes;
        திரும்பு உண்மை;
    }

    எண்ணிக்கை htp_இருமம்_kernel_type kernel_type;
    size_t src1_size = 0;
    size_t src2_size = 0;

    எனில் (is_add_id) {
        GGML_ASSERT(src2 != nullptr);
        kernel_type = HTP_இருமம்_KERNEL_ADD_ID;
        src1_size = hex_round_up(src1->ne[1] * src1_row_size_aligned, 128);
        src2_size = hex_round_up(ggml_nelements(src2) * அளவு(int32_t), 128);
        எனில் (src1_size + src2_size > sess->vtcm_size) {
            திரும்பு பொய்;
        }
    } இல்லையெனில் எனில் (is_row_bcast) {
        kernel_type = HTP_இருமம்_KERNEL_ROW_BCAST;
        src1_size = src1_row_size_aligned;
    } இல்லையெனில் எனில் (is_scalar) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size, 128);
        எனில் (src1_size > sess->vtcm_size) {
            திரும்பு பொய்;
        }
        kernel_type = HTP_இருமம்_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } இல்லையெனில் எனில் (is_same_shape) {
        kernel_type = HTP_இருமம்_KERNEL_SAME_SHAPE;
    } இல்லையெனில் எனில் (is_complex) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        எனில் (src1_size > sess->vtcm_size) {
            திரும்பு பொய்;
        }
        kernel_type = HTP_இருமம்_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } இல்லையெனில் {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        எனில் (src1_size > sess->vtcm_size) {
            திரும்பு பொய்;
        }
        kernel_type = HTP_இருமம்_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_threads = sess->n_threads;
    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;
    kparams->src1_size             = src1_size;
    kparams->src2_size             = src2_size;

    மாறிலி தருக்க src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    மாறிலி தருக்க src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    மாறிலி தருக்க dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    மாறிலி தருக்க src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    மாறிலி தருக்க src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    மாறிலி தருக்க dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    தருக்க split_at_ne01 = பொய்;
    எனில் (src0->ne[2] > 1) {
        எனில் (kernel_type == HTP_இருமம்_KERNEL_SAME_SHAPE) {
            split_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } இல்லையெனில் எனில் (kernel_type == HTP_இருமம்_KERNEL_ROW_BCAST) {
            split_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } இல்லையெனில் {
            split_at_ne01 = உண்மை;
        }
    }
    kparams->split_at_ne01 = split_at_ne01 ? 1 : 0;

    தருக்க split_at_ne02 = பொய்;
    எனில் (src0->ne[3] > 1) {
        எனில் (kernel_type == HTP_இருமம்_KERNEL_SAME_SHAPE) {
            split_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } இல்லையெனில் எனில் (kernel_type == HTP_இருமம்_KERNEL_ROW_BCAST) {
            split_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } இல்லையெனில் {
            split_at_ne02 = உண்மை;
        }
    }
    kparams->split_at_ne02 = split_at_ne02 ? 1 : 0;

    கட்டமைப்பு htp_இருமம்_vtcm_layout L;
    htp_இருமம்_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    எனில் (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {
        எனில் (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            மாறிலி uint32_t total_elems = (uint32_t) ggml_nelements(src0);
            மாறிலி uint32_t n_threads = sess->n_threads;
            மாறிலி uint32_t max_chunk_elems = 32768 / elem_size;
            மாறிலி uint32_t min_chunk_elems = 256;
            மாறிலி uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
            மாறிலி uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
            மாறிலி uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

            kparams->kernel_type           = HTP_இருமம்_KERNEL_CHUNKED;
            kparams->n_threads             = n_threads;
            kparams->rows_per_buffer       = 1;
            kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
            kparams->src1_size             = 0;
            kparams->chunk_size            = chunk_size;
            kparams->chunk_bytes           = chunk_bytes;
            kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

            htp_இருமம்_vtcm_layout_build(&L, kparams, sess->vtcm_size);
            எனில் (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
                திரும்பு பொய்;
            }

            kparams->vtcm_size = L.total_bytes;
            திரும்பு உண்மை;
        }
        திரும்பு பொய்;
    }

    kparams->rows_per_buffer = L.rows_per_buffer;
    kparams->vtcm_size       = L.total_bytes;

    திரும்பு உண்மை;
}

நிலையான வெறுமை ggml_hexagon_precompute_unary_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    uint32_t op,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));

    மாறிலி uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    மாறிலி uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    மாறிலி size_t elem_size = ggml_type_size(src0->type);

    மாறிலி size_t src0_data_row_size = src0->ne[0] * elem_size;
    மாறிலி size_t dst_data_row_size  = dst->ne[0]  * ggml_type_size(dst->type);

    மாறிலி size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    மாறிலி size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = 0;
    தருக்க broadcast_weight = பொய்;

    எனில் (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_data_row_size = src1->ne[0] * ggml_type_size(src1->type);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->broadcast_weight      = broadcast_weight;

    கட்டமைப்பு htp_unary_vtcm_layout L;
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

    மாறிலி uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_மதிப்புகள்(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_மதிப்புகள்(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_மதிப்புகள்(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_மதிப்புகள்(tiles_per_row);
}

நிலையான வெறுமை ggml_hexagon_precompute_get_rows_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_get_rows_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));

    மாறிலி uint32_t ne00 = src0->ne[0];
    மாறிலி uint32_t ne02 = src0->ne[2];
    மாறிலி uint32_t ne03 = src0->ne[3];

    மாறிலி uint32_t ne10 = src1->ne[0];
    மாறிலி uint32_t ne11 = src1->ne[1];
    மாறிலி uint32_t ne12 = src1->ne[2];
    மாறிலி uint32_t nr = ne10 * ne11 * ne12;

    மாறிலி தருக்க tiled = ggml_hexagon_tensor_is_tiled(src0);
    மாறிலி முழு wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    எனில் (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } இல்லையெனில் எனில் (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } இல்லையெனில் {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    மாறிலி uint32_t total_tasks = nr;

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, total_tasks);

    கட்டமைப்பு htp_get_rows_vtcm_layout vtcm_layout = {};
    போது (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, kparams->n_threads);
        எனில் (vtcm_layout.total_bytes <= sess->vtcm_size) {
            நிறுத்து;
        }
        --kparams->n_threads;
    }

    எனில் (kparams->n_threads == 0 && total_tasks > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, 1);
    }

    kparams->vtcm_size = (total_tasks == 0) ? 0 : vtcm_layout.total_bytes;
    kparams->tasks_per_thread = kparams->n_threads > 0 ? (total_tasks + kparams->n_threads - 1) / kparams->n_threads : 0;
    kparams->total_tasks = total_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_மதிப்புகள்(ne10) : fastdiv_மதிப்புகள்{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_மதிப்புகள்(ne10 * ne11) : fastdiv_மதிப்புகள்{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_மதிப்புகள்(ne02) : fastdiv_மதிப்புகள்{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_மதிப்புகள்(ne03) : fastdiv_மதிப்புகள்{0, 0};
}

நிலையான வெறுமை ggml_hexagon_precompute_set_rows_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0, //மதிப்புகள்
    மாறிலி கட்டமைப்பு ggml_tensor * src1, // indices
    மாறிலி கட்டமைப்பு ggml_tensor * dst,  // destination
    கட்டமைப்பு htp_set_rows_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));

    மாறிலி uint32_t nr = src0->ne[1];

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, nr);
    kparams->tasks_per_thread = (nr + kparams->n_threads - 1) / kparams->n_threads;
    kparams->total_tasks = nr;

    kparams->div_ne11 = init_fastdiv_மதிப்புகள்(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_மதிப்புகள்(src1->ne[2]);
    kparams->div_tasks_per_thread = init_fastdiv_மதிப்புகள்(kparams->tasks_per_thread);
    kparams->div_ne02 = init_fastdiv_மதிப்புகள்(src0->ne[2]);

    கட்டமைப்பு htp_set_rows_vtcm_layout vtcm_layout;
    htp_set_rows_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_threads);
    kparams->vtcm_size = vtcm_layout.total_bytes;
}

நிலையான வெறுமை ggml_hexagon_precompute_softmax_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];

    மாறிலி uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    மாறிலி uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);

    தசம scale = 1.0f;
    தசம max_bias = 0.0f;
    memcpy(&scale,    &op->op_params[0], அளவு(தசம));
    memcpy(&max_bias, &op->op_params[1], அளவு(தசம));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    மாறிலி uint32_t n_head = src0->ne[2];
    மாறிலி uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    kparams->n_head      = n_head;
    kparams->n_head_log2 = n_head_log2;

    எனில் (max_bias > 0.0f && n_head_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_head_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    } இல்லையெனில் {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    மாறிலி uint32_t ne00 = src0->ne[0];
    மாறிலி uint32_t ne10 = src1 ? src1->ne[0] : 1;

    கட்டமைப்பு htp_softmax_vtcm_layout layout;
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
    எனில் (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } இல்லையெனில் எனில் (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F16;
    } இல்லையெனில் {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    எனில் (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_மதிப்புகள்(src0->ne[1]);
    எனில் (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_மதிப்புகள்(src0->ne[2]);
    மாறிலி uint32_t ne12 = src1 ? src1->ne[2] : 1;
    மாறிலி uint32_t ne13 = src1 ? src1->ne[3] : 1;
    எனில் (ne12 > 0) kparams->div_ne12 = init_fastdiv_மதிப்புகள்(ne12);
    எனில் (ne13 > 0) kparams->div_ne13 = init_fastdiv_மதிப்புகள்(ne13);
}

நிலையான வெறுமை ggml_hexagon_precompute_rope_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src2 = op->src[2];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    மாறிலி uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    மாறிலி uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    மாறிலி uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    கட்டமைப்பு htp_rope_vtcm_layout layout;
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

    எனில் (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_மதிப்புகள்(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_மதிப்புகள்(dst->ne[1]);
    }
}

நிலையான வெறுமை ggml_hexagon_precompute_ssm_conv_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * src1,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));

    மாறிலி uint32_t d_conv  = (uint32_t) src1->ne[0];
    மாறிலி uint32_t d_inner = (uint32_t) src0->ne[1];
    மாறிலி uint32_t n_t     = (uint32_t) dst->ne[1];
    மாறிலி uint32_t n_s     = (uint32_t) dst->ne[2];
    மாறிலி uint32_t ncs     = (uint32_t) src0->ne[0];

    மாறிலி uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    மாறிலி uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
    மாறிலி uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);

    மாறிலி uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * அளவு(தசம), 128) + 128;
    மாறிலி uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * அளவு(தசம), 128);
    மாறிலி uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

    uint32_t vtcm_src0_per_thread = 0;
    uint32_t vtcm_dst_per_thread  = 0;

    எனில் (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;

        மாறிலி uint32_t src0_tile_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * அளவு(தசம), 128);
        மாறிலி uint32_t src0_T_bytes        = hex_round_up(d_conv * d_inner_per_thread * அளவு(தசம), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw_bytes + src0_T_bytes;

        மாறிலி uint32_t dst_tile_bytes = hex_round_up(d_inner_per_thread * அளவு(தசம), 128);
        vtcm_dst_per_thread = 2 * dst_tile_bytes;
    } இல்லையெனில் {
        மாறிலி size_t vtcm_budget = (sess->vtcm_size > 0 ? sess->vtcm_size / n_threads : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        மாறிலி uint32_t src0_block_T    = hex_round_up(ncs * 32 * அளவு(தசம), 128);
        மாறிலி size_t   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
        மாறிலி size_t   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);

        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
        target_max_tile = (std::max)(target_max_tile, 32u);
        target_max_tile = (std::min)(target_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * அளவு(தசம)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        எனில் (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        எனில் (d_inner_tile > target_max_tile) {
            d_inner_tile = target_max_tile;
        }
        எனில் (d_inner_tile > d_inner_per_thread) {
            d_inner_tile = d_inner_per_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        மாறிலி uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * அளவு(தசம), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw + src0_block_T;

        vtcm_dst_per_thread = 2 * hex_round_up(d_inner_tile * n_t * அளவு(தசம), 128);
    }

    kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
    kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
    kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;

    kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
    kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
    kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
    kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
}

நிலையான வெறுமை ggml_hexagon_precompute_gated_delta_net_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));

    மாறிலி கட்டமைப்பு ggml_tensor * q     = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * k     = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * v     = op->src[2];
    மாறிலி கட்டமைப்பு ggml_tensor * g     = op->src[3];
    மாறிலி கட்டமைப்பு ggml_tensor * state = op->src[5];

    மாறிலி uint32_t S_v      = (uint32_t) v->ne[0];
    மாறிலி uint32_t H        = (uint32_t) v->ne[1];
    மாறிலி uint32_t n_tokens = (uint32_t) v->ne[2];
    மாறிலி uint32_t n_seqs   = (uint32_t) v->ne[3];
    மாறிலி uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    மாறிலி uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    மாறிலி uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    மாறிலி uint32_t total_rows = H * n_seqs;
    மாறிலி uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    மாறிலி தருக்க can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    கட்டமைப்பு htp_gdn_hmx_vtcm_layout hmx_layout;
    கட்டமைப்பு htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    எனில் (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, உண்மை, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_size      = HTP_GDN_CHUNK_SIZE;
        kparams->n_chunks        = (n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_size       = (uint32_t) hmx_layout.total_bytes;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_bytes;
        kparams->vtcm_per_thread = (uint32_t) (hmx_layout.total_bytes / (n_threads > 0 ? n_threads : 1));
    } இல்லையெனில் {
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
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / அளவு(தசம));
    kparams->state_size_per_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((தசம) S_v);

    எனில் (H > 0)                  kparams->div_H         = init_fastdiv_மதிப்புகள்(H);
    எனில் (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_மதிப்புகள்((uint32_t) q->ne[1]);
    எனில் (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_மதிப்புகள்((uint32_t) k->ne[1]);
    எனில் (rq3 > 0)                kparams->div_rq3       = init_fastdiv_மதிப்புகள்(rq3);
    எனில் (rk3 > 0)                kparams->div_rk3       = init_fastdiv_மதிப்புகள்(rk3);
    எனில் (kparams->n_threads > 0) kparams->div_n_threads = init_fastdiv_மதிப்புகள்(kparams->n_threads);
}

நிலையான வெறுமை ggml_hexagon_precompute_sort_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    தருக்க is_உச்சி_k,
    கட்டமைப்பு htp_sort_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    மாறிலி uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    மாறிலி uint32_t ne00       = src0->ne[0];
    மாறிலி uint32_t k          = dst->ne[0];

    int32_t order = GGML_SORT_ORDER_DESC;
    எனில் (!is_உச்சி_k) {
        order = ((மாறிலி int32_t *) op->op_params)[0];
    }

    மாறிலி uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    மாறிலி size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    கட்டமைப்பு htp_sort_vtcm_layout layout;
    தருக்க ok = htp_sort_solve_layout(&layout, ne00, total_rows, k, n_threads_max, vtcm_budget, is_உச்சி_k);
    GGML_ASSERT(ok);

    kparams->n_threads         = (int32_t) layout.n_threads;
    kparams->total_rows        = (int32_t) total_rows;
    kparams->row_start         = 0;
    kparams->row_end           = (int32_t) total_rows;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->order             = order;
    kparams->is_உச்சி_k          = is_உச்சி_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_size         = (int32_t) layout.total_bytes;
    kparams->phase1_slot_size  = (int32_t) layout.phase1_slot_size;
    kparams->merge_மதிப்புகள்_off  = (int32_t) layout.merge_மதிப்புகள்_off;
    kparams->merge_indices_off = (int32_t) layout.merge_indices_off;
    kparams->merge_elems       = (int32_t) layout.merge_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

நிலையான வெறுமை ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    மாறிலி uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    மாறிலி int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    எனில் (numer_hi < 0) {
        *ox_hi = 0;
    } இல்லையெனில் {
        மாறிலி uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    எனில் (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

நிலையான தருக்க ggml_hexagon_pool_shape_fits(
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * dst
) {
    திரும்பு ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

நிலையான வெறுமை ggml_hexagon_precompute_pool_2d_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0,
    மாறிலி கட்டமைப்பு ggml_tensor * dst,
    கட்டமைப்பு htp_pool_2d_kernel_params * kparams,
    தருக்க is_pool_1d
) {
    memset(kparams, 0, அளவு(*kparams));

    மாறிலி uint32_t src_x = (uint32_t) src0->ne[0];
    மாறிலி uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    மாறிலி uint32_t dst_x = (uint32_t) dst->ne[0];
    மாறிலி uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    மாறிலி uint32_t src_plane_bytes = src_x * src_y * அளவு(தசம);
    மாறிலி uint32_t dst_plane_bytes = dst_x * dst_y * அளவு(தசம);
    மாறிலி uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    மாறிலி uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    கட்டமைப்பு htp_pool_vtcm_layout layout;
    மாறிலி தருக்க ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, sess->vtcm_size);
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
    kparams->avg_divide_மொத்தம் = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    மாறிலி தருக்க narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_path = (kparams->fast_path && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (தசம) (kparams->kernel_x * kparams->kernel_y);
}

நிலையான தருக்க ggml_hexagon_precompute_concat_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (!src0 || !src1 || !dst) {
        திரும்பு பொய்;
    }

    முழு dim = ((மாறிலி int32_t *) op->op_params)[0];
    எனில் (dim < 0 || dim >= GGML_MAX_DIMS) {
        திரும்பு பொய்;
    }
    kparams->dim = dim;

    எனில் (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        திரும்பு பொய்;
    }
    எனில் (src0->type != dst->type || src1->type != dst->type) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t type_size = ggml_type_size(dst->type);

    க்கு (முழு d = 0; d < GGML_MAX_DIMS; d++) {
        மாறிலி int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        எனில் (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            திரும்பு பொய்;
        }
    }

    மாறிலி uint32_t max_gather_ne0 = 128 / type_size;
    மாறிலி uint32_t total_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    மாறிலி தருக்க b_dense = (src1->nb[0] == type_size);
    மாறிலி தருக்க gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_size &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_size
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_size);

    எனில் (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_size && dst->nb[1] == (size_t) dst->ne[0] * type_size &&
        src0->nb[0] == type_size &&
        (src1->nb[0] == type_size || src1->nb[1] == type_size) &&
        gather_strides_ok &&
        htp_gather_rows_tensor_outer_contiguous(dst) &&
        htp_gather_rows_tensor_outer_contiguous(src0) &&
        htp_gather_rows_tensor_outer_contiguous(src1)) {

        எனில் (htp_gather_rows_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_size, total_rows,
                                         sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ROWS;
            kparams->n_threads   = (uint8_t) sess->n_threads;
            kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
            திரும்பு உண்மை;
        }
    }

    மாறிலி தருக்க dma_strides_ok = (src0->nb[0] == type_size && src1->nb[0] == type_size && dst->nb[0] == type_size);

    எனில் (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_threads   = 1;
        திரும்பு உண்மை;
    }

    மாறிலி தருக்க is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    மாறிலி தருக்க is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    மாறிலி தருக்க transposed_rows_ok = (src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size);

    எனில் (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        மாறிலி uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 8;
        கட்டமைப்பு htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_size, n_threads);

        எனில் (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
            திரும்பு பொய்;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_threads                         = n_threads;
        kparams->vtcm_size                         = layout.total_bytes;
        kparams->u.transposed.spad0_size_per_thread = layout.src0_spad_size_per_thread;
        kparams->u.transposed.spad1_size_per_thread = layout.src1_spad_size_per_thread;
        திரும்பு உண்மை;
    }

    திரும்பு பொய்;
}

நிலையான தருக்க ggml_hexagon_precompute_cpy_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (!src0 || !dst) {
        திரும்பு பொய்;
    }

    எனில் (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        திரும்பு பொய்;
    }
    எனில் (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        திரும்பு பொய்;
    }

    மாறிலி int64_t nelem_src = ggml_nelements(src0);
    மாறிலி int64_t nelem_dst = ggml_nelements(dst);
    எனில் (nelem_src != nelem_dst || nelem_src < 0) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t src_type_size = ggml_type_size(src0->type);
    மாறிலி uint32_t dst_type_size = ggml_type_size(dst->type);

    kparams->src0_type_size = (uint8_t) src_type_size;
    kparams->dst_type_size  = (uint8_t) dst_type_size;
    kparams->total_elems    = (uint32_t) nelem_src;

    எனில் (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        திரும்பு உண்மை;
    }

    எனில் (nelem_src == 1) {
        எனில் (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            திரும்பு உண்மை;
        }
        எனில் ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            திரும்பு உண்மை;
        }
        திரும்பு பொய்;
    }

    மாறிலி தருக்க sametype = (src0->type == dst->type);

    தருக்க same_extents = உண்மை;
    க்கு (முழு d = 0; d < GGML_MAX_DIMS; d++) {
        எனில் (src0->ne[d] != dst->ne[d]) {
            same_extents = பொய்;
            நிறுத்து;
        }
    }

    மாறிலி தருக்க transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_size) || (dst->nb[0] != dst_type_size) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_size) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_size);
    மாறிலி தருக்க sameshape  = same_extents && !transposed;

    மாறிலி தருக்க src_is_contiguous = ggml_is_contiguous(src0);
    மாறிலி தருக்க dst_is_contiguous = ggml_is_contiguous(dst);

    எனில் (sametype) {
        எனில் (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            திரும்பு உண்மை;
        }

        மாறிலி uint32_t max_gather_ne0 = 128 / src_type_size;
        மாறிலி uint32_t total_rows = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        எனில் ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_size && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_size &&
            total_rows > 1 &&
            htp_gather_rows_tensor_outer_contiguous(src0)) {

            எனில் (htp_gather_rows_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, பொய்,
                                             src_type_size, total_rows,
                                             sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ROWS;
                kparams->n_threads   = (uint8_t) sess->n_threads;
                kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
                திரும்பு உண்மை;
            }
        }

        எனில் (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            திரும்பு உண்மை;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_threads   = sess->n_threads > 0 ? (uint8_t) sess->n_threads : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_மதிப்புகள்((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_மதிப்புகள்((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_மதிப்புகள்((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_மதிப்புகள்((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_மதிப்புகள்((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_மதிப்புகள்((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        திரும்பு உண்மை;
    }

    எனில் (!sameshape) {
        திரும்பு பொய்;
    }

    மாறிலி தருக்க valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    எனில் (!valid_conversion) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 4;
    கட்டமைப்பு htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_size, dst_type_size, n_threads);

    எனில் (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
        திரும்பு பொய்;
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
    kparams->u.convert.div_ne01      = init_fastdiv_மதிப்புகள்((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_மதிப்புகள்((uint32_t) (src0->ne[2] * src0->ne[1]));

    திரும்பு உண்மை;
}

நிலையான வெறுமை ggml_hexagon_precompute_fused_mmnx_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0, // W0
    மாறிலி கட்டமைப்பு ggml_tensor * act,  // x
    int32_t n_weights,
    கட்டமைப்பு htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));
    kparams->n_threads = sess->n_threads;

    மாறிலி முழு ne00 = src0->ne[0];
    மாறிலி முழு ne01 = src0->ne[1];
    மாறிலி முழு ne02 = src0->ne[2];
    மாறிலி முழு ne03 = src0->ne[3];

    மாறிலி முழு ne10 = act->ne[0];
    மாறிலி முழு ne11 = act->ne[1];
    மாறிலி முழு ne12 = act->ne[2];
    மாறிலி முழு ne13 = act->ne[3];

    மாறிலி தருக்க is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    மாறிலி முழு wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    மாறிலி முழு ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    மாறிலி முழு ne11_padded = hex_round_up(ne11, 32);
    மாறிலி முழு ne01_tiled  = hex_round_up(ne01, 32);

    மாறிலி size_t vtcm_budget = sess->vtcm_size;
    மாறிலி தருக்க is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    தருக்க hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    எனில் (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, nullptr, பொய், is_batched)) {
        எனில் (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, பொய், is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            செல் finalize;
        }
    }

    எனில் (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        திரும்பு;
    }

    {
        மாறிலி முழு act_nrows = ne11 * ne12 * ne13;
        மாறிலி size_t act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
        மாறிலி size_t src0_row_size = src0->nb[1];

        மாறிலி uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        க்கு (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            கட்டமைப்பு htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
                0, src0_row_size, act_row_size, 0, d, பொய், உண்மை
            );
            எனில் (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                நிறுத்து;
            }
        }

        கட்டமைப்பு htp_mm_hvx_vtcm_layout L;
        தருக்க try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
            0, src0_row_size, act_row_size, 0, best_n_prefetch, பொய், உண்மை
        );

        எனில் (try_tiled && L.total_bytes <= sess->vtcm_size) {
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
        } இல்லையெனில் {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            திரும்பு;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_மதிப்புகள்(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_மதிப்புகள்(ne11);
    kparams->div_r2       = init_fastdiv_மதிப்புகள்(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_மதிப்புகள்(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_மதிப்புகள்(ne12);
}

நிலையான வெறுமை ggml_hexagon_precompute_fused_mmidnx_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * src0, // W0
    மாறிலி கட்டமைப்பு ggml_tensor * act,  // x
    மாறிலி கட்டமைப்பு ggml_tensor * dst,  // dst0
    int32_t n_weights,
    கட்டமைப்பு htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

நிலையான தருக்க ggml_hexagon_op_mul_mat_use_fwht(மாறிலி கட்டமைப்பு ggml_tensor * dst) {
    எனில் (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        திரும்பு பொய்;
    }

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = dst->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = dst->src[1];

    மாறிலி int64_t n = src1->ne[0];
    எனில் (n < 32 || (n & (n - 1)) != 0) {
        திரும்பு பொய்;
    }

    திரும்பு src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_rows(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

நிலையான தருக்க ggml_hexagon_supported_mul_mat(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * dst) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = dst->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = dst->src[1];

    எனில் (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        எனில் (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            திரும்பு பொய்;
        }
        கட்டமைப்பு htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, nullptr, dst, &ukparams);
        திரும்பு ukparams.vtcm_row_per_thread > 0 && (size_t) ukparams.vtcm_size <= sess->vtcm_size;
    }

    எனில் (dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    எனில் (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        திரும்பு பொய்;
    }

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_Q1_0:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_Q4_0:
        நிலை GGML_TYPE_Q4_1:
        நிலை GGML_TYPE_Q8_0:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_MXFP4:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_K:
        நிலை GGML_TYPE_Q6_K:
        நிலை GGML_TYPE_Q3_K:
        நிலை GGML_TYPE_Q2_K: {
            எனில் (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                திரும்பு பொய்;
            }

            மாறிலி int64_t qk = ggml_blck_size(src0->type);
            எனில் (src0->ne[0] % qk) {
                திரும்பு பொய்;
            }

            எனில் (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                திரும்பு பொய்;
            }
            எனில் (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                திரும்பு பொய்;
            }
            நிறுத்து;
        }

        நிலை GGML_TYPE_F16:
            எனில் (src0->nb[1] < src0->nb[0]) {
                திரும்பு பொய்;
            }
            எனில் (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                திரும்பு பொய்;
            }
            எனில் (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                திரும்பு பொய்;
            }
            நிறுத்து;

        நிலை GGML_TYPE_F32:
            எனில் (src1->type != GGML_TYPE_F32) {
                திரும்பு பொய்;
            }
            எனில் (src0->nb[1] < src0->nb[0]) {
                திரும்பு பொய்;
            }
            எனில் (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                திரும்பு பொய்;
            }
            எனில் (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                திரும்பு பொய்;
            }
            நிறுத்து;

        நிலை GGML_TYPE_BF16:
            எனில் (src0->nb[1] < src0->nb[0]) {
                திரும்பு பொய்;
            }
            எனில் (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                திரும்பு பொய்;
            }
            எனில் (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                திரும்பு பொய்;
            }
            நிறுத்து;

        இயல்பு:
            திரும்பு பொய்;
    }

    கட்டமைப்பு htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    எனில் (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_hexagon_supported_mul_mat_id(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * src2 = op->src[2];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        திரும்பு பொய்;
    }

    எனில் (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        திரும்பு பொய்;
    }

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_Q1_0:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_Q4_0:
        நிலை GGML_TYPE_Q4_1:
        நிலை GGML_TYPE_Q8_0:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_MXFP4:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_K:
        நிலை GGML_TYPE_Q6_K:
        நிலை GGML_TYPE_Q3_K:
        நிலை GGML_TYPE_Q2_K:
            எனில் (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                திரும்பு பொய்;
            }

            எனில் (src0->ne[0] % ggml_blck_size(src0->type) != 0) {
                திரும்பு பொய்;
            }
            நிறுத்து;

        நிலை GGML_TYPE_F16:
        நிலை GGML_TYPE_F32:
        நிலை GGML_TYPE_BF16:
            எனில் (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                திரும்பு பொய்;
            }
            எனில் (src0->ne[0] % 32 != 0) {
                திரும்பு பொய்;
            }
            நிறுத்து;

        இயல்பு:
            திரும்பு பொய்;
    }

    கட்டமைப்பு htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    எனில் (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_hexagon_supported_இருமம்(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src0->type == GGML_TYPE_F32) {
        எனில் (src1->type != GGML_TYPE_F32) {
            திரும்பு பொய்;
        }
        எனில் (dst->type != GGML_TYPE_F32) {
            திரும்பு பொய்;
        }
    }
    இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16) {
        எனில் (src1->type != GGML_TYPE_F16) {
            திரும்பு பொய்;
        }
        எனில் (dst->type != GGML_TYPE_F16) {
            திரும்பு பொய்;
        }
    }
    இல்லையெனில் {
        திரும்பு பொய்;
    }

    எனில் (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        திரும்பு பொய்;
    }
    எனில் (!ggml_are_same_shape(src0, dst)) {
        திரும்பு பொய்;
    }
    எனில் (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        திரும்பு பொய்;
    }

    கட்டமைப்பு htp_இருமம்_kernel_params kparams;
    திரும்பு ggml_hexagon_precompute_இருமம்_params(sess, op_remap_to_htp(op), src0, src1, dst, &kparams);
}

நிலையான தருக்க ggml_hexagon_supported_add_id(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * src2 = op->src[2];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (!src2) {
        திரும்பு பொய்;
    }
    எனில் (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        திரும்பு பொய்;
    }
    எனில் (!ggml_are_same_shape(src0, dst)) {
        திரும்பு பொய்;
    }
    எனில் (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        திரும்பு பொய்;
    }
    எனில் (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        திரும்பு பொய்;
    }
    எனில் (src0->nb[0] != அளவு(தசம) || src1->nb[0] != அளவு(தசம) || dst->nb[0] != அளவு(தசம) || src2->nb[0] != அளவு(int32_t)) {
        திரும்பு பொய்;
    }

    // REVISIT: add support for non-contiguous tensors
    எனில் (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }
    எனில் (src2->nb[1] < src2->ne[0] * அளவு(int32_t)) {
        திரும்பு பொய்;
    }

    கட்டமைப்பு htp_இருமம்_kernel_params kparams;
    திரும்பு ggml_hexagon_precompute_இருமம்_params(sess, HTP_OP_ADD_ID, src0, src1, dst, &kparams, src2);
}

நிலையான தருக்க ggml_hexagon_supported_unary(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        திரும்பு பொய்;
    }
    எனில் (dst->type != src0->type) {
        திரும்பு பொய்;
    }
    எனில் (!ggml_is_contiguous_rows(src0)) {
        திரும்பு பொய்;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    எனில் (src0->type == GGML_TYPE_F16) {
        தேர்வு (op->op) {
            நிலை GGML_OP_NORM:
            நிலை GGML_OP_RMS_NORM:
            நிலை GGML_OP_L2_NORM:
            நிலை GGML_OP_SCALE:
            நிலை GGML_OP_CLAMP:
            நிலை GGML_OP_SQR:
            நிலை GGML_OP_SQRT:
            நிலை GGML_OP_LOG:
                நிறுத்து;
            நிலை GGML_OP_UNARY:
                எனில் (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_STEP) {
                    திரும்பு பொய்;
                }
                நிறுத்து;
            இயல்பு:
                திரும்பு பொய்;
        }
    }

    எனில் (!ggml_are_same_shape(src0, dst)) {
        திரும்பு பொய்;
    }

    // dst must be contiguous; src0 may be non-contiguous
    எனில் (!ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_sum(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src0->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }
    எனில் (dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_sum_rows(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src0->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }
    எனில் (dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    // TODO: add support for non-contigiuos tensors
    எனில் (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_argmax(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src0->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }
    எனில் (dst->type != GGML_TYPE_I32) {
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_pool_2d(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி int32_t * params = op->op_params;

    எனில் (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        திரும்பு பொய்;
    }
    எனில் (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }
    எனில் (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        திரும்பு பொய்;
    }
    எனில் (!ggml_hexagon_pool_shape_fits(src0, op)) {
        திரும்பு பொய்;
    }

    மாறிலி int32_t kernel_x = params[1];
    மாறிலி int32_t kernel_y = params[2];
    மாறிலி int32_t stride_x = params[3];
    மாறிலி int32_t stride_y = params[4];
    மாறிலி int32_t pad_x    = params[5];
    மாறிலி int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    எனில் (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    மாறிலி uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    கட்டமைப்பு htp_pool_vtcm_layout layout;
    திரும்பு htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_threads, sess->vtcm_size);
}

நிலையான தருக்க ggml_hexagon_supported_pool_1d(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி int32_t * params = op->op_params;

    எனில் (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        திரும்பு பொய்;
    }
    எனில் (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }
    எனில் (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        திரும்பு பொய்;
    }
    எனில் (!ggml_hexagon_pool_shape_fits(src0, op)) {
        திரும்பு பொய்;
    }

    மாறிலி int32_t kernel = params[1];
    மாறிலி int32_t stride = params[2];
    மாறிலி int32_t pad    = params[3];
    எனில் (kernel <= 0 || stride <= 0 || pad < 0) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    மாறிலி uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    கட்டமைப்பு htp_pool_vtcm_layout layout;
    திரும்பு htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_threads, sess->vtcm_size);
}

நிலையான தருக்க ggml_hexagon_supported_activations(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        திரும்பு பொய்;
    }
    எனில் (dst->type != src0->type) {
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous_1(src0)) {
        திரும்பு பொய்;
    }
    எனில் (!ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }

    எனில் (src1) {
        எனில் (src1->type != src0->type) {
            திரும்பு பொய்;
        }
        எனில் (!ggml_are_same_shape(src0, src1)) {
            திரும்பு பொய்;
        }
        எனில் (!ggml_is_contiguous_1(src1)) {
            திரும்பு பொய்;
        }
    }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_softmax(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * src2 = op->src[2];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src2) {
        திரும்பு பொய்;  // FIXME: add support for sinks
    }

    எனில் (src0->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }
    எனில் (dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    எனில் (src0->ne[2] > 512) {
        திரும்பு பொய்;
    }

    எனில் (src1) {
        எனில் (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            திரும்பு பொய்;
        }
        எனில் (src0->ne[0] != src1->ne[0]) {
            திரும்பு பொய்;
        }
        எனில் (src1->ne[1] < src0->ne[1]) {
            திரும்பு பொய்;
        }
        எனில் (src0->ne[2] % src1->ne[2] != 0) {
            திரும்பு பொய்;
        }
        எனில் (src0->ne[3] % src1->ne[3] != 0) {
            திரும்பு பொய்;
        }
    }

    எனில் (src1) {
        எனில் (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            திரும்பு பொய்;
        }
    } இல்லையெனில் {
        எனில் (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            திரும்பு பொய்;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    மாறிலி int64_t ne0 = src0->ne[0];
    எனில் (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        திரும்பு பொய்;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    எனில் (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    மாறிலி uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    கட்டமைப்பு htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != nullptr, src1 && src1->type == GGML_TYPE_F16, n_threads);
    எனில் (layout.total_bytes > sess->vtcm_size) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_set_rows(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0]; //மதிப்புகள்
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1]; // indices
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    எனில் (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        திரும்பு பொய்;
    }

    எனில் (src0->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    எனில் (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        திரும்பு பொய்;
    }

    எனில் (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_get_rows(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0]; //மதிப்புகள்
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1]; // indices
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(dst)) {
        திரும்பு பொய்;
    }

    எனில் (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        திரும்பு பொய்;
    }

    எனில் (src0->type == dst->type) {
        எனில் (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            திரும்பு பொய்;
        }
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_I32) {
        திரும்பு பொய்;
    } இல்லையெனில் எனில் (dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    எனில் (ggml_is_empty(op)) {
        திரும்பு உண்மை;
    }

    கட்டமைப்பு htp_get_rows_kernel_params kparams;
    ggml_hexagon_precompute_get_rows_params(sess, src0, src1, dst, &kparams);
    எனில் (kparams.n_threads == 0 || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_argsort(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0]; //மதிப்புகள்
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;         // indices

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    மாறிலி uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    மாறிலி size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    கட்டமைப்பு htp_sort_vtcm_layout layout;
    எனில் (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, பொய்)) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_hexagon_supported_உச்சி_k(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0]; //மதிப்புகள்
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;         // indices

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    மாறிலி uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    மாறிலி size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    கட்டமைப்பு htp_sort_vtcm_layout layout;
    எனில் (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, உண்மை)) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_hexagon_supported_rope(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * src2 = op->src[2];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (!ggml_are_same_shape(src0, dst)) {
        திரும்பு பொய்;
    }

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        திரும்பு பொய்;
    }

    எனில் (src0->ne[0] <= 0) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    எனில் (src0_nrows == 0) {
        திரும்பு பொய்;
    }

    மாறிலி int32_t * op_params = &op->op_params[0];
    மாறிலி முழு n_dims = op_params[1];
    மாறிலி முழு mode   = op_params[2];
    மாறிலி முழு n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    எனில் (n_dims < 0 || n_dims % 2 != 0) {
        திரும்பு பொய்;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    எனில் (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        திரும்பு பொய்;
    }

    தசம freq_base;
    memcpy(&freq_base, op_params + 5, அளவு(தசம));
    எனில் (freq_base < 0.0f) {
        திரும்பு பொய்;
    }

    எனில் (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        திரும்பு பொய்;
    }

    மாறிலி தருக்க is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    எனில் (mode == GGML_ROPE_TYPE_VISION) {
        எனில் (n_dims != (முழு) (src0->ne[0] / 2) || n_offs != 0) {
            திரும்பு பொய்;
        }
    }

    எனில் (is_mrope) {
        மாறிலி int32_t * sections = op_params + 11;
        எனில் (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            திரும்பு பொய்;
        }
    }

    மாறிலி int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    எனில் (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        திரும்பு பொய்;
    }

    எனில் (src2) {
        எனில் (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            திரும்பு பொய்;
        }
        எனில் (src2->ne[0] < (n_dims / 2)) {
            திரும்பு பொய்;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    எனில் (src0->nb[0] != அளவு(தசம) || dst->nb[0] != அளவு(தசம)) {
        திரும்பு பொய்;
    }
    எனில் (src0->nb[1] < src0->ne[0] * அளவு(தசம) || dst->nb[1] < dst->ne[0] * அளவு(தசம)) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    மாறிலி uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    கட்டமைப்பு htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    எனில் (layout.total_bytes > sess->vtcm_size) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_hexagon_supported_ssm_conv(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    // Only support FP32 for now
    எனில் (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    // Check IO tensor shapes and dims
    எனில் (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        திரும்பு பொய்; // src0 should be effectively 3D
    }

    மாறிலி முழு d_conv  = src1->ne[0];
    மாறிலி முழு d_inner = src0->ne[1];
    மாறிலி முழு n_t = dst->ne[1];
    மாறிலி முழு n_s = dst->ne[2];

    எனில் (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        திரும்பு பொய்;
    }
    எனில் (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        திரும்பு பொய்;
    }
    எனில் (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        திரும்பு பொய்;
    }
    எனில் (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        திரும்பு பொய்;
    }
    எனில் (src0->nb[0] != அளவு(தசம) || src1->nb[0] != அளவு(தசம) || dst->nb[0] != அளவு(தசம)) {
        திரும்பு பொய்;
    }
    எனில் (src0->nb[1] != src0->ne[0] * அளவு(தசம) || src1->nb[1] != src1->ne[0] * அளவு(தசம)) {
        திரும்பு பொய்;
    }

    கட்டமைப்பு htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    எனில் ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}

நிலையான தருக்க im2col_3d_calc_output_size(uint32_t உள்ளீடு, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    எனில் (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        திரும்பு பொய்;
    }

    மாறிலி int64_t numerator = (int64_t) உள்ளீடு + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    எனில் (numerator < 0) {
        திரும்பு பொய்;
    }

    மாறிலி uint64_t size = (uint64_t) (numerator / stride) + 1;
    எனில் (size == 0 || size > UINT32_MAX) {
        திரும்பு பொய்;
    }

    *output = (uint32_t) size;
    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_hexagon_precompute_im2col_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (!src0 || !src1 || !dst) {
        திரும்பு பொய்;
    }

    எனில் (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        திரும்பு பொய்;
    }

    எனில் (src1->nb[0] != அளவு(தசம) || !ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }

    மாறிலி int32_t * op_params = (மாறிலி int32_t *) op->op_params;
    மாறிலி int32_t s0 = op_params[0];
    மாறிலி int32_t s1 = op_params[1];
    மாறிலி int32_t p0 = op_params[2];
    மாறிலி int32_t p1 = op_params[3];
    மாறிலி int32_t d0 = op_params[4];
    மாறிலி int32_t d1 = op_params[5];
    மாறிலி தருக்க is_2D = (op_params[6] == 1);

    எனில் (s0 <= 0 || d0 <= 0 || p0 < 0) {
        திரும்பு பொய்;
    }
    எனில் (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    மாறிலி uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    மாறிலி uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    மாறிலி uint32_t IW = (uint32_t) src1->ne[0];
    மாறிலி uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    மாறிலி uint32_t KW = (uint32_t) src0->ne[0];
    மாறிலி uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    மாறிலி uint32_t OW = (uint32_t) dst->ne[1];

    எனில் (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        திரும்பு பொய்;
    }

    மாறிலி uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    மாறிலி uint64_t total_rows_64   = (uint64_t) N * OH;
    எனில் (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX) {
        திரும்பு பொய்;
    }
    மாறிலி uint32_t patch_stride = (uint32_t) patch_stride_64;
    மாறிலி uint32_t total_rows   = (uint32_t) total_rows_64;

    எனில் (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        திரும்பு பொய்;
    }
    எனில் (is_2D && dst->ne[2] != OH) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? அளவு(ggml_fp16_t) : அளவு(தசம);

    தருக்க is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    எனில் (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    மாறிலி uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    மாறிலி uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    மாறிலி uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    எனில் (is_patchembed) {
        மாறிலி uint64_t src_row_bytes_raw = (uint64_t) IC * KH * IW * அளவு(தசம);
        மாறிலி uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        எனில் (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            மாறிலி uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            மாறிலி uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            கட்டமைப்பு htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            எனில் (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                திரும்பு உண்மை;
            }
        }
    }

    க்கு (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        மாறிலி uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        எனில் (Wb > UINT32_MAX) {
            எனில் (owb == 1) நிறுத்து;
            தொடர்;
        }

        மாறிலி uint64_t src_row_bytes_raw = (uint64_t) IC * KH * Wb * அளவு(தசம);
        மாறிலி uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        எனில் (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            எனில் (owb == 1) நிறுத்து;
            தொடர்;
        }

        மாறிலி uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        மாறிலி uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        கட்டமைப்பு htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        எனில் (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            திரும்பு உண்மை;
        }
        எனில் (owb == 1) நிறுத்து;
    }

    திரும்பு பொய்;
}

நிலையான தருக்க ggml_hexagon_precompute_im2col_3d_params(
    மாறிலி கட்டமைப்பு ggml_hexagon_session * sess,
    மாறிலி கட்டமைப்பு ggml_tensor * op,
    கட்டமைப்பு htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, அளவு(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (!src0 || !src1 || !dst) {
        திரும்பு பொய்;
    }

    எனில் (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        திரும்பு பொய்;
    }

    எனில் (src1->nb[0] != அளவு(தசம) || !ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }

    மாறிலி int32_t * params = (மாறிலி int32_t *) op->op_params;
    மாறிலி int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    மாறிலி int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    மாறிலி int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    மாறிலி int32_t ic = params[9];

    எனில் (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        திரும்பு பொய்;
    }

    எனில் (src1->ne[3] % ic != 0) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t IC = (uint32_t) ic;
    மாறிலி uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    எனில் (N == 0) {
        திரும்பு பொய்;
    }

    uint32_t OW, OH, OD;
    எனில் (!im2col_3d_calc_output_size(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_size(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_size(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t KD = (uint32_t) src0->ne[2];
    மாறிலி uint32_t KH = (uint32_t) src0->ne[1];
    மாறிலி uint32_t KW = (uint32_t) src0->ne[0];
    மாறிலி uint32_t ID = (uint32_t) src1->ne[2];
    மாறிலி uint32_t IH = (uint32_t) src1->ne[1];
    மாறிலி uint32_t IW = (uint32_t) src1->ne[0];

    மாறிலி uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    மாறிலி uint64_t total_rows_64   = (uint64_t) N * OD * OH;
    மாறிலி uint64_t dst_ne3_64      = (uint64_t) OD * N;

    எனில் (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        திரும்பு பொய்;
    }
    மாறிலி uint32_t patch_stride = (uint32_t) patch_stride_64;
    மாறிலி uint32_t total_rows   = (uint32_t) total_rows_64;
    மாறிலி uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    எனில் (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        திரும்பு பொய்;
    }

    மாறிலி uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? அளவு(ggml_fp16_t) : அளவு(தசம);

    மாறிலி தருக்க is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    மாறிலி uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    மாறிலி uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    மாறிலி uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    எனில் (is_patchembed) {
        மாறிலி uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * IW * அளவு(தசம);
        மாறிலி uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        எனில் (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            மாறிலி uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            மாறிலி uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            கட்டமைப்பு htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            எனில் (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
                திரும்பு உண்மை;
            }
        }
    }

    க்கு (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        மாறிலி uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        எனில் (Wb > UINT32_MAX) {
            எனில் (owb == 1) நிறுத்து;
            தொடர்;
        }

        மாறிலி uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * Wb * அளவு(தசம);
        மாறிலி uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        எனில் (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            எனில் (owb == 1) நிறுத்து;
            தொடர்;
        }

        மாறிலி uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        மாறிலி uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        கட்டமைப்பு htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        எனில் (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
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
            திரும்பு உண்மை;
        }
        எனில் (owb == 1) நிறுத்து;
    }

    திரும்பு பொய்;
}

நிலையான தருக்க ggml_hexagon_supported_im2col(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    கட்டமைப்பு htp_im2col_kernel_params kparams;
    திரும்பு ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

நிலையான தருக்க ggml_hexagon_supported_im2col_3d(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    கட்டமைப்பு htp_im2col_kernel_params kparams;
    திரும்பு ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

நிலையான தருக்க ggml_hexagon_supported_pad(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    மாறிலி int32_t lp0 = ((மாறிலி int32_t *) op->op_params)[0];
    மாறிலி int32_t rp0 = ((மாறிலி int32_t *) op->op_params)[1];
    மாறிலி int32_t circular = ((மாறிலி int32_t *) op->op_params)[8];

    எனில் (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_cumsum(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_diag(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    // diag only supports F32 currently
    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    // Input must have ne[1] == 1 (vector input)
    எனில் (src0->ne[1] != 1) {
        திரும்பு பொய்;
    }

    // Output must be square in first two dimensions
    எனில் (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_solve_tri(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0]; // A
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = op->src[1]; // B
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;         // X

    எனில் (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    எனில் (src0->ne[0] != src0->ne[1]) {
        திரும்பு பொய்;
    }

    எனில் (src0->ne[1] != src1->ne[1]) {
        திரும்பு பொய்;
    }

    எனில் (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        திரும்பு பொய்;
    }

    எனில் (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_tri(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src0->type != GGML_TYPE_F32) { திரும்பு பொய்; }
    எனில் (dst->type  != GGML_TYPE_F32) { திரும்பு பொய்; }
    எனில் (!ggml_are_same_shape(src0, dst)) { திரும்பு பொய்; }
    எனில் (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { திரும்பு பொய்; }

    திரும்பு உண்மை;

    GGML_UNUSED(sess);
}

நிலையான மாறிலி எழுத்து * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    திரும்பு sess->c_name();
}

நிலையான வெறுமை ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

நிலையான htp_op_code op_remap_to_htp(மாறிலி ggml_tensor * t) {
    தேர்வு (t->op) {
        நிலை GGML_OP_FLASH_ATTN_EXT:  திரும்பு HTP_OP_FLASH_ATTN_EXT;
        நிலை GGML_OP_MUL_MAT:
            எனில் (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                திரும்பு HTP_OP_HADAMARD;
            }
            திரும்பு HTP_OP_MUL_MAT;
        நிலை GGML_OP_MUL_MAT_ID:      திரும்பு HTP_OP_MUL_MAT_ID;
        நிலை GGML_OP_MUL:             திரும்பு HTP_OP_MUL;
        நிலை GGML_OP_ADD:             திரும்பு HTP_OP_ADD;
        நிலை GGML_OP_ADD_ID:          திரும்பு HTP_OP_ADD_ID;
        நிலை GGML_OP_SUB:             திரும்பு HTP_OP_SUB;
        நிலை GGML_OP_DIV:             திரும்பு HTP_OP_DIV;
        நிலை GGML_OP_DUP:
        நிலை GGML_OP_CPY:             திரும்பு HTP_OP_CPY;
        நிலை GGML_OP_CONT:            திரும்பு HTP_OP_CPY;
        நிலை GGML_OP_GET_ROWS:        திரும்பு HTP_OP_GET_ROWS;
        நிலை GGML_OP_SET_ROWS:        திரும்பு HTP_OP_SET_ROWS;
        நிலை GGML_OP_SUM:             திரும்பு HTP_OP_SUM;
        நிலை GGML_OP_SUM_ROWS:        திரும்பு HTP_OP_SUM_ROWS;
        நிலை GGML_OP_POOL_2D:         திரும்பு HTP_OP_POOL_2D;
        நிலை GGML_OP_POOL_1D:         திரும்பு HTP_OP_POOL_1D;
        நிலை GGML_OP_ARGSORT:         திரும்பு HTP_OP_ARGSORT;
        நிலை GGML_OP_உச்சி_K:           திரும்பு HTP_OP_உச்சி_K;
        நிலை GGML_OP_ARGMAX:          திரும்பு HTP_OP_ARGMAX;
        நிலை GGML_OP_NORM:            திரும்பு HTP_OP_NORM;
        நிலை GGML_OP_L2_NORM:         திரும்பு HTP_OP_L2_NORM;
        நிலை GGML_OP_RMS_NORM:        திரும்பு HTP_OP_RMS_NORM;
        நிலை GGML_OP_CONCAT:          திரும்பு HTP_OP_CONCAT;
        நிலை GGML_OP_SCALE:           திரும்பு HTP_OP_SCALE;
        நிலை GGML_OP_CLAMP:           திரும்பு HTP_OP_CLAMP;
        நிலை GGML_OP_LEAKY_RELU:      திரும்பு HTP_OP_LEAKY_RELU;
        நிலை GGML_OP_SQR:             திரும்பு HTP_OP_SQR;
        நிலை GGML_OP_SQRT:            திரும்பு HTP_OP_SQRT;
        நிலை GGML_OP_LOG:             திரும்பு HTP_OP_UNARY_LOG;
        நிலை GGML_OP_SOFT_MAX:        திரும்பு HTP_OP_SOFTMAX;
        நிலை GGML_OP_SSM_CONV:        திரும்பு HTP_OP_SSM_CONV;
        நிலை GGML_OP_GATED_DELTA_NET: திரும்பு HTP_OP_GATED_DELTA_NET;
        நிலை GGML_OP_ROPE:            திரும்பு HTP_OP_ROPE;
        நிலை GGML_OP_REPEAT:          திரும்பு HTP_OP_REPEAT;
        நிலை GGML_OP_CUMSUM:          திரும்பு HTP_OP_CUMSUM;
        நிலை GGML_OP_FILL:            திரும்பு HTP_OP_FILL;
        நிலை GGML_OP_DIAG:            திரும்பு HTP_OP_DIAG;
        நிலை GGML_OP_SOLVE_TRI:       திரும்பு HTP_OP_SOLVE_TRI;
        நிலை GGML_OP_TRI:             திரும்பு HTP_OP_TRI;
        நிலை GGML_OP_PAD:             திரும்பு HTP_OP_PAD;
        நிலை GGML_OP_IM2COL:          திரும்பு HTP_OP_IM2COL;
        நிலை GGML_OP_IM2COL_3D:       திரும்பு HTP_OP_IM2COL_3D;
        நிலை GGML_OP_ROLL:            திரும்பு HTP_OP_ROLL;

        நிலை GGML_OP_UNARY:
            தேர்வு (ggml_get_unary_op(t)) {
                நிலை GGML_UNARY_OP_SILU:       திரும்பு HTP_OP_UNARY_SILU;
                நிலை GGML_UNARY_OP_GELU:       திரும்பு HTP_OP_UNARY_GELU;
                நிலை GGML_UNARY_OP_GELU_QUICK: திரும்பு HTP_OP_UNARY_GELU_QUICK;
                நிலை GGML_UNARY_OP_GELU_ERF:   திரும்பு HTP_OP_UNARY_GELU_ERF;
                நிலை GGML_UNARY_OP_SIGMOID:    திரும்பு HTP_OP_UNARY_SIGMOID;
                நிலை GGML_UNARY_OP_NEG:        திரும்பு HTP_OP_UNARY_NEG;
                நிலை GGML_UNARY_OP_EXP:        திரும்பு HTP_OP_UNARY_EXP;
                நிலை GGML_UNARY_OP_SOFTPLUS:   திரும்பு HTP_OP_UNARY_SOFTPLUS;
                நிலை GGML_UNARY_OP_TANH:       திரும்பு HTP_OP_UNARY_TANH;
                நிலை GGML_UNARY_OP_ABS:        திரும்பு HTP_OP_UNARY_ABS;
                நிலை GGML_UNARY_OP_RELU:       திரும்பு HTP_OP_UNARY_RELU;
                நிலை GGML_UNARY_OP_STEP:       திரும்பு HTP_OP_UNARY_STEP;
            இயல்பு:
                நிறுத்து;
            }
            நிறுத்து;

        நிலை GGML_OP_GLU:
            தேர்வு (ggml_get_glu_op(t)) {
                நிலை GGML_GLU_OP_SWIGLU:       திரும்பு HTP_OP_GLU_SWIGLU;
                நிலை GGML_GLU_OP_SWIGLU_OAI:   திரும்பு HTP_OP_GLU_SWIGLU_OAI;
                நிலை GGML_GLU_OP_SWIGLU_CLAMP: திரும்பு HTP_OP_GLU_SWIGLU_CLAMP;
                நிலை GGML_GLU_OP_GEGLU:        திரும்பு HTP_OP_GLU_GEGLU;
                நிலை GGML_GLU_OP_GEGLU_QUICK:  திரும்பு HTP_OP_GLU_GEGLU_QUICK;
                நிலை GGML_GLU_OP_GEGLU_ERF:    திரும்பு HTP_OP_GLU_GEGLU_ERF;
                இயல்பு: நிறுத்து;
            }
            நிறுத்து;

        இயல்பு:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    திரும்பு HTP_OP_INVALID;
}

நிலையான உள்வரி தருக்க op_is_compute(ggml_tensor *முனை)
{
    திரும்பு !ggml_op_is_empty(முனை->op) && !ggml_is_empty(முனை) && (முனை->flags & GGML_TENSOR_FLAG_COMPUTE);
}

நிலையான தருக்க mm_is_hmx_eligible(மாறிலி ggml_tensor * t) {
    எனில் (opt_nhmx == 0) { திரும்பு பொய்; }

    மாறிலி ggml_tensor * src0 = t->src[0];
    மாறிலி ggml_tensor * src1 = t->src[1];

    எனில் (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        மாறிலி ggml_tensor src1_c = ggml_hexagon_tensor_collapse_rows(src1);
        திரும்பு ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, பொய், பொய்);
    }

    மாறிலி தருக்க is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    மாறிலி தருக்க is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    திரும்பு ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

நிலையான தருக்க is_supported_mul_mat_nx_kernel(மாறிலி ggml_tensor * src0, மாறிலி கட்டமைப்பு htp_mm_kernel_params * kparams) {
    எனில் (kparams->n_hmx) {
        திரும்பு kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    எனில் (!ggml_hexagon_tensor_is_tiled(src0)) {
        திரும்பு பொய்;
    }

    திரும்பு kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
}

நிலையான தருக்க is_supported_mul_mat_id_nx_kernel(மாறிலி ggml_tensor * src0, மாறிலி கட்டமைப்பு htp_mm_kernel_params * kparams) {
    எனில் (kparams->n_hmx) {
        திரும்பு kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    எனில் (!ggml_hexagon_tensor_is_tiled(src0)) {
        திரும்பு பொய்;
    }

    திரும்பு kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK;
}

நிலையான தருக்க is_mergeable_mul_mat(மாறிலி ggml_tensor * t) {
    எனில் (t->op != GGML_OP_MUL_MAT) திரும்பு பொய்;
    எனில் (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) திரும்பு பொய்;

    மாறிலி ggml_tensor * src0 = t->src[0];
    மாறிலி ggml_tensor * src1 = t->src[1];
    எனில் (src0->ne[2] != 1 || src0->ne[3] != 1) திரும்பு பொய்;

    எனில் (mm_is_hmx_eligible(t)) {
        திரும்பு htp_is_hmx_weight_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    எனில் (src1->type != GGML_TYPE_F32) திரும்பு பொய்;

    திரும்பு ggml_hexagon_tensor_is_tiled(src0);
}

நிலையான தருக்க is_mergeable_mul_mat_pair(மாறிலி ggml_tensor * n1, மாறிலி ggml_tensor * n2) {
    எனில் (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        திரும்பு பொய்;
    }
    எனில் (n1->src[1] != n2->src[1]) {
        திரும்பு பொய்;
    }
    எனில் (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        திரும்பு பொய்;
    }
    எனில் (n1->src[0]->type != n2->src[0]->type) {
        திரும்பு பொய்;
    }
    எனில் (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        திரும்பு பொய்;
    }
    எனில் (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        திரும்பு பொய்;
    }
    திரும்பு உண்மை;
}

நிலையான தருக்க is_mergeable_mul_mat_id(மாறிலி ggml_tensor * t) {
    எனில் (t->op != GGML_OP_MUL_MAT_ID) திரும்பு பொய்;

    மாறிலி ggml_tensor * src0 = t->src[0];
    எனில் (mm_is_hmx_eligible(t)) {
        திரும்பு htp_is_hmx_weight_type(src0->type);
    }

    திரும்பு ggml_hexagon_tensor_is_tiled(src0);
}

நிலையான தருக்க is_mergeable_mul_mat_id_pair(மாறிலி ggml_tensor * n1, மாறிலி ggml_tensor * n2) {
    எனில் (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        திரும்பு பொய்;
    }
    எனில் (n1->src[1] != n2->src[1]) {
        திரும்பு பொய்;
    }
    எனில் (n1->src[2] != n2->src[2]) {
        திரும்பு பொய்;
    }
    எனில் (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        திரும்பு பொய்;
    }
    எனில் (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        திரும்பு பொய்;
    }
    எனில் (n1->src[0]->type != n2->src[0]->type) {
        திரும்பு பொய்;
    }
    எனில் (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        திரும்பு பொய்;
    }
    எனில் (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        திரும்பு பொய்;
    }
    திரும்பு உண்மை;
}

நிலையான ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    எனில் (sess->last_error > HTP_STATUS_OK) {
        திரும்பு GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    மாறிலி std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    தருக்க cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
    எனில் (cache_hit) {
        nodes_ptr = &sess->cached_nodes;
    } இல்லையெனில் {
        // Tag fusable tensors in graph
        க்கு (முழு i = 0; i < graph->n_nodes; i++) {
            auto * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            எனில் (!extra) தொடர்;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            எனில் (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } இல்லையெனில் எனில் ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                எனில் ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_முனை_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        க்கு (முழு i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            எனில் (!op_is_compute(n)) {
                தொடர்;
            }

            htp_opnode முனை(HTP_OP_INVALID, n);
            முனை.opcode = op_remap_to_htp(n);
            எனில் (முனை.opcode == HTP_OP_MUL_MAT || முனை.opcode == HTP_OP_MUL_MAT_ID || முனை.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = முனை.முனை->src[0];
                எனில் (ggml_hexagon_needs_repack(src0)) {
                    எனில் (opt_verbose) {
                        எழுத்து dims_buf[64];
                        எழுத்து strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, அளவு(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, அளவு(strides_buf), src0);
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

            எனில் (முனை.opcode == HTP_OP_MUL_MAT || முனை.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    முனை.முனை->src[0], முனை.முனை->src[1], முனை.முனை,
                    (கட்டமைப்பு htp_mm_kernel_params *)முனை.kernel_params
                );
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_MUL || முனை.opcode == HTP_OP_ADD || முனை.opcode == HTP_OP_ADD_ID || முனை.opcode == HTP_OP_SUB || முனை.opcode == HTP_OP_DIV) {
                மாறிலி ggml_tensor * src1 = முனை.முனை->src[1];
                மாறிலி ggml_tensor * src2 = (முனை.opcode == HTP_OP_ADD_ID) ? முனை.முனை->src[2] : nullptr;
                GGML_ASSERT(ggml_hexagon_precompute_இருமம்_params(sess,
                    முனை.opcode, முனை.முனை->src[0], src1, முனை.முனை,
                    (கட்டமைப்பு htp_இருமம்_kernel_params *) முனை.kernel_params, src2));
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    முனை.முனை,
                    (கட்டமைப்பு htp_fa_kernel_params *)முனை.kernel_params
                );
            } இல்லையெனில் எனில் (htp_op_is_unary(முனை.opcode)) {
                auto inputs = முனை.get_inputs();
                மாறிலி கட்டமைப்பு ggml_tensor * src0 = inputs[0];
                மாறிலி கட்டமைப்பு ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    முனை.opcode, src0, src1, முனை.dst(),
                    (கட்டமைப்பு htp_unary_kernel_params *)முனை.kernel_params
                );
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_GET_ROWS) {
                ggml_hexagon_precompute_get_rows_params(sess,
                    முனை.முனை->src[0], முனை.முனை->src[1], முனை.dst(),
                    (கட்டமைப்பு htp_get_rows_kernel_params *)முனை.kernel_params
                );
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_SET_ROWS) {
                ggml_hexagon_precompute_set_rows_params(sess,
                    முனை.முனை->src[0], முனை.முனை->src[1], முனை.dst(),
                    (கட்டமைப்பு htp_set_rows_kernel_params *)முனை.kernel_params
                );
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    முனை.முனை,
                    (கட்டமைப்பு htp_rope_kernel_params *)முனை.kernel_params
                );
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    முனை.முனை->src[0], முனை.முனை->src[1], முனை.dst(),
                    (கட்டமைப்பு htp_ssm_conv_kernel_params *)முனை.kernel_params
                );
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    முனை.முனை,
                    (கட்டமைப்பு htp_softmax_kernel_params *)முனை.kernel_params
                );
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    முனை.முனை,
                    (கட்டமைப்பு htp_gdn_kernel_params *)முனை.kernel_params
                );
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_ARGSORT || முனை.opcode == HTP_OP_உச்சி_K) {
                ggml_hexagon_precompute_sort_params(sess,
                    முனை.முனை,
                    முனை.opcode == HTP_OP_உச்சி_K,
                    (கட்டமைப்பு htp_sort_kernel_params *) முனை.kernel_params
                );
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_POOL_2D || முனை.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, முனை.முனை->src[0], முனை.dst(),
                    (கட்டமைப்பு htp_pool_2d_kernel_params *)முனை.kernel_params,
                    முனை.opcode == HTP_OP_POOL_1D);
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    முனை.முனை,
                    (கட்டமைப்பு htp_concat_kernel_params *) முனை.kernel_params
                );
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_CPY || முனை.opcode == HTP_OP_CPY_FENCE) {
                மாறிலி தருக்க ok = ggml_hexagon_precompute_cpy_params(sess,
                    முனை.முனை,
                    (கட்டமைப்பு htp_copy_kernel_params *) முனை.kernel_params
                );
                மாறிலி auto * kparams = (மாறிலி கட்டமைப்பு htp_copy_kernel_params *) முனை.kernel_params;
                எனில் (ok && முனை.opcode == HTP_OP_CPY && kparams->total_elems == 0) {
                    தொடர்;
                }
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_IM2COL) {
                ggml_hexagon_precompute_im2col_params(sess,
                    முனை.முனை,
                    (கட்டமைப்பு htp_im2col_kernel_params *) முனை.kernel_params
                );
            } இல்லையெனில் எனில் (முனை.opcode == HTP_OP_IM2COL_3D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    முனை.முனை,
                    (கட்டமைப்பு htp_im2col_kernel_params *) முனை.kernel_params
                );
            }
            computed_nodes.push_back(std::move(முனை));
        }

        எனில் (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } இல்லையெனில் {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    க்கு (மாறிலி auto & முனை : *nodes_ptr) {
        sess->enqueue_op(முனை);
    }

    எனில் (sess->last_error > HTP_STATUS_OK) {
        திரும்பு GGML_STATUS_FAILED;
    }

    திரும்பு GGML_STATUS_SUCCESS;
}

நிலையான வெறுமை ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    எனில் (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
}

எண்ணிக்கை ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

கட்டமைப்பு ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

கட்டமைப்பு ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    வெறுமை reset() {
        ranges.clear();
    }

    வெறுமை add(மாறிலி ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    தருக்க check(மாறிலி ggml_hexagon_mem_range & mr) மாறிலி {
        க்கு (மாறிலி auto & cmp : ranges) {
            எனில் (mr.pb != cmp.pb) {
                தொடர்;
            }
            எனில் (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                தொடர்;
            }
            எனில் (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                திரும்பு பொய்;
            }
        }
        திரும்பு உண்மை;
    }
};

நிலையான ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(மாறிலி ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    மாறிலி ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    எனில் (tensor->buffer) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } இல்லையெனில் {
        mr = {
            /*.pb =*/ (uint64_t) base,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    திரும்பு mr;
}

நிலையான வெறுமை ggml_hexagon_mem_ranges_add_முனை(ggml_hexagon_mem_ranges & mrs, மாறிலி htp_opnode & முனை) {
    எனில் (முனை.is_empty()) திரும்பு;

    க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
        எனில் (முனை.முனை->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(முனை.முனை->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    க்கு (மாறிலி auto * fused : முனை.fused) {
        க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
            எனில் (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(முனை.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

நிலையான தருக்க ggml_hexagon_mem_ranges_check_முனை(மாறிலி ggml_hexagon_mem_ranges & mrs, மாறிலி htp_opnode & முனை) {
    எனில் (முனை.is_empty()) திரும்பு உண்மை;

    க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
        எனில் (முனை.முனை->src[i]) {
            எனில் (!mrs.check(ggml_hexagon_mem_range_from_tensor(முனை.முனை->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                திரும்பு பொய்;
            }
        }
    }
    க்கு (மாறிலி auto * fused : முனை.fused) {
        க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
            எனில் (fused->src[i]) {
                எனில் (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    திரும்பு பொய்;
                }
            }
        }
    }
    திரும்பு mrs.check(ggml_hexagon_mem_range_from_tensor(முனை.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

நிலையான std::vector<முழு> ggml_hexagon_graph_optimize_reorder(மாறிலி std::vector<htp_opnode> & nodes) {
    மாறிலி முழு n = nodes.size();

    std::vector<முழு> res;
    res.reserve(n);

    std::vector<தருக்க> used(n, பொய்);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    க்கு (முழு i0 = 0; i0 < n; i0++) {
        எனில் (used[i0]) {
            தொடர்;
        }

        மாறிலி auto & node0 = nodes[i0];

        எனில் (!node0.stackable()) {
            res.push_back(i0);
            used[i0] = உண்மை;
            தொடர்;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr முழு N_FORWARD = 16;

        std::vector<முழு> அடுக்கு;
        அடுக்கு.push_back(i0);

        mrs.reset();

        க்கு (முழு i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            எனில் (used[i1]) {
                தொடர்;
            }

            மாறிலி auto & node1 = nodes[i1];

            எனில் (node1.stackable() && node1.same_உள்ளீடு(node0) && ggml_hexagon_mem_ranges_check_முனை(mrs, node1)) {
                அடுக்கு.push_back(i1);
            } இல்லையெனில் {
                ggml_hexagon_mem_ranges_add_முனை(mrs, node1);
            }
        }

        க்கு (முழு idx : அடுக்கு) {
            res.push_back(idx);
            used[idx] = உண்மை;
        }
    }

    திரும்பு res;
}

நிலையான வெறுமை ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    மாறிலி முழு n = gf->n_nodes;

    constexpr முழு MAX_FUSE = 16;

    எண்ணிக்கை ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    க்கு (முழு i = 0; i < n; i++) {
        htp_opnode முனை(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        எனில் (முனை.op() == GGML_OP_ADD ||
            முனை.op() == GGML_OP_NORM ||
            முனை.op() == GGML_OP_RMS_NORM) {
            ops[0] = முனை.op();

            முழு f = i + 1;
            போது (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                எனில் (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    நிறுத்து;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            க்கு (; f > 1; f--) {
                எனில் (ggml_can_fuse(gf, i, ops, f)) {
                    நிறுத்து;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            க்கு (முழு k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                முனை.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(முனை));
    }

    மாறிலி auto order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        முழு j = 0;
        க்கு (மாறிலி auto i : order) {
            மாறிலி auto & முனை = nodes[i];

            gf->nodes[j++] = முனை.முனை;

            க்கு (auto * fused : முனை.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

நிலையான uint64_t ggml_hexagon_session_சாவி(மாறிலி ggml_hexagon_session * sess) {
    திரும்பு ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

நிலையான தருக்க ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, மாறிலி ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    auto sbuf_dst = sess_dst->mmap_tensor(dst);

    எனில் (!sess_src->clone_buffer(sbuf_dst)) { திரும்பு பொய்; }

    மாறிலி uint64_t src_சாவி = ggml_hexagon_session_சாவி(sess_src);
    auto & fence_slot = sess_dst->cpy_fence_slots[src_சாவி];
    எனில் (!fence_slot) {
        fence_slot = (நிலையற்ற uint32_t *) sess_dst->alloc_fence(1);
    }

    எனில் (!sess_src->clone_buffer(sess_dst->fence_buf)) { திரும்பு பொய்; }

    எனில் (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    நிலையான ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.buffer = &sess_dst->fence_buf->backend_buffer;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.data   = (வெறுமை *) fence_slot;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = அளவு(int32_t);
    fence_tensor.nb[1]  = அளவு(int32_t);
    fence_tensor.nb[2]  = அளவு(int32_t);
    fence_tensor.nb[3]  = அளவு(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ உண்மை);

    sess_dst->add_peer(sess_src);

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, மாறிலி ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    auto sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    எனில் (!sess_dst->clone_buffer(sbuf_src)) { திரும்பு பொய்; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, மாறிலி ggml_tensor * src, ggml_tensor * dst) {
    எனில் (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        திரும்பு பொய்;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    எனில் (!dst->extra) {
        ggml_backend_buffer_init_tensor(dst->buffer, dst);
    }

    auto * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    மாறிலி auto * src_extra = static_cast<மாறிலி ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    எனில் (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        திரும்பு உண்மை;
    }

    எனில் (sess_src->phys_idx != sess_dst->phys_idx)
        திரும்பு ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    திரும்பு ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

நிலையான ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = புதிய ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_slot = (நிலையற்ற uint32_t *) sess->alloc_fence(1);

    நிலையான ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.buffer = &sess->fence_buf->backend_buffer;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.data   = (வெறுமை *) hex_event->fence_slot;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = அளவு(int32_t);
    hex_event->fence_tensor.nb[1]  = அளவு(int32_t);
    hex_event->fence_tensor.nb[2]  = அளவு(int32_t);
    hex_event->fence_tensor.nb[3]  = அளவு(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (வெறுமை *)hex_event, (வெறுமை *)hex_event->fence_slot);

    திரும்பு புதிய ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

நிலையான வெறுமை ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    எனில் (hex_event->seq == 0) {
        திரும்பு;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (வெறுமை *)hex_event, hex_event->seq, (வெறுமை *)hex_event->fence_slot);

    auto * fence = reinterpret_cast<மாறிலி நிலையற்ற std::atomic<uint32_t> *>(hex_event->fence_slot);

    எனில் ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    போது (உண்மை) {
        எனில் ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            எனில் (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            நிறுத்து;
        }
        std::this_thread::yield();
    }
}

நிலையான வெறுமை ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (வெறுமை *)hex_event);
    hex_event->fence_sess->free_fence((வெறுமை *) hex_event->fence_slot, 1);
    delete hex_event;
    delete event;
}

நிலையான வெறுமை ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

நிலையான வெறுமை ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    எனில் (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ பொய்);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (வெறுமை *)hex_event, hex_event->seq, (வெறுமை *)hex_event->fence_slot);
}

நிலையான வெறுமை ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    எனில் (hex_event->seq == 0) {
        திரும்பு;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (வெறுமை *)hex_event, hex_event->seq, (வெறுமை *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    எனில் (sess->phys_idx == hex_event->sess->phys_idx) {
        எனில் (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        திரும்பு;
    }

    sess->clone_buffer(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ உண்மை);
}

நிலையான வெறுமை ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, கட்டமைப்பு ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (முழு) tensor->buffer->usage : -1);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

நிலையான வெறுமை ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, மாறிலி கட்டமைப்பு ggml_tensor * tensor, வெறுமை * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (முழு) tensor->buffer->usage : -1);
    sess->flush_sync();
    எனில் (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

நிலையான வெறுமை ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     கட்டமைப்பு ggml_tensor * tensor,
                                                     மாறிலி வெறுமை * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (முழு) tensor->buffer->usage : -1);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

நிலையான வெறுமை ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     மாறிலி கட்டமைப்பு ggml_tensor * tensor,
                                                     வெறுமை * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (முழு) tensor->buffer->usage : -1);
    sess->flush_sync();
    எனில் (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

நிலையான கட்டமைப்பு ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ வெற்று,
    /* .graph_plan_free         = */ வெற்று,
    /* .graph_plan_update       = */ வெற்று,
    /* .graph_plan_compute      = */ வெற்று,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_record,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

நிலையான ggml_guid_t ggml_backend_hexagon_guid() {
    நிலையான ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    திரும்பு &guid;
}

தருக்க ggml_backend_is_hexagon(ggml_backend_t backend) {
    திரும்பு backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

நிலையான ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, மாறிலி எழுத்து * params) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    திரும்பு புதிய ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

நிலையான மாறிலி எழுத்து * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    திரும்பு dev_ctx->c_name();

    GGML_UNUSED(dev);
}

நிலையான மாறிலி எழுத்து * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    திரும்பு "Hexagon";
    GGML_UNUSED(dev);
}

நிலையான வெறுமை ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * விடுவி, size_t * total) {
    *விடுவி  = opt_mbuf_total;
    *total = opt_mbuf_total;

    GGML_UNUSED(dev);
}

நிலையான எண்ணிக்கை ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    திரும்பு GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

நிலையான வெறுமை ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, கட்டமைப்பு ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ உண்மை,
        /* .host_buffer           = */ பொய்,
        /* .buffer_from_host_ptr  = */ பொய்,
        /* .events                = */ உண்மை,
        /* .mmap_support          = */ பொய்,
    };
}

நிலையான ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    திரும்பு &dev_ctx->buffer_type;
}

நிலையான ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    எனில் (!opt_hostbuf) {
        திரும்பு வெற்று;
    }
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    திரும்பு &dev_ctx->host_buffer_type;
}

நிலையான தருக்க ggml_hexagon_supported_cpy(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    கட்டமைப்பு htp_copy_kernel_params kparams;
    திரும்பு ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

நிலையான தருக்க ggml_hexagon_supported_cont(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    கட்டமைப்பு htp_copy_kernel_params kparams;
    திரும்பு ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

நிலையான தருக்க ggml_hexagon_supported_repeat(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    GGML_UNUSED(sess);
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    // Support f32 and f16
    எனில் (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) திரும்பு பொய்;

    // src and dst must be the same type
    எனில் (src0->type != dst->type) திரும்பு பொய்;

    // dst dims must be multiples of src dims
    எனில் (dst->ne[0] % src0->ne[0] != 0) திரும்பு பொய்;
    எனில் (dst->ne[1] % src0->ne[1] != 0) திரும்பு பொய்;
    எனில் (dst->ne[2] % src0->ne[2] != 0) திரும்பு பொய்;
    எனில் (dst->ne[3] % src0->ne[3] != 0) திரும்பு பொய்;

    // require contiguous tensors (no transposition)
    எனில் (ggml_is_transposed(src0) || ggml_is_transposed(dst)) திரும்பு பொய்;

    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_hexagon_supported_concat(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    கட்டமைப்பு htp_concat_kernel_params kparams;
    திரும்பு ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

நிலையான தருக்க ggml_hexagon_supported_fill(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    மாறிலி கட்டமைப்பு ggml_tensor * dst = op;

    எனில் (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
    GGML_UNUSED(sess);
}

நிலையான தருக்க ggml_hexagon_supported_roll(மாறிலி கட்டமைப்பு ggml_hexagon_session * sess, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    GGML_UNUSED(sess);

    மாறிலி கட்டமைப்பு ggml_tensor * src0 = op->src[0];
    மாறிலி கட்டமைப்பு ggml_tensor * dst  = op;

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    எனில் (!ggml_are_same_shape(src0, dst)) {
        திரும்பு பொய்;
    }

    எனில் (src0->nb[0] != ggml_type_size(src0->type) || dst->nb[0] != ggml_type_size(dst->type)) {
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
நிலையான தருக்க ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, மாறிலி கட்டமைப்பு ggml_tensor * op) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    // reject ops that match the filter
    எனில் (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        திரும்பு பொய்;
    }

    தருக்க supp = பொய்;
    தேர்வு (op->op) {
        நிலை GGML_OP_NONE:
        நிலை GGML_OP_RESHAPE:
        நிலை GGML_OP_VIEW:
        நிலை GGML_OP_PERMUTE:
        நிலை GGML_OP_TRANSPOSE:
            supp = உண்மை;
            நிறுத்து;

        நிலை GGML_OP_MUL:
        நிலை GGML_OP_ADD:
        நிலை GGML_OP_SUB:
        நிலை GGML_OP_DIV:
            supp = ggml_hexagon_supported_இருமம்(sess, op);
            நிறுத்து;

        நிலை GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            நிறுத்து;

        நிலை GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            நிறுத்து;

        நிலை GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            நிறுத்து;

        நிலை GGML_OP_NORM:
        நிலை GGML_OP_L2_NORM:
        நிலை GGML_OP_RMS_NORM:
        நிலை GGML_OP_SCALE:
        நிலை GGML_OP_CLAMP:
        நிலை GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            நிறுத்து;

        நிலை GGML_OP_SQR:
        நிலை GGML_OP_SQRT:
        நிலை GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            நிறுத்து;

        நிலை GGML_OP_SUM:
            supp = ggml_hexagon_supported_sum(sess, op);
            நிறுத்து;

        நிலை GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            நிறுத்து;

        நிலை GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            நிறுத்து;

        நிலை GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            நிறுத்து;

        நிலை GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            நிறுத்து;

        நிலை GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            நிறுத்து;

        நிலை GGML_OP_UNARY:
            தேர்வு (ggml_get_unary_op(op)) {
                நிலை GGML_UNARY_OP_NEG:
                நிலை GGML_UNARY_OP_EXP:
                நிலை GGML_UNARY_OP_SIGMOID:
                நிலை GGML_UNARY_OP_SOFTPLUS:
                நிலை GGML_UNARY_OP_TANH:
                நிலை GGML_UNARY_OP_ABS:
                நிலை GGML_UNARY_OP_SILU:
                நிலை GGML_UNARY_OP_GELU:
                நிலை GGML_UNARY_OP_GELU_QUICK:
                நிலை GGML_UNARY_OP_GELU_ERF:
                நிலை GGML_UNARY_OP_RELU:
                நிலை GGML_UNARY_OP_STEP:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    நிறுத்து;
                இயல்பு:
                    supp = பொய்;
                    நிறுத்து;
            }
            நிறுத்து;

        நிலை GGML_OP_GLU:
            தேர்வு (ggml_get_glu_op(op)) {
                நிலை GGML_GLU_OP_SWIGLU:
                நிலை GGML_GLU_OP_SWIGLU_OAI:
                நிலை GGML_GLU_OP_SWIGLU_CLAMP:
                நிலை GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    நிறுத்து;
                நிலை GGML_GLU_OP_GEGLU_QUICK:
                நிலை GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    நிறுத்து;
                இயல்பு:
                    supp = பொய்;
                    நிறுத்து;
            }
            நிறுத்து;

        நிலை GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            நிறுத்து;

        நிலை GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            நிறுத்து;

        நிலை GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            நிறுத்து;

        நிலை GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            நிறுத்து;

        நிலை GGML_OP_DUP:
        நிலை GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            நிறுத்து;

        நிலை GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            நிறுத்து;

        நிலை GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            நிறுத்து;

        நிலை GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            நிறுத்து;

        நிலை GGML_OP_உச்சி_K:
            supp = ggml_hexagon_supported_உச்சி_k(sess, op);
            நிறுத்து;

        நிலை GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            நிறுத்து;

        நிலை GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            நிறுத்து;

        நிலை GGML_OP_IM2COL_3D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            நிறுத்து;

        நிலை GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            நிறுத்து;

        நிலை GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            நிறுத்து;

        நிலை GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            நிறுத்து;

        நிலை GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            நிறுத்து;

        நிலை GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            நிறுத்து;

        நிலை GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            நிறுத்து;

        நிலை GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            நிறுத்து;

        நிலை GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            நிறுத்து;

        நிலை GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            நிறுத்து;

        இயல்பு:
            நிறுத்து;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    திரும்பு supp;
}

நிலையான தருக்க ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    தருக்க supp = (buft == &dev_ctx->host_buffer_type) || (buft == &dev_ctx->buffer_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    திரும்பு supp;
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ வெற்று,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ வெற்று,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    க்கு (size_t i = 0; i < opt_ndev; i++) {
        மாறிலி auto & cfg = opt_device_configs[i];
        எனில் (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } இல்லையெனில் {
            std::string peers_str;
            க்கு (மாறிலி auto & p : cfg.mdev_group) {
                எனில் (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = புதிய ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    க்கு (size_t i = 0; i < opt_ndev; i++) {
        auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        delete dev_ctx;
    }
}

நிலையான மாறிலி எழுத்து * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    திரும்பு "HTP";
    GGML_UNUSED(reg);
}

நிலையான size_t ggml_backend_hexagon_reg_get_device_மொத்தம்(ggml_backend_reg_t reg) {
    திரும்பு opt_ndev;
    GGML_UNUSED(reg);
}

நிலையான ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    எனில் (index >= opt_ndev || !hreg->devices[index].context) {
        திரும்பு nullptr;
    }

    திரும்பு &hreg->devices[index];
}

// ** communication context for tensor-split allreduce

நிலையான வெறுமை * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    எனில் (n_backends < 2 || n_backends > 4) {
        திரும்பு nullptr;
    }

    க்கு (size_t i = 0; i < n_backends; ++i) {
        எனில் (!ggml_backend_is_hexagon(backends[i])) {
            திரும்பு nullptr;
        }
    }

    க்கு (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        க்கு (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            எனில் (sess_i->phys_idx == sess_j->phys_idx) {
                திரும்பு nullptr;
            }
        }
    }

    auto * ctx = புதிய ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    நிலையான ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    க்கு (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (நிலையற்ற uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].buffer = &sess_i->fence_buf->backend_buffer;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].data   = (வெறுமை *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = அளவு(int32_t);
        ctx->fence_tensors[i].nb[1]  = அளவு(int32_t);
        ctx->fence_tensors[i].nb[2]  = அளவு(int32_t);
        ctx->fence_tensors[i].nb[3]  = அளவு(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    திரும்பு ctx;
}

நிலையான வெறுமை ggml_backend_hexagon_comm_free(வெறுமை * comm_ctx_v) {
    எனில் (!comm_ctx_v) திரும்பு;
    auto * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    க்கு (size_t i = 0; i < ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((வெறுமை *) ctx->fence_slots[i], 1);
    }
    delete ctx;
}

நிலையான தருக்க ggml_backend_hexagon_comm_allreduce_tensor(வெறுமை * comm_ctx_v, கட்டமைப்பு ggml_tensor ** tensors) {
    எனில் (opt_ar_select == 0 || !comm_ctx_v) திரும்பு பொய்;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    மாறிலி size_t n_backends = comm_ctx->n_backends;

    எனில் (n_backends < 2 || n_backends > 4) திரும்பு பொய்;

    க்கு (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        க்கு (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            எனில் (sess_i->phys_idx == sess_j->phys_idx) {
                திரும்பு பொய்;
            }
        }
    }

    க்கு (size_t i = 0; i < n_backends; i++) {
        எனில் (!tensors[i] || !tensors[i]->buffer || !ggml_backend_buffer_is_hexagon(tensors[i]->buffer)) {
            திரும்பு பொய்;
        }
        எனில் (tensors[i]->type != tensors[0]->type) {
            திரும்பு பொய்;
        }
        எனில் (!ggml_is_contiguous(tensors[i])) {
            திரும்பு பொய்;
        }
        எனில் (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            திரும்பு பொய்;
        }
    }

    எனில் (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    க்கு (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        கட்டமைப்பு htp_allreduce_kernel_params kparams;
        எனில் (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, பொய், பொய், /*is_shard_ok=*/ பொய், &kparams)) {
            திரும்பு பொய்;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    க்கு (size_t i = 1; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        எனில் ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    எனில் (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_entry = max_seq;
    எனில் (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    க்கு (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<மாறிலி ggml_tensor *> data_tensors(n_backends);
    std::vector<மாறிலி ggml_tensor *> sync_tensors(n_backends);
    க்கு (size_t i = 0; i < n_backends; i++) {
        data_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    க்கு (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], data_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        க்கு (size_t j = 0; j < n_backends; j++) {
            எனில் (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    திரும்பு உண்மை;
}

நிலையான ggml_backend_buffer_type_t ggml_backend_hexagon_split_buffer_type(முழு main_device, மாறிலி தசம * tensor_split) {
    GGML_UNUSED(tensor_split);
    auto reg = ggml_backend_hexagon_reg();
    auto dev = ggml_backend_reg_dev_get(reg, main_device);
    எனில் (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    எனில் (!dev) திரும்பு nullptr;
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    திரும்பு &dev_ctx->buffer_type;
}

நிலையான வெறுமை * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, மாறிலி எழுத்து * name) {
    GGML_UNUSED(reg);
    எனில் (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        திரும்பு (வெறுமை *) ggml_backend_hexagon_split_buffer_type;
    }
    எனில் (strcmp(name, "ggml_backend_comm_init") == 0) {
        திரும்பு (வெறுமை *) ggml_backend_hexagon_comm_init;
    }
    எனில் (strcmp(name, "ggml_backend_comm_free") == 0) {
        திரும்பு (வெறுமை *) ggml_backend_hexagon_comm_free;
    }
    எனில் (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        திரும்பு (வெறுமை *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    திரும்பு வெற்று;
}

template<typename T> std::vector<T> str_to_vec(மாறிலி எழுத்து* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    போது (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    திரும்பு v;
}

template<typename T, முழு BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    க்கு (auto i : v) { ss << i << ','; }
    auto str = ss.str(); str.pop_back(); // drop last comma
    திரும்பு str;
}

நிலையான வெறுமை ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, தருக்க discovery_supported, மாறிலி std::unordered_map<முழு, fastrpc_domain> & cdsp_map) {
    எனில் (discovery_supported) {
        auto it = cdsp_map.find(cfg.physical_idx);
        எனில் (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } இல்லையெனில் {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } இல்லையெனில் {
        தேர்வு (cfg.physical_idx) {
            நிலை 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                நிறுத்து;
            நிலை 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                நிறுத்து;
            இயல்பு:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                நிறுத்து;
        }
    }
    க்கு (auto & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
நிலையான வெறுமை ggml_hexagon_discover_devices() {
    std::unordered_map<முழு, fastrpc_domain> cdsp_map;
    தருக்க discovery_supported = பொய்;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    முழு err = remote_system_request(&domain_info);
    எனில் (err == AEE_SUCCESS && domain_info.sys.இலக்கம்_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.இலக்கம்_domains);
        domain_info.sys.domains     = domains.data();
        domain_info.sys.max_domains = (முழு) domains.size();

        err = remote_system_request(&domain_info);
        எனில் (err == AEE_SUCCESS) {
            discovery_supported = உண்மை;
            மாறிலி முழு n_domains = std::min(domain_info.sys.இலக்கம்_domains, (முழு) domains.size());
            க்கு (முழு i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (முழு) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                எனில் (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (முழு) domains[i].type);
                    தொடர்;
                }
                எனில் (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    தொடர்;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } இல்லையெனில் {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (unsigned) err);
        }
    } இல்லையெனில் எனில் (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (unsigned) err);
    }

    // Populate domain IDs and names for all configured devices
    க்கு (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

நிலையான வெறுமை ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned முழு) HTP_TYPE_Q4_0 == (unsigned முழு) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned முழு) HTP_TYPE_Q4_1 == (unsigned முழு) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned முழு) HTP_TYPE_Q8_0 == (unsigned முழு) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned முழு) HTP_TYPE_MXFP4 == (unsigned முழு) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned முழு) HTP_TYPE_IQ4_NL == (unsigned முழு) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned முழு) HTP_TYPE_Q4_K == (unsigned முழு) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned முழு) HTP_TYPE_Q5_K == (unsigned முழு) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned முழு) HTP_TYPE_Q6_K == (unsigned முழு) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned முழு) HTP_TYPE_Q3_K == (unsigned முழு) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned முழு) HTP_TYPE_Q2_K == (unsigned முழு) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned முழு) HTP_TYPE_Q1_0 == (unsigned முழு) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned முழு) HTP_TYPE_Q2_0 == (unsigned முழு) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned முழு) HTP_TYPE_BF16 == (unsigned முழு) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    மாறிலி எழுத்து * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    மாறிலி எழுத்து * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    மாறிலி எழுத்து * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    மாறிலி எழுத்து * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    மாறிலி எழுத்து * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    மாறிலி எழுத்து * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    மாறிலி எழுத்து * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    மாறிலி எழுத்து * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    மாறிலி எழுத்து * str_etm      = getenv("GGML_HEXAGON_ETM");
    மாறிலி எழுத்து * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    மாறிலி எழுத்து * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    மாறிலி எழுத்து * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    மாறிலி எழுத்து * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    மாறிலி எழுத்து * str_fa_head_split = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    மாறிலி எழுத்து * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    மாறிலி எழுத்து * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    மாறிலி எழுத்து * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    மாறிலி எழுத்து * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    மாறிலி எழுத்து * str_arch     = getenv("GGML_HEXAGON_ARCH");
    மாறிலி எழுத்து * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    மாறிலி எழுத்து * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    மாறிலி எழுத்து * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    மாறிலி எழுத்து * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    மாறிலி எழுத்து * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    மாறிலி எழுத்து * str_power_mode = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    எனில் (!str_arch) {
        முழு err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        எனில் (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } இல்லையெனில் {
            எனில் (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } இல்லையெனில் எனில் (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } இல்லையெனில் {
        எனில் (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, வெற்று, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? புதிய std::regex(str_opfilter, RE_ICASE) : வெற்று;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, வெற்று, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, வெற்று, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, வெற்று, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  வெற்று, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    எனில் (str_power_mode) {
        முழு m = htp_power_mode_from_str(str_power_mode);
        எனில் (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_mode);
        } இல்லையெனில் {
            opt_power_mode = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, வெற்று, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_head_split = str_fa_head_split ? atoi(str_fa_head_split)       : opt_fa_head_split;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    எனில் (str_mbuf) {
        மாறிலி எழுத்து * p = str_mbuf;
        க்கு (முழு idx = 0; idx < 3 && p && *p; idx++) {
            போது (*p == ' ') p++;
            மாறிலி எழுத்து * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : strlen(p);
            போது (len > 0 && p[len - 1] == ' ') len--;
            எனில் (len > 0) {
                std::string token(p, len);
                எனில் (idx == 0) opt_mbuf_dyn    = parse_size(token.c_str());
                எனில் (idx == 1) opt_mbuf_static = parse_size(token.c_str());
                எனில் (idx == 2) opt_mbuf_total  = parse_size(token.c_str());
            }
            எனில் (!comma) நிறுத்து;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_size(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    மாறிலி எழுத்து * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    எனில் (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    எனில் (str_devices && str_devices[0] != '\0') {
        தருக்க is_single_எண் = உண்மை;
        க்கு (முழு i = 0; str_devices[i] != '\0'; i++) {
            எனில் (!isdigit((unsigned எழுத்து)str_devices[i])) {
                is_single_எண் = பொய்;
                நிறுத்து;
            }
        }
        எனில் (is_single_எண்) {
            முழு n = atoi(str_devices);
            எனில் (n < 1) n = 1;
            எனில் (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            க்கு (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (முழு)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.clear();
            }
        } இல்லையெனில் {
            std::string s_devices(str_devices);
            std::vector<std::string> பொருட்கள்;
            std::string curr_item;
            முழு bracket_depth = 0;
            க்கு (எழுத்து ch : s_devices) {
                எனில் (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } இல்லையெனில் எனில் (ch == ']') {
                    எனில் (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } இல்லையெனில் எனில் (ch == ',' && bracket_depth == 0) {
                    size_t s = curr_item.find_first_not_of(" \t\r\n");
                    size_t e = curr_item.find_last_not_of(" \t\r\n");
                    எனில் (s != std::string::npos) {
                        பொருட்கள்.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.clear();
                } இல்லையெனில் {
                    curr_item += ch;
                }
            }
            size_t s = curr_item.find_first_not_of(" \t\r\n");
            size_t e = curr_item.find_last_not_of(" \t\r\n");
            எனில் (s != std::string::npos) {
                பொருட்கள்.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            க்கு (மாறிலி auto & item : பொருட்கள்) {
                size_t b_open  = item.find('[');
                size_t b_close = item.rfind(']');

                எனில் (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    முழு virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    எனில் (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        முயல் {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } பிடி (...) {
                            virt = 0;
                        }
                    } இல்லையெனில் {
                        size_t dev_colon = dev_name.find(':');
                        எனில் (dev_colon != std::string::npos) {
                            முயல் {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } பிடி (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<முழு> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    போது (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        எனில் (ps == std::string::npos) தொடர்;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        எனில் (dash_pos != std::string::npos) {
                            முயல் {
                                முழு p_start = std::stoi(p_part.substr(0, dash_pos));
                                முழு p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                க்கு (முழு p = p_start; p <= p_end; p++) {
                                    எனில் (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } பிடி (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } இல்லையெனில் {
                            முயல் {
                                முழு p = std::stoi(p_part);
                                எனில் (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } பிடி (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    எனில் (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    எனில் (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        auto & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.clear();

                        க்கு (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_group.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } இல்லையெனில் {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } இல்லையெனில் எனில் (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    size_t colon_pos = rest.find(':');
                    முழு phys = 0;
                    முழு virt = 0;
                    முயல் {
                        எனில் (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } இல்லையெனில் {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } பிடி (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        தொடர்;
                    }

                    எனில் (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_group.clear();
                        opt_ndev++;
                    } இல்லையெனில் {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } இல்லையெனில் {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", item.c_str());
                }
            }
        }
    } இல்லையெனில் {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_group.clear();
    }

#எனில் defined(__ANDROID__)
    எனில் (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    எனில் (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            auto v  = str_to_vec<uint32_t>(str_profile);
            தேர்வு (v.size()) {
                நிலை 1:  opt_profile = v[0]; திரும்பு opt_pmu_evt; // mode with default pmu events
                நிலை 8:  opt_profile = 2;    திரும்பு v;           // mode with custom  pmu events
                இயல்பு: opt_profile = 0;    திரும்பு {};          // garbage input
            }}();
        எனில் (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = புதிய ggml_hexagon_registry(reg);
}

நிலையான மாறிலி கட்டமைப்பு ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_மொத்தம்,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(வெறுமை) {
    நிலையான தருக்க initialized = பொய்;

    நிலையான ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ வெற்று };

    {
        நிலையான std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        எனில் (!initialized) {
            auto nErr = htpdrv_init();
            எனில் (nErr != AEE_SUCCESS) {
                திரும்பு வெற்று;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = உண்மை;
    }

    திரும்பு &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
