#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ USE_MATH_ਨਿਰਧਾਰਤ_ਕਰਦਾ

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <inttypes.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <time.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <atomic>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <chrono>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <thread>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdexcept>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <sstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <iomanip>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_set>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <regex>
#ਅਨ੍ਤਰ੍ਭਾਵ <queue>
#ਅਨ੍ਤਰ੍ਭਾਵ <deque>
#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <initializer_list>

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ M_LOG2E
#    define M_LOG2E 1.44269504088896340736
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
#    define WIN32_LEAN_AND_ਔਸਤ
#    ifndef NOMINMAX
#       define NOMINMAX
#    endif
#    include <windows.h>
#    include <sal.h>
#ਨਹੀਂਤਾਂ
#    include <semaphore.h>
#    include <unistd.h>
#ਜੇ_ਅੰਤ

#ਹਦਾਇਤ clang diagnostic ignored "-Wnested-anon-types"
#ਹਦਾਇਤ clang diagnostic ignored "-Wlanguage-extension-token"
#ਹਦਾਇਤ clang diagnostic ignored "-Wgnu-anonymous-struct"
#ਹਦਾਇਤ clang diagnostic ignored "-Wmicrosoft-enum-value"

#ਅਨ੍ਤਰ੍ਭਾਵ <AEEStdErr.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <dspqueue.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <rpcmem.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_IMPL_CPP
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-alloc.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-hexagon.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-quants.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-opnode.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/matmul-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/binary-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/flash-attn-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/unary-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/get-rows-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/set-rows-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/softmax-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/pool-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/rope-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/ssm-conv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/gated-delta-net-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/argsort-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/concat-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/cpy-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/im2col-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp_iface.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-drv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-repack.h"

ਵਰਤੋਂ intvec  = std::vector<ਪੂਰਨ>;
ਵਰਤੋਂ uintvec = std::vector<ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ>;
ਵਰਤੋਂ u32vec  = std::vector<uint32_t>;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_HEXAGON_MAX_SESSIONS          16

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_HEXAGON_FENCE_ਥਾਂ_ਮਾਪ       128

ਬਣਤਰ ggml_hexagon_device_config {
    ਪੂਰਨ         physical_idx = 0;
    ਪੂਰਨ         virtual_idx  = 0;
    ਪੂਰਨ         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_ਸਮੂਹ;
};

ਸਥਿਰ ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

ਸਥਿਰ ਪੂਰਨ    opt_arch    = 0; // autodetect
ਸਥਿਰ size_t opt_ndev    = 1;
ਸਥਿਰ size_t opt_nhvx    = 0; // use all
ਸਥਿਰ ਪੂਰਨ    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
ਸਥਿਰ size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
ਸਥਿਰ ਪੂਰਨ    opt_etm     = 0;
ਸਥਿਰ ਪੂਰਨ    opt_verbose = 0;
ਸਥਿਰ ਪੂਰਨ    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
ਸਥਿਰ ਤਾਰਕਿਕ   opt_hostbuf = ਝੂਠ;
ਸਥਿਰ ਤਾਰਕਿਕ   opt_dma64   = ਝੂਠ;
ਸਥਿਰ ਪੂਰਨ    opt_power_ਬਹੁਲਕ = HTP_POWER_ਬਹੁਲਕ_ਮਿਆਦ; // DCVS/HMX power profile requested at session start

ਸਥਿਰ ਅਚਲ std::pair<ਅਚਲ ਅੱਖਰ *, ਪੂਰਨ> htp_power_ਬਹੁਲਕ_ਨਾਂ[] = {
    { "low_power_saver",            HTP_POWER_ਬਹੁਲਕ_ਨੀਵਾਂ_POWER_SAVER },
    { "power_saver",                HTP_POWER_ਬਹੁਲਕ_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_ਬਹੁਲਕ_ਉੱਚਾ_POWER_SAVER },
    { "low_balanced",               HTP_POWER_ਬਹੁਲਕ_ਨੀਵਾਂ_ਸੰਤੁਲਿਤ },
    { "balanced",                   HTP_POWER_ਬਹੁਲਕ_ਸੰਤੁਲਿਤ },
    { "high_performance",           HTP_POWER_ਬਹੁਲਕ_ਉੱਚਾ_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_ਬਹੁਲਕ_SUSTAINED_ਉੱਚਾ_PERFORMANCE },
    { "burst",                      HTP_POWER_ਬਹੁਲਕ_ਮਿਆਦ },
    { "default",                    HTP_POWER_ਬਹੁਲਕ_ਮਿਆਦ }, // flip this to change the default mode
};

ਸਥਿਰ ਪੂਰਨ htp_power_ਬਹੁਲਕ_from_str(ਅਚਲ ਅੱਖਰ * s) {
    ਲਈ (ਸਵੈਚਾਲਿਤ & e : htp_power_ਬਹੁਲਕ_ਨਾਂ) {
#ਜੇ defined(_MSC_VER)
        ਜੇ (_stricmp(s, e.ਪਹਿਲਾ) == 0) {
#ਨਹੀਂਤਾਂ
        ਜੇ (strcasecmp(s, e.ਪਹਿਲਾ) == 0) {
#ਜੇ_ਅੰਤ
            ਵਾਪਸ e.second;
        }
    }
    ਵਾਪਸ -1;
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * htp_power_ਬਹੁਲਕ_ਮੰਜ਼ਿਲ_str(ਪੂਰਨ mode) {
    ਲਈ (ਸਵੈਚਾਲਿਤ & e : htp_power_ਬਹੁਲਕ_ਨਾਂ) {
        ਜੇ (e.second == mode) {
            ਵਾਪਸ e.ਪਹਿਲਾ;
        }
    }
    ਵਾਪਸ "unknown";
}

ਸਥਿਰ size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
ਸਥਿਰ size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
ਸਥਿਰ size_t opt_mbuf_ਕੁੱਲ  = 0;                        // total buffer space limit (0 = unconstrained)

ਸਥਿਰ ਪੂਰਨ    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ਸਥਿਰ ਪੂਰਨ    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ਸਥਿਰ ਪੂਰਨ    opt_fa_ਸਿਰ_ਵੰਡੋ = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
ਸਥਿਰ ਪੂਰਨ    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
ਸਥਿਰ ਪੂਰਨ    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
ਸਥਿਰ ਪੂਰਨ    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
ਸਥਿਰ u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

ਸਥਿਰ ਪੂਰਨ opt_opbatch  = 1280; // max number of ops in a batch
ਸਥਿਰ ਪੂਰਨ opt_opqueue  = 8;   // max number of pending batches
ਸਥਿਰ ਪੂਰਨ opt_optrace  = 0;    // trace buffer size per thread (0 means default)
ਸਥਿਰ ਪੂਰਨ opt_oppoll   = 0;    // polling for batch completions
ਸਥਿਰ ਪੂਰਨ opt_opfusion  = 1;    // enable/disable op fusion
ਸਥਿਰ ਪੂਰਨ opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

ਗਿਣਤੀ ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ਜੋੜੋ = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ਜੋੜੋ   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_hexagon_is_fusion_enabled(ਪੂਰਨ flag) {
    ਜੇ (opt_opfusion <= 0) ਵਾਪਸ ਝੂਠ;
    ਜੇ (opt_opfusion == 1) ਵਾਪਸ ਸੱਚ; // 1 enables all
    ਵਾਪਸ (opt_opfusion & flag) != 0;
}

ਸਥਿਰ std::regex* opt_opfilter = ਨਿਰਾ; // regex of ops to not claim

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HEX_VERBOSE(...) \
    ਜੇ (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ਸਥਿਤੀ_ਮੰਜ਼ਿਲ_str(uint32_t ਸਥਿਤੀ) {
    ਚੋਣ (ਸਥਿਤੀ) {
        ਹਾਲਤ HTP_ਸਥਿਤੀ_OK:
            ਵਾਪਸ "OK";
        ਹਾਲਤ HTP_ਸਥਿਤੀ_NO_SUPPORT:
            ਵਾਪਸ "NO-SUPPORT";
        ਹਾਲਤ HTP_ਸਥਿਤੀ_INVAL_PARAMS:
            ਵਾਪਸ "INVAL-PARAMS";
        ਹਾਲਤ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL:
            ਵਾਪਸ "VTCM-TOO-SMALL";
        ਹਾਲਤ HTP_ਸਥਿਤੀ_ਅੰਦਰੂਨੀ_ERR:
            ਵਾਪਸ "INTERNAL-ERROR";
        ਮੂਲ:
            ਵਾਪਸ "UNKNOWN";
    }
}

// ** debug helpers

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_dump_op_exec(ਅਚਲ std::string &sess_name, ਅਚਲ htp_opnode & ਗੰਢ, ਅਚਲ uint32_t req_flags) {
    ਜੇ (!opt_verbose) ਵਾਪਸ;

    htp_opformat fmt(ਗੰਢ);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                ਗੰਢ.op_name().c_str(), fmt.ਨਾਂ, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_dump_op_supp(ਅਚਲ std::string &sess_name, ਅਚਲ ਬਣਤਰ ggml_tensor * op, ਤਾਰਕਿਕ supp) {
    ਜੇ (!opt_verbose) ਵਾਪਸ;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.ਨਾਂ, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * htp_event_name(uint16_t id) {
    ਚੋਣ (id) {
        ਹਾਲਤ HTP_TRACE_EVT_DMA:            ਵਾਪਸ "DMA";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_COMP:       ਵਾਪਸ "HVX_COMP";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_A_QUANT:    ਵਾਪਸ "HVX_A_QUANT";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_A_PREP:     ਵਾਪਸ "HVX_A_PREP";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_W_DEQUANT:  ਵਾਪਸ "HVX_W_DEQUANT";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_W_PREP:     ਵਾਪਸ "HVX_W_PREP";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_O_PROC:     ਵਾਪਸ "HVX_O_PROC";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_FA_QK:      ਵਾਪਸ "HVX_QK_FA";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_FA_SFM:     ਵਾਪਸ "HVX_SFM_FA";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_FA_Q_PREP:  ਵਾਪਸ "HVX_Q_PREP";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_FA_K_PREP:  ਵਾਪਸ "HVX_K_PREP";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_FA_V_PREP:  ਵਾਪਸ "HVX_V_PREP";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_GDN_PREP:   ਵਾਪਸ "HVX_GDN_PREP";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_GDN_SOLVE:  ਵਾਪਸ "HVX_GDN_SOLVE";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_GDN_V_PREP: ਵਾਪਸ "HVX_GDN_V_PREP";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_GDN_D_PREP: ਵਾਪਸ "HVX_GDN_D_PREP";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_GDN_ਬਾਹਰਲਾ:    ਵਾਪਸ "HVX_GDN_OUT";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_GDN_STATE:  ਵਾਪਸ "HVX_GDN_STATE";
        ਹਾਲਤ HTP_TRACE_EVT_HVX_GDN_REM:    ਵਾਪਸ "HVX_GDN_REM";
        ਹਾਲਤ HTP_TRACE_EVT_HMX_COMP:       ਵਾਪਸ "HMX_COMP";
        ਹਾਲਤ HTP_TRACE_EVT_L2FLUSH:        ਵਾਪਸ "L2FLUSH";
        ਹਾਲਤ HTP_TRACE_EVT_INIT:           ਵਾਪਸ "INIT";
        ਹਾਲਤ HTP_TRACE_EVT_BUFF:           ਵਾਪਸ "BUFF";
        ਹਾਲਤ HTP_TRACE_EVT_FENCE:          ਵਾਪਸ "FENCE";
        ਮੂਲ:                           ਵਾਪਸ "UNKNOWN";
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_dump_op_prof(ਅਚਲ std::string &sess_name, ਅਚਲ htp_opnode & ਗੰਢ, ਅਚਲ htp_prof_desc & pd) {
    ਜੇ (!opt_profile) ਵਾਪਸ;

    uint32_t op_usec = pd.usecs;
    uint32_t op_ਚੱਕਰਾਂ = pd.ਚੱਕਰਾਂ_stop - pd.ਚੱਕਰਾਂ_ਸ਼ੁਰੂ;
    ਅਚਲ uint32_t * pmu = pd.pmu;

    ਅੱਖਰ pmu_str[256] = "";
    ਜੇ (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, ਆਕਾਰ(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(ਗੰਢ);
    ਦਸ਼ਮਲਵ mhz = op_usec > 0 ? (ਦਸ਼ਮਲਵ) op_ਚੱਕਰਾਂ / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            ਗੰਢ.op_name().c_str(), fmt.ਨਾਂ, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_ਚੱਕਰਾਂ, pd.ਚੱਕਰਾਂ_ਸ਼ੁਰੂ, mhz, pmu_str);
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_dump_batch_prof(ਅਚਲ std::string & sess_name, ਅਚਲ htp_opbatch_rsp & rsp) {
    uint64_t batch_ਚੱਕਰਾਂ = rsp.ਚੱਕਰਾਂ_stop - rsp.ਚੱਕਰਾਂ_ਸ਼ੁਰੂ;
    ਦਸ਼ਮਲਵ batch_mhz = rsp.usecs > 0 ? (ਦਸ਼ਮਲਵ) batch_ਚੱਕਰਾਂ / rsp.usecs : 0.0f;

    ਅੱਖਰ evt_str[256] = "----";
    ਜੇ (opt_profile == 3) {
        snprintf(evt_str, ਆਕਾਰ(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) batch_ਚੱਕਰਾਂ, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) rsp.ਚੱਕਰਾਂ_ਸ਼ੁਰੂ, batch_mhz);
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_dump_trace_events(ਅਚਲ std::string & sess_name, ਅਚਲ htp_opbatch_rsp & rsp,
                                           ਅਚਲ htp_trace_desc * trace_events, uint32_t n_traces) {
    ਜੇ (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        ਲਈ (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        ਲਈ (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            ਲਈ (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                ਅਚਲ ਸਵੈਚਾਲਿਤ & e = trace_events[t * n_traces + idx];
                ਤਾਰਕਿਕ is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.ਚੱਕਰਾਂ);
            }
        }
    }
}

ਗਿਣਤੀ ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_ਭਾਰ    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

ਬਣਤਰ ggml_hexagon_session;

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_matmul_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_mm_kernel_params * kparams
);

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_fused_matmul_ਜੋੜੋ_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src2,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_mm_kernel_params * kparams
);

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_unary_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    uint32_t op,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_unary_kernel_params * kparams
);

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_ਦੁਆਧਾਰੀ_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    uint32_t op,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_kernel_params * kparams,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src2 = ਖਾਲੀ_ਸੂਚਕ
);

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_get_ਪੰਕਤੀਆਂ_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params * kparams
);

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_ਸੈੱਟ_ਪੰਕਤੀਆਂ_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_kernel_params * kparams
);

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_softmax_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_softmax_kernel_params * kparams
);

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_rope_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_rope_kernel_params * kparams
);

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_ssm_conv_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_ssm_conv_kernel_params * kparams
);

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_gated_delta_net_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_gdn_kernel_params * kparams
);

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_ਛਾਂਟੋ_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਤਾਰਕਿਕ is_ਸਿਖਰ_k,
    ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params * kparams
);

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_pool_2d_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_pool_2d_kernel_params * kparams,
    ਤਾਰਕਿਕ is_pool_1d
);

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_concat_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_concat_kernel_params * kparams
);

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_cpy_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_copy_kernel_params * kparams
);

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_im2col_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_im2col_kernel_params * kparams
);

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_im2col_3d_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_im2col_kernel_params * kparams
);

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_fused_mmnx_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * act,
    int32_t n_weights,
    ਬਣਤਰ htp_mm_kernel_params * kparams
);

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_fused_mmidnx_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * act,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    int32_t n_weights,
    ਬਣਤਰ htp_mm_kernel_params * kparams
);

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_allreduce_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    ਤਾਰਕਿਕ has_ਜੋੜੋ,
    ਤਾਰਕਿਕ is_ਪੰਕਤੀ_bcast,
    ਤਾਰਕਿਕ is_shard_ok,
    ਬਣਤਰ htp_allreduce_kernel_params * kparams
);

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_ਪੰਕਤੀਆਂ_stride(ਅਚਲ int64_t * ne, ਅਚਲ size_t * nb, size_t * stride);
ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_matmul_can_collapse(ਅਚਲ ਬਣਤਰ ggml_tensor * src0, ਅਚਲ ਬਣਤਰ ggml_tensor * src1, ਅਚਲ ਬਣਤਰ ggml_tensor * dst);
ਸਥਿਰ ggml_tensor ggml_hexagon_tensor_collapse_ਪੰਕਤੀਆਂ(ਅਚਲ ਬਣਤਰ ggml_tensor * t);

ਸਥਿਰ ਤਾਰਕਿਕ mm_is_hmx_eligible(ਅਚਲ ggml_tensor * t);
ਸਥਿਰ htp_op_ਕੋਡ op_remap_ਮੰਜ਼ਿਲ_htp(ਅਚਲ ggml_tensor * t);
ਸਥਿਰ ਤਾਰਕਿਕ is_supported_mul_mat_nx_kernel(ਅਚਲ ggml_tensor * src0, ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams);
ਸਥਿਰ ਤਾਰਕਿਕ is_supported_mul_mat_id_nx_kernel(ਅਚਲ ggml_tensor * src0, ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams);
ਸਥਿਰ ਤਾਰਕਿਕ is_mergeable_mul_mat(ਅਚਲ ggml_tensor * t);
ਸਥਿਰ ਤਾਰਕਿਕ is_mergeable_mul_mat_pair(ਅਚਲ ggml_tensor * n1, ਅਚਲ ggml_tensor * n2);
ਸਥਿਰ ਤਾਰਕਿਕ is_mergeable_mul_mat_id(ਅਚਲ ggml_tensor * t);
ਸਥਿਰ ਤਾਰਕਿਕ is_mergeable_mul_mat_id_pair(ਅਚਲ ggml_tensor * n1, ਅਚਲ ggml_tensor * n2);

// ** backend sessions

ਬਣਤਰ ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_ਮਾਪ { 0 };
    uint32_t             flags { 0 };
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_hexagon_tensor_is_fuseable(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਜੇ (!t->extra) ਵਾਪਸ ਝੂਠ;
    ਸਵੈਚਾਲਿਤ extra = (ਅਚਲ ਬਣਤਰ ggml_hexagon_tensor_extra *) t->extra;
    ਵਾਪਸ (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_hexagon_can_rt_repack(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਜੇ (opt_rt_repack == 0 || !htp_is_tiled_ਭਾਰ_tensor(t)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (t->ਬਫਰ && ggml_backend_buft_is_host(t->ਬਫਰ->buft)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_hexagon_tensor_is_tiled(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਅਚਲ ਸਵੈਚਾਲਿਤ * extra = (ਅਚਲ ggml_hexagon_tensor_extra *) t->extra;
    ਜੇ (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਵਾਪਸ ggml_hexagon_can_rt_repack(t);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_hexagon_needs_repack(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਅਚਲ ਸਵੈਚਾਲਿਤ * extra = (ਅਚਲ ggml_hexagon_tensor_extra *) t->extra;
    ਜੇ (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ggml_hexagon_can_rt_repack(t);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_hexagon_tensors_overlap(ਅਚਲ ਬਣਤਰ ggml_tensor * a, ਅਚਲ ਬਣਤਰ ggml_tensor * b) {
    ਅਚਲ uintptr_t a0 = (uintptr_t) a->ਅੰਕੜੇ;
    ਅਚਲ uintptr_t b0 = (uintptr_t) b->ਅੰਕੜੇ;
    ਅਚਲ uintptr_t a1 = a0 + ggml_nbytes(a);
    ਅਚਲ uintptr_t b1 = b0 + ggml_nbytes(b);

    ਵਾਪਸ a0 < b1 && b0 < a1;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_hexagon_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਜੇ (t->ne[1] > 1 && (t->nb[1] & 127) != 0) ਵਾਪਸ ਝੂਠ;
    ਜੇ (t->ne[2] > 1 && (t->nb[2] & 127) != 0) ਵਾਪਸ ਝੂਠ;
    ਜੇ (t->ne[3] > 1 && (t->nb[3] & 127) != 0) ਵਾਪਸ ਝੂਠ;
    ਵਾਪਸ ਸੱਚ;
}

ਬਣਤਰ htp_opnode;

ਬਣਤਰ ggml_hexagon_opbatch;
ਬਣਤਰ ggml_hexagon_opqueue;
ਬਣਤਰ ggml_hexagon_shared_ਬਫਰ;
ਬਣਤਰ ggml_hexagon_fence_ਬਫਰ;
ਬਣਤਰ ggml_hexagon_session;
ਬਣਤਰ ggml_backend_hexagon_device_context;

ਬਣਤਰ ggml_hexagon_mdev_ਸਮੂਹ {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

ਬਣਤਰ ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    ਅਸਥਿਰ uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

ਬਣਤਰ ggml_hexagon_event {
    ggml_hexagon_session * sess         = ਖਾਲੀ_ਸੂਚਕ;
    ggml_hexagon_session * fence_sess   = ਖਾਲੀ_ਸੂਚਕ;
    ਅਸਥਿਰ uint32_t *    fence_ਥਾਂ   = ਖਾਲੀ_ਸੂਚਕ;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

ਬਣਤਰ ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         ਕਤਾਰ_id;
    ਪੂਰਨ              phys_idx;
    ਪੂਰਨ              virt_idx;
    ਤਾਰਕਿਕ             valid_session;
    ਤਾਰਕਿਕ             valid_handle;
    ਤਾਰਕਿਕ             valid_ਕਤਾਰ;
    ਤਾਰਕਿਕ             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_ਕਤਾਰ;

    std::unordered_map<ਪੂਰਨ, std::unique_ptr<ggml_hexagon_shared_ਬਫਰ>> cloned_buffers;
    std::unordered_set<ggml_hexagon_session *>                           virt_peers;
    std::unordered_set<ggml_hexagon_session *>                           phys_peers;

    uint32_t n_ਧਾਗੇ   = 0;
    uint32_t n_hvx       = 0;
    uint32_t n_hmx       = 0;
    uint64_t vtcm_ਮਾਪ   = 0;
    size_t   max_vmem    = 0;
    uint32_t fence_seq   = 0;

    std::atomic<uint64_t> batch_req_seq{0};
    std::atomic<uint64_t> batch_rsp_seq{0};
    std::atomic<uint32_t> ਆਖਰੀ_error{HTP_ਸਥਿਤੀ_OK};

    uint64_t                cached_uid = 0;
    std::vector<htp_opnode> cached_nodes;

    ggml_hexagon_mdev_ਸਮੂਹ                mdev;
    ggml_backend_dev_t                     dev       = ਖਾਲੀ_ਸੂਚਕ;
    ggml_backend_hexagon_device_context *  dev_ctx   = ਖਾਲੀ_ਸੂਚਕ;
    ggml_hexagon_fence_ਬਫਰ *            fence_buf = ਖਾਲੀ_ਸੂਚਕ;

    ggml_hexagon_session(ਅਚਲ ggml_hexagon_device_config & config, ggml_backend_dev_t dev = ਖਾਲੀ_ਸੂਚਕ, uint32_t mdev_idx = 0, uint32_t mdev_ਗਿਣਤੀ = 0) noexcept(ਝੂਠ);
    ~ggml_hexagon_session() noexcept(ਸੱਚ);

    ਅਚਲ ਅੱਖਰ* c_name() ਅਚਲ { ਵਾਪਸ name.c_str(); }

    ਖਾਲੀ allocate(ਅਚਲ ggml_hexagon_device_config & config) noexcept(ਝੂਠ);
    ਖਾਲੀ release() noexcept(ਸੱਚ);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    ਖਾਲੀ      free_fence(ਖਾਲੀ * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_ਥਾਂ = ਖਾਲੀ_ਸੂਚਕ;
    std::unordered_map<uint64_t, ਅਸਥਿਰ uint32_t *> cpy_fence_slots;

    ਖਾਲੀ ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_mdev_ਸਮੂਹ();
    ਖਾਲੀ ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_op(ਅਚਲ htp_opnode & ਗੰਢ);
    ਖਾਲੀ ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_cpy(ਅਚਲ ggml_tensor * src, ggml_tensor * dst, ਅਚਲ ggml_tensor * sync_tensor = ਖਾਲੀ_ਸੂਚਕ, uint32_t fence_seq = 0);
    ਖਾਲੀ ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_fence(ਅਚਲ ggml_tensor * sync_tensor, uint32_t fence_seq = 0, ਤਾਰਕਿਕ wait = ਸੱਚ);
    ਖਾਲੀ ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_allreduce(ਅਚਲ ggml_tensor * dst, ਅਚਲ std::vector<ਅਚਲ ggml_tensor *> & src_tensors,
                           ਅਚਲ std::vector<ਅਚਲ ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_ਇੰਦਰਾਜ = 0, uint32_t fence_seq_exit = 0);

    ਖਾਲੀ ਸ਼ੁਰੂ_batch();
    ਖਾਲੀ flush_sync(ਤਾਰਕਿਕ all = ਸੱਚ);
    ਖਾਲੀ flush_async();
    ਖਾਲੀ flush_batch(size_t min_ops = 1);
    ਖਾਲੀ flush_peers();
    ਖਾਲੀ flush_pending(ਤਾਰਕਿਕ all = ਸੱਚ);

    ggml_hexagon_shared_ਬਫਰ * mmap_tensor(ਅਚਲ ggml_tensor * t);
    ਤਾਰਕਿਕ clone_ਬਫਰ(ਅਚਲ ggml_hexagon_shared_ਬਫਰ*);
    ਖਾਲੀ release_ਬਫਰ(ਅਚਲ ggml_hexagon_shared_ਬਫਰ*);
    ਖਾਲੀ unclone_ਬਫਰ(ਅਚਲ ggml_hexagon_shared_ਬਫਰ*);

    ਖਾਲੀ ਜੋੜੋ_peer(ggml_hexagon_session * peer) {
        ਜੇ (ਇਹ->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } ਨਹੀਂਤਾਂ {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

ਬਣਤਰ ggml_backend_hexagon_device_context {
    ਪੂਰਨ                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = ਖਾਲੀ_ਸੂਚਕ;

    ggml_backend_ਬਫਰ_type ਬਫਰ_type       = {};
    ggml_backend_ਬਫਰ_type host_ਬਫਰ_type  = {};
    ggml_backend_ਬਫਰ_type fence_ਬਫਰ_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(ਪੂਰਨ dev_id, ਅਚਲ ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    ਅਚਲ ਅੱਖਰ * c_name() ਅਚਲ { ਵਾਪਸ config.name.c_str(); }

    ggml_hexagon_session * session() {
        ਜੇ (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        ਵਾਪਸ sess.get();
    }
};

ਬਣਤਰ ggml_backend_hexagon_ਬਫਰ_type_context {
    ggml_backend_hexagon_ਬਫਰ_type_context(ਅਚਲ std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        ਇਹ->dev_ctx = dev_ctx;
        ਇਹ->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

ਬਣਤਰ ggml_hexagon_rpcmem_ਖੰਡ {
    uint8_t * ਅਧਾਰ = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_ਖੰਡ(size_t size) {
        ਅਧਾਰ = (uint8_t *) rpcmem_alloc2(RPCMEM_ਢੇਰ_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        ਜੇ (!ਅਧਾਰ) {
            ਸੁੱਟੋ std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_ਮੰਜ਼ਿਲ_fd(ਅਧਾਰ);
        ਜੇ (fd < 0) {
            rpcmem_free(ਅਧਾਰ);
            ਸੁੱਟੋ std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        ਇਹ->size = size;
    }

    ~ggml_hexagon_rpcmem_ਖੰਡ() {
        ਜੇ (ਅਧਾਰ) {
            rpcmem_free(ਅਧਾਰ);
        }
    }
};

ਬਣਤਰ ggml_hexagon_shared_ਬਫਰ {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_ਖੰਡ> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    ਤਾਰਕਿਕ     mapped;
    ਤਾਰਕਿਕ     pinned;
    ਤਾਰਕਿਕ     extended;

    ਅਚਲ ਅੱਖਰ * c_name() ਅਚਲ { ਵਾਪਸ sess->c_name(); }
    uint8_t *    ਅਧਾਰ()   ਅਚਲ { ਵਾਪਸ mem ? mem->ਅਧਾਰ : ਖਾਲੀ_ਸੂਚਕ; }
    size_t       size()   ਅਚਲ { ਵਾਪਸ mem ? mem->size : 0;  }
    ਪੂਰਨ          fd()     ਅਚਲ { ਵਾਪਸ mem ? mem->fd   : -1; }

    ਖਾਲੀ mmap(ਤਾਰਕਿਕ extended = ਝੂਠ) {
        ਜੇ (!ਇਹ->mem)   ਵਾਪਸ;
        ਜੇ (ਇਹ->mapped) ਵਾਪਸ;

        GGML_ASSERT(!ਇਹ->pinned || !extended);

        ਇਹ->extended = extended;
        fastrpc_map_flags flags = ਇਹ->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        ਪੂਰਨ err = fastrpc_mmap(sess->domain_id, fd(), (ਖਾਲੀ *) ਅਧਾਰ(), 0, size(), flags);
        ਜੇ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
            ਸੁੱਟੋ std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (ਖਾਲੀ *) ਅਧਾਰ(), size(), fd(), pinned, extended);

        ਇਹ->mapped = ਸੱਚ;
    }

    ਖਾਲੀ unmap() {
        ਜੇ (!ਇਹ->mapped) ਵਾਪਸ;

        ਜੇ (!ਇਹ->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        ਜੇ (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (ਖਾਲੀ *) ਅਧਾਰ(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (ਖਾਲੀ *) ਅਧਾਰ(), size(), fd());

        ਇਹ->mapped = ਝੂਠ;
    }

    ਖਾਲੀ alloc(size_t size) {
        ਜੇ (ਇਹ->mem) ਵਾਪਸ;

        ਇਹ->mem = std::make_shared<ggml_hexagon_rpcmem_ਖੰਡ>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (ਖਾਲੀ *) ਅਧਾਰ(), ਇਹ->size(), fd(), (ਪੂਰਨ) pinned);
        ਜੇ (ਇਹ->pinned) {
            mmap();
        }
    }

    ਖਾਲੀ ਮੁਕਤ() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (ਖਾਲੀ *) ਅਧਾਰ(), size(), fd());
        ਇਹ->mem  = ਖਾਲੀ_ਸੂਚਕ;
    }

    ggml_hexagon_shared_ਬਫਰ(ggml_hexagon_session * sess, size_t size, ਤਾਰਕਿਕ pinned = ਝੂਠ) {
        ਇਹ->sess     = sess;
        ਇਹ->mapped   = ਝੂਠ;
        ਇਹ->pinned   = pinned;
        ਇਹ->extended = ਝੂਠ;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_ਵਿੱਥ = (size + 4095) & ~4095;
        size_t ਕੁੱਲ_ਮਾਪ   = guard_ਵਿੱਥ + 4096;
        ਜੇ (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            ਕੁੱਲ_ਮਾਪ = (ਕੁੱਲ_ਮਾਪ + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(ਕੁੱਲ_ਮਾਪ);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_ਬਫਰ(ggml_hexagon_session * sess, ਅਚਲ ggml_hexagon_shared_ਬਫਰ & other) {
        ਇਹ->sess   = sess;
        ਇਹ->mem    = other.mem;
        ਇਹ->mapped = ਝੂਠ;
        ਇਹ->pinned = other.pinned;
        ਇਹ->extended = other.extended;
    }

    ~ggml_hexagon_shared_ਬਫਰ() {
        ਮੁਕਤ();
        ਲਈ (ਸਵੈਚਾਲਿਤ * extra : tensor_extra) {
            ਮਿਟਾਓ extra;
        }
    }
};

ਬਣਤਰ ggml_hexagon_fence_ਬਫਰ : ਜਨਤਕ ggml_hexagon_shared_ਬਫਰ {
    uint32_t              ਥਾਂ_ਗਿਣਤੀ = 0;
    uint32_t              ਥਾਂ_ਸਿਰ  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_ਬਫਰ   backend_ਬਫਰ{};

    ggml_hexagon_fence_ਬਫਰ(ggml_hexagon_session * sess, ggml_backend_ਬਫਰ_type_t buft, size_t size)
        : ggml_hexagon_shared_ਬਫਰ(sess, size, ਝੂਠ /* pinned */),
          ਥਾਂ_ਗਿਣਤੀ(size / GGML_HEXAGON_FENCE_ਥਾਂ_ਮਾਪ),
          ਥਾਂ_ਸਿਰ(0) {
        backend_ਬਫਰ.buft    = buft;
        backend_ਬਫਰ.context = static_cast<ggml_hexagon_shared_ਬਫਰ *>(ਇਹ);
        backend_ਬਫਰ.size    = size;
        mmap(ਝੂਠ);
    }

    uint8_t * alloc_ਥਾਂ(uint32_t n_slots = 1) {
        uint8_t * ptr = ਖਾਲੀ_ਸੂਚਕ;
        ਜੇ (n_slots == 1 && !free_slots.ਖਾਲੀ_ਥਾਵਾਂ()) {
            uint32_t ਥਾਂ = free_slots.back();
            free_slots.pop_back();
            ptr = ਅਧਾਰ() + (size_t) ਥਾਂ * GGML_HEXAGON_FENCE_ਥਾਂ_ਮਾਪ;
        } ਨਹੀਂਤਾਂ ਜੇ (ਥਾਂ_ਸਿਰ + n_slots <= ਥਾਂ_ਗਿਣਤੀ) {
            uint32_t ਥਾਂ = ਥਾਂ_ਸਿਰ;
            ਥਾਂ_ਸਿਰ += n_slots;
            ptr = ਅਧਾਰ() + (size_t) ਥਾਂ * GGML_HEXAGON_FENCE_ਥਾਂ_ਮਾਪ;
        }
        ਜੇ (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_ਥਾਂ_ਮਾਪ);
        }
        ਵਾਪਸ ptr;
    }

    ਖਾਲੀ free_ਥਾਂ(ਖਾਲੀ * ptr, uint32_t n_slots = 1) {
        ਜੇ (!ptr) ਵਾਪਸ;
        uint32_t ਥਾਂ = ((uint8_t *) ptr - ਅਧਾਰ()) / GGML_HEXAGON_FENCE_ਥਾਂ_ਮਾਪ;
        ਲਈ (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(ਥਾਂ + i);
        }
    }
};

ਸਤਰਅੰਦਰ uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_ਥਾਂ(n_slots);
    GGML_ASSERT(ptr);
    ਵਾਪਸ ptr;
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_hexagon_session::free_fence(ਖਾਲੀ * ptr, uint32_t n_slots) {
    ਜੇ (fence_buf) {
        fence_buf->free_ਥਾਂ(ptr, n_slots);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਸਵੈਚਾਲਿਤ sbuf = static_cast<ggml_hexagon_shared_ਬਫਰ *>(ਬਫਰ->context);
    sbuf->sess->unclone_ਬਫਰ(sbuf);
    ਮਿਟਾਓ sbuf;
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_hexagon_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਸਵੈਚਾਲਿਤ sbuf = static_cast<ggml_hexagon_shared_ਬਫਰ *>(ਬਫਰ->context);
    ਵਾਪਸ sbuf->ਅਧਾਰ();
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_hexagon_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor) {
    ਸਵੈਚਾਲਿਤ sbuf = static_cast<ggml_hexagon_shared_ਬਫਰ *>(ਬਫਰ->context);
    ਸਵੈਚਾਲਿਤ sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (ਖਾਲੀ *) sbuf->ਅਧਾਰ(), tensor->ਅੰਕੜੇ, ggml_nbytes(tensor));

    ਸਵੈਚਾਲਿਤ extra = ਨਵਾਂ ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                   ggml_tensor *         tensor,
                                                   ਅਚਲ ਖਾਲੀ *          ਅੰਕੜੇ,
                                                   size_t                ਵਿੱਥ,
                                                   size_t                size) {
    ਸਵੈਚਾਲਿਤ extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    ਸਵੈਚਾਲਿਤ sbuf  = (ggml_hexagon_shared_ਬਫਰ *) ਬਫਰ->context;
    ਸਵੈਚਾਲਿਤ sess  = sbuf->sess;

    ਜੇ (ggml_backend_ਬਫਰ_get_usage(ਬਫਰ) == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_ਭਾਰ;
        ਜੇ (htp_is_tiled_ਭਾਰ_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, ਅੰਕੜੇ, ਵਿੱਥ, size, (ਪੂਰਨ) ਬਫਰ->usage, extra->flags);

    ਜੇ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, ਅੰਕੜੇ, size);
        ਵਾਪਸ;
    }

    ਜੇ (ਵਿੱਥ == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.ਖਾਲੀ_ਥਾਵਾਂ()) {
        htp_repack_tensor_tiled(tensor, ਅੰਕੜੇ, size);
        ਵਾਪਸ;
    }

    ਜੇ (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(extra->shadow_buf.ਅੰਕੜੇ() + ਵਿੱਥ, ਅੰਕੜੇ, size);
    extra->shadow_ਮਾਪ += size;

    ਜੇ (extra->shadow_ਮਾਪ >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.ਅੰਕੜੇ(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_ਮਾਪ = 0;
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                   ਅਚਲ ggml_tensor *   tensor,
                                                   ਖਾਲੀ *                ਅੰਕੜੇ,
                                                   size_t                ਵਿੱਥ,
                                                   size_t                size) {
    ਸਵੈਚਾਲਿਤ extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    ਸਵੈਚਾਲਿਤ sbuf  = (ggml_hexagon_shared_ਬਫਰ *) ਬਫਰ->context;
    ਸਵੈਚਾਲਿਤ sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, ਅੰਕੜੇ, ਵਿੱਥ, size, (ਪੂਰਨ) ਬਫਰ->usage, extra->flags);

    ਜੇ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, size);
        ਵਾਪਸ;
    }

    htp_unpack_tensor_tiled(ਅੰਕੜੇ, tensor, ਵਿੱਥ, size);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_hexagon_ਬਫਰ_cpy_tensor(ggml_backend_ਬਫਰ_t      ਬਫਰ,
                                                   ਅਚਲ ਬਣਤਰ ggml_tensor * src,
                                                   ਬਣਤਰ ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    ਵਾਪਸ ਝੂਠ;

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
    GGML_ਅਣਵਰਤਿਆ(src);
    GGML_ਅਣਵਰਤਿਆ(dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_ਬਫਰ_ਸੈੱਟ_tensor_2d(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                          ggml_tensor *         tensor,
                                                          ਅਚਲ ਖਾਲੀ *          ਅੰਕੜੇ,
                                                          size_t                ਵਿੱਥ,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_ਅੰਕੜੇ) {
    ਸਵੈਚਾਲਿਤ extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    ਸਵੈਚਾਲਿਤ sbuf  = (ggml_hexagon_shared_ਬਫਰ *) ਬਫਰ->context;
    ਸਵੈਚਾਲਿਤ sess  = sbuf->sess;

    ਜੇ (ggml_backend_ਬਫਰ_get_usage(ਬਫਰ) == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_ਭਾਰ;
        ਜੇ (htp_is_tiled_ਭਾਰ_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride_tensor, stride_ਅੰਕੜੇ, (ਪੂਰਨ) ਬਫਰ->usage, extra->flags);

    ਜੇ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        ਲਈ (size_t i = 0; i < n_copies; i++) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ((uint8_t *) tensor->ਅੰਕੜੇ + ਵਿੱਥ + i * stride_tensor, (ਅਚਲ uint8_t *) ਅੰਕੜੇ + i * stride_ਅੰਕੜੇ, size);
        }
        ਵਾਪਸ;
    }

    ਜੇ (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    ਲਈ (size_t i = 0; i < n_copies; i++) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(extra->shadow_buf.ਅੰਕੜੇ() + ਵਿੱਥ + i * stride_tensor, (ਅਚਲ uint8_t *) ਅੰਕੜੇ + i * stride_ਅੰਕੜੇ, size);
    }
    extra->shadow_ਮਾਪ += n_copies * size;

    ਜੇ (extra->shadow_ਮਾਪ >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.ਅੰਕੜੇ(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_ਮਾਪ = 0;
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_ਬਫਰ_get_tensor_2d(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                      ਅਚਲ ggml_tensor *   tensor,
                                                      ਖਾਲੀ *                ਅੰਕੜੇ,
                                                      size_t                ਵਿੱਥ,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_ਅੰਕੜੇ) {
    ਸਵੈਚਾਲਿਤ extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    ਸਵੈਚਾਲਿਤ sbuf  = (ggml_hexagon_shared_ਬਫਰ *) ਬਫਰ->context;
    ਸਵੈਚਾਲਿਤ sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride_tensor, stride_ਅੰਕੜੇ, (ਪੂਰਨ) ਬਫਰ->usage);

    ਜੇ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        ਲਈ (size_t i = 0; i < n_copies; i++) {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ((uint8_t *)ਅੰਕੜੇ + i * stride_ਅੰਕੜੇ, (ਅਚਲ uint8_t *)tensor->ਅੰਕੜੇ + ਵਿੱਥ + i * stride_tensor, size);
        }
        ਵਾਪਸ;
    }

    size_t ਆਰਜ਼ੀ_ਮਾਪ      = n_copies > 0 ? (n_copies - 1) * stride_tensor + size : 0;
    size_t ਟੁਕੜਾ_ਮਾਪ     = tensor->ne[1] * ggml_ਪੰਕਤੀ_ਮਾਪ(tensor->type, tensor->ne[0]);
    size_t ਟੁਕੜਾ_ਵਿੱਥ   = ਵਿੱਥ % ਟੁਕੜਾ_ਮਾਪ;
    size_t ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ = ggml_ਪੰਕਤੀ_ਮਾਪ(tensor->type, tensor->ne[0]);

    GGML_ASSERT((ਟੁਕੜਾ_ਵਿੱਥ % ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ) == 0 && "offset must be aligned to row boundary");
    GGML_ASSERT((ਆਰਜ਼ੀ_ਮਾਪ % ਪੰਕਤੀ_ਮਾਪ_ਬਾਈਟਾਂ)    == 0 && "temp_size must be a multiple of row size");
    GGML_ASSERT((ਵਿੱਥ + ਆਰਜ਼ੀ_ਮਾਪ) <= ggml_nbytes(tensor));

    std::vector<uint8_t> ਆਰਜ਼ੀ_buf(ਆਰਜ਼ੀ_ਮਾਪ);
    htp_unpack_tensor_tiled(ਆਰਜ਼ੀ_buf.ਅੰਕੜੇ(), tensor, ਵਿੱਥ, ਆਰਜ਼ੀ_ਮਾਪ);

    ਲਈ (size_t i = 0; i < n_copies; i++) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ((uint8_t *) ਅੰਕੜੇ + i * stride_ਅੰਕੜੇ, ਆਰਜ਼ੀ_buf.ਅੰਕੜੇ() + i * stride_tensor, size);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    ਸਵੈਚਾਲਿਤ sbuf = (ggml_hexagon_shared_ਬਫਰ *) ਬਫਰ->context;
    ਸਵੈਚਾਲਿਤ sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (ਖਾਲੀ *) sbuf->ਅਧਾਰ(), sbuf->size());
    memset(sbuf->ਅਧਾਰ(), value, sbuf->size());
}

ਸਥਿਰ ggml_backend_ਬਫਰ_i ggml_backend_hexagon_ਬਫਰ_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_hexagon_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ggml_backend_hexagon_ਬਫਰ_init_tensor,
    /* .memset_tensor   = */ ਨਿਰਾ,
    /* .set_tensor      = */ ggml_backend_hexagon_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_ਬਫਰ_ਸੈੱਟ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_ਬਫਰ_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_ਬਫਰ_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,
};

// ** backend buffer type

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_host_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                        ggml_tensor *         tensor,
                                                        ਅਚਲ ਖਾਲੀ *          ਅੰਕੜੇ,
                                                        size_t                ਵਿੱਥ,
                                                        size_t                size) {
    ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, ਅੰਕੜੇ, size);
    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_host_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                        ਅਚਲ ggml_tensor *   tensor,
                                                        ਖਾਲੀ *                ਅੰਕੜੇ,
                                                        size_t                ਵਿੱਥ,
                                                        size_t                size) {
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, size);
    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_i ggml_backend_hexagon_host_ਬਫਰ_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_hexagon_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ggml_backend_hexagon_ਬਫਰ_init_tensor,
    /* .memset_tensor   = */ ਨਿਰਾ,
    /* .set_tensor      = */ ggml_backend_hexagon_host_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ਨਿਰਾ,
    /* .get_tensor_2d   = */ ਨਿਰਾ,
    /* .cpy_tensor      = */ ggml_backend_hexagon_ਬਫਰ_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,
};

// ** backend buffer type

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_hexagon_ਬਫਰ_type_name(ggml_backend_ਬਫਰ_type_t ਬਫਰ_type) {
    ਵਾਪਸ static_cast<ggml_backend_hexagon_ਬਫਰ_type_context *>(ਬਫਰ_type->context)->name.c_str();
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_hexagon_ਬਫਰ_type_alloc_ਬਫਰ(
            ggml_backend_ਬਫਰ_type_t ਬਫਰ_type, size_t size) {
    ਸਵੈਚਾਲਿਤ dev_ctx = static_cast<ggml_backend_hexagon_ਬਫਰ_type_context *>(ਬਫਰ_type->context)->dev_ctx;
    ਸਵੈਚਾਲਿਤ sess    = dev_ctx->session();
    ਜੇ (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਕੋਸ਼ਿਸ਼ {
        ggml_hexagon_shared_ਬਫਰ * sbuf = ਨਵਾਂ ggml_hexagon_shared_ਬਫਰ(sess, size, ਝੂਠ);
        ਵਾਪਸ ggml_backend_ਬਫਰ_init(ਬਫਰ_type, ggml_backend_hexagon_ਬਫਰ_interface, sbuf, size);
    } ਫੜੋ (ਅਚਲ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_hexagon_host_ਬਫਰ_type_alloc_ਬਫਰ(
            ggml_backend_ਬਫਰ_type_t ਬਫਰ_type, size_t size) {
    ਸਵੈਚਾਲਿਤ dev_ctx = static_cast<ggml_backend_hexagon_ਬਫਰ_type_context *>(ਬਫਰ_type->context)->dev_ctx;
    ਸਵੈਚਾਲਿਤ sess    = dev_ctx->session();
    ਜੇ (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਕੋਸ਼ਿਸ਼ {
        ggml_hexagon_shared_ਬਫਰ * sbuf = ਨਵਾਂ ggml_hexagon_shared_ਬਫਰ(sess, size, ਝੂਠ);
        ਵਾਪਸ ggml_backend_ਬਫਰ_init(ਬਫਰ_type, ggml_backend_hexagon_host_ਬਫਰ_interface, sbuf, size);
    } ਫੜੋ (ਅਚਲ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
}

ਸਥਿਰ size_t ggml_backend_hexagon_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ 128;  // HVX alignment
    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ size_t ggml_backend_hexagon_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਜੇ (!ggml_backend_buft_is_host(buft) && htp_is_tiled_ਭਾਰ_tensor(t)) {
        int64_t ne0 = hex_ਗੇੜਾ_ਉੱਪਰ(t->ne[0], 32);
        int64_t ne1 = hex_ਗੇੜਾ_ਉੱਪਰ(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_ਮਾਪ = htp_tiled_ਪੰਕਤੀ_ਮਾਪ(t->type, ne0) * ne1 * ne2 * ne3;
        ਵਾਪਸ std::max(tiled_ਮਾਪ, ggml_nbytes(t));
    }
    ਵਾਪਸ ggml_nbytes(t);
}

ਸਥਿਰ size_t parse_ਮਾਪ(ਅਚਲ ਅੱਖਰ * str, size_t default_unit = 1024 * 1024) {
    ਜੇ (!str || str[0] == '\0') {
        ਵਾਪਸ 0;
    }
    ਅੱਖਰ * end = ਨਿਰਾ;
    ਦੁੱਗਣਾ val = strtod(str, &end);
    ਜੇ (val < 0) {
        ਵਾਪਸ 0;
    }
    ਜੇ (end && *end) {
        ਜਦੋਂ (*end == ' ') end++;
        ਜੇ (*end == 'k' || *end == 'K') {
            ਵਾਪਸ (size_t) (val * 1024);
        } ਨਹੀਂਤਾਂ ਜੇ (*end == 'm' || *end == 'M') {
            ਵਾਪਸ (size_t) (val * 1024 * 1024);
        } ਨਹੀਂਤਾਂ ਜੇ (*end == 'g' || *end == 'G') {
            ਵਾਪਸ (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    ਵਾਪਸ (size_t) (val * default_unit);
}

ਸਥਿਰ size_t ggml_backend_hexagon_ਬਫਰ_type_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ opt_mbuf_dyn;
    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_hexagon_ਬਫਰ_type_is_host(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ ਝੂਠ;
    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_hexagon_host_ਬਫਰ_type_is_host(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ ਸੱਚ;
    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਬਣਤਰ ggml_backend_hexagon_alloc_ਬਫਰ_n_plan_ਵਸਤੂ {
    size_t size;
    ਪੂਰਨ    ਪਹਿਲਾ;
    ਪੂਰਨ    ਆਖਰੀ;
};

ਵਰਤੋਂ ggml_backend_hexagon_alloc_ਬਫਰ_n_plan_t = std::vector<ggml_backend_hexagon_alloc_ਬਫਰ_n_plan_ਵਸਤੂ>;

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_hexagon_kv_layer_suffix(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਜੇ (ਲੜੀ_ਤੁਲਨਾ(t->name, "cache_", 6) != 0) {
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਅਚਲ ਅੱਖਰ * p = ਲੜੀ_ਖੋਜ(t->name, "_l");
    ਜੇ (!p || !ਅੰਕ_ਹੈ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)p[2])) {
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਵਾਪਸ p;
}

ਬਣਤਰ ggml_backend_hexagon_alloc_unit {
    size_t size;
    ਪੂਰਨ    ਪਹਿਲਾ;
    ਪੂਰਨ    ਆਖਰੀ;
};

ਸਥਿਰ ggml_backend_hexagon_alloc_ਬਫਰ_n_plan_t ggml_backend_hexagon_alloc_ਬਫਰ_n_plan(
        ggml_backend_ਬਫਰ_type_t buft, ਬਣਤਰ ggml_tensor ** tensors, ਪੂਰਨ n_tensors) {
    ggml_backend_hexagon_alloc_ਬਫਰ_n_plan_t plan;

    ਅਚਲ size_t alignment = ggml_backend_buft_get_alignment(buft);
    ਅਚਲ size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    ਪੂਰਨ i = 0;
    ਜਦੋਂ (i < n_tensors) {
        ਬਣਤਰ ggml_tensor * t = tensors[i];
        size_t unit_ਮਾਪ = 0;
        ਪੂਰਨ unit_ਪਹਿਲਾ = i;
        ਪੂਰਨ unit_ਆਖਰੀ = i + 1;

        ਜੇ (t->ਅੰਕੜੇ == ਨਿਰਾ && t->view_src == ਨਿਰਾ) {
            unit_ਮਾਪ += GGML_PAD(ggml_backend_buft_get_alloc_ਮਾਪ(buft, t), alignment);
        }

        ਅਚਲ ਅੱਖਰ * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        ਜਦੋਂ (unit_ਆਖਰੀ < n_tensors) {
            ਬਣਤਰ ggml_tensor * ਅਗਲਾ = tensors[unit_ਆਖਰੀ];

            ਜੇ (ਅਗਲਾ->view_src != ਨਿਰਾ) {
                unit_ਆਖਰੀ++;
                ਜਾਰੀ;
            }

            ਜੇ (layer_suffix != ਨਿਰਾ) {
                ਅਚਲ ਅੱਖਰ * ਅਗਲਾ_suffix = ggml_hexagon_kv_layer_suffix(ਅਗਲਾ);
                ਜੇ (ਅਗਲਾ_suffix != ਨਿਰਾ && strcmp(layer_suffix, ਅਗਲਾ_suffix) == 0) {
                    ਜੇ (ਅਗਲਾ->ਅੰਕੜੇ == ਨਿਰਾ) {
                        unit_ਮਾਪ += GGML_PAD(ggml_backend_buft_get_alloc_ਮਾਪ(buft, ਅਗਲਾ), alignment);
                    }
                    unit_ਆਖਰੀ++;
                    ਜਾਰੀ;
                }
            }

            ਰੋਕੋ;
        }

        units.push_back({ unit_ਮਾਪ, unit_ਪਹਿਲਾ, unit_ਆਖਰੀ });
        i = unit_ਆਖਰੀ;
    }

    size_t cur_buf_ਮਾਪ  = 0;
    ਪੂਰਨ    cur_buf_ਪਹਿਲਾ = 0;

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & unit : units) {
        ਜੇ (unit.size == 0) {
            ਜਾਰੀ;
        }

        ਜੇ (cur_buf_ਮਾਪ > 0 && (cur_buf_ਮਾਪ + unit.size) > max_size) {
            plan.push_back({ cur_buf_ਮਾਪ, cur_buf_ਪਹਿਲਾ, unit.ਪਹਿਲਾ });
            cur_buf_ਮਾਪ  = 0;
            cur_buf_ਪਹਿਲਾ = unit.ਪਹਿਲਾ;
        }

        cur_buf_ਮਾਪ += unit.size;
    }

    ਜੇ (cur_buf_ਮਾਪ > 0) {
        plan.push_back({ cur_buf_ਮਾਪ, cur_buf_ਪਹਿਲਾ, n_tensors });
    }

    ਵਾਪਸ plan;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_hexagon_ਬਫਰ_type_alloc_ਬਫਰ_n(
        ggml_backend_ਬਫਰ_type_t buft, ਬਣਤਰ ggml_tensor ** tensors, ਪੂਰਨ n_tensors) {
    ਅਚਲ ggml_backend_hexagon_alloc_ਬਫਰ_n_plan_t plan = ggml_backend_hexagon_alloc_ਬਫਰ_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_ਬਫਰ_t> buffers;
    buffers.reserve(plan.size());

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਵਸਤੂ : plan) {
        ggml_backend_ਬਫਰ_t ਬਫਰ = ggml_backend_buft_alloc_ਬਫਰ(buft, ਵਸਤੂ.size);
        ਜੇ (ਬਫਰ == ਨਿਰਾ) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), ਵਸਤੂ.size);
            ਲਈ (ggml_backend_ਬਫਰ_t b : buffers) {
                ggml_backend_ਬਫਰ_free(b);
            }
            ਵਾਪਸ ਨਿਰਾ;
        }

        ਬਣਤਰ ggml_tallocr tallocr = ggml_tallocr_new(ਬਫਰ);

        ਬਣਤਰ ggml_tensor * t_failed = ਨਿਰਾ;
        ਲਈ (ਪੂਰਨ j = ਵਸਤੂ.ਪਹਿਲਾ; j < ਵਸਤੂ.ਆਖਰੀ; j++) {
            ਬਣਤਰ ggml_tensor * t = tensors[j];
            ਜੇ (t->ਅੰਕੜੇ == ਨਿਰਾ) {
                ਜੇ (t->view_src == ਨਿਰਾ) {
                    ਜੇ (ggml_tallocr_alloc(&tallocr, t) != GGML_ਸਥਿਤੀ_SUCCESS) {
                        t_failed = t;
                        ਰੋਕੋ;
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (t->ਬਫਰ == ਨਿਰਾ) {
                    ਜੇ (ggml_backend_view_init(t) != GGML_ਸਥਿਤੀ_SUCCESS) {
                        t_failed = t;
                        ਰੋਕੋ;
                    }
                }
            } ਨਹੀਂਤਾਂ {
                ਜੇ (t->view_src != ਨਿਰਾ && t->ਬਫਰ == ਨਿਰਾ) {
                    ਜੇ (ggml_backend_view_init(t) != GGML_ਸਥਿਤੀ_SUCCESS) {
                        t_failed = t;
                        ਰੋਕੋ;
                    }
                }
            }
        }
        ਜੇ (t_failed != ਨਿਰਾ) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            ਲਈ (ggml_backend_ਬਫਰ_t b : buffers) {
                ggml_backend_ਬਫਰ_free(b);
            }
            ggml_backend_ਬਫਰ_free(ਬਫਰ);
            ਵਾਪਸ ਨਿਰਾ;
        }

        buffers.push_back(ਬਫਰ);
    }

    ਜੇ (buffers.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਜੇ (buffers.size() == 1) {
        ਵਾਪਸ buffers[0];
    }

    ਵਾਪਸ ggml_backend_multi_ਬਫਰ_alloc_ਬਫਰ(buffers.ਅੰਕੜੇ(), buffers.size());
}

ਸਥਿਰ size_t ggml_backend_hexagon_ਬਫਰ_type_get_alloc_ਮਾਪ_n(
        ggml_backend_ਬਫਰ_type_t buft, ਬਣਤਰ ggml_tensor ** tensors, ਪੂਰਨ n_tensors) {
    ਅਚਲ ggml_backend_hexagon_alloc_ਬਫਰ_n_plan_t plan = ggml_backend_hexagon_alloc_ਬਫਰ_n_plan(buft, tensors, n_tensors);

    size_t ਕੁੱਲ = 0;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਵਸਤੂ : plan) {
        ਕੁੱਲ += ਵਸਤੂ.size;
    }
    ਵਾਪਸ ਕੁੱਲ;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_i ggml_backend_hexagon_ਬਫਰ_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_ਬਫਰ_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_ਬਫਰ_type_alloc_ਬਫਰ,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_ਬਫਰ_type_alloc_ਬਫਰ_n,
    /* .get_alignment       = */ ggml_backend_hexagon_ਬਫਰ_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_ਬਫਰ_type_get_max_ਮਾਪ,
    /* .get_alloc_size      = */ ggml_backend_hexagon_ਬਫਰ_type_get_alloc_ਮਾਪ,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_ਬਫਰ_type_get_alloc_ਮਾਪ_n,
    /* .is_host             = */ ggml_backend_hexagon_ਬਫਰ_type_is_host,
};

ਸਥਿਰ ggml_backend_ਬਫਰ_type_i ggml_backend_hexagon_host_ਬਫਰ_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_ਬਫਰ_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_ਬਫਰ_type_alloc_ਬਫਰ,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_ਬਫਰ_type_alloc_ਬਫਰ_n,
    /* .get_alignment       = */ ggml_backend_hexagon_ਬਫਰ_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_ਬਫਰ_type_get_max_ਮਾਪ,
    /* .get_alloc_size      = */ ggml_backend_hexagon_ਬਫਰ_type_get_alloc_ਮਾਪ,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_ਬਫਰ_type_get_alloc_ਮਾਪ_n,
    /* .is_host             = */ ggml_backend_hexagon_host_ਬਫਰ_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(ਪੂਰਨ dev_id, ਅਚਲ ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    ਬਫਰ_type.device  = dev;
    ਬਫਰ_type.iface   = ggml_backend_hexagon_ਬਫਰ_type_interface;
    ਬਫਰ_type.context = ਨਵਾਂ ggml_backend_hexagon_ਬਫਰ_type_context(config.name, ਇਹ);

    host_ਬਫਰ_type.device  = dev;
    host_ਬਫਰ_type.iface   = ggml_backend_hexagon_host_ਬਫਰ_type_interface;
    host_ਬਫਰ_type.context = ਨਵਾਂ ggml_backend_hexagon_ਬਫਰ_type_context(config.name + "-HOST", ਇਹ);

    fence_ਬਫਰ_type.device  = dev;
    fence_ਬਫਰ_type.iface   = ggml_backend_hexagon_ਬਫਰ_type_interface;
    fence_ਬਫਰ_type.context = ਨਵਾਂ ggml_backend_hexagon_ਬਫਰ_type_context(config.name + "-FENCE", ਇਹ);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    ਮਿਟਾਓ static_cast<ggml_backend_hexagon_ਬਫਰ_type_context *>(ਬਫਰ_type.context);
    ਮਿਟਾਓ static_cast<ggml_backend_hexagon_ਬਫਰ_type_context *>(host_ਬਫਰ_type.context);
    ਮਿਟਾਓ static_cast<ggml_backend_hexagon_ਬਫਰ_type_context *>(fence_ਬਫਰ_type.context);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_ਬਫਰ_is_hexagon(ਅਚਲ ਬਣਤਰ ggml_backend_ਬਫਰ * b) {
    ਵਾਪਸ b->buft->iface.get_alignment == ggml_backend_hexagon_ਬਫਰ_type_get_alignment;
}

ਬਣਤਰ ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<ਪੂਰਨ, ਪੂਰਨ>                b_map; // buffer fd   to index
    std::unordered_map<ਅਚਲ ggml_tensor*, ਪੂਰਨ> t_map; // tensor ptr  to index
    std::unordered_multimap<ਖਾਲੀ*, ਪੂਰਨ>         d_map; // tensor data to index

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n_bufs;     // num buffers in the batch
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n_tens;     // num tensors ...
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n_bufs_max;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n_tens_max;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n_ops_max;
    size_t       b_vmem_max;

    ਖਾਲੀ reset() {
        n_bufs = 0;
        n_tens = 0;
        n_ops  = 0;
        b_vmem = 0;

        b_map.clear();
        t_map.clear();
        d_map.clear();
        ops.resize(n_ops_max);
    }

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_ਮਾਪ, size_t max_vmem) {
        ਇਹ->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_ਮਾਪ;
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

    ਤਾਰਕਿਕ ਖਾਲੀ_ਥਾਵਾਂ() ਅਚਲ { ਵਾਪਸ n_ops == 0; }

    // add buffer and return its index
    ਪੂਰਨ ਜੋੜੋ_ਬਫਰ(ggml_hexagon_shared_ਬਫਰ * sbuf) {
        // Lookup by fd
        ਸਵੈਚਾਲਿਤ it = b_map.find(sbuf->fd());
        ਜੇ (it != b_map.end()) { ਵਾਪਸ it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        ਪੂਰਨ bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.ਅਧਾਰ  = (uint64_t) sbuf->ਅਧਾਰ();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        ਜੇ (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (ਖਾਲੀ*) sbuf->ਅਧਾਰ(), (size_t) b.size, b_vmem);

        ਵਾਪਸ bi;
    }

    ਤਾਰਕਿਕ same_shape(ਅਚਲ htp_tensor * h, ਅਚਲ ggml_tensor * t) ਅਚਲ {
        ਸਵੈਚਾਲਿਤ extra = (ggml_hexagon_tensor_extra *) t->extra;

        ਅਚਲ ਤਾਰਕਿਕ is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ਅਚਲ int64_t ne0_padded = hex_ਗੇੜਾ_ਉੱਪਰ(t->ne[0], 32);
        ਅਚਲ int64_t ne1_padded = hex_ਗੇੜਾ_ਉੱਪਰ(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_ਪੰਕਤੀ_ਮਾਪ(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? ਆਕਾਰ(ggml_fp16_t) : t->nb[0];
        ਗਿਣਤੀ ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        ਵਾਪਸ (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    ਪੂਰਨ ਜੋੜੋ_tensor(ਅਚਲ ggml_tensor * t) {
        ਸਵੈਚਾਲਿਤ extra = (ggml_hexagon_tensor_extra *) t->extra;
        ਸਵੈਚਾਲਿਤ sbuf  = static_cast<ggml_hexagon_shared_ਬਫਰ *>(t->ਬਫਰ->context);

        // First lookup by tensor data
        ਸਵੈਚਾਲਿਤ range = d_map.equal_range(t->ਅੰਕੜੇ);
        ਲਈ (ਸਵੈਚਾਲਿਤ it = range.ਪਹਿਲਾ; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            ਜੇ (same_shape(h, t)) { ਵਾਪਸ it->second; }
        }

        // Lookup by tensor ptr
        ਸਵੈਚਾਲਿਤ it = t_map.find(t);
        ਜੇ (it != t_map.end()) { ਵਾਪਸ it->second; }

        // Add new tensor to the batch
        ਪੂਰਨ ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->ਅੰਕੜੇ, ti});

        uint64_t t_ਵਿੱਥ = (uint8_t *) t->ਅੰਕੜੇ - sbuf->ਅਧਾਰ();
        size_t   t_ਮਾਪ   = ggml_nbytes(t);

        htp_tensor &h = h_tens[ti];
        h.bi    = ਜੋੜੋ_ਬਫਰ(sbuf);
        h.ti    = ti;
        h.ਅੰਕੜੇ  = t_ਵਿੱਥ;
        h.type  = t->type;
        h.ne[0] = t->ne[0];
        h.ne[1] = t->ne[1];
        h.ne[2] = t->ne[2];
        h.ne[3] = t->ne[3];

        ਅਚਲ ਤਾਰਕਿਕ is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ਜੇ (is_tiled) {
            ਜੇ (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = ਆਕਾਰ(ggml_fp16_t);
            } ਨਹੀਂਤਾਂ {
                h.nb[0] = t->nb[0];
            }
            ਅਚਲ int64_t ne0_padded = hex_ਗੇੜਾ_ਉੱਪਰ(t->ne[0], 32);
            ਅਚਲ int64_t ne1_padded = hex_ਗੇੜਾ_ਉੱਪਰ(t->ne[1], 32);
            h.nb[1] = htp_tiled_ਪੰਕਤੀ_ਮਾਪ(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_ਮਾਪ  = h.size;
        } ਨਹੀਂਤਾਂ {
            h.size  = t_ਮਾਪ;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        ਜੇ ((extra->flags & GGML_HEXAGON_TENSOR_ਭਾਰ) != 0) {
            h.flags |= HTP_TENSOR_ਭਾਰ;
        }
        ਜੇ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        ਜੇ ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (ਖਾਲੀ*) t->ਅੰਕੜੇ, (size_t) t_ਵਿੱਥ, t_ਮਾਪ, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        ਵਾਪਸ ti;
    }

    ਤਾਰਕਿਕ fit_op(ਅਚਲ htp_opnode & ਗੰਢ) ਅਚਲ {
        ਜੇ (n_ops >= n_ops_max) ਵਾਪਸ ਝੂਠ;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        ਪੂਰਨ ਵੇਖਿਆ_bufs[HTP_OP_MAX_BUFS];
        ਪੂਰਨ n_ਵੇਖਿਆ_bufs = 0;

        ਸਵੈਚਾਲਿਤ fit_tensor = [&](ਅਚਲ ggml_tensor *t) {
            ਜੇ (!t) ਵਾਪਸ;
            ਜੇ (!t_map.count(t)) {
                extra_tens++;

                ਸਵੈਚਾਲਿਤ sbuf = static_cast<ggml_hexagon_shared_ਬਫਰ *>(t->ਬਫਰ->context);
                ਪੂਰਨ fd = sbuf->fd();
                ਜੇ (!b_map.count(fd)) {
                    ਲਈ (ਪੂਰਨ i = 0; i < n_ਵੇਖਿਆ_bufs; i++) {
                        ਜੇ (ਵੇਖਿਆ_bufs[i] == fd) ਵਾਪਸ;
                    }
                    ਜੇ (n_ਵੇਖਿਆ_bufs < HTP_OP_MAX_BUFS) {
                        ਵੇਖਿਆ_bufs[n_ਵੇਖਿਆ_bufs++] = fd;
                    }
                    ਜੇ (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * src : ਗੰਢ.get_inputs()) {
            fit_tensor(src);
        }
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * output : ਗੰਢ.get_outputs()) {
            fit_tensor(output);
        }

        ਜੇ ((extra_bufs + n_bufs) > n_bufs_max) ਵਾਪਸ ਝੂਠ;
        ਜੇ ((extra_tens + n_tens) > n_tens_max) ਵਾਪਸ ਝੂਠ;
        ਜੇ ((extra_vmem + b_vmem) > b_vmem_max) ਵਾਪਸ ਝੂਠ;

        ਵਾਪਸ ਸੱਚ;
    }

    // assumes that fit_op() was called first and returned true
    ਖਾਲੀ ਜੋੜੋ_op(ਅਚਲ htp_opnode & ਗੰਢ) {
        // Add new op

        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = ਗੰਢ;

        htp_op_desc &o = h_ops[n];
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(o.params,        ਗੰਢ.ਗੰਢ->op_params, ਆਕਾਰ(ਗੰਢ.ਗੰਢ->op_params));
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(o.kernel_params, ਗੰਢ.kernel_params,   ਆਕਾਰ(o.kernel_params));
        o.opcode = ਗੰਢ.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        ਸਵੈਚਾਲਿਤ inputs = ਗੰਢ.get_inputs();
        ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? ਜੋੜੋ_tensor(inputs[i]) : 0xffff;
        }

        ਸਵੈਚਾਲਿਤ outputs = ਗੰਢ.get_outputs();
        ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? ਜੋੜੋ_tensor(outputs[i]) : 0xffff;
        }
    }

    ਖਾਲੀ ਛਾਂਟੋ_buffers() {
        ਜੇ (n_bufs <= 1) ਵਾਪਸ;

        std::vector<ਪੂਰਨ> ਕ੍ਰਮ(n_bufs);
        ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0; i < n_bufs; i++) { ਕ੍ਰਮ[i] = (ਪੂਰਨ) i; }

        std::stable_sort(ਕ੍ਰਮ.begin(), ਕ੍ਰਮ.end(), [&](ਪੂਰਨ a, ਪੂਰਨ b) {
            ਵਾਪਸ h_bufs[a].size > h_bufs[b].size;
        });

        ਤਾਰਕਿਕ already_sorted = ਸੱਚ;
        ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0; i < n_bufs; i++) {
            ਜੇ (ਕ੍ਰਮ[i] != (ਪੂਰਨ) i) {
                already_sorted = ਝੂਠ;
                ਰੋਕੋ;
            }
        }
        ਜੇ (already_sorted) ਵਾਪਸ;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ new_bi = 0; new_bi < n_bufs; new_bi++) {
            ਪੂਰਨ old_bi = ਕ੍ਰਮ[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    ਖਾਲੀ update_mdev_ਸਮੂਹ(uint32_t mdev_idx) {
        ਜੇ (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_ਸਮੂਹ) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    ਤਾਰਕਿਕ try_fuse_common(std::initializer_list<ਅਚਲ ggml_tensor *> tensors) ਅਚਲ {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        ਪੂਰਨ ਵੇਖਿਆ_bufs[HTP_OP_MAX_BUFS];
        ਪੂਰਨ n_ਵੇਖਿਆ_bufs = 0;

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * t : tensors) {
            ਜੇ (!t || t_map.count(t)) {
                ਜਾਰੀ;
            }
            extra_tens++;
            ਸਵੈਚਾਲਿਤ sbuf = static_cast<ggml_hexagon_shared_ਬਫਰ *>(t->ਬਫਰ->context);
            ਪੂਰਨ fd = sbuf->fd();
            ਜੇ (!b_map.count(fd)) {
                ਤਾਰਕਿਕ ਮਿਲਿਆ = ਝੂਠ;
                ਲਈ (ਪੂਰਨ i = 0; i < n_ਵੇਖਿਆ_bufs; i++) {
                    ਜੇ (ਵੇਖਿਆ_bufs[i] == fd) {
                        ਮਿਲਿਆ = ਸੱਚ;
                        ਰੋਕੋ;
                    }
                }
                ਜੇ (!ਮਿਲਿਆ) {
                    ਜੇ (n_ਵੇਖਿਆ_bufs < HTP_OP_MAX_BUFS) {
                        ਵੇਖਿਆ_bufs[n_ਵੇਖਿਆ_bufs++] = fd;
                    }
                    ਜੇ (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        ਜੇ ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ try_fuse_common(ਅਚਲ ggml_tensor * t1, ਅਚਲ ggml_tensor * t2) ਅਚਲ {
        ਵਾਪਸ try_fuse_common({t1, t2});
    }

    ਤਾਰਕਿਕ try_fuse_allreduce_ਜੋੜੋ(ਅਚਲ htp_opnode & ਗੰਢ) {
        ਜੇ (n_ops == 0 || opt_ar_select != 2) ਵਾਪਸ ਝੂਠ;
        ਜੇ (ਗੰਢ.opcode != HTP_OP_ਜੋੜੋ) ਵਾਪਸ ਝੂਠ;

        htp_opnode & ਆਖਰੀ_ਗੰਢ = ops[n_ops - 1];
        ਜੇ (ਆਖਰੀ_ਗੰਢ.opcode != HTP_OP_ALLREDUCE) ਵਾਪਸ ਝੂਠ;

        ਸਵੈਚਾਲਿਤ * ar_kparams = (ਬਣਤਰ htp_allreduce_kernel_params *) ਆਖਰੀ_ਗੰਢ.kernel_params;
        ਅਚਲ uint32_t rank    = (uint32_t) ar_kparams->rank;
        ਅਚਲ uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        ਅਚਲ ggml_tensor * ar_local = ਆਖਰੀ_ਗੰਢ.inputs[rank];
        ਅਚਲ ggml_tensor * ਜੋੜੋ_src0 = ਗੰਢ.src0();
        ਅਚਲ ggml_tensor * ਜੋੜੋ_src1 = ਗੰਢ.src1();
        ਅਚਲ ggml_tensor * ਜੋੜੋ_dst  = ਗੰਢ.dst();

        ਜੇ (!ggml_hexagon_tensor_is_fuseable(ar_local)) ਵਾਪਸ ਝੂਠ;

        ਅਚਲ ggml_tensor * res_tensor;
        ਜੇ (ਜੋੜੋ_src0 == ar_local || ਜੋੜੋ_src0->ਅੰਕੜੇ == ar_local->ਅੰਕੜੇ) {
            res_tensor = ਜੋੜੋ_src1;
        } ਨਹੀਂਤਾਂ ਜੇ (ਜੋੜੋ_src1 == ar_local || ਜੋੜੋ_src1->ਅੰਕੜੇ == ar_local->ਅੰਕੜੇ) {
            res_tensor = ਜੋੜੋ_src0;
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (ar_local->type != res_tensor->type) ਵਾਪਸ ਝੂਠ;

        ਅਚਲ ਤਾਰਕਿਕ is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        ਅਚਲ ਤਾਰਕਿਕ is_ਪੰਕਤੀ_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        ਜੇ (!is_same_shape && !is_ਪੰਕਤੀ_bcast) ਵਾਪਸ ਝੂਠ;

        ਜੇ (is_same_shape) {
            ਜੇ (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
        ਜੇ (ggml_is_contiguous(ar_local) != ggml_is_contiguous(ਜੋੜੋ_dst)) {
            ਵਾਪਸ ਝੂਠ;
        }

        // scatter is only valid for in-place add within max outputs
        ਅਚਲ ਤਾਰਕਿਕ is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && ਜੋੜੋ_dst->ਅੰਕੜੇ == ar_local->ਅੰਕੜੇ;

        ਬਣਤਰ htp_allreduce_kernel_params new_kparams;
        ਜੇ (!ggml_hexagon_precompute_allreduce_params(
            sess, ਜੋੜੋ_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, ਸੱਚ, is_ਪੰਕਤੀ_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਤਾਰਕਿਕ scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_ਭਰਿਆ;

        ਲਈ (uint32_t r = 0; r < n_ranks; r++) {
            ਅਚਲ ggml_tensor * ar_src = ਆਖਰੀ_ਗੰਢ.inputs[r];
            ਜੇ (!ggml_hexagon_tensors_overlap(ਜੋੜੋ_dst, ar_src)) {
                ਜਾਰੀ;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            ਜੇ (scatter_ok && r == rank) {
                ਜਾਰੀ;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (!try_fuse_common(res_tensor, ਜੋੜੋ_dst)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਆਖਰੀ_ਗੰਢ.opcode = HTP_OP_ALLREDUCE_ਜੋੜੋ;
        ਆਖਰੀ_ਗੰਢ.name   = "ALLREDUCE+ADD";
        ਆਖਰੀ_ਗੰਢ.inputs.push_back(res_tensor);
        ਆਖਰੀ_ਗੰਢ.outputs.clear();
        ਆਖਰੀ_ਗੰਢ.outputs.push_back(ਜੋੜੋ_dst);
        ਆਖਰੀ_ਗੰਢ.fused.push_back(ਗੰਢ.ਗੰਢ);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਆਖਰੀ_ਗੰਢ.kernel_params, &new_kparams, ਆਕਾਰ(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ਜੋੜੋ;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(o.kernel_params, &new_kparams, ਆਕਾਰ(new_kparams));

        o.src[2 * n_ranks] = ਜੋੜੋ_tensor(res_tensor);
        ਜੇ (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            ਲਈ (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            ਲਈ (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } ਨਹੀਂਤਾਂ {
            o.dst[0] = ਜੋੜੋ_tensor(ਜੋੜੋ_dst);
            ਲਈ (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (ਪੂਰਨ) new_kparams.mode, (ਪੂਰਨ) new_kparams.n_dsts);
        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ try_fuse_rms_norm_mul(ਅਚਲ htp_opnode & ਗੰਢ) {
        ਜੇ (n_ops == 0) ਵਾਪਸ ਝੂਠ;
        ਜੇ (ਗੰਢ.opcode != HTP_OP_MUL) ਵਾਪਸ ਝੂਠ;

        htp_opnode & ਆਖਰੀ_ਗੰਢ = ops[n_ops - 1];
        ਜੇ (ਆਖਰੀ_ਗੰਢ.opcode != HTP_OP_RMS_NORM) ਵਾਪਸ ਝੂਠ;

        ਅਚਲ ggml_tensor * mul_src0 = ਗੰਢ.src0();
        ਅਚਲ ggml_tensor * mul_src1 = ਗੰਢ.src1();
        ਅਚਲ ggml_tensor * rms_ਬਾਹਰਲਾ  = ਆਖਰੀ_ਗੰਢ.dst();

        ਜੇ (!ggml_hexagon_tensor_is_fuseable(rms_ਬਾਹਰਲਾ)) ਵਾਪਸ ਝੂਠ;

        ਅਚਲ ggml_tensor * ਭਾਰ;
        ਜੇ (mul_src0 == rms_ਬਾਹਰਲਾ || mul_src0->ਅੰਕੜੇ == rms_ਬਾਹਰਲਾ->ਅੰਕੜੇ) {
            ਭਾਰ = mul_src1;
        } ਨਹੀਂਤਾਂ ਜੇ (mul_src1 == rms_ਬਾਹਰਲਾ || mul_src1->ਅੰਕੜੇ == rms_ਬਾਹਰਲਾ->ਅੰਕੜੇ) {
            ਭਾਰ = mul_src0;
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ggml_tensor * src0 = ਆਖਰੀ_ਗੰਢ.src0();

        ਜੇ (src0->ne[0] != ਭਾਰ->ne[0] || src0->ne[0] != ਗੰਢ.dst()->ne[0]) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਤਾਰਕਿਕ is_ਪੰਕਤੀ_bcast = (ਭਾਰ->ne[1] == 1 && ਭਾਰ->ne[2] == 1 && ਭਾਰ->ne[3] == 1);
        ਅਚਲ ਤਾਰਕਿਕ is_same_shape = (src0->ne[0] == ਭਾਰ->ne[0] && src0->ne[1] == ਭਾਰ->ne[1] &&
                                    src0->ne[2] == ਭਾਰ->ne[2] && src0->ne[3] == ਭਾਰ->ne[3]);
        ਜੇ (!is_ਪੰਕਤੀ_bcast && !is_same_shape) ਵਾਪਸ ਝੂਠ;

        ਜੇ (!ggml_are_same_shape(src0, ਗੰਢ.dst())) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (ggml_is_contiguous(src0) != ggml_is_contiguous(ਗੰਢ.dst())) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਬਣਤਰ htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, ਭਾਰ, ਗੰਢ.dst(), &new_kparams
        );

        ਜੇ ((size_t) new_kparams.vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_ਮਾਪ, sess->vtcm_ਮਾਪ);
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (!try_fuse_common(ਭਾਰ, ਗੰਢ.dst())) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਆਖਰੀ_ਗੰਢ.opcode = HTP_OP_RMS_NORM_MUL;
        ਆਖਰੀ_ਗੰਢ.name   = "RMS_NORM+MUL";
        ਆਖਰੀ_ਗੰਢ.inputs.clear();
        ਆਖਰੀ_ਗੰਢ.inputs.push_back(src0);
        ਆਖਰੀ_ਗੰਢ.inputs.push_back(ਭਾਰ);
        ਆਖਰੀ_ਗੰਢ.outputs.clear();
        ਆਖਰੀ_ਗੰਢ.outputs.push_back(ਗੰਢ.dst());
        ਆਖਰੀ_ਗੰਢ.fused.push_back(ਗੰਢ.ਗੰਢ);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਆਖਰੀ_ਗੰਢ.kernel_params, &new_kparams, ਆਕਾਰ(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(o.kernel_params, &new_kparams, ਆਕਾਰ(new_kparams));

        o.src[0] = ਜੋੜੋ_tensor(src0);
        o.src[1] = ਜੋੜੋ_tensor(ਭਾਰ);
        ਲਈ (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = ਜੋੜੋ_tensor(ਗੰਢ.dst());
        ਲਈ (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ try_fuse_mul_mat_ਜੋੜੋ(ਅਚਲ htp_opnode & ਗੰਢ) {
        ਜੇ (n_ops == 0) ਵਾਪਸ ਝੂਠ;
        ਜੇ (ਗੰਢ.opcode != HTP_OP_ਜੋੜੋ) ਵਾਪਸ ਝੂਠ;

        htp_opnode & ਆਖਰੀ_ਗੰਢ = ops[n_ops - 1];
        ਜੇ (ਆਖਰੀ_ਗੰਢ.opcode != HTP_OP_MUL_MAT) ਵਾਪਸ ਝੂਠ;

        ਅਚਲ ggml_tensor * ਜੋੜੋ_src0 = ਗੰਢ.src0();
        ਅਚਲ ggml_tensor * ਜੋੜੋ_src1 = ਗੰਢ.src1();
        ਅਚਲ ggml_tensor * mm_ਬਾਹਰਲਾ   = ਆਖਰੀ_ਗੰਢ.dst();

        ਜੇ (!ggml_hexagon_tensor_is_fuseable(mm_ਬਾਹਰਲਾ)) ਵਾਪਸ ਝੂਠ;

        ਅਚਲ ggml_tensor * src2;
        ਜੇ (ਜੋੜੋ_src0 == mm_ਬਾਹਰਲਾ || ਜੋੜੋ_src0->ਅੰਕੜੇ == mm_ਬਾਹਰਲਾ->ਅੰਕੜੇ) {
            src2 = ਜੋੜੋ_src1;
        } ਨਹੀਂਤਾਂ ਜੇ (ਜੋੜੋ_src1 == mm_ਬਾਹਰਲਾ || ਜੋੜੋ_src1->ਅੰਕੜੇ == mm_ਬਾਹਰਲਾ->ਅੰਕੜੇ) {
            src2 = ਜੋੜੋ_src0;
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ggml_tensor * src0 = ਆਖਰੀ_ਗੰਢ.src0();
        ਅਚਲ ggml_tensor * src1 = ਆਖਰੀ_ਗੰਢ.src1();

        ਜੇ (src2->type != GGML_TYPE_F32) ਵਾਪਸ ਝੂਠ;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        ਜੇ (!ggml_are_same_shape(mm_ਬਾਹਰਲਾ, ਗੰਢ.dst())) ਵਾਪਸ ਝੂਠ;

        ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * orig_kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) ਆਖਰੀ_ਗੰਢ.kernel_params;
        ਬਣਤਰ htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_ਜੋੜੋ_params(sess, src0, src1, src2, ਗੰਢ.dst(), &kparams);
        ਜੇ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (orig_kparams->collapse != kparams.collapse) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਪੂਰਨ src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        ਅਚਲ ਤਾਰਕਿਕ can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        ਜੇ (!can_fuse) ਵਾਪਸ ਝੂਠ;

        ਜੇ ((size_t) kparams.vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_ਮਾਪ, sess->vtcm_ਮਾਪ);
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            ਜੇ (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_ਧਾਗੇ < orig_kparams->n_act_ਧਾਗੇ) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_ਧਾਗੇ, kparams.n_act_ਧਾਗੇ);
                ਵਾਪਸ ਝੂਠ;
            }
        }

        ਜੇ (!try_fuse_common(src2, ਗੰਢ.dst())) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਆਖਰੀ_ਗੰਢ.opcode = HTP_OP_MUL_MAT_ਜੋੜੋ;
        ਆਖਰੀ_ਗੰਢ.name   = "MUL_MAT+ADD";
        ਆਖਰੀ_ਗੰਢ.inputs.clear();
        ਆਖਰੀ_ਗੰਢ.inputs.push_back(src0);
        ਆਖਰੀ_ਗੰਢ.inputs.push_back(src1);
        ਆਖਰੀ_ਗੰਢ.inputs.push_back(src2);
        ਆਖਰੀ_ਗੰਢ.outputs.clear();
        ਆਖਰੀ_ਗੰਢ.outputs.push_back(ਗੰਢ.dst());
        ਆਖਰੀ_ਗੰਢ.fused.push_back(ਗੰਢ.ਗੰਢ);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਆਖਰੀ_ਗੰਢ.kernel_params, &kparams, ਆਕਾਰ(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ਜੋੜੋ;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(o.kernel_params, &kparams, ਆਕਾਰ(kparams));

        o.src[0] = ਜੋੜੋ_tensor(src0);
        o.src[1] = ਜੋੜੋ_tensor(src1);
        o.src[2] = ਜੋੜੋ_tensor(src2);
        ਲਈ (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = ਜੋੜੋ_tensor(ਗੰਢ.dst());
        ਲਈ (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ try_fuse_mul_mat_nx(ਅਚਲ htp_opnode & ਗੰਢ) {
        ਜੇ (n_ops == 0 || ਗੰਢ.opcode != HTP_OP_MUL_MAT) ਵਾਪਸ ਝੂਠ;
        ਜੇ (!is_mergeable_mul_mat(ਗੰਢ.ਗੰਢ)) ਵਾਪਸ ਝੂਠ;

        ਅਚਲ ggml_tensor * w_ਵਿੱਚ = ਗੰਢ.src0();
        ਅਚਲ ggml_tensor * x_ਵਿੱਚ = ਗੰਢ.src1();
        ਅਚਲ ggml_tensor * d_ਵਿੱਚ = ਗੰਢ.dst();

        htp_opnode & ਆਖਰੀ_ਗੰਢ = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        ਜੇ (ਆਖਰੀ_ਗੰਢ.opcode == HTP_OP_MUL_MAT_NX) {
            ਅਚਲ uint32_t curr_n = (uint32_t) ਆਖਰੀ_ਗੰਢ.outputs.size();
            ਜੇ (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਅਚਲ ggml_tensor * w0 = ਆਖਰੀ_ਗੰਢ.inputs[0];
            ਅਚਲ ggml_tensor * x  = ਆਖਰੀ_ਗੰਢ.inputs[curr_n];

            ਜੇ (x_ਵਿੱਚ != x || w_ਵਿੱਚ->type != w0->type || w_ਵਿੱਚ->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_ਵਿੱਚ) != ggml_hexagon_tensor_is_tiled(w0)) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (!ਆਖਰੀ_ਗੰਢ.fused.ਖਾਲੀ_ਥਾਵਾਂ() && (mm_is_hmx_eligible(ਆਖਰੀ_ਗੰਢ.fused[0]) != mm_is_hmx_eligible(ਗੰਢ.ਗੰਢ))) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * orig_kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) ਆਖਰੀ_ਗੰਢ.kernel_params;
            ਅਚਲ ਤਾਰਕਿਕ collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_ਵਿੱਚ, x, d_ਵਿੱਚ);
            ਜੇ (orig_kparams->collapse && !collapse) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਬਣਤਰ htp_mm_kernel_params kparams;
            ਜੇ (collapse) {
                ਅਚਲ ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_ਪੰਕਤੀਆਂ(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } ਨਹੀਂਤਾਂ {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            ਜੇ (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ ((size_t) kparams.vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_ਮਾਪ, sess->vtcm_ਮਾਪ);
                ਵਾਪਸ ਝੂਠ;
            }

            ਜੇ (!try_fuse_common(w_ਵਿੱਚ, d_ਵਿੱਚ)) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਆਖਰੀ_ਗੰਢ.inputs[curr_n] = w_ਵਿੱਚ;
            ਆਖਰੀ_ਗੰਢ.inputs.push_back(x);
            ਆਖਰੀ_ਗੰਢ.outputs.push_back(d_ਵਿੱਚ);
            ਆਖਰੀ_ਗੰਢ.fused.push_back(ਗੰਢ.ਗੰਢ);
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਆਖਰੀ_ਗੰਢ.kernel_params, &kparams, ਆਕਾਰ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(o.kernel_params, &kparams, ਆਕਾਰ(kparams));

            ਲਈ (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = ਜੋੜੋ_tensor(ਆਖਰੀ_ਗੰਢ.inputs[s]);
            }
            ਲਈ (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            ਲਈ (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = ਜੋੜੋ_tensor(ਆਖਰੀ_ਗੰਢ.outputs[d]);
            }
            ਲਈ (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            ਵਾਪਸ ਸੱਚ;
        }

        // Case 2: last_node is single MUL_MAT
        ਜੇ (ਆਖਰੀ_ਗੰਢ.opcode == HTP_OP_MUL_MAT) {
            ਜੇ (!is_mergeable_mul_mat_pair(ਆਖਰੀ_ਗੰਢ.ਗੰਢ, ਗੰਢ.ਗੰਢ)) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਅਚਲ ggml_tensor * w0 = ਆਖਰੀ_ਗੰਢ.src0();
            ਅਚਲ ggml_tensor * x  = ਆਖਰੀ_ਗੰਢ.src1();
            ਅਚਲ ggml_tensor * w1 = ਗੰਢ.src0();
            ਅਚਲ ggml_tensor * dst_0 = ਆਖਰੀ_ਗੰਢ.dst();
            ਅਚਲ ggml_tensor * dst_1 = ਗੰਢ.dst();

            ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * orig_kparams = (ਅਚਲ ਬਣਤਰ htp_mm_kernel_params *) ਆਖਰੀ_ਗੰਢ.kernel_params;
            ਅਚਲ ਤਾਰਕਿਕ collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            ਜੇ (orig_kparams->collapse && !collapse) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਬਣਤਰ htp_mm_kernel_params kparams;
            ਜੇ (collapse) {
                ਅਚਲ ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_ਪੰਕਤੀਆਂ(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } ਨਹੀਂਤਾਂ {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            ਜੇ (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ ((size_t) kparams.vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_ਮਾਪ, sess->vtcm_ਮਾਪ);
                ਵਾਪਸ ਝੂਠ;
            }

            ਜੇ (!try_fuse_common(w1, ਗੰਢ.dst())) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਆਖਰੀ_ਗੰਢ.opcode = HTP_OP_MUL_MAT_NX;
            ਆਖਰੀ_ਗੰਢ.name   = "MUL_MAT_NX";
            ਆਖਰੀ_ਗੰਢ.inputs.clear();
            ਆਖਰੀ_ਗੰਢ.inputs.push_back(w0);
            ਆਖਰੀ_ਗੰਢ.inputs.push_back(w1);
            ਆਖਰੀ_ਗੰਢ.inputs.push_back(x);
            ਆਖਰੀ_ਗੰਢ.outputs.clear();
            ਆਖਰੀ_ਗੰਢ.outputs.push_back(dst_0);
            ਆਖਰੀ_ਗੰਢ.outputs.push_back(dst_1);
            ਆਖਰੀ_ਗੰਢ.fused.push_back(ਗੰਢ.ਗੰਢ);
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਆਖਰੀ_ਗੰਢ.kernel_params, &kparams, ਆਕਾਰ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(o.kernel_params, &kparams, ਆਕਾਰ(kparams));

            o.src[0] = ਜੋੜੋ_tensor(w0);
            o.src[1] = ਜੋੜੋ_tensor(w1);
            o.src[2] = ਜੋੜੋ_tensor(x);
            ਲਈ (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = ਜੋੜੋ_tensor(dst_0);
            o.dst[1] = ਜੋੜੋ_tensor(dst_1);
            ਲਈ (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            ਵਾਪਸ ਸੱਚ;
        }

        ਵਾਪਸ ਝੂਠ;
    }

    ਤਾਰਕਿਕ try_fuse_mul_mat_id_nx(ਅਚਲ htp_opnode & ਗੰਢ) {
        ਜੇ (n_ops == 0 || ਗੰਢ.opcode != HTP_OP_MUL_MAT_ID) ਵਾਪਸ ਝੂਠ;
        ਜੇ (!is_mergeable_mul_mat_id(ਗੰਢ.ਗੰਢ)) ਵਾਪਸ ਝੂਠ;

        ਅਚਲ ggml_tensor * w_ਵਿੱਚ   = ਗੰਢ.src0();
        ਅਚਲ ggml_tensor * x_ਵਿੱਚ   = ਗੰਢ.src1();
        ਅਚਲ ggml_tensor * ids_ਵਿੱਚ = ਗੰਢ.ਗੰਢ->src[2];
        ਅਚਲ ggml_tensor * d_ਵਿੱਚ   = ਗੰਢ.dst();

        htp_opnode & ਆਖਰੀ_ਗੰਢ = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        ਜੇ (ਆਖਰੀ_ਗੰਢ.opcode == HTP_OP_MUL_MAT_ID_NX) {
            ਅਚਲ uint32_t curr_n = (uint32_t) ਆਖਰੀ_ਗੰਢ.outputs.size();
            ਜੇ (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਅਚਲ ggml_tensor * w0  = ਆਖਰੀ_ਗੰਢ.inputs[0];
            ਅਚਲ ggml_tensor * x   = ਆਖਰੀ_ਗੰਢ.inputs[curr_n];
            ਅਚਲ ggml_tensor * ids = ਆਖਰੀ_ਗੰਢ.inputs[curr_n + 1];

            ਜੇ (x_ਵਿੱਚ != x || ids_ਵਿੱਚ != ids || w_ਵਿੱਚ->type != w0->type || w_ਵਿੱਚ->ne[0] != w0->ne[0] || w_ਵਿੱਚ->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_ਵਿੱਚ) != ggml_hexagon_tensor_is_tiled(w0)) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (!ਆਖਰੀ_ਗੰਢ.fused.ਖਾਲੀ_ਥਾਵਾਂ() && (mm_is_hmx_eligible(ਆਖਰੀ_ਗੰਢ.fused[0]) != mm_is_hmx_eligible(ਗੰਢ.ਗੰਢ))) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਬਣਤਰ htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_ਵਿੱਚ, curr_n + 1, &kparams);
            ਜੇ (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ ((size_t) kparams.vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_ਮਾਪ, sess->vtcm_ਮਾਪ);
                ਵਾਪਸ ਝੂਠ;
            }

            ਜੇ (!try_fuse_common(w_ਵਿੱਚ, d_ਵਿੱਚ)) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਆਖਰੀ_ਗੰਢ.inputs[curr_n] = w_ਵਿੱਚ;
            ਆਖਰੀ_ਗੰਢ.inputs[curr_n + 1] = x;
            ਆਖਰੀ_ਗੰਢ.inputs.push_back(ids);
            ਆਖਰੀ_ਗੰਢ.outputs.push_back(d_ਵਿੱਚ);
            ਆਖਰੀ_ਗੰਢ.fused.push_back(ਗੰਢ.ਗੰਢ);
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਆਖਰੀ_ਗੰਢ.kernel_params, &kparams, ਆਕਾਰ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(o.kernel_params, &kparams, ਆਕਾਰ(kparams));

            ਲਈ (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = ਜੋੜੋ_tensor(ਆਖਰੀ_ਗੰਢ.inputs[s]);
            }
            ਲਈ (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            ਲਈ (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = ਜੋੜੋ_tensor(ਆਖਰੀ_ਗੰਢ.outputs[d]);
            }
            ਲਈ (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            ਵਾਪਸ ਸੱਚ;
        }

        // Case 2: last_node is single MUL_MAT_ID
        ਜੇ (ਆਖਰੀ_ਗੰਢ.opcode == HTP_OP_MUL_MAT_ID) {
            ਜੇ (!is_mergeable_mul_mat_id_pair(ਆਖਰੀ_ਗੰਢ.ਗੰਢ, ਗੰਢ.ਗੰਢ)) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਅਚਲ ggml_tensor * w0  = ਆਖਰੀ_ਗੰਢ.src0();
            ਅਚਲ ggml_tensor * x   = ਆਖਰੀ_ਗੰਢ.src1();
            ਅਚਲ ggml_tensor * ids = ਆਖਰੀ_ਗੰਢ.ਗੰਢ->src[2];
            ਅਚਲ ggml_tensor * w1  = ਗੰਢ.src0();

            ਬਣਤਰ htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, ਗੰਢ.dst(), 2, &kparams);
            ਜੇ (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ ((size_t) kparams.vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_ਮਾਪ, sess->vtcm_ਮਾਪ);
                ਵਾਪਸ ਝੂਠ;
            }

            ਜੇ (!try_fuse_common(w1, ਗੰਢ.dst())) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਅਚਲ ggml_tensor * dst_0 = ਆਖਰੀ_ਗੰਢ.dst();
            ਅਚਲ ggml_tensor * dst_1 = ਗੰਢ.dst();

            ਆਖਰੀ_ਗੰਢ.opcode = HTP_OP_MUL_MAT_ID_NX;
            ਆਖਰੀ_ਗੰਢ.name   = "MUL_MAT_ID_NX";
            ਆਖਰੀ_ਗੰਢ.inputs.clear();
            ਆਖਰੀ_ਗੰਢ.inputs.push_back(w0);
            ਆਖਰੀ_ਗੰਢ.inputs.push_back(w1);
            ਆਖਰੀ_ਗੰਢ.inputs.push_back(x);
            ਆਖਰੀ_ਗੰਢ.inputs.push_back(ids);
            ਆਖਰੀ_ਗੰਢ.outputs.clear();
            ਆਖਰੀ_ਗੰਢ.outputs.push_back(dst_0);
            ਆਖਰੀ_ਗੰਢ.outputs.push_back(dst_1);
            ਆਖਰੀ_ਗੰਢ.fused.push_back(ਗੰਢ.ਗੰਢ);
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਆਖਰੀ_ਗੰਢ.kernel_params, &kparams, ਆਕਾਰ(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(o.kernel_params, &kparams, ਆਕਾਰ(kparams));

            o.src[0] = ਜੋੜੋ_tensor(w0);
            o.src[1] = ਜੋੜੋ_tensor(w1);
            o.src[2] = ਜੋੜੋ_tensor(x);
            o.src[3] = ਜੋੜੋ_tensor(ids);
            ਲਈ (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = ਜੋੜੋ_tensor(dst_0);
            o.dst[1] = ਜੋੜੋ_tensor(dst_1);
            ਲਈ (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            ਵਾਪਸ ਸੱਚ;
        }

        ਵਾਪਸ ਝੂਠ;
    }

    ਤਾਰਕਿਕ try_fuse_gdn_cpy(ਅਚਲ htp_opnode & ਗੰਢ) {
        ਜੇ (n_ops == 0 || ਗੰਢ.opcode != HTP_OP_CPY) ਵਾਪਸ ਝੂਠ;

        htp_opnode & ਆਖਰੀ_ਗੰਢ = ops[n_ops - 1];
        ਜੇ (ਆਖਰੀ_ਗੰਢ.opcode != HTP_OP_GATED_DELTA_NET) ਵਾਪਸ ਝੂਠ;
        ਜੇ (ਆਖਰੀ_ਗੰਢ.outputs.size() != 1) ਵਾਪਸ ਝੂਠ;

        ਅਚਲ ggml_tensor * gdn_ਬਾਹਰਲਾ  = ਆਖਰੀ_ਗੰਢ.dst();
        ਅਚਲ ggml_tensor * cpy_ਗੰਢ = ਗੰਢ.ਗੰਢ;
        ਅਚਲ ggml_tensor * cpy_src  = ਗੰਢ.src0();
        ਅਚਲ ggml_tensor * cpy_dst  = ਗੰਢ.dst();

        ਜੇ (!cpy_src || !cpy_dst || !cpy_dst->ਅੰਕੜੇ) ਵਾਪਸ ਝੂਠ;
        ਜੇ (gdn_ਬਾਹਰਲਾ->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) ਵਾਪਸ ਝੂਠ;
        ਜੇ ((gdn_ਬਾਹਰਲਾ->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_ਗੰਢ->flags & GGML_TENSOR_FLAG_OUTPUT)) ਵਾਪਸ ਝੂਠ;

        ਅਚਲ ggml_tensor * v = ਆਖਰੀ_ਗੰਢ.ਗੰਢ->src[2];
        ਜੇ (!v) ਵਾਪਸ ਝੂਠ;

        ਅਚਲ int64_t S_v      = v->ne[0];
        ਅਚਲ int64_t H        = v->ne[1];
        ਅਚਲ int64_t n_ਟੋਕਨ = v->ne[2];
        ਅਚਲ int64_t n_seqs   = v->ne[3];
        ਅਚਲ int64_t K        = ggml_get_op_params_i32(ਆਖਰੀ_ਗੰਢ.ਗੰਢ, 0);
        ਅਚਲ size_t  ਪੂਛ_off = (size_t) S_v * H * n_ਟੋਕਨ * n_seqs * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

        ਅਚਲ int64_t D         = S_v * S_v * H;
        ਅਚਲ int64_t n_written = std::min<int64_t>(n_ਟੋਕਨ, K);

        ਜੇ (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_ਬਾਹਰਲਾ && cpy_src->view_src->ਅੰਕੜੇ != gdn_ਬਾਹਰਲਾ->ਅੰਕੜੇ) ||
            cpy_src->view_offs != ਪੂਛ_off || !ggml_is_contiguous(cpy_src)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (n_written > 1) {
            ਜੇ (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
                ਵਾਪਸ ਝੂਠ;
            }
        }

        ਜੇ (!try_fuse_common({cpy_dst})) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਆਖਰੀ_ਗੰਢ.name += "+CPY";
        ਆਖਰੀ_ਗੰਢ.outputs.push_back(cpy_dst);
        ਆਖਰੀ_ਗੰਢ.fused.push_back(ਗੰਢ.ਗੰਢ);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = ਜੋੜੋ_tensor(cpy_dst);
        ਲਈ (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ try_fuse(ਅਚਲ htp_opnode & ਗੰਢ) {
        ਜੇ (!opt_opfusion) ਵਾਪਸ ਝੂਠ;
        ਜੇ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ਜੋੜੋ) && try_fuse_allreduce_ਜੋੜੋ(ਗੰਢ)) ਵਾਪਸ ਸੱਚ;
        ਜੇ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(ਗੰਢ))  ਵਾਪਸ ਸੱਚ;
        ਜੇ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ਜੋੜੋ)   && try_fuse_mul_mat_ਜੋੜੋ(ਗੰਢ))   ਵਾਪਸ ਸੱਚ;
        ਜੇ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(ਗੰਢ))    ਵਾਪਸ ਸੱਚ;
        ਜੇ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(ਗੰਢ)) ਵਾਪਸ ਸੱਚ;
        ਜੇ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(ਗੰਢ))       ਵਾਪਸ ਸੱਚ;
        ਵਾਪਸ ਝੂਠ;
    }
};

ਬਣਤਰ ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

ਬਣਤਰ ggml_hexagon_opqueue {
    // Shared buffer for storing batches
    ggml_hexagon_shared_ਬਫਰ *shm_buf;
    size_t                      shm_blk_ਮਾਪ;
    size_t                      ਡੂੰਘਾਈ;

    ਵਰਤੋਂ opvec = std::vector<htp_opnode>;

    std::vector<opvec>          op_cache;       // per batch op cache
    std::vector<uint64_t>       ਸ਼ੁਰੂ_usec;     // per batch start time

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, size_t batch_ਮਾਪ, size_t ਡੂੰਘਾਈ) : ਡੂੰਘਾਈ(ਡੂੰਘਾਈ) {
        size_t n_bufs    = HTP_OP_MAX_BUFS;
        size_t n_ops     = batch_ਮਾਪ;
        size_t n_tensors = n_ops * HTP_OP_MAX_OUTPUTS + n_ops * HTP_OP_MAX_INPUTS;

        size_t tr_ਮਾਪ = 0;
        ਜੇ (opt_profile == 3) {
            tr_ਮਾਪ = (HTP_MAX_NTHREADS + 1) * opt_optrace * ਆਕਾਰ(htp_trace_desc);
        }

        shm_blk_ਮਾਪ = ਆਕਾਰ(htp_buf_desc)  * n_bufs    +
                       ਆਕਾਰ(htp_tensor)    * n_tensors +
                       ਆਕਾਰ(htp_op_desc)   * n_ops     +
                       ਆਕਾਰ(htp_prof_desc) * n_ops     +
                       tr_ਮਾਪ;

        shm_buf = ਨਵਾਂ ggml_hexagon_shared_ਬਫਰ(sess, shm_blk_ਮਾਪ * ਡੂੰਘਾਈ, ਸੱਚ /* pinned */);

        op_cache.resize(ਡੂੰਘਾਈ);
        ਸ਼ੁਰੂ_usec.resize(ਡੂੰਘਾਈ, 0);

        ਜੇ (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_ਮਾਪ, ਡੂੰਘਾਈ, shm_buf->size(), shm_blk_ਮਾਪ);
        }
    }

    ~ggml_hexagon_opqueue() {
        ਮਿਟਾਓ shm_buf;
    }

    size_t shm_ਮਾਪ() ਅਚਲ { ਵਾਪਸ shm_buf ? shm_buf->size() : 0; }

    // push new batch
    ਤਾਰਕਿਕ ਧੱਕੋ(htp_opbatch_req& req, dspqueue_ਬਫਰ& dbuf, ਅਚਲ ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(ਆਕਾਰ(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(ਆਕਾਰ(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(ਆਕਾਰ(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(ਆਕਾਰ(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(ਆਕਾਰ(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(ਆਕਾਰ(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        ਜੇ (seq - shm_buf->sess->batch_rsp_seq > ਡੂੰਘਾਈ) { ਵਾਪਸ ਝੂਠ; }

        ਅਚਲ uint32_t ਥਾਂ = (uint32_t) ((seq - 1) % ਡੂੰਘਾਈ);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[ਥਾਂ]   = op_batch->ops;
        ਸ਼ੁਰੂ_usec[ਥਾਂ] = ggml_ਸਮਾਂ_us();

        ਅਚਲ size_t b_ਮਾਪ = ਆਕਾਰ(htp_buf_desc)  * req.n_bufs;
        ਅਚਲ size_t t_ਮਾਪ = ਆਕਾਰ(htp_tensor)    * req.n_tensors;
        ਅਚਲ size_t o_ਮਾਪ = ਆਕਾਰ(htp_op_desc)   * req.n_ops;
        ਅਚਲ size_t p_ਮਾਪ = ਆਕਾਰ(htp_prof_desc) * req.n_ops;

        size_t tr_ਮਾਪ = 0;
        ਜੇ (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_ਮਾਪ = (HTP_MAX_NTHREADS + 1) * req.n_traces * ਆਕਾਰ(htp_trace_desc);
        } ਨਹੀਂਤਾਂ {
            req.n_traces = 0;
        }

        dbuf.ptr      = shm_buf->ਅਧਾਰ() + ((size_t) ਥਾਂ * shm_blk_ਮਾਪ);
        dbuf.fd       = shm_buf->fd();
        dbuf.flags    = DSPQUEUE_ਬਫਰ_FLAG_FLUSH_SENDER | DSPQUEUE_ਬਫਰ_FLAG_INVALIDATE_RECIPIENT;
        dbuf.ਵਿੱਥ   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->ਅਧਾਰ();
        dbuf.size     = b_ਮਾਪ + t_ਮਾਪ + o_ਮਾਪ + p_ਮਾਪ + tr_ਮਾਪ;

        GGML_ASSERT(dbuf.size <= shm_blk_ਮਾਪ);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_ਮਾਪ;
        uint8_t * t_ptr = m_ptr; m_ptr += t_ਮਾਪ;
        uint8_t * o_ptr = m_ptr;

        ਸਮ੍ਰਿਤੀ_ਨਕਲ(b_ptr, (ਖਾਲੀ *) op_batch->h_bufs.ਅੰਕੜੇ(), b_ਮਾਪ);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(t_ptr, (ਖਾਲੀ *) op_batch->h_tens.ਅੰਕੜੇ(), t_ਮਾਪ);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(o_ptr, (ਖਾਲੀ *) op_batch->h_ops.ਅੰਕੜੇ(),  o_ਮਾਪ);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_ਮਾਪ, t_ਮਾਪ, o_ਮਾਪ, (size_t) dbuf.size);

        ਜੇ (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (ਖਾਲੀ *) b[i].ਅਧਾਰ, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) t[i].ਅੰਕੜੇ, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        ਵਾਪਸ ਸੱਚ;
    }

    ਖਾਲੀ ਕੱਢੋ(htp_opbatch_rsp rsp, dspqueue_ਬਫਰ dbuf) {
        ਅਚਲ uint32_t ਥਾਂ = (uint32_t) ((rsp.seq - 1) % ਡੂੰਘਾਈ);

        ਅਚਲ size_t b_ਮਾਪ = ਆਕਾਰ(htp_buf_desc)  * rsp.n_bufs;
        ਅਚਲ size_t t_ਮਾਪ = ਆਕਾਰ(htp_tensor)    * rsp.n_tensors;
        ਅਚਲ size_t o_ਮਾਪ = ਆਕਾਰ(htp_op_desc)   * rsp.n_ops;
        ਅਚਲ size_t p_ਮਾਪ = ਆਕਾਰ(htp_prof_desc) * rsp.n_ops;

        size_t tr_ਮਾਪ = 0;
        uint32_t n_traces = 0;
        ਜੇ (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_ਮਾਪ = (HTP_MAX_NTHREADS + 1) * n_traces * ਆਕਾਰ(htp_trace_desc);
        }

        ਅਚਲ size_t m_ਮਾਪ = b_ਮਾਪ + t_ਮਾਪ + o_ਮਾਪ + p_ਮਾਪ + tr_ਮਾਪ;
        GGML_ASSERT(m_ਮਾਪ <= shm_blk_ਮਾਪ);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_ਮਾਪ, t_ਮਾਪ, o_ਮਾਪ);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_ਮਾਪ + t_ਮਾਪ + o_ਮਾਪ);

        ਜੇ (rsp.n_ops > 0) {
            ਸਵੈਚਾਲਿਤ & ops = op_cache[ਥਾਂ];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            ਅਚਲ htp_prof_desc * pd = (ਅਚਲ htp_prof_desc *) p_ptr;
            ਅਚਲ htp_trace_desc * trace_events = ਖਾਲੀ_ਸੂਚਕ;
            ਜੇ (opt_profile == 3) {
                trace_events = (ਅਚਲ htp_trace_desc *) (p_ptr + p_ਮਾਪ);
            }

            ਜੇ (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            ਲਈ (uint32_t i = 0; i < rsp.n_ops; i++) {
                ਜੇ (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            ਜੇ (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

ਖਾਲੀ ggml_hexagon_session::flush_peers() {
    ਸਵੈਚਾਲਿਤ vpeers = std::move(virt_peers);
    virt_peers.clear();
    ਲਈ (ਸਵੈਚਾਲਿਤ * peer : vpeers) {
        peer->flush_sync();
    }

    ਸਵੈਚਾਲਿਤ ppeers = std::move(phys_peers);
    phys_peers.clear();
    ਲਈ (ਸਵੈਚਾਲਿਤ * peer : ppeers) {
        peer->flush_async();
    }

    ਲਈ (ਸਵੈਚਾਲਿਤ & sub : ਇਹ->mdev.sessions) {
        sub->flush_peers();
    }
}

ਖਾਲੀ ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

ਖਾਲੀ ggml_hexagon_session::flush_pending(ਤਾਰਕਿਕ all) {
    ਲਈ (ਸਵੈਚਾਲਿਤ & sub : ਇਹ->mdev.sessions) {
        sub->flush_pending(all);
        ਜੇ (sub->ਆਖਰੀ_error > HTP_ਸਥਿਤੀ_OK) {
            ਇਹ->ਆਖਰੀ_error = sub->ਆਖਰੀ_error.load();
        }
    }

    ਜਦੋਂ (ਇਹ->batch_rsp_seq < ਇਹ->batch_req_seq) {
        ਬਣਤਰ htp_opbatch_rsp rsp;
        uint32_t               rsp_ਮਾਪ;
        uint32_t               flags;

        ਬਣਤਰ dspqueue_ਬਫਰ dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        ਅਚਲ uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        ਪੂਰਨ err = dspqueue_read(ਇਹ->queue, &flags, 1, &n_dbufs, &dbuf, ਆਕਾਰ(rsp), &rsp_ਮਾਪ, (uint8_t *) &rsp, timeo);
        ਜੇ (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            ਜਾਰੀ;
        }

        ਜੇ (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        }

        // Basic sanity checks
        ਜੇ (rsp_ਮਾਪ != ਆਕਾਰ(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", ਇਹ->c_name(), rsp_ਮਾਪ, n_dbufs);
        }

        ਜੇ (rsp.ਸਥਿਤੀ > HTP_ਸਥਿਤੀ_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", ਇਹ->c_name(), ਸਥਿਤੀ_ਮੰਜ਼ਿਲ_str(rsp.ਸਥਿਤੀ));
            ਇਹ->ਆਖਰੀ_error = rsp.ਸਥਿਤੀ;
            ਲਈ (ਸਵੈਚਾਲਿਤ & sub : ਇਹ->mdev.sessions) {
                sub->ਆਖਰੀ_error = rsp.ਸਥਿਤੀ;
            }
        }

        op_ਕਤਾਰ->ਕੱਢੋ(rsp, dbuf);

        GGML_ASSERT(rsp.seq == ਇਹ->batch_rsp_seq + 1);
        ਇਹ->batch_rsp_seq = rsp.seq;

        ਜੇ (!all) ਰੋਕੋ;
    }
}

ਖਾਲੀ ggml_hexagon_session::flush_sync(ਤਾਰਕਿਕ all) {
    flush_async();
    flush_pending(all);
}

ਖਾਲੀ ggml_hexagon_session::ਸ਼ੁਰੂ_batch() {
    ਜੇ (ਇਹ->mdev.count > 1) {
        ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_mdev_ਸਮੂਹ();
    }
}

ਖਾਲੀ ggml_hexagon_session::flush_batch(size_t min_ops) {
    ਜੇ (op_batch->n_ops < min_ops) { ਵਾਪਸ; }

    op_batch->ਛਾਂਟੋ_buffers();

    htp_opbatch_req req {};
    dspqueue_ਬਫਰ dbuf{};

    ਅਚਲ uint64_t seq = ++ਇਹ->batch_req_seq;

    op_batch->update_mdev_ਸਮੂਹ(ਇਹ->mdev.idx);

    ਜੇ (!op_ਕਤਾਰ->ਧੱਕੋ(req, dbuf, op_batch, seq)) {
        flush_pending(ਝੂਠ);
        op_ਕਤਾਰ->ਧੱਕੋ(req, dbuf, op_batch, seq);
    }

    ਲਈ (ਸਵੈਚਾਲਿਤ & sub : ਇਹ->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_ਬਫਰ sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_ਸਮੂਹ(sub->mdev.idx);

        ਜੇ (!sub->op_ਕਤਾਰ->ਧੱਕੋ(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(ਝੂਠ);
            sub->op_ਕਤਾਰ->ਧੱਕੋ(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        ਪੂਰਨ err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, ਆਕਾਰ(sub_req), (ਅਚਲ uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        ਜੇ (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", ਇਹ->c_name(), dbuf.ptr, dbuf.size);

    ਪੂਰਨ err = dspqueue_write(ਇਹ->queue, 0, 1, &dbuf, ਆਕਾਰ(req), (ਅਚਲ uint8_t*) &req, DSPQUEUE_TIMEOUT);
    ਜੇ (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", ਇਹ->c_name(), (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
    }

    op_batch->reset();
}

ਖਾਲੀ ggml_hexagon_session::ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_op(ਅਚਲ htp_opnode & ਗੰਢ) {
    ਸਵੈਚਾਲਿਤ clone_tensor_ਬਫਰ = [ਇਹ](ਅਚਲ ggml_tensor * t) {
        ਸਵੈਚਾਲਿਤ sbuf = ਇਹ->mmap_tensor(t);
        ਜੇ (!sbuf) ਵਾਪਸ;
        ਜੇ (sbuf->sess != ਇਹ) {
            ਇਹ->clone_ਬਫਰ(sbuf);
        }
        ਲਈ (ਸਵੈਚਾਲਿਤ & sub : ਇਹ->mdev.sessions) {
            sub->clone_ਬਫਰ(sbuf);
        }
    };

    ਲਈ (ਸਵੈਚਾਲਿਤ t : ਗੰਢ.get_inputs()) {
        clone_tensor_ਬਫਰ(t);
    }
    ਲਈ (ਸਵੈਚਾਲਿਤ t : ਗੰਢ.get_outputs()) {
        clone_tensor_ਬਫਰ(t);
    }

    ਜੇ (opt_opfusion && op_batch->try_fuse(ਗੰਢ)) {
        ਵਾਪਸ;
    }

    ਜੇ (!op_batch->fit_op(ਗੰਢ)) {
        flush_async();
    }

    ਜੇ (op_batch->ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਸ਼ੁਰੂ_batch();
    }

    ਜੇ (!op_batch->fit_op(ਗੰਢ)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->ਜੋੜੋ_op(ਗੰਢ);
}

ਖਾਲੀ ggml_hexagon_session::ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_mdev_ਸਮੂਹ() {
    htp_opnode ਸਮੂਹ_ਗੰਢ(HTP_OP_MDEV_ਸਮੂਹ);

    uint8_t * fence_ਥਾਂ = ਇਹ->mdev_fence_ਥਾਂ;

    ਸਥਿਰ ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.ਬਫਰ       = &ਇਹ->fence_buf->backend_ਬਫਰ;
    dummy_t.extra        = &fence_extra;
    dummy_t.ਅੰਕੜੇ         = (ਖਾਲੀ *) fence_ਥਾਂ;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_ਥਾਂ_ਮਾਪ;
    dummy_t.ne[1]        = (int64_t) ਇਹ->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_ਥਾਂ_ਮਾਪ;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) ਇਹ->mdev.idx;

    ggml_tensor * ਗੰਢ = ਸਮੂਹ_ਗੰਢ.ਜੋੜੋ_dummy(dummy_t);
    ਗੰਢ->src[0] = ਗੰਢ;
    ਸਮੂਹ_ਗੰਢ.init(ਗੰਢ);
    ਸਮੂਹ_ਗੰਢ.outputs.clear();
    ਸਮੂਹ_ਗੰਢ.name = "MDEV_GROUP";

    ਲਈ (ਸਵੈਚਾਲਿਤ & sub : ਇਹ->mdev.sessions) {
        sub->clone_ਬਫਰ(ਇਹ->fence_buf);
    }

    op_batch->ਜੋੜੋ_op(ਸਮੂਹ_ਗੰਢ);
}

ਖਾਲੀ ggml_hexagon_session::ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_cpy(ਅਚਲ ggml_tensor * src, ggml_tensor * dst, ਅਚਲ ggml_tensor * sync_tensor, uint32_t fence_seq) {
    ਅਚਲ ਤਾਰਕਿਕ with_fence = sync_tensor != ਖਾਲੀ_ਸੂਚਕ;
    htp_opnode cpy_ਗੰਢ(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* ਗੰਢ = cpy_ਗੰਢ.ਜੋੜੋ_dummy(*dst);
    ਗੰਢ->op     = GGML_OP_CPY;
    ਗੰਢ->src[0] = const_cast<ggml_tensor *>(src);
    ਗੰਢ->src[1] = with_fence ? cpy_ਗੰਢ.ਜੋੜੋ_dummy(*sync_tensor) : ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (with_fence) {
        ਗੰਢ->op_params[0] = (int32_t) fence_seq;
    }

    cpy_ਗੰਢ.init(ਗੰਢ);
    ਜੇ (with_fence) {
        cpy_ਗੰਢ.name = "CPY+FENCE";
    }
    ਅਚਲ ਤਾਰਕਿਕ ok = ggml_hexagon_precompute_cpy_params(ਇਹ, ਗੰਢ, (ਬਣਤਰ htp_copy_kernel_params *) cpy_ਗੰਢ.kernel_params);
    ਅਚਲ ਸਵੈਚਾਲਿਤ * kparams = (ਅਚਲ ਬਣਤਰ htp_copy_kernel_params *) cpy_ਗੰਢ.kernel_params;
    ਜੇ (ok && !with_fence && kparams->ਕੁੱਲ_elems == 0) {
        ਵਾਪਸ;
    }
    ਇਹ->ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_op(cpy_ਗੰਢ);
}

ਖਾਲੀ ggml_hexagon_session::ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_fence(ਅਚਲ ggml_tensor * sync_tensor, uint32_t fence_seq, ਤਾਰਕਿਕ wait) {
    htp_opnode sync_ਗੰਢ(HTP_OP_FENCE);

    ggml_tensor* ਗੰਢ = sync_ਗੰਢ.ਜੋੜੋ_dummy(*sync_tensor);
    ਗੰਢ->op           = GGML_OP_NONE;
    ਗੰਢ->src[0]       = ਗੰਢ;
    ਗੰਢ->op_params[0] = (int32_t) fence_seq;
    ਗੰਢ->op_params[1] = wait ? 0 : 1;

    sync_ਗੰਢ.init(ਗੰਢ);
    sync_ਗੰਢ.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    ਇਹ->ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_op(sync_ਗੰਢ);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_allreduce_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    ਤਾਰਕਿਕ has_ਜੋੜੋ,
    ਤਾਰਕਿਕ is_ਪੰਕਤੀ_bcast,
    ਤਾਰਕਿਕ is_shard_ok,
    ਬਣਤਰ htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_ਪੰਕਤੀ_bcast = (has_ਜੋੜੋ && is_ਪੰਕਤੀ_bcast) ? 1 : 0;

    ਅਚਲ uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    ਅਚਲ uint32_t elem_ਮਾਪ = (dst->type == GGML_TYPE_F16) ? ਆਕਾਰ(ggml_fp16_t) : ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ ਤਾਰਕਿਕ is_contiguous = ggml_is_contiguous(dst);

    ਅਚਲ uint32_t ne0 = (uint32_t) dst->ne[0];
    ਅਚਲ uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    ਅਚਲ ਤਾਰਕਿਕ use_1d = is_contiguous && !(has_ਜੋੜੋ && is_ਪੰਕਤੀ_bcast && ne1 > 1);

    ਜੇ (has_ਜੋੜੋ) {
        // sharded reduce-scatter for contiguous in-place add
        ਜੇ (opt_ar_scatter && is_shard_ok && use_1d && !is_ਪੰਕਤੀ_bcast) {
            ਅਚਲ uint32_t rank_chunk_elems = hex_ਗੇੜਾ_ਉੱਪਰ((nelem + n_ranks - 1) / n_ranks, 128);
            ਅਚਲ uint32_t rank_elem_ਸ਼ੁਰੂ  = (std::min)(rank * rank_chunk_elems, nelem);
            ਅਚਲ uint32_t rank_elem_ਸਿਰਾ    = (std::min)(rank_elem_ਸ਼ੁਰੂ + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_ਸ਼ੁਰੂ = (int32_t) rank_elem_ਸ਼ੁਰੂ;
            kparams->rank_nelem      = (int32_t) (rank_elem_ਸਿਰਾ - rank_elem_ਸ਼ੁਰੂ);
        } ਨਹੀਂਤਾਂ ਜੇ (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_ਸ਼ੁਰੂ = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } ਨਹੀਂਤਾਂ {
            kparams->n_dsts          = 1;
            kparams->rank_elem_ਸ਼ੁਰੂ = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } ਨਹੀਂਤਾਂ {
        kparams->n_dsts = (int32_t) n_ranks;
        ਜੇ (use_1d) {
            ਅਚਲ uint32_t rank_chunk_elems = hex_ਗੇੜਾ_ਉੱਪਰ((nelem + n_ranks - 1) / n_ranks, 128);
            ਅਚਲ uint32_t rank_elem_ਸ਼ੁਰੂ  = (std::min)(rank * rank_chunk_elems, nelem);
            ਅਚਲ uint32_t rank_elem_ਸਿਰਾ    = (std::min)(rank_elem_ਸ਼ੁਰੂ + rank_chunk_elems, nelem);
            ਅਚਲ uint32_t rank_nelem       = rank_elem_ਸਿਰਾ - rank_elem_ਸ਼ੁਰੂ;
            kparams->rank_elem_ਸ਼ੁਰੂ        = (int32_t) rank_elem_ਸ਼ੁਰੂ;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ uint32_t rank_chunk_ਪੰਕਤੀਆਂ = (ne1 + n_ranks - 1) / n_ranks;
            ਅਚਲ uint32_t rank_r0         = (std::min)(rank * rank_chunk_ਪੰਕਤੀਆਂ, ne1);
            ਅਚਲ uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_ਪੰਕਤੀਆਂ, ne1);
            ਅਚਲ uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_ਸ਼ੁਰੂ       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_ਭਰਿਆ;

    ਜੇ (use_1d) {
        ਅਚਲ uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        ਅਚਲ uint32_t n_ਧਾਗੇ  = (std::min)((uint32_t) sess->n_ਧਾਗੇ, (std::max)(1u, rank_nelem / 128));
        kparams->n_ਧਾਗੇ = n_ਧਾਗੇ;
        ਅਚਲ size_t n_vtcm_buffers = htp_allreduce_vtcm_ਬਫਰ_ਗਿਣਤੀ(n_ranks, n_ਧਾਗੇ, has_ਜੋੜੋ, is_ਪੰਕਤੀ_bcast);

        uint32_t ਖੰਡ_elems = 65536;
        ਜੇ (ਖੰਡ_elems > rank_nelem / n_ਧਾਗੇ && rank_nelem / n_ਧਾਗੇ > 128) {
            ਖੰਡ_elems = hex_ਗੇੜਾ_ਉੱਪਰ(rank_nelem / (n_ਧਾਗੇ * 2), 128);
        }
        ਖੰਡ_elems = (std::max)(128u, ਖੰਡ_elems);

        kparams->ਖੰਡ_elems          = ਖੰਡ_elems;
        kparams->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread = 2 * ਖੰਡ_elems * elem_ਮਾਪ;
        kparams->vtcm_ਮਾਪ            = n_vtcm_buffers * kparams->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;

        ਜਦੋਂ ((size_t) kparams->vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ && ਖੰਡ_elems > 128) {
            ਅਚਲ size_t max_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_buf = sess->vtcm_ਮਾਪ / (n_vtcm_buffers * 2);
            ਖੰਡ_elems = (uint32_t) hex_align_ਥੱਲੇ((size_t) (max_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_buf / elem_ਮਾਪ), 128);
            ਜੇ (ਖੰਡ_elems < 128) ਰੋਕੋ;
            kparams->ਖੰਡ_elems          = ਖੰਡ_elems;
            kparams->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread = 2 * ਖੰਡ_elems * elem_ਮਾਪ;
            kparams->vtcm_ਮਾਪ            = n_vtcm_buffers * kparams->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;
        }

        ਜੇ (sess->vtcm_ਮਾਪ < (size_t) kparams->vtcm_ਮਾਪ || ਖੰਡ_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_ਮਾਪ, sess->vtcm_ਮਾਪ);
            ਵਾਪਸ ਝੂਠ;
        }

        kparams->elems_ਪ੍ਰਤੀ_thread = hex_ਗੇੜਾ_ਉੱਪਰ((rank_nelem + n_ਧਾਗੇ - 1) / n_ਧਾਗੇ, ਖੰਡ_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        ਵਾਪਸ ਸੱਚ;
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        ਅਚਲ uint32_t n_ਧਾਗੇ  = (std::min)((uint32_t) sess->n_ਧਾਗੇ, (std::max)(1u, rank_nrows));
        kparams->n_ਧਾਗੇ = n_ਧਾਗੇ;
        ਅਚਲ size_t n_vtcm_buffers = htp_allreduce_vtcm_ਬਫਰ_ਗਿਣਤੀ(n_ranks, n_ਧਾਗੇ, has_ਜੋੜੋ, is_ਪੰਕਤੀ_bcast);

        ਅਚਲ uint32_t ਪੰਕਤੀ_ਬਾਈਟਾਂ = ne0 * elem_ਮਾਪ;
        ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ_aligned = (uint32_t) hex_align_ਉੱਪਰ(ਪੰਕਤੀ_ਬਾਈਟਾਂ, 128);
        kparams->ਪੰਕਤੀ_ਮਾਪ_aligned = ਪੰਕਤੀ_ਮਾਪ_aligned;

        ਅਚਲ uint32_t nrows_ਪ੍ਰਤੀ_thread = (rank_nrows + n_ਧਾਗੇ - 1) / n_ਧਾਗੇ;
        uint32_t ਖੰਡ_ਪੰਕਤੀਆਂ = (std::min)(128u, nrows_ਪ੍ਰਤੀ_thread);
        ਖੰਡ_ਪੰਕਤੀਆਂ = (std::max)(1u, ਖੰਡ_ਪੰਕਤੀਆਂ);
        kparams->ਖੰਡ_elems = ਖੰਡ_ਪੰਕਤੀਆਂ;

        kparams->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread = 2 * (ਖੰਡ_ਪੰਕਤੀਆਂ * ਪੰਕਤੀ_ਮਾਪ_aligned);
        kparams->vtcm_ਮਾਪ            = n_vtcm_buffers * kparams->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;

        ਜਦੋਂ ((size_t) kparams->vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ && ਖੰਡ_ਪੰਕਤੀਆਂ > 1) {
            ਅਚਲ size_t max_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_buf = sess->vtcm_ਮਾਪ / (n_vtcm_buffers * 2 * ਪੰਕਤੀ_ਮਾਪ_aligned);
            ਖੰਡ_ਪੰਕਤੀਆਂ = (std::max)(1u, (uint32_t) max_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_buf);
            kparams->ਖੰਡ_elems          = ਖੰਡ_ਪੰਕਤੀਆਂ;
            kparams->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread = 2 * (ਖੰਡ_ਪੰਕਤੀਆਂ * ਪੰਕਤੀ_ਮਾਪ_aligned);
            kparams->vtcm_ਮਾਪ            = n_vtcm_buffers * kparams->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;
            ਜੇ (max_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_buf == 0) ਰੋਕੋ;
        }

        ਜੇ (sess->vtcm_ਮਾਪ < (size_t) kparams->vtcm_ਮਾਪ || ਖੰਡ_ਪੰਕਤੀਆਂ < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_ਮਾਪ, sess->vtcm_ਮਾਪ);
            ਵਾਪਸ ਝੂਠ;
        }

        kparams->elems_ਪ੍ਰਤੀ_thread = nrows_ਪ੍ਰਤੀ_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        ਵਾਪਸ ਸੱਚ;
    }
}

ਖਾਲੀ ggml_hexagon_session::ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_allreduce(
    ਅਚਲ ggml_tensor * dst,
    ਅਚਲ std::vector<ਅਚਲ ggml_tensor *> & src_tensors,
    ਅਚਲ std::vector<ਅਚਲ ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_ਇੰਦਰਾਜ,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_ਗੰਢ(HTP_OP_ALLREDUCE);

    ggml_tensor* ਗੰਢ = ar_ਗੰਢ.ਜੋੜੋ_dummy(*dst);
    ਗੰਢ->op           = GGML_OP_NONE;
    ਗੰਢ->op_params[0] = (int32_t) fence_seq_ਇੰਦਰਾਜ;
    ਗੰਢ->op_params[1] = (int32_t) fence_seq_exit;

    ar_ਗੰਢ.init(ਗੰਢ);

    ar_ਗੰਢ.inputs.clear();
    ਲਈ (size_t i = 0; i < src_tensors.size(); i++) {
        ar_ਗੰਢ.inputs.push_back(src_tensors[i]);
    }
    ਲਈ (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_ਗੰਢ.inputs.push_back(ar_ਗੰਢ.ਜੋੜੋ_dummy(*sync_tensors[i]));
    }

    ar_ਗੰਢ.outputs.clear();
    ਲਈ (size_t i = 0; i < src_tensors.size(); i++) {
        ar_ਗੰਢ.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        ਇਹ, dst, rank, n_ranks, ਝੂਠ, ਝੂਠ, /*is_shard_ok=*/ ਝੂਠ,
        (ਬਣਤਰ htp_allreduce_kernel_params *) ar_ਗੰਢ.kernel_params);

    ar_ਗੰਢ.name = "ALLREDUCE";
    ਇਹ->ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_op(ar_ਗੰਢ);
}

ggml_hexagon_shared_ਬਫਰ * ggml_hexagon_session::mmap_tensor(ਅਚਲ ggml_tensor * t) {
    ਜੇ (!t) ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;

    ਸਵੈਚਾਲਿਤ sbuf = static_cast<ggml_hexagon_shared_ਬਫਰ *>(t->ਬਫਰ->context);
    ਜੇ (!sbuf->mapped) {
        ਅਚਲ ਤਾਰਕਿਕ is_ਭਾਰ = ggml_backend_ਬਫਰ_get_usage(t->ਬਫਰ) == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS;
        ਅਚਲ ਤਾਰਕਿਕ extended  = opt_dma64 && is_ਭਾਰ;
        sbuf->mmap(extended);
    }
    ਵਾਪਸ sbuf;
}

ਤਾਰਕਿਕ ggml_hexagon_session::clone_ਬਫਰ(ਅਚਲ ggml_hexagon_shared_ਬਫਰ * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    ਜੇ (sbuf->sess == ਇਹ) ਵਾਪਸ ਸੱਚ;

    ਸਵੈਚਾਲਿਤ mem = sbuf->mem;
    ਪੂਰਨ   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    ਜੇ (ਇਹ->cloned_buffers.find(fd) != ਇਹ->cloned_buffers.end()) ਵਾਪਸ ਸੱਚ;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", ਇਹ->name.c_str(),
                sbuf->c_name(), sbuf->ਅਧਾਰ(), sbuf->size(), fd);

    ਸਵੈਚਾਲਿਤ clone = std::make_unique<ggml_hexagon_shared_ਬਫਰ>(ਇਹ, *sbuf);
    ਕੋਸ਼ਿਸ਼ {
        clone->mmap(sbuf->extended);
    } ਫੜੋ (ਅਚਲ std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", ਇਹ->c_name(), exc.what());
        ਵਾਪਸ ਝੂਠ;
    }

    ਇਹ->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(ਇਹ);
    ਵਾਪਸ ਸੱਚ;
}

ਖਾਲੀ ggml_hexagon_session::release_ਬਫਰ(ਅਚਲ ggml_hexagon_shared_ਬਫਰ * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    ਸਵੈਚਾਲਿਤ mem = sbuf->mem;
    ਪੂਰਨ   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    ਸਵੈਚਾਲਿਤ it = ਇਹ->cloned_buffers.find(fd);
    ਜੇ (it != ਇਹ->cloned_buffers.end()) {
        ਸਵੈਚਾਲਿਤ clone = std::move(it->second);
        ਇਹ->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(ਇਹ);
}

ਖਾਲੀ ggml_hexagon_session::unclone_ਬਫਰ(ਅਚਲ ggml_hexagon_shared_ਬਫਰ * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    ਸਵੈਚਾਲਿਤ mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    ਲਈ (ਸਵੈਚਾਲਿਤ * sess : sessions) {
        sess->release_ਬਫਰ(sbuf);
    }
}

ਸਥਿਰ size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_ਬਫਰ *> sbufs;

    ਅਚਲ size_t MiB = 1024 * 1024;
    ਅਚਲ size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t ਕਦਮ = 256u * MiB;

    ਕੋਸ਼ਿਸ਼ {
        sbufs.push_back(ਨਵਾਂ ggml_hexagon_shared_ਬਫਰ(sess, GiB, ਸੱਚ)); vmem += GiB;
        sbufs.push_back(ਨਵਾਂ ggml_hexagon_shared_ਬਫਰ(sess, GiB, ਸੱਚ)); vmem += GiB;
        sbufs.push_back(ਨਵਾਂ ggml_hexagon_shared_ਬਫਰ(sess, GiB, ਸੱਚ)); vmem += GiB;

        ਜਦੋਂ (1) {
            sbufs.push_back(ਨਵਾਂ ggml_hexagon_shared_ਬਫਰ(sess, ਕਦਮ, ਸੱਚ));
            vmem += ਕਦਮ;
        }
    } ਫੜੋ (...) { }

    ਲਈ (ਸਵੈਚਾਲਿਤ b : sbufs) { ਮਿਟਾਓ b; }

    ਵਾਪਸ vmem - ਕਦਮ; // backoff to account for overhead from internal mappings
}

ਖਾਲੀ ggml_hexagon_session::allocate(ਅਚਲ ggml_hexagon_device_config & config) noexcept(ਝੂਠ) {
    ਪੂਰਨ phys_idx = config.physical_idx;
    ਪੂਰਨ virt_idx = config.virtual_idx;

    ਇਹ->valid_session = ਝੂਠ;
    ਇਹ->valid_handle  = ਝੂਠ;
    ਇਹ->valid_ਕਤਾਰ   = ਝੂਠ;
    ਇਹ->valid_iface   = ਝੂਠ;

    ਇਹ->name          = config.name;
    ਇਹ->phys_idx      = phys_idx;
    ਇਹ->virt_idx      = virt_idx;
    ਇਹ->domain_id     = config.domain_id;
    ਇਹ->session_id    = 0;
    ਇਹ->batch_req_seq = 0;
    ਇਹ->batch_rsp_seq = 0;
    ਇਹ->ਆਖਰੀ_error    = HTP_ਸਥਿਤੀ_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", ਇਹ->name.c_str(), ਇਹ->domain_id, phys_idx, virt_idx);

    ਜੇ (config.domain_id < 0 || config.domain_name.ਖਾਲੀ_ਥਾਵਾਂ()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        ਸੁੱਟੋ std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    ਅਚਲ std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    ਜੇ (virt_idx > 0) {
        ਬਣਤਰ remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<ਅੱਖਰ *>(dom_name.c_str());
        n.session_name     = const_cast<ਅੱਖਰ *>(ਇਹ->name.c_str());
        n.session_name_len = ਇਹ->name.size();
        n.session_id       = virt_idx;

        ਪੂਰਨ err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (ਖਾਲੀ *) &n, ਆਕਾਰ(n));
        ਜੇ (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           ਇਹ->c_name(), phys_idx, virt_idx, err);
            ਸੁੱਟੋ std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        ਇਹ->session_id    = n.session_id;
        ਇਹ->domain_id     = n.effective_domain_id;
        ਇਹ->valid_session = ਸੱਚ;
    } ਨਹੀਂਤਾਂ {
        ਬਣਤਰ remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<ਅੱਖਰ *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        ਪੂਰਨ err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (ਖਾਲੀ *) &eff, ਆਕਾਰ(eff));
        ਜੇ (err == AEE_SUCCESS) {
            ਇਹ->domain_id = eff.effective_domain_id;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           ਇਹ->name.c_str(), err, ਇਹ->domain_id);
        }
    }

    // Enable unsigned modules
    {
        ਬਣਤਰ remote_rpc_control_unsigned_module u;
        u.domain = ਇਹ->domain_id;
        u.enable = 1;
        ਪੂਰਨ err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (ਖਾਲੀ *) &u, ਆਕਾਰ(u));
        ਜੇ (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", ਇਹ->c_name(), err);
            ਸੁੱਟੋ std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    ਅੱਖਰ session_uri[256];
    {
        ਅੱਖਰ htp_uri[256];
        snprintf(htp_uri, ਆਕਾਰ(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        ਬਣਤਰ remote_rpc_get_uri u = {};
        u.session_id      = ਇਹ->session_id;
        u.domain_name     = const_cast<ਅੱਖਰ *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<ਅੱਖਰ *>(htp_uri);
        u.module_uri_len  = ਲੜੀ_ਲੰਬਾਈ(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = ਆਕਾਰ(session_uri);

        ਪੂਰਨ err = remote_session_control(FASTRPC_GET_URI, (ਖਾਲੀ *) &u, ਆਕਾਰ(u));
        ਜੇ (err != AEE_SUCCESS) {
            snprintf(session_uri, ਆਕਾਰ(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), ਇਹ->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          ਇਹ->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    ਪੂਰਨ err = htp_iface_open(session_uri, &ਇਹ->handle);
    ਜੇ (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", ਇਹ->c_name(), session_uri, err);
        ਸੁੱਟੋ std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    ਇਹ->valid_handle = ਸੱਚ;

    // Query HW info and resolve session options
    {
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ hw_n_ਧਾਗੇ = 0;
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ hw_n_hvx     = 0;
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ hw_n_hmx     = 0;
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ hw_vtcm_ਮਾਪ = 0;
        ਪੂਰਨ hw_err = htp_iface_hwinfo(ਇਹ->handle, &hw_n_ਧਾਗੇ, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_ਮਾਪ);
        ਜੇ (hw_err == 0) {
            ਅਚਲ uint32_t max_n_ਧਾਗੇ = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_ਧਾਗੇ);
            ਇਹ->n_ਧਾਗੇ = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_ਧਾਗੇ) : max_n_ਧਾਗੇ;
            ਇਹ->n_hvx     = ਇਹ->n_ਧਾਗੇ;
            ਇਹ->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            ਇਹ->vtcm_ਮਾਪ = (uint64_t)hw_vtcm_ਮਾਪ;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          ਇਹ->c_name(), ਇਹ->n_ਧਾਗੇ, ਇਹ->n_hvx, ਇਹ->n_hmx,
                          (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ)(ਇਹ->vtcm_ਮਾਪ / (1024 * 1024)));
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", ਇਹ->c_name(), hw_err);
            ਅਚਲ uint32_t default_n_ਧਾਗੇ = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            ਇਹ->n_ਧਾਗੇ = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_ਧਾਗੇ;
            ਇਹ->n_hvx     = ਇਹ->n_ਧਾਗੇ;
            ਇਹ->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            ਇਹ->vtcm_ਮਾਪ = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        ਬਣਤਰ remote_rpc_control_latency l;
        l.enable = 1;

        ਪੂਰਨ err = remote_handle64_control(ਇਹ->handle, DSPRPC_CONTROL_LATENCY, (ਖਾਲੀ *) &l, ਆਕਾਰ(l));
        ਜੇ (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", ਇਹ->c_name(),
                  ਇਹ->session_id, ਇਹ->domain_id, session_uri, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) ਇਹ->handle);

    ਅਚਲ size_t req_q_ਮਾਪ = (ਆਕਾਰ(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    ਅਚਲ size_t rsp_q_ਮਾਪ = (ਆਕਾਰ(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(ਇਹ->domain_id,
                          0,              // Flags
                          req_q_ਮਾਪ,     // Request  queue size (in bytes)
                          rsp_q_ਮਾਪ,     // Response queue size (in bytes)
                          ਖਾਲੀ_ਸੂਚਕ,        // Read packet callback (we handle reads explicitly)
                          ਖਾਲੀ_ਸੂਚਕ,        // Error callback (we handle errors during reads)
                          (ਖਾਲੀ *) ਇਹ,  // Callback context
                          &queue);
    ਜੇ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", ਇਹ->name.c_str(), (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        ਸੁੱਟੋ std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    ਇਹ->valid_ਕਤਾਰ = ਸੱਚ;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &ਇਹ->ਕਤਾਰ_id);
    ਜੇ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        ਸੁੱਟੋ std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    ਜੇ (opt_etm) {
        err = htp_iface_etm(ਇਹ->handle, 1);
        ਜੇ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        }
    }

    // Allocate buffers and state for op batching
    ਇਹ->op_ਕਤਾਰ = ਨਵਾਂ ggml_hexagon_opqueue(ਇਹ, opt_opbatch, opt_opqueue);

    ਇਹ->fence_buf = ਨਵਾਂ ggml_hexagon_fence_ਬਫਰ(ਇਹ, &dev_ctx->fence_ਬਫਰ_type, 64 * 1024);
    ਜੇ (ਇਹ->mdev.count > 1) {
        ਇਹ->mdev_fence_ਥਾਂ = ਇਹ->alloc_fence(ਇਹ->mdev.count);
    }

    ਜੇ (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(ਇਹ);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", ਇਹ->c_name(), opt_vmem);
    }
    ਅਚਲ size_t shm_ਮਾਪ = ਇਹ->op_ਕਤਾਰ->shm_ਮਾਪ();
    ਇਹ->max_vmem = (opt_vmem > shm_ਮਾਪ) ? (opt_vmem - shm_ਮਾਪ) : opt_vmem;

    ਇਹ->op_batch = ਨਵਾਂ ggml_hexagon_opbatch(ਇਹ, opt_opbatch, ਇਹ->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_ਸ਼ੁਰੂ(ਇਹ->handle, ਇਹ->session_id, ਇਹ->ਕਤਾਰ_id, ਇਹ->n_ਧਾਗੇ, opt_nhmx, ਇਹ->max_vmem, (uint32_t) opt_power_ਬਹੁਲਕ);
    ਜੇ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", ਇਹ->c_name(), (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        ਸੁੱਟੋ std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    ਇਹ->valid_iface = ਸੱਚ;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", ਇਹ->c_name(), htp_power_ਬਹੁਲਕ_ਮੰਜ਼ਿਲ_str(opt_power_ਬਹੁਲਕ));

    ਜੇ (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(ਇਹ->handle, opt_profile, &pmu_conf);
        ਜੇ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        }
    }
}

ਖਾਲੀ ggml_hexagon_session::release() noexcept(ਸੱਚ) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", ਇਹ->name.c_str());

    ਇਹ->mdev.sessions.clear();

    ਪੂਰਨ err;

    ਜੇ (ਇਹ->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(ਇਹ->handle);
        ਜੇ (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        }
    }

    ਮਿਟਾਓ ਇਹ->op_batch;
    ਮਿਟਾਓ ਇਹ->op_ਕਤਾਰ;
    ਲਈ (ਸਵੈਚਾਲਿਤ & it : ਇਹ->cpy_fence_slots) {
        free_fence((ਖਾਲੀ *) it.second, 1);
    }
    ਇਹ->cpy_fence_slots.clear();

    ਜੇ (ਇਹ->fence_buf) {
        unclone_ਬਫਰ(ਇਹ->fence_buf);
        ਮਿਟਾਓ ਇਹ->fence_buf;
        ਇਹ->fence_buf = ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜਦੋਂ (!ਇਹ->cloned_buffers.ਖਾਲੀ_ਥਾਵਾਂ()) {
        release_ਬਫਰ(ਇਹ->cloned_buffers.begin()->second.get());
    }

    ਜੇ (opt_etm) {
        err = htp_iface_etm(ਇਹ->handle, 0);
        ਜੇ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        }
    }

    ਜੇ (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(ਇਹ->handle, 0, &pmu_conf);
        ਜੇ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        }
    }

    ਜੇ (ਇਹ->valid_ਕਤਾਰ) {
        err = dspqueue_close(queue);
        ਜੇ (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        }
    }

    ਜੇ (ਇਹ->valid_handle) {
        htp_iface_close(ਇਹ->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(ਅਚਲ ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_ਗਿਣਤੀ) noexcept(ਝੂਠ) {
    ਇਹ->dev        = dev;
    ਇਹ->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ਇਹ->mdev.idx   = mdev_idx;
    ਇਹ->mdev.count = mdev_ਗਿਣਤੀ > 0 ? mdev_ਗਿਣਤੀ : (uint32_t) (1 + config.mdev_ਸਮੂਹ.size());
    op_batch         = ਖਾਲੀ_ਸੂਚਕ;
    op_ਕਤਾਰ         = ਖਾਲੀ_ਸੂਚਕ;
    fence_buf        = ਖਾਲੀ_ਸੂਚਕ;
    fence_seq        = ((uintptr_t)ਇਹ) & 0xFFFF;

    ਕੋਸ਼ਿਸ਼ {
        allocate(config);
        ਜੇ (ਇਹ->mdev.idx == 0 && !config.mdev_ਸਮੂਹ.ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਲਈ (size_t i = 0; i < config.mdev_ਸਮੂਹ.size(); i++) {
                ਇਹ->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_ਸਮੂਹ[i], ਇਹ->dev, (uint32_t) (i + 1), ਇਹ->mdev.count));
            }
        }
    } ਫੜੋ (ਅਚਲ std::exception & exc) {
        release();
        ਸੁੱਟੋ;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(ਸੱਚ) {
    release();
}

// ** backend interface

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_flash_attn_is_hmx_eligible(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * q,
    ਅਚਲ ਬਣਤਰ ggml_tensor * k,
    ਅਚਲ ਬਣਤਰ ggml_tensor * v,
    ਅਚਲ ਬਣਤਰ ggml_tensor * sinks
) {
    ਜੇ (sess->n_hmx == 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (opt_fa_select < 2) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t DK = q->ne[0];
    ਅਚਲ uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    ਜੇ (DK % 8 != 0 || DV % 8 != 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    GGML_ਅਣਵਰਤਿਆ(sinks);

    // Explicit force mode
    ਜੇ (opt_fa_select > 2) {
        ਵਾਪਸ ਸੱਚ;
    }

    ਅਚਲ uint32_t M = q->ne[1];

    // Prefill or batched decode
    ਜੇ (M > 1) {
        ਵਾਪਸ ਸੱਚ;
    }

    ਅਚਲ uint32_t n_ਸਿਰ = q->ne[2];
    ਅਚਲ uint32_t n_kv_heads = k->ne[2];
    ਅਚਲ uint32_t G = n_kv_heads > 0 ? n_ਸਿਰ / n_kv_heads : 1;
    ਅਚਲ uint32_t S = k->ne[1];
    ਅਚਲ uint32_t ਕੁੱਲ_ਧਾਗੇ = sess->n_ਧਾਗੇ * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    ਜੇ (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < ਕੁੱਲ_ਧਾਗੇ))) {
        ਵਾਪਸ ਸੱਚ;
    }

    // Tile alignment for 32-row HMX tiles
    ਅਚਲ ਤਾਰਕਿਕ is_tile_aligned = (G > 0 && (32 % G == 0));
    ਜੇ (!is_tile_aligned) {
        ਜੇ (sess->n_ਧਾਗੇ >= 6) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਵਾਪਸ S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    ਜੇ (DK == 256) {
        s_cross = 4096;
    } ਨਹੀਂਤਾਂ ਜੇ (DK <= 64) {
        s_cross = (sess->n_ਧਾਗੇ >= 8) ? 2048 : ((sess->n_ਧਾਗੇ >= 6) ? 768 : 512);
    } ਨਹੀਂਤਾਂ {
        s_cross = (sess->n_ਧਾਗੇ >= 8) ? 1024 : 512;
    }

    ਵਾਪਸ S >= s_cross;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_flash_attn_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_fa_kernel_params * kparams
) {
    ਜੇ (opt_fa_select < 1) {
        ਵਾਪਸ ਝੂਠ;
    }

    memset(kparams, 0, ਆਕਾਰ(*kparams));

    ਅਚਲ ਬਣਤਰ ggml_tensor * q    = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * k    = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * v    = op->src[2];
    ਅਚਲ ਬਣਤਰ ggml_tensor * ਨਕਾਬ = op->src[3];

    ਅਚਲ uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ਅਚਲ uint32_t neq1 = q->ne[1];  // n_tokens
    ਅਚਲ uint32_t neq2 = q->ne[2];  // n_heads

    ਅਚਲ uint32_t nek1 = k->ne[1];  // kv_len

    ਅਚਲ uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ਅਚਲ uint32_t DK = neq0;
    ਅਚਲ uint32_t DV = nev0;

    ਅਚਲ uint32_t n_kv_heads = k->ne[2];
    ਅਚਲ uint32_t G          = neq2 / n_kv_heads;

    ਦਸ਼ਮਲਵ scale         = 1.0f;
    ਦਸ਼ਮਲਵ max_bias      = 0.0f;
    ਦਸ਼ਮਲਵ logit_softcap = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,         &op->op_params[0], ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias,      &op->op_params[1], ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, &op->op_params[2], ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਜੇ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->ਸਿਰ_ਵੰਡੋ = (opt_fa_ਸਿਰ_ਵੰਡੋ != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    ਅਚਲ uint32_t n_ਸਿਰ = q->ne[2];
    kparams->n_ਸਿਰ_log2 = 1u << (uint32_t) std::floor(std::log2(n_ਸਿਰ));
    kparams->m0 = std::ਘਾਤ(2.0f, -(max_bias) / kparams->n_ਸਿਰ_log2);
    kparams->m1 = std::ਘਾਤ(2.0f, -(max_bias / 2.0f) / kparams->n_ਸਿਰ_log2);

    // Check HMX eligibility
    ਅਚਲ ਬਣਤਰ ggml_tensor * sinks = op->src[4];
    ਜੇ (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        ਅਚਲ uint32_t DK_pad = hex_ਗੇੜਾ_ਉੱਪਰ(DK, 64);
        ਅਚਲ uint32_t DV_pad = hex_ਗੇੜਾ_ਉੱਪਰ(DV, 64);
        size_t Br = 0, Bc = 0;
        ਪੂਰਨ ret = hmx_fa_ਲੱਭੋ_chunk_ਮਾਪ(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_ਮਾਪ, sess->n_ਧਾਗੇ, (q->type == GGML_TYPE_F32), sinks != ਖਾਲੀ_ਸੂਚਕ, n_ਸਿਰ);
        ਜੇ (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            ਜੇ (logit_softcap == 0.0f) {
                kparams->scale = scale * (ਦਸ਼ਮਲਵ) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } ਨਹੀਂਤਾਂ {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (ਦਸ਼ਮਲਵ) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_ਖੰਡਾਂ = (nek1 + Bc - 1) / Bc;
            kparams->n_ਧਾਗੇ = (kparams->n_kv_ਖੰਡਾਂ >= 3 && sess->n_ਧਾਗੇ >= 2) ? sess->n_ਧਾਗੇ : 1;

            kparams->u.hmx.g_br = hex_align_ਉੱਪਰ(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_ਖੰਡਾਂ >= 3 && sess->n_ਧਾਗੇ >= 2) ? 1 : 0;
            kparams->vtcm_ਮਾਪ = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_ਧਾਗੇ, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != ਖਾਲੀ_ਸੂਚਕ, n_ਸਿਰ);

            ਅਚਲ size_t ਪੰਕਤੀ_vec_ਬਾਈਟਾਂ = hex_align_ਉੱਪਰ(Bc * ਆਕਾਰ(uint16_t), 256);
            kparams->u.hmx.ਪੰਕਤੀ_buf_stride = ਪੰਕਤੀ_vec_ਬਾਈਟਾਂ / 128; // HVX vector is 128 bytes

            ਅਚਲ size_t m_ਸਤਰ_ਬਾਈਟਾਂ = hex_align_ਉੱਪਰ(Bc * ਆਕਾਰ(uint16_t), 128);
            kparams->u.hmx.ਨਕਾਬ_buf_ਪੰਕਤੀ_stride = m_ਸਤਰ_ਬਾਈਟਾਂ / ਆਕਾਰ(uint16_t);
            kparams->u.hmx.ਨਕਾਬ_ਪ੍ਰਸਾਰਣ = (ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ && ਨਕਾਬ->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_ਮੁੱਲਾਂ(G);
            ਜੇ (ਨਕਾਬ) {
                kparams->src3_div2 = init_fastdiv_ਮੁੱਲਾਂ(ਨਕਾਬ->ne[2]);
                kparams->src3_div3 = init_fastdiv_ਮੁੱਲਾਂ(ਨਕਾਬ->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_ਪ੍ਰਤੀ_thread = 0;
            ਵਾਪਸ ਸੱਚ;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_ਖੰਡਾਂ = (k->ne[1] + 64 - 1) / 64;
    kparams->n_ਧਾਗੇ = sess->n_ਧਾਗੇ;

    ਅਚਲ size_t ਮਾਪ_q_ਪੰਕਤੀ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    ਅਚਲ size_t ਮਾਪ_k_ਪੰਕਤੀ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(k->ne[0] * 2, 128);
    ਅਚਲ size_t ਮਾਪ_v_ਪੰਕਤੀ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(v->ne[0] * 2, 128);

    kparams->vtcm_ਮਾਪ = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), ਨਕਾਬ != ਖਾਲੀ_ਸੂਚਕ, sinks != ਖਾਲੀ_ਸੂਚਕ, n_ਸਿਰ, sess->n_ਧਾਗੇ);

    kparams->u.hvx.ਮਾਪ_q_ਪੰਕਤੀ_padded = ਮਾਪ_q_ਪੰਕਤੀ_padded;
    kparams->u.hvx.ਮਾਪ_k_ਪੰਕਤੀ_padded = ਮਾਪ_k_ਪੰਕਤੀ_padded;
    kparams->u.hvx.ਮਾਪ_v_ਪੰਕਤੀ_padded = ਮਾਪ_v_ਪੰਕਤੀ_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_ਮੁੱਲਾਂ(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_ਮੁੱਲਾਂ(q->ne[1]);
    kparams->ਪ੍ਰਸਾਰਣ_rk2 = init_fastdiv_ਮੁੱਲਾਂ(q->ne[2]/k->ne[2]);
    kparams->ਪ੍ਰਸਾਰਣ_rk3 = init_fastdiv_ਮੁੱਲਾਂ(q->ne[3]/k->ne[3]);
    kparams->ਪ੍ਰਸਾਰਣ_rv2 = init_fastdiv_ਮੁੱਲਾਂ(q->ne[2]/v->ne[2]);
    kparams->ਪ੍ਰਸਾਰਣ_rv3 = init_fastdiv_ਮੁੱਲਾਂ(q->ne[3]/v->ne[3]);
    ਜੇ (ਨਕਾਬ) {
        kparams->src3_div2 = init_fastdiv_ਮੁੱਲਾਂ(ਨਕਾਬ->ne[2]);
        kparams->src3_div3 = init_fastdiv_ਮੁੱਲਾਂ(ਨਕਾਬ->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_ਪ੍ਰਤੀ_thread = (kparams->qrows + sess->n_ਧਾਗੇ - 1) / sess->n_ਧਾਗੇ;

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_flash_attn_ext(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src2 = op->src[2];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src3 = op->src[3];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src4 = op->src[4];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    ਜੇ ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src3 && src3->type != GGML_TYPE_F16) {  //ਨਕਾਬ
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        ਵਾਪਸ ਝੂਠ;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    ਜੇ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਬਣਤਰ htp_fa_kernel_params kparams;
    ਜੇ (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ ((size_t) kparams.vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_ਮਾਪ, sess->vtcm_ਮਾਪ);
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_gated_delta_net(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਜੇ (opt_gdn_select < 1) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਬਣਤਰ ggml_tensor * q     = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * k     = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * v     = op->src[2];
    ਅਚਲ ਬਣਤਰ ggml_tensor * g     = op->src[3];
    ਅਚਲ ਬਣਤਰ ggml_tensor * beta  = op->src[4];
    ਅਚਲ ਬਣਤਰ ggml_tensor * state = op->src[5];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst   = op;

    ਜੇ (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous_ਪੰਕਤੀਆਂ(q) || !ggml_is_contiguous_ਪੰਕਤੀਆਂ(k) || !ggml_is_contiguous_ਪੰਕਤੀਆਂ(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int64_t S_v      = v->ne[0];
    ਅਚਲ int64_t H        = v->ne[1];
    ਅਚਲ int64_t n_ਟੋਕਨ = v->ne[2];
    ਅਚਲ int64_t n_seqs   = v->ne[3];
    ਅਚਲ int64_t K        = ggml_get_op_params_i32(op, 0);

    ਜੇ (S_v <= 0 || S_v > 128 || H <= 0 || n_ਟੋਕਨ <= 0 || n_seqs <= 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_ਟੋਕਨ || k->ne[2] != n_ਟੋਕਨ || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        ਵਾਪਸ ਝੂਠ;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    ਜੇ (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (dst->ne[0] != S_v * H || dst->ne[1] != n_ਟੋਕਨ * n_seqs + S_v * n_seqs * K) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = (uint32_t) (H * n_seqs);
    ਅਚਲ uint32_t n_ਧਾਗੇ  = (std::min)((uint32_t) sess->n_ਧਾਗੇ, ਕੁੱਲ_ਪੰਕਤੀਆਂ);

    ਅਚਲ ਤਾਰਕਿਕ can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_ਟੋਕਨ >= HTP_GDN_MIN_ਟੋਕਨ) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    ਜੇ (can_use_hmx) {
        ਬਣਤਰ htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        ਜੇ (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_ਮਾਪ, ਕੁੱਲ_ਪੰਕਤੀਆਂ, sess->vtcm_ਮਾਪ, n_ਧਾਗੇ, ਸੱਚ, &n_heads_batch)) {
            ਵਾਪਸ ਝੂਠ;
        }
    } ਨਹੀਂਤਾਂ {
        ਬਣਤਰ htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_ਧਾਗੇ);
        ਜੇ (layout.ਕੁੱਲ_ਬਾਈਟਾਂ > sess->vtcm_ਮਾਪ) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_matmul_is_hmx_eligible(
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਤਾਰਕਿਕ is_matmul_id,
    ਤਾਰਕਿਕ is_batched
) {
    GGML_ਅਣਵਰਤਿਆ(dst);

    ਜੇ (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਪੂਰਨ ne00  = src0->ne[0];
    ਅਚਲ ਪੂਰਨ ne01  = src0->ne[1];
    ਅਚਲ ਪੂਰਨ ne11  = src1->ne[1];
    ਅਚਲ ਪੂਰਨ ne12  = src1->ne[2];
    ਅਚਲ ਪੂਰਨ wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    ਜੇ (ne01 % HTP_TENSOR_TILE_N_ਪੰਕਤੀਆਂ != 0 && is_matmul_id) {
        ਵਾਪਸ ਝੂਠ;
    }

    // HMX supports F16, F32, and repack quantized types.
    ਜੇ (!htp_is_hmx_ਭਾਰ_type((ggml_type) wtype)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    ਜੇ (ne00 % HTP_TENSOR_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ != 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    ਅਚਲ ਤਾਰਕਿਕ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ਜੇ (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!is_matmul_id && is_batched && is_tiled) {
        ਵਾਪਸ ਝੂਠ;
    }

    // HMX assumes contiguous row-major layout.
    ਜੇ (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        ਵਾਪਸ ਝੂਠ;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    ਅਚਲ ਪੂਰਨ m = is_matmul_id ? ne12 : ne11;
    ਜੇ (m <= HTP_MM_HMX_MIN_NROWS) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(dst);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_hmx_mm_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * act,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਪੂਰਨ wtype,
    ਪੂਰਨ ne00_padded,
    ਪੂਰਨ ne01_padded,
    ਪੂਰਨ ne02,
    ਪੂਰਨ ne11,
    ਪੂਰਨ ne12,
    ਪੂਰਨ ne11_padded,
    ਤਾਰਕਿਕ is_matmul_id,
    ਤਾਰਕਿਕ is_batched,
    size_t src2_ਮਾਪ,
    size_t vtcm_budget,
    ਬਣਤਰ htp_mm_kernel_params * kparams
) {
    ਅਚਲ ਤਾਰਕਿਕ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ਅਚਲ ਪੂਰਨ aligned_tile_ਮਾਪ = is_tiled ? htp_tensor_get_tile_padded_ਮਾਪ(wtype) : 0;
    ਅਚਲ ਪੂਰਨ n_ਧਾਗੇ = (ਪੂਰਨ)sess->n_ਧਾਗੇ;
    ਅਚਲ ਪੂਰਨ ne10 = act->ne[0];

    ਪੂਰਨ m_for_solver = ne11;
    ਪੂਰਨ m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    ਜੇ (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        ਅਚਲ ਤਾਰਕਿਕ dst_ਪੰਕਤੀ_ਵੰਡੋ = dst ? ggml_hexagon_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst) : ਸੱਚ;
        ਅਚਲ ਤਾਰਕਿਕ act_ਪੰਕਤੀ_ਵੰਡੋ = ggml_hexagon_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(act);
        ਜੇ (dst_ਪੰਕਤੀ_ਵੰਡੋ && act_ਪੰਕਤੀ_ਵੰਡੋ) {
            m_for_solver = (ne11 + (ਪੂਰਨ) sess->mdev.count - 1) / (ਪੂਰਨ) sess->mdev.count;
            m_for_solver_padded = hex_ਗੇੜਾ_ਉੱਪਰ(std::max(m_for_solver, 32), 32);
        }
    }
    ਅਚਲ ਤਾਰਕਿਕ pipeline = is_matmul_id ? ਝੂਠ : htp_mm_hmx_pipeline(m_for_solver);

    ਅਚਲ ਤਾਰਕਿਕ is_batched_val = is_matmul_id ? ਝੂਠ : is_batched;
    ਅਚਲ ਪੂਰਨ ਸਮੂਹ_ਮਾਪ = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_ਮਾਪ = 0;
    ਤਾਰਕਿਕ use_grouped = ਝੂਠ;
    ਪੂਰਨ act_ਧਾਗੇ_selected = 0;

    ਜੇ (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && ਸਮੂਹ_ਮਾਪ > 1) {
        // Try grouped path first
        ਜੇ (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, ਸਮੂਹ_ਮਾਪ, n_ਧਾਗੇ, pipeline, src2_ਮਾਪ, vtcm_budget, &m_chunk, &n_chunk, &act_ਧਾਗੇ_selected, &vtcm_ਮਾਪ)) {
            use_grouped = ਸੱਚ;
        }
    }

    ਜੇ (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        ਪੂਰਨ m_id_ਪੰਕਤੀਆਂ = 0;
        ਜੇ (dst && is_matmul_id) {
            ਅਚਲ ਪੂਰਨ n_experts = ne02 > 0 ? ne02 : 1;
            ਅਚਲ size_t ਕੁੱਲ_expert_ਪੰਕਤੀਆਂ = (size_t) dst->ne[1] * dst->ne[2];
            ਪੂਰਨ m_ਪ੍ਰਤੀ_expert = (ਪੂਰਨ) ((ਕੁੱਲ_expert_ਪੰਕਤੀਆਂ + n_experts - 1) / n_experts);
            ਜੇ (sess->mdev.count > 1 && ggml_hexagon_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst)) {
                m_ਪ੍ਰਤੀ_expert = (m_ਪ੍ਰਤੀ_expert + (ਪੂਰਨ) sess->mdev.count - 1) / (ਪੂਰਨ) sess->mdev.count;
            }
            m_id_ਪੰਕਤੀਆਂ = hex_ਗੇੜਾ_ਉੱਪਰ(std::max(m_ਪ੍ਰਤੀ_expert, 32), 32);
        }
        ਅਚਲ uint32_t ਲਾਗਤ_m = is_matmul_id ? (uint32_t) m_id_ਪੰਕਤੀਆਂ : (uint32_t) m_for_solver;
        ਜੇ (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_ਪੰਕਤੀਆਂ, ne01_padded, m_for_solver_padded, ਲਾਗਤ_m, n_ਧਾਗੇ, pipeline, is_matmul_id, aligned_tile_ਮਾਪ, src2_ਮਾਪ, vtcm_budget, &m_chunk, &n_chunk, &act_ਧਾਗੇ_selected, &vtcm_ਮਾਪ)) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    kparams->n_hmx = 1;
    kparams->pipeline = pipeline ? 1 : 0;
    kparams->m_chunk = m_chunk;
    kparams->n_chunk = n_chunk;
    kparams->n_ਧਾਗੇ = n_ਧਾਗੇ;
    kparams->n_act_ਧਾਗੇ = act_ਧਾਗੇ_selected;
    kparams->tile_ਮਾਪ = is_tiled ? htp_tensor_get_tile_ਮਾਪ(wtype) : 0;
    kparams->aligned_tile_ਮਾਪ = aligned_tile_ਮਾਪ;
    kparams->act_ਪੰਕਤੀ_ਮਾਪ = htp_mm_tiled_act_ਪੰਕਤੀ_ਮਾਪ(wtype, ne10);
    kparams->vtcm_ਮਾਪ = vtcm_ਮਾਪ;
    kparams->vtcm_src0_ਮਾਪ = 0;
    kparams->div_n_act_ਧਾਗੇ = init_fastdiv_ਮੁੱਲਾਂ(act_ਧਾਗੇ_selected);
    kparams->div_ne00_padded   = init_fastdiv_ਮੁੱਲਾਂ(ne00_padded);
    kparams->vtcm_act_ਮਾਪ     = 0;
    kparams->vtcm_bias_ਮਾਪ    = (int32_t) src2_ਮਾਪ;
    kparams->vtcm_dst_ਮਾਪ     = 0;

    ਜੇ (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } ਨਹੀਂਤਾਂ {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_hvx_mm_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਪੂਰਨ wtype,
    ਤਾਰਕਿਕ is_tiled,
    ਪੂਰਨ ne02,
    ਪੂਰਨ ne03,
    ਪੂਰਨ ne10,
    ਪੂਰਨ ne11,
    ਪੂਰਨ ne12,
    ਪੂਰਨ ne13,
    ਤਾਰਕਿਕ is_matmul_id,
    ਅਚਲ size_t src2_ਪੰਕਤੀ_ਮਾਪ,
    size_t vtcm_budget,
    ਬਣਤਰ htp_mm_kernel_params * kparams
) {
    ਜੇ (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ਵਾਪਸ;
    }

    kparams->n_hmx = 0;
    kparams->n_ਧਾਗੇ = sess->n_ਧਾਗੇ;

    GGML_ਅਣਵਰਤਿਆ(ne02);
    GGML_ਅਣਵਰਤਿਆ(ne03);

    ਅਚਲ ਪੂਰਨ src1_nrows = ne11 * ne12 * ne13;

    ਜੇ (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_ਮਾਪ = htp_tensor_get_tile_ਮਾਪ(wtype);
        kparams->aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(wtype);

        ਅਚਲ ਤਾਰਕਿਕ k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        ਜੇ (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (ਪੂਰਨ) sess->n_ਧਾਗੇ)
                                  ? HTP_MM_KERNEL_HVX_QUANT_ਖੰਡ : HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ;
            kparams->act_ਪੰਕਤੀ_ਮਾਪ = htp_mm_tiled_act_ਪੰਕਤੀ_ਮਾਪ(wtype, ne10);

            ਬਣਤਰ htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t ਉੱਤਮ_n_prefetch = 2;
            ਲਈ (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_ਧਾਗੇ,
                    0, src0->nb[1], kparams->act_ਪੰਕਤੀ_ਮਾਪ, 0, d, ਸੱਚ, ਝੂਠ
                );
                ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ <= vtcm_budget) {
                    ਉੱਤਮ_n_prefetch = d;
                    ਰੋਕੋ;
                }
            }
            ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                ਵਾਪਸ;
            }
            kparams->n_prefetch     = ਉੱਤਮ_n_prefetch;
            kparams->vtcm_ਮਾਪ      = L.ਕੁੱਲ_ਬਾਈਟਾਂ;
            kparams->vtcm_src0_ਮਾਪ = L.src0_ਬਾਈਟਾਂ;
            kparams->vtcm_act_ਮਾਪ  = L.act_ਬਾਈਟਾਂ;
            kparams->vtcm_bias_ਮਾਪ = 0;
            kparams->vtcm_dst_ਮਾਪ  = L.dst_ਬਾਈਟਾਂ;
            ਵਾਪਸ;
        } ਨਹੀਂਤਾਂ {
            ਤਾਰਕਿਕ try_tiled = (k_align && opt_mm_select >= 1);
            ਜੇ (try_tiled) {
                kparams->act_ਪੰਕਤੀ_ਮਾਪ = htp_mm_tiled_act_ਪੰਕਤੀ_ਮਾਪ(wtype, ne10);
                ਜੇ (src1_nrows < (ਪੂਰਨ) sess->n_ਧਾਗੇ) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ਖੰਡ;
                } ਨਹੀਂਤਾਂ {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ;
                }

                ਬਣਤਰ htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t ਉੱਤਮ_n_prefetch = 2;
                ਲਈ (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_ਧਾਗੇ,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_ਪੰਕਤੀ_ਮਾਪ, d, ਝੂਠ, ਝੂਠ
                    );
                    ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ <= vtcm_budget) {
                        ਉੱਤਮ_n_prefetch = d;
                        ਰੋਕੋ;
                    }
                }

                uint32_t m_chunk = 0;
                ਜੇ (htp_mm_hvx_solve_vtcm_params(
                        kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_ਧਾਗੇ,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_ਪੰਕਤੀ_ਮਾਪ, ਉੱਤਮ_n_prefetch, vtcm_budget,
                        &L, &m_chunk)) {
                    kparams->n_prefetch     = ਉੱਤਮ_n_prefetch;
                    kparams->m_chunk        = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
                    kparams->vtcm_ਮਾਪ      = L.ਕੁੱਲ_ਬਾਈਟਾਂ;
                    kparams->vtcm_src0_ਮਾਪ = L.src0_ਬਾਈਟਾਂ;
                    kparams->vtcm_act_ਮਾਪ  = L.act_ਬਾਈਟਾਂ;
                    kparams->vtcm_bias_ਮਾਪ = L.bias_ਬਾਈਟਾਂ;
                    kparams->vtcm_dst_ਮਾਪ  = L.dst_ਬਾਈਟਾਂ;
                    ਵਾਪਸ;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            ਵਾਪਸ;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ਵਾਪਸ;
    } ਨਹੀਂਤਾਂ ਜੇ (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        ਬਣਤਰ htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ਜੇ (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, sess->n_ਧਾਗੇ,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_ਪੰਕਤੀ_ਮਾਪ, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_ਪੰਕਤੀ_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * 2, 128);
            kparams->vtcm_ਮਾਪ = L.ਕੁੱਲ_ਬਾਈਟਾਂ;
            kparams->vtcm_src0_ਮਾਪ = L.src0_ਬਾਈਟਾਂ;
            kparams->vtcm_act_ਮਾਪ  = L.act_ਬਾਈਟਾਂ;
            kparams->vtcm_bias_ਮਾਪ = L.bias_ਬਾਈਟਾਂ;
            kparams->vtcm_dst_ਮਾਪ = L.dst_ਬਾਈਟਾਂ;
            kparams->n_prefetch = 16;
            ਵਾਪਸ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ਵਾਪਸ;
    } ਨਹੀਂਤਾਂ ਜੇ (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        ਬਣਤਰ htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ਜੇ (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, sess->n_ਧਾਗੇ,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_ਪੰਕਤੀ_ਮਾਪ, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_ਪੰਕਤੀ_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * 4, 128);
            kparams->vtcm_ਮਾਪ = L.ਕੁੱਲ_ਬਾਈਟਾਂ;
            kparams->vtcm_src0_ਮਾਪ = L.src0_ਬਾਈਟਾਂ;
            kparams->vtcm_act_ਮਾਪ  = L.act_ਬਾਈਟਾਂ;
            kparams->vtcm_bias_ਮਾਪ = L.bias_ਬਾਈਟਾਂ;
            kparams->vtcm_dst_ਮਾਪ = L.dst_ਬਾਈਟਾਂ;
            kparams->n_prefetch = 16;
            ਵਾਪਸ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ਵਾਪਸ;
    } ਨਹੀਂਤਾਂ ਜੇ (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        ਬਣਤਰ htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ਜੇ (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_BF16_F32_VTCM, wtype, ne10, src1_nrows, sess->n_ਧਾਗੇ,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_ਪੰਕਤੀ_ਮਾਪ, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_BF16_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_ਪੰਕਤੀ_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * 4, 128);
            kparams->vtcm_ਮਾਪ = L.ਕੁੱਲ_ਬਾਈਟਾਂ;
            kparams->vtcm_src0_ਮਾਪ = L.src0_ਬਾਈਟਾਂ;
            kparams->vtcm_act_ਮਾਪ  = L.act_ਬਾਈਟਾਂ;
            kparams->vtcm_bias_ਮਾਪ = L.bias_ਬਾਈਟਾਂ;
            kparams->vtcm_dst_ਮਾਪ = L.dst_ਬਾਈਟਾਂ;
            kparams->n_prefetch = 16;
            ਵਾਪਸ;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ਵਾਪਸ;
    } ਨਹੀਂਤਾਂ {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ਵਾਪਸ;
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_matmul_params_impl(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਅਚਲ size_t src2_ਪੰਕਤੀ_ਮਾਪ,
    ਅਚਲ size_t src2_ਮਾਪ,
    ਬਣਤਰ htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));

    ਅਚਲ ਪੂਰਨ ne00 = src0->ne[0];
    ਅਚਲ ਪੂਰਨ ne01 = src0->ne[1];
    ਅਚਲ ਪੂਰਨ ne02 = src0->ne[2];
    ਅਚਲ ਪੂਰਨ ne03 = src0->ne[3];

    ਅਚਲ ਪੂਰਨ ne10 = src1->ne[0];
    ਅਚਲ ਪੂਰਨ ne11 = src1->ne[1];
    ਅਚਲ ਪੂਰਨ ne12 = src1->ne[2];
    ਅਚਲ ਪੂਰਨ ne13 = src1->ne[3];

    ਅਚਲ ਤਾਰਕਿਕ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ਅਚਲ ਪੂਰਨ wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ਅਚਲ ਪੂਰਨ ne00_padded = is_tiled ? hex_ਗੇੜਾ_ਉੱਪਰ(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    ਅਚਲ ਪੂਰਨ ne11_padded = hex_ਗੇੜਾ_ਉੱਪਰ(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    ਅਚਲ ਪੂਰਨ ne01_tiled  = hex_ਗੇੜਾ_ਉੱਪਰ(ne01, 32);

    ਅਚਲ ਤਾਰਕਿਕ is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    ਅਚਲ ਤਾਰਕਿਕ is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ਅਚਲ size_t vtcm_budget = sess->vtcm_ਮਾਪ;

    // Check HMX eligibility and try precomputing HMX parameters
    ਤਾਰਕਿਕ hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    ਜੇ (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        ਜੇ (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_ਮਾਪ, vtcm_budget, kparams)) {
            ਜਾਓ finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_ਪੰਕਤੀ_ਮਾਪ, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_ਮੁੱਲਾਂ(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_ਮੁੱਲਾਂ(ne11);
    kparams->div_r2       = init_fastdiv_ਮੁੱਲਾਂ(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_ਮੁੱਲਾਂ(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_ਮੁੱਲਾਂ(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_ਪੰਕਤੀਆਂ_stride(ਅਚਲ int64_t * ne, ਅਚਲ size_t * nb, size_t * stride) {
    size_t s = 0, ਅਗਲਾ = 0;
    ਲਈ (ਪੂਰਨ i = 1; i < GGML_MAX_DIMS; i++) {
        ਜੇ (ne[i] == 1) ਜਾਰੀ;
        ਜੇ (s == 0) { s = nb[i]; ਅਗਲਾ = s * ne[i]; ਜਾਰੀ; }
        ਜੇ (nb[i] != ਅਗਲਾ) ਵਾਪਸ ਝੂਠ;
        ਅਗਲਾ *= ne[i];
    }
    *stride = s ? s : nb[1];
    ਵਾਪਸ ਸੱਚ;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_matmul_can_collapse(ਅਚਲ ਬਣਤਰ ggml_tensor * src0, ਅਚਲ ਬਣਤਰ ggml_tensor * src1, ਅਚਲ ਬਣਤਰ ggml_tensor * dst) {
    size_t s1, sd;
    ਵਾਪਸ (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ਜੋੜੋ) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_ਮਾਪ(src1->type) && ggml_hexagon_ਪੰਕਤੀਆਂ_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_ਮਾਪ(dst->type) && ggml_hexagon_ਪੰਕਤੀਆਂ_stride(dst->ne, dst->nb, &sd);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_matmul_ਜੋੜੋ_can_collapse(
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src2,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst
) {
    ਜੇ (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!src2) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਅਚਲ int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    ਜੇ (src2_nrows == 1) {
        ਵਾਪਸ ਸੱਚ;
    }
    size_t s2;
    ਵਾਪਸ src2->nb[0] == ggml_type_ਮਾਪ(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_ਪੰਕਤੀਆਂ_stride(src2->ne, src2->nb, &s2);
}

ਸਥਿਰ ggml_tensor ggml_hexagon_tensor_collapse_ਪੰਕਤੀਆਂ(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_ਪੰਕਤੀਆਂ_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    ਵਾਪਸ c;
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_matmul_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_mm_kernel_params * kparams
) {
    ਜੇ (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ਅਚਲ ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_ਪੰਕਤੀਆਂ(src1);
        ਅਚਲ ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_ਪੰਕਤੀਆਂ(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        ਵਾਪਸ;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_fused_matmul_ਜੋੜੋ_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src2,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_mm_kernel_params * kparams
) {
    ਜੇ (ggml_hexagon_matmul_ਜੋੜੋ_can_collapse(src0, src1, src2, dst)) {
        ਅਚਲ ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_ਪੰਕਤੀਆਂ(src1);
        ਅਚਲ ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_ਪੰਕਤੀਆਂ(dst);
        ਅਚਲ ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_ਪੰਕਤੀਆਂ(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        ਅਚਲ ਬਣਤਰ ggml_tensor * p_src2 = src2 ? &src2_collapsed : ਖਾਲੀ_ਸੂਚਕ;
        ਅਚਲ size_t src2_ਮਾਪ = p_src2 ? hex_ਗੇੜਾ_ਉੱਪਰ(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_ਮਾਪ, kparams);
        kparams->collapse = 1;
        ਵਾਪਸ;
    }
    ਅਚਲ size_t src2_ਮਾਪ = src2 ? hex_ਗੇੜਾ_ਉੱਪਰ(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_ਮਾਪ, kparams);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_ਦੁਆਧਾਰੀ_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    uint32_t op,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_kernel_params * kparams,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src2
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));

    ਅਚਲ size_t elem_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = src0->ne[0] * elem_ਮਾਪ;
    ਅਚਲ size_t src1_ਪੰਕਤੀ_ਮਾਪ = src1->ne[0] * elem_ਮਾਪ;
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ  = dst->ne[0]  * elem_ਮਾਪ;

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src0_ਪੰਕਤੀ_ਮਾਪ, 128);
    ਅਚਲ size_t src1_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src1_ਪੰਕਤੀ_ਮਾਪ, 128);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = hex_ਗੇੜਾ_ਉੱਪਰ(dst_ਪੰਕਤੀ_ਮਾਪ,  128);

    ਜੇ (!htp_ਦੁਆਧਾਰੀ_strides_ਸੁਰੱਖਿਅਤ(src0, src1, dst, elem_ਮਾਪ)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਤਾਰਕਿਕ is_ਜੋੜੋ_id = op == HTP_OP_ਜੋੜੋ_ID;
    ਅਚਲ ਤਾਰਕਿਕ is_scalar = !is_ਜੋੜੋ_id && src1->ne[0] == 1;
    ਅਚਲ ਤਾਰਕਿਕ is_ਪੰਕਤੀ_bcast = !is_ਜੋੜੋ_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    ਅਚਲ ਤਾਰਕਿਕ is_same_shape = !is_ਜੋੜੋ_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    ਅਚਲ ਤਾਰਕਿਕ is_complex   = !is_ਜੋੜੋ_id && !is_scalar && !is_same_shape && !is_ਪੰਕਤੀ_bcast && (src1->ne[0] == src0->ne[0]);
    ਅਚਲ ਤਾਰਕਿਕ is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    ਅਚਲ ਤਾਰਕਿਕ is_scalar_ਪ੍ਰਸਾਰਣ = !is_ਜੋੜੋ_id && (ggml_nelements(src1) == 1);

    ਜੇ (!is_ਜੋੜੋ_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_ਪ੍ਰਸਾਰਣ)) {
        ਅਚਲ uint32_t ਕੁੱਲ_elems = (uint32_t) ggml_nelements(src0);
        ਅਚਲ uint32_t n_ਧਾਗੇ = sess->n_ਧਾਗੇ;
        ਅਚਲ uint32_t max_chunk_elems = 32768 / elem_ਮਾਪ;
        ਅਚਲ uint32_t min_chunk_elems = 256;
        ਅਚਲ uint32_t ਨਿਸ਼ਾਨਾ_chunk_elems = hex_ਗੇੜਾ_ਉੱਪਰ((ਕੁੱਲ_elems + (2 * n_ਧਾਗੇ) - 1) / (2 * n_ਧਾਗੇ), 32);
        ਅਚਲ uint32_t chunk_ਮਾਪ = (std::min)(max_chunk_elems, (std::max)(ਨਿਸ਼ਾਨਾ_chunk_elems, min_chunk_elems));
        ਅਚਲ uint32_t chunk_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ(chunk_ਮਾਪ * elem_ਮਾਪ, 128);

        kparams->kernel_type           = HTP_ਦੁਆਧਾਰੀ_KERNEL_CHUNKED;
        kparams->n_ਧਾਗੇ             = n_ਧਾਗੇ;
        kparams->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ       = 1;
        kparams->src0_ਪੰਕਤੀ_ਮਾਪ_aligned = src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
        kparams->src1_ਪੰਕਤੀ_ਮਾਪ_aligned = is_scalar_ਪ੍ਰਸਾਰਣ ? 0 : src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
        kparams->dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned;
        kparams->src1_ਮਾਪ             = 0;
        kparams->chunk_ਮਾਪ            = chunk_ਮਾਪ;
        kparams->chunk_ਬਾਈਟਾਂ           = chunk_ਬਾਈਟਾਂ;
        kparams->is_scalar             = is_scalar_ਪ੍ਰਸਾਰਣ ? 1 : 0;

        ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout L;
        htp_ਦੁਆਧਾਰੀ_vtcm_layout_build(&L, kparams, sess->vtcm_ਮਾਪ);
        ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ == 0 || L.ਕੁੱਲ_ਬਾਈਟਾਂ > sess->vtcm_ਮਾਪ) {
            ਵਾਪਸ ਝੂਠ;
        }

        kparams->vtcm_ਮਾਪ = L.ਕੁੱਲ_ਬਾਈਟਾਂ;
        ਵਾਪਸ ਸੱਚ;
    }

    ਗਿਣਤੀ htp_ਦੁਆਧਾਰੀ_kernel_type kernel_type;
    size_t src1_ਮਾਪ = 0;
    size_t src2_ਮਾਪ = 0;

    ਜੇ (is_ਜੋੜੋ_id) {
        GGML_ASSERT(src2 != ਖਾਲੀ_ਸੂਚਕ);
        kernel_type = HTP_ਦੁਆਧਾਰੀ_KERNEL_ਜੋੜੋ_ID;
        src1_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(src1->ne[1] * src1_ਪੰਕਤੀ_ਮਾਪ_aligned, 128);
        src2_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ggml_nelements(src2) * ਆਕਾਰ(int32_t), 128);
        ਜੇ (src1_ਮਾਪ + src2_ਮਾਪ > sess->vtcm_ਮਾਪ) {
            ਵਾਪਸ ਝੂਠ;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (is_ਪੰਕਤੀ_bcast) {
        kernel_type = HTP_ਦੁਆਧਾਰੀ_KERNEL_ਪੰਕਤੀ_BCAST;
        src1_ਮਾਪ = src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    } ਨਹੀਂਤਾਂ ਜੇ (is_scalar) {
        src1_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ggml_nelements(src1) * elem_ਮਾਪ, 128);
        ਜੇ (src1_ਮਾਪ > sess->vtcm_ਮਾਪ) {
            ਵਾਪਸ ਝੂਠ;
        }
        kernel_type = HTP_ਦੁਆਧਾਰੀ_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } ਨਹੀਂਤਾਂ ਜੇ (is_same_shape) {
        kernel_type = HTP_ਦੁਆਧਾਰੀ_KERNEL_SAME_SHAPE;
    } ਨਹੀਂਤਾਂ ਜੇ (is_complex) {
        src1_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ggml_nelements(src1) * elem_ਮਾਪ + 128, 128);
        ਜੇ (src1_ਮਾਪ > sess->vtcm_ਮਾਪ) {
            ਵਾਪਸ ਝੂਠ;
        }
        kernel_type = HTP_ਦੁਆਧਾਰੀ_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } ਨਹੀਂਤਾਂ {
        src1_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ggml_nelements(src1) * elem_ਮਾਪ + 128, 128);
        ਜੇ (src1_ਮਾਪ > sess->vtcm_ਮਾਪ) {
            ਵਾਪਸ ਝੂਠ;
        }
        kernel_type = HTP_ਦੁਆਧਾਰੀ_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_ਧਾਗੇ = sess->n_ਧਾਗੇ;
    kparams->src0_ਪੰਕਤੀ_ਮਾਪ_aligned = src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    kparams->src1_ਪੰਕਤੀ_ਮਾਪ_aligned = src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    kparams->dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned;
    kparams->src1_ਮਾਪ             = src1_ਮਾਪ;
    kparams->src2_ਮਾਪ             = src2_ਮਾਪ;

    ਅਚਲ ਤਾਰਕਿਕ src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    ਅਚਲ ਤਾਰਕਿਕ src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    ਅਚਲ ਤਾਰਕਿਕ dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    ਅਚਲ ਤਾਰਕਿਕ src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    ਅਚਲ ਤਾਰਕਿਕ src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    ਅਚਲ ਤਾਰਕਿਕ dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    ਤਾਰਕਿਕ ਵੰਡੋ_at_ne01 = ਝੂਠ;
    ਜੇ (src0->ne[2] > 1) {
        ਜੇ (kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_SAME_SHAPE) {
            ਵੰਡੋ_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } ਨਹੀਂਤਾਂ ਜੇ (kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_ਪੰਕਤੀ_BCAST) {
            ਵੰਡੋ_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } ਨਹੀਂਤਾਂ {
            ਵੰਡੋ_at_ne01 = ਸੱਚ;
        }
    }
    kparams->ਵੰਡੋ_at_ne01 = ਵੰਡੋ_at_ne01 ? 1 : 0;

    ਤਾਰਕਿਕ ਵੰਡੋ_at_ne02 = ਝੂਠ;
    ਜੇ (src0->ne[3] > 1) {
        ਜੇ (kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_SAME_SHAPE) {
            ਵੰਡੋ_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } ਨਹੀਂਤਾਂ ਜੇ (kernel_type == HTP_ਦੁਆਧਾਰੀ_KERNEL_ਪੰਕਤੀ_BCAST) {
            ਵੰਡੋ_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } ਨਹੀਂਤਾਂ {
            ਵੰਡੋ_at_ne02 = ਸੱਚ;
        }
    }
    kparams->ਵੰਡੋ_at_ne02 = ਵੰਡੋ_at_ne02 ? 1 : 0;

    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_vtcm_layout L;
    htp_ਦੁਆਧਾਰੀ_vtcm_layout_build(&L, kparams, sess->vtcm_ਮਾਪ);
    ਜੇ (L.ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ == 0 || L.ਕੁੱਲ_ਬਾਈਟਾਂ > sess->vtcm_ਮਾਪ) {
        ਜੇ (!is_ਜੋੜੋ_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_ਪ੍ਰਸਾਰਣ)) {
            ਅਚਲ uint32_t ਕੁੱਲ_elems = (uint32_t) ggml_nelements(src0);
            ਅਚਲ uint32_t n_ਧਾਗੇ = sess->n_ਧਾਗੇ;
            ਅਚਲ uint32_t max_chunk_elems = 32768 / elem_ਮਾਪ;
            ਅਚਲ uint32_t min_chunk_elems = 256;
            ਅਚਲ uint32_t ਨਿਸ਼ਾਨਾ_chunk_elems = hex_ਗੇੜਾ_ਉੱਪਰ((ਕੁੱਲ_elems + (2 * n_ਧਾਗੇ) - 1) / (2 * n_ਧਾਗੇ), 32);
            ਅਚਲ uint32_t chunk_ਮਾਪ = (std::min)(max_chunk_elems, (std::max)(ਨਿਸ਼ਾਨਾ_chunk_elems, min_chunk_elems));
            ਅਚਲ uint32_t chunk_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ(chunk_ਮਾਪ * elem_ਮਾਪ, 128);

            kparams->kernel_type           = HTP_ਦੁਆਧਾਰੀ_KERNEL_CHUNKED;
            kparams->n_ਧਾਗੇ             = n_ਧਾਗੇ;
            kparams->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ       = 1;
            kparams->src1_ਪੰਕਤੀ_ਮਾਪ_aligned = is_scalar_ਪ੍ਰਸਾਰਣ ? 0 : src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
            kparams->src1_ਮਾਪ             = 0;
            kparams->chunk_ਮਾਪ            = chunk_ਮਾਪ;
            kparams->chunk_ਬਾਈਟਾਂ           = chunk_ਬਾਈਟਾਂ;
            kparams->is_scalar             = is_scalar_ਪ੍ਰਸਾਰਣ ? 1 : 0;

            htp_ਦੁਆਧਾਰੀ_vtcm_layout_build(&L, kparams, sess->vtcm_ਮਾਪ);
            ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ == 0 || L.ਕੁੱਲ_ਬਾਈਟਾਂ > sess->vtcm_ਮਾਪ) {
                ਵਾਪਸ ਝੂਠ;
            }

            kparams->vtcm_ਮਾਪ = L.ਕੁੱਲ_ਬਾਈਟਾਂ;
            ਵਾਪਸ ਸੱਚ;
        }
        ਵਾਪਸ ਝੂਠ;
    }

    kparams->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ = L.ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਬਫਰ;
    kparams->vtcm_ਮਾਪ       = L.ਕੁੱਲ_ਬਾਈਟਾਂ;

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_unary_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    uint32_t op,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));

    ਅਚਲ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ uint32_t n_ਧਾਗੇ  = (std::min)((uint32_t)sess->n_ਧਾਗੇ, src0_nrows);

    kparams->n_ਧਾਗੇ = n_ਧਾਗੇ;

    ਅਚਲ size_t elem_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);

    ਅਚਲ size_t src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = src0->ne[0] * elem_ਮਾਪ;
    ਅਚਲ size_t dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ  = dst->ne[0]  * ggml_type_ਮਾਪ(dst->type);

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, 128);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = hex_ਗੇੜਾ_ਉੱਪਰ(dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ,  128);

    kparams->src0_ਪੰਕਤੀ_ਮਾਪ_aligned = src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    kparams->dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    size_t src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = 0;
    size_t src1_ਪੰਕਤੀ_ਮਾਪ_aligned = 0;
    ਤਾਰਕਿਕ ਪ੍ਰਸਾਰਣ_ਭਾਰ = ਝੂਠ;

    ਜੇ (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != ਖਾਲੀ_ਸੂਚਕ);
        src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = src1->ne[0] * ggml_type_ਮਾਪ(src1->type);
        src1_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, 128);
        ਪ੍ਰਸਾਰਣ_ਭਾਰ = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_ਪੰਕਤੀ_ਮਾਪ_aligned = src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    kparams->ਪ੍ਰਸਾਰਣ_ਭਾਰ      = ਪ੍ਰਸਾਰਣ_ਭਾਰ;

    ਬਣਤਰ htp_unary_vtcm_layout L;
    uint32_t ਖਾਨਾ_tile = 0;
    uint32_t vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread = 0;

    htp_unary_vtcm_layout_build(&L, op, src0->ne[0], dst->ne[0],
                                op == HTP_OP_RMS_NORM_MUL ? src1->ne[0] : 0,
                                ਪ੍ਰਸਾਰਣ_ਭਾਰ, n_ਧਾਗੇ, sess->vtcm_ਮਾਪ, elem_ਮਾਪ,
                                &ਖਾਨਾ_tile, &vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread);

    kparams->ਖਾਨਾ_tile = ਖਾਨਾ_tile;
    kparams->vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread = vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread;
    kparams->vtcm_ਮਾਪ = L.ਕੁੱਲ_ਬਾਈਟਾਂ;

    kparams->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread = L.src0_ਬਾਈਟਾਂ;
    kparams->vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread = L.src1_ਬਾਈਟਾਂ;
    kparams->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  = L.dst_ਬਾਈਟਾਂ;

    kparams->vtcm_src0_ਮਾਪ = L.src0_ਬਾਈਟਾਂ * n_ਧਾਗੇ;
    kparams->vtcm_src1_ਮਾਪ = L.src1_ਬਾਈਟਾਂ * n_ਧਾਗੇ;
    kparams->vtcm_dst_ਮਾਪ  = L.dst_ਬਾਈਟਾਂ * n_ਧਾਗੇ;

    kparams->ਖੰਡ = ਖਾਨਾ_tile ? 0 : ((L.src0_ਬਾਈਟਾਂ / 2) / src0_ਪੰਕਤੀ_ਮਾਪ_aligned);

    ਅਚਲ uint32_t tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ਖਾਨਾ_tile > 0 ? (src0->ne[0] + ਖਾਨਾ_tile - 1) / ਖਾਨਾ_tile : 1;
    kparams->div_ne01  = init_fastdiv_ਮੁੱਲਾਂ(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_ਮੁੱਲਾਂ(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_ਮੁੱਲਾਂ(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_ਮੁੱਲਾਂ(tiles_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_get_ਪੰਕਤੀਆਂ_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));

    ਅਚਲ uint32_t ne00 = src0->ne[0];
    ਅਚਲ uint32_t ne02 = src0->ne[2];
    ਅਚਲ uint32_t ne03 = src0->ne[3];

    ਅਚਲ uint32_t ne10 = src1->ne[0];
    ਅਚਲ uint32_t ne11 = src1->ne[1];
    ਅਚਲ uint32_t ne12 = src1->ne[2];
    ਅਚਲ uint32_t nr = ne10 * ne11 * ne12;

    ਅਚਲ ਤਾਰਕਿਕ tiled = ggml_hexagon_tensor_is_tiled(src0);
    ਅਚਲ ਪੂਰਨ wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    ਜੇ (tiled) {
        kparams->kernel_type = HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_TILED;
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_SAMETYPE;
    } ਨਹੀਂਤਾਂ {
        kparams->kernel_type = HTP_GET_ਪੰਕਤੀਆਂ_KERNEL_FLAT;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_tasks = nr;

    kparams->n_ਧਾਗੇ = (std::min)((uint32_t)sess->n_ਧਾਗੇ, ਕੁੱਲ_tasks);

    ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_vtcm_layout vtcm_layout = {};
    ਜਦੋਂ (kparams->n_ਧਾਗੇ > 0) {
        htp_get_ਪੰਕਤੀਆਂ_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, ਕੁੱਲ_tasks, src1->type, kparams->n_ਧਾਗੇ);
        ਜੇ (vtcm_layout.ਕੁੱਲ_ਬਾਈਟਾਂ <= sess->vtcm_ਮਾਪ) {
            ਰੋਕੋ;
        }
        --kparams->n_ਧਾਗੇ;
    }

    ਜੇ (kparams->n_ਧਾਗੇ == 0 && ਕੁੱਲ_tasks > 0) {
        htp_get_ਪੰਕਤੀਆਂ_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, ਕੁੱਲ_tasks, src1->type, 1);
    }

    kparams->vtcm_ਮਾਪ = (ਕੁੱਲ_tasks == 0) ? 0 : vtcm_layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
    kparams->tasks_ਪ੍ਰਤੀ_thread = kparams->n_ਧਾਗੇ > 0 ? (ਕੁੱਲ_tasks + kparams->n_ਧਾਗੇ - 1) / kparams->n_ਧਾਗੇ : 0;
    kparams->ਕੁੱਲ_tasks = ਕੁੱਲ_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_ਮੁੱਲਾਂ(ne10) : fastdiv_ਮੁੱਲਾਂ{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_ਮੁੱਲਾਂ(ne10 * ne11) : fastdiv_ਮੁੱਲਾਂ{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_ਮੁੱਲਾਂ(ne02) : fastdiv_ਮੁੱਲਾਂ{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_ਮੁੱਲਾਂ(ne03) : fastdiv_ਮੁੱਲਾਂ{0, 0};
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_ਸੈੱਟ_ਪੰਕਤੀਆਂ_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0, //ਮੁੱਲਾਂ
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1, // indices
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,  // destination
    ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));

    ਅਚਲ uint32_t nr = src0->ne[1];

    kparams->n_ਧਾਗੇ = (std::min)((uint32_t)sess->n_ਧਾਗੇ, nr);
    kparams->tasks_ਪ੍ਰਤੀ_thread = (nr + kparams->n_ਧਾਗੇ - 1) / kparams->n_ਧਾਗੇ;
    kparams->ਕੁੱਲ_tasks = nr;

    kparams->div_ne11 = init_fastdiv_ਮੁੱਲਾਂ(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_ਮੁੱਲਾਂ(src1->ne[2]);
    kparams->div_tasks_ਪ੍ਰਤੀ_thread = init_fastdiv_ਮੁੱਲਾਂ(kparams->tasks_ਪ੍ਰਤੀ_thread);
    kparams->div_ne02 = init_fastdiv_ਮੁੱਲਾਂ(src0->ne[2]);

    ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_vtcm_layout vtcm_layout;
    htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_ਧਾਗੇ);
    kparams->vtcm_ਮਾਪ = vtcm_layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_softmax_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];

    ਅਚਲ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ uint32_t n_ਧਾਗੇ  = (std::min)((uint32_t) sess->n_ਧਾਗੇ, src0_nrows);

    ਦਸ਼ਮਲਵ scale = 1.0f;
    ਦਸ਼ਮਲਵ max_bias = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,    &op->op_params[0], ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, &op->op_params[1], ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    ਅਚਲ uint32_t n_ਸਿਰ = src0->ne[2];
    ਅਚਲ uint32_t n_ਸਿਰ_log2 = 1u << (uint32_t) floor(log2(n_ਸਿਰ));
    kparams->n_ਸਿਰ      = n_ਸਿਰ;
    kparams->n_ਸਿਰ_log2 = n_ਸਿਰ_log2;

    ਜੇ (max_bias > 0.0f && n_ਸਿਰ_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_ਸਿਰ_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_ਸਿਰ_log2);
    } ਨਹੀਂਤਾਂ {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != ਖਾਲੀ_ਸੂਚਕ) ? 1 : 0;
    kparams->use_f16  = (src1 != ਖਾਲੀ_ਸੂਚਕ && src1->type == GGML_TYPE_F16) ? 1 : 0;

    ਅਚਲ uint32_t ne00 = src0->ne[0];
    ਅਚਲ uint32_t ne10 = src1 ? src1->ne[0] : 1;

    ਬਣਤਰ htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, ne00, ne10, kparams->use_src1 != 0, kparams->use_f16 != 0, n_ਧਾਗੇ);

    kparams->n_ਧਾਗੇ                 = n_ਧਾਗੇ;
    kparams->src0_nrows                = src0_nrows;
    kparams->src0_nrows_ਪ੍ਰਤੀ_thread     = (src0_nrows + n_ਧਾਗੇ - 1) / n_ਧਾਗੇ;
    kparams->vtcm_ਮਾਪ                 = (uint32_t) layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
    kparams->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread = (uint32_t) layout.src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    kparams->vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread = (uint32_t) layout.src1_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    kparams->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  = (uint32_t) layout.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    kparams->src0_ਪੰਕਤੀ_ਮਾਪ_aligned     = (uint32_t) layout.src0_spad_half_ਮਾਪ;
    kparams->src1_ਪੰਕਤੀ_ਮਾਪ_aligned     = (uint32_t) layout.src1_spad_half_ਮਾਪ;
    kparams->dst_ਪੰਕਤੀ_ਮਾਪ_aligned      = (uint32_t) layout.dst_spad_half_ਮਾਪ;
    kparams->src0_spad_half_ਮਾਪ       = (uint32_t) layout.src0_spad_half_ਮਾਪ;
    kparams->src1_spad_half_ਮਾਪ       = (uint32_t) layout.src1_spad_half_ਮਾਪ;
    kparams->dst_spad_half_ਮਾਪ        = (uint32_t) layout.dst_spad_half_ਮਾਪ;
    ਜੇ (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } ਨਹੀਂਤਾਂ ਜੇ (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_ਨਕਾਬ_F16;
    } ਨਹੀਂਤਾਂ {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_ਨਕਾਬ_F32;
    }

    ਜੇ (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_ਮੁੱਲਾਂ(src0->ne[1]);
    ਜੇ (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_ਮੁੱਲਾਂ(src0->ne[2]);
    ਅਚਲ uint32_t ne12 = src1 ? src1->ne[2] : 1;
    ਅਚਲ uint32_t ne13 = src1 ? src1->ne[3] : 1;
    ਜੇ (ne12 > 0) kparams->div_ne12 = init_fastdiv_ਮੁੱਲਾਂ(ne12);
    ਜੇ (ne13 > 0) kparams->div_ne13 = init_fastdiv_ਮੁੱਲਾਂ(ne13);
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_rope_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src2 = op->src[2];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਅਚਲ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ uint32_t n_ਧਾਗੇ  = (std::min)((uint32_t) sess->n_ਧਾਗੇ, src0_nrows);
    ਅਚਲ uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ਬਣਤਰ htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_ਧਾਗੇ, n_freq_factors);

    kparams->n_ਧਾਗੇ              = n_ਧਾਗੇ;
    kparams->src0_nrows             = src0_nrows;
    kparams->src0_nrows_ਪ੍ਰਤੀ_thread  = (src0_nrows + n_ਧਾਗੇ - 1) / n_ਧਾਗੇ;
    kparams->vtcm_ਮਾਪ              = (uint32_t) layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
    kparams->spad_ਪ੍ਰਤੀ_thread        = (uint32_t) layout.ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    kparams->theta_cache_ਵਿੱਥ     = (uint32_t) layout.theta_cache_ਮਾਪ_aligned;
    kparams->src0_ਪੰਕਤੀ_ਮਾਪ_aligned  = (uint32_t) layout.src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    kparams->freq_factors_ਵਿੱਥ    = (uint32_t) (layout.ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ);
    kparams->freq_factors_ਮਾਪ      = (uint32_t) layout.freq_factors_ਮਾਪ_aligned;

    ਜੇ (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_ਮੁੱਲਾਂ(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_ਮੁੱਲਾਂ(dst->ne[1]);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_ssm_conv_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));

    ਅਚਲ uint32_t d_conv  = (uint32_t) src1->ne[0];
    ਅਚਲ uint32_t d_inner = (uint32_t) src0->ne[1];
    ਅਚਲ uint32_t n_t     = (uint32_t) dst->ne[1];
    ਅਚਲ uint32_t n_s     = (uint32_t) dst->ne[2];
    ਅਚਲ uint32_t ncs     = (uint32_t) src0->ne[0];

    ਅਚਲ uint32_t n_ਧਾਗੇ = (std::min)((uint32_t) sess->n_ਧਾਗੇ, (d_inner + 31) / 32);

    kparams->n_ਧਾਗੇ = n_ਧਾਗੇ;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    ਅਚਲ uint32_t raw_rpt = (d_inner + n_ਧਾਗੇ - 1) / n_ਧਾਗੇ;
    ਅਚਲ uint32_t d_inner_ਪ੍ਰਤੀ_thread = hex_ਗੇੜਾ_ਉੱਪਰ(raw_rpt, 32);

    ਅਚਲ uint32_t src1_raw_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ(d_inner_ਪ੍ਰਤੀ_thread * d_conv * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128) + 128;
    ਅਚਲ uint32_t src1_T_ਬਾਈਟਾਂ   = hex_ਗੇੜਾ_ਉੱਪਰ(d_conv * d_inner_ਪ੍ਰਤੀ_thread * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
    ਅਚਲ uint32_t vtcm_src1_ਪ੍ਰਤੀ_thread = src1_raw_ਬਾਈਟਾਂ + src1_T_ਬਾਈਟਾਂ;

    uint32_t vtcm_src0_ਪ੍ਰਤੀ_thread = 0;
    uint32_t vtcm_dst_ਪ੍ਰਤੀ_thread  = 0;

    ਜੇ (n_t == 1) {
        kparams->d_inner_tile = d_inner_ਪ੍ਰਤੀ_thread;

        ਅਚਲ uint32_t src0_tile_raw_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ(d_inner_ਪ੍ਰਤੀ_thread * d_conv * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
        ਅਚਲ uint32_t src0_T_ਬਾਈਟਾਂ        = hex_ਗੇੜਾ_ਉੱਪਰ(d_conv * d_inner_ਪ੍ਰਤੀ_thread * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
        vtcm_src0_ਪ੍ਰਤੀ_thread = 2 * src0_tile_raw_ਬਾਈਟਾਂ + src0_T_ਬਾਈਟਾਂ;

        ਅਚਲ uint32_t dst_tile_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ(d_inner_ਪ੍ਰਤੀ_thread * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
        vtcm_dst_ਪ੍ਰਤੀ_thread = 2 * dst_tile_ਬਾਈਟਾਂ;
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ size_t vtcm_budget = (sess->vtcm_ਮਾਪ > 0 ? sess->vtcm_ਮਾਪ / n_ਧਾਗੇ : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        ਅਚਲ uint32_t src0_ਖੰਡ_T    = hex_ਗੇੜਾ_ਉੱਪਰ(ncs * 32 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
        ਅਚਲ size_t   fixed_ਬਾਈਟਾਂ     = vtcm_src1_ਪ੍ਰਤੀ_thread + src0_ਖੰਡ_T;
        ਅਚਲ size_t   avail_for_tiles = vtcm_budget > fixed_ਬਾਈਟਾਂ ? vtcm_budget - fixed_ਬਾਈਟਾਂ : (128 * 1024);

        uint32_t ਨਿਸ਼ਾਨਾ_max_tile = hex_ਗੇੜਾ_ਉੱਪਰ((d_inner_ਪ੍ਰਤੀ_thread + 3) / 4, 32);
        ਨਿਸ਼ਾਨਾ_max_tile = (std::max)(ਨਿਸ਼ਾਨਾ_max_tile, 32u);
        ਨਿਸ਼ਾਨਾ_max_tile = (std::min)(ਨਿਸ਼ਾਨਾ_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * ਆਕਾਰ(ਦਸ਼ਮਲਵ)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        ਜੇ (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        ਜੇ (d_inner_tile > ਨਿਸ਼ਾਨਾ_max_tile) {
            d_inner_tile = ਨਿਸ਼ਾਨਾ_max_tile;
        }
        ਜੇ (d_inner_tile > d_inner_ਪ੍ਰਤੀ_thread) {
            d_inner_tile = d_inner_ਪ੍ਰਤੀ_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        ਅਚਲ uint32_t src0_tile_raw = hex_ਗੇੜਾ_ਉੱਪਰ(d_inner_tile * ncs * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
        vtcm_src0_ਪ੍ਰਤੀ_thread = 2 * src0_tile_raw + src0_ਖੰਡ_T;

        vtcm_dst_ਪ੍ਰਤੀ_thread = 2 * hex_ਗੇੜਾ_ਉੱਪਰ(d_inner_tile * n_t * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
    }

    kparams->vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread = vtcm_src0_ਪ੍ਰਤੀ_thread;
    kparams->vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread = vtcm_src1_ਪ੍ਰਤੀ_thread;
    kparams->vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread  = vtcm_dst_ਪ੍ਰਤੀ_thread;

    kparams->vtcm_src0_ਮਾਪ = vtcm_src0_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
    kparams->vtcm_src1_ਮਾਪ = vtcm_src1_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
    kparams->vtcm_dst_ਮਾਪ  = vtcm_dst_ਪ੍ਰਤੀ_thread  * n_ਧਾਗੇ;
    kparams->vtcm_ਮਾਪ      = kparams->vtcm_src0_ਮਾਪ + kparams->vtcm_src1_ਮਾਪ + kparams->vtcm_dst_ਮਾਪ;
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_gated_delta_net_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));

    ਅਚਲ ਬਣਤਰ ggml_tensor * q     = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * k     = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * v     = op->src[2];
    ਅਚਲ ਬਣਤਰ ggml_tensor * g     = op->src[3];
    ਅਚਲ ਬਣਤਰ ggml_tensor * state = op->src[5];

    ਅਚਲ uint32_t S_v      = (uint32_t) v->ne[0];
    ਅਚਲ uint32_t H        = (uint32_t) v->ne[1];
    ਅਚਲ uint32_t n_ਟੋਕਨ = (uint32_t) v->ne[2];
    ਅਚਲ uint32_t n_seqs   = (uint32_t) v->ne[3];
    ਅਚਲ uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    ਅਚਲ uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    ਅਚਲ uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = H * n_seqs;
    ਅਚਲ uint32_t n_ਧਾਗੇ  = (std::min)((uint32_t) sess->n_ਧਾਗੇ, ਕੁੱਲ_ਪੰਕਤੀਆਂ);

    ਅਚਲ ਤਾਰਕਿਕ can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_ਟੋਕਨ >= HTP_GDN_MIN_ਟੋਕਨ) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    ਬਣਤਰ htp_gdn_hmx_vtcm_layout hmx_layout;
    ਬਣਤਰ htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    ਜੇ (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_ਮਾਪ, ਕੁੱਲ_ਪੰਕਤੀਆਂ, sess->vtcm_ਮਾਪ, n_ਧਾਗੇ, ਸੱਚ, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_ਮਾਪ      = HTP_GDN_CHUNK_ਮਾਪ;
        kparams->n_chunks        = (n_ਟੋਕਨ + HTP_GDN_CHUNK_ਮਾਪ - 1) / HTP_GDN_CHUNK_ਮਾਪ;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_ਮਾਪ       = (uint32_t) hmx_layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_ਬਾਈਟਾਂ;
        kparams->vtcm_ਪ੍ਰਤੀ_thread = (uint32_t) (hmx_layout.ਕੁੱਲ_ਬਾਈਟਾਂ / (n_ਧਾਗੇ > 0 ? n_ਧਾਗੇ : 1));
    } ਨਹੀਂਤਾਂ {
        htp_gdn_vtcm_layout_build(&hvx_layout, S_v, n_ਧਾਗੇ);
        kparams->kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
        kparams->pipeline        = 0;
        kparams->n_heads_batch   = 1;
        kparams->state_aligned   = (uint32_t) hvx_layout.state_aligned;
        kparams->vtcm_ਪ੍ਰਤੀ_thread = (uint32_t) hvx_layout.ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
        kparams->vtcm_ਮਾਪ       = (uint32_t) hvx_layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
    }

    kparams->n_ਧਾਗੇ           = n_ਧਾਗੇ;
    kparams->S_v                 = S_v;
    kparams->H                   = H;
    kparams->n_ਟੋਕਨ            = n_ਟੋਕਨ;
    kparams->n_seqs              = n_seqs;
    kparams->K                   = K;
    kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ          = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    kparams->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread     = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + kparams->n_ਧਾਗੇ - 1) / kparams->n_ਧਾਗੇ;
    kparams->kda                 = (g->ne[0] == S_v) ? 1 : 0;
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    kparams->state_ਮਾਪ_ਪ੍ਰਤੀ_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((ਦਸ਼ਮਲਵ) S_v);

    ਜੇ (H > 0)                  kparams->div_H         = init_fastdiv_ਮੁੱਲਾਂ(H);
    ਜੇ (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) q->ne[1]);
    ਜੇ (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) k->ne[1]);
    ਜੇ (rq3 > 0)                kparams->div_rq3       = init_fastdiv_ਮੁੱਲਾਂ(rq3);
    ਜੇ (rk3 > 0)                kparams->div_rk3       = init_fastdiv_ਮੁੱਲਾਂ(rk3);
    ਜੇ (kparams->n_ਧਾਗੇ > 0) kparams->div_n_ਧਾਗੇ = init_fastdiv_ਮੁੱਲਾਂ(kparams->n_ਧਾਗੇ);
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_ਛਾਂਟੋ_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਤਾਰਕਿਕ is_ਸਿਖਰ_k,
    ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ uint32_t ne00       = src0->ne[0];
    ਅਚਲ uint32_t k          = dst->ne[0];

    int32_t ਕ੍ਰਮ = GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC;
    ਜੇ (!is_ਸਿਖਰ_k) {
        ਕ੍ਰਮ = ((ਅਚਲ int32_t *) op->op_params)[0];
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ_max = sess->n_ਧਾਗੇ > 0 ? sess->n_ਧਾਗੇ : 4;
    ਅਚਲ size_t vtcm_budget = sess->vtcm_ਮਾਪ > 0 ? sess->vtcm_ਮਾਪ : (8 * 1024 * 1024);

    ਬਣਤਰ htp_ਛਾਂਟੋ_vtcm_layout layout;
    ਤਾਰਕਿਕ ok = htp_ਛਾਂਟੋ_solve_layout(&layout, ne00, ਕੁੱਲ_ਪੰਕਤੀਆਂ, k, n_ਧਾਗੇ_max, vtcm_budget, is_ਸਿਖਰ_k);
    GGML_ASSERT(ok);

    kparams->n_ਧਾਗੇ         = (int32_t) layout.n_ਧਾਗੇ;
    kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ        = (int32_t) ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    kparams->ਪੰਕਤੀ_ਸ਼ੁਰੂ         = 0;
    kparams->ਪੰਕਤੀ_ਸਿਰਾ           = (int32_t) ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->ਕ੍ਰਮ             = ਕ੍ਰਮ;
    kparams->is_ਸਿਖਰ_k          = is_ਸਿਖਰ_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_ਮਾਪ         = (int32_t) layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
    kparams->phase1_ਥਾਂ_ਮਾਪ  = (int32_t) layout.phase1_ਥਾਂ_ਮਾਪ;
    kparams->ਮਿਲਾਓ_ਮੁੱਲਾਂ_off  = (int32_t) layout.ਮਿਲਾਓ_ਮੁੱਲਾਂ_off;
    kparams->ਮਿਲਾਓ_indices_off = (int32_t) layout.ਮਿਲਾਓ_indices_off;
    kparams->ਮਿਲਾਓ_elems       = (int32_t) layout.ਮਿਲਾਓ_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    ਅਚਲ uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    ਅਚਲ int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    ਜੇ (numer_hi < 0) {
        *ox_hi = 0;
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    ਜੇ (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_pool_shape_ਸਮਾਉਂਦਾ(
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst
) {
    ਵਾਪਸ ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_pool_2d_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0,
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,
    ਬਣਤਰ htp_pool_2d_kernel_params * kparams,
    ਤਾਰਕਿਕ is_pool_1d
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));

    ਅਚਲ uint32_t src_x = (uint32_t) src0->ne[0];
    ਅਚਲ uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    ਅਚਲ uint32_t dst_x = (uint32_t) dst->ne[0];
    ਅਚਲ uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    ਅਚਲ uint32_t src_plane_ਬਾਈਟਾਂ = src_x * src_y * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t dst_plane_ਬਾਈਟਾਂ = dst_x * dst_y * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    ਅਚਲ uint32_t n_ਧਾਗੇ = (std::min)((uint32_t) sess->n_ਧਾਗੇ, planes > 0 ? planes : 1);

    ਬਣਤਰ htp_pool_vtcm_layout layout;
    ਅਚਲ ਤਾਰਕਿਕ ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_ਧਾਗੇ, sess->vtcm_ਮਾਪ);
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
    kparams->src_plane_ਬਾਈਟਾਂ = src_plane_ਬਾਈਟਾਂ;
    kparams->dst_plane_ਬਾਈਟਾਂ = dst_plane_ਬਾਈਟਾਂ;
    kparams->src_plane_ਬਾਈਟਾਂ_aligned = (uint32_t) layout.src_spad_half_ਮਾਪ;
    kparams->dst_plane_ਬਾਈਟਾਂ_aligned = (uint32_t) layout.dst_spad_half_ਮਾਪ;
    kparams->n_ਧਾਗੇ = n_ਧਾਗੇ;
    kparams->planes = planes;
    kparams->pool_op = (uint32_t) ggml_get_op_params_i32(dst, 0);
    // Fast HVX path requires exact tiling and a supported kernel width.
    kparams->fast_ਰਸਤਾ = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->stride_x == kparams->kernel_x && kparams->stride_y == kparams->kernel_y &&
                           (kparams->kernel_x == 1 || kparams->kernel_x == 2)) ? 1 : 0;
    kparams->global_ਰਸਤਾ = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                            kparams->stride_x == kparams->kernel_x &&
                            kparams->stride_y == kparams->kernel_y &&
                            kparams->kernel_x == kparams->src_x &&
                            kparams->kernel_y == kparams->src_y &&
                            kparams->dst_x == 1 && kparams->dst_y == 1) ? 1 : 0;
    kparams->ਖੰਡ_ਰਸਤਾ = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->kernel_y == 1 && kparams->stride_y == 1 &&
                           kparams->stride_x == kparams->kernel_x &&
                           kparams->kernel_x >= 32) ? 1 : 0;
    kparams->avg_divide_ਗਿਣਤੀ = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    ਅਚਲ ਤਾਰਕਿਕ narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_ਰਸਤਾ = (kparams->fast_ਰਸਤਾ && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (ਦਸ਼ਮਲਵ) (kparams->kernel_x * kparams->kernel_y);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_concat_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (!src0 || !src1 || !dst) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਪੂਰਨ dim = ((ਅਚਲ int32_t *) op->op_params)[0];
    ਜੇ (dim < 0 || dim >= GGML_MAX_DIMS) {
        ਵਾਪਸ ਝੂਠ;
    }
    kparams->dim = dim;

    ਜੇ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src0->type != dst->type || src1->type != dst->type) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    ਲਈ (ਪੂਰਨ d = 0; d < GGML_MAX_DIMS; d++) {
        ਅਚਲ int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        ਜੇ (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਅਚਲ uint32_t max_gather_ne0 = 128 / type_ਮਾਪ;
    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    ਅਚਲ ਤਾਰਕਿਕ b_dense = (src1->nb[0] == type_ਮਾਪ);
    ਅਚਲ ਤਾਰਕਿਕ gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_ਮਾਪ &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_ਮਾਪ
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_ਮਾਪ);

    ਜੇ (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_ਮਾਪ && dst->nb[1] == (size_t) dst->ne[0] * type_ਮਾਪ &&
        src0->nb[0] == type_ਮਾਪ &&
        (src1->nb[0] == type_ਮਾਪ || src1->nb[1] == type_ਮਾਪ) &&
        gather_strides_ok &&
        htp_gather_ਪੰਕਤੀਆਂ_tensor_outer_contiguous(dst) &&
        htp_gather_ਪੰਕਤੀਆਂ_tensor_outer_contiguous(src0) &&
        htp_gather_ਪੰਕਤੀਆਂ_tensor_outer_contiguous(src1)) {

        ਜੇ (htp_gather_ਪੰਕਤੀਆਂ_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_ਮਾਪ, ਕੁੱਲ_ਪੰਕਤੀਆਂ,
                                         sess->n_ਧਾਗੇ, sess->mdev.count, sess->vtcm_ਮਾਪ)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ਪੰਕਤੀਆਂ;
            kparams->n_ਧਾਗੇ   = (uint8_t) sess->n_ਧਾਗੇ;
            kparams->vtcm_ਮਾਪ   = kparams->u.gather.ਟੁਕੜਾ_2 * sess->n_ਧਾਗੇ;
            ਵਾਪਸ ਸੱਚ;
        }
    }

    ਅਚਲ ਤਾਰਕਿਕ dma_strides_ok = (src0->nb[0] == type_ਮਾਪ && src1->nb[0] == type_ਮਾਪ && dst->nb[0] == type_ਮਾਪ);

    ਜੇ (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_ਧਾਗੇ   = 1;
        ਵਾਪਸ ਸੱਚ;
    }

    ਅਚਲ ਤਾਰਕਿਕ is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    ਅਚਲ ਤਾਰਕਿਕ is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    ਅਚਲ ਤਾਰਕਿਕ transposed_ਪੰਕਤੀਆਂ_ok = (src0->nb[0] == type_ਮਾਪ && src1->nb[1] == type_ਮਾਪ && dst->nb[0] == type_ਮਾਪ);

    ਜੇ (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_ਪੰਕਤੀਆਂ_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        ਅਚਲ uint32_t n_ਧਾਗੇ = sess->n_ਧਾਗੇ > 0 ? (uint32_t) sess->n_ਧਾਗੇ : 8;
        ਬਣਤਰ htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_ਮਾਪ, n_ਧਾਗੇ);

        ਜੇ (sess->vtcm_ਮਾਪ > 0 && layout.ਕੁੱਲ_ਬਾਈਟਾਂ > sess->vtcm_ਮਾਪ) {
            ਵਾਪਸ ਝੂਠ;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_ਧਾਗੇ                         = n_ਧਾਗੇ;
        kparams->vtcm_ਮਾਪ                         = layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
        kparams->u.transposed.spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread = layout.src0_spad_ਮਾਪ_ਪ੍ਰਤੀ_thread;
        kparams->u.transposed.spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread = layout.src1_spad_ਮਾਪ_ਪ੍ਰਤੀ_thread;
        ਵਾਪਸ ਸੱਚ;
    }

    ਵਾਪਸ ਝੂਠ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_cpy_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (!src0 || !dst) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int64_t nelem_src = ggml_nelements(src0);
    ਅਚਲ int64_t nelem_dst = ggml_nelements(dst);
    ਜੇ (nelem_src != nelem_dst || nelem_src < 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t src_type_ਮਾਪ = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ uint32_t dst_type_ਮਾਪ = ggml_type_ਮਾਪ(dst->type);

    kparams->src0_type_ਮਾਪ = (uint8_t) src_type_ਮਾਪ;
    kparams->dst_type_ਮਾਪ  = (uint8_t) dst_type_ਮਾਪ;
    kparams->ਕੁੱਲ_elems    = (uint32_t) nelem_src;

    ਜੇ (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (nelem_src == 1) {
        ਜੇ (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            ਵਾਪਸ ਸੱਚ;
        }
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਤਾਰਕਿਕ sametype = (src0->type == dst->type);

    ਤਾਰਕਿਕ same_extents = ਸੱਚ;
    ਲਈ (ਪੂਰਨ d = 0; d < GGML_MAX_DIMS; d++) {
        ਜੇ (src0->ne[d] != dst->ne[d]) {
            same_extents = ਝੂਠ;
            ਰੋਕੋ;
        }
    }

    ਅਚਲ ਤਾਰਕਿਕ transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_ਮਾਪ) || (dst->nb[0] != dst_type_ਮਾਪ) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_ਮਾਪ) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_ਮਾਪ);
    ਅਚਲ ਤਾਰਕਿਕ sameshape  = same_extents && !transposed;

    ਅਚਲ ਤਾਰਕਿਕ src_is_contiguous = ggml_is_contiguous(src0);
    ਅਚਲ ਤਾਰਕਿਕ dst_is_contiguous = ggml_is_contiguous(dst);

    ਜੇ (sametype) {
        ਜੇ (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            ਵਾਪਸ ਸੱਚ;
        }

        ਅਚਲ uint32_t max_gather_ne0 = 128 / src_type_ਮਾਪ;
        ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        ਜੇ ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_ਮਾਪ && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_ਮਾਪ &&
            ਕੁੱਲ_ਪੰਕਤੀਆਂ > 1 &&
            htp_gather_ਪੰਕਤੀਆਂ_tensor_outer_contiguous(src0)) {

            ਜੇ (htp_gather_ਪੰਕਤੀਆਂ_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, ਝੂਠ,
                                             src_type_ਮਾਪ, ਕੁੱਲ_ਪੰਕਤੀਆਂ,
                                             sess->n_ਧਾਗੇ, sess->mdev.count, sess->vtcm_ਮਾਪ)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ਪੰਕਤੀਆਂ;
                kparams->n_ਧਾਗੇ   = (uint8_t) sess->n_ਧਾਗੇ;
                kparams->vtcm_ਮਾਪ   = kparams->u.gather.ਟੁਕੜਾ_2 * sess->n_ਧਾਗੇ;
                ਵਾਪਸ ਸੱਚ;
            }
        }

        ਜੇ (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            ਵਾਪਸ ਸੱਚ;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_ਧਾਗੇ   = sess->n_ਧਾਗੇ > 0 ? (uint8_t) sess->n_ਧਾਗੇ : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (!sameshape) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਤਾਰਕਿਕ valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    ਜੇ (!valid_conversion) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = sess->n_ਧਾਗੇ > 0 ? (uint32_t) sess->n_ਧਾਗੇ : 4;
    ਬਣਤਰ htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_ਮਾਪ, dst_type_ਮਾਪ, n_ਧਾਗੇ);

    ਜੇ (sess->vtcm_ਮਾਪ > 0 && layout.ਕੁੱਲ_ਬਾਈਟਾਂ > sess->vtcm_ਮਾਪ) {
        ਵਾਪਸ ਝੂਠ;
    }

    kparams->kernel_type             = HTP_COPY_KERNEL_SAMESHAPE_CONVERT;
    kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ              = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    kparams->n_ਧਾਗੇ               = (uint8_t) n_ਧਾਗੇ;
    kparams->vtcm_ਮਾਪ               = layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
    kparams->u.convert.src0_buf_ਮਾਪ = layout.src0_buf_ਮਾਪ;
    kparams->u.convert.dst_buf_ਮਾਪ  = layout.dst_buf_ਮਾਪ;
    kparams->u.convert.spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread = layout.spad0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    kparams->u.convert.spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread = layout.spad1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    kparams->u.convert.src0_ਪੰਕਤੀ_stride = layout.src0_ਪੰਕਤੀ_stride;
    kparams->u.convert.dst_ਪੰਕਤੀ_stride  = layout.dst_ਪੰਕਤੀ_stride;
    kparams->u.convert.blk_ਪੰਕਤੀਆਂ        = layout.blk_ਪੰਕਤੀਆਂ;
    kparams->u.convert.div_ne01      = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_ਮੁੱਲਾਂ((uint32_t) (src0->ne[2] * src0->ne[1]));

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_fused_mmnx_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0, // W0
    ਅਚਲ ਬਣਤਰ ggml_tensor * act,  // x
    int32_t n_weights,
    ਬਣਤਰ htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));
    kparams->n_ਧਾਗੇ = sess->n_ਧਾਗੇ;

    ਅਚਲ ਪੂਰਨ ne00 = src0->ne[0];
    ਅਚਲ ਪੂਰਨ ne01 = src0->ne[1];
    ਅਚਲ ਪੂਰਨ ne02 = src0->ne[2];
    ਅਚਲ ਪੂਰਨ ne03 = src0->ne[3];

    ਅਚਲ ਪੂਰਨ ne10 = act->ne[0];
    ਅਚਲ ਪੂਰਨ ne11 = act->ne[1];
    ਅਚਲ ਪੂਰਨ ne12 = act->ne[2];
    ਅਚਲ ਪੂਰਨ ne13 = act->ne[3];

    ਅਚਲ ਤਾਰਕਿਕ is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ਅਚਲ ਪੂਰਨ wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ਅਚਲ ਪੂਰਨ ne00_padded = is_tiled ? hex_ਗੇੜਾ_ਉੱਪਰ(ne00, 32) : ne00;
    ਅਚਲ ਪੂਰਨ ne11_padded = hex_ਗੇੜਾ_ਉੱਪਰ(ne11, 32);
    ਅਚਲ ਪੂਰਨ ne01_tiled  = hex_ਗੇੜਾ_ਉੱਪਰ(ne01, 32);

    ਅਚਲ size_t vtcm_budget = sess->vtcm_ਮਾਪ;
    ਅਚਲ ਤਾਰਕਿਕ is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ਤਾਰਕਿਕ hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    ਜੇ (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, ਖਾਲੀ_ਸੂਚਕ, ਝੂਠ, is_batched)) {
        ਜੇ (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, ਖਾਲੀ_ਸੂਚਕ, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, ਝੂਠ, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            ਜਾਓ finalize;
        }
    }

    ਜੇ (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        ਵਾਪਸ;
    }

    {
        ਅਚਲ ਪੂਰਨ act_nrows = ne11 * ne12 * ne13;
        ਅਚਲ size_t act_ਪੰਕਤੀ_ਮਾਪ = htp_mm_tiled_act_ਪੰਕਤੀ_ਮਾਪ(wtype, ne10);
        ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ = src0->nb[1];

        ਅਚਲ uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t ਉੱਤਮ_n_prefetch = 2;
        ਲਈ (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            ਬਣਤਰ htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ, wtype, ne10, act_nrows, sess->n_ਧਾਗੇ,
                0, src0_ਪੰਕਤੀ_ਮਾਪ, act_ਪੰਕਤੀ_ਮਾਪ, 0, d, ਝੂਠ, ਸੱਚ
            );
            ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ <= sess->vtcm_ਮਾਪ) {
                ਉੱਤਮ_n_prefetch = d;
                ਰੋਕੋ;
            }
        }

        ਬਣਤਰ htp_mm_hvx_vtcm_layout L;
        ਤਾਰਕਿਕ try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ, wtype, ne10, act_nrows, sess->n_ਧਾਗੇ,
            0, src0_ਪੰਕਤੀ_ਮਾਪ, act_ਪੰਕਤੀ_ਮਾਪ, 0, ਉੱਤਮ_n_prefetch, ਝੂਠ, ਸੱਚ
        );

        ਜੇ (try_tiled && L.ਕੁੱਲ_ਬਾਈਟਾਂ <= sess->vtcm_ਮਾਪ) {
            kparams->kernel_type       = HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ;
            kparams->tile_ਮਾਪ         = htp_tensor_get_tile_ਮਾਪ(wtype);
            kparams->aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(wtype);
            kparams->act_ਪੰਕਤੀ_ਮਾਪ      = act_ਪੰਕਤੀ_ਮਾਪ;
            kparams->vtcm_src0_ਮਾਪ    = L.src0_ਬਾਈਟਾਂ;
            kparams->vtcm_act_ਮਾਪ     = L.act_ਬਾਈਟਾਂ;
            kparams->vtcm_bias_ਮਾਪ    = 0;
            kparams->vtcm_dst_ਮਾਪ     = L.dst_ਬਾਈਟਾਂ;
            kparams->vtcm_ਮਾਪ         = L.ਕੁੱਲ_ਬਾਈਟਾਂ;
            kparams->n_prefetch        = ਉੱਤਮ_n_prefetch;
            kparams->n_weights         = n_weights;
        } ਨਹੀਂਤਾਂ {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            ਵਾਪਸ;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_ਮੁੱਲਾਂ(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_ਮੁੱਲਾਂ(ne11);
    kparams->div_r2       = init_fastdiv_ਮੁੱਲਾਂ(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_ਮੁੱਲਾਂ(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_ਮੁੱਲਾਂ(ne12);
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_precompute_fused_mmidnx_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0, // W0
    ਅਚਲ ਬਣਤਰ ggml_tensor * act,  // x
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst,  // dst0
    int32_t n_weights,
    ਬਣਤਰ htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_op_mul_mat_use_fwht(ਅਚਲ ਬਣਤਰ ggml_tensor * dst) {
    ਜੇ (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = dst->src[1];

    ਅਚਲ int64_t n = src1->ne[0];
    ਜੇ (n < 32 || (n & (n - 1)) != 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_ਪੰਕਤੀਆਂ(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_mul_mat(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * dst) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = dst->src[1];

    ਜੇ (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        ਜੇ (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਬਣਤਰ htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, ਖਾਲੀ_ਸੂਚਕ, dst, &ukparams);
        ਵਾਪਸ ukparams.vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread > 0 && (size_t) ukparams.vtcm_ਮਾਪ <= sess->vtcm_ਮਾਪ;
    }

    ਜੇ (dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q2_K: {
            ਜੇ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਅਚਲ int64_t qk = ggml_blck_ਮਾਪ(src0->type);
            ਜੇ (src0->ne[0] % qk) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਜੇ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਰੋਕੋ;
        }

        ਹਾਲਤ GGML_TYPE_F16:
            ਜੇ (src0->nb[1] < src0->nb[0]) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਰੋਕੋ;

        ਹਾਲਤ GGML_TYPE_F32:
            ਜੇ (src1->type != GGML_TYPE_F32) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (src0->nb[1] < src0->nb[0]) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਰੋਕੋ;

        ਹਾਲਤ GGML_TYPE_BF16:
            ਜੇ (src0->nb[1] < src0->nb[0]) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਰੋਕੋ;

        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }

    ਬਣਤਰ htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    ਜੇ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_ਮਾਪ, sess->vtcm_ਮਾਪ);
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_mul_mat_id(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src2 = op->src[2];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q2_K:
            ਜੇ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਜੇ (src0->ne[0] % ggml_blck_ਮਾਪ(src0->type) != 0) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਰੋਕੋ;

        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_F32:
        ਹਾਲਤ GGML_TYPE_BF16:
            ਜੇ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਜੇ (src0->ne[0] % 32 != 0) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਰੋਕੋ;

        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }

    ਬਣਤਰ htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    ਜੇ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_ਮਾਪ, sess->vtcm_ਮਾਪ);
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_ਦੁਆਧਾਰੀ(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src0->type == GGML_TYPE_F32) {
        ਜੇ (src1->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (dst->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16) {
        ਜੇ (src1->type != GGML_TYPE_F16) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (dst->type != GGML_TYPE_F16) {
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ਨਹੀਂਤਾਂ {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_are_same_shape(src0, dst)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_kernel_params kparams;
    ਵਾਪਸ ggml_hexagon_precompute_ਦੁਆਧਾਰੀ_params(sess, op_remap_ਮੰਜ਼ਿਲ_htp(op), src0, src1, dst, &kparams);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_ਜੋੜੋ_id(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src2 = op->src[2];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (!src2) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_are_same_shape(src0, dst)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src0->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || src1->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || dst->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || src2->nb[0] != ਆਕਾਰ(int32_t)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // REVISIT: add support for non-contiguous tensors
    ਜੇ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src2->nb[1] < src2->ne[0] * ਆਕਾਰ(int32_t)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_kernel_params kparams;
    ਵਾਪਸ ggml_hexagon_precompute_ਦੁਆਧਾਰੀ_params(sess, HTP_OP_ਜੋੜੋ_ID, src0, src1, dst, &kparams, src2);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_unary(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (dst->type != src0->type) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    ਜੇ (src0->type == GGML_TYPE_F16) {
        ਚੋਣ (op->op) {
            ਹਾਲਤ GGML_OP_NORM:
            ਹਾਲਤ GGML_OP_RMS_NORM:
            ਹਾਲਤ GGML_OP_L2_NORM:
            ਹਾਲਤ GGML_OP_SCALE:
            ਹਾਲਤ GGML_OP_CLAMP:
            ਹਾਲਤ GGML_OP_SQR:
            ਹਾਲਤ GGML_OP_SQRT:
            ਹਾਲਤ GGML_OP_LOG:
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_UNARY:
                ਜੇ (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_ਕਦਮ) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਰੋਕੋ;
            ਮੂਲ:
                ਵਾਪਸ ਝੂਠ;
        }
    }

    ਜੇ (!ggml_are_same_shape(src0, dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // dst must be contiguous; src0 may be non-contiguous
    ਜੇ (!ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_ਜੋੜ(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src0->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_ਜੋੜ_ਪੰਕਤੀਆਂ(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src0->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    // TODO: add support for non-contigiuos tensors
    ਜੇ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_argmax(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src0->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (dst->type != GGML_TYPE_I32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_pool_2d(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ int32_t * params = op->op_params;

    ਜੇ (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_hexagon_pool_shape_ਸਮਾਉਂਦਾ(src0, op)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int32_t kernel_x = params[1];
    ਅਚਲ int32_t kernel_y = params[2];
    ਅਚਲ int32_t stride_x = params[3];
    ਅਚਲ int32_t stride_y = params[4];
    ਅਚਲ int32_t pad_x    = params[5];
    ਅਚਲ int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    ਜੇ (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    ਅਚਲ uint32_t n_ਧਾਗੇ = (std::min)((uint32_t) sess->n_ਧਾਗੇ, planes > 0 ? planes : 1);

    ਬਣਤਰ htp_pool_vtcm_layout layout;
    ਵਾਪਸ htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_ਧਾਗੇ, sess->vtcm_ਮਾਪ);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_pool_1d(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ int32_t * params = op->op_params;

    ਜੇ (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_hexagon_pool_shape_ਸਮਾਉਂਦਾ(src0, op)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int32_t kernel = params[1];
    ਅਚਲ int32_t stride = params[2];
    ਅਚਲ int32_t pad    = params[3];
    ਜੇ (kernel <= 0 || stride <= 0 || pad < 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ਅਚਲ uint32_t n_ਧਾਗੇ = (std::min)((uint32_t) sess->n_ਧਾਗੇ, planes > 0 ? planes : 1);

    ਬਣਤਰ htp_pool_vtcm_layout layout;
    ਵਾਪਸ htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_ਧਾਗੇ, sess->vtcm_ਮਾਪ);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_activations(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (dst->type != src0->type) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous_1(src0)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src1) {
        ਜੇ (src1->type != src0->type) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (!ggml_are_same_shape(src0, src1)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (!ggml_is_contiguous_1(src1)) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_softmax(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src2 = op->src[2];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src2) {
        ਵਾਪਸ ਝੂਠ;  // FIXME: add support for sinks
    }

    ਜੇ (src0->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src0->ne[2] > 512) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src1) {
        ਜੇ (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (src0->ne[0] != src1->ne[0]) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (src1->ne[1] < src0->ne[1]) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (src0->ne[2] % src1->ne[2] != 0) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (src0->ne[3] % src1->ne[3] != 0) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਜੇ (src1) {
        ਜੇ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            ਵਾਪਸ ਝੂਠ;
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    ਅਚਲ int64_t ne0 = src0->ne[0];
    ਜੇ (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    // HVX vector size constraints for softmax
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SOFTMAX_MAX_ਪੰਕਤੀ_ਮਾਪ 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    ਜੇ (ne0 > SOFTMAX_MAX_ਪੰਕਤੀ_ਮਾਪ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ uint32_t n_ਧਾਗੇ  = (std::min)((uint32_t) sess->n_ਧਾਗੇ, src0_nrows);
    ਬਣਤਰ htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != ਖਾਲੀ_ਸੂਚਕ, src1 && src1->type == GGML_TYPE_F16, n_ਧਾਗੇ);
    ਜੇ (layout.ਕੁੱਲ_ਬਾਈਟਾਂ > sess->vtcm_ਮਾਪ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0]; //ਮੁੱਲਾਂ
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1]; // indices
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    ਜੇ (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src0->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_get_ਪੰਕਤੀਆਂ(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0]; //ਮੁੱਲਾਂ
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1]; // indices
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous_ਪੰਕਤੀਆਂ(src0) || !ggml_is_contiguous_ਪੰਕਤੀਆਂ(src1) || !ggml_is_contiguous_ਪੰਕਤੀਆਂ(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src0->type == dst->type) {
        ਜੇ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            ਵਾਪਸ ਝੂਠ;
        }
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_I32) {
        ਵਾਪਸ ਝੂਠ;
    } ਨਹੀਂਤਾਂ ਜੇ (dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    ਜੇ (ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(op)) {
        ਵਾਪਸ ਸੱਚ;
    }

    ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params kparams;
    ggml_hexagon_precompute_get_ਪੰਕਤੀਆਂ_params(sess, src0, src1, dst, &kparams);
    ਜੇ (kparams.n_ਧਾਗੇ == 0 || (size_t) kparams.vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_argsort(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0]; //ਮੁੱਲਾਂ
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;         // indices

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ uint32_t n_ਧਾਗੇ_max = sess->n_ਧਾਗੇ > 0 ? sess->n_ਧਾਗੇ : 4;
    ਅਚਲ size_t vtcm_budget = sess->vtcm_ਮਾਪ > 0 ? sess->vtcm_ਮਾਪ : (8 * 1024 * 1024);

    ਬਣਤਰ htp_ਛਾਂਟੋ_vtcm_layout layout;
    ਜੇ (!htp_ਛਾਂਟੋ_solve_layout(&layout, src0->ne[0], ਕੁੱਲ_ਪੰਕਤੀਆਂ, dst->ne[0], n_ਧਾਗੇ_max, vtcm_budget, ਝੂਠ)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_ਸਿਖਰ_k(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0]; //ਮੁੱਲਾਂ
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;         // indices

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ uint32_t n_ਧਾਗੇ_max = sess->n_ਧਾਗੇ > 0 ? sess->n_ਧਾਗੇ : 4;
    ਅਚਲ size_t vtcm_budget = sess->vtcm_ਮਾਪ > 0 ? sess->vtcm_ਮਾਪ : (8 * 1024 * 1024);

    ਬਣਤਰ htp_ਛਾਂਟੋ_vtcm_layout layout;
    ਜੇ (!htp_ਛਾਂਟੋ_solve_layout(&layout, src0->ne[0], ਕੁੱਲ_ਪੰਕਤੀਆਂ, dst->ne[0], n_ਧਾਗੇ_max, vtcm_budget, ਸੱਚ)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_rope(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src2 = op->src[2];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (!ggml_are_same_shape(src0, dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src0->ne[0] <= 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਜੇ (src0_nrows == 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int32_t * op_params = &op->op_params[0];
    ਅਚਲ ਪੂਰਨ n_dims = op_params[1];
    ਅਚਲ ਪੂਰਨ mode   = op_params[2];
    ਅਚਲ ਪੂਰਨ n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    ਜੇ (n_dims < 0 || n_dims % 2 != 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    ਜੇ (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਦਸ਼ਮਲਵ freq_ਅਧਾਰ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&freq_ਅਧਾਰ, op_params + 5, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਜੇ (freq_ਅਧਾਰ < 0.0f) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (mode != GGML_ROPE_TYPE_ਸਾਧਾਰਨ &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਤਾਰਕਿਕ is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    ਜੇ (mode == GGML_ROPE_TYPE_VISION) {
        ਜੇ (n_dims != (ਪੂਰਨ) (src0->ne[0] / 2) || n_offs != 0) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਜੇ (is_mrope) {
        ਅਚਲ int32_t * sections = op_params + 11;
        ਜੇ (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਅਚਲ int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    ਜੇ (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src2) {
        ਜੇ (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (src2->ne[0] < (n_dims / 2)) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    ਜੇ (src0->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || dst->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src0->nb[1] < src0->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ) || dst->nb[1] < dst->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = (std::min)((uint32_t) sess->n_ਧਾਗੇ, src0_nrows);
    ਅਚਲ uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ਬਣਤਰ htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_ਧਾਗੇ, n_freq_factors);
    ਜੇ (layout.ਕੁੱਲ_ਬਾਈਟਾਂ > sess->vtcm_ਮਾਪ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_ssm_conv(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    // Only support FP32 for now
    ਜੇ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    // Check IO tensor shapes and dims
    ਜੇ (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        ਵਾਪਸ ਝੂਠ; // src0 should be effectively 3D
    }

    ਅਚਲ ਪੂਰਨ d_conv  = src1->ne[0];
    ਅਚਲ ਪੂਰਨ d_inner = src0->ne[1];
    ਅਚਲ ਪੂਰਨ n_t = dst->ne[1];
    ਅਚਲ ਪੂਰਨ n_s = dst->ne[2];

    ਜੇ (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src0->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || src1->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || dst->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (src0->nb[1] != src0->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ) || src1->nb[1] != src1->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਬਣਤਰ htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    ਜੇ ((size_t) kparams.vtcm_ਮਾਪ > sess->vtcm_ਮਾਪ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ im2col_3d_calc_output_ਮਾਪ(uint32_t ਨਿਵੇਸ਼, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    ਜੇ (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int64_t numerator = (int64_t) ਨਿਵੇਸ਼ + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    ਜੇ (numerator < 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint64_t size = (uint64_t) (numerator / stride) + 1;
    ਜੇ (size == 0 || size > UINT32_MAX) {
        ਵਾਪਸ ਝੂਠ;
    }

    *output = (uint32_t) size;
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_im2col_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));
    kparams->kernel_type = HTP_IM2_ਖਾਨਾ_KERNEL_UNSUPPORTED;

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (!src0 || !src1 || !dst) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src1->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || !ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int32_t * op_params = (ਅਚਲ int32_t *) op->op_params;
    ਅਚਲ int32_t s0 = op_params[0];
    ਅਚਲ int32_t s1 = op_params[1];
    ਅਚਲ int32_t p0 = op_params[2];
    ਅਚਲ int32_t p1 = op_params[3];
    ਅਚਲ int32_t d0 = op_params[4];
    ਅਚਲ int32_t d1 = op_params[5];
    ਅਚਲ ਤਾਰਕਿਕ is_2D = (op_params[6] == 1);

    ਜੇ (s0 <= 0 || d0 <= 0 || p0 < 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    ਅਚਲ uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    ਅਚਲ uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    ਅਚਲ uint32_t IW = (uint32_t) src1->ne[0];
    ਅਚਲ uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    ਅਚਲ uint32_t KW = (uint32_t) src0->ne[0];
    ਅਚਲ uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    ਅਚਲ uint32_t OW = (uint32_t) dst->ne[1];

    ਜੇ (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    ਅਚਲ uint64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ_64   = (uint64_t) N * OH;
    ਜੇ (patch_stride_64 > UINT32_MAX || ਕੁੱਲ_ਪੰਕਤੀਆਂ_64 > UINT32_MAX) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ uint32_t patch_stride = (uint32_t) patch_stride_64;
    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ   = (uint32_t) ਕੁੱਲ_ਪੰਕਤੀਆਂ_64;

    ਜੇ (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (is_2D && dst->ne[2] != OH) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? ਆਕਾਰ(ggml_fp16_t) : ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਤਾਰਕਿਕ is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    ਜੇ (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    ਅਚਲ uint32_t max_ਧਾਗੇ = (sess && sess->n_ਧਾਗੇ > 0) ? (uint32_t) sess->n_ਧਾਗੇ : 8;
    ਅਚਲ uint32_t n_ਧਾਗੇ   = (ਕੁੱਲ_ਪੰਕਤੀਆਂ < max_ਧਾਗੇ) ? ਕੁੱਲ_ਪੰਕਤੀਆਂ : max_ਧਾਗੇ;
    ਅਚਲ uint32_t vtcm_ਮਾਪ   = (sess && sess->vtcm_ਮਾਪ > 0) ? sess->vtcm_ਮਾਪ : 0;

    ਜੇ (is_patchembed) {
        ਅਚਲ uint64_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw = (uint64_t) IC * KH * IW * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ uint64_t dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw = (uint64_t) OW * patch_stride * dst_elem;
        ਜੇ (src_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw <= UINT32_MAX && dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw <= UINT32_MAX) {
            ਅਚਲ uint32_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ((uint32_t) src_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw, 256);
            ਅਚਲ uint32_t dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ((uint32_t) dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw, 256);

            ਬਣਤਰ htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_ਪੰਕਤੀ_ਬਾਈਟਾਂ, dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ, n_ਧਾਗੇ);

            ਜੇ (vtcm_ਮਾਪ == 0 || layout.ਕੁੱਲ_ਬਾਈਟਾਂ <= vtcm_ਮਾਪ) {
                kparams->kernel_type          = HTP_IM2_ਖਾਨਾ_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = is_2D ? 1 : 0;
                kparams->is_3d                = 0;
                kparams->n_ਧਾਗੇ            = (uint8_t) n_ਧਾਗੇ;
                kparams->vtcm_ਮਾਪ            = layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = layout.src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
                kparams->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = layout.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
                kparams->src_ਪੰਕਤੀ_ਬਾਈਟਾਂ        = src_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
                kparams->dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ        = dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
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
                kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ           = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
                ਵਾਪਸ ਸੱਚ;
            }
        }
    }

    ਲਈ (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ਅਚਲ uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        ਜੇ (Wb > UINT32_MAX) {
            ਜੇ (owb == 1) ਰੋਕੋ;
            ਜਾਰੀ;
        }

        ਅਚਲ uint64_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw = (uint64_t) IC * KH * Wb * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ uint64_t dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw = (uint64_t) owb * patch_stride * dst_elem;
        ਜੇ (src_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw > UINT32_MAX || dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw > UINT32_MAX) {
            ਜੇ (owb == 1) ਰੋਕੋ;
            ਜਾਰੀ;
        }

        ਅਚਲ uint32_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ((uint32_t) src_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw, 256);
        ਅਚਲ uint32_t dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ((uint32_t) dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw, 256);

        ਬਣਤਰ htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_ਪੰਕਤੀ_ਬਾਈਟਾਂ, dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ, n_ਧਾਗੇ);

        ਜੇ (vtcm_ਮਾਪ == 0 || layout.ਕੁੱਲ_ਬਾਈਟਾਂ <= vtcm_ਮਾਪ) {
            kparams->kernel_type          = HTP_IM2_ਖਾਨਾ_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = is_2D ? 1 : 0;
            kparams->is_3d                = 0;
            kparams->n_ਧਾਗੇ            = (uint8_t) n_ਧਾਗੇ;
            kparams->vtcm_ਮਾਪ            = layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = layout.src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
            kparams->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = layout.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
            kparams->src_ਪੰਕਤੀ_ਬਾਈਟਾਂ        = src_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
            kparams->dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ        = dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
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
            kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ           = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (owb == 1) ਰੋਕੋ;
    }

    ਵਾਪਸ ਝੂਠ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_precompute_im2col_3d_params(
    ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess,
    ਅਚਲ ਬਣਤਰ ggml_tensor * op,
    ਬਣਤਰ htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, ਆਕਾਰ(*kparams));
    kparams->kernel_type = HTP_IM2_ਖਾਨਾ_KERNEL_UNSUPPORTED;

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (!src0 || !src1 || !dst) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src1->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || !ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int32_t * params = (ਅਚਲ int32_t *) op->op_params;
    ਅਚਲ int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    ਅਚਲ int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    ਅਚਲ int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    ਅਚਲ int32_t ic = params[9];

    ਜੇ (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src1->ne[3] % ic != 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t IC = (uint32_t) ic;
    ਅਚਲ uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    ਜੇ (N == 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    uint32_t OW, OH, OD;
    ਜੇ (!im2col_3d_calc_output_ਮਾਪ(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_ਮਾਪ(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_ਮਾਪ(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t KD = (uint32_t) src0->ne[2];
    ਅਚਲ uint32_t KH = (uint32_t) src0->ne[1];
    ਅਚਲ uint32_t KW = (uint32_t) src0->ne[0];
    ਅਚਲ uint32_t ID = (uint32_t) src1->ne[2];
    ਅਚਲ uint32_t IH = (uint32_t) src1->ne[1];
    ਅਚਲ uint32_t IW = (uint32_t) src1->ne[0];

    ਅਚਲ uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    ਅਚਲ uint64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ_64   = (uint64_t) N * OD * OH;
    ਅਚਲ uint64_t dst_ne3_64      = (uint64_t) OD * N;

    ਜੇ (patch_stride_64 > UINT32_MAX || ਕੁੱਲ_ਪੰਕਤੀਆਂ_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ uint32_t patch_stride = (uint32_t) patch_stride_64;
    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ   = (uint32_t) ਕੁੱਲ_ਪੰਕਤੀਆਂ_64;
    ਅਚਲ uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    ਜੇ (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? ਆਕਾਰ(ggml_fp16_t) : ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ ਤਾਰਕਿਕ is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    ਅਚਲ uint32_t max_ਧਾਗੇ = (sess && sess->n_ਧਾਗੇ > 0) ? (uint32_t) sess->n_ਧਾਗੇ : 8;
    ਅਚਲ uint32_t n_ਧਾਗੇ   = (ਕੁੱਲ_ਪੰਕਤੀਆਂ < max_ਧਾਗੇ) ? ਕੁੱਲ_ਪੰਕਤੀਆਂ : max_ਧਾਗੇ;
    ਅਚਲ uint32_t vtcm_ਮਾਪ   = (sess && sess->vtcm_ਮਾਪ > 0) ? sess->vtcm_ਮਾਪ : 0;

    ਜੇ (is_patchembed) {
        ਅਚਲ uint64_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw = (uint64_t) IC * KD * KH * IW * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ uint64_t dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw = (uint64_t) OW * patch_stride * dst_elem;
        ਜੇ (src_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw <= UINT32_MAX && dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw <= UINT32_MAX) {
            ਅਚਲ uint32_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ((uint32_t) src_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw, 256);
            ਅਚਲ uint32_t dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ((uint32_t) dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw, 256);

            ਬਣਤਰ htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_ਪੰਕਤੀ_ਬਾਈਟਾਂ, dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ, n_ਧਾਗੇ);

            ਜੇ (vtcm_ਮਾਪ == 0 || layout.ਕੁੱਲ_ਬਾਈਟਾਂ <= vtcm_ਮਾਪ) {
                kparams->kernel_type          = HTP_IM2_ਖਾਨਾ_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = 0;
                kparams->is_3d                = 1;
                kparams->n_ਧਾਗੇ            = (uint8_t) n_ਧਾਗੇ;
                kparams->vtcm_ਮਾਪ            = layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = layout.src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
                kparams->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = layout.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
                kparams->src_ਪੰਕਤੀ_ਬਾਈਟਾਂ        = src_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
                kparams->dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ        = dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
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
                kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ           = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
                ਵਾਪਸ ਸੱਚ;
            }
        }
    }

    ਲਈ (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ਅਚਲ uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        ਜੇ (Wb > UINT32_MAX) {
            ਜੇ (owb == 1) ਰੋਕੋ;
            ਜਾਰੀ;
        }

        ਅਚਲ uint64_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw = (uint64_t) IC * KD * KH * Wb * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ uint64_t dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw = (uint64_t) owb * patch_stride * dst_elem;
        ਜੇ (src_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw > UINT32_MAX || dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw > UINT32_MAX) {
            ਜੇ (owb == 1) ਰੋਕੋ;
            ਜਾਰੀ;
        }

        ਅਚਲ uint32_t src_ਪੰਕਤੀ_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ((uint32_t) src_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw, 256);
        ਅਚਲ uint32_t dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ = hex_ਗੇੜਾ_ਉੱਪਰ((uint32_t) dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ_raw, 256);

        ਬਣਤਰ htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_ਪੰਕਤੀ_ਬਾਈਟਾਂ, dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ, n_ਧਾਗੇ);

        ਜੇ (vtcm_ਮਾਪ == 0 || layout.ਕੁੱਲ_ਬਾਈਟਾਂ <= vtcm_ਮਾਪ) {
            kparams->kernel_type          = HTP_IM2_ਖਾਨਾ_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = 0;
            kparams->is_3d                = 1;
            kparams->n_ਧਾਗੇ            = (uint8_t) n_ਧਾਗੇ;
            kparams->vtcm_ਮਾਪ            = layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = layout.src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
            kparams->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = layout.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
            kparams->src_ਪੰਕਤੀ_ਬਾਈਟਾਂ        = src_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
            kparams->dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ        = dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ;
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
            kparams->ਕੁੱਲ_ਪੰਕਤੀਆਂ           = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
            ਵਾਪਸ ਸੱਚ;
        }
        ਜੇ (owb == 1) ਰੋਕੋ;
    }

    ਵਾਪਸ ਝੂਠ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_im2col(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਬਣਤਰ htp_im2col_kernel_params kparams;
    ਵਾਪਸ ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_im2col_3d(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਬਣਤਰ htp_im2col_kernel_params kparams;
    ਵਾਪਸ ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_pad(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int32_t lp0 = ((ਅਚਲ int32_t *) op->op_params)[0];
    ਅਚਲ int32_t rp0 = ((ਅਚਲ int32_t *) op->op_params)[1];
    ਅਚਲ int32_t circular = ((ਅਚਲ int32_t *) op->op_params)[8];

    ਜੇ (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_cumsum(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_diag(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    // diag only supports F32 currently
    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    // Input must have ne[1] == 1 (vector input)
    ਜੇ (src0->ne[1] != 1) {
        ਵਾਪਸ ਝੂਠ;
    }

    // Output must be square in first two dimensions
    ਜੇ (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_solve_tri(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0]; // A
    ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = op->src[1]; // B
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;         // X

    ਜੇ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src0->ne[0] != src0->ne[1]) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src0->ne[1] != src1->ne[1]) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_tri(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src0->type != GGML_TYPE_F32) { ਵਾਪਸ ਝੂਠ; }
    ਜੇ (dst->type  != GGML_TYPE_F32) { ਵਾਪਸ ਝੂਠ; }
    ਜੇ (!ggml_are_same_shape(src0, dst)) { ਵਾਪਸ ਝੂਠ; }
    ਜੇ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { ਵਾਪਸ ਝੂਠ; }

    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_hexagon_name(ggml_backend_t backend) {
    ਸਵੈਚਾਲਿਤ sess = static_cast<ggml_hexagon_session *>(backend->context);
    ਵਾਪਸ sess->c_name();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    ਮਿਟਾਓ backend;
}

ਸਥਿਰ htp_op_ਕੋਡ op_remap_ਮੰਜ਼ਿਲ_htp(ਅਚਲ ggml_tensor * t) {
    ਚੋਣ (t->op) {
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:  ਵਾਪਸ HTP_OP_FLASH_ATTN_EXT;
        ਹਾਲਤ GGML_OP_MUL_MAT:
            ਜੇ (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                ਵਾਪਸ HTP_OP_HADAMARD;
            }
            ਵਾਪਸ HTP_OP_MUL_MAT;
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:      ਵਾਪਸ HTP_OP_MUL_MAT_ID;
        ਹਾਲਤ GGML_OP_MUL:             ਵਾਪਸ HTP_OP_MUL;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:             ਵਾਪਸ HTP_OP_ਜੋੜੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:          ਵਾਪਸ HTP_OP_ਜੋੜੋ_ID;
        ਹਾਲਤ GGML_OP_SUB:             ਵਾਪਸ HTP_OP_SUB;
        ਹਾਲਤ GGML_OP_DIV:             ਵਾਪਸ HTP_OP_DIV;
        ਹਾਲਤ GGML_OP_DUP:
        ਹਾਲਤ GGML_OP_CPY:             ਵਾਪਸ HTP_OP_CPY;
        ਹਾਲਤ GGML_OP_CONT:            ਵਾਪਸ HTP_OP_CPY;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:        ਵਾਪਸ HTP_OP_GET_ਪੰਕਤੀਆਂ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:        ਵਾਪਸ HTP_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ;
        ਹਾਲਤ GGML_OP_ਜੋੜ:             ਵਾਪਸ HTP_OP_ਜੋੜ;
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:        ਵਾਪਸ HTP_OP_ਜੋੜ_ਪੰਕਤੀਆਂ;
        ਹਾਲਤ GGML_OP_POOL_2D:         ਵਾਪਸ HTP_OP_POOL_2D;
        ਹਾਲਤ GGML_OP_POOL_1D:         ਵਾਪਸ HTP_OP_POOL_1D;
        ਹਾਲਤ GGML_OP_ARGSORT:         ਵਾਪਸ HTP_OP_ARGSORT;
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:           ਵਾਪਸ HTP_OP_ਸਿਖਰ_K;
        ਹਾਲਤ GGML_OP_ARGMAX:          ਵਾਪਸ HTP_OP_ARGMAX;
        ਹਾਲਤ GGML_OP_NORM:            ਵਾਪਸ HTP_OP_NORM;
        ਹਾਲਤ GGML_OP_L2_NORM:         ਵਾਪਸ HTP_OP_L2_NORM;
        ਹਾਲਤ GGML_OP_RMS_NORM:        ਵਾਪਸ HTP_OP_RMS_NORM;
        ਹਾਲਤ GGML_OP_CONCAT:          ਵਾਪਸ HTP_OP_CONCAT;
        ਹਾਲਤ GGML_OP_SCALE:           ਵਾਪਸ HTP_OP_SCALE;
        ਹਾਲਤ GGML_OP_CLAMP:           ਵਾਪਸ HTP_OP_CLAMP;
        ਹਾਲਤ GGML_OP_LEAKY_RELU:      ਵਾਪਸ HTP_OP_LEAKY_RELU;
        ਹਾਲਤ GGML_OP_SQR:             ਵਾਪਸ HTP_OP_SQR;
        ਹਾਲਤ GGML_OP_SQRT:            ਵਾਪਸ HTP_OP_SQRT;
        ਹਾਲਤ GGML_OP_LOG:             ਵਾਪਸ HTP_OP_UNARY_LOG;
        ਹਾਲਤ GGML_OP_SOFT_MAX:        ਵਾਪਸ HTP_OP_SOFTMAX;
        ਹਾਲਤ GGML_OP_SSM_CONV:        ਵਾਪਸ HTP_OP_SSM_CONV;
        ਹਾਲਤ GGML_OP_GATED_DELTA_NET: ਵਾਪਸ HTP_OP_GATED_DELTA_NET;
        ਹਾਲਤ GGML_OP_ROPE:            ਵਾਪਸ HTP_OP_ROPE;
        ਹਾਲਤ GGML_OP_REPEAT:          ਵਾਪਸ HTP_OP_REPEAT;
        ਹਾਲਤ GGML_OP_CUMSUM:          ਵਾਪਸ HTP_OP_CUMSUM;
        ਹਾਲਤ GGML_OP_FILL:            ਵਾਪਸ HTP_OP_FILL;
        ਹਾਲਤ GGML_OP_DIAG:            ਵਾਪਸ HTP_OP_DIAG;
        ਹਾਲਤ GGML_OP_SOLVE_TRI:       ਵਾਪਸ HTP_OP_SOLVE_TRI;
        ਹਾਲਤ GGML_OP_TRI:             ਵਾਪਸ HTP_OP_TRI;
        ਹਾਲਤ GGML_OP_PAD:             ਵਾਪਸ HTP_OP_PAD;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:          ਵਾਪਸ HTP_OP_IM2_ਖਾਨਾ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_3_D:       ਵਾਪਸ HTP_OP_IM2_ਖਾਨਾ_3_D;
        ਹਾਲਤ GGML_OP_ROLL:            ਵਾਪਸ HTP_OP_ROLL;

        ਹਾਲਤ GGML_OP_UNARY:
            ਚੋਣ (ggml_get_unary_op(t)) {
                ਹਾਲਤ GGML_UNARY_OP_SILU:       ਵਾਪਸ HTP_OP_UNARY_SILU;
                ਹਾਲਤ GGML_UNARY_OP_GELU:       ਵਾਪਸ HTP_OP_UNARY_GELU;
                ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼: ਵਾਪਸ HTP_OP_UNARY_GELU_ਤੇਜ਼;
                ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:   ਵਾਪਸ HTP_OP_UNARY_GELU_ERF;
                ਹਾਲਤ GGML_UNARY_OP_SIGMOID:    ਵਾਪਸ HTP_OP_UNARY_SIGMOID;
                ਹਾਲਤ GGML_UNARY_OP_NEG:        ਵਾਪਸ HTP_OP_UNARY_NEG;
                ਹਾਲਤ GGML_UNARY_OP_EXP:        ਵਾਪਸ HTP_OP_UNARY_EXP;
                ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:   ਵਾਪਸ HTP_OP_UNARY_SOFTPLUS;
                ਹਾਲਤ GGML_UNARY_OP_TANH:       ਵਾਪਸ HTP_OP_UNARY_TANH;
                ਹਾਲਤ GGML_UNARY_OP_ABS:        ਵਾਪਸ HTP_OP_UNARY_ABS;
                ਹਾਲਤ GGML_UNARY_OP_RELU:       ਵਾਪਸ HTP_OP_UNARY_RELU;
                ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:       ਵਾਪਸ HTP_OP_UNARY_ਕਦਮ;
            ਮੂਲ:
                ਰੋਕੋ;
            }
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_GLU:
            ਚੋਣ (ggml_get_glu_op(t)) {
                ਹਾਲਤ GGML_GLU_OP_SWIGLU:       ਵਾਪਸ HTP_OP_GLU_SWIGLU;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:   ਵਾਪਸ HTP_OP_GLU_SWIGLU_OAI;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP: ਵਾਪਸ HTP_OP_GLU_SWIGLU_CLAMP;
                ਹਾਲਤ GGML_GLU_OP_GEGLU:        ਵਾਪਸ HTP_OP_GLU_GEGLU;
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:  ਵਾਪਸ HTP_OP_GLU_GEGLU_ਤੇਜ਼;
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:    ਵਾਪਸ HTP_OP_GLU_GEGLU_ERF;
                ਮੂਲ: ਰੋਕੋ;
            }
            ਰੋਕੋ;

        ਮੂਲ:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    ਵਾਪਸ HTP_OP_INVALID;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ op_is_compute(ggml_tensor *ਗੰਢ)
{
    ਵਾਪਸ !ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ->op) && !ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ) && (ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE);
}

ਸਥਿਰ ਤਾਰਕਿਕ mm_is_hmx_eligible(ਅਚਲ ggml_tensor * t) {
    ਜੇ (opt_nhmx == 0) { ਵਾਪਸ ਝੂਠ; }

    ਅਚਲ ggml_tensor * src0 = t->src[0];
    ਅਚਲ ggml_tensor * src1 = t->src[1];

    ਜੇ (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        ਅਚਲ ggml_tensor src1_c = ggml_hexagon_tensor_collapse_ਪੰਕਤੀਆਂ(src1);
        ਵਾਪਸ ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, ਝੂਠ, ਝੂਠ);
    }

    ਅਚਲ ਤਾਰਕਿਕ is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    ਅਚਲ ਤਾਰਕਿਕ is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    ਵਾਪਸ ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

ਸਥਿਰ ਤਾਰਕਿਕ is_supported_mul_mat_nx_kernel(ਅਚਲ ggml_tensor * src0, ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams) {
    ਜੇ (kparams->n_hmx) {
        ਵਾਪਸ kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    ਜੇ (!ggml_hexagon_tensor_is_tiled(src0)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ;
}

ਸਥਿਰ ਤਾਰਕਿਕ is_supported_mul_mat_id_nx_kernel(ਅਚਲ ggml_tensor * src0, ਅਚਲ ਬਣਤਰ htp_mm_kernel_params * kparams) {
    ਜੇ (kparams->n_hmx) {
        ਵਾਪਸ kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    ਜੇ (!ggml_hexagon_tensor_is_tiled(src0)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਖੰਡ;
}

ਸਥਿਰ ਤਾਰਕਿਕ is_mergeable_mul_mat(ਅਚਲ ggml_tensor * t) {
    ਜੇ (t->op != GGML_OP_MUL_MAT) ਵਾਪਸ ਝੂਠ;
    ਜੇ (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) ਵਾਪਸ ਝੂਠ;

    ਅਚਲ ggml_tensor * src0 = t->src[0];
    ਅਚਲ ggml_tensor * src1 = t->src[1];
    ਜੇ (src0->ne[2] != 1 || src0->ne[3] != 1) ਵਾਪਸ ਝੂਠ;

    ਜੇ (mm_is_hmx_eligible(t)) {
        ਵਾਪਸ htp_is_hmx_ਭਾਰ_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    ਜੇ (src1->type != GGML_TYPE_F32) ਵਾਪਸ ਝੂਠ;

    ਵਾਪਸ ggml_hexagon_tensor_is_tiled(src0);
}

ਸਥਿਰ ਤਾਰਕਿਕ is_mergeable_mul_mat_pair(ਅਚਲ ggml_tensor * n1, ਅਚਲ ggml_tensor * n2) {
    ਜੇ (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (n1->src[1] != n2->src[1]) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (n1->src[0]->type != n2->src[0]->type) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ is_mergeable_mul_mat_id(ਅਚਲ ggml_tensor * t) {
    ਜੇ (t->op != GGML_OP_MUL_MAT_ID) ਵਾਪਸ ਝੂਠ;

    ਅਚਲ ggml_tensor * src0 = t->src[0];
    ਜੇ (mm_is_hmx_eligible(t)) {
        ਵਾਪਸ htp_is_hmx_ਭਾਰ_type(src0->type);
    }

    ਵਾਪਸ ggml_hexagon_tensor_is_tiled(src0);
}

ਸਥਿਰ ਤਾਰਕਿਕ is_mergeable_mul_mat_id_pair(ਅਚਲ ggml_tensor * n1, ਅਚਲ ggml_tensor * n2) {
    ਜੇ (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (n1->src[1] != n2->src[1]) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (n1->src[2] != n2->src[2]) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (n1->src[0]->type != n2->src[0]->type) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ggml_ਸਥਿਤੀ ggml_backend_hexagon_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    ਸਵੈਚਾਲਿਤ sess = static_cast<ggml_hexagon_session *>(backend->context);

    ਜੇ (sess->ਆਖਰੀ_error > HTP_ਸਥਿਤੀ_OK) {
        ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    ਅਚਲ std::vector<htp_opnode> * nodes_ptr = ਖਾਲੀ_ਸੂਚਕ;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    ਤਾਰਕਿਕ cache_ਪ੍ਰਾਪਤੀ = (graph->uid != 0 && sess->cached_uid == graph->uid);
    ਜੇ (cache_ਪ੍ਰਾਪਤੀ) {
        nodes_ptr = &sess->cached_nodes;
    } ਨਹੀਂਤਾਂ {
        // Tag fusable tensors in graph
        ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; i++) {
            ਸਵੈਚਾਲਿਤ * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            ਜੇ (!extra) ਜਾਰੀ;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            ਜੇ (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } ਨਹੀਂਤਾਂ ਜੇ ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                ਜੇ ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ਜੋੜੋ && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ਜੋੜੋ })) ||
                    ggml_ਗੰਢ_has_n_ਵਰਤਦਾ(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            ਜੇ (!op_is_compute(n)) {
                ਜਾਰੀ;
            }

            htp_opnode ਗੰਢ(HTP_OP_INVALID, n);
            ਗੰਢ.opcode = op_remap_ਮੰਜ਼ਿਲ_htp(n);
            ਜੇ (ਗੰਢ.opcode == HTP_OP_MUL_MAT || ਗੰਢ.opcode == HTP_OP_MUL_MAT_ID || ਗੰਢ.opcode == HTP_OP_GET_ਪੰਕਤੀਆਂ) {
                ggml_tensor * src0 = ਗੰਢ.ਗੰਢ->src[0];
                ਜੇ (ggml_hexagon_needs_repack(src0)) {
                    ਜੇ (opt_verbose) {
                        ਅੱਖਰ dims_buf[64];
                        ਅੱਖਰ strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, ਆਕਾਰ(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, ਆਕਾਰ(strides_buf), src0);
                        GGML_LOG_DEBUG("ggml-hex: %s rt-repack %s|%s|%s|%s|size %zu\n",
                                       sess->c_name(), src0->name, dims_buf, ggml_type_name(src0->type), strides_buf, ggml_nbytes(src0));
                    }
                    std::vector<uint8_t> raw(ggml_nbytes(src0));
                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(raw.ਅੰਕੜੇ(), src0->ਅੰਕੜੇ, raw.size());
                    htp_repack_tensor_tiled(src0, raw.ਅੰਕੜੇ(), raw.size());
                    ਸਵੈਚਾਲਿਤ * extra = (ggml_hexagon_tensor_extra *) src0->extra;
                    extra->flags |= GGML_HEXAGON_TENSOR_TILED;
                }
            }

            ਜੇ (ਗੰਢ.opcode == HTP_OP_MUL_MAT || ਗੰਢ.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    ਗੰਢ.ਗੰਢ->src[0], ਗੰਢ.ਗੰਢ->src[1], ਗੰਢ.ਗੰਢ,
                    (ਬਣਤਰ htp_mm_kernel_params *)ਗੰਢ.kernel_params
                );
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_MUL || ਗੰਢ.opcode == HTP_OP_ਜੋੜੋ || ਗੰਢ.opcode == HTP_OP_ਜੋੜੋ_ID || ਗੰਢ.opcode == HTP_OP_SUB || ਗੰਢ.opcode == HTP_OP_DIV) {
                ਅਚਲ ggml_tensor * src1 = ਗੰਢ.ਗੰਢ->src[1];
                ਅਚਲ ggml_tensor * src2 = (ਗੰਢ.opcode == HTP_OP_ਜੋੜੋ_ID) ? ਗੰਢ.ਗੰਢ->src[2] : ਖਾਲੀ_ਸੂਚਕ;
                GGML_ASSERT(ggml_hexagon_precompute_ਦੁਆਧਾਰੀ_params(sess,
                    ਗੰਢ.opcode, ਗੰਢ.ਗੰਢ->src[0], src1, ਗੰਢ.ਗੰਢ,
                    (ਬਣਤਰ htp_ਦੁਆਧਾਰੀ_kernel_params *) ਗੰਢ.kernel_params, src2));
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    ਗੰਢ.ਗੰਢ,
                    (ਬਣਤਰ htp_fa_kernel_params *)ਗੰਢ.kernel_params
                );
            } ਨਹੀਂਤਾਂ ਜੇ (htp_op_is_unary(ਗੰਢ.opcode)) {
                ਸਵੈਚਾਲਿਤ inputs = ਗੰਢ.get_inputs();
                ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = inputs[0];
                ਅਚਲ ਬਣਤਰ ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : ਖਾਲੀ_ਸੂਚਕ;
                ggml_hexagon_precompute_unary_params(sess,
                    ਗੰਢ.opcode, src0, src1, ਗੰਢ.dst(),
                    (ਬਣਤਰ htp_unary_kernel_params *)ਗੰਢ.kernel_params
                );
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_GET_ਪੰਕਤੀਆਂ) {
                ggml_hexagon_precompute_get_ਪੰਕਤੀਆਂ_params(sess,
                    ਗੰਢ.ਗੰਢ->src[0], ਗੰਢ.ਗੰਢ->src[1], ਗੰਢ.dst(),
                    (ਬਣਤਰ htp_get_ਪੰਕਤੀਆਂ_kernel_params *)ਗੰਢ.kernel_params
                );
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ) {
                ggml_hexagon_precompute_ਸੈੱਟ_ਪੰਕਤੀਆਂ_params(sess,
                    ਗੰਢ.ਗੰਢ->src[0], ਗੰਢ.ਗੰਢ->src[1], ਗੰਢ.dst(),
                    (ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_kernel_params *)ਗੰਢ.kernel_params
                );
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    ਗੰਢ.ਗੰਢ,
                    (ਬਣਤਰ htp_rope_kernel_params *)ਗੰਢ.kernel_params
                );
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    ਗੰਢ.ਗੰਢ->src[0], ਗੰਢ.ਗੰਢ->src[1], ਗੰਢ.dst(),
                    (ਬਣਤਰ htp_ssm_conv_kernel_params *)ਗੰਢ.kernel_params
                );
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    ਗੰਢ.ਗੰਢ,
                    (ਬਣਤਰ htp_softmax_kernel_params *)ਗੰਢ.kernel_params
                );
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    ਗੰਢ.ਗੰਢ,
                    (ਬਣਤਰ htp_gdn_kernel_params *)ਗੰਢ.kernel_params
                );
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_ARGSORT || ਗੰਢ.opcode == HTP_OP_ਸਿਖਰ_K) {
                ggml_hexagon_precompute_ਛਾਂਟੋ_params(sess,
                    ਗੰਢ.ਗੰਢ,
                    ਗੰਢ.opcode == HTP_OP_ਸਿਖਰ_K,
                    (ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params *) ਗੰਢ.kernel_params
                );
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_POOL_2D || ਗੰਢ.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, ਗੰਢ.ਗੰਢ->src[0], ਗੰਢ.dst(),
                    (ਬਣਤਰ htp_pool_2d_kernel_params *)ਗੰਢ.kernel_params,
                    ਗੰਢ.opcode == HTP_OP_POOL_1D);
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    ਗੰਢ.ਗੰਢ,
                    (ਬਣਤਰ htp_concat_kernel_params *) ਗੰਢ.kernel_params
                );
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_CPY || ਗੰਢ.opcode == HTP_OP_CPY_FENCE) {
                ਅਚਲ ਤਾਰਕਿਕ ok = ggml_hexagon_precompute_cpy_params(sess,
                    ਗੰਢ.ਗੰਢ,
                    (ਬਣਤਰ htp_copy_kernel_params *) ਗੰਢ.kernel_params
                );
                ਅਚਲ ਸਵੈਚਾਲਿਤ * kparams = (ਅਚਲ ਬਣਤਰ htp_copy_kernel_params *) ਗੰਢ.kernel_params;
                ਜੇ (ok && ਗੰਢ.opcode == HTP_OP_CPY && kparams->ਕੁੱਲ_elems == 0) {
                    ਜਾਰੀ;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_IM2_ਖਾਨਾ) {
                ggml_hexagon_precompute_im2col_params(sess,
                    ਗੰਢ.ਗੰਢ,
                    (ਬਣਤਰ htp_im2col_kernel_params *) ਗੰਢ.kernel_params
                );
            } ਨਹੀਂਤਾਂ ਜੇ (ਗੰਢ.opcode == HTP_OP_IM2_ਖਾਨਾ_3_D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    ਗੰਢ.ਗੰਢ,
                    (ਬਣਤਰ htp_im2col_kernel_params *) ਗੰਢ.kernel_params
                );
            }
            computed_nodes.push_back(std::move(ਗੰਢ));
        }

        ਜੇ (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } ਨਹੀਂਤਾਂ {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਗੰਢ : *nodes_ptr) {
        sess->ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_op(ਗੰਢ);
    }

    ਜੇ (sess->ਆਖਰੀ_error > HTP_ਸਥਿਤੀ_OK) {
        ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
    }

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    ਸਵੈਚਾਲਿਤ sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    ਜੇ (sess->ਆਖਰੀ_error > HTP_ਸਥਿਤੀ_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), ਸਥਿਤੀ_ਮੰਜ਼ਿਲ_str(sess->ਆਖਰੀ_error));
    }
}

ਗਿਣਤੀ ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

ਬਣਤਰ ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

ਬਣਤਰ ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    ਖਾਲੀ reset() {
        ranges.clear();
    }

    ਖਾਲੀ ਜੋੜੋ_2(ਅਚਲ ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    ਤਾਰਕਿਕ check(ਅਚਲ ggml_hexagon_mem_range & mr) ਅਚਲ {
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & cmp : ranges) {
            ਜੇ (mr.pb != cmp.pb) {
                ਜਾਰੀ;
            }
            ਜੇ (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                ਜਾਰੀ;
            }
            ਜੇ (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
        ਵਾਪਸ ਸੱਚ;
    }
};

ਸਥਿਰ ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(ਅਚਲ ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    ਅਚਲ ggml_tensor * ਅਧਾਰ = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    ਜੇ (tensor->ਬਫਰ) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->ਬਫਰ,
            /*.p0 =*/ (uint64_t) tensor->ਅੰਕੜੇ,
            /*.p1 =*/ (uint64_t) tensor->ਅੰਕੜੇ + ggml_backend_buft_get_alloc_ਮਾਪ(tensor->ਬਫਰ->buft, tensor),
            /*.pt =*/ pt,
        };
    } ਨਹੀਂਤਾਂ {
        mr = {
            /*.pb =*/ (uint64_t) ਅਧਾਰ,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    ਵਾਪਸ mr;
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_mem_ranges_ਜੋੜੋ_ਗੰਢ(ggml_hexagon_mem_ranges & mrs, ਅਚਲ htp_opnode & ਗੰਢ) {
    ਜੇ (ਗੰਢ.is_ਖਾਲੀ_ਥਾਵਾਂ()) ਵਾਪਸ;

    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        ਜੇ (ਗੰਢ.ਗੰਢ->src[i]) {
            mrs.ਜੋੜੋ_2(ggml_hexagon_mem_range_from_tensor(ਗੰਢ.ਗੰਢ->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * fused : ਗੰਢ.fused) {
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ਜੇ (fused->src[i]) {
                mrs.ਜੋੜੋ_2(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.ਜੋੜੋ_2(ggml_hexagon_mem_range_from_tensor(ਗੰਢ.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_mem_ranges_check_ਗੰਢ(ਅਚਲ ggml_hexagon_mem_ranges & mrs, ਅਚਲ htp_opnode & ਗੰਢ) {
    ਜੇ (ਗੰਢ.is_ਖਾਲੀ_ਥਾਵਾਂ()) ਵਾਪਸ ਸੱਚ;

    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        ਜੇ (ਗੰਢ.ਗੰਢ->src[i]) {
            ਜੇ (!mrs.check(ggml_hexagon_mem_range_from_tensor(ਗੰਢ.ਗੰਢ->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ * fused : ਗੰਢ.fused) {
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ਜੇ (fused->src[i]) {
                ਜੇ (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    ਵਾਪਸ ਝੂਠ;
                }
            }
        }
    }
    ਵਾਪਸ mrs.check(ggml_hexagon_mem_range_from_tensor(ਗੰਢ.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ਸਥਿਰ std::vector<ਪੂਰਨ> ggml_hexagon_ਗ੍ਰਾਫ_optimize_reorder(ਅਚਲ std::vector<htp_opnode> & nodes) {
    ਅਚਲ ਪੂਰਨ n = nodes.size();

    std::vector<ਪੂਰਨ> res;
    res.reserve(n);

    std::vector<ਤਾਰਕਿਕ> ਵਰਤਿਆ(n, ਝੂਠ);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    ਲਈ (ਪੂਰਨ i0 = 0; i0 < n; i0++) {
        ਜੇ (ਵਰਤਿਆ[i0]) {
            ਜਾਰੀ;
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ & node0 = nodes[i0];

        ਜੇ (!node0.stackable()) {
            res.push_back(i0);
            ਵਰਤਿਆ[i0] = ਸੱਚ;
            ਜਾਰੀ;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr ਪੂਰਨ N_FORWARD = 16;

        std::vector<ਪੂਰਨ> ਢੇਰ;
        ਢੇਰ.push_back(i0);

        mrs.reset();

        ਲਈ (ਪੂਰਨ i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            ਜੇ (ਵਰਤਿਆ[i1]) {
                ਜਾਰੀ;
            }

            ਅਚਲ ਸਵੈਚਾਲਿਤ & node1 = nodes[i1];

            ਜੇ (node1.stackable() && node1.same_ਨਿਵੇਸ਼(node0) && ggml_hexagon_mem_ranges_check_ਗੰਢ(mrs, node1)) {
                ਢੇਰ.push_back(i1);
            } ਨਹੀਂਤਾਂ {
                ggml_hexagon_mem_ranges_ਜੋੜੋ_ਗੰਢ(mrs, node1);
            }
        }

        ਲਈ (ਪੂਰਨ idx : ਢੇਰ) {
            res.push_back(idx);
            ਵਰਤਿਆ[idx] = ਸੱਚ;
        }
    }

    ਵਾਪਸ res;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_ਗ੍ਰਾਫ_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_ਗ੍ਰਾਫ_optimize_params * params) {
    GGML_ਅਣਵਰਤਿਆ(params);

    ਅਚਲ ਪੂਰਨ n = gf->n_nodes;

    constexpr ਪੂਰਨ MAX_FUSE = 16;

    ਗਿਣਤੀ ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        htp_opnode ਗੰਢ(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        ਜੇ (ਗੰਢ.op() == GGML_OP_ਜੋੜੋ ||
            ਗੰਢ.op() == GGML_OP_NORM ||
            ਗੰਢ.op() == GGML_OP_RMS_NORM) {
            ops[0] = ਗੰਢ.op();

            ਪੂਰਨ f = i + 1;
            ਜਦੋਂ (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                ਜੇ (gf->nodes[f]->op != GGML_OP_ਜੋੜੋ &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    ਰੋਕੋ;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            ਲਈ (; f > 1; f--) {
                ਜੇ (ggml_can_fuse(gf, i, ops, f)) {
                    ਰੋਕੋ;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            ਲਈ (ਪੂਰਨ k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                ਗੰਢ.ਜੋੜੋ_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(ਗੰਢ));
    }

    ਅਚਲ ਸਵੈਚਾਲਿਤ ਕ੍ਰਮ = ggml_hexagon_ਗ੍ਰਾਫ_optimize_reorder(nodes);

    // unfuse
    {
        ਪੂਰਨ j = 0;
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ i : ਕ੍ਰਮ) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ & ਗੰਢ = nodes[i];

            gf->nodes[j++] = ਗੰਢ.ਗੰਢ;

            ਲਈ (ਸਵੈਚਾਲਿਤ * fused : ਗੰਢ.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_ਅਣਵਰਤਿਆ(backend);
}

ਸਥਿਰ uint64_t ggml_hexagon_session_ਕੁੰਜੀ(ਅਚਲ ggml_hexagon_session * sess) {
    ਵਾਪਸ ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, ਅਚਲ ggml_tensor * src, ggml_tensor * dst) {
    ਸਵੈਚਾਲਿਤ sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    ਸਵੈਚਾਲਿਤ sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    ਸਵੈਚਾਲਿਤ sbuf_dst = sess_dst->mmap_tensor(dst);

    ਜੇ (!sess_src->clone_ਬਫਰ(sbuf_dst)) { ਵਾਪਸ ਝੂਠ; }

    ਅਚਲ uint64_t src_ਕੁੰਜੀ = ggml_hexagon_session_ਕੁੰਜੀ(sess_src);
    ਸਵੈਚਾਲਿਤ & fence_ਥਾਂ = sess_dst->cpy_fence_slots[src_ਕੁੰਜੀ];
    ਜੇ (!fence_ਥਾਂ) {
        fence_ਥਾਂ = (ਅਸਥਿਰ uint32_t *) sess_dst->alloc_fence(1);
    }

    ਜੇ (!sess_src->clone_ਬਫਰ(sess_dst->fence_buf)) { ਵਾਪਸ ਝੂਠ; }

    ਜੇ (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    ਸਥਿਰ ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.ਬਫਰ = &sess_dst->fence_buf->backend_ਬਫਰ;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.ਅੰਕੜੇ   = (ਖਾਲੀ *) fence_ਥਾਂ;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = ਆਕਾਰ(int32_t);
    fence_tensor.nb[1]  = ਆਕਾਰ(int32_t);
    fence_tensor.nb[2]  = ਆਕਾਰ(int32_t);
    fence_tensor.nb[3]  = ਆਕਾਰ(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_fence(&fence_tensor, fence_seq, /* wait = */ ਸੱਚ);

    sess_dst->ਜੋੜੋ_peer(sess_src);

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, ਅਚਲ ggml_tensor * src, ggml_tensor * dst) {
    ਸਵੈਚਾਲਿਤ sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    ਸਵੈਚਾਲਿਤ sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    ਸਵੈਚਾਲਿਤ sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    ਜੇ (!sess_dst->clone_ਬਫਰ(sbuf_src)) { ਵਾਪਸ ਝੂਠ; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_cpy(src, dst);
    sess_dst->ਜੋੜੋ_peer(sess_src);

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ਅਚਲ ggml_tensor * src, ggml_tensor * dst) {
    ਜੇ (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    ਜੇ (!dst->extra) {
        ggml_backend_ਬਫਰ_init_tensor(dst->ਬਫਰ, dst);
    }

    ਸਵੈਚਾਲਿਤ * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    ਅਚਲ ਸਵੈਚਾਲਿਤ * src_extra = static_cast<ਅਚਲ ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    ਸਵੈਚਾਲਿਤ sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    ਸਵੈਚਾਲਿਤ sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    ਜੇ (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_cpy(src, dst);
        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (sess_src->phys_idx != sess_dst->phys_idx)
        ਵਾਪਸ ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    ਵਾਪਸ ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

ਸਥਿਰ ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    ਸਵੈਚਾਲਿਤ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ਸਵੈਚਾਲਿਤ sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = ਨਵਾਂ ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_ਥਾਂ = (ਅਸਥਿਰ uint32_t *) sess->alloc_fence(1);

    ਸਥਿਰ ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.ਬਫਰ = &sess->fence_buf->backend_ਬਫਰ;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.ਅੰਕੜੇ   = (ਖਾਲੀ *) hex_event->fence_ਥਾਂ;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = ਆਕਾਰ(int32_t);
    hex_event->fence_tensor.nb[1]  = ਆਕਾਰ(int32_t);
    hex_event->fence_tensor.nb[2]  = ਆਕਾਰ(int32_t);
    hex_event->fence_tensor.nb[3]  = ਆਕਾਰ(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (ਖਾਲੀ *)hex_event, (ਖਾਲੀ *)hex_event->fence_ਥਾਂ);

    ਵਾਪਸ ਨਵਾਂ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    ਜੇ (hex_event->seq == 0) {
        ਵਾਪਸ;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (ਖਾਲੀ *)hex_event, hex_event->seq, (ਖਾਲੀ *)hex_event->fence_ਥਾਂ);

    ਸਵੈਚਾਲਿਤ * fence = reinterpret_cast<ਅਚਲ ਅਸਥਿਰ std::atomic<uint32_t> *>(hex_event->fence_ਥਾਂ);

    ਜੇ ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    ਜਦੋਂ (ਸੱਚ) {
        ਜੇ ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t ਸਥਿਤੀ = fence[1].load(std::memory_order_acquire);
            ਜੇ (ਸਥਿਤੀ > HTP_ਸਥਿਤੀ_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), ਸਥਿਤੀ_ਮੰਜ਼ਿਲ_str(ਸਥਿਤੀ));
            }
            ਰੋਕੋ;
        }
        std::this_thread::yield();
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ਸਵੈਚਾਲਿਤ * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (ਖਾਲੀ *)hex_event);
    hex_event->fence_sess->free_fence((ਖਾਲੀ *) hex_event->fence_ਥਾਂ, 1);
    ਮਿਟਾਓ hex_event;
    ਮਿਟਾਓ event;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ਸਵੈਚਾਲਿਤ * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_event_ਰਿਕਾਰਡ(ggml_backend_t backend, ggml_backend_event_t event) {
    ਸਵੈਚਾਲਿਤ sess = static_cast<ggml_hexagon_session *>(backend->context);
    ਸਵੈਚਾਲਿਤ hex_event = static_cast<ggml_hexagon_event *>(event->context);

    ਜੇ (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ ਝੂਠ);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (ਖਾਲੀ *)hex_event, hex_event->seq, (ਖਾਲੀ *)hex_event->fence_ਥਾਂ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ਸਵੈਚਾਲਿਤ sess = static_cast<ggml_hexagon_session *>(backend->context);
    ਸਵੈਚਾਲਿਤ hex_event = static_cast<ggml_hexagon_event *>(event->context);

    ਜੇ (hex_event->seq == 0) {
        ਵਾਪਸ;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (ਖਾਲੀ *)hex_event, hex_event->seq, (ਖਾਲੀ *)hex_event->fence_ਥਾਂ);

    // same physical NPU runs sequentially in FIFO order
    ਜੇ (sess->phys_idx == hex_event->sess->phys_idx) {
        ਜੇ (sess != hex_event->sess) {
            sess->ਜੋੜੋ_peer(hex_event->sess);
        }
        ਵਾਪਸ;
    }

    sess->clone_ਬਫਰ(hex_event->fence_sess->fence_buf);
    sess->ਜੋੜੋ_peer(hex_event->sess);
    sess->ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ ਸੱਚ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_ਸੈੱਟ_tensor_async(ggml_backend_t backend, ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ਸਵੈਚਾਲਿਤ sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, ਅੰਕੜੇ, ਵਿੱਥ, size, tensor->ਬਫਰ ? (ਪੂਰਨ) tensor->ਬਫਰ->usage : -1);
    ggml_backend_tensor_ਸੈੱਟ(tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ਸਵੈਚਾਲਿਤ sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, ਅੰਕੜੇ, ਵਿੱਥ, size, tensor->ਬਫਰ ? (ਪੂਰਨ) tensor->ਬਫਰ->usage : -1);
    sess->flush_sync();
    ਜੇ (sess->ਆਖਰੀ_error > HTP_ਸਥਿਤੀ_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), ਸਥਿਤੀ_ਮੰਜ਼ਿਲ_str(sess->ਆਖਰੀ_error));
    }
    ggml_backend_tensor_get(tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_ਸੈੱਟ_tensor_2d_async(ggml_backend_t backend,
                                                     ਬਣਤਰ ggml_tensor * tensor,
                                                     ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ,
                                                     size_t ਵਿੱਥ,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_ਅੰਕੜੇ) {
    ਸਵੈਚਾਲਿਤ sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride_tensor, stride_ਅੰਕੜੇ, tensor->ਬਫਰ ? (ਪੂਰਨ) tensor->ਬਫਰ->usage : -1);
    ggml_backend_tensor_ਸੈੱਟ_2d(tensor, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride_tensor, stride_ਅੰਕੜੇ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     ਅਚਲ ਬਣਤਰ ggml_tensor * tensor,
                                                     ਖਾਲੀ * ਅੰਕੜੇ,
                                                     size_t ਵਿੱਥ,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_ਅੰਕੜੇ) {
    ਸਵੈਚਾਲਿਤ sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride_tensor, stride_ਅੰਕੜੇ, tensor->ਬਫਰ ? (ਪੂਰਨ) tensor->ਬਫਰ->usage : -1);
    sess->flush_sync();
    ਜੇ (sess->ਆਖਰੀ_error > HTP_ਸਥਿਤੀ_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), ਸਥਿਤੀ_ਮੰਜ਼ਿਲ_str(sess->ਆਖਰੀ_error));
    }
    ggml_backend_tensor_get_2d(tensor, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride_tensor, stride_ਅੰਕੜੇ);
}

ਸਥਿਰ ਬਣਤਰ ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_ਸੈੱਟ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_ਸੈੱਟ_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ ਨਿਰਾ,
    /* .graph_plan_free         = */ ਨਿਰਾ,
    /* .graph_plan_update       = */ ਨਿਰਾ,
    /* .graph_plan_compute      = */ ਨਿਰਾ,
    /* .graph_compute           = */ ggml_backend_hexagon_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_ਰਿਕਾਰਡ,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_ਗ੍ਰਾਫ_optimize,
};

ਸਥਿਰ ggml_guid_t ggml_backend_hexagon_guid() {
    ਸਥਿਰ ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    ਵਾਪਸ &guid;
}

ਤਾਰਕਿਕ ggml_backend_is_hexagon(ggml_backend_t backend) {
    ਵਾਪਸ backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

ਸਥਿਰ ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    ਸਵੈਚਾਲਿਤ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ਸਵੈਚਾਲਿਤ sess    = dev_ctx->session();

    ਵਾਪਸ ਨਵਾਂ ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_ਅਣਵਰਤਿਆ(params);
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    ਸਵੈਚਾਲਿਤ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ਵਾਪਸ dev_ctx->c_name();

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    ਵਾਪਸ "Hexagon";
    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    *ਮੁਕਤ  = opt_mbuf_ਕੁੱਲ;
    *ਕੁੱਲ = opt_mbuf_ਕੁੱਲ;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, ਬਣਤਰ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);
    props->caps = {
        /* .async                 = */ ਸੱਚ,
        /* .host_buffer           = */ ਝੂਠ,
        /* .buffer_from_host_ptr  = */ ਝੂਠ,
        /* .events                = */ ਸੱਚ,
        /* .mmap_support          = */ ਝੂਠ,
    };
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_hexagon_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ਸਵੈਚਾਲਿਤ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ਵਾਪਸ &dev_ctx->ਬਫਰ_type;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_hexagon_device_get_host_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ਜੇ (!opt_hostbuf) {
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਸਵੈਚਾਲਿਤ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ਵਾਪਸ &dev_ctx->host_ਬਫਰ_type;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_cpy(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਬਣਤਰ htp_copy_kernel_params kparams;
    ਵਾਪਸ ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_cont(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਬਣਤਰ htp_copy_kernel_params kparams;
    ਵਾਪਸ ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_repeat(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    GGML_ਅਣਵਰਤਿਆ(sess);
    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    // Support f32 and f16
    ਜੇ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ਵਾਪਸ ਝੂਠ;

    // src and dst must be the same type
    ਜੇ (src0->type != dst->type) ਵਾਪਸ ਝੂਠ;

    // dst dims must be multiples of src dims
    ਜੇ (dst->ne[0] % src0->ne[0] != 0) ਵਾਪਸ ਝੂਠ;
    ਜੇ (dst->ne[1] % src0->ne[1] != 0) ਵਾਪਸ ਝੂਠ;
    ਜੇ (dst->ne[2] % src0->ne[2] != 0) ਵਾਪਸ ਝੂਠ;
    ਜੇ (dst->ne[3] % src0->ne[3] != 0) ਵਾਪਸ ਝੂਠ;

    // require contiguous tensors (no transposition)
    ਜੇ (ggml_is_transposed(src0) || ggml_is_transposed(dst)) ਵਾਪਸ ਝੂਠ;

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_concat(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਬਣਤਰ htp_concat_kernel_params kparams;
    ਵਾਪਸ ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_fill(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst = op;

    ਜੇ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
    GGML_ਅਣਵਰਤਿਆ(sess);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_hexagon_supported_roll(ਅਚਲ ਬਣਤਰ ggml_hexagon_session * sess, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    GGML_ਅਣਵਰਤਿਆ(sess);

    ਅਚਲ ਬਣਤਰ ggml_tensor * src0 = op->src[0];
    ਅਚਲ ਬਣਤਰ ggml_tensor * dst  = op;

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_are_same_shape(src0, dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (src0->nb[0] != ggml_type_ਮਾਪ(src0->type) || dst->nb[0] != ggml_type_ਮਾਪ(dst->type)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ਸਵੈਚਾਲਿਤ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ਸਵੈਚਾਲਿਤ sess    = dev_ctx->session();

    // reject ops that match the filter
    ਜੇ (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਤਾਰਕਿਕ supp = ਝੂਠ;
    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_RESHAPE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
            supp = ਸੱਚ;
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_MUL:
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_SUB:
        ਹਾਲਤ GGML_OP_DIV:
            supp = ggml_hexagon_supported_ਦੁਆਧਾਰੀ(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
            supp = ggml_hexagon_supported_ਜੋੜੋ_id(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_NORM:
        ਹਾਲਤ GGML_OP_L2_NORM:
        ਹਾਲਤ GGML_OP_RMS_NORM:
        ਹਾਲਤ GGML_OP_SCALE:
        ਹਾਲਤ GGML_OP_CLAMP:
        ਹਾਲਤ GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_SQR:
        ਹਾਲਤ GGML_OP_SQRT:
        ਹਾਲਤ GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_ਜੋੜ:
            supp = ggml_hexagon_supported_ਜੋੜ(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            supp = ggml_hexagon_supported_ਜੋੜ_ਪੰਕਤੀਆਂ(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_UNARY:
            ਚੋਣ (ggml_get_unary_op(op)) {
                ਹਾਲਤ GGML_UNARY_OP_NEG:
                ਹਾਲਤ GGML_UNARY_OP_EXP:
                ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
                ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
                ਹਾਲਤ GGML_UNARY_OP_TANH:
                ਹਾਲਤ GGML_UNARY_OP_ABS:
                ਹਾਲਤ GGML_UNARY_OP_SILU:
                ਹਾਲਤ GGML_UNARY_OP_GELU:
                ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
                ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
                ਹਾਲਤ GGML_UNARY_OP_RELU:
                ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    ਰੋਕੋ;
                ਮੂਲ:
                    supp = ਝੂਠ;
                    ਰੋਕੋ;
            }
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_GLU:
            ਚੋਣ (ggml_get_glu_op(op)) {
                ਹਾਲਤ GGML_GLU_OP_SWIGLU:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
                ਹਾਲਤ GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    ਰੋਕੋ;
                ਮੂਲ:
                    supp = ਝੂਠ;
                    ਰੋਕੋ;
            }
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            supp = ggml_hexagon_supported_ਸੈੱਟ_ਪੰਕਤੀਆਂ(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            supp = ggml_hexagon_supported_get_ਪੰਕਤੀਆਂ(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_DUP:
        ਹਾਲਤ GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
            supp = ggml_hexagon_supported_ਸਿਖਰ_k(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
            supp = ggml_hexagon_supported_im2col(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_3_D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            ਰੋਕੋ;

        ਹਾਲਤ GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            ਰੋਕੋ;

        ਮੂਲ:
            ਰੋਕੋ;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    ਵਾਪਸ supp;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    ਸਵੈਚਾਲਿਤ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    ਤਾਰਕਿਕ supp = (buft == &dev_ctx->host_ਬਫਰ_type) || (buft == &dev_ctx->ਬਫਰ_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    ਵਾਪਸ supp;
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_ਬਫਰ_type,
    /* .buffer_from_host_ptr = */ ਨਿਰਾ,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ ਨਿਰਾ,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    ਲਈ (size_t i = 0; i < opt_ndev; i++) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ & cfg = opt_device_configs[i];
        ਜੇ (cfg.mdev_ਸਮੂਹ.ਖਾਲੀ_ਥਾਵਾਂ()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } ਨਹੀਂਤਾਂ {
            std::string peers_str;
            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & p : cfg.mdev_ਸਮੂਹ) {
                ਜੇ (!peers_str.ਖਾਲੀ_ਥਾਵਾਂ()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = ਨਵਾਂ ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    ਲਈ (size_t i = 0; i < opt_ndev; i++) {
        ਸਵੈਚਾਲਿਤ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        ਮਿਟਾਓ dev_ctx;
    }
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    ਵਾਪਸ "HTP";
    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ size_t ggml_backend_hexagon_reg_get_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ਵਾਪਸ opt_ndev;
    GGML_ਅਣਵਰਤਿਆ(reg);
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t ਸੂਚਕਾਂਕ) {
    ਸਵੈਚਾਲਿਤ hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    ਜੇ (ਸੂਚਕਾਂਕ >= opt_ndev || !hreg->devices[ਸੂਚਕਾਂਕ].context) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਵਾਪਸ &hreg->devices[ਸੂਚਕਾਂਕ];
}

// ** communication context for tensor-split allreduce

ਸਥਿਰ ਖਾਲੀ * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    ਜੇ (n_backends < 2 || n_backends > 4) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਲਈ (size_t i = 0; i < n_backends; ++i) {
        ਜੇ (!ggml_backend_is_hexagon(backends[i])) {
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    }

    ਲਈ (size_t i = 0; i < n_backends; i++) {
        ਸਵੈਚਾਲਿਤ sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ਲਈ (size_t j = i + 1; j < n_backends; j++) {
            ਸਵੈਚਾਲਿਤ sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            ਜੇ (sess_i->phys_idx == sess_j->phys_idx) {
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
        }
    }

    ਸਵੈਚਾਲਿਤ * ctx = ਨਵਾਂ ggml_backend_hexagon_comm_context();
    ctx->backends.ਸੌਂਪੋ(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    ਸਥਿਰ ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ਲਈ (size_t i = 0; i < n_backends; i++) {
        ਸਵੈਚਾਲਿਤ sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (ਅਸਥਿਰ uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].ਬਫਰ = &sess_i->fence_buf->backend_ਬਫਰ;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].ਅੰਕੜੇ   = (ਖਾਲੀ *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = ਆਕਾਰ(int32_t);
        ctx->fence_tensors[i].nb[1]  = ਆਕਾਰ(int32_t);
        ctx->fence_tensors[i].nb[2]  = ਆਕਾਰ(int32_t);
        ctx->fence_tensors[i].nb[3]  = ਆਕਾਰ(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    ਵਾਪਸ ctx;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_hexagon_comm_free(ਖਾਲੀ * comm_ctx_v) {
    ਜੇ (!comm_ctx_v) ਵਾਪਸ;
    ਸਵੈਚਾਲਿਤ * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ਲਈ (size_t i = 0; i < ctx->n_backends; i++) {
        ਸਵੈਚਾਲਿਤ sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((ਖਾਲੀ *) ctx->fence_slots[i], 1);
    }
    ਮਿਟਾਓ ctx;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_hexagon_comm_allreduce_tensor(ਖਾਲੀ * comm_ctx_v, ਬਣਤਰ ggml_tensor ** tensors) {
    ਜੇ (opt_ar_select == 0 || !comm_ctx_v) ਵਾਪਸ ਝੂਠ;
    ਸਵੈਚਾਲਿਤ * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ਅਚਲ size_t n_backends = comm_ctx->n_backends;

    ਜੇ (n_backends < 2 || n_backends > 4) ਵਾਪਸ ਝੂਠ;

    ਲਈ (size_t i = 0; i < n_backends; i++) {
        ਸਵੈਚਾਲਿਤ sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        ਲਈ (size_t j = i + 1; j < n_backends; j++) {
            ਸਵੈਚਾਲਿਤ sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            ਜੇ (sess_i->phys_idx == sess_j->phys_idx) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
    }

    ਲਈ (size_t i = 0; i < n_backends; i++) {
        ਜੇ (!tensors[i] || !tensors[i]->ਬਫਰ || !ggml_backend_ਬਫਰ_is_hexagon(tensors[i]->ਬਫਰ)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (tensors[i]->type != tensors[0]->type) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (!ggml_is_contiguous(tensors[i])) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਜੇ (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਲਈ (size_t r = 0; r < n_backends; r++) {
        ਸਵੈਚਾਲਿਤ sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        ਬਣਤਰ htp_allreduce_kernel_params kparams;
        ਜੇ (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, ਝੂਠ, ਝੂਠ, /*is_shard_ok=*/ ਝੂਠ, &kparams)) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    ਲਈ (size_t i = 1; i < n_backends; i++) {
        ਸਵੈਚਾਲਿਤ sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        ਜੇ ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    ਜੇ (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_ਇੰਦਰਾਜ = max_seq;
    ਜੇ (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    ਲਈ (size_t i = 0; i < n_backends; i++) {
        ਸਵੈਚਾਲਿਤ sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<ਅਚਲ ggml_tensor *> ਅੰਕੜੇ_tensors(n_backends);
    std::vector<ਅਚਲ ggml_tensor *> sync_tensors(n_backends);
    ਲਈ (size_t i = 0; i < n_backends; i++) {
        ਅੰਕੜੇ_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    ਲਈ (size_t r = 0; r < n_backends; r++) {
        ਸਵੈਚਾਲਿਤ sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->ਕਤਾਰ_ਵਿੱਚ_ਪਾਓ_allreduce(tensors[r], ਅੰਕੜੇ_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_ਇੰਦਰਾਜ, fence_seq_exit);
        ਲਈ (size_t j = 0; j < n_backends; j++) {
            ਜੇ (r != j) {
                sess->ਜੋੜੋ_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_hexagon_ਵੰਡੋ_ਬਫਰ_type(ਪੂਰਨ main_device, ਅਚਲ ਦਸ਼ਮਲਵ * tensor_ਵੰਡੋ) {
    GGML_ਅਣਵਰਤਿਆ(tensor_ਵੰਡੋ);
    ਸਵੈਚਾਲਿਤ reg = ggml_backend_hexagon_reg();
    ਸਵੈਚਾਲਿਤ dev = ggml_backend_reg_dev_get(reg, main_device);
    ਜੇ (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    ਜੇ (!dev) ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    ਸਵੈਚਾਲਿਤ dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ਵਾਪਸ &dev_ctx->ਬਫਰ_type;
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_hexagon_get_proc_ਪਤਾ(ggml_backend_reg_t reg, ਅਚਲ ਅੱਖਰ * name) {
    GGML_ਅਣਵਰਤਿਆ(reg);
    ਜੇ (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *) ggml_backend_hexagon_ਵੰਡੋ_ਬਫਰ_type;
    }
    ਜੇ (strcmp(name, "ggml_backend_comm_init") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *) ggml_backend_hexagon_comm_init;
    }
    ਜੇ (strcmp(name, "ggml_backend_comm_free") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *) ggml_backend_hexagon_comm_free;
    }
    ਜੇ (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    ਵਾਪਸ ਨਿਰਾ;
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T> std::vector<T> str_ਮੰਜ਼ਿਲ_vec(ਅਚਲ ਅੱਖਰ* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    ਜਦੋਂ (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, ਖਾਲੀ_ਸੂਚਕ, 0));
    }

    ਵਾਪਸ v;
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ ਅਧਾਰ_2=10> std::string vec_ਮੰਜ਼ਿਲ_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(ਅਧਾਰ_2) << std::showbase;
    ਲਈ (ਸਵੈਚਾਲਿਤ i : v) { ss << i << ','; }
    ਸਵੈਚਾਲਿਤ str = ss.str(); str.pop_back(); // drop last comma
    ਵਾਪਸ str;
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, ਤਾਰਕਿਕ discovery_supported, ਅਚਲ std::unordered_map<ਪੂਰਨ, fastrpc_domain> & cdsp_map) {
    ਜੇ (discovery_supported) {
        ਸਵੈਚਾਲਿਤ it = cdsp_map.find(cfg.physical_idx);
        ਜੇ (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } ਨਹੀਂਤਾਂ {
        ਚੋਣ (cfg.physical_idx) {
            ਹਾਲਤ 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                ਰੋਕੋ;
            ਹਾਲਤ 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                ਰੋਕੋ;
            ਮੂਲ:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                ਰੋਕੋ;
        }
    }
    ਲਈ (ਸਵੈਚਾਲਿਤ & sub_cfg : cfg.mdev_ਸਮੂਹ) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
ਸਥਿਰ ਖਾਲੀ ggml_hexagon_discover_devices() {
    std::unordered_map<ਪੂਰਨ, fastrpc_domain> cdsp_map;
    ਤਾਰਕਿਕ discovery_supported = ਝੂਠ;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = ਖਾਲੀ_ਸੂਚਕ;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_ਸੈੱਟ_TYPE(0, FASTRPC_NSP);

    ਪੂਰਨ err = remote_system_request(&domain_info);
    ਜੇ (err == AEE_SUCCESS && domain_info.sys.ਅੰਕ_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.ਅੰਕ_domains);
        domain_info.sys.domains     = domains.ਅੰਕੜੇ();
        domain_info.sys.max_domains = (ਪੂਰਨ) domains.size();

        err = remote_system_request(&domain_info);
        ਜੇ (err == AEE_SUCCESS) {
            discovery_supported = ਸੱਚ;
            ਅਚਲ ਪੂਰਨ n_domains = std::min(domain_info.sys.ਅੰਕ_domains, (ਪੂਰਨ) domains.size());
            ਲਈ (ਪੂਰਨ i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (ਪੂਰਨ) domains[i].type, domains[i].id, domains[i].name, domains[i].ਸਥਿਤੀ, domains[i].instance_id);
                ਜੇ (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (ਪੂਰਨ) domains[i].type);
                    ਜਾਰੀ;
                }
                ਜੇ (!domains[i].ਸਥਿਤੀ) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    ਜਾਰੀ;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
        }
    } ਨਹੀਂਤਾਂ ਜੇ (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (ਬਿਨਾਨਿਸ਼ਾਨੀ) err);
    }

    // Populate domain IDs and names for all configured devices
    ਲਈ (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_Q4_0 == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_Q4_1 == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_Q8_0 == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_MXFP4 == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_IQ4_NL == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_Q4_K == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_Q5_K == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_Q6_K == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_Q3_K == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_Q2_K == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_Q1_0 == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_Q2_0 == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) HTP_TYPE_BF16 == (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    ਅਚਲ ਅੱਖਰ * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    ਅਚਲ ਅੱਖਰ * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    ਅਚਲ ਅੱਖਰ * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    ਅਚਲ ਅੱਖਰ * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    ਅਚਲ ਅੱਖਰ * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    ਅਚਲ ਅੱਖਰ * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    ਅਚਲ ਅੱਖਰ * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    ਅਚਲ ਅੱਖਰ * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    ਅਚਲ ਅੱਖਰ * str_etm      = getenv("GGML_HEXAGON_ETM");
    ਅਚਲ ਅੱਖਰ * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    ਅਚਲ ਅੱਖਰ * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    ਅਚਲ ਅੱਖਰ * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    ਅਚਲ ਅੱਖਰ * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    ਅਚਲ ਅੱਖਰ * str_fa_ਸਿਰ_ਵੰਡੋ = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    ਅਚਲ ਅੱਖਰ * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    ਅਚਲ ਅੱਖਰ * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    ਅਚਲ ਅੱਖਰ * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    ਅਚਲ ਅੱਖਰ * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    ਅਚਲ ਅੱਖਰ * str_arch     = getenv("GGML_HEXAGON_ARCH");
    ਅਚਲ ਅੱਖਰ * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    ਅਚਲ ਅੱਖਰ * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    ਅਚਲ ਅੱਖਰ * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    ਅਚਲ ਅੱਖਰ * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    ਅਚਲ ਅੱਖਰ * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    ਅਚਲ ਅੱਖਰ * str_power_ਬਹੁਲਕ = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    ਜੇ (!str_arch) {
        ਪੂਰਨ err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        ਜੇ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } ਨਹੀਂਤਾਂ {
            ਜੇ (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } ਨਹੀਂਤਾਂ ਜੇ (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, ਨਿਰਾ, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    ਸਵੈਚਾਲਿਤ RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? ਨਵਾਂ std::regex(str_opfilter, RE_ICASE) : ਨਿਰਾ;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, ਨਿਰਾ, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, ਨਿਰਾ, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, ਨਿਰਾ, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  ਨਿਰਾ, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    ਜੇ (str_power_ਬਹੁਲਕ) {
        ਪੂਰਨ m = htp_power_ਬਹੁਲਕ_from_str(str_power_ਬਹੁਲਕ);
        ਜੇ (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_ਬਹੁਲਕ);
        } ਨਹੀਂਤਾਂ {
            opt_power_ਬਹੁਲਕ = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, ਨਿਰਾ, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_ਸਿਰ_ਵੰਡੋ = str_fa_ਸਿਰ_ਵੰਡੋ ? atoi(str_fa_ਸਿਰ_ਵੰਡੋ)       : opt_fa_ਸਿਰ_ਵੰਡੋ;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    ਜੇ (str_mbuf) {
        ਅਚਲ ਅੱਖਰ * p = str_mbuf;
        ਲਈ (ਪੂਰਨ idx = 0; idx < 3 && p && *p; idx++) {
            ਜਦੋਂ (*p == ' ') p++;
            ਅਚਲ ਅੱਖਰ * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : ਲੜੀ_ਲੰਬਾਈ(p);
            ਜਦੋਂ (len > 0 && p[len - 1] == ' ') len--;
            ਜੇ (len > 0) {
                std::string token(p, len);
                ਜੇ (idx == 0) opt_mbuf_dyn    = parse_ਮਾਪ(token.c_str());
                ਜੇ (idx == 1) opt_mbuf_static = parse_ਮਾਪ(token.c_str());
                ਜੇ (idx == 2) opt_mbuf_ਕੁੱਲ  = parse_ਮਾਪ(token.c_str());
            }
            ਜੇ (!comma) ਰੋਕੋ;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_ਮਾਪ(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    ਅਚਲ ਅੱਖਰ * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    ਜੇ (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    ਜੇ (str_devices && str_devices[0] != '\0') {
        ਤਾਰਕਿਕ is_single_ਸੰਖਿਆ = ਸੱਚ;
        ਲਈ (ਪੂਰਨ i = 0; str_devices[i] != '\0'; i++) {
            ਜੇ (!ਅੰਕ_ਹੈ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)str_devices[i])) {
                is_single_ਸੰਖਿਆ = ਝੂਠ;
                ਰੋਕੋ;
            }
        }
        ਜੇ (is_single_ਸੰਖਿਆ) {
            ਪੂਰਨ n = atoi(str_devices);
            ਜੇ (n < 1) n = 1;
            ਜੇ (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            ਲਈ (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (ਪੂਰਨ)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_ਸਮੂਹ.clear();
            }
        } ਨਹੀਂਤਾਂ {
            std::string s_devices(str_devices);
            std::vector<std::string> ਚੀਜ਼ਾਂ;
            std::string curr_ਵਸਤੂ;
            ਪੂਰਨ bracket_ਡੂੰਘਾਈ = 0;
            ਲਈ (ਅੱਖਰ ch : s_devices) {
                ਜੇ (ch == '[') {
                    bracket_ਡੂੰਘਾਈ++;
                    curr_ਵਸਤੂ += ch;
                } ਨਹੀਂਤਾਂ ਜੇ (ch == ']') {
                    ਜੇ (bracket_ਡੂੰਘਾਈ > 0) bracket_ਡੂੰਘਾਈ--;
                    curr_ਵਸਤੂ += ch;
                } ਨਹੀਂਤਾਂ ਜੇ (ch == ',' && bracket_ਡੂੰਘਾਈ == 0) {
                    size_t s = curr_ਵਸਤੂ.find_first_not_of(" \t\r\n");
                    size_t e = curr_ਵਸਤੂ.find_last_not_of(" \t\r\n");
                    ਜੇ (s != std::string::npos) {
                        ਚੀਜ਼ਾਂ.push_back(curr_ਵਸਤੂ.substr(s, e - s + 1));
                    }
                    curr_ਵਸਤੂ.clear();
                } ਨਹੀਂਤਾਂ {
                    curr_ਵਸਤੂ += ch;
                }
            }
            size_t s = curr_ਵਸਤੂ.find_first_not_of(" \t\r\n");
            size_t e = curr_ਵਸਤੂ.find_last_not_of(" \t\r\n");
            ਜੇ (s != std::string::npos) {
                ਚੀਜ਼ਾਂ.push_back(curr_ਵਸਤੂ.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਵਸਤੂ : ਚੀਜ਼ਾਂ) {
                size_t b_open  = ਵਸਤੂ.find('[');
                size_t b_close = ਵਸਤੂ.rfind(']');

                ਜੇ (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = ਵਸਤੂ.substr(0, b_open);
                    std::string content  = ਵਸਤੂ.substr(b_open + 1, b_close - b_open - 1);

                    ਪੂਰਨ virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    ਜੇ (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        ਕੋਸ਼ਿਸ਼ {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } ਫੜੋ (...) {
                            virt = 0;
                        }
                    } ਨਹੀਂਤਾਂ {
                        size_t dev_colon = dev_name.find(':');
                        ਜੇ (dev_colon != std::string::npos) {
                            ਕੋਸ਼ਿਸ਼ {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } ਫੜੋ (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<ਪੂਰਨ> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    ਜਦੋਂ (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        ਜੇ (ps == std::string::npos) ਜਾਰੀ;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        ਜੇ (dash_pos != std::string::npos) {
                            ਕੋਸ਼ਿਸ਼ {
                                ਪੂਰਨ p_ਸ਼ੁਰੂ = std::stoi(p_part.substr(0, dash_pos));
                                ਪੂਰਨ p_ਸਿਰਾ   = std::stoi(p_part.substr(dash_pos + 1));
                                ਲਈ (ਪੂਰਨ p = p_ਸ਼ੁਰੂ; p <= p_ਸਿਰਾ; p++) {
                                    ਜੇ (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } ਫੜੋ (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } ਨਹੀਂਤਾਂ {
                            ਕੋਸ਼ਿਸ਼ {
                                ਪੂਰਨ p = std::stoi(p_part);
                                ਜੇ (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } ਫੜੋ (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    ਜੇ (phys_list.ਖਾਲੀ_ਥਾਵਾਂ()) {
                        phys_list.push_back(0);
                    }

                    ਜੇ (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        ਸਵੈਚਾਲਿਤ & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_ਸਮੂਹ.clear();

                        ਲਈ (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_ਸਮੂਹ.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } ਨਹੀਂਤਾਂ {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, ਵਸਤੂ.c_str());
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (ਵਸਤੂ.rfind("HTP", 0) == 0) {
                    std::string rest = ਵਸਤੂ.substr(3);
                    size_t colon_pos = rest.find(':');
                    ਪੂਰਨ phys = 0;
                    ਪੂਰਨ virt = 0;
                    ਕੋਸ਼ਿਸ਼ {
                        ਜੇ (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } ਨਹੀਂਤਾਂ {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } ਫੜੋ (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", ਵਸਤੂ.c_str());
                        ਜਾਰੀ;
                    }

                    ਜੇ (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_ਸਮੂਹ.clear();
                        opt_ndev++;
                    } ਨਹੀਂਤਾਂ {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, ਵਸਤੂ.c_str());
                    }
                } ਨਹੀਂਤਾਂ {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", ਵਸਤੂ.c_str());
                }
            }
        }
    } ਨਹੀਂਤਾਂ {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_ਸਮੂਹ.clear();
    }

#ਜੇ defined(__ANDROID__)
    ਜੇ (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#ਜੇ_ਅੰਤ

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    ਜੇ (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            ਸਵੈਚਾਲਿਤ v  = str_ਮੰਜ਼ਿਲ_vec<uint32_t>(str_profile);
            ਚੋਣ (v.size()) {
                ਹਾਲਤ 1:  opt_profile = v[0]; ਵਾਪਸ opt_pmu_evt; // mode with default pmu events
                ਹਾਲਤ 8:  opt_profile = 2;    ਵਾਪਸ v;           // mode with custom  pmu events
                ਮੂਲ: opt_profile = 0;    ਵਾਪਸ {};          // garbage input
            }}();
        ਜੇ (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_ਮੰਜ਼ਿਲ_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = ਨਵਾਂ ggml_hexagon_registry(reg);
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_ਗਿਣਤੀ,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_ਪਤਾ,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(ਖਾਲੀ) {
    ਸਥਿਰ ਤਾਰਕਿਕ initialized = ਝੂਠ;

    ਸਥਿਰ ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ ਨਿਰਾ };

    {
        ਸਥਿਰ std::mutex           mutex;
        std::lock_guard<std::mutex> ਤਾਲਾ(mutex);
        ਜੇ (!initialized) {
            ਸਵੈਚਾਲਿਤ nErr = htpdrv_init();
            ਜੇ (nErr != AEE_SUCCESS) {
                ਵਾਪਸ ਨਿਰਾ;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = ਸੱਚ;
    }

    ਵਾਪਸ &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
